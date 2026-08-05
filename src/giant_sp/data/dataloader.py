"""Data loading and preprocessing utilities for DeepSensor."""

import glob
import json
import logging
import os
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.data.sources import get_gldas_land_mask
from deepsensor.data.utils import construct_x1x2_ds

from giant_sp.utils.constants import region_extents
from giant_sp.utils.data import construct_circ_and_linear_time_ds

logger = logging.getLogger(__name__)

cache_dir = ".datacache"


class BaseDataLoader:
    """Base class for data loaders.

    Reads the common ``data`` and ``dataloader`` sections of a configuration
    dictionary and exposes helpers shared by all concrete loaders (anomaly
    computation, feature derivation, spatiotemporal auxiliary variables).
    Subclasses must implement :meth:`load_ds`, :meth:`load_aux_ds` and
    :meth:`get_dataproc_taskload`.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary, typically loaded from a YAML file. Must
        contain a ``"data"`` section with keys ``data_path``,
        ``data_pattern``, ``region``, ``basin``, ``context_var_IDs``,
        ``target_var_IDs``, ``train_start``, ``train_end``, ``val_start``,
        ``val_end``, and a ``"dataloader"`` section with keys ``lsm_mask``
        and ``frequency``.

    Attributes
    ----------
    config : dict[str, Any]
        The configuration dictionary passed at construction.
    source : pathlib.Path
        Root path where the source data lives.
    data_pattern : str
        Glob pattern (relative to ``source/region``) used to find data files.
    region : str
        Region key, used to index ``giant_sp.utils.constants.region_extents``
        and to build paths under ``source``.
    basin : str | None
        Basin code to spatially subset the data to (e.g. ``"SE"``), or
        ``None`` to keep the full region.
    context_vars : list[list[str]] | list[str] | str
        Context variable identifiers, as given in the configuration. May be
        a flat list, a list of groups (each a list of variable IDs), or a
        single variable ID.
    target_vars : str
        Target variable identifier.
    train_start, train_end, val_start, val_end : str
        Date strings (parseable by :class:`pandas.Timestamp`) bounding the
        training and validation periods.
    lsm_mask : bool
        Whether to mask out non-land points using the land-sea mask.
    frequency : {"D", "H", "MS", "ME"}
        Frequency code used to derive circular time features.
    variables : list[str]
        Flat list of variable IDs to load, populated by :meth:`load_ds` in
        subclasses.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the data loader from a configuration dictionary.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary. See the class docstring for the
            required keys.
        """
        self.config = config
        self.source = Path(config["data"]["data_path"])
        self.data_pattern: str = config["data"]["data_pattern"]
        self.region: str = config["data"]["region"]
        self.basin: str | None = config["data"]["basin"]
        self.context_vars: list[list[str]] | list[str] | str = config["data"][
            "context_var_IDs"
        ]
        self.context_delta_t = config["dataloader"]["context_delta_t"]
        self.context_delta_t = config["dataloader"]["context_delta_t"]
        self.target_vars: str = config["data"]["target_var_IDs"]
        self.train_start: str = self.config["data"]["train_start"]
        self.train_end: str = self.config["data"]["train_end"]
        self.val_start: str = config["data"]["val_start"]
        self.val_end: str = config["data"]["val_end"]
        self.lsm_mask: bool = config["dataloader"]["lsm_mask"]
        self.frequency: Literal["D", "H", "MS", "ME"] = config["dataloader"][
            "frequency"
        ]
        self.variables: list[str] = []

    def add_spatiotemporal_aux(
        self, aux_ds: xr.Dataset, ref_ds: xr.Dataset
    ) -> xr.Dataset:
        """Augment auxiliary data with spatial and circular time features.

        Adds normalised spatial coordinate arrays (``x1_arr``, ``x2_arr``)
        derived from ``aux_ds``, and circular/linear time features (e.g.
        ``cos_D``/``sin_D``) derived from the time range of ``ref_ds`` at
        ``self.frequency``.

        Parameters
        ----------
        aux_ds : xarray.Dataset
            Auxiliary dataset to augment in place (e.g. the normalised
            land-sea mask).
        ref_ds : xarray.Dataset
            Reference dataset whose ``time`` coordinate defines the date
            range used to construct the time features.

        Returns
        -------
        xarray.Dataset
            ``aux_ds`` with ``x1_arr``, ``x2_arr``, ``cos_{frequency}`` and
            ``sin_{frequency}`` variables added.
        """
        x1x2_ds = construct_x1x2_ds(aux_ds)
        aux_ds["x1_arr"] = x1x2_ds["x1_arr"]
        aux_ds["x2_arr"] = x1x2_ds["x2_arr"]

        dates = pd.date_range(
            ref_ds.time.values.min(), ref_ds.time.values.max(), freq=self.frequency
        )
        doy_ds = construct_circ_and_linear_time_ds(dates, freq=self.frequency)
        aux_ds[f"cos_{self.frequency}"] = doy_ds[f"cos_{self.frequency}"]
        aux_ds[f"sin_{self.frequency}"] = doy_ds[f"sin_{self.frequency}"]

        return aux_ds

    def compute_gridded_anomalies(self, var_ID: str, da: xr.DataArray) -> xr.DataArray:
        """Compute anomalies of ``da`` relative to its training climatology.

        Loads a cached daily climatology (mean over day-of-year, computed
        over ``[self.train_start, self.train_end]``) if one already exists
        on disk, otherwise computes and caches it. The climatology is then
        subtracted from ``da`` to obtain anomalies.

        Parameters
        ----------
        var_ID : str
            Variable identifier, used to build the climatology cache
            filename.
        da : xarray.DataArray
            Absolute-value data array (with a ``time`` dimension) to convert
            to anomalies.

        Returns
        -------
        xarray.DataArray
            Anomalies of ``da`` relative to its training-period daily
            climatology.

        Notes
        -----
        The climatology is always computed at daily resolution
        (``time.dayofyear``); there is currently no option for other
        temporal resolutions.
        """
        # Check if climatology already computed
        climatology_folder = os.path.join(self.source, self.region)
        climatology_fpath = os.path.join(
            climatology_folder,
            "{}_climatology_{}_{}_{}.nc".format(
                var_ID,
                pd.Timestamp(self.train_start).year,
                pd.Timestamp(self.train_end).year,
                self.data_pattern.split("*.")[0][-3:],
            ),
        )
        if os.path.exists(climatology_fpath):
            with xr.open_dataset(climatology_fpath) as ds:
                climatology = next(iter(ds.data_vars.values()))
        else:
            climatology = (
                da.sel(
                    time=slice(
                        pd.Timestamp(self.train_start), pd.Timestamp(self.train_end)
                    )
                )
                .groupby("time.dayofyear", restore_coord_dims=True)
                .mean("time")
            )
            climatology.to_netcdf(climatology_fpath)

        da = da.groupby("time.dayofyear", restore_coord_dims=True) - climatology

        return da

    def featurize_vars(
        self,
        var: str,
        dims: Sequence[Hashable],
        chunks: Sequence[tuple[int, ...]],
        abs: xr.DataArray,
    ) -> xr.Dataset:
        """Derive anomaly features for a variable.

        Computes the anomaly of ``abs`` (via :meth:`compute_gridded_anomalies`)
        and returns it as a single-variable dataset, re-chunked along
        ``dims`` using the given ``chunks``.

        Parameters
        ----------
        var : str
            Base variable name from which the anomaly feature is derived.
            The output variable is named ``f"{var}_anm"``.
        dims : Sequence[Hashable]
            Dimension names of ``abs``, in the same order as ``chunks``.
        chunks : Sequence[tuple[int, ...]]
            Dask chunk sizes for each dimension in ``dims``, in the same
            order.
        abs : xarray.DataArray
            Absolute-value data array from which the anomaly feature is
            computed.

        Returns
        -------
        xarray.Dataset
            Dataset containing the single derived variable ``f"{var}_anm"``.
        """
        ds = xr.Dataset()

        anom = self.compute_gridded_anomalies(var, abs)

        ds[var + "_anm"] = anom
        ds = ds.chunk(dict(zip(dims, chunks, strict=True)))

        return ds

    def init_data_processor(self) -> DataProcessor:
        """Create a default :class:`DataProcessor`.

        Returns
        -------
        deepsensor.data.processor.DataProcessor
            A new, unfitted data processor configured with ``x1_name="lat"``
            and ``x2_name="lon"``.
        """
        data_processor = DataProcessor(
            x1_name="lat",
            x2_name="lon",
            verbose=True,
        )
        return data_processor

    def load_ds(self) -> xr.Dataset:
        """Load and preprocess the main gridded dataset.

        Returns
        -------
        xarray.Dataset
            The preprocessed dataset.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement load_ds method")

    def load_aux_ds(self) -> xr.DataArray:
        """Load the auxiliary (land-sea mask) dataset.

        Returns
        -------
        xarray.DataArray
            The auxiliary data array.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement load_aux_ds method")

    def get_dataproc_taskload(
        self,
        dates: Sequence[np.datetime64] | pd.DatetimeIndex | None = None,
        task: Any | None = None,
        data_processor: DataProcessor | None = None,
    ) -> tuple[xr.Dataset, DataProcessor, TaskLoader]:
        """Build the raw dataset, fitted data processor and task loader.

        Parameters
        ----------
        dates : Sequence[numpy.datetime64] | pandas.DatetimeIndex, optional
            Dates used by subclasses to fit the data processor's
            normalisation parameters.
        task : Any, optional
            Reserved for a pre-built task; not currently used by any
            subclass implementation.
        data_processor : deepsensor.data.processor.DataProcessor, optional
            Existing data processor to reuse. If ``None``, a new one is
            created via :meth:`init_data_processor`.

        Returns
        -------
        tuple[xarray.Dataset, deepsensor.data.processor.DataProcessor, deepsensor.data.loader.TaskLoader]
            The raw dataset, the (possibly newly-fitted) data processor, and
            the resulting task loader.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override this method.
        """
        raise NotImplementedError(
            "Subclasses must implement get_dataproc_taskload method"
        )


class CARRA2DataLoader(BaseDataLoader):
    """Load and preprocess CARRA2 data for DeepSensor models.

    Attributes
    ----------
    ancils_path : str
        Path to the directory containing BedMachine ancillary files (basin
        and land-sea mask rasters).
    basin_x, basin_y : xarray.DataArray
        ``x``/``y`` coordinate arrays covering the bounding box of
        ``self.basin``. Only set if ``self.basin is not None``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the CARRA2 data loader.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary. Must additionally contain
            ``config["data"]["ancillary_path"]``.
        """
        super().__init__(config)
        self.ancils_path: str = config["data"]["ancillary_path"]
        if self.basin is not None:
            self.basin_x, self.basin_y = self.load_basin(target_basin=self.basin)

    def load_basin(self, target_basin: str = "SE") -> tuple[xr.DataArray, xr.DataArray]:
        """Compute the bounding ``x``/``y`` coordinates of a basin.

        Reads the CARRA2 (2500 m) basin mask and returns the ``x`` and ``y``
        coordinate values for which the given basin's mask is non-zero along
        at least one row/column.

        Parameters
        ----------
        target_basin : str, default "SE"
            Basin label (as used in the basin mask's ``subregion_labels``
            attribute), e.g. ``"SE"``.

        Returns
        -------
        basin_x : xarray.DataArray
            ``x`` coordinate values covered by ``target_basin``.
        basin_y : xarray.DataArray
            ``y`` coordinate values covered by ``target_basin``.
        """
        basin_ds = xr.open_dataset(os.path.join(self.ancils_path, "basins_2500m.nc"))[
            "subregion"
        ]
        label_to_code = {
            v: int(k) for k, v in json.loads(basin_ds.attrs["subregion_labels"]).items()
        }
        se_code = label_to_code[target_basin]
        se_mask = basin_ds == se_code
        basin_x = basin_ds.x.where(se_mask.any("y"), drop=True)
        basin_y = basin_ds.y.where(se_mask.any("x"), drop=True)
        return basin_x, basin_y

    def load_ds(self) -> xr.Dataset:
        """Load and preprocess the CARRA2 gridded dataset.

        Opens all files matching ``source/region/data_pattern``, selects the
        configured context variables, converts 2 m temperature (``2t``) from
        Kelvin to Celsius, derives anomaly features for any variable ID
        ending in ``anm``, renames the remaining (absolute-value) variables
        to end in ``_abs``, optionally subsets to ``self.basin``'s bounding
        box, and renames the ``x``/``y`` coordinates to ``lon``/``lat``.

        Returns
        -------
        xarray.Dataset
            The preprocessed dataset, with coordinates ``lon``/``lat``.
        """
        if not isinstance(self.context_vars, list):
            self.variables = [self.context_vars]
        if any(isinstance(el, list) for el in self.context_vars):
            self.variables = [var for group in self.context_vars for var in group]

        files = glob.glob(f"{self.source}/{self.region}/{self.data_pattern}")
        ds = xr.open_mfdataset(files)
        ds = ds[[var[:-4] for var in self.variables]]

        if any(v.startswith("2t") for v in self.variables):
            scale = 1.0
            shift = -273.15
            ds["2t"] = ds["2t"] * scale + shift

        for var in self.variables:
            if var[-3:] == "anm":
                target = var[:-4]
                ds.update(
                    self.featurize_vars(
                        var=target,
                        dims=ds[target].dims,
                        chunks=cast(tuple[tuple[int, ...], ...], ds[target].chunks),
                        abs=ds[target],
                    )
                )

        rename_map = {
            var: f"{var}_abs"
            for var in ds.data_vars
            if not (cast(str, var).endswith("_anm") or cast(str, var).endswith("_abs"))
        }

        ds = ds.rename(rename_map)

        if self.basin is not None:
            ds = ds.sel(
                x=slice(float(self.basin_x.min()), float(self.basin_x.max())),
                y=slice(
                    float(self.basin_y.max()), float(self.basin_y.min())
                ),  # y is decreasing
            )

        rename_map = {"x": "lon", "y": "lat"}

        ds = ds.rename(rename_map)

        return ds

    def load_aux_ds(self, resolution: str = "public") -> xr.DataArray:
        """Load the land-sea mask auxiliary data.

        Parameters
        ----------
        resolution : str, default "public"
            Source of the mask. ``"public"`` downloads (and caches) the
            GLDAS land mask for ``self.region``. ``"bedmachine"`` loads the
            higher-resolution BedMachine land-sea mask and, if
            ``self.basin`` is set, subsets it to the basin's bounding box.

        Returns
        -------
        xarray.DataArray
            Land-sea mask, with ``x``/``y`` coordinates renamed to
            ``lon``/``lat`` when loaded from BedMachine.
        """
        # Download static auxiliary features (elevation/land mask)
        if resolution == "public":
            reg_extent = region_extents[self.region]
            # aux_var_IDs = ["elevation"]
            # aux_raw_ds = get_earthenv_auxiliary_data(
            #     aux_var_IDs, reg_extent, "100KM", cache=True, cache_dir=cache_dir
            # )
            lsm_raw_ds = get_gldas_land_mask(
                reg_extent, cache=True, cache_dir=cache_dir
            )

        elif resolution == "bedmachine":
            lsm_raw_ds = xr.open_dataset(
                os.path.join(self.ancils_path, "lsm_2500m.nc")
            )["lsm"]

            if self.basin is not None:
                lsm_raw_ds = lsm_raw_ds.sel(
                    x=slice(float(self.basin_x.min()), float(self.basin_x.max())),
                    y=slice(float(self.basin_y.max()), float(self.basin_y.min())),
                )

            rename_map = {"x": "lon", "y": "lat"}

            lsm_raw_ds = lsm_raw_ds.rename(rename_map)

        return lsm_raw_ds

    def get_dataproc_taskload(
        self,
        dates: Sequence[np.datetime64] | pd.DatetimeIndex | None = None,
        task: Any | None = None,
        data_processor: DataProcessor | None = None,
    ) -> tuple[xr.Dataset, DataProcessor, TaskLoader]:
        """Build the raw dataset, fitted data processor and task loader.

        Loads the raw dataset and (BedMachine) land-sea mask, optionally
        applies the land-sea mask, fits/applies the data processor to the
        selected ``dates`` (to compute normalisation parameters) and to the
        full dataset, adds spatiotemporal auxiliary features, and assembles
        a :class:`TaskLoader` from the context and target variables.

        Parameters
        ----------
        dates : Sequence[numpy.datetime64] | pandas.DatetimeIndex
            Dates used to fit the data processor's normalisation parameters
            (via ``raw_ds.sel(time=dates)``).
        task : Any, optional
            Reserved for a pre-built task; not currently used.
        data_processor : deepsensor.data.processor.DataProcessor, optional
            Existing data processor to reuse. If ``None``, a new one is
            created via :meth:`init_data_processor`.

        Returns
        -------
        tuple[xarray.Dataset, deepsensor.data.processor.DataProcessor, deepsensor.data.loader.TaskLoader]
            The raw (unnormalised) dataset, the fitted data processor, and
            the resulting task loader.
        """
        # Load all data
        raw_ds = self.load_ds()
        lsm_raw_ds = self.load_aux_ds(resolution="bedmachine")
        if self.lsm_mask:
            raw_ds = raw_ds.where(lsm_raw_ds == 1)

        # Data processor
        if data_processor is None:
            data_processor = self.init_data_processor()

        # Apply/learn
        _ = data_processor(raw_ds.sel(time=dates))
        ds = data_processor(raw_ds)
        aux_ds = data_processor(lsm_raw_ds, method="min_max")

        # Spatiotemporal aux
        aux_ds = self.add_spatiotemporal_aux(aux_ds.to_dataset(), ds)

        # Make task loader
        if len(self.variables) > 1:
            context = [ds[sublist] for sublist in self.context_vars] + [aux_ds]
        else:
            context = [ds[self.context_vars], aux_ds]
        target = [ds[self.target_vars]]

        if task is None:
            task_loader = TaskLoader(context=context, target=target)

        return raw_ds, data_processor, task_loader


class HIRHAM5DataLoader(BaseDataLoader):
    """Load and preprocess HIRHAM5 data for DeepSensor models.

    Attributes
    ----------
    ancils_path : str
        Path to the directory containing BedMachine ancillary files (basin
        and ablation-zone mask rasters).
    basin_x, basin_y : xarray.DataArray
        ``x``/``y`` coordinate arrays covering the bounding box of
        ``self.basin``. Only set if ``self.basin is not None``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the HIRHAM5 data loader.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary. Must additionally contain
            ``config["data"]["ancillary_path"]``.
        """
        super().__init__(config)
        self.ancils_path: str = config["data"]["ancillary_path"]
        if self.basin is not None:
            self.basin_x, self.basin_y = self.load_basin(target_basin=self.basin)

    def load_basin(self, target_basin: str = "SE") -> tuple[xr.DataArray, xr.DataArray]:
        """Compute the bounding ``x``/``y`` coordinates of a basin.

        Reads the HIRHAM5 (5500 m) basin mask and returns the ``x`` and ``y``
        coordinate values for which the given basin's mask is non-zero along
        at least one row/column.

        Parameters
        ----------
        target_basin : str, default "SE"
            Basin label (as used in the basin mask's ``subregion_labels``
            attribute), e.g. ``"SE"``.

        Returns
        -------
        basin_x : xarray.DataArray
            ``x`` coordinate values covered by ``target_basin``.
        basin_y : xarray.DataArray
            ``y`` coordinate values covered by ``target_basin``.
        """
        basin_ds = xr.open_dataset(os.path.join(self.ancils_path, "basins_5500m.nc"))[
            "subregion"
        ]
        label_to_code = {
            v: int(k) for k, v in json.loads(basin_ds.attrs["subregion_labels"]).items()
        }
        se_code = label_to_code[target_basin]
        se_mask = basin_ds == se_code
        basin_x = basin_ds.x.where(se_mask.any("y"), drop=True)
        basin_y = basin_ds.y.where(se_mask.any("x"), drop=True)
        return basin_x, basin_y

    def load_ds(self) -> xr.Dataset:
        """Load and preprocess the HIRHAM5 gridded dataset.

        Opens all files matching ``source/region/data_pattern``, clips
        negative snow-melt (``snmel``) values to zero, selects the
        configured context variables, converts near-surface air temperature
        (``tas``) from Kelvin to Celsius, derives anomaly features for any
        variable ID ending in ``anm``, renames the remaining (absolute-value)
        variables to end in ``_abs``, optionally subsets to ``self.basin``'s
        bounding box, and renames the ``x``/``y`` coordinates to
        ``lon``/``lat``.

        Returns
        -------
        xarray.Dataset
            The preprocessed dataset, with coordinates ``lon``/``lat``.
        """
        if not isinstance(self.context_vars, list):
            self.variables = [self.context_vars]
        if any(isinstance(el, list) for el in self.context_vars):
            self.variables = [var for group in self.context_vars for var in group]

        files = glob.glob(f"{self.source}/{self.region}/{self.data_pattern}")
        ds = xr.open_mfdataset(files)

        # preprocess data to fix negative snmel
        ds["snmel"] = ds["snmel"].where(ds["snmel"] >= 0, 0)

        ds = ds[[var[:-4] for var in self.variables]]

        if any(v.startswith("tas") for v in self.variables):
            scale = 1.0
            shift = -273.15
            ds["tas"] = ds["tas"] * scale + shift

        for var in self.variables:
            if var[-3:] == "anm":
                target = var[:-4]
                ds.update(
                    self.featurize_vars(
                        var=target,
                        dims=ds[target].dims,
                        chunks=cast(tuple[tuple[int, ...], ...], ds[target].chunks),
                        abs=ds[target],
                    )
                )

        rename_map = {
            var: f"{var}_abs"
            for var in ds.data_vars
            if not (cast(str, var).endswith("_anm") or cast(str, var).endswith("_abs"))
        }

        ds = ds.rename(rename_map)

        if self.basin is not None:
            ds = ds.sel(
                x=slice(float(self.basin_x.min()), float(self.basin_x.max())),
                y=slice(
                    float(self.basin_y.max()), float(self.basin_y.min())
                ),  # y is decreasing
            )

        rename_map = {"x": "lon", "y": "lat"}
        ds = ds.rename(rename_map)

        return ds

    def load_aux_ds(self, resolution: str = "public") -> xr.DataArray:
        """Load the land-sea/ablation mask auxiliary data.

        Parameters
        ----------
        resolution : str, default "public"
            Source of the mask. ``"public"`` downloads (and caches) the
            GLDAS land mask for ``self.region``. ``"bedmachine"`` loads the
            BedMachine ablation-zone mask, renaming its ``lat``/``lon``
            coordinates to ``x``/``y`` first if necessary, then, if
            ``self.basin`` is set, subsets it to the basin's bounding box.

        Returns
        -------
        xarray.DataArray
            Land-sea/ablation mask, with ``x``/``y`` coordinates renamed to
            ``lon``/``lat`` when loaded from BedMachine.
        """
        # Download static auxiliary features (elevation/land mask)
        if resolution == "public":
            reg_extent = region_extents[self.region]
            # aux_var_IDs = ["elevation"]
            # aux_raw_ds = get_earthenv_auxiliary_data(
            #     aux_var_IDs, reg_extent, "100KM", cache=True, cache_dir=cache_dir
            # )
            lsm_raw_ds = get_gldas_land_mask(
                reg_extent, cache=True, cache_dir=cache_dir
            )

        elif resolution == "bedmachine":
            lsm_raw_ds = xr.open_dataset(
                os.path.join(self.ancils_path, "ablation_5500m.nc")
            )["ablation"]

            if lsm_raw_ds.dims == ("lat", "lon"):
                rename_map = {"lon": "x", "lat": "y"}
                lsm_raw_ds = lsm_raw_ds.rename(rename_map)

            if self.basin is not None:
                lsm_raw_ds = lsm_raw_ds.sel(
                    x=slice(float(self.basin_x.min()), float(self.basin_x.max())),
                    y=slice(float(self.basin_y.max()), float(self.basin_y.min())),
                )

            rename_map = {"x": "lon", "y": "lat"}
            lsm_raw_ds = lsm_raw_ds.rename(rename_map)

        return lsm_raw_ds

    def get_dataproc_taskload(
        self,
        dates: Sequence[np.datetime64] | pd.DatetimeIndex | None = None,
        task: Any | None = None,
        data_processor: DataProcessor | None = None,
    ) -> tuple[xr.Dataset, DataProcessor, TaskLoader]:
        """Build the raw dataset, fitted data processor and task loader.

        Loads the raw dataset and (BedMachine) mask, optionally applies the
        mask, fits/applies the data processor, adds spatiotemporal auxiliary
        features, and assembles a :class:`TaskLoader` from the context and
        target variables.

        When ``self.context_vars`` contains more than one variable group,
        the first group is normalised with the ``"positive_semidefinite"``
        method (used for the target-bearing group, e.g. snow melt) and the
        remaining group with the default method; otherwise all context
        variables are normalised together with the default method.

        Parameters
        ----------
        dates : Sequence[numpy.datetime64] | pandas.DatetimeIndex
            Dates to select (via ``.sel(time=dates)``) before fitting/
            applying the data processor.
        task : Any, optional
            Reserved for a pre-built task; not currently used.
        data_processor : deepsensor.data.processor.DataProcessor, optional
            Existing data processor to reuse. If ``None``, a new one is
            created via :meth:`init_data_processor`.

        Returns
        -------
        tuple[xarray.Dataset, deepsensor.data.processor.DataProcessor, deepsensor.data.loader.TaskLoader]
            The raw (unnormalised) dataset, the fitted data processor, and
            the resulting task loader.
        """
        # Load all data
        raw_ds = self.load_ds()
        lsm_raw_ds = self.load_aux_ds(resolution="bedmachine")
        if self.lsm_mask:
            raw_ds = raw_ds.where(lsm_raw_ds == 1)

        # Data processor
        if data_processor is None:
            data_processor = self.init_data_processor()

        if isinstance(self.context_delta_t, list) and any(
            isinstance(el, list) for el in self.context_delta_t
        ):
            delta_factor_context: list[int] = [
                len(group) for group in self.context_delta_t
            ]
            flat_context_delta_t: list[int] = [
                v for group in self.context_delta_t for v in group
            ] + [0]  # the last 0 is for the aux data
        else:
            raise NotImplementedError(
                "HIRHAM5DataLoader only supports context_delta_t as a nested "
                f"list of per-group deltas, e.g. [[0], [-2, -1, 0]], got {self.context_delta_t!r}."
            )

        max_negative_context_delta_t = min(
            (v for v in flat_context_delta_t if v is not None and v < 0), default=0
        )

        if dates is not None and max_negative_context_delta_t < 0:
            dates_index = pd.DatetimeIndex(dates)
            negative_deltas = {
                v for v in flat_context_delta_t if v is not None and v < 0
            }
            padding = pd.DatetimeIndex(
                [
                    date + pd.Timedelta(days=delta)
                    for date in dates_index
                    for delta in negative_deltas
                ]
            )
            padded_dates = dates_index.union(padding).unique()
        else:
            padded_dates = dates

        # Apply/learn
        aux_ds = data_processor(lsm_raw_ds, method="min_max")

        # Make task loader
        if len(self.variables) > 1:
            ds_target = data_processor(
                raw_ds[self.context_vars[0]].sel(time=padded_dates),
                raw_ds[self.context_vars[0]].sel(time=padded_dates),
                method="positive_semidefinite",
            )
            ds_others = data_processor(
                raw_ds[self.context_vars[1]].sel(time=padded_dates)
            )
            aux_ds = self.add_spatiotemporal_aux(aux_ds.to_dataset(), ds_target)
            context = [
                obj
                for obj, n in zip(
                    [ds_target, ds_others], delta_factor_context, strict=False
                )
                for _ in range(n)
            ] + [aux_ds]
            target = [ds_target[self.target_vars]]
        else:
            ds = data_processor(raw_ds.sel(time=padded_dates))
            ds = data_processor(raw_ds.sel(time=padded_dates))
            aux_ds = self.add_spatiotemporal_aux(aux_ds.to_dataset(), ds)
            context = [
                obj
                for obj, n in zip(
                    [self.context_vars], delta_factor_context, strict=False
                )
                for _ in range(n)
            ] + [aux_ds]
            context = [
                obj
                for obj, n in zip(
                    [self.context_vars], delta_factor_context, strict=False
                )
                for _ in range(n)
            ] + [aux_ds]
            target = [ds[self.target_vars]]

        if task is None:
            task_loader = TaskLoader(
                context=context, target=target, context_delta_t=flat_context_delta_t
            )

        return raw_ds, data_processor, task_loader


class HIRHAM5ZarrDataLoader(BaseDataLoader):
    """Load and preprocess HIRHAM5 data stored in Zarr for DeepSensor models.

    Attributes
    ----------
    ancils_path : str
        Path to the directory containing ancillary files.
    basin_x, basin_y : xarray.DataArray
        ``x``/``y`` coordinate arrays covering the bounding box of
        ``self.basin``. Only set if ``self.basin is not None``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the HIRHAM5 Zarr data loader.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary. Must additionally contain
            ``config["data"]["ancillary_path"]``.
        """
        super().__init__(config)
        self.ancils_path: str = config["data"]["ancillary_path"]
        self.train_start: str = config["data"]["train_start"]
        self.val_end: str = config["data"]["val_end"]
        if self.basin is not None:
            self.basin_x, self.basin_y = self.load_basin(target_basin=self.basin)

    def load_basin(self, target_basin: str = "SE") -> tuple[xr.DataArray, xr.DataArray]:
        """Compute the bounding ``x``/``y`` coordinates of a basin.

        Reads the ``GRLmask.zarr`` basin mask and returns the ``x`` and ``y``
        coordinate values for which the given basin's mask is non-zero along
        at least one row/column.

        Parameters
        ----------
        target_basin : str, default "SE"
            Basin label. One of ``"N"``, ``"NE"``, ``"CE"``, ``"SE"``,
            ``"SW"``, ``"NW"``.

        Returns
        -------
        basin_x : xarray.DataArray
            ``x`` coordinate values covered by ``target_basin``.
        basin_y : xarray.DataArray
            ``y`` coordinate values covered by ``target_basin``.
        """
        basin_ds = xr.open_zarr(
            os.path.join(self.ancils_path, "GRLmask.zarr"), chunks="auto"
        )["maskbas"]
        basin_ds = basin_ds.set_index(z=["y", "x"]).unstack("z").compute()
        label_to_code = {"N": 1, "NE": 2, "CE": 3, "SE": 4, "SW": 5, "NW": 6}
        se_code = label_to_code[target_basin]
        se_mask = basin_ds == se_code
        basin_x = basin_ds.x.where(se_mask.any("y"), drop=True)
        basin_y = basin_ds.y.where(se_mask.any("x"), drop=True)
        return basin_x, basin_y

    def load_ds(self) -> xr.Dataset:
        """Load and preprocess the HIRHAM5 Zarr gridded dataset.

        Opens the Zarr store at ``source/data_pattern``, unstacks the
        ``z`` multi-index back into ``x``/``y``, drops the redundant
        ``lat``/``lon`` variables, clips negative snow-melt (``snmel``)
        values to zero, selects the configured context variables, converts
        near-surface air temperature (``tas``) from Kelvin to Celsius,
        derives anomaly features for any variable ID ending in ``anm``,
        renames the remaining (absolute-value) variables to end in
        ``_abs``, optionally subsets to ``self.basin``'s bounding box, and
        renames the ``x``/``y`` coordinates to ``lon``/``lat``.

        Returns
        -------
        xarray.Dataset
            The preprocessed dataset, with coordinates ``lon``/``lat``.
        """
        if not isinstance(self.context_vars, list):
            self.variables = [self.context_vars]
        if any(isinstance(el, list) for el in self.context_vars):
            self.variables = [var for group in self.context_vars for var in group]

        ds = xr.open_zarr(os.path.join(self.source, self.data_pattern), chunks="auto")

        ds = ds.set_index(z=["y", "x"]).unstack("z")
        ds = ds.drop_vars(["lat", "lon"])

        # preprocess data to fix negative snmel
        ds["snmel"] = ds["snmel"].where(ds["snmel"] >= 0, 0)

        # split variables into those that are 10-year averages and those that are not
        self.long_vars = [var for var in self.variables if var.endswith("_10yr-avg")]
        self.base_vars = [var for var in self.variables if var not in self.long_vars]

        # preprocess base variables (non-10-year averages)
        ds = ds[[var[:-4] for var in self.base_vars]]

        if any(v.startswith("tas") for v in self.base_vars):
            scale = 1.0
            shift = -273.15
            ds["tas"] = ds["tas"] * scale + shift

        for var in self.base_vars:
            if var[-3:] == "anm":
                target = var[:-4]
                ds.update(
                    self.featurize_vars(
                        var=target,
                        dims=ds[target].dims,
                        chunks=cast(tuple[tuple[int, ...], ...], ds[target].chunks),
                        abs=ds[target],
                    )
                )

        rename_map = {
            var: f"{var}_abs"
            for var in ds.data_vars
            if not (cast(str, var).endswith("_anm") or cast(str, var).endswith("_abs"))
        }

        ds = ds.rename(rename_map)

        # open the 10-year average dataset and merge it with the base dataset
        if len(self.long_vars) > 0:
            ds_long = xr.open_zarr(
                os.path.join(self.source, "base_dataset_10yr.zarr"), chunks="auto"
            )
            ds_long = ds_long.set_index(z=["y", "x"]).unstack("z")
            ds_long = ds_long.drop_vars(["lat", "lon"])
            if any(v.startswith("tas") for v in self.long_vars):
                scale = 1.0
                shift = -273.15
                ds_long["tas_10yr-avg"] = ds_long["tas_10yr-avg"] * scale + shift
            ds = ds.merge(ds_long[self.long_vars])

        if self.basin is not None:
            ds = ds.sel(
                x=slice(float(self.basin_x.min()), float(self.basin_x.max())),
                y=slice(
                    float(self.basin_y.min()), float(self.basin_y.max())
                ),  # y is decreasing
            )

        rename_map = {"x": "lon", "y": "lat"}
        ds = ds.rename(rename_map)

        # update the base_vars list to exclude any target variables, so that they are not included in the context forcing variables
        self.base_vars = [var for var in self.base_vars if var not in self.target_vars]

        return ds

    def load_aux_ds(self, resolution: str = "public") -> xr.DataArray:
        """Load the land-sea/glacier mask auxiliary data.

        Parameters
        ----------
        resolution : str, default "public"
            Source of the mask. ``"public"`` downloads (and caches) the
            GLDAS land mask for ``self.region``. ``"bedmachine"`` loads the
            ``glacGRL`` mask from ``GRLmask.zarr``, unstacks its ``z``
            multi-index, drops the ``lat``/``lon`` variables, fills missing
            values with 0, and, if ``self.basin`` is set, subsets it to the
            basin's bounding box.

        Returns
        -------
        xarray.DataArray
            Land-sea/glacier mask, with ``x``/``y`` coordinates renamed to
            ``lon``/``lat`` when loaded from BedMachine.
        """
        # Download static auxiliary features (elevation/land mask)
        if resolution == "public":
            reg_extent = region_extents[self.region]
            lsm_raw_ds = get_gldas_land_mask(
                reg_extent, cache=True, cache_dir=cache_dir
            )

        elif resolution == "bedmachine":
            mask_native = xr.open_zarr(
                os.path.join(self.ancils_path, "GRLmask.zarr"), chunks="auto"
            )
            mask_native = mask_native["glacGRL"].set_index(z=["y", "x"]).unstack("z")
            lsm_raw_ds = mask_native.drop_vars(["lat", "lon"]).fillna(0)

            if self.basin is not None:
                lsm_raw_ds = lsm_raw_ds.sel(
                    x=slice(float(self.basin_x.min()), float(self.basin_x.max())),
                    y=slice(float(self.basin_y.min()), float(self.basin_y.max())),
                )

            rename_map = {"x": "lon", "y": "lat"}
            lsm_raw_ds = lsm_raw_ds.rename(rename_map)

        return lsm_raw_ds

    def get_dataproc_taskload(
        self,
        dates: Sequence[np.datetime64] | pd.DatetimeIndex | None = None,
        task: Any | None = None,
        data_processor: DataProcessor | None = None,
    ) -> tuple[xr.Dataset, DataProcessor, TaskLoader]:
        """Build the raw dataset, fitted data processor and task loader.

        Loads the raw dataset and (BedMachine) mask, optionally applies the
        mask, fits/applies the data processor, adds spatiotemporal auxiliary
        features, and assembles a :class:`TaskLoader` from the context and
        target variables.

        When ``self.context_vars`` contains more than one variable group,
        the first group is normalised with the ``"positive_semidefinite"``
        method (used for the target-bearing group, e.g. snow melt) and the
        remaining group with the default method; otherwise all context
        variables are normalised together with the default method.

        Parameters
        ----------
        dates : Sequence[numpy.datetime64] | pandas.DatetimeIndex
            Dates to select (via ``.sel(time=dates)``) before fitting/
            applying the data processor.
        task : Any, optional
            Reserved for a pre-built task; not currently used.
        data_processor : deepsensor.data.processor.DataProcessor, optional
            Existing data processor to reuse. If ``None``, a new one is
            created via :meth:`init_data_processor`.

        Returns
        -------
        tuple[xarray.Dataset, deepsensor.data.processor.DataProcessor, deepsensor.data.loader.TaskLoader]
            The raw (unnormalised) dataset, the fitted data processor, and
            the resulting task loader.
        """
        # Load all data
        raw_ds = self.load_ds()
        lsm_raw_ds = self.load_aux_ds(resolution="bedmachine")
        if self.lsm_mask:
            raw_ds = raw_ds.where(lsm_raw_ds == 1)

        # Data processor
        if data_processor is None:
            data_processor = self.init_data_processor()

        if isinstance(self.context_delta_t, list) and any(
            isinstance(el, list) for el in self.context_delta_t
        ):
            delta_factor_context: list[int] = [
                len(group) for group in self.context_delta_t
            ]
            flat_context_delta_t: list[int] = [
                v for group in self.context_delta_t for v in group
            ] + [0]  # the last 0 is for the aux data
        else:
            raise NotImplementedError(
                "HIRHAM5ZarrDataLoader only supports context_delta_t as a nested "
                f"list of per-group deltas, e.g. [[0], [-2, -1, 0]], got {self.context_delta_t!r}."
            )

        max_negative_context_delta_t = min(
            (v for v in flat_context_delta_t if v is not None and v < 0), default=0
        )

        if dates is not None and max_negative_context_delta_t < 0:
            dates_index = pd.DatetimeIndex(dates)
            negative_deltas = {
                v for v in flat_context_delta_t if v is not None and v < 0
            }
            padding = pd.DatetimeIndex(
                [
                    date + pd.Timedelta(days=delta)
                    for date in dates_index
                    for delta in negative_deltas
                ]
            )
            padded_dates = dates_index.union(padding).unique()
        else:
            padded_dates = dates

        if isinstance(self.context_delta_t, list) and any(
            isinstance(el, list) for el in self.context_delta_t
        ):
            delta_factor_context: list[int] = [
                len(group) for group in self.context_delta_t
            ]
            flat_context_delta_t: list[int] = [
                v for group in self.context_delta_t for v in group
            ] + [0]  # the last 0 is for the aux data
        else:
            raise NotImplementedError(
                "HIRHAM5ZarrDataLoader only supports context_delta_t as a nested "
                f"list of per-group deltas, e.g. [[0], [-2, -1, 0]], got {self.context_delta_t!r}."
            )

        max_negative_context_delta_t = min(
            (v for v in flat_context_delta_t if v is not None and v < 0), default=0
        )

        if dates is not None and max_negative_context_delta_t < 0:
            dates_index = pd.DatetimeIndex(dates)
            negative_deltas = {
                v for v in flat_context_delta_t if v is not None and v < 0
            }
            padding = pd.DatetimeIndex(
                [
                    date + pd.Timedelta(days=delta)
                    for date in dates_index
                    for delta in negative_deltas
                ]
            )
            padded_dates = dates_index.union(padding).unique()
        else:
            padded_dates = dates

        # Apply/learn
        aux_ds = data_processor(lsm_raw_ds, method="min_max")

        # Make task loader
        if len(self.variables) > 1:
            ds_target = data_processor(
                raw_ds[self.target_vars].sel(time=padded_dates),
                method="positive_semidefinite",
            )
            ds_others = data_processor(raw_ds[self.base_vars].sel(time=padded_dates))

            aux_ds = self.add_spatiotemporal_aux(
                aux_ds.to_dataset(name="GRLmask"), ds_target
            )
            if len(self.long_vars) > 0:
                ds_others_long = data_processor(raw_ds[self.long_vars].sel(time=dates))
                context = [
                    obj
                    for obj, n in zip(
                        [ds_target, ds_others, ds_others_long],
                        delta_factor_context,
                        strict=False,
                    )
                    for _ in range(n)
                ] + [aux_ds]
            else:
                context = [
                    obj
                    for obj, n in zip(
                        [ds_target, ds_others], delta_factor_context, strict=False
                    )
                    for _ in range(n)
                ] + [aux_ds]
        else:
            ds = data_processor(raw_ds.sel(time=padded_dates))
            aux_ds = self.add_spatiotemporal_aux(aux_ds.to_dataset(name="GRLmask"), ds)
            context = [
                obj
                for obj, n in zip(
                    [ds[self.context_vars]], delta_factor_context, strict=False
                )
                for _ in range(n)
            ] + [aux_ds]
            ds_target = [ds[self.target_vars]]

        if task is None:
            task_loader = TaskLoader(
                context=context, target=ds_target, context_delta_t=flat_context_delta_t
            )

        return raw_ds, data_processor, task_loader
