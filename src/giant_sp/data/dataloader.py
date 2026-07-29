"""
Data loading and preprocessing utilities for DeepSensor
"""

import glob
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

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
    """Base class for data loaders"""

    def __init__(self, config: dict):
        """
        Initialize data loader

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.source = Path(config["data"]["data_path"])
        self.data_pattern = config["data"]["data_pattern"]
        self.region = config["data"]["region"]
        self.basin = config["data"]["basin"]
        self.context_vars = config["data"]["context_var_IDs"]
        self.target_vars = config["data"]["target_var_IDs"]
        self.train_start = self.config["data"]["train_start"]
        self.train_end = self.config["data"]["train_end"]
        self.val_start = config["data"]["val_start"]
        self.val_end = config["data"]["val_end"]
        self.lsm_mask = config["dataloader"]["lsm_mask"]
        self.frequency = config["dataloader"]["frequency"]
        self.variables = []

    def add_spatiotemporal_aux(self, aux_ds, ref_ds):
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

    def compute_gridded_anomalies(self, var_ID, da):
        """
        Load or compute the daily climatology of the xarray variable over
        the training dates, then subtract
        from the original DataArray to compute anomalies.

        TODO:
        - option for other temporal resolution (don't pin to daily with time.dayofyear)
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
        self, var: str, dims: Sequence[str], chunks: Sequence[str], abs: np.ndarray
    ) -> Mapping[str, xr.Variable]:
        """Derives features

        Args:
          name: Base variable name from which features are derived.
          dims: List of the output feature dimensions.
          abs: Array of abs values.

        Returns:
          Dictionary of xarray variables derived.
        """

        ds = xr.Dataset()

        anom = self.compute_gridded_anomalies(var, abs)

        ds[var + "_anm"] = anom
        ds = ds.chunk(dict(zip(dims, chunks, strict=True)))

        return ds

    def init_data_processor(self):
        data_processor = DataProcessor(
            x1_name="lat",
            x2_name="lon",
            verbose=True,
        )
        return data_processor

    def load_ds(self):
        raise NotImplementedError("Subclasses must implement load_ds method")

    def load_aux_ds(self):
        raise NotImplementedError("Subclasses must implement load_aux_ds method")

    def get_dataproc_taskload(self, task=None, data_processor=None):
        raise NotImplementedError(
            "Subclasses must implement get_dataproc_taskload method"
        )


class CARRA2DataLoader(BaseDataLoader):
    """Load and preprocess CARRA2 data for DeepSensor models"""

    def __init__(self, config: dict):
        """
        Initialize data loader

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.bm_ancils_path = config["data"]["bedmachine_ancillary_path"]
        if self.basin is not None:
            self.basin_x, self.basin_y = self.load_basin(target_basin=self.basin)

    def load_basin(self, target_basin="SE"):
        basin_ds = xr.open_dataset(
            os.path.join(self.bm_ancils_path, "basins_2500m.nc")
        )["subregion"]
        label_to_code = {
            v: int(k) for k, v in json.loads(basin_ds.attrs["subregion_labels"]).items()
        }
        se_code = label_to_code[target_basin]
        se_mask = basin_ds == se_code
        basin_x = basin_ds.x.where(se_mask.any("y"), drop=True)
        basin_y = basin_ds.y.where(se_mask.any("x"), drop=True)
        return basin_x, basin_y

    def load_ds(self):
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
                        chunks=ds[target].chunks,
                        abs=ds[target],
                    )
                )

        rename_map = {
            var: f"{var}_abs"
            for var in ds.data_vars
            if not (var.endswith("_anm") or var.endswith("_abs"))
        }

        # apply rename
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

    def load_aux_ds(self, resolution="public"):
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
                os.path.join(self.bm_ancils_path, "lsm_2500m.nc")
            )["lsm"]

            if self.basin is not None:
                lsm_raw_ds = lsm_raw_ds.sel(
                    x=slice(float(self.basin_x.min()), float(self.basin_x.max())),
                    y=slice(float(self.basin_y.max()), float(self.basin_y.min())),
                )

            rename_map = {"x": "lon", "y": "lat"}

            lsm_raw_ds = lsm_raw_ds.rename(rename_map)

        return lsm_raw_ds

    def get_dataproc_taskload(self, dates, task=None, data_processor=None):
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
    """Load and preprocess HIRHAM5 data for DeepSensor models"""

    def __init__(self, config: dict):
        """
        Initialize data loader

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.bm_ancils_path = config["data"]["bedmachine_ancillary_path"]
        if self.basin is not None:
            self.basin_x, self.basin_y = self.load_basin(target_basin=self.basin)

    def load_basin(self, target_basin="SE"):
        basin_ds = xr.open_dataset(
            os.path.join(self.bm_ancils_path, "basins_5500m.nc")
        )["subregion"]
        label_to_code = {
            v: int(k) for k, v in json.loads(basin_ds.attrs["subregion_labels"]).items()
        }
        se_code = label_to_code[target_basin]
        se_mask = basin_ds == se_code
        basin_x = basin_ds.x.where(se_mask.any("y"), drop=True)
        basin_y = basin_ds.y.where(se_mask.any("x"), drop=True)
        return basin_x, basin_y

    # Load gridded
    def load_ds(self):
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
                        chunks=ds[target].chunks,
                        abs=ds[target],
                    )
                )

        rename_map = {
            var: f"{var}_abs"
            for var in ds.data_vars
            if not (var.endswith("_anm") or var.endswith("_abs"))
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

    def load_aux_ds(self, resolution="public"):
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
                os.path.join(self.bm_ancils_path, "ablation_5500m.nc")
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

    def get_dataproc_taskload(self, dates, task=None, data_processor=None):
        # Load all data
        raw_ds = self.load_ds()
        lsm_raw_ds = self.load_aux_ds(resolution="bedmachine")
        if self.lsm_mask:
            raw_ds = raw_ds.where(lsm_raw_ds == 1)

        # Data processor
        if data_processor is None:
            data_processor = self.init_data_processor()

        # Apply/learn
        aux_ds = data_processor(lsm_raw_ds, method="min_max")

        # Spatiotemporal aux
        aux_ds = self.add_spatiotemporal_aux(aux_ds.to_dataset(), ds_target)

        # Make task loader
        if len(self.variables) > 1:
            ds_target = data_processor(
                raw_ds[self.context_vars[0]].sel(time=dates),
                method="positive_semidefinite",
            )
            ds_others = data_processor(raw_ds[self.context_vars[1]].sel(time=dates))
            context = [ds_target, ds_others, aux_ds]
            target = [ds_target[self.target_vars]]
        else:
            ds = data_processor(raw_ds.sel(time=dates))
            context = [ds[self.context_vars], aux_ds]
            target = [ds[self.target_vars]]

        if task is None:
            task_loader = TaskLoader(context=context, target=target)

        return raw_ds, data_processor, task_loader


class HIRHAM5ZarrDataLoader(BaseDataLoader):
    """Load and preprocess HIRHAM5 data for DeepSensor models"""

    def __init__(self, config: dict):
        """
        Initialize data loader

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.bm_ancils_path = config["data"]["bedmachine_ancillary_path"]
        self.train_start = config["data"]["train_start"]
        self.val_end = config["data"]["val_end"]
        if self.basin is not None:
            self.basin_x, self.basin_y = self.load_basin(target_basin=self.basin)

    def load_basin(self, target_basin="SE"):
        basin_ds = xr.open_zarr(
            os.path.join(self.source, "GRLmask.zarr"), chunks="auto"
        )["maskbas"]
        basin_ds = basin_ds.set_index(z=["y", "x"]).unstack("z").compute()
        label_to_code = {"N": 1, "NE": 2, "CE": 3, "SE": 4, "SW": 5, "NW": 6}
        se_code = label_to_code[target_basin]
        se_mask = basin_ds == se_code
        basin_x = basin_ds.x.where(se_mask.any("y"), drop=True)
        basin_y = basin_ds.y.where(se_mask.any("x"), drop=True)
        return basin_x, basin_y

    # Load gridded
    def load_ds(self):
        if not isinstance(self.context_vars, list):
            self.variables = [self.context_vars]
        if any(isinstance(el, list) for el in self.context_vars):
            self.variables = [var for group in self.context_vars for var in group]

        ds = xr.open_zarr(os.path.join(self.source, self.data_pattern), chunks="auto")
        ds = ds.set_index(z=["y", "x"]).unstack("z")
        ds = ds.drop_vars(["lat", "lon"])

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
                        chunks=ds[target].chunks,
                        abs=ds[target],
                    )
                )

        rename_map = {
            var: f"{var}_abs"
            for var in ds.data_vars
            if not (var.endswith("_anm") or var.endswith("_abs"))
        }

        ds = ds.rename(rename_map)

        if self.basin is not None:
            ds = ds.sel(
                x=slice(float(self.basin_x.min()), float(self.basin_x.max())),
                y=slice(
                    float(self.basin_y.min()), float(self.basin_y.max())
                ),  # y is decreasing
            )

        rename_map = {"x": "lon", "y": "lat"}
        ds = ds.rename(rename_map)

        return ds

    def load_aux_ds(self, resolution="public"):
        # Download static auxiliary features (elevation/land mask)
        if resolution == "public":
            reg_extent = region_extents[self.region]
            lsm_raw_ds = get_gldas_land_mask(
                reg_extent, cache=True, cache_dir=cache_dir
            )

        elif resolution == "bedmachine":
            mask_native = xr.open_zarr(
                os.path.join(self.source, "GRLmask.zarr"), chunks="auto"
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

    def get_dataproc_taskload(self, dates, task=None, data_processor=None):
        # Load all data
        raw_ds = self.load_ds()
        lsm_raw_ds = self.load_aux_ds(resolution="bedmachine")
        if self.lsm_mask:
            raw_ds = raw_ds.where(lsm_raw_ds == 1)

        # Data processor
        if data_processor is None:
            data_processor = self.init_data_processor()

        # Apply/learn
        aux_ds = data_processor(lsm_raw_ds, method="min_max")

        # Spatiotemporal aux
        aux_ds = self.add_spatiotemporal_aux(
            aux_ds.to_dataset(name="GRLmask"), ds_target
        )

        # Make task loader
        # Make task loader
        if len(self.variables) > 1:
            ds_target = data_processor(
                raw_ds[self.context_vars[0]].sel(time=dates),
                method="positive_semidefinite",
            )
            ds_others = data_processor(raw_ds[self.context_vars[1]].sel(time=dates))
            context = [ds_target, ds_others, aux_ds]
            target = [ds_target[self.target_vars]]
        else:
            ds = data_processor(raw_ds.sel(time=dates))
            context = [ds[self.context_vars], aux_ds]
            target = [ds[self.target_vars]]

        if task is None:
            task_loader = TaskLoader(context=context, target=target)

        return raw_ds, data_processor, task_loader
