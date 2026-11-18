"""Script for creating ZARR-analysis ready base dataset from netcdf files.

Source: https://github.com/eschlager/MeltEmulation
Created on 05.03.2024, Edited 16.07.2026
@author: eschlager
@modified: acocac
"""

import logging
import os
from collections.abc import Mapping, Sequence
from typing import TypeVar

import dask
import hydra
import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar
from omegaconf import DictConfig, OmegaConf

from ds_greenland.utils.data import ten_year_store_name

logger = logging.getLogger(__name__)

XarrayT = TypeVar("XarrayT", xr.Dataset, xr.DataArray)


def flatten_spatial(obj: XarrayT, dropna: bool = True) -> XarrayT:
    """Stack the ``x``/``y`` dimensions into a single ``z`` dimension.

    Parameters
    ----------
    obj : xr.Dataset or xr.DataArray
        Gridded data with ``x`` and ``y`` dimensions.
    dropna : bool, optional
        Drop ``z`` locations where every value is NaN. Default ``True``.

    Returns
    -------
    xr.Dataset or xr.DataArray
        ``obj`` with ``x``/``y`` flattened into ``z`` (``x``/``y`` kept as
        non-index coordinates).

    """
    flat = obj.stack(z=("x", "y")).reset_index("z")
    if dropna:
        flat = flat.dropna(dim="z", how="all")
    return flat


def create_coords_dataset(src: xr.Dataset) -> xr.Dataset:
    """Extract the original ``x``/``y``/``lat``/``lon`` grid for later reconstruction.

    Parameters
    ----------
    src : xr.Dataset
        Source dataset on the native grid (e.g. ``GRLmask.nc``).

    Returns
    -------
    xr.Dataset
        Dataset holding only ``x``/``y`` (as coordinates when they are
        coordinates in ``src``, otherwise as variables) and ``lat``/``lon``
        (with their original dims), carrying ``src``'s attributes.

    """
    meta = xr.Dataset()
    for dim in ("x", "y"):
        if dim in src.coords:
            meta = meta.assign_coords({dim: (dim, src[dim].values)})
        elif dim in src.variables:
            meta[dim] = (dim, src[dim].values)

    # lat/lon may be 1D or 2D; include them exactly as in source
    for name in ("lat", "lon"):
        if name in src:
            meta[name] = (src[name].dims, src[name].values)

    meta.attrs = src.attrs
    return meta


def clip_to_ranges(
    ds: xr.Dataset, var_ranges: Mapping[str, Sequence[float | None]]
) -> xr.Dataset:
    """Clip variables to their valid ranges, logging how many samples were clipped.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to clip.
    var_ranges : Mapping[str, Sequence[float | None]]
        ``{var: (vmin, vmax)}``; ``None`` leaves that side unbounded. Variables
        missing from ``ds`` are logged and skipped.

    Returns
    -------
    xr.Dataset
        Copy of ``ds`` with the listed variables clipped.

    """
    ds = ds.copy()
    for var, (vmin, vmax) in var_ranges.items():
        if var not in ds.data_vars:
            logger.info("%s: not found in dataset", var)
            continue

        data = ds[var]
        logger.info(" check if %s in [%s, %s]...", var, vmin, vmax)
        if vmin is not None:
            n_clipped_min = (data < vmin).sum().compute().item()
            if n_clipped_min > 0:
                logger.info("   # samples clipped at min: %d", n_clipped_min)
        if vmax is not None:
            n_clipped_max = (data > vmax).sum().compute().item()
            if n_clipped_max > 0:
                logger.info("   # samples clipped at max: %d", n_clipped_max)
        ds[var] = data.clip(min=vmin, max=vmax)
    return ds


def build_base_dataset(ds: xr.Dataset, chunks: Mapping[str, int]) -> xr.Dataset:
    """Flatten the grid, drop all-NaN locations and add the ``energyin`` variable.

    Parameters
    ----------
    ds : xr.Dataset
        Daily gridded dataset with ``time``/``x``/``y`` dims, containing at
        least ``tas``, ``ahfl``, ``ahfs`` and ``dlwrad``.
    chunks : Mapping[str, int]
        Chunk sizes for the output (over ``time`` and ``z``).

    Returns
    -------
    xr.Dataset
        Dataset over ``time``/``z`` with an extra
        ``energyin = ahfl + ahfs + dlwrad`` variable.

    """
    logger.info("Flatten spatial dimension...")
    ds = flatten_spatial(ds, dropna=False)

    logger.info("Drop nans...")
    # compute null only from tas, because either all vars or none are nan for a specific location!
    valid = ~ds["tas"].isnull().all(dim="time").compute()
    logger.info("amount of all z: %d", ds.sizes["z"])
    ds = ds.isel(z=valid)
    logger.info("amount of non-nan z: %d", ds.sizes["z"])

    logger.info("Rechunk time dimension...")
    ds = ds.chunk(dict(chunks))
    ds["energyin"] = (
        ds["ahfl"].astype(np.float32)
        + ds["ahfs"].astype(np.float32)
        + ds["dlwrad"].astype(np.float32)
    )
    return ds


def rolling_means(
    ds: xr.Dataset,
    variables: Sequence[str],
    window: int,
    start_date: str,
    chunks: Mapping[str, int],
) -> xr.Dataset:
    """Trailing rolling means over ``time``, renamed ``<var>_10yr-avg``.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a ``time`` dimension.
    variables : Sequence[str]
        Variables of ``ds`` to average.
    window : int
        Window length in time steps; only full windows give a value.
    start_date : str
        First date kept in the output.
    chunks : Mapping[str, int]
        Chunk sizes for the output.

    Returns
    -------
    xr.Dataset
        Rolling means from ``start_date`` onwards.

    """
    means = ds[list(variables)].rolling(time=window, min_periods=window).mean()
    means = means.sel(time=slice(start_date, None))
    means = means.rename({var: f"{var}_10yr-avg" for var in means.data_vars})
    return means.chunk(dict(chunks))


def create_basefile(params: DictConfig) -> None:
    """Build the HIRHAM5 zarr base files from the interim NetCDF files.

    Reads ``GRLmask.nc``/``GRLzones.nc`` from ``params.aux_dir`` and
    ``Daily2D_GRL_<year>.nc`` (``params.start_year - 10`` to ``params.end_year``,
    the extra 10 years feeding the rolling means) from ``params.daily_dir``,
    both relative to ``params.interim_path``. The base dataset keeps those
    extra 10 years; the rolling means start at ``params.rolling.start_date``.
    Writes ``coords_only.zarr``, ``GRLmask.zarr``, ``GRLzones.zarr``,
    ``base_dataset_<start>_<end>.zarr`` and ``base_dataset_<start>_<end>_10yr.zarr``
    to ``<params.output_dir>/<folder>/v_<params.version:02d>``, where
    ``<folder>`` is ``HIRHAM5-ERAInterim``, ``HIRHAM5-CESM2`` or
    ``HIRHAM5-ECEARTH3`` for ``params.daily_dir`` ``erai``, ``cesm2`` or
    ``ecearth3``, ``<start>`` is ``params.start_year`` and ``<end>`` is
    ``params.end_year``.

    Parameters
    ----------
    params : DictConfig
        The ``preprocessing`` section of the composed config.

    Raises
    ------
    KeyError
        If ``params.daily_dir`` is not ``erai``, ``cesm2`` or ``ecearth3``.

    """
    folder_mapping = {
        "erai": "HIRHAM5-ERAInterim",
        "erai_albedo": "HIRHAM5-ERAInterim-Albedo",
        "cesm2": "HIRHAM5-CESM2",
        "ecearth3": "HIRHAM5-ECEARTH3",
    }
    out_path = os.path.join(
        params.output_dir, f"{folder_mapping[params.daily_dir]}/v_{params.version:02d}"
    )
    os.makedirs(out_path, exist_ok=True)
    in_path_aux = os.path.join(params.interim_path, params.aux_dir)
    in_path_h5 = os.path.join(params.interim_path, params.daily_dir)
    load_chunks: dict[str, int] = OmegaConf.to_container(params.load_chunks)  # type: ignore[assignment]
    save_chunks: dict[str, int] = OmegaConf.to_container(params.save_chunks)  # type: ignore[assignment]

    # %% --------- Create file with original coordinates for later reconstruction ---------
    with xr.open_dataset(os.path.join(in_path_aux, "GRLmask.nc")) as src:
        meta = create_coords_dataset(src)
    logger.info("Coordinates:\n%s", meta)
    filepath = os.path.join(out_path, "coords_only.zarr")
    meta.to_zarr(filepath, mode="w", consolidated=True)
    logger.info("Coordinates saved to %s", filepath)

    # %% -------------------- Create stacked files for AuxFiles --------------------
    filepath = os.path.join(out_path, "GRLmask.zarr")
    with xr.open_dataset(os.path.join(in_path_aux, "GRLmask.nc")) as ds_basins:
        flatten_spatial(ds_basins[["glacGRL", "maskbas"]]).to_zarr(
            filepath, mode="w", consolidated=True
        )
    logger.info("GRL mask saved to %s", filepath)

    filepath = os.path.join(out_path, "GRLzones.zarr")
    with xr.open_dataset(os.path.join(in_path_aux, "GRLzones.nc")) as ds_zones:
        flatten_spatial(ds_zones["zones"]).to_zarr(
            filepath, mode="w", consolidated=True
        )
    logger.info("GRL zones saved to %s", filepath)

    # %% Correct daily files: clip to valid ranges
    logger.info("Load all daily files ...")
    years = range(params.start_year - 10, params.end_year + 1)
    ds_all = xr.open_mfdataset(
        [os.path.join(in_path_h5, f"Daily2D_GRL_{year}.nc") for year in years],
        concat_dim="time",
        combine="nested",
        chunks=load_chunks,
    ).drop_vars(list(params.drop_vars))
    ds_all = ds_all.sortby("time")

    logger.info("Correct data ranges ...")
    var_ranges: dict[str, list[float | None]] = OmegaConf.to_container(  # type: ignore[assignment]
        params.var_ranges
    )
    ds_all = clip_to_ranges(ds_all, var_ranges)

    # %% -------------------- Restructure dataset, and save daily data to zarr files --------------------
    filepath = os.path.join(
        out_path, f"base_dataset_{params.start_year}_{params.end_year}.zarr"
    )
    logger.info("Re-structure dataset...")
    ds = build_base_dataset(ds_all, save_chunks)
    logger.info("Saving data to %s...", filepath)
    with ProgressBar():
        ds.to_zarr(filepath, mode="w", consolidated=True)

    # %% -------------------- Create 10yr rolling averages --------------------
    outpath_roll = ten_year_store_name(filepath)
    logger.info("Create 10yrs rolling mean for %s...", list(params.rolling.vars))
    rolling_window = 3652  # days (~10 years)
    means_10yr = rolling_means(
        ds,
        list(params.rolling.vars),
        rolling_window,
        params.rolling.start_date,
        save_chunks,
    )
    logger.info("Saving 10years rolling mean results to %s ...", outpath_roll)
    with ProgressBar():
        means_10yr.to_zarr(outpath_roll, mode="w", consolidated=True)


@hydra.main(
    version_base=None, config_path="../configs", config_name="hirham5_create_basefile"
)
def main(config: DictConfig) -> None:
    """Run the HIRHAM5 base file preprocessing."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))
    dask.config.set(scheduler="threads")
    create_basefile(config.preprocessing)


if __name__ == "__main__":
    main()
