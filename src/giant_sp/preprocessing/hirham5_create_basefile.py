"""
Source: https://github.com/eschlager/MeltEmulation
Created on 05.03.2024, Edited 16.07.2026
@author: eschlager
@modified: acocac
Script for creating ZARR-analysis ready base dataset from netcdf files;
"""

import logging
import os
import sys

import dask
import xarray as xr
from dask.diagnostics import ProgressBar

script_dir = os.path.abspath(os.path.dirname(__file__))
project_dir = os.path.sep.join([script_dir, "..", ".."])
sys.path.append(os.path.sep.join([project_dir, "src"]))


def setup_logging() -> None:
    """
    Configure the root logger for console output.

    Sets the root logger level to ``INFO`` and attaches a
    :class:`logging.StreamHandler` with a timestamped format.
    """
    log_level = logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


if __name__ == "__main__":
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    dask.config.set(scheduler="threads")

    out_path = "/gws/pw/j07/aria_giant/wip/draft/acocac/data/hirham5/processed/HIRHAM5-ERAInterim/v_03"
    os.makedirs(out_path, exist_ok=True)

    interim_path = "/gws/pw/j07/aria_giant/wip/draft/acocac/data/hirham5/interim"
    in_path_aux = os.path.sep.join([interim_path, "AuxFiles"])
    in_path_h5 = os.path.sep.join([interim_path, "firnpack"])

    # %% --------- Create file with original coordinates for later reconstruction ---------
    meta = xr.Dataset()
    src = xr.open_dataset(os.path.sep.join([in_path_aux, "GRLmask.nc"]))
    # include x and y dims/coords (common names)
    if "x" in src.coords:
        meta = meta.assign_coords(x=("x", src["x"].values))
    elif "x" in src.variables:
        # if x is just a variable (not a coord), include it as a variable
        meta["x"] = ("x", src["x"].values)

    if "y" in src.coords:
        meta = meta.assign_coords(y=("y", src["y"].values))
    elif "y" in src.variables:
        meta["y"] = ("y", src["y"].values)

    # lat/lon may be 1D or 2D; include them exactly as in source
    if "lat" in src:
        # ensure values are concrete (compute if dask)
        lat_vals = src["lat"].values
        meta["lat"] = (src["lat"].dims, lat_vals)

    if "lon" in src:
        lon_vals = src["lon"].values
        meta["lon"] = (src["lon"].dims, lon_vals)

    print(meta)
    meta.attrs = src.attrs
    filepath = os.path.sep.join([out_path, "coords_only.zarr"])
    logger.info("GRL mask saved to %s", filepath)
    meta.to_zarr(filepath, mode="w", consolidated=True)

    # %% -------------------- Create stacked nc files for AuxFiles --------------------
    # -------------------- Create file with GRL mask --------------------
    filepath = os.path.sep.join([out_path, "GRLmask.zarr"])
    ds_basins = xr.open_dataset(os.path.sep.join([in_path_aux, "GRLmask.nc"]))
    ds_mask = ds_basins[["glacGRL", "maskbas"]]
    ds_mask = ds_mask.stack(z=("x", "y")).reset_index("z")  # flatten spatial dimension
    ds_mask = ds_mask.dropna(dim="z", how="all")  # drop all NaN values
    ds_mask.to_zarr(filepath, mode="w", consolidated=True)
    logger.info("GRL mask saved to %s", filepath)
    ds_mask.close()

    # -------------------- Create file with zones --------------------
    filepath = os.path.sep.join(
        [out_path, "GRLzones.zarr"]
    )  # output file from create_spatial_subsampling.py
    ds_zones = xr.open_dataset(os.path.sep.join([in_path_aux, "GRLzones.nc"]))
    da_zones = ds_zones["zones"]
    da_zones = da_zones.stack(z=("x", "y")).reset_index(
        "z"
    )  # flatten spatial dimension
    da_zones = da_zones.dropna(dim="z", how="all")  # drop all NaN values
    da_zones.to_zarr(filepath, mode="w", consolidated=True)
    logger.info("GRL mask saved to %s", filepath)
    ds_zones.close()

    # %% Correct daily files: clip to valid ranges, correct runoff values using available water estimates
    logger.info("Load all daily files ...")
    years = range(1980, 2017)
    dropvars = ["tsl", "sn", "evspsbl", "gld"]
    chunks_init = {"time": 1024, "y": 64, "x": 64}
    ds_all = xr.open_mfdataset(
        [f"{in_path_h5}/Daily2D_GRL_{year}.nc" for year in years],
        concat_dim="time",
        combine="nested",
        chunks=chunks_init,
    ).drop_vars(dropvars)
    ds_all = ds_all.sortby("time")

    logger.info("Correct data ranges ...")
    vars_ranges = {
        "ahfs": (-140, None),
    }

    for var, (vmin, vmax) in vars_ranges.items():
        if var in ds_all.data_vars:
            data = ds_all[var]
            logger.info(" check if %s in [%s, %s]...", var, vmin, vmax)
            clipped_at_min = (
                (data < vmin) if vmin is not None else xr.zeros_like(data, dtype=bool)
            )
            clipped_at_max = (
                (data > vmax) if vmax is not None else xr.zeros_like(data, dtype=bool)
            )
            n_clipped_min = clipped_at_min.sum().compute().item()
            n_clipped_max = clipped_at_max.sum().compute().item()

            if n_clipped_min > 0:
                logger.info("   # samples clipped at min: %d", n_clipped_min)
            if n_clipped_max > 0:
                logger.info("   # samples clipped at max: %d", n_clipped_max)
            ds_all[var] = data.clip(min=vmin, max=vmax)

        else:
            logger.info("%s: not found in dataset", var)

    # %% -------------------- Restructure dataset, and save daily data to zarr files --------------------
    filepath = os.path.sep.join([out_path, "base_dataset.zarr"])
    chunksize = {"time": 2048, "z": -1}
    logger.info("Re-structure dataset...")
    # ds = ds.to_dataarray(dim='variable_names') # stack all variables into an extra dimension
    # ds.name = "data"
    logger.info("Flatten spatial dimension...")
    ds = ds_all.stack(z=("x", "y")).reset_index("z")  # flatten spatial dimension
    logger.info("Drop nans...")
    # ds = ds.dropna(dim='z', how='all')  # drop all NaN values
    nan_mask = (
        ~ds["tas"].isnull().all(dim=["time"]).compute()
    )  # compute null only from tas, because either all vars or none are nan for a specific location!
    logger.info("Shape of nan-maks: %s", nan_mask.shape)
    logger.info("amount of all z: %d", len(ds.z))
    ds = ds.isel(z=nan_mask)
    logger.info("amount of non-nan z: %d", len(ds.z))

    logger.info("Rechunk time dimension...")
    ds = ds.chunk(chunksize)

    logger.info("Saving data to %s...", filepath)

    with ProgressBar():
        ds.to_zarr(filepath, mode="w", consolidated=True)

    # %% -------------------- Create 10yr rolling averages --------------------
    outpath_roll = filepath.replace(".zarr", "_10yr.zarr")
    vars_to_roll = ["tas", "rainfall", "snfall"]

    ds_sel = ds[vars_to_roll]

    logger.info("Create 10yrs rolling mean for %s...", vars_to_roll)
    window_size = 3652
    rolling_10yr = ds_sel.rolling(time=window_size, min_periods=window_size)

    logger.info("  drop nans ...")
    means_10yr = rolling_10yr.mean()
    means_10yr = means_10yr.sel(time=slice("1990-01-01", None))

    means_10yr = means_10yr.rename(
        {var: f"{var}_10yr-avg" for var in means_10yr.data_vars}
    )

    print(f"Saving 10years rolling mean results to {outpath_roll} ...")
    means_10yr = means_10yr.chunk(chunksize)
    with ProgressBar():
        means_10yr.to_zarr(outpath_roll, mode="w", consolidated=True)
