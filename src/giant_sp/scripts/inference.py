"""Main script for inference using a trained ConvNP model."""

import glob
import json
import logging
import os
from typing import cast

import deepsensor.torch  # noqa: F401
import geopandas as gpd
import hydra
import numpy as np
import pandas as pd
import torch
import wandb
import xarray as xr
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from deepsensor.model.pred import Prediction
from omegaconf import DictConfig

import giant_sp.utils.data as du
import giant_sp.utils.plots as myplt
from giant_sp.data import CARRA2DataLoader, HIRHAM5DataLoader, HIRHAM5ZarrDataLoader
from giant_sp.models.mdn import MDNZIGammaConvNP, install_mdn_zigamma_likelihood
from giant_sp.utils.data import context_sampling_with_delta
from giant_sp.utils.model import get_available_device

logger = logging.getLogger(__name__)


def download_wandb_file(run_ref: str, files_path: str, file: str) -> None:
    """Download a single file logged to a Weights & Biases run.

    Parameters
    ----------
    run_ref : str
        Fully qualified W&B run reference, e.g. ``"entity/project/run_id"``.
    files_path : str
        Local directory to download the file into.
    file : str
        Name of the file to download, relative to the run's ``files/``
        directory (e.g. ``"model_config.json"``).

    """
    api = wandb.Api()
    therun = api.run(run_ref)

    therun.file(os.path.join("files", file)).download(root=files_path, exist_ok=True)


def load_wandb_model(
    data_processor: DataProcessor,
    task_loader: TaskLoader,
    config_path: str,
    weights_path: str,
) -> ConvNP:
    """Rebuild a ConvNP model from a saved config and load its trained weights.

    Parameters
    ----------
    data_processor : DataProcessor
        Fitted data processor used by the model to (de)normalise inputs
        and outputs.
    task_loader : TaskLoader
        Task loader describing the context/target variables the model
        was trained on.
    config_path : str
        Path to the model's JSON config file (e.g. ``model_config.json``
        as saved by W&B), containing ``unet_channels``, ``unet_kernels``
        and ``likelihood``.
    weights_path : str
        Path to the PyTorch checkpoint (``.pt`` file) with the trained
        model weights.

    Returns
    -------
    ConvNP
        A ConvNP model built from `config_path` with the weights from
        `weights_path` loaded, mapped onto CUDA if available and
        otherwise the CPU.

    """
    with open(config_path) as f:
        config = json.load(f)

    # Build model from config
    model_args = {
        "unet_channels": config["unet_channels"],
        "unet_kernels": config["unet_kernels"],
        "likelihood": config["likelihood"],
    }

    if config["likelihood"] == "mdn-zigamma":
        model_args["mixture_components"] = config["mixture_components"]
        install_mdn_zigamma_likelihood(config["mixture_components"])
        model = MDNZIGammaConvNP(data_processor, task_loader, **model_args)
    else:
        model = ConvNP(data_processor, task_loader, **model_args)

    # Load model weights
    model.model.load_state_dict(
        torch.load(weights_path, map_location=get_available_device())
    )

    return model


def export_zarr(
    target_var: str,
    pred_ds: Prediction,
    truth_ds: xr.Dataset,
    files_path: str,
    fname: str,
    source: str,
    chunk_size: int,
    data_path: str | None = None,
) -> None:
    """Align predictions with ground truth and write the result to a Zarr store.

    Parameters
    ----------
    target_var : str
        Name of the target variable to export.
    pred_ds : Prediction
        Model predictions keyed by target variable, as returned by
        ``ConvNP.predict``. Must contain ``"mean"`` and ``"std"`` entries
        for `target_var`.
    truth_ds : xr.Dataset
        Ground-truth dataset containing `target_var`, used to align
        predictions onto observed grid points and to store the true
        values alongside the predictions.
    files_path : str
        Directory the resulting ``<fname>.zarr`` store is written into.
    fname : str
        Base filename (without extension) for the output Zarr store.
    source : str
        Data source identifier (e.g. ``"hirham5zarr"``, ``"hirham5"``,
        ``"carra2"``). When ``"hirham5zarr"``, the output is reindexed
        onto the reference x/y grid read from ``coords_only.zarr`` under
        `data_path`; otherwise it keeps its native lat/lon grid.
    chunk_size : int
        Chunk size along the ``time`` dimension of the output store.
        Spatial dimensions are stored as single chunks.
    data_path : str | None, optional
        Directory containing ``coords_only.zarr``. Required when `source`
        is ``"hirham5zarr"``. Default is None.

    """
    pred_ds, truth_da = xr.align(
        pred_ds[target_var], truth_ds[target_var], join="inner"
    )

    pred_final = xr.concat(
        [truth_da, pred_ds["mean"], pred_ds["std"]], dim="variable"
    ).assign_coords(
        variable=[f"{target_var}_true", f"{target_var}_pred", f"{target_var}_std"]
    )
    pred_final = pred_final.to_dataset(dim="variable")

    if source == "hirham5zarr":
        # `data_path` is required (and always a str) for this source; the
        # `str | None` annotation only reflects the function's default value.
        coords_file = os.path.sep.join([cast(str, data_path), "coords_only.zarr"])
        coords_ref = xr.open_zarr(coords_file, chunks="auto")
        xy_coords_ref = (coords_ref["x"], coords_ref["y"])
        lonlat_coords_ref = (coords_ref["lon"], coords_ref["lat"])

        rename_map = {"lon": "x", "lat": "y"}
        pred_final = pred_final.rename(rename_map)

        ds_xr = pred_final.reindex({"x": xy_coords_ref[0], "y": xy_coords_ref[1]})
        ds_xr = ds_xr.assign_coords(lon=lonlat_coords_ref[0], lat=lonlat_coords_ref[1])
        chunk_dims = {"time": chunk_size, "x": -1, "y": -1}
    else:
        ds_xr = pred_final
        chunk_dims = {"time": chunk_size, "lat": -1, "lon": -1}

    ds_xr = ds_xr.chunk(chunk_dims)

    ds_xr.to_zarr(
        store=os.path.join(files_path, f"{fname}.zarr"),
        mode="w",
        compute=True,
        consolidated=True,
        encoding={
            var: {"chunks": ds_xr[var].data.chunksize, "compressor": None}
            for var in ds_xr.data_vars
        },
    )


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(config: DictConfig) -> None:
    """Run inference with a trained ConvNP model and export predictions.

    Loads the run's configuration and trained model (downloading checkpoint
    files from Weights & Biases if they are not already cached locally),
    builds validation tasks for the requested sampling strategy, generates
    predictions on the full grid, writes them to a Zarr store, and
    optionally saves diagnostic plots.

    Parameters
    ----------
    config : DictConfig
        Hydra-composed run configuration (see ``src/giant_sp/configs/``).
        ``config.inference`` holds the run-specific parameters:

        val_set : {"station", "random-land", "no-obs"}
            Task-generation strategy: real station locations, randomly
            sampled land locations, or no context observations at all.
        n_obs : int
            Number of context observations to sample per task when
            `val_set` is ``"random-land"``.
        run_ref : str
            W&B run reference (``"entity/project/run_id"``) the model and
            data processor were logged under.
        weights_filename : str
            Filename of the model weights checkpoint within the run's files.
        date_split : str
            Config key prefix (e.g. ``"test"``) used to look up
            ``{date_split}_start``/``{date_split}_end`` date-range bounds.
        every : int
            Stride used to subsample the date range (keep every `every`-th
            date).
        seed : int
            Random seed used for reproducible sampling of random-land
            locations and, when plotting, of PIT histogram dates.
        plot : bool
            If set, additionally saves an averaged-prediction plot and a PIT
            histogram plot.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the configured data `source` is not one of ``"carra2"``,
        ``"hirham5"`` or ``"hirham5zarr"``; if no station locations
        remain after filtering to the model domain/land mask; or if
        `val_set` is not one of ``"station"``, ``"random-land"`` or
        ``"no-obs"``.

    """
    inference_config = config.inference
    run_ref = inference_config.run_ref
    weights_fn = inference_config.weights_filename
    n_obs = inference_config.n_obs
    val_set = inference_config.val_set
    date_split = inference_config.date_split
    every = inference_config.every
    seed = inference_config.seed
    plot = inference_config.plot

    # Set up output folders
    run_id = run_ref.split("/")[-1]
    output_path = f"outputs/{run_id}"
    savefolder = os.path.join(output_path, "test_results")
    figfolder = os.path.join(savefolder, "figures")
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    if not os.path.exists(figfolder):
        os.makedirs(figfolder)

    source = config["data"]["source"]
    chunk_size = 100
    target_var = config["data"]["target_var_IDs"]
    context_delta_t = config["dataloader"]["context_delta_t"]

    # Initialize data loader
    data_loader: CARRA2DataLoader | HIRHAM5DataLoader | HIRHAM5ZarrDataLoader
    if source == "carra2":
        data_loader = CARRA2DataLoader(config)
        td = np.timedelta64(0, "ms")
    elif source == "hirham5":
        data_loader = HIRHAM5DataLoader(config)
        td = np.timedelta64(12, "h")  # offset 12hours
    elif source == "hirham5zarr":
        data_loader = HIRHAM5ZarrDataLoader(config)
        td = np.timedelta64(12, "h")
    else:
        raise ValueError(f"Unknown source: {source}")

    dates_start = config["data"][f"{date_split}_start"]
    dates_end = config["data"][f"{date_split}_end"]
    frequency = config["dataloader"]["frequency"]

    dates = pd.date_range(start=dates_start, end=dates_end, freq=frequency)[::every]
    dates = list(np.sort(dates + td).astype("datetime64[ns]"))

    # Locate data
    target_files = ["model_config.json", "data_processor_config.json", weights_fn]
    try:
        files_path = sorted(glob.glob(f"wandb/run*{run_id}/files"), reverse=True)[0]
        if not all(os.path.exists(os.path.join(files_path, f)) for f in target_files):
            raise IndexError
    except IndexError:
        root_path = os.path.join("wandb", f"run-{run_id}")
        os.makedirs(root_path, exist_ok=True)
        for f in target_files:
            download_wandb_file(run_ref, root_path, f)
        files_path = os.path.join(root_path, "files")
    logger.info("Loading run data from: %s", files_path)

    data_processor = DataProcessor(files_path)
    raw_ds, data_processor, task_loader = data_loader.get_dataproc_taskload(
        dates=dates, data_processor=data_processor
    )
    lsm = data_loader.load_aux_ds(resolution="bedmachine")

    model_config_file = os.path.join(files_path, "model_config.json")
    weights_file = os.path.join(files_path, weights_fn)

    model = load_wandb_model(
        data_processor, task_loader, model_config_file, weights_file
    )
    likelihood = model.config["likelihood"]

    # Bernoulli-Gamma/mdn-zigamma predictions are zero-inflated (non-continuous): PIT
    # needs the zero-mass probability and Gamma (mixture) parameters, not just mean/std.
    # Must be a list: deepsensor's Model.predict mutates pred_params in place
    # (list.remove/.append) when "mixture_probs" is requested.
    pred_params = ["mean", "std"]
    if plot and likelihood == "bernoulli-gamma":
        pred_params += ["mixture_probs", "k", "scale"]
    elif plot and likelihood == "mdn-zigamma":
        n_slab_components = model.config["mixture_components"]
        pred_params += ["mixture_probs"]
        for component in range(n_slab_components):
            pred_params += [
                f"slab_probs_{component}",
                f"slab_k_{component}",
                f"slab_scale_{component}",
            ]

    # Determine delta_factor_context for context_sampling_with_delta
    if context_delta_t is None:
        delta_factor_context = [1] * len(config["data"]["context_var_IDs"])
    else:
        delta_factor_context = [len(group) for group in context_delta_t]

    weights_fn = weights_fn.split(".")[0]
    if val_set == "no-obs":
        val_tasks = task_loader(
            dates,
            context_sampling=context_sampling_with_delta(0, delta_factor_context),
            target_sampling="all",
        )

        # Predict
        logger.info("Starting prediction of %d tasks", len(val_tasks))
        val_pred = model.predict(
            val_tasks, X_t=raw_ds, X_t_mask=lsm, pred_params=pred_params
        )
        logger.info("Done")

        fname = f"{target_var}_{date_split}_{val_set}_e{every}_{weights_fn}_predictions"
        export_zarr(
            target_var,
            val_pred,
            raw_ds,
            savefolder,
            fname,
            source,
            chunk_size,
            config["data"]["data_path"],
        )
        logger.info("Saved")

    elif val_set == "random-land":
        logger.info("Starting task generation (%d)", len(dates))
        val_tasks = [
            task_loader(
                date,
                context_sampling=context_sampling_with_delta(
                    data_processor.map_coord_array(
                        du.sample_land_locations(
                            n_obs=n_obs,
                            lsm=lsm,
                            seed=(seed, i),
                        )
                    ),
                    delta_factor_context,
                ),
                target_sampling="all",
                datewise_deterministic=True,
            )
            for i, date in enumerate(dates)
        ]
        logger.info("Finished task generation")

        # Predict
        logger.info("Starting prediction of %d tasks", len(val_tasks))
        val_pred = model.predict(
            val_tasks, X_t=raw_ds, X_t_mask=lsm, pred_params=pred_params
        )
        logger.info("Done")

        fname = f"{target_var}_{date_split}_{val_set}_seed{seed}_e{every}_{weights_fn}_predictions"
        export_zarr(
            target_var,
            val_pred,
            raw_ds,
            savefolder,
            fname,
            source,
            chunk_size,
            config["data"]["data_path"],
        )
        logger.info("Saved")

    elif val_set == "station":
        # All station locations ever
        stn_file = config["data"]["station_locations_file"]
        stn_locs_df = pd.read_csv(stn_file)[
            ["latitude_last_valid", "longitude_last_valid"]
        ]  # get lat/lon only

        if source == "hirham5zarr":
            coords_file = os.path.sep.join(
                [config["data"]["data_path"], "coords_only.zarr"]
            )
            coords_ref = xr.open_zarr(coords_file, chunks="auto")
            xy_coords_ref = (coords_ref["x"], coords_ref["y"])
            lonlat_coords_ref = (coords_ref["lon"], coords_ref["lat"])

            rename_map = {"lon": "x", "lat": "y"}

            # subset based on lsm mask
            mask_reference = lsm.rename(rename_map)
            mask_reference = mask_reference.reindex(
                {"x": xy_coords_ref[0], "y": xy_coords_ref[1]}
            )
            mask_reference = mask_reference.assign_coords(
                lon=lonlat_coords_ref[0], lat=lonlat_coords_ref[1]
            )

            gdf_points = gpd.GeoDataFrame(
                stn_locs_df,
                crs="epsg:4326",
                geometry=gpd.points_from_xy(
                    stn_locs_df["longitude_last_valid"],
                    stn_locs_df["latitude_last_valid"],
                ),
            )

            # Drop stations outside the model domain before snapping to nearest grid cell.
            lon_min = float(lonlat_coords_ref[0].min().values)
            lon_max = float(lonlat_coords_ref[0].max().values)
            lat_min = float(lonlat_coords_ref[1].min().values)
            lat_max = float(lonlat_coords_ref[1].max().values)
            gdf_points = gdf_points.cx[lon_min:lon_max, lat_min:lat_max]

            # mask_reference is a DataArray; find_xy_for_points only uses
            # coords/dims present on both Dataset and DataArray, but its
            # signature currently only declares xr.Dataset.
            gdf_points = du.find_xy_for_points(mask_reference, gdf_points)

            gdf_points["mask_val"] = mask_reference.sel(
                x=xr.DataArray(gdf_points["x"].values, dims="points"),
                y=xr.DataArray(gdf_points["y"].values, dims="points"),
                method="nearest",
            ).values

            stn_locs_df_subset = gdf_points[gdf_points["mask_val"] == 1]
            if stn_locs_df_subset.empty:
                raise ValueError(
                    "No station locations remain after applying the land mask/domain filter."
                )

            raw_stn_locs = pd.DataFrame(
                {"lat": stn_locs_df_subset.y, "lon": stn_locs_df_subset.x}
            )
        else:
            # Non-zarr sources (hirham5, carra2): raw_ds has projected x/y coords renamed to lon/lat
            gdf_points = gpd.GeoDataFrame(
                stn_locs_df,
                crs="epsg:4326",
                geometry=gpd.points_from_xy(
                    stn_locs_df["longitude_last_valid"],
                    stn_locs_df["latitude_last_valid"],
                ),
            ).to_crs("epsg:3413")
            lon_min = float(raw_ds.lon.min().values)
            lon_max = float(raw_ds.lon.max().values)
            lat_min = float(raw_ds.lat.min().values)
            lat_max = float(raw_ds.lat.max().values)
            stn_locs_df_subset = gdf_points.cx[lon_min:lon_max, lat_min:lat_max]
            if stn_locs_df_subset.empty:
                raise ValueError(
                    "No station locations remain after filtering to the model domain."
                )
            raw_stn_locs = pd.DataFrame(
                {
                    "lat": stn_locs_df_subset.geometry.y,
                    "lon": stn_locs_df_subset.geometry.x,
                }
            )

        stn_locs = data_processor.map_coord_array(raw_stn_locs.to_numpy().transpose())

        # Create set of stnval tasks
        logger.info("Starting task generation (%d)", len(dates))
        val_tasks = task_loader(
            dates,
            context_sampling=context_sampling_with_delta(
                stn_locs, delta_factor_context
            ),
            target_sampling="all",
        )
        logger.info("Finished task generation")

        # Predict
        logger.info("Starting prediction of %d tasks", len(val_tasks))
        val_pred = model.predict(
            val_tasks, X_t=raw_ds, X_t_mask=lsm, pred_params=pred_params
        )
        logger.info("Done")

        fname = f"{target_var}_{date_split}_{val_set}_e{every}_{weights_fn}_predictions"
        export_zarr(
            target_var,
            val_pred,
            raw_ds,
            savefolder,
            fname,
            source,
            chunk_size,
            config["data"]["data_path"],
        )
        logger.info("Saved")

    else:
        raise ValueError(
            "Unknown val_set, should be one of [station, random-land, no-obs]"
        )

    # Averaged prediction plots
    if plot:
        date = pd.Timestamp(dates[0])
        task = next(t for t in val_tasks if t["time"] == date)
        _mean_da, raw_da = xr.align(
            val_pred[target_var]["mean"], raw_ds[target_var], join="inner"
        )
        fig, _ax = myplt.averaged_prediction_plot(
            raw_da,
            val_pred[target_var]["mean"],
            val_pred[target_var]["std"],
            cmap="viridis",
            task=task,
            dataproc=data_processor,
            highlight_negatives=True,
            plot_xc=True,
            crs=None,
        )
        fig.savefig(os.path.join(figfolder, f"{fname}_avgpred.png"))

        # PIT
        # Randomly select dates
        test_dates = [val_tasks[i]["time"] for i in range(len(val_tasks))]
        n_dates = min(len(test_dates), 30)
        np.random.seed(42)
        dates_plt = np.random.choice(test_dates, n_dates, replace=False)

        # Subselect and flatten datasets
        true_plt = raw_ds[target_var].sel(time=dates_plt).values.flatten()
        val_pred_plt = val_pred[target_var].sel(time=dates_plt)

        fig, _ = myplt.PIT_histogram(
            true_plt,
            val_pred_plt,
            likelihood=likelihood,
            n_slab_components=(
                model.config["mixture_components"]
                if likelihood == "mdn-zigamma"
                else None
            ),
        )

        fig.savefig(os.path.join(figfolder, f"{fname}_pit_histogram.png"))


if __name__ == "__main__":
    main()
