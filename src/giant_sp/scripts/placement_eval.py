"""Main script for evaluating placement experiments using a trained ConvNP model."""

import logging
import os

# src
import deepsensor.torch as deepsensor  # noqa: F401
import geopandas as gpd
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr
from deepsensor.data.processor import DataProcessor
from deepsensor.train.train import set_gpu_default_device
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import giant_sp.utils.data as du
from giant_sp.data import CARRA2DataLoader, HIRHAM5DataLoader, HIRHAM5ZarrDataLoader
from giant_sp.data.dataloader import BaseDataLoader
from giant_sp.utils.data import context_sampling_with_delta, sample_dates_by_year_chunks
from giant_sp.utils.wandb_utils import (
    DATA_PROCESSOR_CONFIG_FILE,
    MODEL_CONFIG_FILE,
    fetch_run_files,
    load_wandb_model,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="placement_eval")
def main(config: DictConfig) -> None:
    """Evaluate how adding proposed station placements changes model skill.

    Loads a trained ConvNP model together with its data processor and task
    loader from a Weights & Biases run, samples evaluation dates, and builds
    the existing station network (snapped to the model grid and filtered to
    the domain/land mask). The proposed placements in ``newplace_file`` are
    then added one at a time, in iteration order, to the existing stations
    as context: for each step ``i = 0, ..., n_iter`` (``i = 0`` being the
    existing network alone), the model predicts over the full grid for every
    sampled date and the following are recorded:

    - ``rmse``: per-grid-cell RMSE against the gridded truth over time,
      averaged over space;
    - ``mae``: mean absolute error over all dates and grid cells;
    - ``avgstd``: mean predictive standard deviation;
    - ``nll``: normalised negative log-likelihood (``model.loss_fn``),
      averaged over tasks (one per date), as in `Trainer`'s validation loss.

    Parameters
    ----------
    config : DictConfig
        Hydra-composed run configuration (see ``src/giant_sp/configs/``).
        ``config.placement_eval`` holds the run-specific parameters:

        run_ref : str
            Weights & Biases run reference (``"entity/project/run_id"``)
            to load the trained model from.
        newplace_file : str
            Path to a CSV of proposed placements (as written by
            ``placement.py``), indexed by an ``iteration`` column. The
            weights file to load is derived from the last ``_``-separated
            token of its file name, with ``.csv`` replaced by ``.pt``.
        search_start_date, search_end_date : str
            Date range (inclusive) over which evaluation dates are sampled.
        temporal_subsampling_method : str
            Either ``"seasonal"`` (Gaussian-weighted by day of year) or
            ``"regular"`` (fixed-stride date range).
        temporal_subsampling_every : int
            Number of dates sampled per year (seasonal) or stride between
            dates (regular).
        output_dir : str
            Root directory for outputs; results go under
            ``<output_dir>/<run_id>/``.

    Returns
    -------
    None
        Outputs are written under
        ``<output_dir>/<run_id>/placement/metrics/<method>_e<every>/``, each named
        after ``newplace_file`` with a ``_metrics`` suffix: a NetCDF of the
        predicted mean/std per iteration, a CSV of the metric curves above,
        and a PNG plotting them.

    Raises
    ------
    ValueError
        If ``config["data"]["source"]`` is not one of ``"carra2"``,
        ``"hirham5"``, ``"hirham5zarr"``; if the temporal sampling method is
        not ``"seasonal"`` or ``"regular"``; or if no station locations
        remain after filtering to the model domain/land mask.
    """
    placement_config = config.placement_eval
    run_ref = placement_config.run_ref
    newplace_file = placement_config.newplace_file
    dates_start = placement_config.search_start_date
    dates_end = placement_config.search_end_date
    output_dir = placement_config.output_dir

    weights_fn = newplace_file.split("/")[-1].split("_")[-1].replace(".csv", ".pt")
    run_id = run_ref.split("/")[-1]

    # Path and files
    output_path = os.path.join(output_dir, run_id)

    # coords_path = config["data"]["data_path"]
    source = config["data"]["source"]
    context_delta_t = config["dataloader"]["context_delta_t"]

    logger.info(
        "Active devices before updating default: %s", torch.get_default_device()
    )
    if torch.cuda.is_available():
        set_gpu_default_device()

    logger.info("Active devices after updating default: %s", torch.get_default_device())
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    # Initialize data loader
    data_loader: BaseDataLoader
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

    target_var = config["data"]["target_var_IDs"]
    frequency = config["dataloader"]["frequency"]
    temporal_sampling_method = placement_config.temporal_subsampling_method
    temporal_sampling_every = placement_config.temporal_subsampling_every

    if temporal_sampling_method == "seasonal":
        # source: https://github.com/eschlager/MeltEmulation/tree/main
        # temporal sub-sampling of training data
        doy = np.arange(1, 367)  # days 1 to 366
        mu = 205  # 24th of July ~ peak of melt season
        sigma = 60
        # min_weight_doy = 0.0
        gauss_weights = np.exp(-0.5 * ((doy - mu) / sigma) ** 2)
        # weights = min_weight_doy + (1 - min_weight_doy) * gauss_weights
        weights_doy = gauss_weights / gauss_weights.sum()

        n_samples_per_year = temporal_sampling_every

        np.random.seed(42)
        dates = sample_dates_by_year_chunks(
            weights_doy, dates_start, dates_end, n_samples_per_year
        )

        # numpy resolves list[np.datetime64] + np.timedelta64 via timedelta64's
        # __radd__ at runtime (broadcasting the list to an array), which
        # mypy's operator overload resolution does not model.
        dates = list(np.sort(dates + td).astype("datetime64[ns]"))  # type: ignore[operator]
    elif temporal_sampling_method == "regular":
        dates = pd.date_range(start=dates_start, end=dates_end, freq=frequency)[
            ::temporal_sampling_every
        ]
        dates = list(np.sort(dates + td).astype("datetime64[ns]"))
    else:
        raise ValueError(
            f"Unknown temporal sampling method: {temporal_sampling_method}"
        )

    # Locate data
    files_path = fetch_run_files(
        run_ref, [MODEL_CONFIG_FILE, DATA_PROCESSOR_CONFIG_FILE, weights_fn]
    )

    data_processor = DataProcessor(files_path)
    raw_ds, data_processor, task_loader = data_loader.get_dataproc_taskload(
        dates=dates, data_processor=data_processor
    )
    lsm = data_loader.load_aux_ds(resolution="bedmachine")

    config_file = os.path.join(files_path, MODEL_CONFIG_FILE)
    weights_file = os.path.join(files_path, weights_fn)

    model = load_wandb_model(data_processor, task_loader, config_file, weights_file)

    stn_file = config["data"]["station_locations_file"]
    stn_locs_df = pd.read_csv(stn_file)[
        ["latitude_last_valid", "longitude_last_valid"]
    ]  # get lat/lon only
    if source == "hirham5zarr":
        coords_file = os.path.sep.join(
            [config["data"]["data_path"], "coords_only.zarr"]
        )
        coords_ref = xr.open_zarr(coords_file, chunks="auto")
        # coords_only.zarr stores x/y as unlabelled dims, so xarray falls back to a
        # synthetic 0-based index on access. Materialize it as a real coordinate so
        # later .sel(..., method='nearest') lookups against this grid work.
        coords_ref = coords_ref.assign_coords(
            x=np.arange(coords_ref.sizes["x"]),
            y=np.arange(coords_ref.sizes["y"]),
        )
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

        # find_xy_for_points is typed for xr.Dataset, but only ever accesses
        # attributes (coords, dims) that xr.DataArray shares with Dataset, so
        # passing the DataArray mask_reference here works at runtime.
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
            {"lat": stn_locs_df_subset.geometry.y, "lon": stn_locs_df_subset.geometry.x}
        )

    X_stn = data_processor.map_coord_array(raw_stn_locs.to_numpy().transpose())

    # Get new placements df
    newplace_df = pd.read_csv(newplace_file, index_col="iteration")
    raw_newplace = newplace_df.to_numpy().transpose()
    X_new = data_processor.map_coord_array(raw_newplace)
    n_iter = newplace_df.shape[0]

    # Determine delta_factor_context for context_sampling_with_delta
    if context_delta_t is None:
        delta_factor_context = [1] * len(config["data"]["context_var_IDs"])
    else:
        delta_factor_context = [len(group) for group in context_delta_t]

    truth_da = raw_ds[target_var].sel(time=dates)

    pred_means = []
    pred_stds = []
    nlls = []
    for iter in tqdm(range(n_iter + 1), desc="Predict"):
        X_added = np.concatenate((X_stn, X_new[:, :iter]), axis=1)
        tasks = task_loader(
            dates,
            context_sampling=context_sampling_with_delta(X_added, delta_factor_context),
            target_sampling="all",
        )

        pred = model.predict(tasks, X_t=raw_ds)
        mean_da = pred[target_var]["mean"]
        std_da = pred[target_var]["std"]

        pred_means.append(mean_da)
        pred_stds.append(std_da)

        # Marginal NLL, averaged over tasks (one per date), matching Trainer's
        # val-loss computation (model.loss_fn(task, normalise=True) per task).
        task_nlls = [
            model.loss_fn(task.remove_target_nans(), normalise=True)
            .detach()
            .cpu()
            .numpy()
            for task in tasks
        ]
        nlls.append(float(np.mean(task_nlls)))

    # Create output directory for metrics
    metric_dir = os.path.join(
        output_path,
        "placement",
        "metrics",
        f"{temporal_sampling_method}_e{temporal_sampling_every}",
    )
    if not os.path.exists(metric_dir):
        os.makedirs(metric_dir)

    # Save predictions
    pred_mean_all = xr.concat(pred_means, dim="iteration")
    pred_std_all = xr.concat(pred_stds, dim="iteration")
    pred_all_out = xr.Dataset({"mean": pred_mean_all, "std": pred_std_all})
    pred_all_out.to_netcdf(
        os.path.join(
            metric_dir, newplace_file.split("/")[-1].replace(".csv", "_metrics.nc")
        )
    )

    # Metrics
    rmses = []
    avgstds = []
    maes = []
    for iter in tqdm(range(n_iter + 1), desc="Compute metrics"):
        mean_da = pred_all_out["mean"].sel(iteration=iter)
        std_da = pred_all_out["std"].sel(iteration=iter)
        err_da = mean_da - truth_da

        rmse = np.sqrt(np.power(err_da, 2).mean(dim="time")).mean().values.item()
        avgstd = std_da.where(truth_da.notnull()).mean().values.item()
        mae = np.abs(err_da).mean().values.item()

        rmses.append(rmse)
        avgstds.append(avgstd)
        maes.append(mae)

    curves_arr = np.array([rmses, maes, avgstds, nlls]).T
    out_df = pd.DataFrame(curves_arr, columns=["rmse", "mae", "avgstd", "nll"])
    out_df.to_csv(
        os.path.join(
            metric_dir, newplace_file.split("/")[-1].replace(".csv", "_metrics.csv")
        )
    )

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].plot(rmses, label="RMSE")
    axs[1].plot(maes, label="MAE")
    axs[2].plot(avgstds, label="Avg. std")
    axs[3].plot(nlls, label="NLL")
    for ax in axs:
        ax.legend()
    fig.savefig(
        os.path.join(
            metric_dir, newplace_file.split("/")[-1].replace(".csv", "_metrics.png")
        ),
        dpi=300,
    )


if __name__ == "__main__":
    main()
