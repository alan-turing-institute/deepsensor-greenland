"""Main script for placement experiments using a trained ConvNP model."""

import glob
import json
import logging
import os

# src
import deepsensor.active_learning.acquisition_fns as acq
import deepsensor.torch as deepsensor  # noqa: F401
import geopandas as gpd
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import wandb
import xarray as xr
from deepsensor.active_learning.algorithms import GreedyAlgorithm
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from deepsensor.train.train import set_gpu_default_device
from omegaconf import DictConfig, OmegaConf

import giant_sp.utils.data as du
import giant_sp.utils.plots as myplt
from giant_sp.data import CARRA2DataLoader, HIRHAM5DataLoader, HIRHAM5ZarrDataLoader
from giant_sp.data.dataloader import BaseDataLoader
from giant_sp.models.mdn import MDNZIGammaConvNP, install_mdn_zigamma_likelihood
from giant_sp.utils.acquisition import (
    ExpectedEpistemicReduction,
    ExpectedMagnitudeVarianceReduction,
    ExpectedOccurrenceEntropyReduction,
    MeanMagnitudeVariance,
    MeanOccurrenceEntropy,
    OracleBrierScore,
    OracleEpistemicReduction,
    OracleMagnitudeMAE,
    OracleMagnitudeVarianceReduction,
    OracleOccurrenceEntropyReduction,
)
from giant_sp.utils.data import context_sampling_with_delta, sample_dates_by_year_chunks
from giant_sp.utils.model import get_available_device

logger = logging.getLogger(__name__)


def download_wandb_file(run_ref: str, files_path: str, file: str) -> None:
    """Download a single file logged to a Weights & Biases run.

    Parameters
    ----------
    run_ref : str
        Weights & Biases run reference, e.g. ``"entity/project/run_id"``.
    files_path : str
        Local directory to download the file into (passed as ``root`` to
        the W&B download call).
    file : str
        Name of the file to download, relative to the run's ``files/``
        directory.

    Returns
    -------
    None
        The file is downloaded to disk as a side effect; nothing is
        returned.

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
    """Reconstruct a trained ConvNP model from a Weights & Biases run.

    Builds a `ConvNP` model using the architecture recorded in the run's
    model config, then loads its trained weights onto the CUDA device if
    one is available, otherwise the CPU.

    Parameters
    ----------
    data_processor : DataProcessor
        Fitted DeepSensor data processor used to construct the model.
    task_loader : TaskLoader
        Task loader describing the model's context/target sets.
    config_path : str
        Path to the run's ``model_config.json`` file.
    weights_path : str
        Path to the ``.pt`` file containing the trained model weights.

    Returns
    -------
    ConvNP
        The reconstructed model with weights loaded from `weights_path`.

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


@hydra.main(version_base=None, config_path="../configs", config_name="placement")
def main(config: DictConfig) -> None:
    """Run a greedy active-learning search for new station placements.

    Loads a trained ConvNP model together with its data processor and task
    loader from a Weights & Biases run, builds the target prediction grid
    and search-space mask, samples evaluation dates, and runs deepsensor's
    `GreedyAlgorithm` with the requested acquisition function to propose
    `n_new` new station locations. Results (the acquisition function values
    and proposed placements) are saved to disk and a summary figure is
    produced, unless a previous run's output already exists on disk, in
    which case that output is loaded and only re-plotted.

    Parameters
    ----------
    config : DictConfig
        Hydra-composed run configuration (see ``src/giant_sp/configs/``).
        ``config.placement`` holds the run-specific parameters:

        run_ref : str
            Weights & Biases run reference (``"entity/project/run_id"``)
            to load the trained model from.
        weights_filename : str
            Filename of the model weights within the run's files.
        acq_fn : str
            Name of the deepsensor acquisition function to use.
        n_new : int
            Number of new station locations to propose.
        search_start_date, search_end_date : str
            Date range (inclusive) over which evaluation dates are sampled.
        temporal_subsampling_method : str
            Either ``"seasonal"`` (Gaussian-weighted by day of year) or
            ``"regular"`` (fixed-stride date range).
        temporal_subsampling_every : int
            Number of dates sampled per year (seasonal) or stride between
            dates (regular).
        pred_resolution : int
            Coarsening factor for the target prediction grid; ``0`` uses
            the native model grid.
        search_mask_file : str
            Optional path to a pre-computed search-space mask; if empty, a
            default mask is derived from the land-sea mask.
        search_resolution : int
            Coarsening factor used to build the default search-space mask.
        min_land : float
            Minimum land fraction for a grid cell to be included in the
            target/search masks.

    Returns
    -------
    None
        Outputs are written under ``outputs/<run_id>/placement/<acq_fn>/``
        (acquisition function NetCDF and new-placements CSV) and
        ``outputs/<run_id>/placement/figures/`` (plot).

    Raises
    ------
    ValueError
        If ``config["data"]["source"]`` is not one of ``"carra2"``,
        ``"hirham5"``, ``"hirham5zarr"``; if the temporal sampling method is
        not ``"seasonal"`` or ``"regular"``; or if no station locations
        remain after filtering to the model domain/land mask.
    NotImplementedError
        If ``acq_fn`` does not match a known deepsensor acquisition
        function name.
    """
    placement_config = config.placement
    run_ref = placement_config.run_ref
    weights_fn = placement_config.weights_filename
    dates_start = placement_config.search_start_date
    dates_end = placement_config.search_end_date
    search_resolution = placement_config.search_resolution
    acq_fn = placement_config.acq_fn
    seed = placement_config.seed
    n_new = placement_config.n_new
    pred_resolution = placement_config.pred_resolution
    search_mask_file = placement_config.search_mask_file
    min_land = placement_config.min_land

    run_id = run_ref.split("/")[-1]

    # Path and files
    output_path = f"outputs/{run_id}"
    savedir = os.path.join(output_path, "placement", acq_fn)
    figdir = os.path.join(output_path, "placement", "figures")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    if not os.path.exists(figdir):
        os.makedirs(figdir)

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

    # chunk_size = 100

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

        np.random.seed(seed)
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

    config_file = os.path.join(files_path, "model_config.json")
    weights_file = os.path.join(files_path, weights_fn)

    model = load_wandb_model(data_processor, task_loader, config_file, weights_file)

    # Load raw/GT data
    ds = data_processor(raw_ds)

    # Define target grid and gridbox weights
    if pred_resolution == 0:
        target_grid = raw_ds
        target_mask = lsm.where(lsm > min_land, other=0)
    else:
        lsm = data_loader.load_aux_ds(resolution="bedmachine")
        coarsen_factor = int(pred_resolution / 1.0)
        # xarray-stubs does not expose reduction methods (e.g. `.mean()`) on
        # DataArrayCoarsen, though they exist at runtime.
        target_grid = lsm.coarsen(
            {"lat": coarsen_factor, "lon": coarsen_factor}, boundary="trim"
        ).mean()  # type: ignore[attr-defined]
        target_mask = target_grid.where(target_grid > min_land, other=0)

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

    # Search space definition
    search_grid_is_default = False
    if search_mask_file == "":
        coarsen_factor = int(search_resolution / 1.0)
        lsm = data_loader.load_aux_ds(resolution="bedmachine")
        # xarray-stubs does not expose reduction methods (e.g. `.mean()`) on
        # DataArrayCoarsen, though they exist at runtime.
        aux_raw_ds = lsm.coarsen(
            {"lat": coarsen_factor, "lon": coarsen_factor}, boundary="trim"
        ).mean()  # type: ignore[attr-defined]
        search_mask = aux_raw_ds.where(
            aux_raw_ds > min_land, other=0
        )  # search where True
        search_grid_is_default = True
    else:
        search_mask = xr.open_dataarray(search_mask_file)

    # Determine delta_factor_context for context_sampling_with_delta
    if context_delta_t is None:
        delta_factor_context = [1] * len(config["data"]["context_var_IDs"])
    else:
        delta_factor_context = [len(group) for group in context_delta_t]

    # Generate tasks
    tasks = task_loader(dates, context_sampling_with_delta(X_stn, delta_factor_context))
    weights_fn = weights_fn.split(".")[0]

    # Acquisition functions
    # Parallel
    seeds = [seed]
    if acq_fn == "Random":
        n_random_seeds = placement_config.N_random_seeds
        random_seeds = np.arange(seed, seed + n_random_seeds)
        acq_fn_type = [acq.Random(seed=s) for s in random_seeds]
        seeds = random_seeds.tolist()
        tasks = tasks[:1]
    elif acq_fn == "ContextDist":
        acq_fn_type = [acq.ContextDist(context_set_idx=0)]
    elif acq_fn == "Stddev":
        acq_fn_type = [acq.Stddev(model)]
    elif acq_fn == "MeanOccurrenceEntropy":
        acq_fn_type = [MeanOccurrenceEntropy(model)]
    elif acq_fn == "MeanMagnitudeVariance":
        acq_fn_type = [MeanMagnitudeVariance(model)]
    # Sequential
    elif acq_fn == "MeanStddev":
        acq_fn_type = [acq.MeanStddev(model)]
    elif acq_fn == "MeanVariance":
        acq_fn_type = [acq.MeanVariance(model)]
    elif acq_fn == "MeanMarginalEntropy":
        acq_fn_type = [acq.MeanMarginalEntropy(model)]
    elif acq_fn == "JointEntropy":
        acq_fn_type = [acq.JointEntropy(model)]
    elif acq_fn == "ExpectedImprovement":
        acq_fn_type = [acq.ExpectedImprovement(model)]
    elif acq_fn == "ExpectedEpistemicReduction":
        acq_fn_type = [ExpectedEpistemicReduction(model)]
    elif acq_fn == "ExpectedOccurrenceEntropyReduction":
        acq_fn_type = [ExpectedOccurrenceEntropyReduction(model)]
    elif acq_fn == "ExpectedMagnitudeVarianceReduction":
        acq_fn_type = [ExpectedMagnitudeVarianceReduction(model)]
    # Oracle
    elif acq_fn == "OracleRMSE":
        acq_fn_type = [acq.OracleRMSE(model)]
    elif acq_fn == "OracleMAE":
        acq_fn_type = [acq.OracleMAE(model)]
    elif acq_fn == "OracleMarginalNLL":
        acq_fn_type = [acq.OracleMarginalNLL(model)]
    elif acq_fn == "OracleJointNLL":
        acq_fn_type = [acq.OracleJointNLL(model)]
    elif acq_fn == "OracleEpistemicReduction":
        acq_fn_type = [OracleEpistemicReduction(model)]
    elif acq_fn == "OracleOccurrenceEntropyReduction":
        acq_fn_type = [OracleOccurrenceEntropyReduction(model)]
    elif acq_fn == "OracleMagnitudeVarianceReduction":
        acq_fn_type = [OracleMagnitudeVarianceReduction(model)]
    elif acq_fn == "OracleBrierScore":
        acq_fn_type = [OracleBrierScore(model)]
    elif acq_fn == "OracleMagnitudeMAE":
        acq_fn_type = [OracleMagnitudeMAE(model)]
    else:
        raise NotImplementedError(f"No deepsensor acquisition function named {acq_fn}")

    if acq_fn[:6] == "Oracle":
        # Args for fns needing ground truth
        greedy_gt_args = {
            "query_infill": ds[target_var],
            "proposed_infill": ds[target_var],
            "task_loader": task_loader,
        }
    else:
        greedy_gt_args = {}

    # The six "*Reduction" acquisition functions (Expected-/Oracle- x Epistemic-/
    # OccurrenceEntropy-/MagnitudeVariance-) score their target quantity directly;
    # GreedyAlgorithm's `diff=True` turns that into `after - before` so the acquisition
    # value reads as a reduction. None of them are AcquisitionFunctionParallel (for
    # which `diff=True` would raise), so this is safe for all six. Other acquisition
    # functions
    # (OracleMAE/OracleRMSE/OracleMarginalNLL/OracleJointNLL/OracleBrierScore/
    # OracleMagnitudeMAE included)
    # use GreedyAlgorithm's default `diff=False`, scoring the resulting task directly.
    greedy_call_args = {"diff": True} if acq_fn.endswith("Reduction") else {}

    # Greedy algorithm
    greedy_alg = GreedyAlgorithm(
        model=model,
        X_t=target_grid,
        X_s=search_mask.astype(bool),
        X_s_mask=search_mask.astype(bool),
        X_t_mask=target_mask.astype(bool),
        N_new_context=n_new,
        progress_bar=True,
        **greedy_gt_args,
    )

    for run_seed, acq_fn_run in zip(seeds, acq_fn_type, strict=True):
        run_suffix = f"{acq_fn}_seed{run_seed}_{n_new}-iter_s{temporal_sampling_method}_e{temporal_sampling_every}_sr{search_resolution}_{weights_fn}"
        acq_fn_path = os.path.join(savedir, f"AcqFn_{run_suffix}.nc")
        new_place_path = os.path.join(savedir, f"NewPlace_{run_suffix}.csv")

        if not os.path.exists(acq_fn_path):
            # New placements
            X_new_df, acq_fn_da = greedy_alg(acq_fn_run, tasks, **greedy_call_args)

            # Save
            acq_fn_da.to_netcdf(acq_fn_path)
            X_new_df.to_csv(new_place_path)

        else:
            logger.info(
                "Acquisition function file already exists. Skipping computation: %s",
                acq_fn_path,
            )
            logger.info("Plotting existing acquisition function file.")
            search_grid_is_default = True
            acq_fn_da = xr.open_dataarray(acq_fn_path)
            X_new_df = pd.read_csv(new_place_path)[["lat", "lon"]]

        if source == "hirham5zarr" and search_grid_is_default:
            # acq_fn_da's lat/lon coords are native-grid-index averages from coarsening
            # lsm (which is cropped to land pixels), so they don't share block
            # boundaries with an independently-coarsened coords_ref and a label-based
            # reindex against it returns all NaN. Look up the nearest matching lon/lat
            # degrees directly from the full-resolution reference grid instead.
            acq_fn_da_plot = acq_fn_da.rename(rename_map)
            lonlat_lookup = coords_ref.sel(
                x=acq_fn_da_plot["x"], y=acq_fn_da_plot["y"], method="nearest"
            )
            # lonlat_lookup's x/y coords are coords_ref's matched (integer) grid indices,
            # not acq_fn_da_plot's (coarsened, non-integer) ones, so assigning the
            # DataArrays directly would realign by label and produce NaN again. Attach
            # the raw values positionally instead.
            acq_fn_da_plot = acq_fn_da_plot.assign_coords(
                lon=(("y", "x"), lonlat_lookup["lon"].data),
                lat=(("y", "x"), lonlat_lookup["lat"].data),
            )

            # X_new_df's lat/lon columns are the same native-grid-index values as
            # acq_fn_da's original coords, so look up their real lon/lat the same
            # way to match acq_fn_da_plot's geographic coords when plotting.
            lonlat_lookup_new = coords_ref.sel(
                x=xr.DataArray(X_new_df[greedy_alg.x2_name].values, dims="points"),
                y=xr.DataArray(X_new_df[greedy_alg.x1_name].values, dims="points"),
                method="nearest",
            )
            X_new_df_plot = X_new_df.copy()
            X_new_df_plot[greedy_alg.x2_name] = lonlat_lookup_new["lon"].values
            X_new_df_plot[greedy_alg.x1_name] = lonlat_lookup_new["lat"].values

            # raw_stn_locs's lat/lon columns are likewise the full-resolution
            # grid-index values used to build X_stn (map_coord_array needs them
            # in that space), so look up their real lon/lat for plotting too.
            lonlat_lookup_stn = coords_ref.sel(
                x=xr.DataArray(raw_stn_locs["lon"].values, dims="points"),
                y=xr.DataArray(raw_stn_locs["lat"].values, dims="points"),
                method="nearest",
            )
            raw_stn_locs_plot = raw_stn_locs.copy()
            raw_stn_locs_plot["lon"] = lonlat_lookup_stn["lon"].values
            raw_stn_locs_plot["lat"] = lonlat_lookup_stn["lat"].values

            ax = myplt.new_placements_plot_xy_hirham5native(
                acq_fn_da_plot,
                X_new_df_plot,
                raw_stn_locs_plot[["lat", "lon"]],
                cmap="YlGnBu",
                title=acq_fn,
            )
            ax.get_figure().savefig(
                os.path.join(
                    figdir,
                    f"newplacements_{run_suffix}.jpg",
                ),
                bbox_inches="tight",
                dpi=300,
            )

        else:
            fig, ax = myplt.new_placements_plot_xy(
                acq_fn_da, X_new_df, raw_stn_locs[["lat", "lon"]], title=acq_fn
            )
            fig.savefig(
                os.path.join(
                    figdir,
                    f"newplacements_{run_suffix}.jpg",
                )
            )
        plt.close("all")


if __name__ == "__main__":
    main()
