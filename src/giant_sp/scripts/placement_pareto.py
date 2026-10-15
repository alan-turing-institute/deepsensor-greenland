"""Main script for Pareto multi-objective experiments using acquisition and cost functions."""

import glob
import json
import os

import deepsensor.torch as deepsensor  # noqa: F401
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# src
import paretoset
import torch
import wandb
import xarray as xr
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.model.convnp import ConvNP
from deepsensor.train.train import set_gpu_default_device
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from omegaconf import DictConfig
from scipy import ndimage

import giant_sp.utils.plots as myplt
from giant_sp.data import CARRA2DataLoader, HIRHAM5DataLoader, HIRHAM5ZarrDataLoader
from giant_sp.data.dataloader import BaseDataLoader
from giant_sp.models.mdn import MDNZIGammaConvNP, install_mdn_zigamma_likelihood
from giant_sp.scripts.placement_overview_placements_map import (
    context_dist_km_factor,
    km_per_grid_cell,
)
from giant_sp.utils.data import sample_dates_by_year_chunks
from giant_sp.utils.model import get_available_device

set_gpu_default_device() if torch.cuda.is_available() else torch.device("cpu")

DATA_PATH = "/gws/pw/j07/aria_giant/wip/draft/acocac/data/hirham5/processed/HIRHAM5-ERAInterim/v_02"
COORDS_FILE = os.path.sep.join([DATA_PATH, "coords_only.zarr"])
STATION_FILE = "/gws/pw/j07/aria_giant/wip/draft/acocac/deepsensor/data/greenland/station/processed/promice_locations.csv"


def load_existing_stations(coords_ref: xr.Dataset) -> pd.DataFrame:
    """Station file's own lat/lon, filtered to the grid's geographic bounding box."""
    stn_df = pd.read_csv(STATION_FILE)[["latitude_last_valid", "longitude_last_valid"]]
    stn_df = stn_df.rename(
        columns={"latitude_last_valid": "lat", "longitude_last_valid": "lon"}
    )
    lon_min, lon_max = float(coords_ref["lon"].min()), float(coords_ref["lon"].max())
    lat_min, lat_max = float(coords_ref["lat"].min()), float(coords_ref["lat"].max())
    in_domain = stn_df["lon"].between(lon_min, lon_max) & stn_df["lat"].between(
        lat_min, lat_max
    )
    return stn_df[in_domain].reset_index(drop=True)


def load_coords_ref() -> xr.Dataset:
    """Full-resolution lon/lat reference grid, indexed by integer x/y (as in placement.py L320-327)."""
    coords_ref = xr.open_zarr(COORDS_FILE, chunks="auto")
    return coords_ref.assign_coords(
        x=np.arange(coords_ref.sizes["x"]), y=np.arange(coords_ref.sizes["y"])
    )


def reproject_field(field: xr.DataArray, coords_ref: xr.Dataset) -> xr.DataArray:
    """Attach true 2D lon/lat coords to a field indexed by native-grid lat/lon *indices*."""
    field = field.rename({"lon": "x", "lat": "y"})
    lookup = coords_ref.sel(x=field["x"], y=field["y"], method="nearest")
    return field.assign_coords(
        lon=(("y", "x"), lookup["lon"].data), lat=(("y", "x"), lookup["lat"].data)
    )


def reproject_points(df: pd.DataFrame, coords_ref: xr.Dataset) -> pd.DataFrame:
    """Look up true lon/lat for new-placement rows (whose lat/lon columns are grid indices)."""
    lookup = coords_ref.sel(
        x=xr.DataArray(df["lon"].values, dims="points"),
        y=xr.DataArray(df["lat"].values, dims="points"),
        method="nearest",
    )
    return pd.DataFrame({"lat": lookup["lat"].values, "lon": lookup["lon"].values})


def coarsen_to_field(
    da: xr.DataArray, field: xr.DataArray, coords_ref: xr.Dataset
) -> xr.DataArray:
    """Block-average a full-resolution (y, x) field onto ``field``'s coarse grid.

    ``da`` is indexed by native (y, x) dims without coordinates; the integer
    x/y indices of ``coords_ref`` are attached so it can be matched to
    ``field``, whose lat/lon coords are full-resolution grid indices with a
    spacing >1. Each output cell is the mean of the spacing x spacing window
    centred on that index.

    Parameters
    ----------
    da : xr.DataArray
        Full-resolution 2D field with dims (y, x), same shape as ``coords_ref``.
    field : xr.DataArray
        Coarse 2D reference field with dims (lat, lon) holding grid indices.
    coords_ref : xr.Dataset
        Full-resolution reference grid, indexed by integer x/y.

    Returns
    -------
    xr.DataArray
        ``da`` averaged onto ``field``'s (lat, lon) grid.
    """
    da = da.drop_vars(["lon", "lat"], errors="ignore").assign_coords(
        x=coords_ref["x"].values, y=coords_ref["y"].values
    )
    spacing = {
        d: round(float(np.abs(np.median(np.diff(field[c].values)))))
        for d, c in (("y", "lat"), ("x", "lon"))
    }
    smoothed = da.rolling(spacing, center=True, min_periods=1).mean()
    out = smoothed.sel(y=field["lat"].values, x=field["lon"].values, method="nearest")
    return (
        out.rename({"y": "lat", "x": "lon"})
        .assign_coords(lat=field["lat"].values, lon=field["lon"].values)
        .transpose(*field.dims)
    )


def distance_to_border(field: xr.DataArray) -> xr.DataArray:
    """Distance (in full-resolution grid cells) from each valid cell to the edge of the valid region.

    The border is the edge of the non-NaN region of ``field`` -- i.e. the
    transition to NaN cells (e.g. ocean/masked areas) as well as the outer
    edges of the array itself. ``field``'s coords are full-resolution grid
    indices, so their spacing (>1 for a coarsened grid) is used as the
    per-axis cell size.

    Parameters
    ----------
    field : xr.DataArray
        2D field whose non-NaN cells define the domain.

    Returns
    -------
    xr.DataArray
        Same shape/coords as ``field``; Euclidean distance to the nearest
        border cell, NaN outside the valid region.
    """
    valid = ~np.isnan(field.data)
    spacing = [float(np.abs(np.median(np.diff(field[d].values)))) for d in field.dims]
    # Pad with invalid cells so the array edges also count as border.
    dist = ndimage.distance_transform_edt(
        np.pad(valid, 1, constant_values=False), sampling=spacing
    )[1:-1, 1:-1]
    out = field.copy(data=dist.astype(float))
    out.data[~valid] = np.nan
    return out


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


@hydra.main(version_base=None, config_path="../configs", config_name="placement_pareto")
def main(config: DictConfig) -> None:
    """Find Pareto-optimal placements trading off informativeness against a cost.

    Treats each grid cell as a candidate placement scored on two objectives,
    both minimised:

    1. the acquisition function from ``acq_file`` at iteration 0 (i.e. with
       only the existing station network as context), averaged over time;
    2. a cost map on the same grid, selected by ``cost``:

       - ``"distance2stations"``: distance to existing stations [km], taken
         from a precomputed ``ContextDist`` acquisition-function file;
       - ``"distance2border"``: distance to the edge of the valid
         (non-NaN) domain [km];
       - ``"elevation"``: HIRHAM5 surface elevation [m], block-averaged
         onto the acquisition grid;
       - ``"albedo"``: mean ``albedom`` over the search date range,
         block-averaged onto the acquisition grid.

    The Pareto front and Pareto rank of every cell are computed with
    `paretoset`. A two-panel figure is then produced: on the left, cost
    against informativeness (the negated acquisition value) with the
    Pareto-optimal cells highlighted; on the right, a Greenland map of
    Pareto rank showing the Pareto-optimal and existing stations.

    The run's data processor is fetched from Weights & Biases (or a local
    ``wandb/`` cache) so that the data loader is built consistently with
    the trained model; the model itself is not loaded.

    Parameters
    ----------
    config : DictConfig
        Hydra-composed run configuration (see ``src/giant_sp/configs/``).
        ``config.placement_analysis`` holds the run-specific parameters:

        run_ref : str
            Weights & Biases run reference (``"entity/project/run_id"``)
            whose data processor config is loaded.
        acq_file : str
            Path to a NetCDF of acquisition-function values with
            ``iteration`` and ``time`` dims (as written by ``placement.py``).
            The acquisition-function name is the second ``_``-separated
            token of its file name; the weights file name is derived from the
            last token, with ``.nc`` replaced by ``.pt``.
        cost : str
            Cost objective; one of ``"distance2stations"``,
            ``"distance2border"``, ``"elevation"``, ``"albedo"``.
        search_start_date, search_end_date : str
            Date range (inclusive) over which dates are sampled, and over
            which albedo is averaged for ``cost="albedo"``.
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
        Written under ``<output_dir>/<run_id>/placement/``:

        - ``pareto/``: the two objective maps
          (``acq_<run_id>_<acq_fn>.npy``, ``cost_<cost>.npy``), the Pareto
          mask and rank (``mask_*.npy``, ``rank_*.npy``), and the
          Pareto-optimal placements in lon/lat (``placements_*.csv``);
        - ``figures/``: the combined plot,
          ``<acq_file stem>_<cost>_pareto_combined.png``.

    Raises
    ------
    ValueError
        If ``config["data"]["source"]`` is not one of ``"carra2"``,
        ``"hirham5"``, ``"hirham5zarr"``; if the temporal sampling method is
        not ``"seasonal"`` or ``"regular"``; or if ``cost`` is not one of
        the values listed above.
    """
    placement_config = config.placement_pareto
    run_ref = placement_config.run_ref
    acq_file = placement_config.acq_file
    cost = placement_config.cost
    dates_start = placement_config.search_start_date
    dates_end = placement_config.search_end_date
    output_dir = placement_config.output_dir

    weights_fn = acq_file.split("/")[-1].split("_")[-1].replace(".nc", ".pt")
    run_id = run_ref.split("/")[-1]
    acqfn_name = acq_file.split("/")[-1].split("_")[1]

    # Path and files
    output_path = f"{output_dir}/{run_id}"
    figdir = os.path.join(output_path, "placement", "figures")
    paretodir = os.path.join(output_path, "placement", "pareto")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(paretodir, exist_ok=True)

    # coords_path = config["data"]["data_path"]
    source = config["data"]["source"]

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
    print("Loading run data from: ", files_path)

    data_processor = DataProcessor(files_path)
    _, data_processor, _ = data_loader.get_dataproc_taskload(
        dates=dates, data_processor=data_processor
    )

    acqfn = xr.open_dataarray(acq_file)
    acqfn0 = acqfn.sel(iteration=0).mean(dim="time").drop_vars(("iteration"))
    np.save(f"{paretodir}/acq_{run_id}_{acqfn_name}.npy", acqfn0.data)

    coords_ref = load_coords_ref()

    if cost == "distance2stations":
        acqfn = xr.open_dataarray(
            "outputs/vloyijco/placement/ContextDist/AcqFn_ContextDist_10-iter_sseasonal_e50_sr9_weights-ep95.nc"
        )
        acqfn1 = acqfn.sel(iteration=0).mean(dim="time").drop_vars(("iteration"))
        # ContextDist is in normalised-coordinate units; convert to km.
        acqfn1 = acqfn1 * context_dist_km_factor(coords_ref)
    elif cost == "distance2border":
        # Grid-cell distance; convert to km.
        acqfn1 = distance_to_border(acqfn0) * km_per_grid_cell(coords_ref)
    elif cost == "elevation":
        # Elevation is already in the correct units.
        elev = xr.open_dataarray("data/hirham5/h5_elevation_GRL.nc")
        acqfn1 = coarsen_to_field(elev, acqfn0, coords_ref)
    elif cost == "albedo":
        ds = xr.open_zarr(os.path.join(DATA_PATH, "base_dataset.zarr"), chunks="auto")
        albedo = ds["albedom"].sel(time=slice(dates_start, dates_end)).mean(dim="time")
        # base_dataset only holds valid cells, stacked along ``z``; unstack to
        # (y, x) and pad back to the full native grid.
        albedo = albedo.set_index(z=["y", "x"]).unstack("z")
        albedo = albedo.reindex(x=coords_ref["x"].values, y=coords_ref["y"].values)
        acqfn1 = coarsen_to_field(albedo, acqfn0, coords_ref)
    else:
        raise ValueError(f"Unknown cost: {cost}")

    np.save(f"{paretodir}/cost_{cost}.npy", acqfn1.data)

    das = [acqfn0, acqfn1]
    dfs = [da.to_dataframe(name=f"alpha_{i + 1}") for i, da in enumerate(das)]
    alpha_df = pd.concat(dfs, axis=1)
    alpha_df = alpha_df.dropna(axis=0)[[f"alpha_{i + 1}" for i in range(len(das))]]

    data = alpha_df.values
    senses = ["min", "min"]  # Both objectives are to be minimised
    mask = paretoset.paretoset(data, sense=senses[: len(das)], distinct=False)
    rank = paretoset.paretorank(data, sense=senses[: len(das)], distinct=False)
    np.save(f"{paretodir}/mask_{run_id}_{acqfn_name}_{cost}.npy", mask)
    np.save(f"{paretodir}/rank_{run_id}_{acqfn_name}_{cost}.npy", rank)

    print(f"Found {mask.sum()} pareto optimal solutions with {rank.max()} ranks")

    # Combined scatter + spatial plot of the Pareto front
    existing_df = load_existing_stations(coords_ref)
    display_crs = myplt.ccrs.NorthPolarStereo(central_longitude=-40)

    X_pareto = alpha_df.reset_index()[["lat", "lon"]][mask]

    rank_da = acqfn0.copy()
    rank_da.data[~np.isnan(rank_da.data)] = rank

    field = reproject_field(rank_da, coords_ref)
    background = field  # iteration-0 field only
    newplace_df = reproject_points(X_pareto, coords_ref)  # all 10 placements
    newplace_df.to_csv(f"{paretodir}/placements_{run_id}_{acqfn_name}_{cost}.csv")

    fig = plt.figure(figsize=(16, 8), frameon=False)
    gs = fig.add_gridspec(1, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(
        acqfn1.data.flatten(),
        acqfn0.data.flatten() * -1,
        marker="o",
        color="k",
        s=6**2,
        linewidths=0.35,
        label="All solutions",
        facecolor="none",
        alpha=0.3,
    )
    y_pareto = alpha_df.reset_index()[["alpha_1", "alpha_2"]].values[mask]
    ax0.scatter(
        y_pareto[:, 1],
        y_pareto[:, 0] * -1,
        marker="o",
        color="r",
        s=6**2,
        linewidths=0.35,
        label="Pareto optimal",
    )
    ax0.legend(loc="lower right", prop={"size": 15})
    label_fontsize = 20
    cost_label = {
        "distance2stations": "Distance to existing stations [km]",
        "distance2border": "Distance to border [km]",
        "elevation": "Elevation [m]",
        "albedo": "Albedo [unitless]",
    }[cost]
    ax0.set_xlabel(f"Cost ({cost_label})", size=label_fontsize)
    ax0.set_ylabel(
        "Informativeness\n(ConvCNP Adquisition Function)", size=label_fontsize
    )
    ax0.set_title("")
    ax0.tick_params(axis="both", which="major", labelsize=18)
    for spine in ax0.spines.values():
        spine.set_color("gray")

    ax1 = fig.add_subplot(gs[0, 1], projection=display_crs)
    ax1, pcm = myplt.plot_greenland_only(
        background,
        newplace_df,
        existing_df,
        ax=ax1,
        pcolormesh_kwargs={"cmap": "Greys"},
    )

    # Enlarge the new/existing placement markers and drop the row-index
    # annotations, without touching plot_greenland_only's own defaults.
    for coll in ax1.collections:
        if isinstance(coll, PathCollection):
            coll.set_sizes([8**2])
    for txt in list(ax1.texts):
        txt.remove()

    existing_handle = Line2D(
        [],
        [],
        marker="x",
        color="k",
        linestyle="None",
        markersize=8,
        label="Existing stations",
    )
    ax1.legend(
        handles=[existing_handle], loc="lower right", frameon=True, prop={"size": 15}
    )
    ax1.spines["geo"].set_color("gray")

    fig.tight_layout(rect=(0.0, 0.0, 0.9, 1.0))

    # Position the colorbar against ax1's *actual* (aspect-locked) bounding
    # box, computed only after layout/draw, so it sits flush against the map
    # rather than at a fixed guessed offset.
    fig.canvas.draw()
    pos1 = ax1.get_position()
    cbar_ax = fig.add_axes((pos1.x1 + 0.01, pos1.y0, 0.03, pos1.height))
    cbar_kwargs = {
        "fraction": 0.0,
        "pad": 0.00,
        "ticks": np.arange(0, rank.max() + 1, 5),
    }
    cb = plt.colorbar(pcm, cax=cbar_ax, **cbar_kwargs)
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=16)
    cb.set_label("Pareto rank", size=label_fontsize)

    fig.savefig(
        os.path.join(
            figdir,
            acq_file.split("/")[-1].replace(".nc", f"_{cost}_pareto_combined.png"),
        ),
        dpi=200,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
