"""Main script for Pareto multi-objective experiments using acquisition and cost functions."""

import os

import deepsensor.active_learning.acquisition_fns as ds_acq
import deepsensor.torch as deepsensor  # noqa: F401
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# src
import paretoset
import xarray as xr
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
from omegaconf import DictConfig
from scipy import ndimage

import giant_sp.utils.acquisition as gs_acq
import giant_sp.utils.plots as myplt
from giant_sp.scripts.placement_overview_placements_map import (
    context_dist_km_factor,
    km_per_grid_cell,
)


def load_existing_stations(station_file: str, coords_ref: xr.Dataset) -> pd.DataFrame:
    """Station file's own lat/lon, filtered to the grid's geographic bounding box."""
    stn_df = pd.read_csv(station_file)[["latitude_last_valid", "longitude_last_valid"]]
    stn_df = stn_df.rename(
        columns={"latitude_last_valid": "lat", "longitude_last_valid": "lon"}
    )
    lon_min, lon_max = float(coords_ref["lon"].min()), float(coords_ref["lon"].max())
    lat_min, lat_max = float(coords_ref["lat"].min()), float(coords_ref["lat"].max())
    in_domain = stn_df["lon"].between(lon_min, lon_max) & stn_df["lat"].between(
        lat_min, lat_max
    )
    return stn_df[in_domain].reset_index(drop=True)


def load_coords_ref(coords_file: str) -> xr.Dataset:
    """Full-resolution lon/lat reference grid, indexed by integer x/y (as in placement.py L320-327)."""
    coords_ref = xr.open_zarr(coords_file, chunks="auto")
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


@hydra.main(version_base=None, config_path="../configs", config_name="placement_pareto")
def main(config: DictConfig) -> None:
    """Find Pareto-optimal placements trading off informativeness against a cost.

    Treats each grid cell as a candidate placement scored on two objectives:

    1. the acquisition function from ``acq_file`` at iteration 0 (i.e. with
       only the existing station network as context), averaged over time,
       optimised in its own direction (the acquisition class's
       ``min_or_max``, looked up in `giant_sp.utils.acquisition` and then
       `deepsensor.active_learning.acquisition_fns`);
    2. a cost map on the same grid, minimised, selected by ``cost``:

       - ``"distance2stations"``: distance to existing stations [km], taken
         from the precomputed ``ContextDist`` acquisition-function file
         ``distance2stations_file``;
       - ``"distance2border"``: distance to the edge of the valid
         (non-NaN) domain [km];
       - ``"elevation"``: HIRHAM5 surface elevation [m]
         (``data/hirham5/h5_elevation_GRL.nc``), block-averaged onto the
         acquisition grid;
       - ``"albedo"``: mean ``albedom`` from ``<data_path>/base_dataset.zarr``
         between ``search_start_date`` and ``search_end_date``,
         block-averaged onto the acquisition grid.

    The Pareto front and Pareto rank of every cell are computed with
    `paretoset`. A two-panel figure is then produced: on the left, cost
    against informativeness (the acquisition value, negated if lower is
    better) with the Pareto-optimal cells highlighted; on the right, a
    Greenland map of Pareto rank showing the Pareto-optimal and existing
    stations.

    No model or data processor is loaded: everything is read from
    precomputed files and the data directory.

    Parameters
    ----------
    config : DictConfig
        Hydra-composed run configuration (see ``src/giant_sp/configs/``).
        From ``config.data``, ``data_path`` (holding ``coords_only.zarr``
        and, for ``cost="albedo"``, ``base_dataset.zarr``) and
        ``station_locations_file`` (existing stations to plot) are used.
        ``config.placement_pareto`` holds the run-specific parameters:

        run_ref : str
            Weights & Biases run reference (``"entity/project/run_id"``);
            only its ``run_id`` is used, to name outputs.
        acq_file : str
            Path to a NetCDF of acquisition-function values with
            ``iteration`` and ``time`` dims (as written by ``placement.py``).
            The acquisition-function name is the second ``_``-separated
            token of its file name.
        cost : str
            Cost objective; one of ``"distance2stations"``,
            ``"distance2border"``, ``"elevation"``, ``"albedo"``.
        distance2stations_file : str
            Path to a ``ContextDist`` acquisition-function NetCDF (same
            layout as ``acq_file``); only read for
            ``cost="distance2stations"``.
        search_start_date : str
            Start (inclusive) of the date range over which albedo is
            averaged; only read for ``cost="albedo"``.
        search_end_date : str
            End (inclusive) of the date range over which albedo is
            averaged; only read for ``cost="albedo"``.
        output_dir : str
            Root directory for outputs; results go under
            ``<output_dir>/<run_id>/``.

    Raises
    ------
    ValueError
        If the acquisition-function name parsed from ``acq_file`` or
        ``cost`` is not recognised.

    Returns
    -------
    None
        Written under ``<output_dir>/<run_id>/placement/``:

        - ``pareto/``: the two objective maps
          (``acq_<run_id>_<acq_fn>.npy``, ``cost_<cost>.npy``), the Pareto
          mask and rank (``mask_<run_id>_<acq_fn>_<cost>.npy``,
          ``rank_<run_id>_<acq_fn>_<cost>.npy``), and the Pareto-optimal
          placements in lon/lat (``placements_<run_id>_<acq_fn>_<cost>.csv``);
        - ``figures/``: the combined plot,
          ``<acq_file stem>_<cost>_pareto_combined.png``.
    """
    data_config = config.data
    data_path = data_config.data_path
    station_file = data_config.station_locations_file
    placement_config = config.placement_pareto
    run_ref = placement_config.run_ref
    acq_file = placement_config.acq_file
    cost = placement_config.cost
    output_dir = placement_config.output_dir

    run_id = run_ref.split("/")[-1]
    acqfn_name = acq_file.split("/")[-1].split("_")[1]

    # Determine whether lower ("min") or higher ("max") acquisition values are better
    acq_cls = getattr(gs_acq, acqfn_name, None) or getattr(ds_acq, acqfn_name, None)
    if acq_cls is None:
        raise ValueError(f"Unknown acquisition function in acq_file name: {acqfn_name}")
    acq_sense = acq_cls.min_or_max

    # Path and files
    output_path = f"{output_dir}/{run_id}"
    figdir = os.path.join(output_path, "placement", "figures")
    paretodir = os.path.join(output_path, "placement", "pareto")
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(paretodir, exist_ok=True)

    # Load acquisition function and compute its mean over time at iteration 0
    acqfn = xr.open_dataarray(acq_file)
    acqfn0 = acqfn.sel(iteration=0).mean(dim="time").drop_vars("iteration")
    np.save(f"{paretodir}/acq_{run_id}_{acqfn_name}.npy", acqfn0.data)

    coords_ref_file = os.path.join(data_path, "coords_only.zarr")
    coords_ref = load_coords_ref(coords_ref_file)

    if cost == "distance2stations":
        distance2stations_file = placement_config.distance2stations_file
        acqfn = xr.open_dataarray(distance2stations_file)
        acqfn1 = acqfn.sel(iteration=0).mean(dim="time").drop_vars("iteration")
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
        dates_start = placement_config.search_start_date
        dates_end = placement_config.search_end_date
        ds = xr.open_zarr(os.path.join(data_path, "base_dataset.zarr"), chunks="auto")
        albedo = ds["albedom"].sel(time=slice(dates_start, dates_end)).mean(dim="time")
        # base_dataset only holds valid cells, stacked along ``z``; unstack to
        # (y, x) and pad back to the full native grid.
        albedo = albedo.set_index(z=["y", "x"]).unstack("z")
        albedo = albedo.reindex(x=coords_ref["x"].values, y=coords_ref["y"].values)
        acqfn1 = coarsen_to_field(albedo, acqfn0, coords_ref)
    else:
        raise ValueError(f"Unknown cost: {cost}")

    # Both objectives must share a grid, else the (lat, lon) join below silently
    # drops the non-overlapping cells.
    try:
        xr.align(acqfn0, acqfn1, join="exact")
    except ValueError as e:
        raise ValueError(
            f"Acquisition and '{cost}' cost fields are on different grids; "
            "check they share the same search_resolution."
        ) from e

    np.save(f"{paretodir}/cost_{cost}.npy", acqfn1.data)

    das = [acqfn0, acqfn1]
    dfs = [da.to_dataframe(name=f"alpha_{i + 1}") for i, da in enumerate(das)]
    alpha_df = pd.concat(dfs, axis=1)
    alpha_df = alpha_df.dropna(axis=0)[[f"alpha_{i + 1}" for i in range(len(das))]]

    data = alpha_df.values
    senses = [acq_sense, "min"]  # Acquisition in its own direction; cost minimised
    mask = paretoset.paretoset(data, sense=senses[: len(das)], distinct=False)
    rank = paretoset.paretorank(data, sense=senses[: len(das)], distinct=False)
    np.save(f"{paretodir}/mask_{run_id}_{acqfn_name}_{cost}.npy", mask)
    np.save(f"{paretodir}/rank_{run_id}_{acqfn_name}_{cost}.npy", rank)

    print(f"Found {mask.sum()} pareto optimal solutions with {rank.max()} ranks")

    # Combined scatter + spatial plot of the Pareto front
    existing_df = load_existing_stations(station_file, coords_ref)
    display_crs = myplt.ccrs.NorthPolarStereo(central_longitude=-40)

    X_pareto = alpha_df.reset_index()[["lat", "lon"]][mask]

    # Map ranks back by (lat, lon), not by position: alpha_df only holds cells
    # where both objectives are non-NaN, which can be fewer than acqfn0's.
    rank_da = (
        pd.Series(rank, index=alpha_df.index, name="rank")
        .to_xarray()
        .reindex_like(acqfn0)
    )

    field = reproject_field(rank_da, coords_ref)
    background = field  # iteration-0 field only
    newplace_df = reproject_points(
        X_pareto, coords_ref
    )  # all pareto-optimal placements
    newplace_df.to_csv(f"{paretodir}/placements_{run_id}_{acqfn_name}_{cost}.csv")

    fig = plt.figure(figsize=(16, 8), frameon=False)
    gs = fig.add_gridspec(1, 2)

    # Plot "up = more informative": flip min-is-better scores, leave max-is-better ones.
    informativeness_sign = -1 if acq_sense == "min" else 1

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(
        alpha_df["alpha_2"],
        alpha_df["alpha_1"] * informativeness_sign,
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
        y_pareto[:, 0] * informativeness_sign,
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
        "Informativeness\n(ConvCNP Acquisition Function)", size=label_fontsize
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
    cb = plt.colorbar(pcm, cax=cbar_ax, ticks=np.arange(0, rank.max() + 1, 5))
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
