"""Plotting utilities for DeepSensor predictions and placement diagnostics.

Includes plots for comparing predicted vs. true fields (with uncertainty
and error), probability-integral-transform (PIT) and hexbin parity
diagnostics, and maps of acquisition-function values with new/existing
sensor placements over Greenland.
"""

from typing import Any, Literal, cast

import cartopy.crs as ccrs
import cartopy.feature as cf
import deepsensor.plot as dsplt
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from cartopy.feature import ShapelyFeature
from cartopy.mpl.geoaxes import GeoAxes
from deepsensor.data.processor import DataProcessor
from deepsensor.data.task import Task
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray

from giant_sp.utils.data import (
    find_lon_lat,
    load_greenland_shape,
    normalize_lons,
    points_in_polygons,
)
from giant_sp.utils.metrics import compute_pit_bernoulli_gamma


def averaged_prediction_plot(
    truth_da: xr.DataArray,
    mean_da: xr.DataArray,
    std_da: xr.DataArray,
    cmap: str = "seismic",
    crs: ccrs.CRS | None = None,
    highlight_negatives: bool = False,
    task: Task | None = None,
    dataproc: DataProcessor | None = None,
    plot_xc: bool = False,
) -> tuple[Figure, NDArray[Any]]:
    """Plot time-averaged truth, prediction, uncertainty and error fields.

    Produces a 2x2 grid: ground truth, predicted mean, predicted uncertainty
    (std), and prediction error (mean - truth), each averaged over the
    ``time`` dimension. `mean_da` and `truth_da` are aligned (inner join)
    before computing the error.

    Parameters
    ----------
    truth_da : xarray.DataArray
        Ground truth field with a ``time`` dimension.
    mean_da : xarray.DataArray
        Predicted mean field with a ``time`` dimension.
    std_da : xarray.DataArray
        Predicted standard deviation field with a ``time`` dimension.
    cmap : str, default "seismic"
        Colormap used for the truth and (non-highlighted) mean plots.
    crs : cartopy.crs.CRS or None, default None
        If provided, subplots use this as their map projection and
        coastlines are drawn on every axis.
    highlight_negatives : bool, default False
        If True, the predicted mean is plotted with the ``"viridis"``
        colormap using the same color limits as the ground truth colorbar,
        which makes negative/out-of-range values easier to spot. If False,
        the mean is plotted with `cmap` using symmetric limits derived from
        both fields.
    task : deepsensor.data.task.Task or None, default None
        Task whose off-grid context locations are overlaid on the ground
        truth axis. Only used if `plot_xc` is True.
    dataproc : deepsensor.data.processor.DataProcessor or None, default None
        Data processor used to un-normalise context locations when plotting
        them. Only used if `plot_xc` is True.
    plot_xc : bool, default False
        If True, overlay `task`'s off-grid context locations on the ground
        truth axis.

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
        The figure and a flattened array of its 4 :class:`~matplotlib.axes.Axes`
        (or ``GeoAxes`` if `crs` is given), in the order
        ``[truth, mean, std, error]``.
    """
    mean_da, truth_da = xr.align(mean_da, truth_da, join="inner")
    # Calculate prediction errors
    err_da = mean_da - truth_da

    # Set up subplots
    if crs is None:
        fig, axes = plt.subplots(
            nrows=2, ncols=2, figsize=(10, 5), layout="constrained"
        )
    else:
        fig, axes = plt.subplots(
            subplot_kw={"projection": crs},
            nrows=2,
            ncols=2,
            figsize=(10, 5),
            layout="constrained",
        )
    axes = axes.flatten()
    # Title
    date_min = pd.Timestamp(truth_da.time.values.min()).strftime("%Y-%m-%d")
    date_max = pd.Timestamp(truth_da.time.values.max()).strftime("%Y-%m-%d")
    fig.suptitle(f"{truth_da.name} predictions ({date_min} - {date_max} average)")

    vmin = min(
        mean_da.mean(dim="time").min().compute(),
        truth_da.mean(dim="time").min().compute(),
    )
    vmax = max(
        mean_da.mean(dim="time").max().compute(),
        truth_da.mean(dim="time").max().compute(),
    )

    maxabs = np.max(np.abs([vmin, vmax]))
    var_clim = (-maxabs, maxabs)

    # Plot ground truth
    im_ratio = truth_da.shape[1] / truth_da.shape[2]
    img_truth = truth_da.mean(dim="time").plot(  # type: ignore[call-arg]
        ax=axes[0],
        cmap=cmap,
        vmin=var_clim[0],
        vmax=var_clim[1],
        cbar_kwargs={"fraction": 0.046 * im_ratio, "pad": 0.04, "label": None},
    )
    axes[0].set_title(f"{truth_da.name}")

    # Plot predicted mean
    if not highlight_negatives:
        mean_da.mean(dim="time").plot(  # type: ignore[call-arg]
            ax=axes[1],
            cmap=cmap,
            vmin=var_clim[0],
            vmax=var_clim[1],
            cbar_kwargs={"fraction": 0.046 * im_ratio, "pad": 0.04, "label": None},
        )
    else:
        # DataArray.plot()'s `cmap` also accepts a Colormap instance (not just
        # a name); use a separate variable since `cmap` is typed as `str`.
        highlight_cmap = plt.get_cmap("viridis").copy()
        # highlight_cmap.set_under('orange')
        norm = plt.Normalize(img_truth.colorbar.vmin, img_truth.colorbar.vmax)
        mean_da.mean(dim="time").plot(  # type: ignore[call-arg]
            ax=axes[1],
            vmin=img_truth.colorbar.vmin,
            vmax=img_truth.colorbar.vmax,
            cmap=highlight_cmap,
            norm=norm,
            cbar_kwargs={"fraction": 0.046 * im_ratio, "pad": 0.04, "label": None},
        )
    axes[1].set_title("ConvNP mean")

    # Plot predicted uncertainty
    std_da.mean(dim="time").plot(  # type: ignore[call-arg]
        ax=axes[2],
        #    vmin=0.,
        cmap="Greys",
        cbar_kwargs={"fraction": 0.046 * im_ratio, "pad": 0.04, "label": None},
    )
    axes[2].set_title("ConvNP uncertainty (std)")

    # Plot prediction error
    err_da.mean(dim="time").plot(  # type: ignore[call-arg]
        ax=axes[3],
        cmap="seismic",  # "RdBu_r"
        cbar_kwargs={"fraction": 0.046 * im_ratio, "pad": 0.04, "label": None},
    )
    axes[3].set_title("ConvNP error")

    # Plot context locations on ground truth plot
    if plot_xc:
        dsplt.offgrid_context(
            axes[0],
            task,
            dataproc,
            s=3,
            markers="x",
            linewidths=0.5,
            colors="green",
            add_legend=False,
        )

    if crs is not None:
        for ax in axes:
            ax.coastlines(linewidth=0.5)

    return fig, axes


def PIT_histogram(
    truth: NDArray[np.floating],
    val_pred: xr.Dataset,
    likelihood: str,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a probability integral transform (PIT) histogram.

    Computes PIT values for the (flattened, NaN-masked) predictions and
    plots their histogram against the uniform density expected for a
    well-calibrated model.

    Parameters
    ----------
    truth : NDArray[np.floating]
        True values, broadcastable/indexable in the same flattened order as
        `val_pred`'s data variables.
    val_pred : xarray.Dataset
        Predictive distribution parameters. Must contain a ``"mean"`` and
        ``"std"`` variable; if `likelihood` is ``"bernoulli-gamma"`` it must
        also contain ``"mixture_probs_0"``, ``"k"`` and ``"scale"``.
    likelihood : str
        Predictive likelihood family. If ``"bernoulli-gamma"``, PIT values
        are computed with :func:`~giant_sp.utils.metrics.compute_pit_bernoulli_gamma`
        to account for the point mass at zero; otherwise a Normal CDF
        (``scipy.stats.norm.cdf``) is used with `val_pred`'s mean/std.
    **kwargs : Any
        Accepted for interface compatibility but currently unused.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axis containing the PIT histogram.
    """
    pred_mean_plt_all = val_pred["mean"].values.flatten()
    pred_std_plt = val_pred["std"].values.flatten()
    valid = ~np.isnan(pred_mean_plt_all)
    pred_mean_plt = pred_mean_plt_all[valid]
    pred_std_plt = pred_std_plt[valid]
    true_plt = truth[valid]

    # PIT histogram (area weighted)
    # Compute PIT values
    if likelihood == "bernoulli-gamma":
        # Zero-inflated (non-continuous) predictive distribution: PIT must
        # account for the point mass at zero, not just mean/std.
        p_zero_plt = val_pred["mixture_probs_0"].values.flatten()[valid]
        k_plt = val_pred["k"].values.flatten()[valid]
        scale_plt = val_pred["scale"].values.flatten()[valid]
        pit_values = compute_pit_bernoulli_gamma(true_plt, p_zero_plt, k_plt, scale_plt)
    else:
        pit_values = sp.stats.norm.cdf(true_plt, loc=pred_mean_plt, scale=pred_std_plt)
    # Plot
    fig, ax = plt.subplots()
    ax.hist(pit_values, bins=100, range=(0, 1), density=True, label="ConvNP model")
    ax.set_xlabel("Probability integral transform")
    ax.set_ylabel("Density")
    ax.set_title("PIT histogram")
    ax.grid()
    # Plot uniform line
    ax.axhline(1.0, color="k", linestyle="--")

    return fig, ax


def hexbin_parity_plot(x: ArrayLike, y: ArrayLike) -> tuple[Figure, Axes]:
    """Plot a hexbin parity (predicted vs. true) diagram with a 1:1 line.

    Parameters
    ----------
    x : array-like
        True values, plotted on the x-axis.
    y : array-like
        Predicted values, plotted on the y-axis.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axis containing the hexbin plot, with equal aspect
        ratio and a black bisecting (1:1) line.
    """
    # Plot
    fig, ax = plt.subplots()
    # hb = ax.hexbin(x, y, bins='log')
    gridsize = 50
    hb = ax.hexbin(
        x,
        y,
        gridsize=gridsize,
        bins="log",
        cmap="RdYlBu_r",
        linewidths=(0,),  # type: ignore[arg-type]
        mincnt=1,
        vmin=1,
    )
    fig.colorbar(hb, ax=ax, label="counts")
    ax.set_xlabel("True")
    ax.set_ylabel("Pred.")
    ax.grid()

    # Manage limits and aspect ratio
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # maxes of both axes
    ]
    # Plot bisecting 1:1 line
    ax.plot(lims, lims, "k-", alpha=0.75, linewidth=0.1, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(lims)  # type: ignore[arg-type]
    ax.set_ylim(lims)  # type: ignore[arg-type]

    return fig, ax


def new_placements_plot_xy(
    acqfn_da: xr.DataArray,
    newplace_df: pd.DataFrame,
    Xc_df: pd.DataFrame,
    cmap: str = "Greys",
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the acquisition function (in x/y index space) with placements.

    Plots the first iteration's acquisition function values (averaged over
    ``time``) as an image, without geographic coordinates or axes ticks, and
    overlays new and existing placement locations.

    Parameters
    ----------
    acqfn_da : xarray.DataArray
        Acquisition function values with ``iteration`` and ``time``
        dimensions. Only ``iteration == 0`` is plotted.
    newplace_df : pandas.DataFrame
        New placement coordinates, with 2 columns (e.g. y, x), plotted as
        green ``"+"`` markers.
    Xc_df : pandas.DataFrame
        Existing placement coordinates, with 2 columns, plotted as black
        ``"x"`` markers.
    cmap : str, default "Greys"
        Colormap for the acquisition function image.
    title : str or None, default None
        Title placed above the plot.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The figure and axis containing the plot.
    """
    fig, ax = plt.subplots(layout="constrained")

    plot_da = acqfn_da.sel(iteration=0).mean(dim="time")
    vmax = float(plot_da.max().compute())
    vmin = float(plot_da.min().compute())
    plot_da.plot(  # type: ignore[call-arg]
        ax=ax,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        add_colorbar=True,
        center=False,
        cbar_kwargs={"label": "Acquisition function value"},
    )
    ax.set_title(title, pad=10)  # type: ignore[arg-type]

    ax.scatter(  # type: ignore[misc]
        *newplace_df.values.T[::-1], marker="+", s=20**2, c="g", label="New placements"
    )
    ax.scatter(  # type: ignore[misc]
        *Xc_df.values.T[::-1], marker="x", s=10**2, c="k", label="Existing placements"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax


def plot_greenland_only(
    gris_data: xr.DataArray,
    newplace_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    ax: GeoAxes | None = None,
    masked: bool = True,
    ocean_color: str | None = "white",
    gridlines: bool = False,
    pcolormesh_kwargs: dict[str, Any] | None = None,
) -> tuple[GeoAxes, QuadMesh]:
    """Plot a 2D field over Greenland only, on a north-polar stereographic map.

    Modified from: https://github.com/eschlager/MeltEmulation/

    Masks `gris_data` to the Greenland land polygon (from Natural Earth),
    draws the Greenland outline and (optionally) the ocean, and overlays new
    and existing placement locations annotated with their (1-indexed) row
    number.

    Parameters
    ----------
    gris_data : xarray.DataArray
        2D field (e.g. ``lat``/``lon`` or ``y``/``x``) to plot.
    newplace_df : pandas.DataFrame
        New placement locations, indexed as ``(lat, lon)`` pairs (row order
        matches ``.values``), plotted as red ``"+"`` markers and annotated
        with ``index + 1``.
    existing_df : pandas.DataFrame
        Existing placement locations, indexed as ``(lat, lon)`` pairs,
        plotted as black ``"x"`` markers.
    ax : cartopy.mpl.geoaxes.GeoAxes or None, default None
        Axis to plot on. If None, a new figure and
        :class:`~cartopy.crs.NorthPolarStereo` axis are created.
    masked : bool, default True
        If True, mask out data outside the Greenland polygon before
        plotting.
    ocean_color : str or None, default "white"
        Facecolor for the ocean feature. If None, the ocean is left
        transparent.
    gridlines : bool, default False
        If True, draw latitude/longitude gridlines.
    pcolormesh_kwargs : dict[str, Any] or None, default None
        Extra keyword arguments forwarded to ``ax.pcolormesh`` (e.g.
        ``cmap``, ``vmin``, ``vmax``). Must be provided (unpacked with
        ``**``), so a plain call without this argument will raise a
        ``TypeError``.

    Returns
    -------
    tuple[cartopy.mpl.geoaxes.GeoAxes, matplotlib.collections.QuadMesh]
        The axis plotted on, and the ``pcolormesh`` mappable (useful for
        building a shared colorbar).

    Raises
    ------
    ValueError
        If `gris_data` is not 2D, or if its lon/lat coordinate shapes do not
        match its data shape.
    """
    # Modified from: https://github.com/eschlager/MeltEmulation/
    ds = gris_data

    if ds.ndim != 2:
        raise ValueError("Expected a 2D field (lat,lon or y,x).")

    lon, lat = find_lon_lat(ds)

    # If lon/lat are 1D, make meshgrid
    Lon: NDArray[np.floating] | xr.DataArray
    if lon.ndim == 1 and lat.ndim == 1:
        Lon, Lat = np.meshgrid(lon, lat)
    elif lon.shape == ds.shape and lat.shape == ds.shape:
        Lon, Lat = lon, np.array(lat)
    else:
        raise ValueError(
            f"lon/lat shapes {lon.shape},{lat.shape} do not match data shape {ds.shape}"
        )

    # Normalize longitudes to -180..180 to match Natural Earth polygons
    Lon = normalize_lons(Lon)
    # load Greenland shapely geometry
    greenland_geom = load_greenland_shape(resolution="50m")

    # Create mask of points inside Greenland polygons (using cell centers)
    inside_mask = points_in_polygons(Lon, Lat, greenland_geom)

    # Mask the data outside Greenland
    data = ds.values
    if masked:
        data = np.ma.array(data, mask=~inside_mask)

    # Plot
    display_crs = ccrs.NorthPolarStereo(central_longitude=-40)
    data_crs = ccrs.PlateCarree()

    if ax is None:
        fig = plt.figure(figsize=(8, 10), frameon=False)
        ax = fig.add_subplot(1, 1, 1, projection=display_crs)
        plt.axis("off")

    # Make ocean white if not None, otherwise default is transparent!
    if ocean_color is not None:
        ax.add_feature(cf.OCEAN.with_scale("50m"), facecolor=ocean_color, zorder=0)
    # Fill Greenland with light gray (use ShapelyFeature for correct reprojection)
    greenland_feature = ShapelyFeature(
        [greenland_geom],
        crs=ccrs.PlateCarree(),
        facecolor="lightgray",
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.add_feature(greenland_feature)

    # Set a Greenland-focused extent in lon/lat
    ax.set_extent((-55, -25, 58, 85), crs=ccrs.PlateCarree())

    # Latitude gridlines visible
    if gridlines:
        ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=False,
            linewidth=1,
            color="gray",
            alpha=0.7,
            linestyle="--",
        )

    ax.scatter(
        *newplace_df.values.T[::-1],
        transform=data_crs,
        marker="+",
        s=3**2,
        c="r",
        zorder=5,
        label="New placements",
    )
    ax.scatter(
        *existing_df.values.T[::-1],
        transform=data_crs,
        marker="x",
        s=3**2,
        c="k",
        zorder=5,
        label="Existing placements",
    )

    # annotate new placements with their row index
    for row_idx, (lat_val, lon_val) in zip(
        newplace_df.index, newplace_df.values, strict=True
    ):
        ax.text(
            lon_val,
            lat_val,
            str(row_idx + 1),
            transform=data_crs,
            color="r",
            fontsize=6,
            fontweight="bold",
            zorder=6,
            ha="left",
            va="bottom",
        )

    pcm = ax.pcolormesh(
        Lon,
        Lat,
        data,
        transform=data_crs,
        shading="auto",
        rasterized=True,
        zorder=4,
        # Note: raises TypeError if pcolormesh_kwargs is left as None (the
        # default); every current caller always passes a dict.
        **pcolormesh_kwargs,  # type: ignore[arg-type]
    )

    return ax, pcm


def _get_value_limits(
    minval: float,
    maxval: float,
    value_lim: tuple[float | None, float | None] | None,
) -> tuple[float, float, Literal["neither", "min", "max", "both"]]:
    """Determine effective value limits and colorbar extension.

    Source: https://github.com/eschlager/MeltEmulation/

    Parameters
    ----------
    minval : float
        Actual minimum value of the data.
    maxval : float
        Actual maximum value of the data.
    value_lim : tuple[float or None, float or None] or None
        Desired ``(value_min, value_max)`` limits. Either element may be
        None to fall back to `minval`/`maxval` respectively. If the whole
        tuple is None, no clipping is applied.

    Returns
    -------
    tuple[float, float, {"neither", "min", "max", "both"}]
        The effective ``(minval, maxval, extend_colorbar)``, where
        `extend_colorbar` indicates whether the data range exceeds the
        requested limits on the low end (``"min"``), high end (``"max"``),
        both (``"both"``), or neither (``"neither"``).
    """
    # Source: https://github.com/eschlager/MeltEmulation/
    extend_colorbar: Literal["neither", "min", "max", "both"]
    if value_lim is not None:
        value_min, value_max = value_lim
        if value_max is None:
            value_max = maxval
        if value_min is None:
            value_min = minval

        if maxval > value_max:
            extend_colorbar = "both" if minval < value_min else "max"
        elif minval < value_min:
            extend_colorbar = "min"
        else:
            extend_colorbar = "neither"
        maxval = value_max
        minval = value_min
    else:
        extend_colorbar = "neither"

    return minval, maxval, extend_colorbar


def new_placements_plot_xy_hirham5native(
    ds: xr.DataArray,
    newplace_df: pd.DataFrame,
    existing_df: pd.DataFrame,
    cmap: str = "Grays",
    title: str | None = None,
) -> GeoAxes:
    """Plot acquisition maps for iterations 1, 5 and 10 with a shared colorbar.

    Modified from: https://github.com/eschlager/MeltEmulation/

    Draws three side-by-side :func:`plot_greenland_only` panels (for
    ``iteration`` indices 0, 4 and 9), sharing a single colorbar with value
    limits and extension computed from all three panels combined.

    Parameters
    ----------
    ds : xarray.DataArray
        Acquisition function values with ``iteration`` and ``time``
        dimensions, passed to :func:`plot_greenland_only` (averaged over
        ``time``) for each selected iteration.
    newplace_df : pandas.DataFrame
        New placement locations. For panel ``i`` (0-indexed among
        ``[0, 4, 9]``), only ``newplace_df.iloc[:idx + 1]`` is shown, i.e.
        placements accumulate across panels.
    existing_df : pandas.DataFrame
        Existing placement locations, shown on every panel.
    cmap : str, default "Grays"
        Colormap for the acquisition function panels.
    title : str or None, default None
        Currently unused (accepted for interface compatibility).

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes
        The first (leftmost) panel's axis.
    """
    # Modified from: https://github.com/eschlager/MeltEmulation/
    fig = plt.figure(figsize=(9, 3))
    axs: list[GeoAxes | None] = [None] * 3
    display_crs = ccrs.NorthPolarStereo(central_longitude=-40)

    # gap between each subplot
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 0.1, 1, 0.1, 1], wspace=0)
    axs[0] = fig.add_subplot(gs[0, 0], projection=display_crs)
    axs[1] = fig.add_subplot(gs[0, 2], projection=display_crs)
    axs[2] = fig.add_subplot(gs[0, 4], projection=display_crs)

    value_lim = None

    cbar_label = "Acquisition function value"

    iterations = [0, 4, 9]
    data_by_idx = {idx: ds.isel(iteration=idx).mean(dim="time") for idx in iterations}

    # shared value limits across all subplots
    min_x = np.nanmin([np.nanmin(xx.values.flatten()) for xx in data_by_idx.values()])
    max_x = np.nanmax([np.nanmax(xx.values.flatten()) for xx in data_by_idx.values()])
    minval_x, maxval_x, extend_colorbar_x = _get_value_limits(min_x, max_x, value_lim)

    # plot true
    for i, idx in enumerate(iterations):
        xx = data_by_idx[idx]

        axs[i], p0 = plot_greenland_only(
            xx,
            newplace_df.iloc[: idx + 1],
            existing_df,
            ax=axs[i],
            pcolormesh_kwargs={"cmap": cmap, "vmin": minval_x, "vmax": maxval_x},
        )
        cast(GeoAxes, axs[i]).set_title(f"Iteration {idx + 1}")
        cast(GeoAxes, axs[i]).set_axis_off()

    # single shared colorbar spanning all subplots
    extra_gap = (
        0.02  # horizontal gap (figure fraction) between axes right edge and colorbar
    )
    cbar_width = 0.03  # thickness of the colorbar (figure fraction)
    bbox_last = cast(GeoAxes, axs[-1]).get_position()

    x0 = bbox_last.x1 + extra_gap
    y0 = min(cast(GeoAxes, ax).get_position().y0 for ax in axs)
    y1 = max(cast(GeoAxes, ax).get_position().y1 for ax in axs)
    height = y1 - y0
    cax = fig.add_axes((x0, y0, cbar_width, height))
    cbar = fig.colorbar(p0, cax=cax, orientation="vertical", extend=extend_colorbar_x)
    cbar.set_label(cbar_label)

    # plt.suptitle(title)
    return cast(GeoAxes, axs[0])
