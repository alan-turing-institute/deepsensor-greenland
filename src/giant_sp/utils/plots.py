import logging

import numpy as np
import pandas as pd
import scipy as sp
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.feature import ShapelyFeature
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

import deepsensor.plot as dsplt

from giant_sp.utils.data import find_lon_lat, normalize_lons, load_greenland_shape, points_in_polygons
import matplotlib.gridspec as gridspec
from giant_sp.utils.metrics import compute_pit_bernoulli_gamma

def averaged_prediction_plot(truth_da, mean_da, std_da, cmap='seismic', crs=None, highlight_negatives=False, task=None, dataproc=None, plot_xc=False):
    mean_da, truth_da = xr.align(mean_da, truth_da, join='inner')
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
    img_truth = truth_da.mean(dim="time").plot(
        ax=axes[0],
        cmap=cmap,
        vmin=var_clim[0],
        vmax=var_clim[1],
        cbar_kwargs=dict(fraction=0.046 * im_ratio, pad=0.04, label=None),
    )
    axes[0].set_title(f"{truth_da.name}")

    # Plot predicted mean
    if not highlight_negatives:
        mean_da.mean(dim="time").plot(
            ax=axes[1],
            cmap=cmap,
            vmin=var_clim[0],
            vmax=var_clim[1],
            cbar_kwargs=dict(fraction=0.046 * im_ratio, pad=0.04, label=None),
        )
    else:
        cmap = plt.get_cmap("viridis").copy()
        # cmap.set_under('orange')
        norm = plt.Normalize(img_truth.colorbar.vmin, img_truth.colorbar.vmax)
        mean_da.mean(dim="time").plot(
            ax=axes[1],
            vmin=img_truth.colorbar.vmin,
            vmax=img_truth.colorbar.vmax,
            cmap=cmap,
            norm=norm,
            cbar_kwargs=dict(fraction=0.046 * im_ratio, pad=0.04, label=None),
        )
    axes[1].set_title("ConvNP mean")

    # Plot predicted uncertainty
    std_da.mean(dim="time").plot(
        ax=axes[2],
        #    vmin=0.,
        cmap="Greys",
        cbar_kwargs=dict(fraction=0.046 * im_ratio, pad=0.04, label=None),
    )
    axes[2].set_title("ConvNP uncertainty (std)")

    # Plot prediction error
    err_da.mean(dim="time").plot(
        ax=axes[3],
        cmap="seismic",  # "RdBu_r"
        cbar_kwargs=dict(fraction=0.046 * im_ratio, pad=0.04, label=None),
    )
    axes[3].set_title("ConvNP error")

    # Plot context locations on ground truth plot
    if plot_xc == True:
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


def PIT_histogram(truth, val_pred, likelihood, **kwargs):

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


def hexbin_parity_plot(x, y):
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
        linewidths=(0,),
        mincnt=1,
        vmin=1,
    )
    cb = fig.colorbar(hb, ax=ax, label="counts")
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
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    return fig, ax


def new_placements_plot_xy(acqfn_da, newplace_df, Xc_df, cmap="Greys", title=None):
    fig, ax = plt.subplots(layout="constrained")

    plot_da = acqfn_da.sel(iteration=0).mean(dim="time")
    vmax = float(plot_da.max().compute())
    vmin = float(plot_da.min().compute())
    plot_da.plot(
        ax=ax,
        cmap=cmap,
        vmax=vmax,
        vmin=vmin,
        add_colorbar=True,
        center=False,
        cbar_kwargs={"label": "Acquisition function value"},
    )
    ax.set_title(title, pad=10)

    ax.scatter(
        *newplace_df.values.T[::-1], marker="+", s=20**2, c="g", label="New placements"
    )
    ax.scatter(
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


def plot_greenland_only(gris_data, newplace_df, existing_df, ax=None, masked=True, ocean_color='white', gridlines=False, pcolormesh_kwargs=None):
    # Modified from: https://github.com/eschlager/MeltEmulation/
    ds = gris_data

    if ds.ndim != 2:
        raise ValueError("Expected a 2D field (lat,lon or y,x).")

    lon, lat = find_lon_lat(ds)

    # If lon/lat are 1D, make meshgrid
    if lon.ndim == 1 and lat.ndim == 1:
        Lon, Lat = np.meshgrid(lon, lat)
    elif lon.shape == ds.shape and lat.shape == ds.shape:
        Lon, Lat = lon, np.array(lat)
    else:
        raise ValueError(f"lon/lat shapes {lon.shape},{lat.shape} do not match data shape {ds.shape}")

    # Normalize longitudes to -180..180 to match Natural Earth polygons
    Lon = normalize_lons(Lon)
    # load Greenland shapely geometry
    greenland_geom = load_greenland_shape(resolution='50m')

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
        fig = plt.figure(figsize=(8,10), frameon=False)
        ax = fig.add_subplot(1,1,1, projection=display_crs)
        plt.axis('off')

    # Make ocean white if not None, otherwise default is transparent!
    if ocean_color is not None:
        ax.add_feature(cf.OCEAN.with_scale('50m'), facecolor=ocean_color, zorder=0)
    # Fill Greenland with light gray (use ShapelyFeature for correct reprojection)
    greenland_feature = ShapelyFeature([greenland_geom], crs=ccrs.PlateCarree(), facecolor='lightgray', edgecolor='black', linewidth=.5, zorder=3)
    ax.add_feature(greenland_feature)

    # Set a Greenland-focused extent in lon/lat
    ax.set_extent((-55, -25, 58, 85), crs=ccrs.PlateCarree())

    # Latitude gridlines visible
    if gridlines:
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='gray', alpha=0.7, linestyle='--')

    ax.scatter(*newplace_df.values.T[::-1], transform=data_crs, marker='+', s=3 ** 2,  c='r', zorder=5, label='New placements')
    ax.scatter(*existing_df.values.T[::-1], transform=data_crs, marker='x', s=3 ** 2, c='k', zorder=5, label='Existing placements')

    # annotate new placements with their row index
    for row_idx, (lat_val, lon_val) in zip(newplace_df.index, newplace_df.values):
        ax.text(lon_val, lat_val, str(row_idx+1), transform=data_crs, color='r', fontsize=6, fontweight='bold', zorder=6, ha='left', va='bottom')

    pcm = ax.pcolormesh(Lon, Lat, data, transform=data_crs, shading='auto', rasterized=True, zorder=4, **pcolormesh_kwargs)

    return ax, pcm


def _get_value_limits(minval, maxval, value_lim):
    # Source: https://github.com/eschlager/MeltEmulation/
    """
    Determine value limits and colorbar extension based on provided value limits.
    """
    if value_lim is not None:
        value_min, value_max = value_lim
        if value_max is None:
            value_max = maxval
        if value_min is None:
            value_min = minval

        if maxval > value_max:
            if minval < value_min:
                extend_colorbar = 'both'
            else:
                extend_colorbar = 'max'
        elif minval < value_min:
            extend_colorbar = 'min'
        else:
            extend_colorbar = 'neither'
        maxval = value_max
        minval = value_min
    else:
        extend_colorbar = 'neither'
        
    return minval, maxval, extend_colorbar

def new_placements_plot_xy_hirham5native(ds, newplace_df, existing_df, cmap='Grays', title=None):
    # Modified from: https://github.com/eschlager/MeltEmulation/
    """
    plot acquisition maps
    """
    fig = plt.figure(figsize=(9,3))            
    axs = [None]*3
    display_crs = ccrs.NorthPolarStereo(central_longitude=-40)
    
    # gap between each subplot
    gs = gridspec.GridSpec(1, 5, width_ratios=[1, 0.1, 1, 0.1, 1], wspace=0)
    axs[0] = fig.add_subplot(gs[0, 0], projection=display_crs)
    axs[1] = fig.add_subplot(gs[0, 2], projection=display_crs)
    axs[2] = fig.add_subplot(gs[0, 4], projection=display_crs)

    value_lim = None

    cbar_label = 'Acquisition function value'

    iterations = [0, 4, 9]
    data_by_idx = {idx: ds.isel(iteration=idx).mean(dim='time') for idx in iterations}

    # shared value limits across all subplots
    min_x = np.nanmin([np.nanmin(xx.values.flatten()) for xx in data_by_idx.values()])
    max_x = np.nanmax([np.nanmax(xx.values.flatten()) for xx in data_by_idx.values()])
    minval_x, maxval_x, extend_colorbar_x = _get_value_limits(min_x, max_x, value_lim)

    # plot true
    for i, idx in enumerate(iterations):
        xx = data_by_idx[idx]

        axs[i], p0 = plot_greenland_only(xx, newplace_df.iloc[:idx+1], existing_df, ax=axs[i], pcolormesh_kwargs={'cmap': cmap, 'vmin': minval_x, 'vmax': maxval_x})
        axs[i].set_title(f'Iteration {idx+1}')
        axs[i].set_axis_off()

    # single shared colorbar spanning all subplots
    extra_gap = 0.02    # horizontal gap (figure fraction) between axes right edge and colorbar
    cbar_width = 0.03   # thickness of the colorbar (figure fraction)
    bbox_last = axs[-1].get_position()

    x0 = bbox_last.x1 + extra_gap
    y0 = min(ax.get_position().y0 for ax in axs)
    y1 = max(ax.get_position().y1 for ax in axs)
    height = y1 - y0
    cax = fig.add_axes([x0, y0, cbar_width, height])
    cbar = fig.colorbar(p0, cax=cax, orientation='vertical', extend=extend_colorbar_x)
    cbar.set_label(cbar_label)

    # plt.suptitle(title)
    return axs[0]
