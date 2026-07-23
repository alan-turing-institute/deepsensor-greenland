import cartopy.crs as ccrs
import cartopy.feature as cf
import deepsensor.plot as dsplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr

from giant_sp.utils.metrics import compute_pit_bernoulli_gamma


def averaged_prediction_plot(
    truth_da,
    mean_da,
    std_da,
    cmap="seismic",
    crs=None,
    highlight_negatives=False,
    task=None,
    dataproc=None,
    plot_xc=False,
):
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
