"""Generate the Pareto placement figures shown on the demo site's Results tab.

Every result set under ``pareto/`` is discovered from its Pareto mask file,
named ``mask/<run>_<acq_fn>_<cost>.npy``, alongside:

- ``acq/<run>_<acq_fn>.npy``: acquisition-function value per grid cell
- ``cost/<cost>.npy``: cost per grid cell
- ``rank/<run>_<acq_fn>_<cost>.npy``: Pareto rank per candidate site
- ``placements/<run>_<acq_fn>_<cost>.csv``: lat/lon of the Pareto-optimal sites
- ``existing_stations.csv``: lat/lon of existing stations (shared by all results)

One figure per result set is written to ``demo/images/pareto/<run>_<acq_fn>_<cost>.png``,
plus a ``manifest.json`` there listing them, which ``demo/results.js`` reads
to populate the Results tab. Run from anywhere with
``uv run python demo/results/placement_pareto_demo.py``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cartopy.crs as ccrs
import colorcet as cc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

import giant_sp.utils.plots as myplt

RESULTS_DIR = Path(__file__).resolve().parent / "pareto"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "images" / "pareto"

LABEL_FONTSIZE = 20
# Fixed layout, in inches, shared by every figure so they line up when switched
# between. Rectangles are (left, bottom, width, height) from the figure's
# bottom-left corner. The figure's width isn't fixed: it's trimmed to fit the
# colorbar label (see _fit_figure_width).
FIG_HEIGHT = 8.0
# The left margin is sized for the widest tick labels seen (MeanVariance's,
# e.g. "-0.12026"), so that their y-label sits as far from the ticks as the
# x-label does from its ticks; narrower tick labels leave a bigger gap.
SCATTER_RECT = (1.8, 0.8, 6.84, 6.96)
# Gap between the scatter plot and the map.
PLOT_GAP = 0.48
# The map (and its colorbar) leaves the same gap above and below it as there
# is above the scatter plot. Its width only needs to fit the map's aspect
# ratio, since the map is drawn against its left edge.
_TOP_GAP = FIG_HEIGHT - (SCATTER_RECT[1] + SCATTER_RECT[3])
MAP_RECT = (
    SCATTER_RECT[0] + SCATTER_RECT[2] + PLOT_GAP,
    _TOP_GAP,
    5.12,
    FIG_HEIGHT - 2 * _TOP_GAP,
)
COLORBAR_PAD = 0.16  # gap between the map and its colorbar
COLORBAR_WIDTH = 0.32
DISPLAY_CRS = ccrs.NorthPolarStereo(central_longitude=-40)
# The demo site's accent blue (--accent in demo/styles.css); stands out
# against the warm surface-melt colormap.
PARETO_COLOR = "#4da3ff"

# Display names for the Results tab's "Cost" picker, keyed on cost file stem;
# other costs fall back to the stem with underscores as spaces.
COST_LABELS = {
    "distance2stations": "Distance to stations",
    "distance2border": "Distance to border",
    "elevation": "Elevation",
    "albedo": "Albedo",
}


@dataclass(frozen=True)
class ParetoResult:
    """One (run, acquisition function, cost) result set under ``RESULTS_DIR``."""

    run: str
    acq_fn: str
    cost: str

    @property
    def stem(self) -> str:
        """File stem shared by this result's mask, rank, placements and figure."""
        return f"{self.run}_{self.acq_fn}_{self.cost}"

    @property
    def acq_fn_label(self) -> str:
        """Name shown in the Results tab's "Informativeness" picker."""
        acq_words = re.sub(r"(?<!^)(?=[A-Z])", " ", self.acq_fn)
        return acq_words.capitalize()

    @property
    def cost_label(self) -> str:
        """Name shown in the Results tab's "Cost" picker."""
        return COST_LABELS.get(self.cost, self.cost.replace("_", " ").capitalize())


def discover_results(results_dir: Path) -> list[ParetoResult]:
    """Find every complete result set in ``results_dir``, sorted by name."""
    costs = sorted((p.stem for p in (results_dir / "cost").glob("*.npy")), key=len)
    found = []
    for mask_file in sorted((results_dir / "mask").glob("*.npy")):
        # Run ids have no underscores; the cost name is the known cost file
        # the stem ends with (longest match, in case one contains another).
        run, _, rest = mask_file.stem.partition("_")
        cost = next((c for c in reversed(costs) if rest.endswith(f"_{c}")), None)
        if not run or cost is None:
            print(f"Skipping {mask_file.name}: no matching cost/*.npy")
            continue
        result = ParetoResult(run, rest[: -len(cost) - 1], cost)
        missing = [
            p
            for p in (
                results_dir / "acq" / f"{run}_{result.acq_fn}.npy",
                results_dir / "rank" / f"{result.stem}.npy",
                results_dir / "placements" / f"{result.stem}.csv",
            )
            if not p.exists()
        ]
        if missing:
            print(
                f"Skipping {result.stem}: missing {', '.join(p.name for p in missing)}"
            )
            continue
        found.append(result)
    return found


def _power_of_ten(values: np.ndarray) -> int:
    """Power of ten to scale ``values`` by so their ticks read e.g. 0.5 to 1.5.

    Returns 0 (no scaling) when the largest magnitude is between 1e-3 and 1e4,
    where plain tick labels are already readable; otherwise the exponent of
    the largest magnitude. Plotting ``values / 10**p`` and putting the scale in
    the axis label replaces matplotlib's multiplier text (e.g. "1e-5") at the
    end of the axis.
    """
    largest = np.nanmax(np.abs(values))
    if not np.isfinite(largest) or largest == 0:
        return 0
    power = int(np.floor(np.log10(largest)))
    return power if power <= -3 or power >= 4 else 0


def _xlabel_margins(fig: Figure, ax: Axes) -> tuple[float, float]:
    """Return ``ax``'s x-label gaps, in pixels, as ``(to_edge, to_ticks)``.

    ``to_edge`` is the gap from the figure's bottom edge to the x-label;
    ``to_ticks`` is the gap from the x-label to the tick labels above it. The
    other figure labels are placed to match. Must be called after
    ``fig.canvas.draw()``, since it measures the drawn labels.
    """
    renderer = fig.canvas.get_renderer()
    label = ax.xaxis.label.get_window_extent(renderer)
    ticks_y0 = min(
        t.get_window_extent(renderer).y0 for t in ax.get_xticklabels() if t.get_text()
    )
    return label.y0, ticks_y0 - label.y1


def _match_ylabel_margin(fig: Figure, ax: Axes, to_edge: float) -> None:
    """Move ``ax``'s y-label so it is ``to_edge`` pixels from the left edge."""
    left = ax.yaxis.label.get_window_extent(fig.canvas.get_renderer()).x0
    x, y = ax.yaxis.label.get_position()  # axes coords, from set_label_coords
    ax.yaxis.set_label_coords(x + (to_edge - left) / ax.bbox.width, y)


Rect = tuple[float, float, float, float]


def _to_fraction(fig: Figure, rect: Rect) -> Rect:
    """Convert a ``(left, bottom, width, height)`` rect in inches to fractions."""
    width, height = fig.get_size_inches()
    left, bottom, w, h = rect
    return (left / width, bottom / height, w / width, h / height)


def _fit_figure_width(
    fig: Figure,
    cbar_ax: Axes,
    rects: dict[Axes, Rect],
    to_edge: float,
    to_ticks: float,
) -> None:
    """Fit the colorbar label's margins by placing it and trimming the figure.

    Places the colorbar label ``to_ticks`` pixels right of its tick labels,
    then sets the figure's width so the label ends ``to_edge`` pixels from
    its right edge, mirroring the x-label's margins. ``rects`` gives each
    axes' rectangle in inches, re-applied after the resize so nothing else
    moves. Draws the figure twice to measure the labels.
    """
    renderer = fig.canvas.get_renderer()
    label = cbar_ax.yaxis.label

    cbar_ax.yaxis.set_label_coords(1.0, 0.5)  # axes coords, so it moves with cbar_ax
    fig.canvas.draw()
    ticks_x1 = max(
        t.get_window_extent(renderer).x1
        for t in cbar_ax.get_yticklabels()
        if t.get_text()
    )
    left = label.get_window_extent(renderer).x0
    cbar_ax.yaxis.set_label_coords(
        1.0 + (ticks_x1 + to_ticks - left) / cbar_ax.bbox.width, 0.5
    )

    fig.canvas.draw()
    excess = fig.bbox.width - to_edge - label.get_window_extent(renderer).x1
    width, height = fig.get_size_inches()
    fig.set_size_inches(width - excess / fig.dpi, height)
    for ax, rect in rects.items():
        ax.set_position(_to_fraction(fig, rect))


def plot_result(
    result: ParetoResult, existing_df: pd.DataFrame, out_file: Path
) -> None:
    """Draw the cost/informativeness scatter and Pareto-rank map for ``result``."""
    acqfn = np.load(RESULTS_DIR / "acq" / f"{result.run}_{result.acq_fn}.npy")
    cost = np.load(RESULTS_DIR / "cost" / f"{result.cost}.npy")
    mask = np.load(RESULTS_DIR / "mask" / f"{result.stem}.npy")
    # rank_da = xr.open_dataarray(RESULTS_DIR / "background" / f"{result.stem}.nc")
    rank_da = xr.open_dataarray(
        RESULTS_DIR / "background" / "hirham5_mean_temporalsampling_50dates_2016.nc"
    )
    newplace_df = pd.read_csv(
        RESULTS_DIR / "placements" / f"{result.stem}.csv", index_col=[0]
    )

    cmap = mcolors.ListedColormap(cc.linear_kryw_5_100_c67[::-1])

    # acq/cost are grids; the mask indexes their non-NaN (land) cells in
    # row-major order.
    alpha_df = pd.DataFrame({"alpha_1": acqfn.ravel(), "alpha_2": cost.ravel()}).dropna(
        axis=0
    )

    # Informativeness is the negated acquisition value, scaled so its ticks are
    # readable; the scale goes in the axis label.
    informativeness = alpha_df["alpha_1"].to_numpy() * -1
    y_power = _power_of_ten(informativeness)
    y_scale = 10.0**-y_power
    y_label = "Informativeness"
    if y_power:
        y_label += rf" ($\times10^{{{y_power}}}$)"

    # Fixed axes positions (not tight_layout) so every figure lines up the
    # same, whatever the width of its tick labels.
    # Start wide enough for everything; _fit_figure_width trims it at the end.
    fig = plt.figure(figsize=(20, FIG_HEIGHT), frameon=False)

    ax0 = fig.add_axes(_to_fraction(fig, SCATTER_RECT))
    ax0.scatter(
        alpha_df["alpha_2"],
        informativeness * y_scale,
        marker="o",
        color="k",
        s=6**2,
        linewidths=0,
        label="All solutions",
        alpha=0.25,
    )
    y_pareto = alpha_df.reset_index()[["alpha_1", "alpha_2"]].values[mask]
    ax0.scatter(
        y_pareto[:, 1],
        y_pareto[:, 0] * -1 * y_scale,
        marker="o",
        color=PARETO_COLOR,
        edgecolors="white",
        s=8**2,
        linewidths=0.8,
        label="Pareto optimal",
        zorder=3,
    )
    ax0.legend(loc="lower right", prop={"size": 15})
    # Generic labels: which cost/acquisition function is shown is picked on the
    # Results tab.
    ax0.set_xlabel("Cost", size=LABEL_FONTSIZE)
    ax0.set_ylabel(y_label, size=LABEL_FONTSIZE)
    # Pin the axis labels rather than letting them sit next to the tick
    # labels, whose width varies between figures. The y-label is moved again
    # by _match_label_margins once the figure is drawn.
    ax0.xaxis.set_label_coords(0.5, -0.065)
    ax0.yaxis.set_label_coords(-0.235, 0.5)
    ax0.tick_params(axis="both", which="major", labelsize=18)
    # At most ~5 intervals, so long cost labels (e.g. "0.025") don't crowd.
    ax0.xaxis.set_major_locator(MaxNLocator(nbins=5))
    for spine in ax0.spines.values():
        spine.set_color("gray")

    ax1 = fig.add_axes(_to_fraction(fig, MAP_RECT), projection=DISPLAY_CRS)
    # The map keeps its aspect ratio inside MAP_RECT; hug the left edge so it
    # sits PLOT_GAP from the scatter plot.
    ax1.set_anchor("W")
    new_style = {
        "marker": "*",
        "c": PARETO_COLOR,
        "s": 22**2,
        "edgecolors": "white",
        "linewidths": 0.8,
        "zorder": 6,
    }
    ax1, pcm = myplt.plot_greenland_only(
        rank_da,
        newplace_df,
        existing_df,
        ax=ax1,
        pcolormesh_kwargs={"cmap": cmap},
        newplace_kwargs=new_style,
    )
    # Enlarge the existing-station markers only; new placements keep new_style.
    for coll in ax1.collections:
        if isinstance(coll, PathCollection) and coll.get_label() != "New placements":
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
    new_handle = Line2D(
        [],
        [],
        marker="*",
        color=PARETO_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="None",
        markersize=22,
        label=f"Proposed sites ({len(newplace_df)})",
    )
    ax1.legend(
        handles=[existing_handle, new_handle],
        loc="lower right",
        frameon=True,
        prop={"size": 15},
    )
    ax1.spines["geo"].set_color("gray")

    fig.canvas.draw()
    to_edge, to_ticks = _xlabel_margins(fig, ax0)
    _match_ylabel_margin(fig, ax0, to_edge)
    # The colorbar sits just right of the map as drawn (narrower than
    # MAP_RECT, since the map keeps its aspect ratio), and matches its height.
    fig_width, fig_height = fig.get_size_inches()
    pos1 = ax1.get_position()
    cbar_rect = (
        pos1.x1 * fig_width + COLORBAR_PAD,
        pos1.y0 * fig_height,
        COLORBAR_WIDTH,
        pos1.height * fig_height,
    )
    cbar_ax = fig.add_axes(_to_fraction(fig, cbar_rect))
    cb = plt.colorbar(
        pcm,
        cax=cbar_ax,
        fraction=0.0,
        pad=0.0,
        ticks=np.arange(0, float(rank_da.max()) + 1, 5),
    )
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=16)
    cb.set_label("Surface melt (mm w.e. per day)", size=LABEL_FONTSIZE)
    _fit_figure_width(
        fig,
        cbar_ax,
        {ax0: SCATTER_RECT, ax1: MAP_RECT, cbar_ax: cbar_rect},
        to_edge,
        to_ticks,
    )

    # No bbox_inches="tight": cropping to the content would shift each figure
    # differently and undo the fixed layout.
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    rank_da.close()


def main() -> None:
    """Plot every result set under ``RESULTS_DIR`` and write the manifest."""
    results = discover_results(RESULTS_DIR)
    if not results:
        raise SystemExit(f"No complete result sets found under {RESULTS_DIR}")
    existing_df = pd.read_csv(RESULTS_DIR / "existing_stations.csv", index_col=[0])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for result in results:
        out_file = OUTPUT_DIR / f"{result.stem}.png"
        print(f"Plotting {result.stem} -> {out_file}")
        plot_result(result, existing_df, out_file)
        manifest.append(
            {
                "file": f"images/pareto/{out_file.name}",
                "run": result.run,
                "acq_fn": result.acq_fn,
                "acq_fn_label": result.acq_fn_label,
                "cost": result.cost,
                "cost_label": result.cost_label,
            }
        )
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
