"""Generate the Pareto placement figures shown on the demo site's Results tab.

Every result set under ``pareto/`` is discovered from its Pareto mask file,
named ``mask/<run>_<acq_fn>_<cost>.npy``, alongside:

- ``acq/<run>_<acq_fn>.npy``: acquisition-function value per grid cell
- ``cost/<cost>.npy``: cost per grid cell
- ``rank/<run>_<acq_fn>_<cost>.nc``: Pareto rank map
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

import giant_sp.utils.plots as myplt

RESULTS_DIR = Path(__file__).resolve().parent / "pareto"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "images" / "pareto"

LABEL_FONTSIZE = 20
DISPLAY_CRS = ccrs.NorthPolarStereo(central_longitude=-40)


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
    def label(self) -> str:
        """Human-readable name shown in the Results tab's picker."""
        acq_words = re.sub(r"(?<!^)(?=[A-Z])", " ", self.acq_fn)
        return f"{acq_words} vs {self.cost.replace('_', ' ')} (run {self.run})"


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
                results_dir / "rank" / f"{result.stem}.nc",
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


def plot_result(
    result: ParetoResult, existing_df: pd.DataFrame, out_file: Path
) -> None:
    """Draw the cost/informativeness scatter and Pareto-rank map for ``result``."""
    acqfn = np.load(RESULTS_DIR / "acq" / f"{result.run}_{result.acq_fn}.npy")
    cost = np.load(RESULTS_DIR / "cost" / f"{result.cost}.npy")
    mask = np.load(RESULTS_DIR / "mask" / f"{result.stem}.npy")
    rank_da = xr.open_dataarray(RESULTS_DIR / "rank" / f"{result.stem}.nc")
    newplace_df = pd.read_csv(
        RESULTS_DIR / "placements" / f"{result.stem}.csv", index_col=[0]
    )

    alpha_df = pd.DataFrame({"alpha_1": acqfn, "alpha_2": cost}).dropna(axis=0)

    fig = plt.figure(figsize=(16, 8), frameon=False)
    gs = fig.add_gridspec(1, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.scatter(
        cost,
        acqfn * -1,
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
    ax0.set_xlabel("Cost (Remoteness)", size=LABEL_FONTSIZE)
    ax0.set_ylabel(
        "Informativeness\n(ConvCNP Acquisition Function)", size=LABEL_FONTSIZE
    )
    ax0.tick_params(axis="both", which="major", labelsize=18)
    for spine in ax0.spines.values():
        spine.set_color("gray")

    ax1 = fig.add_subplot(gs[0, 1], projection=DISPLAY_CRS)
    ax1, pcm = myplt.plot_greenland_only(
        rank_da, newplace_df, existing_df, ax=ax1, pcolormesh_kwargs={"cmap": "Greys"}
    )
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
    fig.canvas.draw()
    pos1 = ax1.get_position()
    cbar_ax = fig.add_axes((pos1.x1 + 0.01, pos1.y0, 0.03, pos1.height))
    cb = plt.colorbar(
        pcm,
        cax=cbar_ax,
        fraction=0.0,
        pad=0.0,
        ticks=np.arange(0, rank_da.max() + 1, 5),
    )
    cb.ax.minorticks_off()
    cb.ax.tick_params(labelsize=16)
    cb.set_label("Pareto rank", size=LABEL_FONTSIZE)

    fig.savefig(out_file, dpi=200, bbox_inches="tight")
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
                "label": result.label,
                "run": result.run,
                "acq_fn": result.acq_fn,
                "cost": result.cost,
            }
        )
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
