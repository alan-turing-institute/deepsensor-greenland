"""Save a static copy of the data dashboard's API for offline use.

``uv run python -m giant_sp.demo.snapshot`` writes, under
``demo/dashboard/static-api/``, the same payloads the live ``/dashboard/api``
serves: ``repo.json``, ``catalog.json``, ``series/<experiment>/<variable>.json``
for every variable, and map PNGs for every (or chosen) variable at a regular date
interval (``maps/<experiment>/<variable>/<YYYY-MM-DD>.png``, plus
``<YYYY-MM-DD>-full.png`` coloured by the full data range and
``<YYYY-MM-DD>-date.png`` coloured by that date's own range, listed in
``maps/index.json``). It also saves a copy of Plotly, so the dashboard works
without internet access. When the live API is unreachable, the dashboard
falls back to these files, and ``demo/`` can then be served by any static
file server.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from giant_sp.demo.dashboard import Dashboard, MapScale
from giant_sp.demo.server import DEFAULT_SITE_DIR
from giant_sp.demo.source import ArraylakeSource

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = DEFAULT_SITE_DIR / "dashboard" / "static-api"

# Each map's file name suffix per colour scale; must match `api.mapUrl` in
# demo/dashboard/app.js.
MAP_SCALE_SUFFIXES: dict[MapScale, str] = {
    "shared": "",
    "full": "-full",
    "date": "-date",
}

# Must match the Plotly <script> in demo/dashboard/index.html.
PLOTLY_URL = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def map_dates(times: Sequence[str | int], freq: str) -> list[str]:
    """Pick the dates to pre-render maps for from a series' time axis.

    Parameters
    ----------
    times : Sequence[str | int]
        A series' ``time`` values (ISO strings, or sample indices when the
        time coordinate couldn't be decoded).
    freq : str
        Pandas offset alias, e.g. ``"MS"`` for the first of each month.

    Returns
    -------
    list[str]
        ``YYYY-MM-DD`` dates within the series' range, or ``[]`` if its time
        axis isn't made of dates.
    """
    if not times or not isinstance(times[0], str):
        return []
    start = pd.Timestamp(str(times[0])[:10])
    end = pd.Timestamp(str(times[-1])[:10])
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq=freq)]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_snapshot(
    dashboard: Dashboard,
    out_dir: Path,
    map_variables: Sequence[str] | None = None,
    map_freq: str = "MS",
    plotly_url: str | None = PLOTLY_URL,
) -> int:
    """Write a static copy of ``dashboard``'s API to ``out_dir``.

    Parameters
    ----------
    dashboard : Dashboard
        Source of the payloads.
    out_dir : Path
        Destination directory. A previous snapshot there is replaced; any
        other non-empty directory is refused.
    map_variables : Sequence[str] | None
        Variables to pre-render maps for (where an experiment has them), or
        None for every 2-D variable in the catalog.
    map_freq : str
        Pandas offset alias for the map dates; see ``map_dates``.
    plotly_url : str | None
        Where to download Plotly from, or None to skip saving it.

    Returns
    -------
    int
        Number of files written.

    Raises
    ------
    FileExistsError
        If ``out_dir`` is a non-empty directory that isn't a snapshot.
    """
    if out_dir.exists() and any(out_dir.iterdir()):
        if not (out_dir / "repo.json").exists():
            raise FileExistsError(f"{out_dir} is not empty and isn't a snapshot")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    repo = dashboard.repo_info()
    repo["saved_at"] = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    _write_json(out_dir / "repo.json", repo)
    catalog = dashboard.catalog()
    _write_json(out_dir / "catalog.json", catalog)
    n_files = 2

    maps_index: dict[str, dict[str, list[str]]] = {}
    for exp in catalog["experiments"]:
        name = exp["name"]
        for var in exp["variables"]:
            series = dashboard.series(name, var)
            _write_json(out_dir / "series" / name / f"{var}.json", series)
            n_files += 1
            wanted = map_variables is None or var in map_variables
            if not wanted or not series["spatial_mean"]:
                continue
            dates = map_dates(series["time"], map_freq)
            for date in dates:
                logger.info("map %s/%s %s", name, var, date)
                for scale, suffix in MAP_SCALE_SUFFIXES.items():
                    png = dashboard.map_png(name, var, date, scale)
                    path = out_dir / "maps" / name / var / f"{date}{suffix}.png"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(png)
            n_files += len(MAP_SCALE_SUFFIXES) * len(dates)
            if dates:
                maps_index.setdefault(name, {})[var] = dates
    _write_json(out_dir / "maps" / "index.json", maps_index)
    n_files += 1

    if plotly_url:
        with urllib.request.urlopen(plotly_url, timeout=60) as r:
            (out_dir / "plotly.min.js").write_bytes(r.read())
        n_files += 1
    return n_files


def main() -> None:
    """Parse CLI args and write a snapshot from Arraylake."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--map-variables",
        nargs="*",
        default=None,
        help="variable keys to pre-render maps for (default: every mappable variable)",
    )
    parser.add_argument(
        "--map-freq",
        default="MS",
        help="pandas offset alias for map dates (default: %(default)s, "
        "the first of each month)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
    n_files = write_snapshot(
        Dashboard(ArraylakeSource.from_env()),
        args.out_dir,
        map_variables=args.map_variables,
        map_freq=args.map_freq,
    )
    logger.info("wrote %d files to %s", n_files, args.out_dir)


if __name__ == "__main__":
    main()
