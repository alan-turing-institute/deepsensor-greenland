"""Catalog discovery, time-series reads and map rendering for the data dashboard.

The dataset layout is discovered rather than hardcoded:

- Any group (up to one level of nesting below the root) that directly holds a
  recognized time coordinate is an "experiment"; static groups such as
  land masks or zone ids have no time coordinate and are skipped, as is
  anything in ``EXCLUDED_EXPERIMENTS``.
- Every 1-D or 2-D array in an experiment is a plottable variable, except the
  time coordinate and any array named in another array's CF ``coordinates``
  attribute (e.g. ``lat``/``lon``) and anything in ``EXCLUDED_VARIABLES``.
- A 2-D ``(time, points)`` variable is reduced to a domain-mean time series.

Results are cached per snapshot: each request makes one cheap metastore call
for the tip-of-branch snapshot id and only re-reads data if it has moved.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal, cast

import cartopy.crs as ccrs
import cftime
import colorcet as cc
import numpy as np
import zarr
from matplotlib.colors import Colormap, ListedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from numpy.typing import NDArray

from giant_sp.demo.source import DataSource, Snapshot

logger = logging.getLogger(__name__)

TIME_NAMES = ("time", "time_centered", "valid_time", "time_counter")

# Variables present in the data but deliberately hidden from the dashboard
# (runoff and albedo).
EXCLUDED_VARIABLES = frozenset({"rogl", "albedom"})

# Experiments present in the data but deliberately hidden from the dashboard
# (the 10-year rolling-mean baseline).
EXCLUDED_EXPERIMENTS = frozenset({"baseline_10year"})

# NSIDC Sea Ice Polar Stereographic North (EPSG:3413), defined by its parameters
# rather than ccrs.epsg(3413), which clips to the CRS's registered area of use.
MAP_PROJECTION = ccrs.Stereographic(
    central_latitude=90, central_longitude=-45, true_scale_latitude=70
)

# Map colormap per variable, keyed on lowercased ``long_name``; anything not
# listed uses ``DEFAULT_MAP_CMAP``. Surface melt matches the Results tab's
# figure (``demo/results/placement_pareto_demo.py``).
DEFAULT_MAP_CMAP: str | Colormap = "viridis"
MAP_CMAPS: dict[str, str | Colormap] = {
    "acc snow melt": ListedColormap(cc.linear_kryw_5_100_c67[::-1]),
}

# Percentiles (over every date and point) bounding each variable's map colour
# scale, so all its dates share one scale.
MAP_PERCENTILES = (1, 99)

# How a map's colour scale is chosen (the dashboard sidebar's "Colour scale"):
# ``"shared"`` by ``MAP_PERCENTILES`` over every date, ``"full"`` by the
# data's minimum to maximum over every date, or ``"date"`` by the mapped date's
# own minimum to maximum, so each date gets its own (always linear, even for
# ``LOG_SCALE_VARIABLES``).
MapScale = Literal["shared", "full", "date"]

# Heavily skewed variables (mostly zero/near-zero with rare large events), keyed
# on lowercased ``long_name``, mapped on a log colour scale instead: it tops out
# at the ``MAP_LOG_UPPER_PERCENTILE`` of the positive values and spans
# ``MAP_LOG_DECADES`` decades below that; anything smaller (including zeros)
# gets the lowest colour.
LOG_SCALE_VARIABLES = frozenset({"acc snow melt", "precipitation", "snowfall flux"})
MAP_LOG_UPPER_PERCENTILE = 99.9
MAP_LOG_DECADES = 3


def _array(group: zarr.Group, name: str) -> zarr.Array:
    return cast(zarr.Array, group[name])


def time_array_name(group: zarr.Group) -> str | None:
    """Return the name of ``group``'s time coordinate, if it has one.

    Parameters
    ----------
    group : zarr.Group
        Group to inspect.

    Returns
    -------
    str | None
        The first of ``TIME_NAMES`` present in ``group``, or None.
    """
    names = set(group.array_keys())
    return next((name for name in TIME_NAMES if name in names), None)


def coordinate_names(group: zarr.Group) -> set[str]:
    """Return the arrays in ``group`` that are coordinates rather than data.

    Parameters
    ----------
    group : zarr.Group
        Experiment group.

    Returns
    -------
    set[str]
        The time coordinate plus every array referenced by a CF
        ``coordinates`` attribute.
    """
    time_name = time_array_name(group)
    names = {time_name} if time_name else set()
    for var_name in group.array_keys():
        coords_attr = _array(group, var_name).attrs.get("coordinates")
        if coords_attr:
            names.update(str(coords_attr).split())
    return names


def discover_experiments(base: zarr.Group) -> dict[str, zarr.Group]:
    """Find the experiment groups under ``base``.

    Parameters
    ----------
    base : zarr.Group
        Root group of the dataset.

    Returns
    -------
    dict[str, zarr.Group]
        ``{name: group}`` for every child (or grandchild, named
        ``"child/grandchild"``) of ``base`` holding a time coordinate, or
        ``{"root": base}`` if there are none but ``base`` itself has one,
        minus anything in ``EXCLUDED_EXPERIMENTS``.
    """
    found: dict[str, zarr.Group] = {}
    for name, sub in base.groups():
        if time_array_name(sub) is not None:
            found[name] = sub
            continue
        for name2, sub2 in sub.groups():
            if time_array_name(sub2) is not None:
                found[f"{name}/{name2}"] = sub2
    if not found and time_array_name(base) is not None:
        found["root"] = base
    return {k: v for k, v in found.items() if k not in EXCLUDED_EXPERIMENTS}


def build_catalog(base: zarr.Group) -> dict[str, Any]:
    """Describe the experiments and variables available under ``base``.

    Parameters
    ----------
    base : zarr.Group
        Root group of the dataset.

    Returns
    -------
    dict[str, Any]
        ``{"experiments": [...], "variables": [...]}``, each sorted by name.
        Each experiment lists its ``variables`` keys; each variable carries
        ``units``, ``long_name`` and ``spatial_mean`` (whether it is a 2-D
        field that is plotted as a domain mean).
    """
    experiments = []
    variables: dict[str, dict[str, Any]] = {}
    for name, group in discover_experiments(base).items():
        time_name = time_array_name(group)
        coord_names = coordinate_names(group)
        exp_variables = []
        for var_name in group.array_keys():
            if var_name in coord_names or var_name in EXCLUDED_VARIABLES:
                continue
            arr = _array(group, var_name)
            if arr.ndim not in (1, 2):
                continue
            exp_variables.append(var_name)
            variables.setdefault(
                var_name,
                {
                    "key": var_name,
                    "units": str(arr.attrs.get("units", "")),
                    "long_name": str(
                        arr.attrs.get("long_name", arr.attrs.get("standard_name", ""))
                    ),
                    "spatial_mean": arr.ndim > 1,
                },
            )
        experiments.append(
            {
                "name": name,
                "n_points": int(_array(group, time_name).shape[0])
                if time_name
                else None,
                "has_time": time_name is not None,
                "variables": sorted(exp_variables),
            }
        )
    return {
        "experiments": sorted(experiments, key=lambda e: e["name"]),
        "variables": sorted(variables.values(), key=lambda v: v["key"]),
    }


def _sanitize_floats(values: Iterable[float]) -> list[float | None]:
    """Convert to JSON-safe floats, mapping NaN/inf to None."""
    out: list[float | None] = []
    for v in values:
        fv = float(v)
        out.append(None if math.isnan(fv) or math.isinf(fv) else fv)
    return out


def decode_time(group: zarr.Group, time_name: str) -> list[str] | list[int]:
    """Decode a CF time coordinate to ISO 8601 strings.

    Parameters
    ----------
    group : zarr.Group
        Experiment group.
    time_name : str
        Name of its time coordinate.

    Returns
    -------
    list[str] | list[int]
        ISO timestamps, or a plain sample index ``0..n-1`` if the coordinate
        has no ``units`` or can't be decoded.
    """
    time_arr = _array(group, time_name)
    n = int(time_arr.shape[0])
    units = time_arr.attrs.get("units")
    calendar = str(time_arr.attrs.get("calendar", "standard"))
    if not units:
        return list(range(n))
    try:
        raw = np.asarray(time_arr[:], dtype=float)
        dates = cftime.num2date(raw, units=str(units), calendar=calendar)
        return [d.isoformat() for d in np.atleast_1d(dates)]
    except Exception as e:  # best effort: fall back to a sample index
        logger.warning("time decode failed for %s: %s", time_name, e)
        return list(range(n))


def _experiment_variable(
    base: zarr.Group, experiment: str, variable: str
) -> tuple[zarr.Group, zarr.Array]:
    experiments = discover_experiments(base)
    if experiment not in experiments:
        raise KeyError(f"unknown experiment: {experiment}")
    group = experiments[experiment]
    if variable in EXCLUDED_VARIABLES or variable not in set(group.array_keys()):
        raise KeyError(f"unknown variable: {variable} in experiment {experiment}")
    return group, _array(group, variable)


def build_series(base: zarr.Group, experiment: str, variable: str) -> dict[str, Any]:
    """Read one experiment/variable as a time series.

    Parameters
    ----------
    base : zarr.Group
        Root group of the dataset.
    experiment : str
        Experiment name, as returned by ``discover_experiments``.
    variable : str
        Variable (array) name within the experiment.

    Returns
    -------
    dict[str, Any]
        ``time`` and ``values`` lists plus ``units``/``long_name`` metadata.
        A 2-D variable is averaged over its non-time axis (ignoring NaNs).

    Raises
    ------
    KeyError
        If ``experiment`` or ``variable`` doesn't exist.
    """
    group, data_arr = _experiment_variable(base, experiment, variable)
    raw = np.asarray(data_arr[:], dtype=float)
    time_name = time_array_name(group)
    n = int(_array(group, time_name).shape[0]) if time_name else raw.shape[0]

    values: NDArray[np.float64]
    if raw.ndim == 1:
        values = raw
    else:
        time_axis = next((i for i, s in enumerate(raw.shape) if s == n), 0)
        space_axes = tuple(i for i in range(raw.ndim) if i != time_axis)
        # All-NaN time steps (e.g. fully masked) legitimately average to NaN.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            values = np.nanmean(raw, axis=space_axes)

    return {
        "experiment": experiment,
        "variable": variable,
        "time": decode_time(group, time_name) if time_name else list(range(n)),
        "values": _sanitize_floats(values.tolist()),
        "units": str(data_arr.attrs.get("units", "")),
        "long_name": str(data_arr.attrs.get("long_name", "")),
        "n_points": int(values.size),
        "spatial_mean": raw.ndim > 1,
    }


def nearest_time_index(group: zarr.Group, time_name: str, date: str) -> int:
    """Return the index of the time step nearest ``date``.

    Parameters
    ----------
    group : zarr.Group
        Experiment group.
    time_name : str
        Name of its time coordinate.
    date : str
        ISO date (only the ``YYYY-MM-DD`` part is used), or a plain sample
        index if the time coordinate has no CF ``units``.

    Returns
    -------
    int
        Index into the time coordinate.
    """
    time_arr = _array(group, time_name)
    raw = np.asarray(time_arr[:], dtype=float)
    units = time_arr.attrs.get("units")
    if not units:
        return max(0, min(int(date), len(raw) - 1))
    calendar = str(time_arr.attrs.get("calendar", "standard"))
    parsed = dt.date.fromisoformat(date[:10])
    target = cftime.date2num(
        cftime.datetime(parsed.year, parsed.month, parsed.day, calendar=calendar),
        units=str(units),
        calendar=calendar,
    )
    return int(np.nanargmin(np.abs(raw - target)))


def is_log_scale(arr: zarr.Array) -> bool:
    """Return whether ``arr`` is mapped on a log colour scale.

    Parameters
    ----------
    arr : zarr.Array
        Variable array; matched on its ``long_name`` attribute.

    Returns
    -------
    bool
        ``True`` if its lowercased ``long_name`` is in ``LOG_SCALE_VARIABLES``.
    """
    return str(arr.attrs.get("long_name", "")).strip().lower() in LOG_SCALE_VARIABLES


def _color_range(
    values: NDArray[Any], log: bool, full: bool = False
) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if log:
        positive = finite[finite > 0]
        if positive.size:
            vmax = float(
                positive.max()
                if full
                else np.percentile(positive, MAP_LOG_UPPER_PERCENTILE)
            )
            return vmax / 10**MAP_LOG_DECADES, vmax
        # Nothing positive to put on a log scale: fall back to a linear one.
    if not finite.size:
        return None
    if full:
        return float(finite.min()), float(finite.max())
    vmin, vmax = np.percentile(finite, MAP_PERCENTILES)
    return float(vmin), float(vmax)


def map_color_range(
    base: zarr.Group, experiment: str, variable: str, full: bool = False
) -> tuple[float, float] | None:
    """Return the colour range shared by every date's map of ``variable``.

    The 1st to 99th percentile over all dates and points, so a few extreme
    values don't squash most maps into one end of the scale. Variables in
    ``LOG_SCALE_VARIABLES`` (see ``is_log_scale``) instead span
    ``MAP_LOG_DECADES`` decades up to the ``MAP_LOG_UPPER_PERCENTILE`` of their
    positive values. With ``full``, the range is the data's minimum to maximum
    instead (for a log scale, ``MAP_LOG_DECADES`` decades up to the maximum).

    Parameters
    ----------
    base : zarr.Group
        Root group of the dataset.
    experiment : str
        Experiment name.
    variable : str
        Variable name.

    Returns
    -------
    tuple[float, float] | None
        ``(vmin, vmax)``, or ``None`` if the variable has no finite values.

    Raises
    ------
    KeyError
        If ``experiment`` or ``variable`` doesn't exist.
    """
    _, arr = _experiment_variable(base, experiment, variable)
    return _color_range(np.asarray(arr[:], dtype=float), is_log_scale(arr), full)


def _colorbar_extend(
    color_range: tuple[float, float] | None, full_range: bool, log: bool
) -> Literal["neither", "both", "min", "max"]:
    # Arrows mark values beyond a fixed range: clipped by the percentiles, or
    # (on a full range) only those under a log scale's floor, e.g. zeros.
    if not color_range:
        return "neither"
    if full_range:
        return "min" if log else "neither"
    return "both"


def render_map_png(
    base: zarr.Group,
    experiment: str,
    variable: str,
    date: str,
    color_range: tuple[float, float] | None = None,
    full_range: bool = False,
) -> bytes:
    """Render a map of ``variable`` at the time step nearest ``date``.

    The experiment's points are scatter-plotted by their ``lat``/``lon`` in
    polar stereographic (EPSG:3413) with coastlines.

    Parameters
    ----------
    base : zarr.Group
        Root group of the dataset.
    experiment : str
        Experiment name.
    variable : str
        A ``(time, points)`` variable in the experiment.
    date : str
        Date to map; see ``nearest_time_index``.
    color_range : tuple[float, float] | None, optional
        Fixed ``(vmin, vmax)`` for the colour scale (see ``map_color_range``),
        on which variables in ``LOG_SCALE_VARIABLES`` use a log colour scale.
        By default the scale is instead fitted to this date's minimum to
        maximum, on a linear scale for every variable.
    full_range : bool, optional
        Whether ``color_range`` spans all of the variable's data (see
        ``map_color_range``'s ``full``), so the colour bar has no arrows for
        values beyond it, except below a log scale's floor.

    Returns
    -------
    bytes
        PNG image.

    Raises
    ------
    KeyError
        If ``experiment`` or ``variable`` doesn't exist.
    ValueError
        If ``variable`` isn't 2-D, or the experiment has no time coordinate or
        ``lat``/``lon`` arrays.
    """
    group, arr = _experiment_variable(base, experiment, variable)
    if arr.ndim != 2:
        raise ValueError(f"{variable} has no time dimension to map a single date from")
    time_name = time_array_name(group)
    if not time_name:
        raise ValueError(f"experiment {experiment} has no time coordinate")
    if "lat" not in group or "lon" not in group:
        raise ValueError(f"experiment {experiment} has no lat/lon coordinates to map")

    idx = nearest_time_index(group, time_name, date)
    values = np.asarray(arr[idx, :], dtype=float)
    lat = np.asarray(_array(group, "lat")[:], dtype=float)
    lon = np.asarray(_array(group, "lon")[:], dtype=float)
    finite = np.isfinite(values) & np.isfinite(lat) & np.isfinite(lon)
    lon, lat, values = lon[finite], lat[finite], values[finite]

    # A bare Figure (not pyplot) so rendering is safe from server worker threads.
    fig = Figure(figsize=(3, 4.2))
    ax: Any = fig.add_subplot(1, 1, 1, projection=MAP_PROJECTION)
    # Project the points ourselves so the view can be fitted to the data's
    # bounds in map coordinates (a lon/lat extent this far north leaves a
    # skewed, mostly-empty frame).
    xy = MAP_PROJECTION.transform_points(ccrs.PlateCarree(), lon, lat)
    x, y = xy[:, 0], xy[:, 1]
    norm: Normalize
    log = is_log_scale(arr)
    if color_range is None:
        # Fitted to this date alone: its min to max, on a linear scale.
        color_range = _color_range(values, log=False, full=True)
        full_range = True
        log = False
    log_range = color_range if log else None
    if log_range and log_range[0] > 0:
        # LogNorm masks values <= 0, so lift them to the bottom of the scale.
        values = np.maximum(values, log_range[0])
        norm = LogNorm(*log_range)
    else:
        norm = Normalize(*(color_range or (None, None)))
    sc = ax.scatter(
        x,
        y,
        c=values,
        cmap=MAP_CMAPS.get(
            str(arr.attrs.get("long_name", "")).strip().lower(), DEFAULT_MAP_CMAP
        ),
        norm=norm,
        s=2,
        linewidths=0,
        transform=MAP_PROJECTION,
        zorder=2,
    )
    if x.size:
        pad_x = 0.03 * (x.max() - x.min())
        pad_y = 0.03 * (y.max() - y.min())
        ax.set_extent(
            [x.min() - pad_x, x.max() + pad_x, y.min() - pad_y, y.max() + pad_y],
            crs=MAP_PROJECTION,
        )
    ax.coastlines(resolution="50m", linewidth=0.5, color="0.3", zorder=3)
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.3,
        color="0.6",
        alpha=0.7,
        x_inline=False,
        y_inline=False,
        rotate_labels=False,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = gl.ylabel_style = {"size": 7}
    cbar = fig.colorbar(
        sc,
        ax=ax,
        shrink=0.8,
        orientation="horizontal",
        pad=0.06,
        extend=_colorbar_extend(color_range, full_range, isinstance(norm, LogNorm)),
    )
    cbar.ax.tick_params(labelsize=9)
    if isinstance(norm, LogNorm):
        # Plain-number labels (0.1, 0.3, 1, 3, ...) on the log-spaced bar.
        cbar.ax.xaxis.set_major_locator(LogLocator(subs=(1.0, 3.0)))
        cbar.ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        cbar.ax.xaxis.set_minor_formatter(NullFormatter())
    cbar.set_label(str(arr.attrs.get("units", "")), fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    return buf.getvalue()


@dataclass
class _SnapshotState:
    snapshot: Snapshot
    group: zarr.Group
    catalog: dict[str, Any] | None = None
    series: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    map_ranges: dict[tuple[str, str, MapScale], tuple[float, float] | None] = field(
        default_factory=dict
    )


class Dashboard:
    """Snapshot-cached catalog/series/map reads from a ``DataSource``.

    Parameters
    ----------
    source : DataSource
        Where the data lives.
    """

    def __init__(self, source: DataSource) -> None:
        self.source = source
        self._state: _SnapshotState | None = None
        self._lock = Lock()

    def _ensure_state(self) -> _SnapshotState:
        tip = self.source.tip_snapshot()
        state = self._state
        if state is not None and state.snapshot.id == tip.id:
            return state
        with self._lock:
            if self._state is None or self._state.snapshot.id != tip.id:
                self._state = _SnapshotState(tip, self.source.open_group(tip.id))
            return self._state

    def snapshot(self) -> Snapshot:
        """Return the snapshot currently being served.

        Returns
        -------
        Snapshot
            Tip-of-branch snapshot.
        """
        return self._ensure_state().snapshot

    def repo_info(self) -> dict[str, Any]:
        """Describe the repo and the snapshot currently being served.

        Returns
        -------
        dict[str, Any]
            ``repo``, ``branch``, ``root_path`` and ``snapshot`` (``id``,
            ``written_at``, ``message``).
        """
        snapshot = self.snapshot()
        return {
            "repo": self.source.repo,
            "branch": self.source.branch,
            "root_path": self.source.root_path,
            "snapshot": {
                "id": snapshot.id,
                "written_at": snapshot.written_at,
                "message": snapshot.message,
            },
        }

    def catalog(self) -> dict[str, Any]:
        """Return the (cached) ``build_catalog`` result for the current snapshot.

        Returns
        -------
        dict[str, Any]
            See ``build_catalog``.
        """
        state = self._ensure_state()
        if state.catalog is None:
            state.catalog = build_catalog(state.group)
        return state.catalog

    def series(self, experiment: str, variable: str) -> dict[str, Any]:
        """Return the (cached) ``build_series`` result for the current snapshot.

        Parameters
        ----------
        experiment : str
            Experiment name.
        variable : str
            Variable name.

        Returns
        -------
        dict[str, Any]
            See ``build_series``.
        """
        state = self._ensure_state()
        key = (experiment, variable)
        if key not in state.series:
            state.series[key] = build_series(state.group, experiment, variable)
        return state.series[key]

    def map_png(
        self, experiment: str, variable: str, date: str, scale: MapScale = "shared"
    ) -> bytes:
        """Render ``render_map_png`` against the current snapshot.

        With a shared ``scale``, every date of one experiment/variable shares
        the same (cached) ``map_color_range`` colour scale.

        Parameters
        ----------
        experiment : str
            Experiment name.
        variable : str
            Variable name.
        date : str
            Date to map.
        scale : MapScale, optional
            How the colour scale is chosen; see ``MapScale``.

        Returns
        -------
        bytes
            PNG image.
        """
        state = self._ensure_state()
        if scale == "date":
            return render_map_png(state.group, experiment, variable, date)
        full_range = scale == "full"
        key = (experiment, variable, scale)
        if key not in state.map_ranges:
            state.map_ranges[key] = map_color_range(
                state.group, experiment, variable, full_range
            )
        return render_map_png(
            state.group, experiment, variable, date, state.map_ranges[key], full_range
        )
