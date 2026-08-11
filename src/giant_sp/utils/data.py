"""Utility functions for data handling and sampling."""

from __future__ import annotations

import datetime
import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
from cartopy.io import shapereader
from matplotlib.path import Path
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    import geopandas as gpd
    from shapely.geometry.base import BaseGeometry

# Sample sizes are often produced by ``numpy.random.Generator.integers``,
# which returns a numpy integer scalar rather than a plain ``int``.
IntLike = int | np.integer

# Anything accepted by ``numpy.random.default_rng``, including the
# ``(seed, i)``-style sequences used to derive per-iteration seeds.
SeedLike = int | np.integer | Sequence[int] | None


class NumpyFloatValuesEncoder(json.JSONEncoder):
    """JSON encoder that converts ``numpy.float32`` values to native floats.

    ``json.JSONEncoder`` does not know how to serialize numpy scalar types,
    so this encoder converts ``numpy.float32`` instances to plain Python
    ``float`` before falling back to the default behaviour for anything
    else.
    """

    def default(self, obj: Any) -> Any:
        """Return a JSON-serializable representation of `obj`.

        Parameters
        ----------
        obj : Any
            Object being encoded.

        Returns
        -------
        Any
            ``float(obj)`` if `obj` is a ``numpy.float32``, otherwise the
            result of the parent class's `default`.
        """
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


def sample_land_locations(
    n_obs: IntLike,
    lsm: xr.DataArray | None = None,
    seed: SeedLike = None,
    invert_mask: bool = False,
) -> NDArray[np.floating]:
    """Sample latitude/longitude locations over land, weighted by area.

    Points are drawn (with replacement) from grid cells where `lsm` is
    truthy, with sampling probability proportional to ``cos(latitude)`` to
    account for the change in cell area with latitude.

    Parameters
    ----------
    n_obs : int or numpy.integer
        Number of points to sample.
    lsm : xarray.DataArray or None, default=None
        Land-sea mask with ``lat``/``lon`` dimensions, where sea cells are
        ``0`` (falsy) and land cells are non-zero (truthy). Must be
        provided.
    seed : SeedLike, default=None
        Seed for ``numpy.random.default_rng`` (an int, a sequence of ints,
        or ``None``).
    invert_mask : bool, default=False
        If True, sample from sea locations instead of land locations.

    Returns
    -------
    NDArray[np.floating]
        Array of shape ``(2, n_obs)`` with sampled ``[lat, lon]`` pairs
        stacked along axis 0.

    Raises
    ------
    ValueError
        If `lsm` is None.
    """
    if lsm is None:
        raise ValueError("Land-sea mask (lsm) must be provided.")
    lsm = lsm.astype(bool)
    if invert_mask:
        lsm = ~lsm

    lat_weights = np.cos(np.radians(lsm["lat"]))
    lsm_weighted = lsm * lat_weights

    lsm_stacked = lsm_weighted.stack(points=("lat", "lon")).compute()
    land_points = lsm_stacked.where(lsm_stacked, drop=True)

    probs = np.abs(land_points) / np.abs(land_points).sum()

    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(
        land_points.points.size, n_obs, replace=True, p=probs.values
    )
    sampled_points = land_points.isel(points=sampled_idx)

    lats = sampled_points["lat"].values
    lons = sampled_points["lon"].values
    raw_locs = np.array(list(zip(lats, lons, strict=True))).transpose()  # (2, n_obs)

    return raw_locs


def sample_station_locations(
    stn_locs_df: pd.DataFrame, n_obs: IntLike, seed: SeedLike = None
) -> NDArray[np.floating]:
    """Sample station locations without replacement.

    Parameters
    ----------
    stn_locs_df : pandas.DataFrame
        DataFrame of station locations, with one row per station and
        location columns (e.g. latitude, longitude).
    n_obs : int or numpy.integer
        Number of stations to sample.
    seed : SeedLike, default=None
        Seed for ``numpy.random.default_rng`` (an int, a sequence of ints,
        or ``None``).

    Returns
    -------
    NDArray[np.floating]
        Array of shape ``(n_columns, n_obs)``, i.e. `stn_locs_df` rows
        sampled and transposed so that each column is one sampled station.
    """
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(stn_locs_df), n_obs, replace=False)
    sampled_stns = stn_locs_df.iloc[sampled_idx]

    raw_locs = sampled_stns.to_numpy().transpose()

    return raw_locs


def mixed_sampling_strategy(
    stn_locs_df: pd.DataFrame,
    n_obs: IntLike,
    frac_stn: float = 0.1,
    lsm: xr.DataArray | None = None,
    seed: SeedLike = None,
) -> NDArray[np.floating]:
    """Sample a mix of station and land locations.

    A fraction `frac_stn` of `n_obs` is sampled from `stn_locs_df` (without
    replacement) via `sample_station_locations`, and the remainder is
    sampled over land via `sample_land_locations`.

    Parameters
    ----------
    stn_locs_df : pandas.DataFrame
        DataFrame of station locations, passed to `sample_station_locations`.
    n_obs : int or numpy.integer
        Total number of locations to sample.
    frac_stn : float, default=0.1
        Fraction of `n_obs` to sample from stations; the rest is sampled
        over land.
    lsm : xarray.DataArray or None, default=None
        Land-sea mask passed to `sample_land_locations`.
    seed : SeedLike, default=None
        Seed passed to both `sample_station_locations` and
        `sample_land_locations`.

    Returns
    -------
    NDArray[np.floating]
        Array of shape ``(2, n_obs)`` with station locations followed by
        land locations, concatenated along axis 1.
    """
    n_stn = round(frac_stn * n_obs)
    n_land = n_obs - n_stn
    stn_locs = sample_station_locations(stn_locs_df, n_stn, seed=seed)
    land_locs = sample_land_locations(n_land, lsm=lsm, seed=seed)

    raw_locs = np.concatenate((stn_locs, land_locs), axis=1)
    return raw_locs


def construct_circ_and_linear_time_ds(dates: ArrayLike, freq: str) -> xr.Dataset:
    """Construct circular and linear time covariates for a set of dates.

    Returns an :class:`xarray.Dataset` containing a circular representation
    (sine and cosine) of time, plus a linearly normalised year variable. The
    `freq` argument dictates the cycle length of the circular variable. E.g.:

        - ``'H'``: cycles once per day at hourly intervals
        - ``'D'``: cycles once per year at daily intervals
        - ``'MS'``/``'ME'``: cycles once per year at monthly intervals

    Note: can be applied after preprocessing since outputs are normalised.

    Parameters
    ----------
    dates : array-like
        Datetime-like values, coercible to a :class:`pandas.DatetimeIndex`.
    freq : {'D', 'H', 'MS', 'ME'}
        Frequency governing the cycle length of the circular variable, as
        described above.

    Returns
    -------
    xarray.Dataset
        Dataset with a ``time`` coordinate (from `dates`) and data
        variables ``cos_{freq}``, ``sin_{freq}`` (the circular time
        representation, both in [-1, 1]) and ``year`` (the year of each
        date, linearly normalised to [-1, 1] over the range of `dates`).

    Raises
    ------
    ValueError
        If `freq` is not one of ``'D'``, ``'H'``, ``'MS'`` or ``'ME'``.
    """
    dt_index = pd.DatetimeIndex(dates)
    if freq == "D":
        time_var = dt_index.dayofyear
        mod = 365.25
    elif freq == "H":
        time_var = dt_index.hour
        mod = 24
    elif freq == "MS" or freq == "ME":
        time_var = dt_index.month
        mod = 12
    else:
        raise ValueError("Circular time variable not implemented for this frequency.")

    cos_time = np.cos(2 * np.pi * time_var / mod)
    sin_time = np.sin(2 * np.pi * time_var / mod)
    # normalised year
    year_time = (
        (dt_index.year - dt_index.year.min())
        / (dt_index.year.max() - dt_index.year.min())
        * 2
    ) - 1

    ds = xr.Dataset(
        coords={"time": dt_index},
        data_vars={
            f"cos_{freq}": ("time", cos_time),
            f"sin_{freq}": ("time", sin_time),
            "year": ("time", year_time),
        },
    )
    return ds


def sample_dates_by_year_chunks(
    weights: NDArray[np.floating],
    start_date: str | datetime.date | pd.Timestamp,
    end_date: str | datetime.date | pd.Timestamp,
    n_samples_per_year: int,
) -> list[np.datetime64]:
    """Sample dates within successive 12-month chunks, weighted by day-of-year.

    The range ``[start_date, end_date)`` is split into consecutive 12-month
    periods starting at `start_date`. Within each period, `n_samples_per_year`
    dates are sampled without replacement, using day-of-year weights taken
    from `weights`. Periods containing fewer than `n_samples_per_year` daily
    dates are skipped.

    Parameters
    ----------
    weights : NDArray[np.floating]
        Weight for each day of year, indexed by ``dayofyear - 1`` (i.e.
        index 0 is January 1st). Must have at least 366 entries to cover
        leap years.
    start_date : str, datetime.date, or pandas.Timestamp
        Start of the overall date range (inclusive).
    end_date : str, datetime.date, or pandas.Timestamp
        End of the overall date range (inclusive for the daily grid, but
        acts as the exclusive upper bound for the last 12-month chunk).
    n_samples_per_year : int
        Number of dates to sample per 12-month period.

    Returns
    -------
    list[np.datetime64]
        Sampled dates, in chunk order (not globally sorted).

    Notes
    -----
    Source: https://github.com/eschlager/MeltEmulation/
    """
    sampled_dates: list[np.datetime64] = []

    # Ensure dates and weights are pandas Series
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    weights = weights[dates.dayofyear - 1]

    current_start = pd.Timestamp(start_date)
    while current_start < pd.Timestamp(end_date):
        # Define the end of the 12-month period
        next_year = current_start + pd.DateOffset(years=1)
        current_end = min(next_year, pd.Timestamp(end_date))

        # Select dates within this period
        mask = (dates >= current_start) & (dates < current_end)
        period_dates = dates[mask]
        period_weights = weights[mask]

        if len(period_dates) < n_samples_per_year:
            current_start = next_year
            continue  # Skip periods with too few dates

        # Normalize weights
        period_weights = period_weights / period_weights.sum()

        # Sample without replacement
        sampled = np.random.choice(
            period_dates, size=n_samples_per_year, replace=False, p=period_weights
        )

        sampled_dates.extend(sampled)
        current_start = next_year

    return sampled_dates


def find_lon_lat(ds: xr.Dataset | xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Find the longitude and latitude coordinates of a dataset.

    Coordinates are matched case-insensitively against the names
    ``lon``/``longitude``/``long`` and ``lat``/``latitude``.

    Source: https://github.com/eschlager/MeltEmulation/

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Object whose coordinates are searched for longitude and latitude.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray]
        The ``(lon, lat)`` coordinate arrays.

    Raises
    ------
    ValueError
        If no matching longitude or latitude coordinate is found.
    """
    # Try common names (coords)
    lon_names = [n for n in ds.coords if str(n).lower() in ("lon", "longitude", "long")]
    lat_names = [n for n in ds.coords if str(n).lower() in ("lat", "latitude")]
    if not lon_names or not lat_names:
        raise ValueError("Couldn't find lon/lat variables!")
    lon = ds.coords[lon_names[0]]
    lat = ds.coords[lat_names[0]]
    return lon, lat


def find_xy_for_points(
    ds: xr.Dataset | xr.DataArray, gdf_points: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Find the nearest grid cell x/y coordinates for a set of points.

    For each point in `gdf_points` (with a point geometry in lon/lat), finds
    the nearest grid cell in `ds` (by 2D lon/lat coordinates) and attaches
    its ``x``/``y`` coordinate values.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset or data array with 2D longitude/latitude coordinates and 1D
        ``x``/``y`` dimensions.
    gdf_points : geopandas.GeoDataFrame
        Points with a lon/lat point geometry.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of `gdf_points` with added ``x`` and ``y`` columns giving the
        nearest grid cell's coordinate values.
    """
    lon_da, lat_da = find_lon_lat(ds)
    lon2d, lat2d = lon_da.values, lat_da.values

    X2d_da, Y2d_da = xr.broadcast(ds.coords["x"], ds.coords["y"])
    X2d = X2d_da.transpose(*lon_da.dims).values
    Y2d = Y2d_da.transpose(*lon_da.dims).values

    tree = sp.spatial.cKDTree(np.column_stack([lon2d.ravel(), lat2d.ravel()]))
    query_pts = np.column_stack(
        [gdf_points.geometry.x.values, gdf_points.geometry.y.values]
    )
    _, idx = tree.query(query_pts)

    gdf_points = gdf_points.copy()
    gdf_points["x"] = X2d.ravel()[idx]
    gdf_points["y"] = Y2d.ravel()[idx]
    return gdf_points


def normalize_lons(lon: ArrayLike) -> NDArray[np.floating]:
    """Convert longitudes from 0..360 to -180..180 if needed.

    Source: https://github.com/eschlager/MeltEmulation/

    Parameters
    ----------
    lon : array-like
        Longitude value(s).

    Returns
    -------
    NDArray[np.floating]
        Longitudes in the [-180, 180) range if the input was entirely
        non-negative and exceeded 180, otherwise the input unchanged (as an
        array).
    """
    lon = np.array(lon)
    if lon.min() >= 0 and lon.max() > 180:
        lon = ((lon + 180) % 360) - 180
    return lon


def load_greenland_shape(resolution: str = "10m") -> BaseGeometry:
    """Load the Greenland geometry from Natural Earth's countries dataset.

    Source: https://github.com/eschlager/MeltEmulation/

    Parameters
    ----------
    resolution : str, default='10m'
        Resolution of the Natural Earth ``admin_0_countries`` dataset to
        load (e.g. ``'10m'``, ``'50m'``, ``'110m'``).

    Returns
    -------
    shapely.geometry.base.BaseGeometry
        Shapely geometry (Polygon or MultiPolygon) for Greenland.

    Raises
    ------
    RuntimeError
        If no matching "Greenland" record is found in the Natural Earth
        data.
    """
    shpname = shapereader.natural_earth(
        resolution=resolution, category="cultural", name="admin_0_countries"
    )
    reader = shapereader.Reader(shpname)
    for rec in reader.records():
        props = rec.attributes
        name = props.get("NAME") or props.get("NAME_LONG") or props.get("ADMIN")
        if name and name.lower() == "greenland":
            geom = rec.geometry  # shapely geometry
            return geom
    raise RuntimeError("Greenland geometry not found in Natural Earth data.")


def points_in_polygons(
    lon: NDArray[np.floating], lat: NDArray[np.floating], polygon: BaseGeometry
) -> NDArray[np.bool_]:
    """Test which points fall inside a (multi)polygon.

    Uses :class:`matplotlib.path.Path` for each polygon part, which is fast
    for repeated point-in-polygon tests.

    Source: https://github.com/eschlager/MeltEmulation/

    Parameters
    ----------
    lon : NDArray[np.floating]
        2D array of longitudes.
    lat : NDArray[np.floating]
        2D array of latitudes, same shape as `lon`.
    polygon : shapely.geometry.base.BaseGeometry
        Polygon or MultiPolygon geometry to test against. Holes (interior
        rings) are excluded from the resulting mask.

    Returns
    -------
    NDArray[np.bool_]
        Boolean mask, same shape as `lon`/`lat`, True where the
        corresponding point lies inside `polygon` (and outside any holes).
    """
    pts = np.column_stack((lon.ravel(), lat.ravel()))
    inside = np.zeros(pts.shape[0], dtype=bool)
    # polygon may be MultiPolygon or Polygon
    if polygon.geom_type == "Polygon":
        polys = [polygon.exterior.coords]
        # include interiors (holes) as negative masks
        interiors = list(polygon.interiors)
    else:
        polys = [poly.exterior.coords for poly in polygon.geoms]
        interiors = [interior for poly in polygon.geoms for interior in poly.interiors]

    for poly_coords in polys:
        path = Path(np.array(poly_coords))
        inside |= path.contains_points(pts)

    # subtract holes
    for interior in interiors:
        path_h = Path(np.array(interior.coords))
        inside &= ~path_h.contains_points(pts)

    return inside.reshape(lon.shape)


def context_sampling_with_delta(
    first_set_obj: Any, delta_factor_context: list[int]
) -> list[Any]:
    group_objs = [first_set_obj] + ["all"] * (len(delta_factor_context) - 1)
    return [
        obj
        for obj, n in zip(group_objs, delta_factor_context, strict=True)
        for _ in range(n)
    ] + ["all"]
