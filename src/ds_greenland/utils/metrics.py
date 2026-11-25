"""Utility functions for computing metrics."""

import cartopy.crs as ccrs
import numpy as np
import scipy as sp
import xarray as xr
from numpy.typing import NDArray

# Equal-area projection centred on Greenland, used to measure grid-cell areas.
_GREENLAND_EQUAL_AREA = ccrs.LambertAzimuthalEqualArea(
    central_longitude=-40, central_latitude=72
)


def compute_pit_bernoulli_gamma(
    true: NDArray[np.floating],
    p_zero: NDArray[np.floating],
    k: NDArray[np.floating],
    scale: NDArray[np.floating],
    epsilon: float = 1e-6,
    seed: int | None = 42,
) -> NDArray[np.float64]:
    """PIT values for a zero-inflated Bernoulli-Gamma predictive distribution.

    The predictive CDF has a jump at zero, so a plain CDF evaluation (as used
    for continuous likelihoods) is not uniform under a correct model. Follows
    the non-randomized PIT for discrete/mixed distributions (Czado et al.,
    2009): values falling in the zero point mass are assigned a value drawn
    uniformly between 0 and the mass P(Y=0); values above the point mass use
    the standard Gamma CDF shifted by that mass.

    Parameters
    ----------
    true : NDArray[np.floating]
        Observed values, some of which may be (approximately) zero.
    p_zero : NDArray[np.floating]
        Predicted probability mass at zero, ``P(Y=0)``, broadcastable to the
        shape of `true`.
    k : NDArray[np.floating]
        Gamma shape parameter of the predictive distribution, broadcastable
        to the shape of `true`.
    scale : NDArray[np.floating]
        Gamma scale parameter of the predictive distribution, broadcastable
        to the shape of `true`.
    epsilon : float, default=1e-6
        Absolute-value threshold below which an observation in `true` is
        treated as a zero (point mass) rather than a continuous draw.
    seed : int or None, default=42
        Seed for the random number generator used to draw PIT values within
        the zero point mass.

    Returns
    -------
    NDArray[np.float64]
        PIT values, same shape as `true`. Under a correctly specified model
        these are approximately uniformly distributed on [0, 1].

    """
    rng = np.random.default_rng(seed)
    is_zero = np.abs(true) < epsilon
    pit = np.empty_like(true, dtype=float)
    pit[is_zero] = rng.uniform(0, p_zero[is_zero])
    pit[~is_zero] = p_zero[~is_zero] + (1 - p_zero[~is_zero]) * sp.stats.gamma.cdf(
        true[~is_zero], a=k[~is_zero], scale=scale[~is_zero]
    )
    return pit


def compute_pit_zigamma_mixture(
    true: NDArray[np.floating],
    p_zero: NDArray[np.floating],
    pi: NDArray[np.floating],
    k: NDArray[np.floating],
    scale: NDArray[np.floating],
    epsilon: float = 1e-6,
    seed: int | None = 42,
) -> NDArray[np.float64]:
    """PIT values for a zero-inflated ``K``-component Gamma-mixture predictive distribution.

    Generalises :func:`compute_pit_bernoulli_gamma` to a ``K``-component Gamma "slab"
    (see :class:`ds_greenland.models.mdn.GammaMixture`) in place of a single Gamma: the slab
    CDF at a point is the mixture-weighted sum of each component's Gamma CDF,
    ``sum_k pi_k * GammaCDF(y; k_k, scale_k)``, substituted for the plain Gamma CDF in
    the same non-randomized PIT construction (Czado et al., 2009) that
    `compute_pit_bernoulli_gamma` uses around the zero point mass. At ``K=1`` (``pi``
    identically ``1``) this reduces exactly to `compute_pit_bernoulli_gamma`.

    Parameters
    ----------
    true : NDArray[np.floating]
        Observed values, some of which may be (approximately) zero. Shape ``(N,)``.
    p_zero : NDArray[np.floating]
        Predicted probability mass at zero, ``P(Y=0)``, broadcastable to the shape of
        `true`.
    pi : NDArray[np.floating]
        Mixture weights for each of the ``K`` Gamma components, shape ``(K, N)``
        (broadcastable to ``(K, *true.shape)``), summing to ``1`` along axis ``0``.
    k : NDArray[np.floating]
        Gamma shape parameter of each component, shape ``(K, N)``.
    scale : NDArray[np.floating]
        Gamma scale parameter of each component, shape ``(K, N)``.
    epsilon : float, default=1e-6
        Absolute-value threshold below which an observation in `true` is treated as a
        zero (point mass) rather than a continuous draw.
    seed : int or None, default=42
        Seed for the random number generator used to draw PIT values within the zero
        point mass.

    Returns
    -------
    NDArray[np.float64]
        PIT values, same shape as `true`. Under a correctly specified model these are
        approximately uniformly distributed on [0, 1].

    """
    rng = np.random.default_rng(seed)
    is_zero = np.abs(true) < epsilon
    pit = np.empty_like(true, dtype=float)
    pit[is_zero] = rng.uniform(0, p_zero[is_zero])
    # Model outputs are float32, so `pi` can sum to slightly more than 1 (e.g.
    # 1 + 1e-7); clip so the slab CDF (and hence the PIT) can't round above 1.
    slab_cdf = np.clip(
        np.sum(
            pi[:, ~is_zero]
            * sp.stats.gamma.cdf(
                true[~is_zero], a=k[:, ~is_zero], scale=scale[:, ~is_zero]
            ),
            axis=0,
        ),
        0.0,
        1.0,
    )
    pit[~is_zero] = p_zero[~is_zero] + (1 - p_zero[~is_zero]) * slab_cdf
    return pit


def compute_grid_metrics(
    mean: xr.DataArray, std: xr.DataArray, truth: xr.DataArray
) -> dict[str, float]:
    """Summary skill metrics of gridded predictions against the gridded truth.

    Uses the same definitions as ``placement_eval.py``, so the two scripts'
    numbers are comparable. NaNs (e.g. grid cells outside the land mask) are
    skipped.

    Parameters
    ----------
    mean : xr.DataArray
        Predicted mean, with a ``time`` dimension plus spatial dimensions.
    std : xr.DataArray
        Predicted standard deviation, same dimensions as `mean`.
    truth : xr.DataArray
        Ground truth, aligned with `mean` (same coordinates).

    Returns
    -------
    dict[str, float]
        ``"rmse"``: per-grid-cell RMSE over time, averaged over space;
        ``"mae"``: mean absolute error over all dates and grid cells;
        ``"avgstd"``: mean predictive std over points where `truth` is defined.

    """
    err = mean - truth
    return {
        "rmse": float(np.sqrt((err**2).mean(dim="time")).mean()),
        "mae": float(np.abs(err).mean()),
        "avgstd": float(std.where(truth.notnull()).mean()),
    }


def compute_grid_metric_maps(
    mean: xr.DataArray, std: xr.DataArray, truth: xr.DataArray
) -> xr.Dataset:
    """Per-grid-cell skill maps of gridded predictions against the gridded truth.

    Each metric is computed separately for every grid cell, from that cell's
    values over all dates (i.e. averaged over ``time`` only, not over space),
    giving one map per metric. The spatial mean of the ``"rmse"`` map equals
    `compute_grid_metrics`' RMSE. NaNs (e.g. outside the land mask) are skipped.

    Parameters
    ----------
    mean : xr.DataArray
        Predicted mean, with a ``time`` dimension plus spatial dimensions.
    std : xr.DataArray
        Predicted standard deviation, same dimensions as `mean`.
    truth : xr.DataArray
        Ground truth, aligned with `mean` (same coordinates).

    Returns
    -------
    xr.Dataset
        ``"rmse"``, ``"mae"`` and ``"avgstd"`` maps, on the spatial
        dimensions of `mean`.

    """
    err = mean - truth
    return xr.Dataset(
        {
            "rmse": np.sqrt((err**2).mean(dim="time")),
            "mae": abs(err).mean(dim="time"),
            "avgstd": std.where(truth.notnull()).mean(dim="time"),
        }
    )


def grid_cell_areas(lon: xr.DataArray, lat: xr.DataArray) -> xr.DataArray:
    """Approximate area of each cell of a (possibly curvilinear) grid.

    Projects the cell centres to an equal-area projection centred on
    Greenland and takes the Jacobian of the grid-index -> projected-position
    mapping, so it works for any smooth grid (rotated-pole, polar
    stereographic, regular lon/lat).

    Parameters
    ----------
    lon, lat : xr.DataArray
        2D geographic longitudes/latitudes (degrees) of the cell centres,
        at least 2 cells along each dimension.

    Returns
    -------
    xr.DataArray
        Cell areas in m², with the dimensions and coordinates of `lon`.

    """
    pts = _GREENLAND_EQUAL_AREA.transform_points(
        ccrs.PlateCarree(), np.asarray(lon), np.asarray(lat)
    )
    dx_drow, dx_dcol = np.gradient(pts[..., 0])
    dy_drow, dy_dcol = np.gradient(pts[..., 1])
    area = np.abs(dx_dcol * dy_drow - dx_drow * dy_dcol)
    return xr.DataArray(area, coords=lon.coords, dims=lon.dims)


def total_mass_gt(values: xr.DataArray, cell_area: xr.DataArray) -> float:
    """Total mass (Gt) of a water-equivalent field, summed over cells and time.

    Parameters
    ----------
    values : xr.DataArray
        Field in mm w.e. (1 mm w.e. = 1 kg m⁻²), e.g. daily melt; any
        ``time`` dimension is summed over too. NaNs are skipped.
    cell_area : xr.DataArray
        Cell areas in m² (see `grid_cell_areas`), on the spatial dimensions
        of `values`.

    Returns
    -------
    float
        Total mass in Gt (10¹² kg).

    """
    return float((values * cell_area).sum()) / 1e12


def melt_extent_km2(
    values: xr.DataArray, cell_area: xr.DataArray, threshold: float = 1.0
) -> float:
    """Melt extent: total area of cells with melt above `threshold`.

    Parameters
    ----------
    values : xr.DataArray
        Melt field (mm w.e. per day). If it has a ``time`` dimension, the
        extent of each date is averaged over time (mean daily melt extent).
    cell_area : xr.DataArray
        Cell areas in m², on the spatial dimensions of `values`.
    threshold : float, default 1.0
        Melt above which a cell counts as melting (mm w.e. per day), as in
        MeltEmulation's ``plot_map``.

    Returns
    -------
    float
        Melt extent in km².

    """
    area = cell_area.where(values > threshold, 0).sum(dim=cell_area.dims)
    return float(area.mean()) / 1e6
