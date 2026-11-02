"""Utility functions for computing metrics."""

import numpy as np
import scipy as sp
import xarray as xr
from numpy.typing import NDArray


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
    (see :class:`giant_sp.models.mdn.GammaMixture`) in place of a single Gamma: the slab
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
    slab_cdf = np.sum(
        pi[:, ~is_zero]
        * sp.stats.gamma.cdf(
            true[~is_zero], a=k[:, ~is_zero], scale=scale[:, ~is_zero]
        ),
        axis=0,
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
