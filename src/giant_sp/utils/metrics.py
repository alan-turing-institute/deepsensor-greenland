"""Utility functions for computing metrics."""

import numpy as np
import scipy as sp
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
