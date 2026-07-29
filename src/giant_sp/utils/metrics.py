import numpy as np
import scipy as sp


def compute_pit_bernoulli_gamma(true, p_zero, k, scale, epsilon=1e-6, seed=42):
    """PIT values for a zero-inflated Bernoulli-Gamma predictive distribution.

    The predictive CDF has a jump at zero, so a plain CDF evaluation (as used
    for continuous likelihoods) is not uniform under a correct model. Follows
    the non-randomized PIT for discrete/mixed distributions (Czado et al.,
    2009): values falling in the zero point mass are assigned a value drawn
    uniformly between 0 and the mass P(Y=0); values above the point mass use
    the standard Gamma CDF shifted by that mass.
    """
    rng = np.random.default_rng(seed)
    is_zero = np.abs(true) < epsilon
    pit = np.empty_like(true, dtype=float)
    pit[is_zero] = rng.uniform(0, p_zero[is_zero])
    pit[~is_zero] = p_zero[~is_zero] + (1 - p_zero[~is_zero]) * sp.stats.gamma.cdf(
        true[~is_zero], a=k[~is_zero], scale=scale[~is_zero]
    )
    return pit
