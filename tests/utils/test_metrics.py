import numpy as np
import pytest
import xarray as xr

from giant_sp.utils.metrics import (
    compute_grid_metrics,
    compute_pit_bernoulli_gamma,
    compute_pit_zigamma_mixture,
)


@pytest.fixture
def zero_inflated_gamma_params():
    true = np.array([0.0, 5.0, 0.0, 2.0])
    p_zero = np.array([0.3, 0.3, 0.6, 0.1])
    k = np.array([2.0, 2.0, 2.0, 2.0])
    scale = np.array([1.0, 1.0, 1.0, 1.0])
    return true, p_zero, k, scale


class TestComputePitBernoulliGamma:
    def test_pit_values_are_within_unit_interval(self, zero_inflated_gamma_params):
        true, p_zero, k, scale = zero_inflated_gamma_params
        pit = compute_pit_bernoulli_gamma(true, p_zero, k, scale, seed=42)
        assert np.all(pit >= 0)
        assert np.all(pit <= 1)

    def test_zero_observations_fall_within_the_point_mass(
        self, zero_inflated_gamma_params
    ):
        true, p_zero, k, scale = zero_inflated_gamma_params
        pit = compute_pit_bernoulli_gamma(true, p_zero, k, scale, seed=42)
        is_zero = np.abs(true) < 1e-6
        assert np.all(pit[is_zero] <= p_zero[is_zero])

    def test_nonzero_observations_fall_above_the_point_mass(
        self, zero_inflated_gamma_params
    ):
        true, p_zero, k, scale = zero_inflated_gamma_params
        pit = compute_pit_bernoulli_gamma(true, p_zero, k, scale, seed=42)
        is_zero = np.abs(true) < 1e-6
        assert np.all(pit[~is_zero] >= p_zero[~is_zero])

    def test_seed_is_reproducible(self, zero_inflated_gamma_params):
        true, p_zero, k, scale = zero_inflated_gamma_params
        first = compute_pit_bernoulli_gamma(true, p_zero, k, scale, seed=1)
        second = compute_pit_bernoulli_gamma(true, p_zero, k, scale, seed=1)
        np.testing.assert_array_equal(first, second)

    def test_all_zero_point_mass_gives_pit_in_range(self):
        true = np.zeros(5)
        p_zero = np.full(5, 0.5)
        k = np.full(5, 2.0)
        scale = np.full(5, 1.0)
        pit = compute_pit_bernoulli_gamma(true, p_zero, k, scale, seed=0)
        assert np.all(pit <= p_zero)


@pytest.fixture
def zero_inflated_gamma_mixture_params():
    true = np.array([0.0, 5.0, 0.0, 2.0])
    p_zero = np.array([0.3, 0.3, 0.6, 0.1])
    pi = np.array([[0.4, 0.4, 0.4, 0.4], [0.6, 0.6, 0.6, 0.6]])
    k = np.array([[2.0, 2.0, 2.0, 2.0], [5.0, 5.0, 5.0, 5.0]])
    scale = np.array([[1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]])
    return true, p_zero, pi, k, scale


class TestComputePitZigammaMixture:
    def test_pit_values_are_within_unit_interval(
        self, zero_inflated_gamma_mixture_params
    ):
        true, p_zero, pi, k, scale = zero_inflated_gamma_mixture_params
        pit = compute_pit_zigamma_mixture(true, p_zero, pi, k, scale, seed=42)
        assert np.all(pit >= 0)
        assert np.all(pit <= 1)

    def test_zero_observations_fall_within_the_point_mass(
        self, zero_inflated_gamma_mixture_params
    ):
        true, p_zero, pi, k, scale = zero_inflated_gamma_mixture_params
        pit = compute_pit_zigamma_mixture(true, p_zero, pi, k, scale, seed=42)
        is_zero = np.abs(true) < 1e-6
        assert np.all(pit[is_zero] <= p_zero[is_zero])

    def test_nonzero_observations_fall_above_the_point_mass(
        self, zero_inflated_gamma_mixture_params
    ):
        true, p_zero, pi, k, scale = zero_inflated_gamma_mixture_params
        pit = compute_pit_zigamma_mixture(true, p_zero, pi, k, scale, seed=42)
        is_zero = np.abs(true) < 1e-6
        assert np.all(pit[~is_zero] >= p_zero[~is_zero])

    def test_seed_is_reproducible(self, zero_inflated_gamma_mixture_params):
        true, p_zero, pi, k, scale = zero_inflated_gamma_mixture_params
        first = compute_pit_zigamma_mixture(true, p_zero, pi, k, scale, seed=1)
        second = compute_pit_zigamma_mixture(true, p_zero, pi, k, scale, seed=1)
        np.testing.assert_array_equal(first, second)

    def test_single_component_matches_bernoulli_gamma(self, zero_inflated_gamma_params):
        """At K=1 (`pi` identically 1), the mixture reduces to a single Gamma."""
        true, p_zero, k, scale = zero_inflated_gamma_params
        pi = np.ones((1, *true.shape))
        mixture_pit = compute_pit_zigamma_mixture(
            true, p_zero, pi, k[np.newaxis, :], scale[np.newaxis, :], seed=42
        )
        single_pit = compute_pit_bernoulli_gamma(true, p_zero, k, scale, seed=42)
        np.testing.assert_array_almost_equal(mixture_pit, single_pit)


def _grid(values):
    """Wrap a (time, cell) array as a DataArray."""
    return xr.DataArray(np.array(values, dtype=float), dims=("time", "cell"))


class TestComputeGridMetrics:
    def test_hand_computed_values(self):
        truth = _grid([[0.0, 0.0], [0.0, 0.0]])
        mean = _grid([[1.0, 3.0], [1.0, 1.0]])
        std = _grid([[0.5, 1.5], [0.5, 1.5]])

        metrics = compute_grid_metrics(mean, std, truth)

        # Per-cell RMSE over time: cell 0 -> 1, cell 1 -> sqrt((9 + 1) / 2).
        assert metrics["rmse"] == pytest.approx((1.0 + np.sqrt(5.0)) / 2)
        assert metrics["mae"] == pytest.approx(1.5)
        assert metrics["avgstd"] == pytest.approx(1.0)

    def test_skips_points_without_truth(self):
        # A cell outside the land mask has no truth; its (large) std and error
        # must not count.
        truth = _grid([[0.0, np.nan], [0.0, np.nan]])
        mean = _grid([[2.0, 100.0], [2.0, 100.0]])
        std = _grid([[1.0, 100.0], [1.0, 100.0]])

        metrics = compute_grid_metrics(mean, std, truth)

        assert metrics == pytest.approx({"rmse": 2.0, "mae": 2.0, "avgstd": 1.0})
