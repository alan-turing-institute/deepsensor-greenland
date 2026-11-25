import numpy as np
import pytest
import xarray as xr

from ds_greenland.utils.metrics import (
    compute_grid_metric_maps,
    compute_grid_metrics,
    compute_pit_bernoulli_gamma,
    compute_pit_zigamma_mixture,
    grid_cell_areas,
    melt_extent_km2,
    total_mass_gt,
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

    def test_float32_weights_summing_above_one_stay_within_unit_interval(self):
        """Float32 model outputs can give `pi` summing to just over 1.

        Regression test: with a large observation (every component CDF ~1),
        these weights used to give a PIT of 1.00000001.
        """
        true = np.array([4.0])
        p_zero = np.array([0.87], dtype=np.float32)
        pi = np.array([[0.21973227], [0.78026783]], dtype=np.float32)
        assert pi.astype(np.float64).sum() > 1.0
        k = np.array([[0.06], [0.17]], dtype=np.float32)
        scale = np.array([[0.33], [0.2]], dtype=np.float32)
        pit = compute_pit_zigamma_mixture(true, p_zero, pi, k, scale, seed=42)
        assert np.all(pit >= 0)
        assert np.all(pit <= 1)


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


class TestComputeGridMetricMaps:
    def test_hand_computed_values(self):
        truth = _grid([[0.0, 0.0], [0.0, 0.0]])
        mean = _grid([[1.0, 3.0], [1.0, 1.0]])
        std = _grid([[0.5, 1.5], [0.5, 1.5]])

        maps = compute_grid_metric_maps(mean, std, truth)

        np.testing.assert_allclose(maps["rmse"], [1.0, np.sqrt(5.0)])
        np.testing.assert_allclose(maps["mae"], [1.0, 2.0])
        np.testing.assert_allclose(maps["avgstd"], [0.5, 1.5])

    def test_spatial_mean_rmse_matches_compute_grid_metrics(self):
        rng = np.random.default_rng(0)
        truth, mean, std = (_grid(rng.random((5, 4))) for _ in range(3))

        maps = compute_grid_metric_maps(mean, std, truth)

        assert float(maps["rmse"].mean()) == pytest.approx(
            compute_grid_metrics(mean, std, truth)["rmse"]
        )


def _lonlat_grid(lons, lats):
    lon2d, lat2d = np.meshgrid(lons, lats)
    dims = ("y", "x")
    return xr.DataArray(lon2d, dims=dims), xr.DataArray(lat2d, dims=dims)


class TestGridCellAreas:
    def test_one_degree_cell_matches_spherical_area(self):
        lon, lat = _lonlat_grid([-41.0, -40.0, -39.0], [71.0, 72.0, 73.0])

        area = grid_cell_areas(lon, lat)

        # Spherical 1x1 degree cell centred on 72N (cartopy uses the WGS84
        # ellipsoid, hence the 1% tolerance).
        r = 6_371_000.0
        expected = (
            r**2
            * np.deg2rad(1.0)
            * (np.sin(np.deg2rad(72.5)) - np.sin(np.deg2rad(71.5)))
        )
        assert area.dims == ("y", "x")
        assert float(area[1, 1]) == pytest.approx(expected, rel=0.01)


class TestMeltTotals:
    @pytest.fixture
    def cell_area(self):
        return xr.DataArray(np.full(2, 1e6), dims="cell")  # 1 km² cells

    def test_total_mass_gt(self, cell_area):
        melt = _grid([[2.0, 0.0], [np.nan, 4.0]])  # mm w.e., NaN skipped

        # (2 + 4) mm w.e. x 1e6 m² = 6e6 kg = 6e-6 Gt
        assert total_mass_gt(melt, cell_area) == pytest.approx(6e-6)

    def test_melt_extent_is_mean_daily_area_above_threshold(self, cell_area):
        melt = _grid([[2.0, 0.5], [3.0, 5.0]])  # day 1: 1 cell melts, day 2: 2

        assert melt_extent_km2(melt, cell_area) == pytest.approx(1.5)
