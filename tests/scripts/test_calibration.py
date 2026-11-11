import numpy as np
import pandas as pd
import pytest

import giant_sp.scripts.calibration as cal
from giant_sp.utils.data import sample_dates_by_year_chunks, seasonal_doy_weights

TD = np.timedelta64(12, "h")


class TestCandidateSplitDates:
    def test_regular_strides_and_offsets(self):
        dates = cal.candidate_split_dates(
            "2014-01-01", "2014-12-31", "regular", TD, frequency="D", every=200
        )
        assert dates == [
            np.datetime64("2014-01-01T12:00", "ns"),
            np.datetime64("2014-07-20T12:00", "ns"),
        ]

    def test_seasonal_count_and_order(self):
        dates = cal.candidate_split_dates(
            "2014-01-01", "2015-12-31", "seasonal", TD, samples_per_year=50
        )
        assert len(dates) == 2 * 50
        assert dates == sorted(dates)
        assert all(np.asarray(dates).astype("datetime64[h]").astype(int) % 24 == 12)

    def test_seasonal_reproduces_training_dates(self):
        # interpolation.py seeds np.random with 42 before sampling each split.
        np.random.seed(42)
        expected = sample_dates_by_year_chunks(
            seasonal_doy_weights(), "2014-01-01", "2014-12-31", 150
        )
        dates = cal.candidate_split_dates(
            "2014-01-01", "2014-12-31", "seasonal", TD, samples_per_year=150, seed=42
        )
        assert dates == list(
            np.sort(np.asarray(expected) + TD).astype("datetime64[ns]")
        )

    def test_seasonal_favours_melt_season(self):
        dates = cal.candidate_split_dates(
            "2000-01-01", "2009-12-31", "seasonal", TD, samples_per_year=150
        )
        months = pd.DatetimeIndex(dates).month
        assert np.isin(months, [6, 7, 8]).mean() > 0.4
        assert np.isin(months, [12, 1, 2]).mean() < 0.1

    def test_seasonal_needs_samples_per_year(self):
        with pytest.raises(ValueError, match="samples_per_year"):
            cal.candidate_split_dates("2014-01-01", "2014-12-31", "seasonal", TD)

    def test_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown date sampling"):
            cal.candidate_split_dates("2014-01-01", "2014-12-31", "monthly", TD)


class TestSeasonalSamplesPerYear:
    def test_configured_value_wins(self):
        dataloader = {
            "temporal_sampling_method": "seasonal",
            "temporal_sampling_every": 150,
        }
        assert cal.seasonal_samples_per_year(20, dataloader) == 20

    def test_falls_back_to_training_value(self):
        dataloader = {
            "temporal_sampling_method": "seasonal",
            "temporal_sampling_every": 150,
        }
        assert cal.seasonal_samples_per_year(None, dataloader) == 150

    def test_requires_value_when_training_was_regular(self):
        dataloader = {
            "temporal_sampling_method": "regular",
            "temporal_sampling_every": 1,
        }
        with pytest.raises(ValueError, match="seasonal_samples_per_year"):
            cal.seasonal_samples_per_year(None, dataloader)


def test_seasonal_doy_weights():
    weights = seasonal_doy_weights()
    assert weights.shape == (366,)
    assert weights.sum() == pytest.approx(1.0)
    assert weights.argmax() == 205 - 1


class TestEnsembleCoverage:
    def test_matches_autocast_quantile_interval(self):
        samples = np.arange(101, dtype=float)[None, :].repeat(3, axis=0)
        # 0.5 coverage -> [25, 75] quantiles of 0..100.
        covered = cal.ensemble_coverage(samples, np.array([25.0, 50.0, 75.5]), 0.5)
        np.testing.assert_array_equal(covered, [1.0, 1.0, 0.0])

    def test_invalid_level(self):
        with pytest.raises(ValueError, match="coverage_level"):
            cal.ensemble_coverage(np.zeros((1, 5)), np.zeros(1), 1.0)


def test_sample_marginal_ensemble_shape_and_zero_fraction():
    rng = np.random.default_rng(0)
    n = 2000
    samples = cal.sample_marginal_ensemble(
        np.full(n, 0.3), np.full(n, 2.0), np.full(n, 1.5), 100, rng
    )
    assert samples.shape == (n, 100)
    assert (samples > 0).mean() == pytest.approx(0.3, abs=0.01)
    assert samples[samples > 0].mean() == pytest.approx(3.0, rel=0.02)


def test_coverage_table_calibrated_when_truth_from_model():
    rng = np.random.default_rng(1)
    n = 20_000
    shape, scale = rng.uniform(1, 3, n), rng.uniform(0.5, 2, n)
    y = rng.gamma(shape, scale)  # always melt, drawn from the predictive
    diag = pd.DataFrame(
        {
            "split": "val",
            "melt_prob": 1.0,
            "gamma_shape": shape,
            "gamma_scale_norm": scale,
            "y_true_norm": y,
            "observed_melt": 1.0,
        }
    )
    table = cal.coverage_table(diag, n_samples=100, seed=0, chunk_size=7000)
    assert list(table["coverage_level"]) == cal.default_coverage_levels()
    assert table["observed_coverage_no_melt"].isna().all()
    # Sample quantiles from 100 members widen intervals slightly.
    assert table["abs_error"].max() < 0.03
    error = cal.coverage_calibration_error(table)
    assert error.loc["val", "coverage_calibration_error"] < 0.02
    assert error.loc["val", "coverage_calibration_error_randomized"] < 0.02


def test_randomized_coverage_calibrated_with_zero_inflation():
    rng = np.random.default_rng(2)
    n = 20_000
    melt_prob = rng.uniform(0.0, 0.3, n)
    shape, scale = rng.uniform(1, 3, n), rng.uniform(0.5, 2, n)
    melt = rng.uniform(size=n) < melt_prob
    y = np.where(melt, rng.gamma(shape, scale), 0.0)
    diag = pd.DataFrame(
        {
            "split": "val",
            "melt_prob": melt_prob,
            "gamma_shape": shape,
            "gamma_scale_norm": scale,
            "y_true_norm": y,
            "observed_melt": melt.astype(float),
        }
    )
    table = cal.coverage_table(diag, n_samples=100, seed=0)
    # Ties at zero make autocast's coverage ~1 at every level ...
    assert table["observed_coverage"].min() > 0.8
    # ... while random tie-breaking recovers the nominal level.
    assert table["abs_error_randomized"].max() < 0.02


def test_randomized_ensemble_pit_ties_uniform():
    rng = np.random.default_rng(3)
    samples = np.zeros((50_000, 100))
    pit = cal.randomized_ensemble_pit(samples, np.zeros(50_000), rng)
    assert pit.min() >= 0
    assert pit.max() <= 1
    np.testing.assert_allclose(
        np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit), 0.1, atol=0.01
    )
