import numpy as np
import pytest
from deepsensor.active_learning.acquisition_fns import AcquisitionFunctionParallel
from deepsensor.data.task import Task

from giant_sp.utils.acquisition import (
    ExpectedEpistemicReduction,
    ExpectedMagnitudeVarianceReduction,
    ExpectedOccurrenceEntropyReduction,
    MeanMagnitudeVariance,
    MeanOccurrenceEntropy,
    OracleBrierScore,
    OracleEpistemicReduction,
    OracleMagnitudeMAE,
    OracleMagnitudeVarianceReduction,
    OracleOccurrenceEntropyReduction,
)


class _StubModel:
    """Minimal stand-in for `MDNZIGammaConvNP`.

    Exposes `mean`/`epistemic_variance`/`occurrence_entropy`/`magnitude_variance`/
    `p_zero`, each returning shape ``(1, N_targets)`` -- matching the real model's
    convention (a leading ``N_features``/``dim_y`` axis of size 1) -- so tests catch
    shape bugs like forgetting to index ``[target_set_idx]`` to drop that leading axis
    (see ``MeanOccurrenceEntropy``). `epistemic_variance`, `occurrence_entropy`, and
    `magnitude_variance` all decrease as more context points are added (mimicking a
    model that becomes more certain with more observations), so the "expected
    reduction"/"Oracle*Reduction" acquisition functions should return positive
    reductions.
    """

    def mean(self, task: Task) -> np.ndarray:
        n_t = task["X_t"].shape[-1]
        return np.zeros((1, n_t))

    def epistemic_variance(self, task: Task) -> np.ndarray:
        n_context = task["X_c"][0].shape[-1]
        n_t = task["X_t"].shape[-1] if task["X_t"] is not None else 1
        # Epistemic variance shrinks with more context points, floored at 0.1.
        value = max(1.0 - 0.1 * n_context, 0.1)
        return np.full((1, n_t), value)

    def occurrence_entropy(self, task: Task) -> np.ndarray:
        n_context = task["X_c"][0].shape[-1]
        n_t = task["X_t"].shape[-1] if task["X_t"] is not None else 1
        # Occurrence entropy shrinks with more context points, floored at 0.05.
        value = max(np.log(2) - 0.1 * n_context, 0.05)
        return np.full((1, n_t), value)

    def magnitude_variance(self, task: Task) -> np.ndarray:
        n_context = task["X_c"][0].shape[-1]
        n_t = task["X_t"].shape[-1] if task["X_t"] is not None else 1
        # Magnitude-only epistemic variance shrinks with more context, floored at 0.02,
        # using a different slope/floor than `epistemic_variance` so tests can't
        # accidentally pass by confusing the two quantities.
        value = max(0.5 - 0.05 * n_context, 0.02)
        return np.full((1, n_t), value)

    def p_zero(self, task: Task) -> np.ndarray:
        n_t = task["X_t"].shape[-1] if task["X_t"] is not None else 1
        return np.full((1, n_t), 0.3)

    def magnitude_mean(self, task: Task) -> np.ndarray:
        n_t = task["X_t"].shape[-1] if task["X_t"] is not None else 1
        return np.full((1, n_t), 1.5)


def _make_task(n_context: int, n_target: int) -> Task:
    rng = np.random.default_rng(0)
    return Task(
        {
            "X_c": [rng.uniform(size=(2, n_context))],
            "Y_c": [rng.uniform(size=(1, n_context))],
            "X_t": rng.uniform(size=(2, n_target)),
            "Y_t": None,
        }
    )


class TestExpectedEpistemicReduction:
    def test_scores_task_directly(self):
        model = _StubModel()
        acq_fn = ExpectedEpistemicReduction(model)
        task = _make_task(n_context=3, n_target=5)

        result = acq_fn(task)

        assert result == pytest.approx(model.epistemic_variance(task).mean())

    def test_matches_oracle_counterpart_given_the_same_task(self):
        # Expected*/Oracle* share their scoring logic via a mixin -- given the same
        # task, they must return identical values; they differ only in which class
        # GreedyAlgorithm treats specially (query_infill sampling/task_loader
        # requirement), not in what __call__ computes.
        model = _StubModel()
        task = _make_task(n_context=3, n_target=5)

        assert ExpectedEpistemicReduction(model)(task) == OracleEpistemicReduction(
            model
        )(task)

    def test_min_or_max_is_min(self):
        # `GreedyAlgorithm(..., diff=True)` computes after-before, so "min" selects the
        # most negative (largest) reduction.
        assert ExpectedEpistemicReduction.min_or_max == "min"

    def test_not_parallel(self):
        # Must not be an AcquisitionFunctionParallel: GreedyAlgorithm raises if
        # `diff=True` is combined with a Parallel acquisition function, and these rely
        # on `diff=True` for their reduction framing.
        assert not isinstance(
            ExpectedEpistemicReduction(_StubModel()), AcquisitionFunctionParallel
        )


class TestExpectedOccurrenceEntropyReduction:
    def test_scores_task_directly(self):
        model = _StubModel()
        acq_fn = ExpectedOccurrenceEntropyReduction(model)
        task = _make_task(n_context=3, n_target=5)

        result = acq_fn(task)

        assert result == pytest.approx(model.occurrence_entropy(task).mean())

    def test_matches_oracle_counterpart_given_the_same_task(self):
        model = _StubModel()
        task = _make_task(n_context=3, n_target=5)

        assert ExpectedOccurrenceEntropyReduction(model)(
            task
        ) == OracleOccurrenceEntropyReduction(model)(task)

    def test_min_or_max_is_min(self):
        assert ExpectedOccurrenceEntropyReduction.min_or_max == "min"

    def test_not_parallel(self):
        assert not isinstance(
            ExpectedOccurrenceEntropyReduction(_StubModel()),
            AcquisitionFunctionParallel,
        )


class TestExpectedMagnitudeVarianceReduction:
    def test_scores_task_directly(self):
        model = _StubModel()
        acq_fn = ExpectedMagnitudeVarianceReduction(model)
        task = _make_task(n_context=3, n_target=5)

        result = acq_fn(task)

        assert result == pytest.approx(model.magnitude_variance(task).mean())

    def test_matches_oracle_counterpart_given_the_same_task(self):
        model = _StubModel()
        task = _make_task(n_context=3, n_target=5)

        assert ExpectedMagnitudeVarianceReduction(model)(
            task
        ) == OracleMagnitudeVarianceReduction(model)(task)

    def test_min_or_max_is_min(self):
        assert ExpectedMagnitudeVarianceReduction.min_or_max == "min"

    def test_not_parallel(self):
        assert not isinstance(
            ExpectedMagnitudeVarianceReduction(_StubModel()),
            AcquisitionFunctionParallel,
        )


class TestOracleMagnitudeVarianceReduction:
    def test_scores_task_directly(self):
        model = _StubModel()
        acq_fn = OracleMagnitudeVarianceReduction(model)
        task = _make_task(n_context=3, n_target=5)

        result = acq_fn(task)

        assert result == pytest.approx(model.magnitude_variance(task).mean())

    def test_min_or_max_is_min(self):
        assert OracleMagnitudeVarianceReduction.min_or_max == "min"


class TestMeanOccurrenceEntropy:
    def test_output_is_one_dimensional(self):
        # Regression test: `occurrence_entropy` returns shape (N_features, N_targets)
        # -- forgetting to index `[target_set_idx]` left a 2D (1, N_search) array,
        # which `GreedyAlgorithm`'s boolean-mask assignment rejects with
        # "NumPy boolean array indexing assignment requires a 0 or 1-dimensional
        # input, input has 2 dimensions".
        model = _StubModel()
        acq_fn = MeanOccurrenceEntropy(model)
        task = _make_task(n_context=3, n_target=5)
        X_s = np.random.default_rng(4).uniform(size=(2, 6))

        result = acq_fn(task, X_s)

        assert result.shape == (6,)

    def test_output_matches_model_directly(self):
        model = _StubModel()
        acq_fn = MeanOccurrenceEntropy(model)
        task = _make_task(n_context=3, n_target=5)
        X_s = np.random.default_rng(4).uniform(size=(2, 6))

        result = acq_fn(task, X_s)

        # Stub's `occurrence_entropy` depends only on `n_context` (unchanged here) and
        # `n_t` (== number of search points, since `__call__` sets X_t = X_s).
        expected = model.occurrence_entropy(
            Task({"X_c": task["X_c"], "Y_c": task["Y_c"], "X_t": X_s, "Y_t": None})
        )[0]
        np.testing.assert_allclose(result, expected)

    def test_original_task_is_not_mutated(self):
        model = _StubModel()
        acq_fn = MeanOccurrenceEntropy(model)
        task = _make_task(n_context=3, n_target=5)
        original_x_t = task["X_t"]
        X_s = np.random.default_rng(5).uniform(size=(2, 2))

        acq_fn(task, X_s)

        assert task["X_t"] is original_x_t

    def test_min_or_max_is_max(self):
        assert MeanOccurrenceEntropy.min_or_max == "max"


class TestMeanMagnitudeVariance:
    def test_output_is_one_dimensional(self):
        # Regression test: `magnitude_variance` returns shape (N_features, N_targets)
        # -- forgetting to index `[target_set_idx]` left a 2D (1, N_search) array,
        # which `GreedyAlgorithm`'s boolean-mask assignment rejects with
        # "NumPy boolean array indexing assignment requires a 0 or 1-dimensional
        # input, input has 2 dimensions".
        model = _StubModel()
        acq_fn = MeanMagnitudeVariance(model)
        task = _make_task(n_context=3, n_target=5)
        X_s = np.random.default_rng(4).uniform(size=(2, 6))

        result = acq_fn(task, X_s)

        assert result.shape == (6,)

    def test_output_matches_model_directly(self):
        model = _StubModel()
        acq_fn = MeanMagnitudeVariance(model)
        task = _make_task(n_context=3, n_target=5)
        X_s = np.random.default_rng(4).uniform(size=(2, 6))

        result = acq_fn(task, X_s)

        # Stub's `magnitude_variance` depends only on `n_context` (unchanged here) and
        # `n_t` (== number of search points, since `__call__` sets X_t = X_s).
        expected = model.magnitude_variance(
            Task({"X_c": task["X_c"], "Y_c": task["Y_c"], "X_t": X_s, "Y_t": None})
        )[0]
        np.testing.assert_allclose(result, expected)

    def test_original_task_is_not_mutated(self):
        model = _StubModel()
        acq_fn = MeanMagnitudeVariance(model)
        task = _make_task(n_context=3, n_target=5)
        original_x_t = task["X_t"]
        X_s = np.random.default_rng(5).uniform(size=(2, 2))

        acq_fn(task, X_s)

        assert task["X_t"] is original_x_t

    def test_min_or_max_is_max(self):
        assert MeanMagnitudeVariance.min_or_max == "max"


class TestOracleEpistemicReduction:
    def test_scores_task_directly(self):
        model = _StubModel()
        acq_fn = OracleEpistemicReduction(model)
        task = _make_task(n_context=3, n_target=5)

        result = acq_fn(task)

        assert result == pytest.approx(model.epistemic_variance(task).mean())

    def test_min_or_max_is_min(self):
        # `GreedyAlgorithm(..., diff=True)` computes after-before, so "min" selects the
        # most negative (largest) reduction.
        assert OracleEpistemicReduction.min_or_max == "min"


class TestOracleOccurrenceEntropyReduction:
    def test_scores_task_directly(self):
        model = _StubModel()
        acq_fn = OracleOccurrenceEntropyReduction(model)
        task = _make_task(n_context=3, n_target=5)

        result = acq_fn(task)

        assert result == pytest.approx(model.occurrence_entropy(task).mean())

    def test_min_or_max_is_min(self):
        assert OracleOccurrenceEntropyReduction.min_or_max == "min"


class TestOracleBrierScore:
    def test_matches_manual_brier_score(self):
        model = _StubModel()
        acq_fn = OracleBrierScore(model)
        task = _make_task(n_context=3, n_target=4)
        true = np.array([0.0, 1.0, 0.0, 2.0])
        task["Y_t"] = [true]

        result = acq_fn(task)

        p_zero = model.p_zero(task)
        true_is_nonzero = np.array([0.0, 1.0, 0.0, 1.0])
        expected = np.mean(((1 - p_zero) - true_is_nonzero) ** 2)
        assert result == pytest.approx(expected)

    def test_perfect_calibration_gives_zero_score(self):
        class _PerfectModel(_StubModel):
            def p_zero(self, task: Task) -> np.ndarray:
                true = task["Y_t"][0]
                return (np.abs(true) < 1e-6).astype(float)

        model = _PerfectModel()
        acq_fn = OracleBrierScore(model)
        task = _make_task(n_context=3, n_target=4)
        task["Y_t"] = [np.array([0.0, 1.0, 0.0, 2.0])]

        assert acq_fn(task) == pytest.approx(0.0)

    def test_min_or_max_is_min(self):
        assert OracleBrierScore.min_or_max == "min"


class TestOracleMagnitudeMAE:
    def test_matches_manual_mae(self):
        model = _StubModel()
        acq_fn = OracleMagnitudeMAE(model)
        task = _make_task(n_context=3, n_target=4)
        true = np.array([0.0, 1.0, 0.0, 2.0])
        task["Y_t"] = [true]

        result = acq_fn(task)

        magmean = model.magnitude_mean(task)[0]
        true_is_nonzero = np.abs(true) >= 1e-6
        expected = np.mean(np.abs(magmean[true_is_nonzero] - true[true_is_nonzero]))
        assert result == pytest.approx(expected)

    def test_perfect_prediction_gives_zero_score(self):
        class _PerfectModel(_StubModel):
            def magnitude_mean(self, task: Task) -> np.ndarray:
                return np.array([task["Y_t"][0]])

        model = _PerfectModel()
        acq_fn = OracleMagnitudeMAE(model)
        task = _make_task(n_context=3, n_target=4)
        task["Y_t"] = [np.array([0.0, 1.0, 0.0, 2.0])]

        assert acq_fn(task) == pytest.approx(0.0)

    def test_returns_nan_when_no_occurrence(self):
        model = _StubModel()
        acq_fn = OracleMagnitudeMAE(model)
        task = _make_task(n_context=3, n_target=3)
        task["Y_t"] = [np.zeros(3)]

        assert np.isnan(acq_fn(task))

    def test_min_or_max_is_min(self):
        assert OracleMagnitudeMAE.min_or_max == "min"
