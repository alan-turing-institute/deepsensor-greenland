import numpy as np
from deepsensor.data.task import Task

from giant_sp.utils.acquisition import ExpectedEpistemicReduction


class _StubModel:
    """Minimal stand-in for `MDNZIGammaConvNP` exposing `mean`/`epistemic_variance`.

    `epistemic_variance` decreases as more context points are added (mimicking a model
    that becomes more certain with more observations), so the acquisition function
    should return positive reductions.
    """

    def mean(self, task: Task) -> np.ndarray:
        n_t = task["X_t"].shape[-1]
        return np.zeros(n_t)

    def epistemic_variance(self, task: Task) -> np.ndarray:
        n_context = task["X_c"][0].shape[-1]
        n_t = task["X_t"].shape[-1] if task["X_t"] is not None else 1
        # Epistemic variance shrinks with more context points, floored at 0.1.
        value = max(1.0 - 0.1 * n_context, 0.1)
        return np.full(n_t, value)


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
    def test_output_shape(self):
        model = _StubModel()
        acq_fn = ExpectedEpistemicReduction(model)
        task = _make_task(n_context=3, n_target=5)
        X_s = np.random.default_rng(1).uniform(size=(2, 4))

        result = acq_fn(task, X_s)

        assert result.shape == (4,)

    def test_reductions_match_baseline_minus_candidate(self):
        model = _StubModel()
        acq_fn = ExpectedEpistemicReduction(model)
        task = _make_task(n_context=3, n_target=5)
        X_s = np.random.default_rng(2).uniform(size=(2, 3))

        result = acq_fn(task, X_s)

        baseline = model.epistemic_variance(task).mean()
        # One candidate point added -> n_context + 1 -> lower epistemic variance ->
        # positive reduction, identical for every candidate given the stub's logic.
        expected_each = baseline - max(1.0 - 0.1 * 4, 0.1)
        np.testing.assert_allclose(result, np.full(3, expected_each))

    def test_original_task_is_not_mutated(self):
        model = _StubModel()
        acq_fn = ExpectedEpistemicReduction(model)
        task = _make_task(n_context=3, n_target=5)
        original_n_context = task["X_c"][0].shape[-1]
        X_s = np.random.default_rng(3).uniform(size=(2, 2))

        acq_fn(task, X_s)

        assert task["X_c"][0].shape[-1] == original_n_context

    def test_min_or_max_is_max(self):
        assert ExpectedEpistemicReduction.min_or_max == "max"
