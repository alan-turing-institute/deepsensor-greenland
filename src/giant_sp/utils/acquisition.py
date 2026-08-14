"""Custom DeepSensor acquisition functions.

`ExpectedEpistemicReduction` places sensors where adding an observation is expected to most reduce the model's
*epistemic* variance over the target set. Requires a model exposing
``epistemic_variance`` (e.g. :class:`giant_sp.models.mdn.MDNZIGammaConvNP`).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from deepsensor.active_learning.acquisition_fns import AcquisitionFunctionParallel
from deepsensor.data.task import Task

logger = logging.getLogger(__name__)

__all__ = ["ExpectedEpistemicReduction"]


class ExpectedEpistemicReduction(AcquisitionFunctionParallel):
    """Expected reduction in mean epistemic variance over the target set.

    For each candidate search location, hypothetically adds it to the context set
    (using the model's own predicted mean as a pseudo-observation, since ground truth
    is unknown at candidate locations) and computes the reduction in mean epistemic
    variance over the task's target set relative to the current context.

    .. warning::
        Unlike ``Stddev``/``ContextDist``, this requires one model forward pass *per
        candidate search location* (each hypothetically augments the context set
        differently), so it is materially more expensive on large search grids.
    """

    min_or_max = "max"

    def __call__(self, task: Task, X_s: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the expected epistemic-variance reduction at each search point.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.
        X_s : numpy.ndarray
            Search points. Shape ``(2, N_search)``.

        Returns
        -------
        numpy.ndarray
            Expected reduction in mean epistemic variance, shape ``(N_search,)``.
            Larger values indicate a candidate location expected to most reduce
            epistemic uncertainty over the target set.
        """
        baseline = self._mean_epistemic_variance(task)

        n_search = X_s.shape[-1]
        reductions = np.empty(n_search)
        for i in range(n_search):
            x_i = X_s[:, i : i + 1]
            candidate_task = self._augment_context_with_candidate(task, x_i)
            reductions[i] = baseline - self._mean_epistemic_variance(candidate_task)

        return reductions

    def _mean_epistemic_variance(self, task: Task) -> float:
        epvar = self.model.epistemic_variance(task)
        if isinstance(epvar, list):
            epvar = epvar[self.target_set_idx]
        return float(np.mean(epvar))

    def _augment_context_with_candidate(self, task: Task, x_i: np.ndarray) -> Task:
        """Return a copy of `task` with `x_i` (and its predicted mean) appended to context."""
        candidate_task = copy.deepcopy(task)
        x_c = candidate_task["X_c"][self.context_set_idx]
        y_c = candidate_task["Y_c"][self.context_set_idx]

        # Predict at the single candidate location to get its pseudo-observation, since
        # the ground-truth value at a not-yet-placed sensor is unknown.
        task_at_candidate = copy.deepcopy(task)
        task_at_candidate["X_t"] = x_i
        y_hat_i = self.model.mean(task_at_candidate)
        if isinstance(y_hat_i, list):
            y_hat_i = y_hat_i[self.target_set_idx]
        y_hat_i = np.asarray(y_hat_i).reshape(y_c.shape[0], 1)

        candidate_task["X_c"][self.context_set_idx] = np.concatenate(
            [x_c, x_i], axis=-1
        )
        candidate_task["Y_c"][self.context_set_idx] = np.concatenate(
            [y_c, y_hat_i], axis=-1
        )
        return candidate_task
