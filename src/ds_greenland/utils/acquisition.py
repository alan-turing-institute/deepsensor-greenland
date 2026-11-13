"""Custom DeepSensor acquisition functions.

`ExpectedEpistemicReduction`/`OracleEpistemicReduction` place sensors where adding an
observation is expected/known to most reduce the model's *epistemic* variance over the
target set. Since the underlying `MDNZIGammaConvNP.epistemic_frac` decomposes *both*
mixture levels of the zero-inflated Gamma predictive distribution (see its docstring),
this is not purely uncertainty in melt *magnitude* -- it also includes uncertainty in
*whether* melt is occurring at all, weighted by how large the melt would be if it did
occur (``p_slab*(1-p_slab)*slab.mean**2``).
`MeanOccurrenceEntropy`/`ExpectedOccurrenceEntropyReduction`/
`OracleOccurrenceEntropyReduction`/`OracleBrierScore` isolate that occurrence
uncertainty *on its own*, unweighted by predicted magnitude, via the Bernoulli entropy
of the zero-inflation spike -- a complementary but overlapping view, not a fully
disjoint one. Unlike the epistemic-variance family, this is *not* decomposed into
epistemic/aleatoric parts: ``p_zero`` is a single deterministic scalar with no
inter-component disagreement signal to split (see
`MDNZIGammaConvNP.occurrence_entropy`'s docstring), so ``H(p_zero)`` mixes reducible
and irreducible occurrence uncertainty -- the "expected reduction" framing still holds
on average (more/closer context generally does shrink it) but isn't a strict epistemic
guarantee the way ``ExpectedEpistemicReduction`` is.

`MeanMagnitudeVariance`/`ExpectedMagnitudeVarianceReduction`/
`OracleMagnitudeVarianceReduction` are the occurrence-independent mirror image of the
occurrence-entropy family: they isolate *magnitude* uncertainty on its own, via
`MDNZIGammaConvNP.magnitude_variance` (the slab's raw inter-component disagreement,
``sum_k pi_k*(mean_k-mean)**2``, *without* `_epistemic_frac_raw`'s ``p_slab``
weighting). That weighting is exactly what makes `ExpectedEpistemicReduction`'s
magnitude contribution shrink for locations where melt is unlikely to occur at all,
regardless of how much the candidate Gamma components disagree about its size if it
did -- these functions target that disagreement directly, independent of occurrence
probability. Together with the occurrence-entropy family, this makes each mixture
level's uncertainty independently targetable: occurrence-only, magnitude-only, or
combined (`ExpectedEpistemicReduction`/`OracleEpistemicReduction`).
`OracleMagnitudeMAE` is `MeanMagnitudeVariance`'s calibration/accuracy counterpart,
scoring predicted magnitude against ground truth (restricted to points where melt
actually occurs) exactly as `OracleBrierScore` does for occurrence.

All require a model exposing
``epistemic_variance``/``occurrence_entropy``/``magnitude_variance``/``magnitude_mean``/
``p_zero`` (e.g. :class:`ds_greenland.models.mdn.MDNZIGammaConvNP`).

The ``Expected*``/``Oracle*Reduction`` pairs share their scoring logic (via a mixin --
see ``_MeanEpistemicVarianceScorer``/``_MeanOccurrenceEntropyScorer`` below) and differ
only in which base class they use: plain
:class:`~deepsensor.active_learning.acquisition_fns.AcquisitionFunction` for the
model-based ``Expected*`` versions, or
:class:`~deepsensor.active_learning.acquisition_fns.AcquisitionFunctionOracle` for the
ground-truth-informed ``Oracle*`` versions. Both rely on
:class:`~deepsensor.active_learning.algorithms.GreedyAlgorithm`'s own "sequential"
per-candidate context-augmentation engine rather than reimplementing one: it samples a
pseudo-observation for each candidate from ``query_infill`` (the model's own predicted
mean by default, or a true ground-truth grid if passed explicitly, as `placement.py`
does for the ``Oracle*`` names) and, called with ``diff=True``, reports
``after - before`` so the acquisition value reads as a *reduction*.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import numpy as np
from deepsensor.active_learning.acquisition_fns import (
    AcquisitionFunction,
    AcquisitionFunctionOracle,
    AcquisitionFunctionParallel,
)
from deepsensor.data.task import Task

logger = logging.getLogger(__name__)

__all__ = [
    "ExpectedEpistemicReduction",
    "ExpectedMagnitudeVarianceReduction",
    "ExpectedOccurrenceEntropyReduction",
    "MeanMagnitudeVariance",
    "MeanOccurrenceEntropy",
    "OracleBrierScore",
    "OracleEpistemicReduction",
    "OracleMagnitudeMAE",
    "OracleMagnitudeVarianceReduction",
    "OracleOccurrenceEntropyReduction",
]


def _mean_over_target_set(values: Any, target_set_idx: int) -> float:
    if isinstance(values, list):
        values = values[target_set_idx]
    return float(np.mean(values))


class _MeanEpistemicVarianceScorer:
    """Shared scoring logic for `ExpectedEpistemicReduction`/`OracleEpistemicReduction`.

    Call ``GreedyAlgorithm(...)(acq_fn, tasks, diff=True)`` (note ``diff=True``) so the
    "importance" of a candidate is the *reduction* in epistemic variance
    (``after - before``); with ``diff=False`` it instead scores the *absolute*
    post-candidate variance.
    """

    min_or_max = "min"

    def __call__(self, task: Task) -> float:
        """Return the mean epistemic variance over `task`'s target set.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.

        Returns
        -------
        float
            Mean epistemic variance over the target set.
        """
        epvar = self.model.epistemic_variance(task)  # type: ignore[attr-defined]
        return _mean_over_target_set(epvar, self.target_set_idx)  # type: ignore[attr-defined]


class _MeanOccurrenceEntropyScorer:
    """Shared scoring logic for the occurrence-entropy-reduction acquisition functions.

    See `_MeanEpistemicVarianceScorer` for the ``diff=True`` usage note.
    """

    min_or_max = "min"

    def __call__(self, task: Task) -> float:
        """Return the mean melt-occurrence entropy over `task`'s target set.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.

        Returns
        -------
        float
            Mean occurrence entropy over the target set.
        """
        entropy = self.model.occurrence_entropy(task)  # type: ignore[attr-defined]
        return _mean_over_target_set(entropy, self.target_set_idx)  # type: ignore[attr-defined]


class _MeanMagnitudeVarianceScorer:
    """Shared scoring logic for the magnitude-variance-reduction acquisition functions.

    See `_MeanEpistemicVarianceScorer` for the ``diff=True`` usage note.
    """

    min_or_max = "min"

    def __call__(self, task: Task) -> float:
        """Return the mean magnitude-only epistemic variance over `task`'s target set.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.

        Returns
        -------
        float
            Mean magnitude-only epistemic variance over the target set.
        """
        magvar = self.model.magnitude_variance(task)  # type: ignore[attr-defined]
        return _mean_over_target_set(magvar, self.target_set_idx)  # type: ignore[attr-defined]


class ExpectedEpistemicReduction(_MeanEpistemicVarianceScorer, AcquisitionFunction):
    """Expected reduction in mean epistemic variance over the target set.

    Targets uncertainty that is reducible by observing more (as opposed to irreducible
    aleatoric noise) -- both in melt *magnitude* (disagreement between the K Gamma
    mixture components) and in *whether* melt is occurring at all (uncertainty in the
    zero-inflation spike, weighted by the predicted magnitude if it did occur); see
    `ds_greenland.models.mdn.MDNZIGammaConvNP.epistemic_frac`'s docstring for the exact
    two-level decomposition. Pseudo-observations at each candidate come from the
    model's own predicted mean (`GreedyAlgorithm`'s default ``query_infill``), computed
    once before the greedy loop starts -- like `deepsensor`'s other built-in
    "sequential" acquisition functions (e.g. ``MeanVariance``), it is *not* refreshed
    after each greedy pick.
    """


class OracleEpistemicReduction(_MeanEpistemicVarianceScorer, AcquisitionFunctionOracle):
    """Ground-truth-informed counterpart to `ExpectedEpistemicReduction`.

    Unlike `ExpectedEpistemicReduction` (which hypothesises a pseudo-observation from
    the model's own predicted mean), this scores mean epistemic variance over the
    target set *after* a candidate has been added to context with its *true* observed
    value (sampled by `GreedyAlgorithm` from a ``query_infill`` you pass explicitly,
    e.g. ``ds[target_var]``) -- isolating how good the model-based pseudo-observation
    approximation is, for offline benchmarking against real held-out data.
    """


class ExpectedOccurrenceEntropyReduction(
    _MeanOccurrenceEntropyScorer, AcquisitionFunction
):
    """Expected reduction in mean melt-occurrence entropy over the target set.

    Targets uncertainty in *whether* melt is occurring at all (the zero-inflation
    spike), rather than uncertainty in its magnitude. See
    `ExpectedEpistemicReduction`'s docstring for the pseudo-observation/``diff``
    mechanics, which are identical here.
    """


class OracleOccurrenceEntropyReduction(
    _MeanOccurrenceEntropyScorer, AcquisitionFunctionOracle
):
    """Ground-truth-informed counterpart to `ExpectedOccurrenceEntropyReduction`.

    Same relationship to `ExpectedOccurrenceEntropyReduction` as
    `OracleEpistemicReduction` has to `ExpectedEpistemicReduction`.
    """


class ExpectedMagnitudeVarianceReduction(
    _MeanMagnitudeVarianceScorer, AcquisitionFunction
):
    """Expected reduction in mean magnitude-only epistemic variance over the target set.

    Targets uncertainty in melt *magnitude* alone -- disagreement between the K Gamma
    mixture components' means, weighted by how undecided the mixture is between them --
    independent of *whether* melt is occurring at all. The occurrence-independent
    mirror image of `ExpectedOccurrenceEntropyReduction`: together they let occurrence
    and magnitude uncertainty be targeted separately, while `ExpectedEpistemicReduction`
    targets their combined (occurrence-weighted) effect. See
    `ds_greenland.models.mdn.MDNZIGammaConvNP.magnitude_variance`'s docstring for the exact
    quantity. Pseudo-observation/``diff`` mechanics are identical to
    `ExpectedEpistemicReduction`'s.
    """


class OracleMagnitudeVarianceReduction(
    _MeanMagnitudeVarianceScorer, AcquisitionFunctionOracle
):
    """Ground-truth-informed counterpart to `ExpectedMagnitudeVarianceReduction`.

    Same relationship to `ExpectedMagnitudeVarianceReduction` as
    `OracleEpistemicReduction` has to `ExpectedEpistemicReduction`.
    """


class MeanOccurrenceEntropy(AcquisitionFunctionParallel):
    """Bernoulli entropy of the predicted melt/no-melt occurrence probability.

    Cheap, single-pass counterpart to `ExpectedOccurrenceEntropyReduction`: evaluates
    the model's current occurrence-entropy directly at each search location (no
    hypothetical context augmentation, hence ``Parallel`` rather than needing
    `GreedyAlgorithm`'s per-candidate sequential engine), mirroring how ``Stddev``
    relates to ``ExpectedEpistemicReduction``.
    """

    min_or_max = "max"

    def __call__(self, task: Task, X_s: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the occurrence entropy at each search point.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.
        X_s : numpy.ndarray
            Search points. Shape ``(2, N_search)``.

        Returns
        -------
        numpy.ndarray
            Occurrence entropy at each search point, shape ``(N_search,)``.
        """
        task = copy.deepcopy(task)
        task["X_t"] = X_s
        # `occurrence_entropy` returns shape `(N_features, *N_targets)` (or a list of
        # such arrays, one per target set) -- always index `[target_set_idx]`, exactly
        # like `Stddev`'s `self.model.stddev(task)[self.target_set_idx]`, to drop the
        # leading features axis and get the `(N_targets,)` shape GreedyAlgorithm's
        # boolean-mask assignment requires.
        entropy = self.model.occurrence_entropy(task)
        if isinstance(entropy, list):
            entropy = entropy[self.target_set_idx]

        return entropy[0]


class MeanMagnitudeVariance(AcquisitionFunctionParallel):
    """Magnitude-only epistemic variance of the predicted melt amount, if any occurs.

    Cheap, single-pass counterpart to `ExpectedMagnitudeVarianceReduction`: evaluates
    the model's current magnitude-only epistemic variance directly at each search
    location (no hypothetical context augmentation, hence ``Parallel`` rather than
    needing `GreedyAlgorithm`'s per-candidate sequential engine), mirroring how
    `MeanOccurrenceEntropy` relates to `ExpectedOccurrenceEntropyReduction` (and
    ``Stddev`` to ``ExpectedEpistemicReduction``).
    """

    min_or_max = "max"

    def __call__(self, task: Task, X_s: np.ndarray, **kwargs: Any) -> np.ndarray:
        """Compute the magnitude-only epistemic variance at each search point.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.
        X_s : numpy.ndarray
            Search points. Shape ``(2, N_search)``.

        Returns
        -------
        numpy.ndarray
            Magnitude-only epistemic variance at each search point, shape
            ``(N_search,)``.
        """
        task = copy.deepcopy(task)
        task["X_t"] = X_s
        # `magnitude_variance` returns shape `(N_features, *N_targets)` (or a list of
        # such arrays, one per target set) -- always index `[target_set_idx]`, exactly
        # like `MeanOccurrenceEntropy`'s `self.model.occurrence_entropy(task)[...]`, to
        # drop the leading features axis and get the `(N_targets,)` shape
        # GreedyAlgorithm's boolean-mask assignment requires.
        magvar = self.model.magnitude_variance(task)
        if isinstance(magvar, list):
            magvar = magvar[self.target_set_idx]

        return magvar[0]


class OracleBrierScore(AcquisitionFunctionOracle):
    """Brier score of the predicted melt-occurrence probability against the truth.

    Ground-truth-informed calibration/accuracy counterpart to `MeanOccurrenceEntropy`
    (which scores pure model-uncertainty, with no notion of whether that uncertainty is
    well-calibrated). Mirrors the existing ``OracleMAE``/``OracleRMSE`` pattern
    directly: scored against ``task["Y_t"]`` as-is (no context augmentation), so use it
    with :class:`~deepsensor.active_learning.algorithms.GreedyAlgorithm`'s default
    ``diff=False``.
    """

    min_or_max = "min"

    def __call__(self, task: Task) -> float:
        """Return the Brier score of predicted vs. true melt occurrence.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.

        Returns
        -------
        float
            Mean squared error between the predicted probability of melt occurrence
            (``1 - p_zero``) and the true occurrence indicator.
        """
        p_zero = self.model.p_zero(task)
        if isinstance(p_zero, list):
            p_zero = p_zero[self.target_set_idx]
        true = task["Y_t"][self.target_set_idx]
        true_is_nonzero = (np.abs(true) >= 1e-6).astype(float)
        return float(np.mean(((1 - p_zero) - true_is_nonzero) ** 2))


class OracleMagnitudeMAE(AcquisitionFunctionOracle):
    """MAE of the predicted melt magnitude against the truth, given melt occurs.

    Ground-truth-informed calibration/accuracy counterpart to `MeanMagnitudeVariance`
    (which scores pure model disagreement about magnitude, with no notion of whether
    that magnitude estimate is accurate) -- the magnitude-uncertainty analogue of
    `OracleBrierScore`, which plays the same role for occurrence. Scored against
    ``task["Y_t"]`` as-is (no context augmentation), so use it with
    :class:`~deepsensor.active_learning.algorithms.GreedyAlgorithm`'s default
    ``diff=False``. Restricted to target points where melt actually occurs (true
    value nonzero), since predicted magnitude is only meaningful conditional on
    occurrence.
    """

    min_or_max = "min"

    def __call__(self, task: Task) -> float:
        """Return the MAE of predicted vs. true melt magnitude, given melt occurs.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            Task object containing context and target sets.

        Returns
        -------
        float
            Mean absolute error between predicted melt magnitude (the slab mean,
            conditional on occurrence) and the true value, over target points where
            melt actually occurs. ``nan`` if no such points exist.
        """
        magmean = self.model.magnitude_mean(task)
        if isinstance(magmean, list):
            magmean = magmean[self.target_set_idx]
        # Drop the leading `N_features` axis (size 1) so the boolean mask below indexes
        # cleanly against `true`'s `(N_targets,)` shape -- exactly like
        # `MeanMagnitudeVariance.__call__`'s `magvar[0]`.
        magmean = magmean[0]
        true = task["Y_t"][self.target_set_idx]
        true_is_nonzero = np.abs(true) >= 1e-6
        if not np.any(true_is_nonzero):
            return float("nan")
        return float(np.mean(np.abs(magmean[true_is_nonzero] - true[true_is_nonzero])))
