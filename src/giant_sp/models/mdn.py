"""Zero-inflated Gamma Mixture Density Network (MDN) likelihood for ConvNP.

Extends DeepSensor/neuralprocesses with an ``"mdn-zigamma"`` predictive likelihood: a
Dirac spike at zero (matching the zero-inflation already used by the existing
``"bernoulli-gamma"`` likelihood for non-negative, often-exactly-zero melt data) plus a
``K``-component Gamma mixture "slab".

The total predictive variance is decomposed via the law of total variance into:

- ``aleatoric_var`` = ``sum_k pi_k * var_k`` (expected within-component variance)
- ``epistemic_var`` = ``sum_k pi_k * (mean_k - mean)**2`` (inter-component disagreement)

``neuralprocesses``'s coder/likelihood classes are registered against its own shared,
open ``plum`` dispatcher (``neuralprocesses._dispatch``), so ``GammaMixture`` and
``ZIGammaMixtureLikelihood`` below register into that same table with no monkeypatching.
The one closed extension point is
``neuralprocesses.architectures.util.construct_likelihood``, a plain string ``if/elif``
bound by name into ``neuralprocesses.architectures.convgnp`` at import time -- adding the
``"mdn-zigamma"`` spec string requires patching that bound name, done by
:func:`install_mdn_zigamma_likelihood`.
"""

from __future__ import annotations

import logging
from typing import Any

import lab as B
import neuralprocesses as nps
import neuralprocesses.architectures.convgnp as convgnp_module
import numpy as np
from deepsensor.data.task import Task
from deepsensor.model.convnp import ConvNP
from neuralprocesses import _dispatch
from neuralprocesses.aggregate import Aggregate, AggregateInput
from neuralprocesses.datadims import data_dims
from neuralprocesses.dist import AbstractDistribution, AbstractMultiOutputDistribution
from neuralprocesses.dist.spikeslab import SpikesSlab
from neuralprocesses.likelihood import AbstractLikelihood
from neuralprocesses.mask import Masked
from neuralprocesses.util import register_module, split, split_dimension
from plum import dispatch

logger = logging.getLogger(__name__)

__all__ = [
    "GammaMixture",
    "MDNZIGammaConvNP",
    "ZIGammaMixtureLikelihood",
    "install_mdn_zigamma_likelihood",
]


def _bernoulli_entropy(p: np.ndarray) -> np.ndarray:
    """Compute the Bernoulli entropy of `p`, clipped to avoid ``log(0)``.

    Clips in ``float64`` regardless of `p`'s own dtype: in ``float32``, the upper
    bound ``1 - 1e-12`` rounds to exactly ``1.0`` (float32 only has ~7 decimal digits
    of precision), so clipping a float32 array against it silently leaves an exact
    ``1.0`` unclipped, and ``log(1 - 1.0) == log(0)`` raises
    ``RuntimeWarning: divide by zero encountered in log``.
    """
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


class GammaMixture(AbstractDistribution):
    """A `K`-component Gamma mixture, without zero-inflation.

    Zero-inflation is added by wrapping this distribution as the ``slab`` of a
    :class:`~neuralprocesses.dist.spikeslab.SpikesSlab`, exactly as
    ``BernoulliGammaLikelihood`` wraps a plain
    :class:`~neuralprocesses.dist.gamma.Gamma`.

    Parameters
    ----------
    logits : tensor
        Unnormalized mixture logits, shape ``(..., dim_y, K, *spatial)``.
    k : tensor
        Per-component Gamma shape parameters, same shape as `logits`.
    scale : tensor
        Per-component Gamma scale parameters, same shape as `logits`.
    d : int
        Number of trailing axes (``dim_y`` + spatial, i.e. everything except the ``K``
        component axis) to sum over in :meth:`logpdf`, matching the `d + 1` convention
        used by :class:`~neuralprocesses.dist.gamma.Gamma` and
        :class:`~neuralprocesses.dist.spikeslab.SpikesSlab`.

    Notes
    -----
    Only the single-output-variable (``dim_y == 1``, non-:class:`Aggregate`) case is
    supported; this repo only ever configures a single target variable. Sampling is not
    implemented, since nothing in this repo's training/decomposition/acquisition-function
    code paths needs it (only ``logpdf`` for training and ``mean``/``var``/
    ``epistemic_var``/``aleatoric_var`` for prediction and placement).
    """

    def __init__(self, logits: B.Numeric, k: B.Numeric, scale: B.Numeric, d: int):
        self.k = k
        self.scale = scale
        self.d = d
        # `k`/`scale`/`logits` have shape (..., dim_y, K, *spatial), where `spatial`
        # occupies the last `d - 1` axes and `dim_y` the one before that -- so `K` sits
        # at position `-d` (immediately after `dim_y`, immediately before `spatial`).
        self._k_axis = -d
        self.logits = logits
        self.logprobs = logits - B.logsumexp(logits, axis=self._k_axis, squeeze=False)

    @property
    def probs(self) -> B.Numeric:
        """tensor: Normalized mixture weights, same shape as `logits`."""
        return B.exp(self.logprobs)

    @property
    def component_mean(self) -> B.Numeric:
        """tensor: Per-component Gamma means (``k * scale``)."""
        return self.k * self.scale

    @property
    def component_var(self) -> B.Numeric:
        """tensor: Per-component Gamma variances (``k * scale**2``)."""
        return self.k * self.scale**2

    @property
    def mean(self) -> B.Numeric:
        """tensor: Mixture mean, ``sum_k pi_k * mean_k``."""
        return B.sum(self.probs * self.component_mean, axis=self._k_axis)

    @property
    def aleatoric_var(self) -> B.Numeric:
        """tensor: Expected within-component variance, ``sum_k pi_k * var_k``."""
        return B.sum(self.probs * self.component_var, axis=self._k_axis)

    @property
    def epistemic_var(self) -> B.Numeric:
        """tensor: Inter-component variance, ``sum_k pi_k * (mean_k - mean)**2``."""
        mean = B.expand_dims(self.mean, axis=self._k_axis, times=1)
        return B.sum(self.probs * (self.component_mean - mean) ** 2, axis=self._k_axis)

    @property
    def var(self) -> B.Numeric:
        """tensor: Total mixture variance, ``aleatoric_var + epistemic_var``."""
        return self.aleatoric_var + self.epistemic_var

    @_dispatch
    def logpdf(self, x: Masked) -> B.Numeric:
        """Log-density at `x`, masking out points that fall in the zero-spike.

        `SpikesSlab.logpdf` calls `self.slab.logpdf(Masked(x, ind_slab))`, masking out
        points that are actually the zero-spike before scoring them under the slab.
        Mirrors `Gamma.logpdf`'s exact `Masked` handling: substitute a safe dummy value
        at masked-out points (so `log(x)` doesn't blow up there), then re-run with the
        mask multiplied into the per-point log-density before the final `d`-axis sum.
        """
        x_arr, mask = x.y, x.mask
        with B.on_device(self.k):
            safe = B.to_active_device(B.one(B.dtype(self.k)))
        x_arr = mask * x_arr + (1 - mask) * safe
        return self.logpdf(x_arr, mask=mask)

    @_dispatch  # type: ignore[no-redef]
    def logpdf(self, x: B.Numeric, *, mask: B.Numeric | int = 1) -> B.Numeric:  # noqa: F811
        """Log-density at `x`, optionally multiplied by a precomputed `mask`."""
        x = B.expand_dims(x, axis=self._k_axis, times=1)
        logz = B.loggamma(self.k) + self.k * B.log(self.scale)
        comp_logpdf = (self.k - 1) * B.log(x) - x / self.scale - logz
        mix_logpdf = B.logsumexp(self.logprobs + comp_logpdf, axis=self._k_axis)
        mix_logpdf = mix_logpdf * mask
        if self.d == 0:
            return mix_logpdf
        return B.sum(mix_logpdf, axis=tuple(range(B.rank(mix_logpdf)))[-self.d :])

    def __repr__(self) -> str:
        """Return a debug string with the distribution's raw parameter tensors."""
        return (
            f"GammaMixture(k={self.k!r}, scale={self.scale!r}, logits={self.logits!r})"
        )

    def __str__(self) -> str:
        """Return the same debug string as `__repr__`."""
        return repr(self)


@register_module
class ZIGammaMixtureLikelihood(AbstractLikelihood):
    """Mixture of a `K`-component Gamma mixture and a Dirac delta at zero.

    Parameters
    ----------
    K : int
        Number of Gamma mixture components.
    epsilon : float, default 1e-6
        Tolerance for equality checking (passed through to
        :class:`~neuralprocesses.dist.spikeslab.SpikesSlab`).
    """

    @_dispatch
    def __init__(self, K: int, epsilon: float = 1e-6):
        self.K = K
        self.epsilon = epsilon

    def __str__(self) -> str:
        """Return the same debug string as `__repr__`."""
        return repr(self)

    def __repr__(self) -> str:
        """Return a debug string with `K` and `epsilon`."""
        return f"ZIGammaMixtureLikelihood(K={self.K!r}, epsilon={self.epsilon!r})"


def _make_torch_module_class(cls: type) -> type:
    """Mix a plain `@register_module`-style class with `torch.nn.Module`.

    `@register_module` only appends to a list (`neuralprocesses.util.modules`) that
    `neuralprocesses.torch` walks *once*, at its own import time, to build the actual
    `(cls, torch.nn.Module)`-mixed classes used at construction time (see
    `neuralprocesses/torch/__init__.py`'s `create_init`/`create_forward` loop) --
    registering `ZIGammaMixtureLikelihood` after that import has already happened is
    too late for it to pick this class up automatically, so replicate that same
    mixing here instead of relying on it.
    """
    from neuralprocesses.torch.nn import Module as NPSModule

    def __init__(self: Any, *args: Any, **kw_args: Any) -> None:
        NPSModule.__init__(self)
        cls.__init__(self, *args, **kw_args)  # type: ignore[misc]

    def forward(self: Any, *args: Any, **kw_args: Any) -> Any:
        return cls.__call__(self, *args, **kw_args)

    return type(
        cls.__name__, (cls, NPSModule), {"__init__": __init__, "forward": forward}
    )


_TorchZIGammaMixtureLikelihood = _make_torch_module_class(ZIGammaMixtureLikelihood)


@_dispatch
def code(
    coder: ZIGammaMixtureLikelihood,
    xz: Any,
    z: Any,
    x: Any,
    *,
    dtype_lik: Any = None,
    **kw_args: Any,
) -> Any:
    k, scale, logits, logp0, logps, d = _zigammamixture(coder, xz, z)

    if dtype_lik:
        k = B.cast(dtype_lik, k)
        scale = B.cast(dtype_lik, scale)
        logits = B.cast(dtype_lik, logits)
        logp0 = B.cast(dtype_lik, logp0)
        logps = B.cast(dtype_lik, logps)

    with B.on_device(z):
        dtype = dtype_lik or B.dtype(z)
        spikes = B.stack(B.zero(dtype))

    return xz, SpikesSlab(
        spikes,
        GammaMixture(logits, k, scale, d),
        B.stack(logp0, logps, axis=-1),
        d,
        epsilon=coder.epsilon,
    )


@_dispatch
def _zigammamixture(
    coder: ZIGammaMixtureLikelihood, xz: AggregateInput, z: Aggregate
) -> Any:
    raise NotImplementedError(
        "ZIGammaMixtureLikelihood does not support multiple aggregate target sets "
        "(dim_y > 1 via neuralprocesses.aggregate.Aggregate). giant_sp only ever "
        "configures a single target variable."
    )


@_dispatch  # type: ignore[no-redef]
def _zigammamixture(coder: ZIGammaMixtureLikelihood, xz: Any, z: B.Numeric) -> Any:
    d = data_dims(xz)
    dim_y = B.shape(z, -d - 1) // (2 + 3 * coder.K)

    z_k, z_scale, z_logits, z_logp0, z_logps = split(
        z,
        (dim_y * coder.K, dim_y * coder.K, dim_y * coder.K, dim_y, dim_y),
        -d - 1,
    )
    z_k = split_dimension(z_k, -d - 1, (dim_y, coder.K))
    z_scale = split_dimension(z_scale, -d - 1, (dim_y, coder.K))
    z_logits = split_dimension(z_logits, -d - 1, (dim_y, coder.K))

    # Transform into parameters.
    k = 1e-3 + B.softplus(z_k)
    scale = 1e-3 + B.softplus(z_scale)
    logits = z_logits
    logp0 = z_logp0
    logps = z_logps

    return k, scale, logits, logp0, logps, d + 1


_original_construct_likelihood = convgnp_module.construct_likelihood


def install_mdn_zigamma_likelihood(K: int) -> None:
    """Register the ``"mdn-zigamma"`` likelihood spec with ``neuralprocesses``.

    Patches ``neuralprocesses.architectures.convgnp.construct_likelihood`` (the bound
    name actually called when building a ConvNP, per
    ``neuralprocesses.architectures.convgnp:6,221``) to recognise a new
    ``spec == "mdn-zigamma"`` in addition to the built-in specs. Safe to call multiple
    times (e.g. once per model build) -- each call wraps the *original* unpatched
    function, so repeated calls don't stack wrappers, and the number of mixture
    components `K` used by the most recent call takes effect for subsequent model
    builds.

    Parameters
    ----------
    K : int
        Number of Gamma mixture components for the ``"mdn-zigamma"`` likelihood.
    """

    def patched_construct_likelihood(
        nps_module: Any = nps,
        *,
        spec: str,
        dim_y: int,
        num_basis_functions: int,
        dtype: Any,
    ) -> Any:
        if spec == "mdn-zigamma":
            num_channels = dim_y * (2 + 3 * K)  # k, scale, logits (x K) + 2 log-probs
            selector = nps_module.SelectFromChannels(
                dim_y * K, dim_y * K, dim_y * K, dim_y, dim_y
            )
            lik = _TorchZIGammaMixtureLikelihood(K)
            return num_channels, selector, lik
        return _original_construct_likelihood(
            nps_module,
            spec=spec,
            dim_y=dim_y,
            num_basis_functions=num_basis_functions,
            dtype=dtype,
        )

    convgnp_module.construct_likelihood = patched_construct_likelihood
    logger.info("Installed 'mdn-zigamma' likelihood (K=%d) into neuralprocesses", K)


class MDNZIGammaConvNP(ConvNP):
    """`ConvNP` subclass exposing epistemic/aleatoric decomposition for `"mdn-zigamma"`.

    Requires :func:`install_mdn_zigamma_likelihood` to have been called (with the same
    `K` as ``config["model"]["mixture_components"]``) before construction.
    """

    def __init__(
        self, data_processor: Any, task_loader: Any, *args: Any, **kwargs: Any
    ) -> None:
        # `mixture_components` is giant_sp/`install_mdn_zigamma_likelihood`-specific
        # config, not a `neuralprocesses.construct_convgnp` argument -- pop it before
        # delegating, then restore it into `self.config` so it round-trips through
        # `model_config.json` for later reloads (see `placement.py::load_wandb_model`).
        mixture_components = kwargs.pop("mixture_components", None)
        super().__init__(data_processor, task_loader, *args, **kwargs)
        if mixture_components is not None:
            self.config["mixture_components"] = mixture_components

    def _set_num_mixture_components(self) -> None:
        if self.config["likelihood"] == "mdn-zigamma":
            # P(zero) vs P(nonzero), matching the outer SpikesSlab split -- same
            # semantics `mixture_probs` already has for "bernoulli-gamma".
            self.N_mixture_components = 2
        else:
            super()._set_num_mixture_components()

    def _assert_mdn_zigamma(self, method_name: str) -> None:
        if self.config["likelihood"] != "mdn-zigamma":
            raise NotImplementedError(
                f"ConvNP.{method_name} method not supported for likelihood "
                f"{self.config['likelihood']!r}. Valid likelihoods: 'mdn-zigamma'."
            )

    def _epistemic_frac_raw(self, dist: AbstractMultiOutputDistribution) -> B.Numeric:
        p_slab = B.exp(dist.logprobs[..., -1])
        return (p_slab * dist.slab.epistemic_var) / dist.var

    @dispatch
    def epistemic_frac(self, dist: AbstractMultiOutputDistribution):
        """Fraction of total predictive variance attributable to epistemic uncertainty.

        Returned numpy arrays have shape ``(N_features, *N_targets)``, clipped to
        ``[0, 1]``.
        """
        self._assert_mdn_zigamma("epistemic_frac")
        frac = self._cast_numpy_and_squeeze(self._epistemic_frac_raw(dist))
        frac = self._maybe_concat_multi_targets(frac)
        if isinstance(frac, (list, tuple)):
            return [np.clip(f, 0.0, 1.0) for f in frac]
        return np.clip(frac, 0.0, 1.0)

    @dispatch  # type: ignore[no-redef]
    def epistemic_frac(self, task: Task):  # noqa: F811
        """Fraction of total predictive variance attributable to epistemic uncertainty.

        Returned numpy arrays have shape ``(N_features, *N_targets)``, clipped to
        ``[0, 1]``.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            The task containing the context and target data.

        Returns
        -------
        numpy.ndarray | list[numpy.ndarray]
            Epistemic-variance fractions.

        """
        dist = self(task)
        return self.epistemic_frac(dist)

    @dispatch
    def epistemic_variance(self, dist: AbstractMultiOutputDistribution):
        """Epistemic component of the total predictive variance (model/normalized units).

        Returned numpy arrays have shape ``(N_features, *N_targets)``.
        """
        self._assert_mdn_zigamma("epistemic_variance")
        epvar = self._epistemic_frac_raw(dist) * dist.var
        epvar = self._cast_numpy_and_squeeze(epvar)
        return self._maybe_concat_multi_targets(epvar)

    @dispatch  # type: ignore[no-redef]
    def epistemic_variance(self, task: Task):  # noqa: F811
        """Epistemic component of the total predictive variance (model/normalized units).

        Returned numpy arrays have shape ``(N_features, *N_targets)``.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            The task containing the context and target data.

        Returns
        -------
        numpy.ndarray | list[numpy.ndarray]
            Epistemic variance values.

        """
        dist = self(task)
        return self.epistemic_variance(dist)

    @dispatch
    def p_zero(self, dist: AbstractMultiOutputDistribution):
        """Predicted probability mass at exactly zero.

        Returned numpy arrays have shape ``(N_features, *N_targets)``.
        """
        self._assert_mdn_zigamma("p_zero")
        p0 = self._cast_numpy_and_squeeze(B.exp(dist.logprobs[..., 0]))
        return self._maybe_concat_multi_targets(p0)

    @dispatch  # type: ignore[no-redef]
    def p_zero(self, task: Task):  # noqa: F811
        """Predicted probability mass at exactly zero.

        Returned numpy arrays have shape ``(N_features, *N_targets)``.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            The task containing the context and target data.

        Returns
        -------
        numpy.ndarray | list[numpy.ndarray]
            Probability mass at zero.

        """
        dist = self(task)
        return self.p_zero(dist)

    @dispatch
    def occurrence_entropy(self, dist: AbstractMultiOutputDistribution):
        """Bernoulli entropy of the predicted melt/no-melt occurrence probability.

        ``H(p_zero) = -p_zero*log(p_zero) - (1-p_zero)*log(1-p_zero)``: uncertainty in
        *whether* melt is occurring at all, independent of its magnitude given that it
        is. Complements ``epistemic_frac``/``epistemic_variance`` (magnitude
        uncertainty) with the uncertainty captured by the zero-inflation spike itself.

        Returned numpy arrays have shape ``(N_features, *N_targets)``.
        """
        p0 = self.p_zero(dist)
        if isinstance(p0, (list, tuple)):
            return [_bernoulli_entropy(p) for p in p0]
        return _bernoulli_entropy(p0)

    @dispatch  # type: ignore[no-redef]
    def occurrence_entropy(self, task: Task):  # noqa: F811
        """Bernoulli entropy of the predicted melt/no-melt occurrence probability.

        Returned numpy arrays have shape ``(N_features, *N_targets)``.

        Parameters
        ----------
        task : deepsensor.data.task.Task
            The task containing the context and target data.

        Returns
        -------
        numpy.ndarray | list[numpy.ndarray]
            Occurrence entropy values.

        """
        dist = self(task)
        return self.occurrence_entropy(dist)
