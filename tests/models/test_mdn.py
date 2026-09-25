from typing import Any

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats
import torch
import xarray as xr
from deepsensor.data import DataProcessor, TaskLoader

from giant_sp.models.mdn import (
    GammaMixture,
    MDNZIGammaConvNP,
    _bernoulli_entropy,
    install_mdn_zigamma_likelihood,
)
from giant_sp.utils.metrics import compute_pit_zigamma_mixture


@pytest.fixture(scope="module")
def tiny_zigamma_dp_and_task_loader():
    """Non-negative, zero-inflated analogue of `tiny_dp_and_task_loader`.

    `"mdn-zigamma"` (like the existing `"bernoulli-gamma"` likelihood it generalizes)
    is only valid for non-negative targets, so it needs its own synthetic dataset --
    the shared `tiny_dp_and_task_loader` fixture uses plain `rng.normal` values
    (including negatives), which would give a `Gamma` log-density of `nan`.
    """
    lat = np.linspace(0, 1, 5)
    lon = np.linspace(0, 1, 5)
    dates = pd.date_range("2020-01-01", periods=3)
    rng = np.random.default_rng(0)
    values = rng.gamma(shape=2.0, scale=1.0, size=(len(dates), len(lat), len(lon)))
    # Zero-inflate ~30% of points, matching the melt-like data this likelihood targets.
    values[rng.uniform(size=values.shape) < 0.3] = 0.0
    da = xr.DataArray(
        values,
        coords={"time": dates, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="var",
    )
    ds = da.to_dataset()

    data_processor = DataProcessor(x1_name="lat", x2_name="lon", verbose=False)
    # "positive_semidefinite" normalises to min=0 (unlike "min_max"'s min=-1), keeping
    # the exact-zero melt-like points at exactly zero after normalisation, matching
    # what this likelihood's zero-inflation spike needs.
    ds_norm = data_processor(ds, method="positive_semidefinite")
    task_loader = TaskLoader(context=ds_norm, target=ds_norm)
    return data_processor, task_loader, dates, ds


def _brute_force_moments(probs, k, scale):
    """Compute reference mean/var/epistemic/aleatoric via plain numpy, no `lab`/torch ops."""
    comp_mean = k * scale
    comp_var = k * scale**2
    mean = np.sum(probs * comp_mean, axis=-1)
    aleatoric = np.sum(probs * comp_var, axis=-1)
    epistemic = np.sum(probs * (comp_mean - mean[..., None]) ** 2, axis=-1)
    return mean, aleatoric, epistemic


class TestBernoulliEntropy:
    def test_no_warning_for_exact_zero_or_one_float32(self, recwarn):
        # Regression test: `1 - 1e-12` rounds to exactly 1.0 in float32, so clipping a
        # float32 array against it (without first upcasting) silently fails to clip
        # p==1.0, and log(1 - 1.0) raises "RuntimeWarning: divide by zero in log".
        p = np.array([0.0, 1.0, 0.5], dtype=np.float32)

        entropy = _bernoulli_entropy(p)

        assert len(recwarn) == 0
        assert np.all(np.isfinite(entropy))

    def test_matches_analytic_bernoulli_entropy(self):
        p = np.array([0.1, 0.5, 0.9])
        entropy = _bernoulli_entropy(p)
        expected = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        np.testing.assert_allclose(entropy, expected, rtol=1e-6)


class TestGammaMixtureMoments:
    def test_matches_brute_force_reference(self):
        rng = np.random.default_rng(0)
        batch, dim_y, K, spatial = 2, 1, 3, 4
        # Tensors shaped (batch, dim_y, K, spatial) -- K sits at axis -d-1 with d=2
        # (dim_y + spatial), matching the `ZIGammaMixtureLikelihood` channel layout.
        logits_np = rng.normal(size=(batch, dim_y, K, spatial))
        k_np = rng.uniform(0.5, 3.0, size=(batch, dim_y, K, spatial))
        scale_np = rng.uniform(0.5, 3.0, size=(batch, dim_y, K, spatial))

        logits = torch.tensor(logits_np)
        k = torch.tensor(k_np)
        scale = torch.tensor(scale_np)

        dist = GammaMixture(logits, k, scale, d=2)

        probs_np = np.exp(logits_np) / np.exp(logits_np).sum(axis=2, keepdims=True)
        # Move K to the last axis to match `_brute_force_moments`'s convention.
        probs_ref = np.moveaxis(probs_np, 2, -1)
        k_ref = np.moveaxis(k_np, 2, -1)
        scale_ref = np.moveaxis(scale_np, 2, -1)
        mean_ref, aleatoric_ref, epistemic_ref = _brute_force_moments(
            probs_ref, k_ref, scale_ref
        )

        np.testing.assert_allclose(dist.mean.numpy(), mean_ref, rtol=1e-6)
        np.testing.assert_allclose(dist.aleatoric_var.numpy(), aleatoric_ref, rtol=1e-6)
        np.testing.assert_allclose(dist.epistemic_var.numpy(), epistemic_ref, rtol=1e-6)
        np.testing.assert_allclose(
            dist.var.numpy(), (aleatoric_ref + epistemic_ref), rtol=1e-6
        )

    def test_degenerate_k1_matches_plain_gamma(self):
        rng = np.random.default_rng(1)
        batch, dim_y, K, spatial = 2, 1, 1, 3
        logits = torch.tensor(rng.normal(size=(batch, dim_y, K, spatial)))
        k = torch.tensor(rng.uniform(0.5, 3.0, size=(batch, dim_y, K, spatial)))
        scale = torch.tensor(rng.uniform(0.5, 3.0, size=(batch, dim_y, K, spatial)))

        dist = GammaMixture(logits, k, scale, d=2)

        expected_mean = (k * scale).squeeze(2)
        expected_var = (k * scale**2).squeeze(2)

        np.testing.assert_allclose(dist.mean.numpy(), expected_mean.numpy(), rtol=1e-6)
        np.testing.assert_allclose(dist.var.numpy(), expected_var.numpy(), rtol=1e-6)
        np.testing.assert_allclose(
            dist.epistemic_var.numpy(), np.zeros_like(expected_mean.numpy()), atol=1e-8
        )


class TestGammaMixtureLogpdf:
    def test_matches_scipy_mixture_logpdf(self):
        rng = np.random.default_rng(2)
        dim_y, K, spatial = 1, 3, 5
        logits_np = rng.normal(size=(dim_y, K, spatial))
        k_np = rng.uniform(0.5, 3.0, size=(dim_y, K, spatial))
        scale_np = rng.uniform(0.5, 3.0, size=(dim_y, K, spatial))
        x_np = rng.uniform(0.1, 5.0, size=(dim_y, spatial))

        dist = GammaMixture(
            torch.tensor(logits_np), torch.tensor(k_np), torch.tensor(scale_np), d=2
        )
        logpdf = dist.logpdf(torch.tensor(x_np)).numpy()

        probs_np = np.exp(logits_np) / np.exp(logits_np).sum(axis=1, keepdims=True)
        # Reference: sum over dim_y+spatial of log(sum_k pi_k * gamma_pdf_k(x)).
        pdf_per_point = np.zeros((dim_y, spatial))
        for i in range(dim_y):
            for s in range(spatial):
                mix_pdf = sum(
                    probs_np[i, kk, s]
                    * stats.gamma.pdf(
                        x_np[i, s], a=k_np[i, kk, s], scale=scale_np[i, kk, s]
                    )
                    for kk in range(K)
                )
                pdf_per_point[i, s] = mix_pdf
        expected = np.log(pdf_per_point).sum()

        np.testing.assert_allclose(logpdf, expected, rtol=1e-5)


class TestEpistemicFracRaw:
    """Unit tests for `MDNZIGammaConvNP._epistemic_frac_raw`'s two-level decomposition.

    Uses `MDNZIGammaConvNP.__new__` (skipping `__init__`) since `_epistemic_frac_raw`
    is a pure function of its `dist` argument and doesn't touch instance state -- no
    need to build a real model just to test it.
    """

    @staticmethod
    def _make_dist(
        p_slab: float, k: np.ndarray, scale: np.ndarray, pi: np.ndarray
    ) -> Any:
        """Build a `SpikesSlab(spike=0, slab=GammaMixture(...))` with given params."""
        from neuralprocesses.dist.spikeslab import SpikesSlab

        K = len(k)
        logits = torch.log(torch.tensor(pi, dtype=torch.float64)).reshape(1, K)
        k_t = torch.tensor(k, dtype=torch.float64).reshape(1, K)
        scale_t = torch.tensor(scale, dtype=torch.float64).reshape(1, K)
        slab = GammaMixture(logits, k_t, scale_t, d=1)

        logp0 = torch.log(torch.tensor([1 - p_slab], dtype=torch.float64))
        logps = torch.log(torch.tensor([p_slab], dtype=torch.float64))
        logprobs = torch.stack([logp0, logps], dim=-1)
        return SpikesSlab(torch.tensor([0.0]), slab, logprobs, 1)

    def _flat_reference(self, p_slab, k, scale, pi):
        """Compute an independent reference by flattening the two mixture levels.

        Treats {spike, K gamma components} as one flat (K+1)-component mixture and
        applies the textbook law of total variance once, rather than composing the two
        nested mixture levels as the implementation does -- an independent check of
        the same underlying identity.
        """
        means = np.concatenate([[0.0], k * scale])
        variances = np.concatenate([[0.0], k * scale**2])
        weights = np.concatenate([[1 - p_slab], p_slab * pi])
        grand_mean = np.sum(weights * means)
        epistemic = np.sum(weights * (means - grand_mean) ** 2)
        aleatoric = np.sum(weights * variances)
        return epistemic, aleatoric

    def test_matches_flat_mixture_reference(self):
        rng = np.random.default_rng(0)
        k = rng.uniform(0.5, 3.0, size=3)
        scale = rng.uniform(0.5, 3.0, size=3)
        logits = rng.normal(size=3)
        pi = np.exp(logits) / np.exp(logits).sum()
        p_slab = 0.4

        dist = self._make_dist(p_slab, k, scale, pi)
        model = MDNZIGammaConvNP.__new__(MDNZIGammaConvNP)
        frac = model._epistemic_frac_raw(dist).item()

        epistemic_ref, aleatoric_ref = self._flat_reference(p_slab, k, scale, pi)
        expected_frac = epistemic_ref / (epistemic_ref + aleatoric_ref)
        assert frac == pytest.approx(expected_frac, rel=1e-6)

    def test_outer_uncertainty_counted_even_with_single_component(self):
        # Regression test: with K=1 (a single Gamma component), the *inner* epistemic
        # variance is exactly 0 (no component disagreement possible) -- an earlier,
        # incomplete version of `_epistemic_frac_raw` returned epistemic_frac == 0 in
        # this case regardless of `p_slab`, silently folding all "is this location
        # zero-inflated or not" uncertainty into "aleatoric". At p_slab=0.5 (maximal
        # uncertainty about whether the point is in the zero-spike or the slab), the
        # fixed formula must report nonzero epistemic uncertainty.
        dist = self._make_dist(
            p_slab=0.5, k=np.array([2.0]), scale=np.array([1.0]), pi=np.array([1.0])
        )
        model = MDNZIGammaConvNP.__new__(MDNZIGammaConvNP)

        frac = model._epistemic_frac_raw(dist).item()

        assert frac > 0.0

    def test_epistemic_variance_vanishes_as_p_slab_approaches_zero(self):
        # As p_slab -> 0 (the model is confident the point is always exactly zero),
        # the *absolute* epistemic variance (not the fraction -- total variance
        # vanishes too, so the fraction need not) must shrink to ~0, since every term
        # in `outer_epistemic + inner_epistemic` is proportional to `p_slab`.
        k = np.array([1.0, 5.0])
        scale = np.array([1.0, 1.0])
        pi = np.array([0.5, 0.5])
        dist = self._make_dist(1e-6, k, scale, pi)

        p_slab = torch.exp(dist.logprobs[..., -1])
        epistemic_var = (
            p_slab * (1 - p_slab) * dist.slab.mean**2 + p_slab * dist.slab.epistemic_var
        )

        assert epistemic_var.item() < 1e-4


class TestMDNZIGammaConvNPEndToEnd:
    def test_build_loss_and_predict(self, tiny_zigamma_dp_and_task_loader):
        data_processor, task_loader, dates, ds = tiny_zigamma_dp_and_task_loader
        K = 2
        install_mdn_zigamma_likelihood(K)

        config = {
            "unet_channels": (4, 4),
            "likelihood": "mdn-zigamma",
            "mixture_components": K,
        }
        model = MDNZIGammaConvNP(data_processor, task_loader, **config)

        task = task_loader(dates[0], context_sampling="all", target_sampling="all")

        loss = model.loss_fn(task, normalise=True)
        assert torch.isfinite(loss)

        preds = model.predict(
            [task],
            X_t=ds,
            pred_params=["mean", "std", "mixture_probs", "epistemic_frac"],
        )
        var_name = next(iter(task_loader.target_var_IDs[0]))
        epistemic_frac = preds[var_name]["epistemic_frac"].data
        assert np.all(epistemic_frac >= 0.0)
        assert np.all(epistemic_frac <= 1.0)
        assert np.all(np.isfinite(preds[var_name]["mean"].data))
        assert np.all(preds[var_name]["std"].data >= 0.0)

    def test_epistemic_variance_task_level(self, tiny_zigamma_dp_and_task_loader):
        data_processor, task_loader, dates, _ds = tiny_zigamma_dp_and_task_loader
        K = 2
        install_mdn_zigamma_likelihood(K)
        model = MDNZIGammaConvNP(
            data_processor,
            task_loader,
            unet_channels=(4, 4),
            likelihood="mdn-zigamma",
            mixture_components=K,
        )
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")

        epvar = model.epistemic_variance(task)
        total_var = model.variance(task)
        if isinstance(epvar, list):
            epvar, total_var = epvar[0], total_var[0]

        assert np.all(epvar >= -1e-6)
        assert np.all(epvar <= total_var + 1e-6)

    def test_occurrence_entropy_task_level(self, tiny_zigamma_dp_and_task_loader):
        data_processor, task_loader, dates, _ds = tiny_zigamma_dp_and_task_loader
        K = 2
        install_mdn_zigamma_likelihood(K)
        model = MDNZIGammaConvNP(
            data_processor,
            task_loader,
            unet_channels=(4, 4),
            likelihood="mdn-zigamma",
            mixture_components=K,
        )
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")

        entropy = model.occurrence_entropy(task)
        p0 = model.p_zero(task)
        if isinstance(entropy, list):
            entropy, p0 = entropy[0], p0[0]

        # Bernoulli entropy is bounded in [0, log(2)] and matches -p*log(p)-(1-p)*log(1-p).
        assert np.all(entropy >= 0.0)
        assert np.all(entropy <= np.log(2) + 1e-6)
        expected = -(p0 * np.log(p0) + (1 - p0) * np.log(1 - p0))
        np.testing.assert_allclose(entropy, expected, atol=1e-6)

    def test_magnitude_variance_task_level(self, tiny_zigamma_dp_and_task_loader):
        data_processor, task_loader, dates, _ds = tiny_zigamma_dp_and_task_loader
        K = 2
        install_mdn_zigamma_likelihood(K)
        model = MDNZIGammaConvNP(
            data_processor,
            task_loader,
            unet_channels=(4, 4),
            likelihood="mdn-zigamma",
            mixture_components=K,
        )
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")

        magvar = model.magnitude_variance(task)
        epvar = model.epistemic_variance(task)
        p_zero = model.p_zero(task)
        if isinstance(magvar, list):
            magvar, epvar, p_zero = magvar[0], epvar[0], p_zero[0]

        # magnitude_variance is `_epistemic_frac_raw`'s "inner" term unweighted by
        # `p_slab`; its `p_slab`-weighted contribution can never exceed the combined
        # (occurrence + magnitude) epistemic_variance it's a non-negative summand of.
        p_slab = 1 - p_zero
        assert np.all(magvar >= -1e-6)
        assert np.all(p_slab * magvar <= epvar + 1e-6)

    def test_magnitude_mean_task_level(self, tiny_zigamma_dp_and_task_loader):
        data_processor, task_loader, dates, _ds = tiny_zigamma_dp_and_task_loader
        K = 2
        install_mdn_zigamma_likelihood(K)
        model = MDNZIGammaConvNP(
            data_processor,
            task_loader,
            unet_channels=(4, 4),
            likelihood="mdn-zigamma",
            mixture_components=K,
        )
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")

        magmean = model.magnitude_mean(task)
        p_zero = model.p_zero(task)
        mean = model.mean(task)
        if isinstance(magmean, list):
            magmean, p_zero, mean = magmean[0], p_zero[0], mean[0]

        # magnitude_mean is the slab-only mean (conditional on occurrence); the overall
        # predictive mean is the zero-spike-weighted average of it and 0.
        assert np.all(magmean >= -1e-6)
        np.testing.assert_allclose(mean, (1 - p_zero) * magmean, atol=1e-5)

    def test_slab_component_params_and_pit(self, tiny_zigamma_dp_and_task_loader):
        """Verify the per-component `pred_params` accessors and PIT computation.

        `install_mdn_zigamma_likelihood` attaches per-component `pred_params`
        accessors (`slab_probs_i`/`slab_k_i`/`slab_scale_i`) that plug directly into
        `compute_pit_zigamma_mixture`, matching how `k`/`scale` already do for
        `"bernoulli-gamma"` and `compute_pit_bernoulli_gamma`.
        """
        data_processor, task_loader, dates, ds = tiny_zigamma_dp_and_task_loader
        K = 2
        install_mdn_zigamma_likelihood(K)
        model = MDNZIGammaConvNP(
            data_processor,
            task_loader,
            unet_channels=(4, 4),
            likelihood="mdn-zigamma",
            mixture_components=K,
        )
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")

        pred_params = ["mean", "std", "mixture_probs"]
        for component in range(K):
            pred_params += [
                f"slab_probs_{component}",
                f"slab_k_{component}",
                f"slab_scale_{component}",
            ]
        preds = model.predict([task], X_t=ds, pred_params=pred_params)
        var_name = next(iter(task_loader.target_var_IDs[0]))
        pred_ds = preds[var_name]

        probs = np.stack([pred_ds[f"slab_probs_{i}"].data for i in range(K)], axis=0)
        k = np.stack([pred_ds[f"slab_k_{i}"].data for i in range(K)], axis=0)
        scale = np.stack([pred_ds[f"slab_scale_{i}"].data for i in range(K)], axis=0)

        assert probs.shape == k.shape == scale.shape
        np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-5)
        assert np.all(k > 0.0)
        assert np.all(scale > 0.0)

        true = ds[var_name].sel(time=dates[0]).values.flatten()
        p_zero = pred_ds["mixture_probs_0"].data.flatten()
        pit = compute_pit_zigamma_mixture(
            true,
            p_zero,
            probs.reshape(K, -1),
            k.reshape(K, -1),
            scale.reshape(K, -1),
        )
        assert np.all(pit >= 0.0)
        assert np.all(pit <= 1.0)

    def test_wrong_likelihood_raises(self, tiny_zigamma_dp_and_task_loader):
        data_processor, task_loader, dates, _ds = tiny_zigamma_dp_and_task_loader
        model = MDNZIGammaConvNP(
            data_processor, task_loader, unet_channels=(4, 4), likelihood="cnp"
        )
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")
        with pytest.raises(NotImplementedError):
            model.epistemic_frac(task)
        with pytest.raises(NotImplementedError):
            model.occurrence_entropy(task)
