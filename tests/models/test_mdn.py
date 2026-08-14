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
    install_mdn_zigamma_likelihood,
)


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

    def test_wrong_likelihood_raises(self, tiny_zigamma_dp_and_task_loader):
        data_processor, task_loader, dates, _ds = tiny_zigamma_dp_and_task_loader
        model = MDNZIGammaConvNP(
            data_processor, task_loader, unet_channels=(4, 4), likelihood="cnp"
        )
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")
        with pytest.raises(NotImplementedError):
            model.epistemic_frac(task)
