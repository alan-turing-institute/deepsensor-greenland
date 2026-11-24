"""Tests for the FiLM adapters in ``ds_greenland.models.film``."""

from typing import Any

import pytest
import torch
from deepsensor.data import Task
from deepsensor.model import ConvNP
from torch.nn.modules.conv import _ConvNd

from ds_greenland.models import DeepSensorModel
from ds_greenland.models.film import (
    FiLM,
    add_film_layers,
    folded_state_dict,
    load_folded_state_dict,
)

UNET = "decoder.links.0"
# The convolution of each layer of a 2-layer UNet (`unet_channels=(4, 4)`); the
# second layer of each path wraps its convolution with pooling/upsampling.
UNET_CONVS = {
    True: [  # resize convolutions: Upsample, then Conv2d
        f"{UNET}.before_turn_layers.0",
        f"{UNET}.before_turn_layers.1.0",
        f"{UNET}.after_turn_layers.0",
        f"{UNET}.after_turn_layers.1.1",
    ],
    False: [  # transposed convolutions
        f"{UNET}.before_turn_layers.0",
        f"{UNET}.before_turn_layers.1.0",
        f"{UNET}.after_turn_layers.0",
        f"{UNET}.after_turn_layers.1",
    ],
}


def _build(tiny_dp_and_task_loader: Any, **kwargs: Any) -> ConvNP:
    data_processor, task_loader, _ = tiny_dp_and_task_loader
    return ConvNP(
        data_processor,
        task_loader,
        unet_channels=(4, 4),
        likelihood="cnp",
        verbose=False,
        **kwargs,
    )


def _task(tiny_dp_and_task_loader):
    _, task_loader, dates = tiny_dp_and_task_loader
    return task_loader(
        dates[0], context_sampling=10, target_sampling="all", seed_override=0
    )


def _loss(model: ConvNP, task: Task) -> float:
    with torch.no_grad():
        return float(model.loss_fn(task, normalise=True))


def _films(module: torch.nn.Module) -> list[FiLM]:
    return [m for m in module.modules() if isinstance(m, FiLM)]


def _randomise_films(module: torch.nn.Module, seed: int = 0) -> None:
    """Move every FiLM layer away from the identity."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for film in _films(module):
            for param in film.parameters():
                param.copy_(0.5 * torch.randn(param.shape, generator=generator))


class TestFiLM:
    def test_starts_at_identity(self):
        x = torch.randn(2, 3, 4, 5)

        torch.testing.assert_close(FiLM(3)(x), x)

    def test_scales_and_shifts_each_channel(self):
        film = FiLM(2)
        with torch.no_grad():
            film.scale_delta.copy_(torch.tensor([1.0, -0.5]))
            film.shift.copy_(torch.tensor([0.0, 3.0]))
        x = torch.ones(1, 2, 3, 3)

        out = film(x)

        torch.testing.assert_close(out[:, 0], torch.full((1, 3, 3), 2.0))
        torch.testing.assert_close(out[:, 1], torch.full((1, 3, 3), 3.5))


class TestAddFilmLayers:
    @pytest.mark.parametrize("resize_convs", [True, False])
    def test_adds_a_film_layer_after_every_unet_convolution(
        self, tiny_dp_and_task_loader, resize_convs
    ):
        model = _build(tiny_dp_and_task_loader, unet_resize_convs=resize_convs)

        names = add_film_layers(model.model)

        assert names == UNET_CONVS[resize_convs]
        modules = dict(model.model.named_modules())
        for name in names:
            conv = modules[name]
            assert isinstance(conv, _ConvNd)
            assert isinstance(conv.film, FiLM)
            assert conv.film.shift.shape == (conv.out_channels,)
        assert not hasattr(modules[f"{UNET}.final_linear"], "film")
        film_params = {n for n, _ in model.model.named_parameters() if ".film." in n}
        assert film_params == {
            f"{name}.film.{p}" for name in names for p in ("scale_delta", "shift")
        }

    def test_leaves_the_output_unchanged(self, tiny_dp_and_task_loader):
        model = _build(tiny_dp_and_task_loader)
        task = _task(tiny_dp_and_task_loader)
        before = _loss(model, task)

        add_film_layers(model.model)

        assert _loss(model, task) == pytest.approx(before, rel=1e-6)

    def test_film_layers_change_the_output(self, tiny_dp_and_task_loader):
        model = _build(tiny_dp_and_task_loader)
        task = _task(tiny_dp_and_task_loader)
        before = _loss(model, task)
        add_film_layers(model.model)

        _randomise_films(model.model)

        assert _loss(model, task) != pytest.approx(before, rel=1e-3)

    def test_rejects_a_model_without_a_unet(self):
        module = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 3))

        with pytest.raises(ValueError, match="needs a UNet"):
            add_film_layers(module)

    def test_rejects_a_model_that_already_has_film_layers(
        self, tiny_dp_and_task_loader
    ):
        model = _build(tiny_dp_and_task_loader)
        add_film_layers(model.model)

        with pytest.raises(ValueError, match="already has FiLM layers"):
            add_film_layers(model.model)

    def test_rejects_a_separable_unet_without_changing_it(
        self, tiny_dp_and_task_loader
    ):
        model = _build(tiny_dp_and_task_loader, conv_arch="unet-sep")
        before = set(model.model.state_dict())

        with pytest.raises(ValueError, match="plain UNets"):
            add_film_layers(model.model)

        assert set(model.model.state_dict()) == before


class TestFoldedStateDict:
    @pytest.mark.parametrize("resize_convs", [True, False])
    def test_plain_model_loaded_with_it_matches_the_film_model(
        self, tiny_dp_and_task_loader, resize_convs
    ):
        model = _build(tiny_dp_and_task_loader, unet_resize_convs=resize_convs)
        add_film_layers(model.model)
        _randomise_films(model.model)
        task = _task(tiny_dp_and_task_loader)

        plain = _build(tiny_dp_and_task_loader, unet_resize_convs=resize_convs)
        plain.model.load_state_dict(folded_state_dict(model.model))  # strict

        assert _loss(plain, task) == pytest.approx(_loss(model, task), rel=1e-5)

    def test_is_the_state_dict_of_a_model_without_film_layers(
        self, tiny_dp_and_task_loader
    ):
        model = _build(tiny_dp_and_task_loader)
        expected = model.model.state_dict()

        state = folded_state_dict(model.model)

        assert state.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(state[key], value)

    def test_leaves_the_model_unchanged(self, tiny_dp_and_task_loader):
        model = _build(tiny_dp_and_task_loader)
        add_film_layers(model.model)
        _randomise_films(model.model)
        before = {k: v.clone() for k, v in model.model.state_dict().items()}

        folded_state_dict(model.model)

        after = model.model.state_dict()
        assert after.keys() == before.keys()
        for key, value in before.items():
            torch.testing.assert_close(after[key], value)


class TestLoadFoldedStateDict:
    def test_loads_folded_weights_and_resets_film_layers(self, tiny_dp_and_task_loader):
        source = _build(tiny_dp_and_task_loader)
        add_film_layers(source.model)
        _randomise_films(source.model, seed=0)
        task = _task(tiny_dp_and_task_loader)

        target = _build(tiny_dp_and_task_loader)
        add_film_layers(target.model)
        _randomise_films(target.model, seed=1)
        load_folded_state_dict(target.model, folded_state_dict(source.model))

        for film in _films(target.model):
            assert not film.scale_delta.any()
            assert not film.shift.any()
        assert _loss(target, task) == pytest.approx(_loss(source, task), rel=1e-5)

    def test_loads_film_parameters_of_an_unfolded_state_dict(
        self, tiny_dp_and_task_loader
    ):
        source = _build(tiny_dp_and_task_loader)
        add_film_layers(source.model)
        _randomise_films(source.model)
        target = _build(tiny_dp_and_task_loader)
        add_film_layers(target.model)

        load_folded_state_dict(target.model, source.model.state_dict())

        for source_film, target_film in zip(
            _films(source.model), _films(target.model), strict=True
        ):
            torch.testing.assert_close(target_film.shift, source_film.shift)
            torch.testing.assert_close(target_film.scale_delta, source_film.scale_delta)

    def test_is_strict_for_a_model_without_film_layers(self, tiny_dp_and_task_loader):
        model = _build(tiny_dp_and_task_loader)
        state = model.model.state_dict()
        del state[next(iter(state))]

        with pytest.raises(RuntimeError, match="Missing key"):
            load_folded_state_dict(model.model, state)


class TestDeepSensorModelCheckpoints:
    def test_film_model_checkpoint_loads_into_a_plain_model(
        self, tiny_model_config, tiny_dp_and_task_loader, tmp_path
    ):
        data_processor, task_loader, _ = tiny_dp_and_task_loader
        task = _task(tiny_dp_and_task_loader)
        source = DeepSensorModel(tiny_model_config)
        source_model = source.build_model(data_processor, task_loader)
        add_film_layers(source_model.model)
        _randomise_films(source_model.model)
        filepath = str(tmp_path / "model.pt")

        source.save_model(filepath)
        target = DeepSensorModel(tiny_model_config)
        target_model = target.build_model(data_processor, task_loader)
        target.load_model(filepath)

        assert _loss(target_model, task) == pytest.approx(
            _loss(source_model, task), rel=1e-5
        )
