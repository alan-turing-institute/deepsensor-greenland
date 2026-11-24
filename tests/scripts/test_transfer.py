"""Tests for the transfer-learning helpers in ``ds_greenland.scripts.transfer``."""

import copy
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr
from deepsensor.data import TaskLoader
from omegaconf import OmegaConf, open_dict

import ds_greenland.scripts.transfer as tr
from ds_greenland.models import DeepSensorModel
from ds_greenland.utils.data import NumpyFloatValuesEncoder
from ds_greenland.utils.training import task_collate_fn, train_epoch_with_dataloader

UNET_DOWN = "decoder.links.0.before_turn_layers"


def _grid_ds(n_points: int, dates: pd.DatetimeIndex, n_vars: int = 1) -> xr.Dataset:
    """Build a random dataset on an `n_points` x `n_points` grid over [0, 1]^2."""
    coords = np.linspace(0, 1, n_points)
    rng = np.random.default_rng(1)
    return xr.Dataset(
        {
            "var" if i == 0 else f"var{i}": (
                ("time", "lat", "lon"),
                rng.normal(size=(len(dates), n_points, n_points)),
            )
            for i in range(n_vars)
        },
        coords={"time": dates, "lat": coords, "lon": coords},
    )


@pytest.fixture
def data_processor(tiny_dp_and_task_loader):
    """Copy the session-wide data processor, so tests can normalise new variables with it."""
    return copy.deepcopy(tiny_dp_and_task_loader[0])


@pytest.fixture
def pretrained_run(tmp_path, tiny_model_config, tiny_dp_and_task_loader):
    """Build a tiny model and save its config/weights the way `Trainer` saves them to W&B."""
    data_processor, task_loader, _ = tiny_dp_and_task_loader
    wrapper = DeepSensorModel(tiny_model_config)
    model = wrapper.build_model(data_processor, task_loader)
    with open(tmp_path / tr.MODEL_CONFIG_FILE, "w") as f:
        json.dump(model.config, f, cls=NumpyFloatValuesEncoder)
    wrapper.save_model(str(tmp_path / "weights.pt"))
    return tmp_path, model


def _rebuild(files_path, data_processor, task_loader):
    """Rebuild the pretrained model the way `transfer.main` does."""
    # The experiment's own `model` section, which the pretrained one replaces.
    config = OmegaConf.create(
        {"model": {"unet_channels": [8, 8, 8], "likelihood": "gnp"}}
    )
    with open_dict(config):
        config["model"] = tr.load_pretrained_model_config(
            str(files_path / tr.MODEL_CONFIG_FILE)
        )
    wrapper = DeepSensorModel(config)
    model = wrapper.build_model(data_processor, task_loader)
    wrapper.load_model(str(files_path / "weights.pt"))
    return model


class TestRebuildPretrainedModel:
    def test_same_architecture_and_weights(
        self, pretrained_run, tiny_dp_and_task_loader
    ):
        files_path, pretrained = pretrained_run
        data_processor, task_loader, _ = tiny_dp_and_task_loader

        model = _rebuild(files_path, data_processor, task_loader)

        saved = json.loads(json.dumps(pretrained.config, cls=NumpyFloatValuesEncoder))
        assert model.config == saved
        pretrained_state = pretrained.model.state_dict()
        for name, value in model.model.state_dict().items():
            assert torch.equal(value, pretrained_state[name]), name

    def test_keeps_pretrained_internal_density_on_a_finer_grid(
        self, pretrained_run, data_processor, tiny_dp_and_task_loader
    ):
        files_path, pretrained = pretrained_run
        dates = tiny_dp_and_task_loader[2]
        fine_ds = data_processor(_grid_ds(17, dates))
        fine_task_loader = TaskLoader(context=fine_ds, target=fine_ds)

        model = _rebuild(files_path, data_processor, fine_task_loader)

        assert (
            tr.compute_greatest_data_density(fine_task_loader)
            > (pretrained.config["internal_density"])
        )
        assert model.config["internal_density"] == pretrained.config["internal_density"]


class TestLoadPretrainedModelConfig:
    def test_reads_json(self, tmp_path):
        path = tmp_path / tr.MODEL_CONFIG_FILE
        path.write_text(json.dumps({"likelihood": "het", "dim_yc": [1]}))

        assert tr.load_pretrained_model_config(str(path)) == {
            "likelihood": "het",
            "dim_yc": [1],
        }


class TestCheckTaskLoaderCompatible:
    def test_accepts_same_structure(self, pretrained_run, tiny_dp_and_task_loader):
        _, pretrained = pretrained_run
        _, task_loader, _ = tiny_dp_and_task_loader

        tr.check_task_loader_compatible(pretrained.config, task_loader)

    def test_accepts_same_structure_on_a_different_grid(
        self, pretrained_run, data_processor, tiny_dp_and_task_loader
    ):
        _, pretrained = pretrained_run
        dates = tiny_dp_and_task_loader[2]
        ds = data_processor(_grid_ds(9, dates))

        tr.check_task_loader_compatible(
            pretrained.config, TaskLoader(context=ds, target=ds)
        )

    def test_rejects_extra_context_set(
        self, pretrained_run, data_processor, tiny_dp_and_task_loader
    ):
        _, pretrained = pretrained_run
        dates = tiny_dp_and_task_loader[2]
        ds = data_processor(_grid_ds(5, dates))

        with pytest.raises(
            ValueError, match=r"dim_yc: pretrained \[1\], new dataset \[1, 1\]"
        ):
            tr.check_task_loader_compatible(
                pretrained.config, TaskLoader(context=[ds, ds], target=ds)
            )

    def test_rejects_extra_target_variable(
        self, pretrained_run, data_processor, tiny_dp_and_task_loader
    ):
        _, pretrained = pretrained_run
        dates = tiny_dp_and_task_loader[2]
        ds = data_processor(_grid_ds(5, dates, n_vars=2))

        with pytest.raises(ValueError, match="dim_yt: pretrained 1, new dataset 2"):
            tr.check_task_loader_compatible(
                pretrained.config, TaskLoader(context=ds["var"], target=ds)
            )


class TestFreezeParameters:
    def test_freezes_only_matching_parameters(self, pretrained_run):
        _, model = pretrained_run
        before = {n: p.requires_grad for n, p in model.model.named_parameters()}

        frozen = tr.freeze_parameters(model.model, [UNET_DOWN])

        assert frozen
        assert all(n.startswith(f"{UNET_DOWN}.") for n in frozen)
        for name, param in model.model.named_parameters():
            expected = False if name in frozen else before[name]
            assert param.requires_grad is expected, name

    def test_matches_whole_name_components_only(self, pretrained_run):
        _, model = pretrained_run

        with pytest.raises(ValueError, match="before_turn"):
            tr.freeze_parameters(model.model, ["decoder.links.0.before_turn"])

    def test_rejects_unknown_prefix_without_freezing_anything(self, pretrained_run):
        _, model = pretrained_run
        before = {n: p.requires_grad for n, p in model.model.named_parameters()}

        with pytest.raises(ValueError, match="no_such_layer"):
            tr.freeze_parameters(model.model, [UNET_DOWN, "no_such_layer"])

        after = {n: p.requires_grad for n, p in model.model.named_parameters()}
        assert after == before

    def test_frozen_parameters_survive_a_training_step(
        self, pretrained_run, tiny_dp_and_task_loader
    ):
        _, model = pretrained_run
        _, task_loader, dates = tiny_dp_and_task_loader
        frozen = set(tr.freeze_parameters(model.model, [UNET_DOWN]))
        before = {n: p.detach().clone() for n, p in model.model.named_parameters()}
        # Built over every parameter, with weight decay, as `Trainer` builds it.
        optimizer = torch.optim.Adam(
            model.model.parameters(), lr=0.1, weight_decay=0.0001
        )
        task = task_loader(dates[0], context_sampling=10, target_sampling="all")

        train_epoch_with_dataloader(model, [task_collate_fn([task])], optimizer)

        changed = {
            n
            for n, p in model.model.named_parameters()
            if not torch.equal(p, before[n])
        }
        assert changed
        assert not changed & frozen


class TestCheckTransferMethod:
    @pytest.mark.parametrize(
        ("method", "freeze"), [("full", []), ("full", [UNET_DOWN]), ("film", [])]
    )
    def test_accepts_valid_combinations(self, method, freeze):
        tr.check_transfer_method(method, freeze)

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="Unknown transfer method 'lora'"):
            tr.check_transfer_method("lora", [])

    def test_rejects_freeze_with_film(self):
        with pytest.raises(ValueError, match="only applies to method 'full'"):
            tr.check_transfer_method("film", [UNET_DOWN])


class TestPrepareForFineTuning:
    def test_full_leaves_the_model_as_it_is_by_default(self, pretrained_run):
        _, model = pretrained_run
        before = {n: p.requires_grad for n, p in model.model.named_parameters()}

        tr.prepare_for_fine_tuning(model.model, "full", [])

        after = {n: p.requires_grad for n, p in model.model.named_parameters()}
        assert after == before

    def test_full_freezes_given_prefixes(self, pretrained_run):
        _, model = pretrained_run
        before = {n: p.requires_grad for n, p in model.model.named_parameters()}

        tr.prepare_for_fine_tuning(model.model, "full", [UNET_DOWN])

        for name, param in model.model.named_parameters():
            expected = before[name] and not name.startswith(f"{UNET_DOWN}.")
            assert param.requires_grad is expected, name

    def test_film_trains_only_film_layers(self, pretrained_run):
        _, model = pretrained_run

        tr.prepare_for_fine_tuning(model.model, "film", [])

        trainable = {n for n, p in model.model.named_parameters() if p.requires_grad}
        assert trainable
        assert all(".film." in n for n in trainable)
        assert all(
            not p.requires_grad
            for n, p in model.model.named_parameters()
            if ".film." not in n
        )

    def test_film_rejects_freeze_before_changing_the_model(self, pretrained_run):
        _, model = pretrained_run
        before = {n: p.requires_grad for n, p in model.model.named_parameters()}

        with pytest.raises(ValueError, match="only applies to method 'full'"):
            tr.prepare_for_fine_tuning(model.model, "film", [UNET_DOWN])

        after = {n: p.requires_grad for n, p in model.model.named_parameters()}
        assert after == before

    def test_film_fine_tuned_checkpoint_loads_into_rebuilt_model(
        self, pretrained_run, tiny_dp_and_task_loader
    ):
        files_path, _ = pretrained_run
        data_processor, task_loader, dates = tiny_dp_and_task_loader
        model = _rebuild(files_path, data_processor, task_loader)
        tr.prepare_for_fine_tuning(model.model, "film", [])
        pretrained = {n: p.detach().clone() for n, p in model.model.named_parameters()}
        # Built over every parameter, with weight decay, as `Trainer` builds it.
        optimizer = torch.optim.Adam(
            model.model.parameters(), lr=0.1, weight_decay=0.0001
        )
        task = task_loader(dates[0], context_sampling=10, target_sampling="all")

        train_epoch_with_dataloader(model, [task_collate_fn([task])], optimizer)

        changed = {
            n
            for n, p in model.model.named_parameters()
            if not torch.equal(p, pretrained[n])
        }
        assert changed
        assert all(".film." in n for n in changed)

        # Saved as `Trainer` saves checkpoints, then loaded as `inference.py`/a
        # chained transfer would: into a model rebuilt from `model_config.json`.
        wrapper = DeepSensorModel(OmegaConf.create({"model": {}}))
        wrapper.model = model
        wrapper.save_model(str(files_path / "weights.pt"))
        reloaded = _rebuild(files_path, data_processor, task_loader)

        assert tr.mean_task_loss(reloaded, [task]) == pytest.approx(
            tr.mean_task_loss(model, [task]), rel=1e-5
        )


class TestMeanTaskLoss:
    def test_is_mean_of_task_losses_without_tracking_gradients(
        self, pretrained_run, tiny_dp_and_task_loader
    ):
        _, model = pretrained_run
        _, task_loader, dates = tiny_dp_and_task_loader
        tasks = [
            task_loader(
                date, context_sampling=10, target_sampling="all", seed_override=i
            )
            for i, date in enumerate(dates)
        ]
        expected = np.mean(
            [model.loss_fn(t, normalise=True).detach().cpu().numpy() for t in tasks]
        )
        model.model.zero_grad(set_to_none=True)

        loss = tr.mean_task_loss(model, tasks)

        assert isinstance(loss, float)
        assert loss == pytest.approx(expected)
        assert all(p.grad is None for p in model.model.parameters())


class TestFetchPretrainedFiles:
    RUN_REF = "entity/project/abc123"
    WEIGHTS = "best-weights.pt"
    FILES = (tr.MODEL_CONFIG_FILE, tr.DATA_PROCESSOR_CONFIG_FILE, WEIGHTS)

    def _make_run_dir(self, name, files):
        path = os.path.join("wandb", name, "files")
        os.makedirs(path)
        for f in files:
            open(os.path.join(path, f), "w").close()
        return path

    def test_reuses_newest_complete_local_copy(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._make_run_dir("run-20260101_000000-abc123", self.FILES)
        newest = self._make_run_dir("run-20260201_000000-abc123", self.FILES)

        def no_download(*args):
            raise AssertionError("should not download")

        monkeypatch.setattr(tr, "download_wandb_file", no_download)

        assert tr.fetch_pretrained_files(self.RUN_REF, self.WEIGHTS) == newest

    @pytest.mark.parametrize("local_files", [None, FILES[:2]])
    def test_downloads_when_missing_or_incomplete(
        self, tmp_path, monkeypatch, local_files
    ):
        monkeypatch.chdir(tmp_path)
        if local_files is not None:
            self._make_run_dir("run-20260101_000000-abc123", local_files)
        calls = []
        monkeypatch.setattr(tr, "download_wandb_file", lambda *args: calls.append(args))

        files_path = tr.fetch_pretrained_files(self.RUN_REF, self.WEIGHTS)

        root = os.path.join("wandb", "run-abc123")
        assert files_path == os.path.join(root, "files")
        assert calls == [(self.RUN_REF, root, f) for f in self.FILES]
