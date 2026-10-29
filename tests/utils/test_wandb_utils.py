import json
import os
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch
from deepsensor.model.convnp import ConvNP
from omegaconf import OmegaConf

import giant_sp.utils.wandb_utils as wu
from giant_sp.models.mdn import MDNZIGammaConvNP
from giant_sp.utils.data import NumpyFloatValuesEncoder


@pytest.fixture
def wandb_config():
    return OmegaConf.create(
        {
            "logging": {
                "wandb": {
                    "project": "proj",
                    "entity": "ent",
                    "name": "run-name",
                    "tags": ["a"],
                    "notes": "some notes",
                }
            },
            "model": {"likelihood": "cnp"},
            "training": {"num_epochs": 1},
            "data": {"source": "carra2"},
        }
    )


@pytest.fixture
def fake_wandb_init(monkeypatch):
    calls = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(name="run-name", url="https://wandb.example/run")

    monkeypatch.setattr(wu.wandb, "init", fake_init)
    return calls


class TestWandbEnabled:
    @pytest.mark.parametrize(
        ("logging", "expected"),
        [
            ({"wandb": {"enabled": True}}, True),
            ({"wandb": {"enabled": False}}, False),
            ({"wandb": {}}, False),
            ({}, False),
        ],
    )
    def test_reads_enabled_flag(self, logging, expected):
        config = OmegaConf.create({"logging": logging})
        assert wu.wandb_enabled(config) is expected


class TestInitWandbRun:
    def test_passes_logging_settings_and_config_sections(
        self, wandb_config, fake_wandb_init
    ):
        run = wu.init_wandb_run(wandb_config)

        assert run.name == "run-name"
        (kwargs,) = fake_wandb_init
        assert kwargs["project"] == "proj"
        assert kwargs["entity"] == "ent"
        assert kwargs["name"] == "run-name"
        assert kwargs["tags"] == ["a"]
        assert kwargs["notes"] == "some notes"
        assert kwargs["config"] == {
            "model": {"likelihood": "cnp"},
            "training": {"num_epochs": 1},
            "data": {"source": "carra2"},
        }

    def test_merges_extra_run_config(self, wandb_config, fake_wandb_init):
        wu.init_wandb_run(wandb_config, {"model_parameters": 42})

        assert fake_wandb_init[0]["config"]["model_parameters"] == 42

    def test_defaults_when_logging_settings_missing(
        self, wandb_config, fake_wandb_init
    ):
        wandb_config.logging.wandb = {}

        wu.init_wandb_run(wandb_config)

        kwargs = fake_wandb_init[0]
        assert kwargs["project"] == "deepsensor-spatial-interpolation"
        assert kwargs["entity"] is None
        assert kwargs["tags"] == []
        assert kwargs["notes"] == ""


class TestFetchRunFiles:
    FILES: ClassVar[list[str]] = [
        wu.MODEL_CONFIG_FILE,
        wu.DATA_PROCESSOR_CONFIG_FILE,
        "weights.pt",
    ]

    @pytest.fixture
    def downloads(self, monkeypatch):
        calls = []

        def fake_download(run_ref, files_path, file):
            calls.append((run_ref, files_path, file))
            os.makedirs(os.path.join(files_path, "files"), exist_ok=True)
            open(os.path.join(files_path, "files", file), "w").close()

        monkeypatch.setattr(wu, "download_wandb_file", fake_download)
        return calls

    def test_uses_local_run_dir_when_all_files_present(
        self, tmp_path, monkeypatch, downloads
    ):
        monkeypatch.chdir(tmp_path)
        files_dir = tmp_path / "wandb" / "run-20260101_000000-abc123" / "files"
        files_dir.mkdir(parents=True)
        for f in self.FILES:
            (files_dir / f).touch()

        files_path = wu.fetch_run_files("ent/proj/abc123", self.FILES)

        assert files_path == os.path.join(
            "wandb", "run-20260101_000000-abc123", "files"
        )
        assert downloads == []

    def test_downloads_when_no_local_run_dir(self, tmp_path, monkeypatch, downloads):
        monkeypatch.chdir(tmp_path)

        files_path = wu.fetch_run_files("ent/proj/abc123", self.FILES)

        root_path = os.path.join("wandb", "run-abc123")
        assert files_path == os.path.join(root_path, "files")
        assert downloads == [("ent/proj/abc123", root_path, f) for f in self.FILES]

    def test_downloads_when_local_run_dir_is_missing_a_file(
        self, tmp_path, monkeypatch, downloads
    ):
        monkeypatch.chdir(tmp_path)
        files_dir = tmp_path / "wandb" / "run-20260101_000000-abc123" / "files"
        files_dir.mkdir(parents=True)
        (files_dir / wu.MODEL_CONFIG_FILE).touch()

        files_path = wu.fetch_run_files("ent/proj/abc123", self.FILES)

        assert files_path == os.path.join("wandb", "run-abc123", "files")
        assert [f for _, _, f in downloads] == self.FILES


class TestLoadWandbModel:
    def test_round_trips_saved_config_and_weights(
        self, tmp_path, tiny_dp_and_task_loader
    ):
        data_processor, task_loader, _ = tiny_dp_and_task_loader
        model = ConvNP(
            data_processor,
            task_loader,
            unet_channels=(4, 4),
            likelihood="cnp",
            num_basis_functions=16,
        )
        config_path = tmp_path / wu.MODEL_CONFIG_FILE
        weights_path = tmp_path / "weights.pt"
        # Saved the same way as Trainer._setup_existing_wandb_run.
        with open(config_path, "w") as f:
            json.dump(model.config, f, cls=NumpyFloatValuesEncoder)
        torch.save(model.model.state_dict(), weights_path)

        loaded = wu.load_wandb_model(
            data_processor, task_loader, str(config_path), str(weights_path)
        )

        assert loaded.config["num_basis_functions"] == 16
        original_state = model.model.state_dict()
        loaded_state = loaded.model.state_dict()
        assert original_state.keys() == loaded_state.keys()
        for key, value in original_state.items():
            torch.testing.assert_close(loaded_state[key], value.to(loaded_state[key]))

    def test_bernoulli_gamma_model_exposes_p_zero(
        self, tmp_path, tiny_dp_and_task_loader
    ):
        # placement.py's occurrence-entropy acquisition functions call `p_zero` on
        # bernoulli-gamma checkpoints, which only `MDNZIGammaConvNP` provides.
        data_processor, task_loader, dates = tiny_dp_and_task_loader
        model = ConvNP(
            data_processor,
            task_loader,
            unet_channels=(4, 4),
            likelihood="bernoulli-gamma",
        )
        config_path = tmp_path / wu.MODEL_CONFIG_FILE
        weights_path = tmp_path / "weights.pt"
        with open(config_path, "w") as f:
            json.dump(model.config, f, cls=NumpyFloatValuesEncoder)
        torch.save(model.model.state_dict(), weights_path)

        loaded = wu.load_wandb_model(
            data_processor, task_loader, str(config_path), str(weights_path)
        )

        assert isinstance(loaded, MDNZIGammaConvNP)
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")
        p0 = loaded.p_zero(task)
        assert ((p0 >= 0) & (p0 <= 1)).all()
