import json
import os
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch
from deepsensor.model.convnp import ConvNP
from omegaconf import OmegaConf

import giant_sp.utils.wandb_utils as wu
from giant_sp.models.mdn import MDNZIGammaConvNP, install_mdn_zigamma_likelihood
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


SOURCE_RUN_REF = "train-ent/train-proj/src123"
SOURCE_RUN_URL = "https://wandb.example/train-ent/train-proj/runs/src123"


@pytest.fixture
def fake_source_run(monkeypatch):
    """Make `wandb.Api().run(...)` return a fake training run; records summary writes."""
    summary_writes: list[dict] = []
    source_run = SimpleNamespace(
        entity="train-ent",
        id="src123",
        url=SOURCE_RUN_URL,
        summary=SimpleNamespace(update=summary_writes.append),
    )
    requested: list[str] = []

    def fake_api():
        def run(ref):
            requested.append(ref)
            return source_run

        return SimpleNamespace(run=run)

    monkeypatch.setattr(wu.wandb, "Api", fake_api)
    return SimpleNamespace(requested=requested, summary_writes=summary_writes)


class TestInitEvalRun:
    def test_links_to_and_groups_under_training_run(
        self, fake_wandb_init, fake_source_run
    ):
        wu.init_eval_run(
            SOURCE_RUN_REF,
            job_type="inference",
            project="eval-proj",
            run_config={"inference": {"val_set": "station"}},
        )

        assert fake_source_run.requested == [SOURCE_RUN_REF]
        (kwargs,) = fake_wandb_init
        assert kwargs["project"] == "eval-proj"
        assert kwargs["job_type"] == "inference"
        assert kwargs["group"] == "src123"
        assert SOURCE_RUN_URL in kwargs["notes"]
        assert kwargs["config"] == {
            "source_run": SOURCE_RUN_REF,
            "source_run_url": SOURCE_RUN_URL,
            "inference": {"val_set": "station"},
        }

    def test_defaults_to_training_run_entity(self, fake_wandb_init, fake_source_run):
        wu.init_eval_run(SOURCE_RUN_REF, job_type="inference", project="eval-proj")

        assert fake_wandb_init[0]["entity"] == "train-ent"

    def test_explicit_entity_overrides_training_run_entity(
        self, fake_wandb_init, fake_source_run
    ):
        wu.init_eval_run(
            SOURCE_RUN_REF, job_type="inference", project="eval-proj", entity="me"
        )

        assert fake_wandb_init[0]["entity"] == "me"

    def test_run_name_defaults_to_wandb_generated(
        self, fake_wandb_init, fake_source_run
    ):
        wu.init_eval_run(SOURCE_RUN_REF, job_type="inference", project="eval-proj")
        wu.init_eval_run(
            SOURCE_RUN_REF,
            job_type="placement_eval",
            project="eval-proj",
            name="OracleRMSE_seasonal_e50",
        )

        assert fake_wandb_init[0]["name"] is None
        assert fake_wandb_init[1]["name"] == "OracleRMSE_seasonal_e50"


class TestLinkEvalRunToSource:
    def test_writes_eval_run_url_to_training_run_summary(self, fake_source_run):
        eval_run = SimpleNamespace(id="ev456", url="https://wandb.example/ev456")

        wu.link_eval_run_to_source(SOURCE_RUN_REF, eval_run, job_type="inference")

        assert fake_source_run.requested == [SOURCE_RUN_REF]
        assert fake_source_run.summary_writes == [
            {"inference/ev456": "https://wandb.example/ev456"}
        ]

    def test_warns_instead_of_raising_when_write_fails(self, monkeypatch, caplog):
        # e.g. no write access to the training run's project.
        def denied_update(_):
            raise RuntimeError("permission denied")

        source_run = SimpleNamespace(summary=SimpleNamespace(update=denied_update))
        monkeypatch.setattr(
            wu.wandb, "Api", lambda: SimpleNamespace(run=lambda ref: source_run)
        )
        eval_run = SimpleNamespace(id="ev456", url="https://wandb.example/ev456")

        with caplog.at_level("WARNING"):
            wu.link_eval_run_to_source(SOURCE_RUN_REF, eval_run, job_type="inference")

        assert "Could not link inference run ev456" in caplog.text
        assert "permission denied" in caplog.text


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


def _save_and_reload(model, tmp_path, data_processor, task_loader):
    """Save `model` the same way as Trainer._setup_existing_wandb_run, then reload it."""
    config_path = tmp_path / wu.MODEL_CONFIG_FILE
    weights_path = tmp_path / "weights.pt"
    with open(config_path, "w") as f:
        json.dump(model.config, f, cls=NumpyFloatValuesEncoder)
    torch.save(model.model.state_dict(), weights_path)
    return wu.load_wandb_model(
        data_processor, task_loader, str(config_path), str(weights_path)
    )


def _assert_same_weights(original, loaded):
    original_state = original.model.state_dict()
    loaded_state = loaded.model.state_dict()
    assert original_state.keys() == loaded_state.keys()
    for key, value in original_state.items():
        torch.testing.assert_close(loaded_state[key], value.to(loaded_state[key]))


class TestLoadWandbModel:
    @pytest.mark.parametrize("likelihood", ["cnp", "gnp"])
    def test_default_likelihood_loads_as_plain_convnp(
        self, tmp_path, tiny_dp_and_task_loader, likelihood
    ):
        # Non-melt variables should get deepsensor's default ConvNP, not the
        # surface-melt-specific MDNZIGammaConvNP subclass.
        data_processor, task_loader, dates = tiny_dp_and_task_loader
        model = ConvNP(
            data_processor,
            task_loader,
            unet_channels=(4, 4),
            likelihood=likelihood,
            num_basis_functions=16,
        )

        loaded = _save_and_reload(model, tmp_path, data_processor, task_loader)

        assert type(loaded) is ConvNP
        assert loaded.config["num_basis_functions"] == 16
        _assert_same_weights(model, loaded)
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")
        np.testing.assert_allclose(loaded.mean(task), model.mean(task))

    def test_bernoulli_gamma_loads_as_mdn_zigamma_convnp_with_p_zero(
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

        loaded = _save_and_reload(model, tmp_path, data_processor, task_loader)

        assert isinstance(loaded, MDNZIGammaConvNP)
        _assert_same_weights(model, loaded)
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")
        p0 = loaded.p_zero(task)
        assert ((p0 >= 0) & (p0 <= 1)).all()

    def test_mdn_zigamma_loads_with_mixture_components(
        self, tmp_path, tiny_dp_and_task_loader
    ):
        data_processor, task_loader, _ = tiny_dp_and_task_loader
        install_mdn_zigamma_likelihood(2)
        model = MDNZIGammaConvNP(
            data_processor,
            task_loader,
            unet_channels=(4, 4),
            likelihood="mdn-zigamma",
            mixture_components=2,
        )

        loaded = _save_and_reload(model, tmp_path, data_processor, task_loader)

        assert isinstance(loaded, MDNZIGammaConvNP)
        assert loaded.config["mixture_components"] == 2
        _assert_same_weights(model, loaded)
