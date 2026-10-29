import json
import os
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import giant_sp.training.trainer as trainer_module
import giant_sp.utils.wandb_utils as wu
from giant_sp.models.model import DeepSensorModel
from giant_sp.training import Trainer


@pytest.fixture
def trainer_config():
    return OmegaConf.create(
        {
            "model": {"unet_channels": [4, 4], "likelihood": "cnp"},
            "training": {"optimizer": "adam", "learning_rate": 1e-3, "resume": False},
            "data": {"target_var_IDs": "var"},
            "logging": {"wandb": {"enabled": True, "watch_model": False}},
            "visualization": {
                "plot_dates": None,
                "res_factor_maps": 1,
                "crs": "plate_carree",
                "cmap": "viridis",
                "fig_size": 3.0,
            },
        }
    )


@pytest.fixture
def built_model(trainer_config, tiny_dp_and_task_loader):
    data_processor, task_loader, _ = tiny_dp_and_task_loader
    model = DeepSensorModel(trainer_config)
    model.build_model(data_processor, task_loader)
    return model


@pytest.fixture
def fake_wandb(monkeypatch, tmp_path):
    """Replace W&B calls with fakes; returns the run and the list of saved paths."""
    run_config: dict = {}
    run = SimpleNamespace(
        dir=str(tmp_path),
        name="run",
        url="url",
        config=SimpleNamespace(update=run_config.update, values=run_config),
    )
    saved: list[str] = []

    monkeypatch.setattr(wu.wandb, "init", lambda **kwargs: run)
    monkeypatch.setattr(trainer_module.wandb, "save", lambda path, **kw: saved.append(path))
    return run, saved


class TestTrainerWandbSetup:
    @pytest.mark.parametrize("pass_run", [False, True], ids=["new_run", "existing_run"])
    def test_saves_files_needed_to_reload_model(
        self, trainer_config, built_model, fake_wandb, tmp_path, pass_run
    ):
        run, saved = fake_wandb

        Trainer(built_model, trainer_config, wandb_run=run if pass_run else None)

        files_dir = tmp_path / "files"
        for name in [wu.MODEL_CONFIG_FILE, wu.DATA_PROCESSOR_CONFIG_FILE]:
            assert (files_dir / name).exists()
            assert os.path.join(str(files_dir), name) in saved
        with open(files_dir / wu.MODEL_CONFIG_FILE) as f:
            saved_likelihood = json.load(f)["likelihood"]
        assert saved_likelihood == built_model.model.config["likelihood"]
        assert run.config.values["model_parameters"] == built_model.count_parameters()
