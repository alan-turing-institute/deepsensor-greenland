import pytest
import torch

from ds_greenland.models.model import DeepSensorModel


class TestInit:
    def test_initial_state(self, tiny_model_config):
        model = DeepSensorModel(tiny_model_config)
        assert model.config == tiny_model_config
        assert model.model is None
        assert model.task_loader is None
        assert model.normalization_params is None


class TestBuildModel:
    def test_returns_and_stores_model(self, tiny_model_config, tiny_dp_and_task_loader):
        data_processor, task_loader, _ = tiny_dp_and_task_loader
        model = DeepSensorModel(tiny_model_config)

        built = model.build_model(data_processor, task_loader)

        assert built is model.model
        assert model.task_loader is task_loader


class TestCountParameters:
    def test_zero_before_build(self, tiny_model_config):
        model = DeepSensorModel(tiny_model_config)
        assert model.count_parameters() == 0

    def test_matches_manual_count_after_build(
        self, tiny_model_config, tiny_dp_and_task_loader
    ):
        data_processor, task_loader, _ = tiny_dp_and_task_loader
        model = DeepSensorModel(tiny_model_config)
        model.build_model(data_processor, task_loader)

        expected = sum(
            p.numel() for p in model.model.model.parameters() if p.requires_grad
        )
        assert model.count_parameters() == expected
        assert model.count_parameters() > 0


class TestGetModelSummary:
    def test_warns_when_not_built(self, tiny_model_config, caplog):
        model = DeepSensorModel(tiny_model_config)

        with caplog.at_level("WARNING"):
            model.get_model_summary()

        assert "Model not built yet" in caplog.text

    def test_logs_parameter_count_when_built(
        self, tiny_model_config, tiny_dp_and_task_loader, caplog
    ):
        data_processor, task_loader, _ = tiny_dp_and_task_loader
        model = DeepSensorModel(tiny_model_config)
        model.build_model(data_processor, task_loader)

        with caplog.at_level("INFO"):
            model.get_model_summary()

        assert "Parameters:" in caplog.text


class TestSaveModel:
    def test_raises_when_not_built(self, tiny_model_config, tmp_path):
        model = DeepSensorModel(tiny_model_config)

        with pytest.raises(ValueError, match="No model to save"):
            model.save_model(str(tmp_path / "model.pt"))

    def test_writes_checkpoint_file(
        self, tiny_model_config, tiny_dp_and_task_loader, tmp_path
    ):
        data_processor, task_loader, _ = tiny_dp_and_task_loader
        model = DeepSensorModel(tiny_model_config)
        model.build_model(data_processor, task_loader)

        filepath = tmp_path / "model.pt"
        model.save_model(str(filepath))

        assert filepath.exists()


class TestLoadModel:
    def test_raises_when_not_built(self, tiny_model_config, tmp_path):
        model = DeepSensorModel(tiny_model_config)

        with pytest.raises(
            ValueError, match="Model must be built before loading weights"
        ):
            model.load_model(str(tmp_path / "model.pt"))

    def test_round_trips_weights(
        self, tiny_model_config, tiny_dp_and_task_loader, tmp_path
    ):
        data_processor, task_loader, _ = tiny_dp_and_task_loader

        source = DeepSensorModel(tiny_model_config)
        source.build_model(data_processor, task_loader)
        filepath = tmp_path / "model.pt"
        source.save_model(str(filepath))

        target = DeepSensorModel(tiny_model_config)
        target.build_model(data_processor, task_loader)
        target.load_model(str(filepath))

        for source_param, target_param in zip(
            source.model.model.parameters(),
            target.model.model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(source_param, target_param)


class TestPredict:
    def test_raises_when_not_built(self, tiny_model_config):
        model = DeepSensorModel(tiny_model_config)

        with pytest.raises(ValueError, match="Model not built yet"):
            model.predict(task=None)

    def test_returns_distribution_for_task(
        self, tiny_model_config, tiny_dp_and_task_loader
    ):
        data_processor, task_loader, dates = tiny_dp_and_task_loader
        model = DeepSensorModel(tiny_model_config)
        model.build_model(data_processor, task_loader)

        task = task_loader(dates[0], context_sampling="all", target_sampling="all")
        pred_dist = model.predict(task, num_samples=1)

        assert pred_dist is not None
