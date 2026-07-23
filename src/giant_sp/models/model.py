"""
DeepSensor model definitions and utilities
"""

import contextlib
import io
import logging

import torch
import torch.nn as nn
from deepsensor.data import DataProcessor, TaskLoader
from deepsensor.model import ConvNP

logger = logging.getLogger(__name__)


class DeepSensorModel:
    """Wrapper for DeepSensor ConvNP model with training utilities"""

    def __init__(self, config: dict):
        """
        Initialize DeepSensor model

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model = None
        self.task_loader = None
        self.normalization_params = None

    def build_model(self, data_processor: DataProcessor, task_loader: TaskLoader):
        """
        Build ConvNP model with specified architecture

        Args:
            task_loader: DeepSensor TaskLoader instance
        """
        model_config = self.config["model"]

        logger.info("Building ConvNP model with PyTorch backend...")
        logger.info(f"Model likehood: {model_config['likelihood']}")

        # Initialize ConvNP model, capturing verbose print() output into logger
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.model = ConvNP(
                data_processor,
                task_loader,
                verbose=True,
                **model_config,
            )
        verbose_output = buf.getvalue()
        if verbose_output:
            for line in verbose_output.strip().splitlines():
                logger.info(line)

        self.task_loader = task_loader
        logger.info("Model built successfully with PyTorch backend")

        return self.model

    def get_model_summary(self):
        """Print model architecture summary"""
        if self.model is None:
            logger.warning("Model not built yet")
            return

        # Model summary will be printed by DeepSensor
        logger.info("Model architecture:")
        logger.info(f"Parameters: {self.count_parameters()}")

    def count_parameters(self) -> int:
        """Count total trainable parameters in model"""
        if self.model is None:
            return 0

        total_params = sum(
            p.numel() for p in self.model.model.parameters() if p.requires_grad
        )

        return int(total_params)

    def save_model(self, filepath: str):
        """
        Save model checkpoint

        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save")

        logger.info(f"Saving model to {filepath}")
        torch.save(self.model.model.state_dict(), filepath)

    def load_model(self, filepath: str):
        """
        Load model checkpoint

        Args:
            filepath: Path to model checkpoint
        """
        if self.model is None:
            raise ValueError("Model must be built before loading weights")

        logger.info(f"Loading model from {filepath}")
        model_state_dict = torch.load(filepath)
        self.model.model.load_state_dict(model_state_dict)

    def predict(self, task, num_samples: int = 1):
        """
        Make predictions with uncertainty quantification

        Args:
            task: DeepSensor Task object
            num_samples: Number of samples for uncertainty estimation

        Returns:
            Predictions with mean and standard deviation
        """
        if self.model is None:
            raise ValueError("Model not built yet")

        logger.info(f"Making predictions with {num_samples} samples")

        # Sample from posterior
        pred_dist = self.model(task, n_samples=num_samples)

        return pred_dist
