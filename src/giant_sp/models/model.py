"""DeepSensor model definitions and utilities."""

import contextlib
import io
import logging
from typing import Any

import torch
from deepsensor.data import DataProcessor, Task, TaskLoader
from deepsensor.model import ConvNP

from giant_sp.models.mdn import MDNZIGammaConvNP, install_mdn_zigamma_likelihood

logger = logging.getLogger(__name__)


class DeepSensorModel:
    """Wrapper for a DeepSensor ``ConvNP`` model with training utilities.

    Attributes
    ----------
    config : dict[str, Any]
        Configuration dictionary, typically loaded from a YAML config file.
        Expected to contain a ``"model"`` key holding the keyword arguments
        passed to :class:`~deepsensor.model.ConvNP` (e.g. ``likelihood``,
        ``unet_channels``, ``unet_kernels``, ``num_basis_functions``).
    model : deepsensor.model.ConvNP | None
        The underlying DeepSensor model, set once :meth:`build_model` has
        been called.
    task_loader : deepsensor.data.TaskLoader | None
        The :class:`~deepsensor.data.TaskLoader` used to build ``model``,
        set once :meth:`build_model` has been called.
    normalization_params : None
        Reserved for normalization parameters. Currently unused.

    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the DeepSensor model wrapper.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary. Must contain a ``"model"`` key whose
            value is a dictionary of keyword arguments for
            :class:`~deepsensor.model.ConvNP` (used by :meth:`build_model`).

        """
        self.config = config
        self.model: ConvNP | None = None
        self.task_loader: TaskLoader | None = None
        self.normalization_params = None

    def build_model(
        self, data_processor: DataProcessor, task_loader: TaskLoader
    ) -> ConvNP:
        """Build the ``ConvNP`` model with the architecture from ``self.config``.

        Parameters
        ----------
        data_processor : deepsensor.data.DataProcessor
            The :class:`~deepsensor.data.DataProcessor` used to normalize
            inputs/outputs for the model.
        task_loader : deepsensor.data.TaskLoader
            The :class:`~deepsensor.data.TaskLoader` describing the context
            and target sets the model should be built for.

        Returns
        -------
        deepsensor.model.ConvNP
            The newly constructed model, also stored on ``self.model``.

        """
        model_config = self.config["model"]

        logger.info("Building ConvNP model with PyTorch backend...")
        logger.info("Model likelihood: %s", model_config["likelihood"])

        if model_config["likelihood"] == "mdn-zigamma":
            install_mdn_zigamma_likelihood(model_config["mixture_components"])
            convnp_cls = MDNZIGammaConvNP
        else:
            convnp_cls = ConvNP

        # Initialize ConvNP model, capturing verbose print() output into logger
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.model = convnp_cls(
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

    def get_model_summary(self) -> None:
        """Log the model's architecture summary and parameter count.

        Logs a warning and returns without doing anything if the model has
        not been built yet.
        """
        if self.model is None:
            logger.warning("Model not built yet")
            return

        # Model summary will be printed by DeepSensor
        logger.info("Model architecture:")
        logger.info("Parameters: %d", self.count_parameters())

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model.

        Returns
        -------
        int
            The number of trainable parameters, or ``0`` if the model has
            not been built yet.

        """
        if self.model is None:
            return 0

        total_params = sum(
            p.numel() for p in self.model.model.parameters() if p.requires_grad
        )

        return int(total_params)

    def save_model(self, filepath: str) -> None:
        """Save the model's state dict to disk.

        Parameters
        ----------
        filepath : str
            Path to save the model checkpoint to.

        Raises
        ------
        ValueError
            If the model has not been built yet.

        """
        if self.model is None:
            raise ValueError("No model to save")

        logger.info("Saving model to %s", filepath)
        torch.save(self.model.model.state_dict(), filepath)

    def load_model(self, filepath: str) -> None:
        """Load a model checkpoint's state dict from disk.

        Parameters
        ----------
        filepath : str
            Path to the model checkpoint to load.

        Raises
        ------
        ValueError
            If the model has not been built yet (weights are loaded into
            the existing model, so it must exist first).

        """
        if self.model is None:
            raise ValueError("Model must be built before loading weights")

        logger.info("Loading model from %s", filepath)
        model_state_dict = torch.load(filepath)
        self.model.model.load_state_dict(model_state_dict)

    def predict(self, task: Task, num_samples: int = 1) -> Any:
        """Make predictions with uncertainty quantification.

        Parameters
        ----------
        task : deepsensor.data.Task
            The :class:`~deepsensor.data.Task` (context/target sets) to
            predict on.
        num_samples : int, default 1
            Number of samples to draw from the model's posterior for
            uncertainty estimation.

        Returns
        -------
        Any
            The ``neuralprocesses`` distribution returned by the underlying
            model, from which samples, mean and standard deviation can be
            derived.

        Raises
        ------
        ValueError
            If the model has not been built yet.

        """
        if self.model is None:
            raise ValueError("Model not built yet")

        logger.info("Making predictions with %d samples", num_samples)

        # Sample from posterior
        pred_dist = self.model(task, n_samples=num_samples)

        return pred_dist
