"""Training manager for DeepSensor models.

Provides the :class:`Trainer` class, which orchestrates the full training
loop for a DeepSensor ``ConvNP`` model: optimizer creation, epoch-by-epoch
training and validation, best/periodic/final checkpointing, optional
station-based validation metrics, and (optionally) experiment tracking and
visualization logging to Weights & Biases (W&B).
"""

import json
import logging
import os
import random
from collections.abc import Callable
from typing import Any

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")

import deepsensor.plot as dsplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import wandb
import xarray as xr
from deepsensor.data.task import Task
from deepsensor.model import ConvNP
from deepsensor.model.pred import Prediction
from torch.utils.data import DataLoader
from tqdm import tqdm
from wandb.sdk.wandb_run import Run

import giant_sp.utils.plots as myplt
import giant_sp.utils.training as tu
from giant_sp.models.model import DeepSensorModel
from giant_sp.utils.data import NumpyFloatValuesEncoder
from giant_sp.utils.model import (
    compute_pred_ds,
    get_available_device,
    get_pred_DAs_taskstream,
)

logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for DeepSensor models.

    Wraps a :class:`~giant_sp.models.model.DeepSensorModel`, its optimizer,
    and (optionally) a Weights & Biases run, and drives the full training
    loop via :meth:`train`.

    Attributes
    ----------
    model : DeepSensorModel
        The wrapped DeepSensor model being trained.
    config : dict[str, Any]
        Full configuration dictionary passed at construction.
    train_config : dict[str, Any]
        Shortcut to ``config["training"]``.
    optimizer : torch.optim.Optimizer
        Optimizer created from ``train_config``.
    history : dict[str, list[float]]
        Per-epoch training history with keys ``"train_loss"``,
        ``"val_loss"``, and ``"epoch"``.
    best_val_loss : float
        Lowest validation loss observed so far (``inf`` until training runs).
    epochs_without_improvement : int
        Number of consecutive epochs since ``best_val_loss`` last improved.
    best_stnval_loss : float
        Lowest station-validation loss observed so far (``inf`` initially).
    use_wandb : bool
        Whether W&B logging is enabled, from
        ``config["logging"]["wandb"]["enabled"]``.
    wandb_run : wandb.sdk.wandb_run.Run | None
        Active W&B run, if ``use_wandb`` is ``True``; otherwise ``None``.
    plot_dates : list[str] | None
        Fixed dates to use for prediction map plots, from
        ``config["visualization"]["plot_dates"]``. If ``None``, dates are
        sampled randomly each time maps are plotted.
    res_factor_maps : int
        Resolution factor used when predicting for map visualizations.
    crs : str
        Name of the map projection to use for plotting (e.g.
        ``"north_polar"``).
    cmap : str
        Colormap name used for prediction/error map plots.
    fig_size : float
        Per-subplot figure size (inches) used when building map grids.
    projection : cartopy.crs.CRS
        Cartopy projection instance derived from ``crs``.
    local_save_dir : str
        Local directory (inside the W&B run directory) where checkpoints
        and config files are written. Only set when ``use_wandb`` is
        ``True`` (see the W&B setup methods).
    pred_params : list[str]
        Prediction outputs to request from the model (e.g. ``["mean",
        "std"]``, extended with mixture parameters for the
        ``"bernoulli-gamma"`` likelihood). Set in :meth:`train`.

    """

    def __init__(
        self,
        model: DeepSensorModel,
        config: dict[str, Any],
        wandb_run: Run | None = None,
    ) -> None:
        """Initialize the trainer and configure optimization and experiment logging.

        Creates the optimizer, resets training-history and best-loss
        tracking, configures the plotting projection, and — if W&B logging
        is enabled in the config — either restores a previous W&B run,
        attaches to an already-initialized run, or starts a new one.

        Parameters
        ----------
        model : DeepSensorModel
            Wrapper around the DeepSensor ``ConvNP`` model to train. Its
            ``model.model`` attribute (the underlying ``ConvNP``) and
            ``model.model.model`` (the underlying `torch.nn.Module`) are
            used for optimization.
        config : dict[str, Any]
            Full configuration dictionary, expected to contain
            ``"training"``, ``"logging"``, ``"visualization"``, ``"model"``,
            and ``"data"`` sections.
        wandb_run : wandb.sdk.wandb_run.Run | None, optional
            An already-initialized W&B run to attach to, e.g. when the
            calling script has already called ``wandb.init()``. If
            ``None`` and W&B logging is enabled, a new run is created (or
            an existing one restored, if
            ``config["training"]["resume"]`` is set). Default is ``None``.

        """
        self.model = model
        self.config = config
        self.train_config = config["training"]

        # Setup optimizer
        self.optimizer = self._create_optimizer()

        # Training history
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "epoch": [],
        }

        # Best model tracking
        self.best_val_loss = np.inf
        self.epochs_without_improvement = 0
        self.best_stnval_loss = np.inf

        # Initialize wandb if enabled
        self.use_wandb = config["logging"].get("wandb", {}).get("enabled", False)
        self.wandb_run = wandb_run  # may be pre-initialized from the training script

        # Plotting settings
        self.plot_dates = config["visualization"]["plot_dates"]
        self.res_factor_maps = config["visualization"]["res_factor_maps"]
        self.crs = config["visualization"]["crs"]
        self.cmap = config["visualization"]["cmap"]
        self.fig_size = config["visualization"]["fig_size"]

        if self.crs == "north_polar":
            self.projection = ccrs.NorthPolarStereo(central_longitude=-45)
        else:
            self.projection = ccrs.PlateCarree()

        if self.use_wandb and self.config["training"]["resume"]:
            self._restore_wandb()
        elif self.use_wandb and (not self.config["training"]["resume"]):
            if self.wandb_run is not None:
                self._setup_existing_wandb_run()
            else:
                self._init_wandb()

    def _create_optimizer(self) -> optim.Optimizer:
        """Build the optimizer specified in the training configuration.

        Reads ``self.train_config["optimizer"]`` (one of ``"adam"``,
        ``"adamw"``, or ``"sgd"``, case-insensitive) along with the
        learning rate and optional weight decay, and constructs the
        corresponding `torch.optim` optimizer over the parameters of the
        underlying `torch.nn.Module` (``self.model.model.model``).

        Returns
        -------
        torch.optim.Optimizer
            The constructed optimizer instance.

        Raises
        ------
        ValueError
            If ``self.train_config["optimizer"]`` is not one of the
            supported optimizer names.

        """
        lr = self.train_config["learning_rate"]
        weight_decay = self.train_config.get("weight_decay", 0.0)

        assert self.model.model is not None, "model.model must be built before training"
        optimizer: optim.Optimizer
        if self.train_config["optimizer"].lower() == "adam":
            optimizer = optim.Adam(
                self.model.model.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.train_config["optimizer"].lower() == "adamw":
            optimizer = optim.AdamW(
                self.model.model.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif self.train_config["optimizer"].lower() == "sgd":
            optimizer = optim.SGD(
                self.model.model.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.train_config['optimizer']}")

        logger.info(
            "Created %s optimizer with lr=%f", self.train_config["optimizer"], lr
        )
        return optimizer

    def _restore_wandb(self) -> None:
        """Resume a previous W&B run and reload its latest checkpoint.

        Reinitializes the W&B run identified by
        ``config["training"]["wandb_run_id"]`` in resume mode, recreates
        the local save directory, and loads the model weights saved as
        ``"files/last-weights.pt"`` in that run into
        ``self.model.model.model``.

        Notes
        -----
        Sets ``self.wandb_run`` and ``self.local_save_dir`` as a side
        effect.

        """
        run_id = self.config["training"]["wandb_run_id"]
        entity_id = self.config["logging"]["wandb"]["entity"]
        project_id = self.config["logging"]["wandb"]["project"]

        # Restore wandb run
        self.wandb_run = wandb.init(
            entity=entity_id,
            project=project_id,
            id=run_id,
            resume="auto",
        )

        self.local_save_dir = os.path.join(self.wandb_run.dir, "files")
        os.makedirs(self.local_save_dir)

        # Restore best-loss tracking so post-resume epochs only overwrite
        # the best checkpoints if they actually improve on the pre-resume best.
        self.best_val_loss = self.wandb_run.summary.get("best_val_loss", np.inf)
        self.best_stnval_loss = self.wandb_run.summary.get("best_stnval_loss", np.inf)

        self.latest_model_epoch = self.wandb_run.summary["epoch"]

        # Load model weights
        logger.info("Loading model weights of epoch %03d...", self.latest_model_epoch)
        weights_file = wandb.restore("files/last-weights.pt")
        assert weights_file is not None, "wandb.restore did not find the weights file"
        assert self.model.model is not None, "model.model must be built before training"
        self.model.model.model.load_state_dict(
            torch.load(weights_file.name, map_location=get_available_device())
        )

        logger.info("Restore W&B run: %s", self.wandb_run.name)
        logger.info("W&B URL: %s", self.wandb_run.url)

    def _setup_existing_wandb_run(self) -> None:
        """Configure a W&B run that was already initialized before construction.

        Updates the pre-existing run's config with the model's parameter
        count, optionally attaches `wandb.watch` to the model, creates
        the local save directory, and saves the data-processor and model
        configuration as JSON files locally and to W&B.

        Notes
        -----
        Sets ``self.local_save_dir`` as a side effect. Assumes
        ``self.wandb_run`` is already set to an active
        `wandb.sdk.wandb_run.Run`.

        """
        assert self.wandb_run is not None, (
            "wandb_run must already be initialized by the caller"
        )
        assert self.model.model is not None, "model.model must be built before training"
        wandb_config = self.config["logging"]["wandb"]

        # Add model parameters now that the model is built
        self.wandb_run.config.update(
            {"model_parameters": self.model.count_parameters()}
        )

        if wandb_config.get("watch_model", True):
            wandb.watch(
                self.model.model.model,
                log="all",
                log_freq=wandb_config.get("watch_freq", 100),
            )

        self.local_save_dir = os.path.join(self.wandb_run.dir, "files")
        os.makedirs(self.local_save_dir)

        logger.info("Saving data processor config")
        with open(
            os.path.join(self.local_save_dir, "data_processor_config.json"), "w"
        ) as f:
            json.dump(
                self.model.model.data_processor.config, f, indent=4, sort_keys=False
            )
        wandb.save(
            os.path.join(self.local_save_dir, "data_processor_config.json"),
            base_path=self.wandb_run.dir,
        )

        logger.info("Saving model config")
        config_fpath = os.path.join(self.local_save_dir, "model_config.json")
        with open(config_fpath, "w") as f:
            json.dump(
                self.model.model.config,
                f,
                cls=NumpyFloatValuesEncoder,
                indent=4,
                sort_keys=False,
            )
        wandb.save(
            os.path.join(self.local_save_dir, "model_config.json"),
            base_path=self.wandb_run.dir,
        )

    def _init_wandb(self) -> None:
        """Start a new W&B run and log initial configuration artifacts.

        Creates a new run via `wandb.init` using the
        ``config["logging"]["wandb"]`` settings, optionally attaches
        `wandb.watch` to the model, creates the local save directory, and
        saves the data-processor configuration as a JSON file locally and
        to W&B.

        Notes
        -----
        Sets ``self.wandb_run`` and ``self.local_save_dir`` as a side
        effect.

        """
        assert self.model.model is not None, "model.model must be built before training"
        wandb_config = self.config["logging"]["wandb"]

        # Initialize wandb run
        self.wandb_run = wandb.init(
            project=wandb_config.get("project", "deepsensor-spatial-interpolation"),
            entity=wandb_config.get("entity"),
            name=wandb_config.get("name"),
            tags=wandb_config.get("tags", []),
            notes=wandb_config.get("docs", ""),
            config={
                "model": self.config["model"],
                "training": self.train_config,
                "data": self.config["data"],
                "model_parameters": self.model.count_parameters(),
            },
        )

        # Watch model if enabled
        if wandb_config.get("watch_model", True):
            wandb.watch(
                self.model.model.model,
                log="all",
                log_freq=wandb_config.get("watch_freq", 100),
            )

        self.local_save_dir = os.path.join(self.wandb_run.dir, "files")
        os.makedirs(self.local_save_dir)

        logger.info("Initialized W&B run: %s", self.wandb_run.name)
        logger.info("W&B URL: %s", self.wandb_run.url)

        # Save norm params
        logger.info("Saving data processor config")
        with open(
            os.path.join(self.local_save_dir, "data_processor_config.json"), "w"
        ) as f:
            json.dump(
                self.model.model.data_processor.config, f, indent=4, sort_keys=False
            )
        wandb.save(
            os.path.join(self.local_save_dir, "data_processor_config.json"),
            base_path=self.wandb_run.dir,
        )

    def train(
        self,
        truth_ds: xr.Dataset,
        task_stream_tr: tu.TaskStreamDataset | tu.GapFillTaskStreamDataset,
        val_tasks: list[Task],
        mask: xr.DataArray | None = None,
        stnval_tasks: list[Task] | None = None,
        callback: Callable | None = None,
    ) -> None:
        """Run the full training loop.

        For each epoch (or, when ``train_config["resume"]`` is set, from
        the epoch after the last recorded one), this:

        1. Trains one epoch over ``task_stream_tr`` via
           :func:`giant_sp.utils.training.train_epoch_with_dataloader`.
        2. Computes training-set predictions/errors (via
           :func:`~giant_sp.utils.model.get_pred_DAs_taskstream`) and
           derives train NLL/MAE/RMSE.
        3. Computes the validation loss over ``val_tasks`` and appends
           train/val loss and the epoch number to ``self.history``.
        4. Saves best-validation-loss weights (if improved) and applies
           early stopping based on ``train_config["patience"]``.
        5. Periodically checkpoints weights every
           ``train_config["save_checkpoint_every"]`` epochs.
        6. Computes validation-set predictions/errors (via
           :func:`~giant_sp.utils.model.compute_pred_ds`) and derives val
           NLL/MAE/RMSE (and, if the predictions have a ``lead_time``
           dimension, an RMSE-by-lead-time array).
        7. If ``stnval_tasks`` is given and station-validation logging is
           enabled, computes station-validation loss/metrics every other
           epoch (and on the final epoch), and tracks the best
           station-validation loss.
        8. If W&B logging is enabled, logs all of the above metrics, and
           periodically logs prediction maps and charts, then saves the
           "last" checkpoint.

        After the loop, saves a "final" checkpoint and, if W&B logging is
        enabled, uploads it and finishes the run.

        Parameters
        ----------
        truth_ds : xarray.Dataset
            Ground-truth dataset used for computing prediction errors.
        task_stream_tr : giant_sp.utils.training.TaskStreamDataset or giant_sp.utils.training.GapFillTaskStreamDataset
            Iterable dataset of training tasks, wrapped in a
            `torch.utils.data.DataLoader` for the training step.
        val_tasks : list[deepsensor.data.task.Task]
            Validation tasks used for validation loss and metrics.
        mask : xarray.DataArray | None, optional
            Optional target mask passed through to prediction calls.
            Default is ``None``.
        stnval_tasks : list[deepsensor.data.task.Task] | None, optional
            Optional station-validation tasks. Required if
            ``config["logging"]["log_stnval"]`` is ``True``. Default is
            ``None``.
        callback : collections.abc.Callable | None, optional
            Accepted for future use; currently not invoked anywhere in
            this method. Default is ``None``.

        Notes
        -----
        Mutates ``self.history``, ``self.best_val_loss``,
        ``self.epochs_without_improvement``, ``self.best_stnval_loss``,
        and ``self.pred_params``. Writes checkpoint files under
        ``self.local_save_dir`` (which must already be set — i.e. W&B
        logging must have been enabled during construction — since
        weights are saved there unconditionally).

        """
        num_epochs = self.train_config["num_epochs"]

        if not hasattr(self, "local_save_dir"):
            self.local_save_dir = os.path.join("outputs", "training")
            os.makedirs(self.local_save_dir, exist_ok=True)

        assert self.model.model is not None, "model.model must be built before training"
        model = self.model.model
        self.train_config["learning_rate"]
        target_var = self.config["data"]["target_var_IDs"]
        batch_size = self.config["training"]["batch_size"]
        log_stnval = self.config["logging"]["log_stnval"]

        self.pred_params = ["mean", "std"]
        if self.model.config["model"]["likelihood"] == "bernoulli-gamma":
            self.pred_params += ["mixture_probs", "k", "scale"]

        logger.info("Starting training for %d epochs", num_epochs)
        logger.info("Training samples: %d", len(task_stream_tr.date_list))
        logger.info("Validation samples: %d", len(val_tasks))

        # Plot encoding and log it to wandb
        fig = dsplt.context_encoding(model, val_tasks[0], model.task_loader)
        if self.use_wandb:
            wandb.log({"encoding": wandb.Image(fig)})
        plt.close(fig)

        # Wrap with torch DataLoader for training loop
        task_dataloader_tr = DataLoader(
            task_stream_tr, batch_size=batch_size, collate_fn=tu.task_collate_fn
        )

        if self.train_config["resume"]:
            assert self.wandb_run is not None, (
                "wandb_run must be set when resuming (see _restore_wandb)"
            )
            n0 = self.latest_model_epoch + 1
            nend = self.train_config["num_epochs"]
        else:
            n0 = 0
            nend = self.train_config["num_epochs"]
        for ep in tqdm(range(n0, nend), desc="Training"):
            # Train
            batch_losses = tu.train_epoch_with_dataloader(
                model, task_dataloader_tr, self.optimizer
            )
            epoch_loss = np.mean(batch_losses)
            logger.info("Starting train set prediction")
            _, train_std_da, train_err_da = get_pred_DAs_taskstream(
                model, task_stream_tr, 120, truth_ds, target_var, mask=mask
            )
            logger.info("Done, computing metrics")
            train_nll = (
                (
                    np.log(train_std_da)
                    + np.power(train_err_da, 2) / (2 * np.power(train_std_da, 2))
                )
                .mean()
                .values.item()
            )
            train_mae = np.abs(train_err_da).mean().values.item()
            train_rmse = np.sqrt(np.power(train_err_da, 2).mean()).values.item()

            # Val
            # Compute validation loss
            logger.info("Starting val loss computation")
            val_task_losses = []
            for task in val_tasks:
                val_task_losses.append(
                    model.loss_fn(task, normalise=True).detach().cpu().numpy()
                )
            epoch_loss_val = np.mean(val_task_losses)

            # Update history
            self.history["train_loss"].append(epoch_loss)
            self.history["val_loss"].append(epoch_loss_val)
            self.history["epoch"].append(ep + 1)

            # Save best val loss weights
            if epoch_loss_val < self.best_val_loss:
                self.best_val_loss = epoch_loss_val
                self.epochs_without_improvement = 0

                if self.train_config.get("save_best_only", True):
                    weights_path = os.path.join(self.local_save_dir, "best-weights.pt")
                    self.model.save_model(weights_path)
                    if self.use_wandb:
                        assert wandb.run is not None
                        assert self.wandb_run is not None
                        wandb.run.summary["best_val_loss"] = self.best_val_loss
                        wandb.run.summary["best_val_loss_epoch"] = ep + 1
                        wandb.save(weights_path, base_path=self.wandb_run.dir)
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if (
                self.train_config.get("early_stopping", False)
                and self.epochs_without_improvement >= self.train_config["patience"]
            ):
                logger.info("Early stopping triggered after %d epochs", ep + 1)
                break

            # Periodic checkpointing
            if (ep + 1) % self.train_config.get("save_checkpoint_every", 10) == 0:
                torch.save(
                    model.model.state_dict(),
                    os.path.join(self.local_save_dir, f"weights-ep{ep:02}.pt"),
                )
                if self.use_wandb:
                    assert self.wandb_run is not None
                    wandb.save(
                        os.path.join(self.local_save_dir, f"weights-ep{ep:02}.pt"),
                        base_path=self.wandb_run.dir,
                    )

            # Track val metrics
            logger.info("Starting val set prediction")
            val_pred, val_err_ds = compute_pred_ds(
                model, val_tasks, truth_ds, target_var, self.pred_params, mask=mask
            )
            val_mean_ds = val_pred[target_var]["mean"]
            val_std_ds = val_pred[target_var]["std"]
            logger.info("Done, computing metrics")

            val_nll = (
                (
                    np.log(val_std_ds)
                    + np.power(val_err_ds, 2) / (2 * np.power(val_std_ds, 2))
                )
                .mean()
                .values.item()
            )
            val_mae = np.abs(val_err_ds).mean().values.item()
            val_rmse = np.sqrt(np.power(val_err_ds, 2).mean()).values.item()
            if "lead_time" in val_mean_ds.dims:
                val_rmse_lt = (
                    np.power(val_err_ds, 2).mean(["init_time", "lat", "lon"])  # type: ignore[list-item]
                    ** 0.5
                )

            # Stnval
            if log_stnval and ((ep % 2 == 0) or ep == nend - 1):
                assert stnval_tasks is not None, (
                    "stnval_tasks must be provided when log_stnval is enabled"
                )
                # Compute validation loss
                logger.info("Starting stnval loss computation")
                stnval_task_losses = []
                for task in stnval_tasks:
                    stnval_task_losses.append(
                        model.loss_fn(task, normalise=True).detach().cpu().numpy()
                    )
                epoch_loss_stnval = np.mean(stnval_task_losses)
                # Save best val loss weights
                if epoch_loss_stnval < self.best_stnval_loss:
                    self.best_stnval_loss = epoch_loss_stnval
                    weights_path = os.path.join(
                        self.local_save_dir, "best-weights-stnval.pt"
                    )
                    torch.save(model.model.state_dict(), weights_path)
                    if self.use_wandb:
                        assert wandb.run is not None
                        assert self.wandb_run is not None
                        wandb.run.summary["best_stnval_loss"] = self.best_stnval_loss
                        wandb.run.summary["best_stnval_loss_epoch"] = ep + 1
                        wandb.save(weights_path, base_path=self.wandb_run.dir)
                # Track val metrics
                logger.info("Starting stnval set prediction")
                stnval_pred, stnval_err_ds = compute_pred_ds(
                    model, stnval_tasks, truth_ds, target_var, self.pred_params
                )
                stnval_std_ds = stnval_pred[target_var]["std"]
                logger.info("Done, computing metrics")
                stnval_nll = (
                    (
                        np.log(stnval_std_ds)
                        + np.power(stnval_err_ds, 2) / (2 * np.power(stnval_std_ds, 2))
                    )
                    .mean()
                    .values.item()
                )
                stnval_mae = np.abs(stnval_err_ds).mean().values.item()
                stnval_rmse = np.sqrt(np.power(stnval_err_ds, 2).mean()).values.item()

                if self.use_wandb:
                    ep_log = ep + 1
                    wandb.log(
                        {f"val/loss_station/{target_var}": epoch_loss_stnval},
                        step=ep_log,
                    )
                    wandb.log({f"val/nll_stn/{target_var}": stnval_nll}, step=ep_log)
                    wandb.log({f"val/mae_stn/{target_var}": stnval_mae}, step=ep_log)
                    wandb.log({f"val/rmse_stn/{target_var}": stnval_rmse}, step=ep_log)

            if self.use_wandb:
                assert self.wandb_run is not None
                ep_log = ep + 1
                self._log_wandb_metrics(ep_log, epoch_loss, epoch_loss_val)
                wandb.log({f"train/nll/{target_var}": train_nll}, step=ep_log)
                wandb.log({f"train/mae/{target_var}": train_mae}, step=ep_log)
                wandb.log({f"train/rmse/{target_var}": train_rmse}, step=ep_log)
                wandb.log({f"val/nll/{target_var}": val_nll}, step=ep_log)
                wandb.log({f"val/mae/{target_var}": val_mae}, step=ep_log)
                wandb.log({f"val/rmse/{target_var}": val_rmse}, step=ep_log)

                # Log images periodically
                wandb_config = self.config["logging"]["wandb"]
                if (
                    wandb_config.get("log_maps", True)
                    and (ep + 1) % wandb_config.get("log_maps_frequency") == 0
                ):
                    self._plot_additional_maps(
                        "val", val_tasks, model, truth_ds, mask, ep_log
                    )

                    if log_stnval and ((ep % 2 == 0) or ep == nend - 1):
                        assert stnval_tasks is not None, (
                            "stnval_tasks must be provided when log_stnval is enabled"
                        )
                        self._plot_additional_maps(
                            "stnval", stnval_tasks, model, truth_ds, mask, ep_log
                        )

                if (
                    wandb_config.get("log_charts", True)
                    and (ep + 1) % wandb_config.get("log_charts_frequency") == 0
                ):
                    # self._log_wandb_predictions(truth_ds, ep)
                    self._plot_additional_charts(
                        "val", val_tasks, truth_ds, val_pred, ep_log
                    )
                    if "lead_time" in val_mean_ds.dims:
                        self._plot_additional_charts_forecast(val_rmse_lt, ep_log)

                    if log_stnval and ((ep % 2 == 0) or ep == nend - 1):
                        assert stnval_tasks is not None, (
                            "stnval_tasks must be provided when log_stnval is enabled"
                        )
                        self._plot_additional_charts(
                            "stnval", stnval_tasks, truth_ds, stnval_pred, ep_log
                        )

                # Save last checkpoint weights
                torch.save(
                    model.model.state_dict(),
                    os.path.join(self.local_save_dir, "last-weights.pt"),
                )
                wandb.save(
                    os.path.join(self.local_save_dir, "last-weights.pt"),
                    base_path=self.wandb_run.dir,
                )

        # Save final model weights
        torch.save(
            model.model.state_dict(),
            os.path.join(self.local_save_dir, "final-weights.pt"),
        )

        if self.use_wandb:
            assert self.wandb_run is not None
            logger.info("Saved model config")
            wandb.save(
                os.path.join(self.local_save_dir, "final-weights.pt"),
                base_path=self.wandb_run.dir,
            )
            self.wandb_run.finish()
            logger.info("W&B run finished")

    def _plot_additional_maps(
        self,
        partition: str,
        tasks: list[Task],
        model: ConvNP,
        truth_ds: xr.Dataset,
        mask_ds: xr.DataArray | None,
        epoch: int,
    ) -> None:
        """Log prediction map visualizations for a set of tasks to W&B.

        Computes predictions over ``tasks`` at resolution
        ``self.res_factor_maps``, then logs (1) a multi-date grid of
        ground truth / predicted mean / predicted std / error maps via
        :meth:`_log_wandb_predictions_task`, and (2) a
        temporally-averaged prediction plot via
        `giant_sp.utils.plots.averaged_prediction_plot`, both under W&B
        keys prefixed with ``partition``.

        Parameters
        ----------
        partition : str
            Label identifying the data split (e.g. ``"val"`` or
            ``"stnval"``), used in plot titles and W&B log keys.
        tasks : list[deepsensor.data.task.Task]
            Tasks to predict on and plot.
        model : deepsensor.model.ConvNP
            Model used to generate predictions.
        truth_ds : xarray.Dataset
            Ground-truth dataset used for error computation and plotting.
        mask_ds : xarray.DataArray | None
            Optional target mask passed through to the prediction call.
        epoch : int
            Epoch number used as the W&B logging step.

        """
        target_var = self.model.config["data"]["target_var_IDs"]
        pred, _ = compute_pred_ds(
            model,
            tasks,
            truth_ds,
            target_var,
            self.pred_params,
            mask=mask_ds,
            res_factor=self.res_factor_maps,
        )
        mean_ds = pred[target_var]["mean"]
        std_ds = pred[target_var]["std"]
        # Multiple dates prediction plots
        self._log_wandb_predictions_task(
            partition,
            tasks,
            truth_ds[target_var],
            mean_ds,
            std_ds,
            epoch,
            cmap=self.cmap,
            crs=self.projection,
            plot_xc=False,
        )
        # Averaged prediction plots
        fig, _ax = myplt.averaged_prediction_plot(
            truth_ds[target_var],
            mean_ds,
            std_ds,
            cmap=self.cmap,
            crs=self.projection,
            highlight_negatives=False,
        )
        wandb.log(
            {f"predictions/plot_{partition}_average": wandb.Image(fig)}, step=epoch
        )
        plt.close(fig)

    def _plot_additional_charts(
        self,
        partition: str,
        tasks: list[Task],
        truth_ds: xr.Dataset,
        val_pred: Prediction,
        epoch: int,
    ) -> None:
        """Log parity and calibration charts for a random sample of dates to W&B.

        Randomly samples up to 30 dates (fixed seed ``42``) from the
        timestamps of ``tasks``, flattens the ground-truth and predicted
        mean values at those dates, and logs a hexbin parity plot
        (`giant_sp.utils.plots.hexbin_parity_plot`) and a PIT calibration
        histogram (`giant_sp.utils.plots.PIT_histogram`) to W&B.

        Parameters
        ----------
        partition : str
            Label identifying the data split (e.g. ``"val"`` or
            ``"stnval"``), used in W&B log keys.
        tasks : list[deepsensor.data.task.Task]
            Tasks whose ``"time"`` entries define the candidate dates to
            sample from.
        truth_ds : xarray.Dataset
            Ground-truth dataset used for the parity plot.
        val_pred : deepsensor.model.pred.Prediction
            Predictions (containing at least ``"mean"``) used for the
            parity and PIT plots.
        epoch : int
            Epoch number used as the W&B logging step.

        """
        target_var = self.model.config["data"]["target_var_IDs"]
        val_dates = [tasks[i]["time"] for i in range(len(tasks))]
        # Randomly select dates
        n_dates = min(len(val_dates), 30)
        np.random.seed(42)
        dates_plt = np.random.choice(val_dates, n_dates, replace=False)
        # Subselect and flatten datasets
        true_plt = truth_ds[target_var].sel(time=dates_plt).values.flatten()
        pred_plt = val_pred[target_var].sel(time=dates_plt)
        pred_mean_plt = pred_plt["mean"].values.flatten()

        # Parity plot (hexbin)
        fig, _ = myplt.hexbin_parity_plot(true_plt, pred_mean_plt)
        wandb.log(
            {f"predictions/hexbin_{partition}_{n_dates}_dates": wandb.Image(fig)},
            step=epoch,
        )
        plt.close(fig)
        # PIT histogram (uncertainty calibration)
        fig, _ = myplt.PIT_histogram(
            true_plt, pred_plt, likelihood=self.model.config["model"]["likelihood"]
        )
        wandb.log(
            {f"predictions/pit_{partition}_{n_dates}_dates": wandb.Image(fig)},
            step=epoch,
        )
        plt.close(fig)

    def _plot_additional_charts_forecast(
        self, rmse_lt: xr.DataArray, epoch: int
    ) -> None:
        """Log a forecast-skill (RMSE vs. lead time) chart to W&B.

        Parameters
        ----------
        rmse_lt : xarray.DataArray
            RMSE values indexed by a ``lead_time`` (timedelta) dimension.
        epoch : int
            Epoch number used as the W&B logging step.

        """
        fig, ax = plt.subplots(figsize=(4, 3))
        rmse_lt.plot(ax=ax, marker="o")  # type: ignore[call-arg]
        lead_time_nanoseconds = rmse_lt.lead_time.data.astype(int)
        ax.set_xticks(lead_time_nanoseconds)
        lead_time_days = rmse_lt.lead_time.dt.days.data
        ax.set_xticklabels(lead_time_days)
        ax.set_ylabel("RMSE (m $d^-1$)")
        ax.set_xlabel("Lead time (months)")
        ax.set_title("Forecast skill")
        wandb.log({"predictions/forecastskill": wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    def _log_wandb_metrics(
        self, epoch: int, train_loss: float, val_loss: float
    ) -> None:
        """Log the current epoch's train/val loss and learning rate to W&B.

        Parameters
        ----------
        epoch : int
            Current epoch number, used both as a logged field and as the
            W&B logging step.
        train_loss : float
            Training loss for the current epoch.
        val_loss : float
            Validation loss for the current epoch.

        """
        # Get current learning rate from optimizer
        current_lr = self.optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "learning_rate": current_lr,
            },
            step=epoch,
        )

    def _log_wandb_predictions_task(
        self,
        partition: str,
        tasks: list[Task],
        truth_ds: xr.DataArray,
        mean_da: xr.DataArray,
        std_da: xr.DataArray,
        epoch: int,
        cmap: str = "seismic",
        crs: ccrs.CRS | None = None,
        plot_xc: bool = True,
    ) -> None:
        """Generate and log a grid of per-date prediction maps to W&B.

        For either the configured ``self.plot_dates`` or (if unset) three
        randomly sampled dates (fixed seed ``0``), builds one row of four
        maps — ground truth, predicted mean, predicted std, and error —
        for the matching task, optionally overlaying context observation
        locations on the ground-truth panel, and logs the resulting
        figure to W&B. Any exception raised while building or logging the
        figure is caught and logged as a warning rather than propagated.

        Parameters
        ----------
        partition : str
            Label identifying the data split (e.g. ``"val"`` or
            ``"stnval"``), used in the W&B log key.
        tasks : list[deepsensor.data.task.Task]
            Tasks to search for the ones matching the plotted dates (used
            to read context locations and context count ``N_c``).
        truth_ds : xarray.DataArray
            Ground-truth data array for the target variable.
        mean_da : xarray.DataArray
            Predicted mean data array for the target variable.
        std_da : xarray.DataArray
            Predicted standard-deviation data array for the target
            variable.
        epoch : int
            Epoch number used as the W&B logging step.
        cmap : str, optional
            Colormap used for the ground-truth and mean panels. Default
            is ``"seismic"``.
        crs : cartopy.crs.CRS | None, optional
            Map projection for the subplots. If ``None``, defaults to
            `cartopy.crs.PlateCarree`. Default is ``None``.
        plot_xc : bool, optional
            Whether to overlay off-grid context locations on the
            ground-truth panel. Default is ``True``.

        """
        if crs is None:
            crs = ccrs.PlateCarree()

        assert self.model.model is not None, "model.model must be built before training"

        try:
            # Make predictions
            ### Prepare tasks =============================================================
            if self.plot_dates is None:
                random.seed(0)
                dates_all = [t["time"] for t in tasks]
                dates = [
                    dates_all[i]
                    for i in sorted(random.sample(range(len(dates_all)), 3))
                ]
            else:
                dates = self.plot_dates
                dates = [pd.Timestamp(d) for d in dates]

            mean_da, truth_da = xr.align(mean_da, truth_ds, join="inner")
            # Calculate prediction errors
            err_da = mean_da - truth_da

            fig1, axes = plt.subplots(
                nrows=len(dates),
                ncols=4,
                figsize=(4 * self.fig_size, len(dates) * self.fig_size),
                subplot_kw={"projection": crs},
            )

            fig1.suptitle(
                f"{truth_da.name} predictions multiple dates - Epoch: {epoch}"
            )

            for row_i, date in enumerate(dates):
                task = next(t for t in tasks if t["time"] == date)
                N_c = task["X_c"][0].shape[1]

                vmin = min(
                    mean_da.mean(dim="time").min().compute(),
                    truth_da.mean(dim="time").min().compute(),
                )
                vmax = max(
                    mean_da.mean(dim="time").max().compute(),
                    truth_da.mean(dim="time").max().compute(),
                )
                if self.model.config["data"]["target_var_IDs"][-3:] == "anm":
                    maxabs = np.max(np.abs([vmin, vmax]))
                    var_clim = (-maxabs, maxabs)
                else:
                    var_clim = (vmin, vmax)

                ax = axes[row_i, 0]
                truth_da.sel(time=date).plot(  # type: ignore[call-arg]
                    ax=ax, center=False, cmap=cmap, vmin=var_clim[0], vmax=var_clim[1]
                )
                ax.coastlines(linewidth=0.5)
                ax.set_title(f"Ground Truth \n Date: {date}'")

                ax = axes[row_i, 1]
                mean_da.sel(time=date).plot(  # type: ignore[call-arg]
                    ax=ax, center=False, cmap=cmap, vmin=var_clim[0], vmax=var_clim[1]
                )
                ax.coastlines(linewidth=0.5)
                ax.set_title(f"ConvNP mean \n 'N_context: {N_c}")

                ax = axes[row_i, 2]
                std_da.sel(time=date).plot(  # type: ignore[call-arg]
                    ax=ax, center=False, cmap="Greys"
                )
                ax.coastlines(linewidth=0.5)
                ax.set_title("ConvNP uncertainty (std)")

                ax = axes[row_i, 3]
                err_da.sel(time=date).plot(  # type: ignore[call-arg]
                    ax=ax, center=False, cmap="RdBu_r"
                )
                ax.coastlines(linewidth=0.5)
                ax.set_title("ConvNP error")

                # Plot context locations on ground truth plot
                if plot_xc:
                    dsplt.offgrid_context(
                        axes[row_i, 0],
                        task,
                        self.model.model.data_processor,
                        s=3,
                        markers="x",
                        linewidths=0.5,
                        colors="green",
                        add_legend=False,
                    )

            plt.tight_layout()

            # Log to wandb
            wandb.log(
                {
                    f"predictions/plot_{partition}_dates": wandb.Image(fig1),
                },
                step=epoch,
            )

            plt.close(fig1)

        except Exception as e:
            logger.warning("Failed to log predictions to W&B: %s", e)
