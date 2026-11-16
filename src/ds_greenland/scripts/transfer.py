"""Transfer learning: fine-tune a pretrained DeepSensor model on a new dataset.

Loads a model trained by ``interpolation.py`` from its W&B run, rebuilds it with
exactly the pretrained architecture, and keeps training it on the dataset of the
selected ``experiment``. The run is saved to W&B in the same format as
``interpolation.py``'s, so ``inference.py``/``placement.py`` (and this script, to
chain transfers) can load the fine-tuned model from its new run.
"""

import glob
import json
import logging
import os
from collections.abc import Sequence
from typing import Any

import deepsensor.torch  # noqa: F401
import hydra
import numpy as np
import torch
import wandb
from deepsensor.data import DataProcessor, Task, TaskLoader
from deepsensor.model import ConvNP
from deepsensor.model.defaults import compute_greatest_data_density
from deepsensor.train import set_gpu_default_device
from omegaconf import DictConfig, OmegaConf, open_dict

from ds_greenland.models import DeepSensorModel
from ds_greenland.scripts.inference import download_wandb_file
from ds_greenland.scripts.interpolation import prepare_training_data
from ds_greenland.training import Trainer
from ds_greenland.utils.config import to_plain

logger = logging.getLogger(__name__)

MODEL_CONFIG_FILE = "model_config.json"
DATA_PROCESSOR_CONFIG_FILE = "data_processor_config.json"


def fetch_pretrained_files(run_ref: str, weights_filename: str) -> str:
    """Return a local directory holding a W&B run's model config, data processor and weights.

    Reuses the newest local ``wandb/run*<run_id>/files`` directory if it already
    holds all three files; otherwise downloads them into
    ``wandb/run-<run_id>/files``.

    Parameters
    ----------
    run_ref : str
        Fully qualified W&B run reference, e.g. ``"entity/project/run_id"``.
    weights_filename : str
        Name of the weights checkpoint within the run's files (e.g.
        ``"best-weights.pt"``).

    Returns
    -------
    str
        Directory containing ``model_config.json``,
        ``data_processor_config.json`` and `weights_filename`.

    """
    run_id = run_ref.split("/")[-1]
    target_files = [MODEL_CONFIG_FILE, DATA_PROCESSOR_CONFIG_FILE, weights_filename]

    cached = sorted(glob.glob(os.path.join("wandb", f"run*{run_id}", "files")))
    if cached and all(
        os.path.exists(os.path.join(cached[-1], f)) for f in target_files
    ):
        return cached[-1]

    root_path = os.path.join("wandb", f"run-{run_id}")
    os.makedirs(root_path, exist_ok=True)
    for f in target_files:
        download_wandb_file(run_ref, root_path, f)
    return os.path.join(root_path, "files")


def load_pretrained_model_config(path: str) -> dict[str, Any]:
    """Read a pretrained run's ``model_config.json``.

    Parameters
    ----------
    path : str
        Path to the ``model_config.json`` saved by `Trainer` for the run. It
        holds every argument the ``ConvNP`` was built with (including the
        data-derived ``dim_yc``/``dim_yt``/``internal_density``/scales), so
        passing it back to ``ConvNP`` rebuilds exactly the same architecture.

    Returns
    -------
    dict[str, Any]
        The model config.

    """
    with open(path) as f:
        config: dict[str, Any] = json.load(f)
    return config


def check_task_loader_compatible(
    model_config: dict[str, Any], task_loader: TaskLoader
) -> None:
    """Check the new dataset has the context/target structure the pretrained model expects.

    Parameters
    ----------
    model_config : dict[str, Any]
        The pretrained model's config (see `load_pretrained_model_config`).
    task_loader : deepsensor.data.TaskLoader
        Task loader built from the new dataset.

    Raises
    ------
    ValueError
        If the number of context sets, the number of variables in any of them,
        the target dimension or the auxiliary-at-target dimension differs from
        the pretrained model's.

    """
    expected = {
        "dim_yc": list(model_config["dim_yc"]),
        "dim_yt": model_config["dim_yt"],
        "dim_aux_t": model_config["dim_aux_t"],
    }
    actual = {
        "dim_yc": list(task_loader.context_dims),
        "dim_yt": sum(task_loader.target_dims),
        "dim_aux_t": task_loader.aux_at_target_dims,
    }
    mismatches = [
        f"{key}: pretrained {expected[key]}, new dataset {actual[key]}"
        for key in expected
        if expected[key] != actual[key]
    ]
    if mismatches:
        raise ValueError(
            "The new dataset doesn't match the pretrained model's inputs/outputs ("
            + "; ".join(mismatches)
            + "). Its experiment's data.context_var_IDs, data.target_var_IDs and "
            "dataloader.context_delta_t must have the same structure (number of "
            "context sets, variables per set, time offsets) as the pretrained run's."
        )


def freeze_parameters(module: torch.nn.Module, prefixes: Sequence[str]) -> list[str]:
    """Stop the parameters under the given name prefixes from being trained.

    A prefix matches a parameter whose dotted name equals it or starts with it
    followed by ``"."``, so ``"decoder.links.0.before_turn_layers.1"`` doesn't
    also match ``"...before_turn_layers.10"``.

    Parameters
    ----------
    module : torch.nn.Module
        The model's torch module (``ConvNP.model``).
    prefixes : Sequence[str]
        Dotted parameter-name prefixes to freeze, e.g.
        ``"decoder.links.0.before_turn_layers"`` for the UNet's downsampling
        path.

    Returns
    -------
    list[str]
        Names of the parameters that were frozen.

    Raises
    ------
    ValueError
        If any prefix matches no parameter (most likely a typo).

    """

    def matches(name: str, prefix: str) -> bool:
        return name == prefix or name.startswith(f"{prefix}.")

    names = [name for name, _ in module.named_parameters()]
    unmatched = [p for p in prefixes if not any(matches(n, p) for n in names)]
    if unmatched:
        raise ValueError(
            f"No model parameters match the freeze prefix(es) {unmatched}. "
            f"Parameter names are: {names}"
        )

    frozen = []
    for name, param in module.named_parameters():
        if any(matches(name, p) for p in prefixes):
            param.requires_grad_(False)
            frozen.append(name)
    return frozen


def mean_task_loss(model: ConvNP, tasks: Sequence[Task]) -> float:
    """Mean normalised loss over `tasks`, as `Trainer` computes its validation loss.

    Parameters
    ----------
    model : deepsensor.model.ConvNP
        Model to evaluate.
    tasks : Sequence[deepsensor.data.Task]
        Tasks to evaluate it on.

    Returns
    -------
    float
        Mean of ``model.loss_fn(task, normalise=True)`` over `tasks`.

    """
    with torch.no_grad():
        losses = [
            model.loss_fn(task, normalise=True).detach().cpu().numpy() for task in tasks
        ]
    return float(np.mean(losses))


@hydra.main(version_base=None, config_path="../configs", config_name="transfer")
def main(config: DictConfig) -> None:
    """Fine-tune a pretrained DeepSensor model on the experiment's dataset.

    Fetches the pretrained run's model config, data processor and weights
    from W&B; builds the training/validation tasks for the selected
    experiment's dataset, normalised with the pretrained data processor; then
    rebuilds the pretrained model, loads its weights, optionally freezes some
    of its layers and logs its zero-shot validation loss, and fine-tunes it
    with `Trainer.train`.

    Parameters
    ----------
    config : DictConfig
        Hydra-composed run configuration (see ``src/ds_greenland/configs/``).
        Its ``model`` section is replaced with the pretrained run's
        architecture. ``config.transfer`` holds the run-specific parameters:

        run_ref : str
            W&B run reference (``"entity/project/run_id"``) of the pretrained
            model.
        weights_filename : str
            Weights checkpoint within that run's files to start from.
        refit_normalisation : bool
            If false, normalise the new dataset with the pretrained run's
            variable statistics. If true, fit them on the new dataset instead
            (the coordinate normalisation is always the pretrained run's).
        freeze : list[str]
            Parameter-name prefixes to freeze (see `freeze_parameters`).
        zero_shot_eval : bool
            If set, log the pretrained model's loss on the new validation
            tasks before fine-tuning, as a baseline.

    Returns
    -------
    None
        Training progress and results are logged and checkpointed as a
        side effect (see `Trainer.train`); nothing is returned.

    Raises
    ------
    ValueError
        If the new dataset's structure doesn't match the pretrained model's
        (see `check_task_loader_compatible`), if a freeze prefix matches no
        parameters, or for any of the reasons `prepare_training_data` raises.

    """
    transfer_config = config["transfer"]
    run_ref = transfer_config["run_ref"]
    weights_fn = transfer_config["weights_filename"]
    resume = config["training"].get("resume", False)

    # Fetch the pretrained run first: its architecture replaces the experiment's
    # `model` section, which W&B records below and `Trainer` reads.
    files_path = fetch_pretrained_files(run_ref, weights_fn)
    logger.info("Loading pretrained run %s from: %s", run_ref, files_path)
    pretrained_model_config = load_pretrained_model_config(
        os.path.join(files_path, MODEL_CONFIG_FILE)
    )
    with open_dict(config):
        config["model"] = pretrained_model_config
    logger.info(
        "Using the pretrained model's architecture; the experiment's `model` "
        "section is ignored"
    )

    # Initialize W&B early to track the full run (data loading, model build, training)
    wandb_run = None
    use_wandb = config["logging"].get("wandb", {}).get("enabled", False)
    if use_wandb and not resume:
        wandb_cfg = config["logging"]["wandb"]
        wandb_run = wandb.init(
            project=wandb_cfg.get("project", "deepsensor-spatial-interpolation"),
            entity=wandb_cfg.get("entity"),
            name=wandb_cfg.get("name"),
            tags=[*wandb_cfg.get("tags", []), "transfer-learning"],
            notes=wandb_cfg.get("notes", ""),
            config={
                "model": to_plain(config["model"]),
                "training": to_plain(config["training"]),
                "data": to_plain(config["data"]),
                "transfer": to_plain(transfer_config),
            },
        )
        logger.info("Initialized W&B run: %s", wandb_run.name)
        logger.info("W&B URL: %s", wandb_run.url)

    logger.info(
        "Active devices before updating default: %s", torch.get_default_device()
    )
    if torch.cuda.is_available():
        set_gpu_default_device()

    logger.info("Active devices after updating default: %s", torch.get_default_device())
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    # Normalise the new data the way the pretrained model saw its training data.
    data_processor = DataProcessor(files_path)
    pretrained_vars = set(data_processor.config) - {"coords"}
    if transfer_config["refit_normalisation"]:
        for var_ID in pretrained_vars:
            del data_processor.config[var_ID]
        logger.info(
            "Fitting variable normalisation on the new dataset (keeping the "
            "pretrained coordinate normalisation)"
        )

    data = prepare_training_data(config, data_processor=data_processor)

    if not transfer_config["refit_normalisation"]:
        fitted_vars = sorted(
            set(data.data_processor.config) - {"coords"} - pretrained_vars
        )
        if fitted_vars:
            logger.warning(
                "No pretrained normalisation statistics for %s; fitted them on "
                "the new dataset",
                fitted_vars,
            )

    check_task_loader_compatible(pretrained_model_config, data.task_loader)
    new_density = compute_greatest_data_density(data.task_loader)
    if new_density != pretrained_model_config["internal_density"]:
        logger.warning(
            "The new dataset's grid density (%d points per unit) differs from the "
            "pretrained model's internal_density (%d); keeping the pretrained value "
            "so its convolutions keep the spatial scale they were trained at",
            new_density,
            pretrained_model_config["internal_density"],
        )

    # Build the pretrained model
    logger.info("Building model...")
    model_wrapper = DeepSensorModel(config)
    model = model_wrapper.build_model(data.data_processor, data.task_loader)
    model_wrapper.load_model(os.path.join(files_path, weights_fn))

    freeze = list(transfer_config["freeze"])
    if freeze:
        frozen = freeze_parameters(model.model, freeze)
        logger.info("Froze %d parameter tensors under %s", len(frozen), freeze)
    logger.info("Trainable model parameters: %d", model_wrapper.count_parameters())

    # When resuming, `Trainer` replaces these weights with the run's last checkpoint.
    if transfer_config["zero_shot_eval"] and not resume:
        logger.info("Computing zero-shot loss of the pretrained model...")
        zero_shot = {"zero_shot/val_loss": mean_task_loss(model, data.val_tasks)}
        if config["logging"]["log_stnval"]:
            zero_shot["zero_shot/stnval_loss"] = mean_task_loss(
                model, data.stnval_tasks
            )
        for key, value in zero_shot.items():
            logger.info("%s: %.6f", key, value)
        if wandb_run is not None:
            wandb_run.summary.update(zero_shot)

    trainer = Trainer(model_wrapper, config, wandb_run=wandb_run)

    logger.info("Starting fine-tuning...")
    trainer.train(
        data.truth_ds,
        data.task_stream_tr,
        data.val_tasks,
        mask=data.lsm,
        stnval_tasks=data.stnval_tasks,
    )

    logger.info("Fine-tuning complete!")
    logger.info("Best validation loss: %.6f", trainer.best_val_loss)


if __name__ == "__main__":
    main()
