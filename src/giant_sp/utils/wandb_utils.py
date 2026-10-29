"""Weights & Biases helpers shared by the training, inference and placement scripts.

Covers starting a new W&B run from the ``logging.wandb`` config section, and
fetching/rebuilding a trained model from the files a training run logged
(see `Trainer`, which writes `MODEL_CONFIG_FILE` and
`DATA_PROCESSOR_CONFIG_FILE` to the run's ``files/`` directory).
"""

import glob
import json
import logging
import os
from typing import Any

import torch
import wandb
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from omegaconf import DictConfig
from wandb.sdk.wandb_run import Run

from giant_sp.models.mdn import MDNZIGammaConvNP, install_mdn_zigamma_likelihood
from giant_sp.utils.config import to_plain
from giant_sp.utils.model import get_available_device

logger = logging.getLogger(__name__)

MODEL_CONFIG_FILE = "model_config.json"
DATA_PROCESSOR_CONFIG_FILE = "data_processor_config.json"


def wandb_enabled(config: DictConfig) -> bool:
    """Return whether W&B logging is enabled in ``config["logging"]["wandb"]``.

    Parameters
    ----------
    config : DictConfig
        Full run configuration.

    Returns
    -------
    bool
        ``config["logging"]["wandb"]["enabled"]``, or ``False`` if the
        ``wandb`` section or its ``enabled`` key is missing.

    """
    return bool(config["logging"].get("wandb", {}).get("enabled", False))


def init_wandb_run(config: DictConfig, run_config: dict[str, Any] | None = None) -> Run:
    """Start a new W&B run configured from ``config["logging"]["wandb"]``.

    Parameters
    ----------
    config : DictConfig
        Full run configuration. ``logging.wandb`` supplies the
        ``project``/``entity``/``name``/``tags``/``notes`` of the run, and
        the ``model``/``training``/``data`` sections are logged as the run's
        config.
    run_config : dict[str, Any] | None, optional
        Extra entries merged into the logged run config (e.g.
        ``{"model_parameters": ...}``). Default is None.

    Returns
    -------
    wandb.sdk.wandb_run.Run
        The newly started run.

    """
    wandb_cfg = config["logging"]["wandb"]
    run = wandb.init(
        project=wandb_cfg.get("project", "deepsensor-spatial-interpolation"),
        entity=wandb_cfg.get("entity"),
        name=wandb_cfg.get("name"),
        tags=wandb_cfg.get("tags", []),
        notes=wandb_cfg.get("notes", ""),
        config={
            "model": to_plain(config["model"]),
            "training": to_plain(config["training"]),
            "data": to_plain(config["data"]),
            **(run_config or {}),
        },
    )
    logger.info("Initialized W&B run: %s", run.name)
    logger.info("W&B URL: %s", run.url)
    return run


def download_wandb_file(run_ref: str, files_path: str, file: str) -> None:
    """Download a single file logged to a Weights & Biases run.

    Parameters
    ----------
    run_ref : str
        Fully qualified W&B run reference, e.g. ``"entity/project/run_id"``.
    files_path : str
        Local directory to download the file into (passed as ``root`` to
        the W&B download call).
    file : str
        Name of the file to download, relative to the run's ``files/``
        directory (e.g. ``"model_config.json"``).

    """
    api = wandb.Api()
    therun = api.run(run_ref)

    therun.file(os.path.join("files", file)).download(root=files_path, exist_ok=True)


def fetch_run_files(run_ref: str, files: list[str]) -> str:
    """Locate a W&B run's files locally, downloading them if any are missing.

    Looks for the most recent local ``wandb/run*<run_id>/files`` directory
    containing all of `files`; if there is none, downloads each file into
    ``wandb/run-<run_id>/files``.

    Parameters
    ----------
    run_ref : str
        Fully qualified W&B run reference, e.g. ``"entity/project/run_id"``.
    files : list[str]
        Names of the files required, relative to the run's ``files/``
        directory.

    Returns
    -------
    str
        Local directory containing all of `files`.

    """
    run_id = run_ref.split("/")[-1]
    try:
        files_path = sorted(glob.glob(f"wandb/run*{run_id}/files"), reverse=True)[0]
        if not all(os.path.exists(os.path.join(files_path, f)) for f in files):
            raise IndexError
    except IndexError:
        root_path = os.path.join("wandb", f"run-{run_id}")
        os.makedirs(root_path, exist_ok=True)
        for f in files:
            download_wandb_file(run_ref, root_path, f)
        files_path = os.path.join(root_path, "files")
    logger.info("Loading run data from: %s", files_path)
    return files_path


def load_wandb_model(
    data_processor: DataProcessor,
    task_loader: TaskLoader,
    config_path: str,
    weights_path: str,
) -> MDNZIGammaConvNP:
    """Rebuild a ConvNP model from a saved config and load its trained weights.

    Parameters
    ----------
    data_processor : DataProcessor
        Fitted data processor used by the model to (de)normalise inputs
        and outputs.
    task_loader : TaskLoader
        Task loader describing the context/target variables the model
        was trained on.
    config_path : str
        Path to the model's JSON config file (`MODEL_CONFIG_FILE` as saved
        by W&B), containing ``unet_channels``, ``unet_kernels``,
        ``likelihood`` and ``num_basis_functions``.
    weights_path : str
        Path to the PyTorch checkpoint (``.pt`` file) with the trained
        model weights.

    Returns
    -------
    MDNZIGammaConvNP
        A ConvNP model built from `config_path` with the weights from
        `weights_path` loaded, mapped onto CUDA if available and
        otherwise the CPU. Always an `MDNZIGammaConvNP` (a `ConvNP`
        subclass that behaves as a plain `ConvNP` for likelihoods other
        than ``"mdn-zigamma"``), so ``p_zero``/``occurrence_entropy`` are
        available for ``"bernoulli-gamma"`` models too.

    """
    with open(config_path) as f:
        config = json.load(f)

    # Build model from config
    model_args = {
        "unet_channels": config["unet_channels"],
        "unet_kernels": config["unet_kernels"],
        "likelihood": config["likelihood"],
        "num_basis_functions": config["num_basis_functions"],
    }

    if config["likelihood"] == "mdn-zigamma":
        model_args["mixture_components"] = config["mixture_components"]
        install_mdn_zigamma_likelihood(config["mixture_components"])
    # MDNZIGammaConvNP is used for "bernoulli-gamma" too (not just "mdn-zigamma"): its
    # _set_num_mixture_components falls back to the base ConvNP behaviour for any other
    # likelihood, so this is a no-op for "bernoulli-gamma" beyond additionally exposing
    # p_zero/occurrence_entropy (both acquisition functions need), which share the same
    # SpikesSlab outer mixture as "mdn-zigamma".
    model = MDNZIGammaConvNP(data_processor, task_loader, **model_args)

    # Load model weights
    model.model.load_state_dict(
        torch.load(weights_path, map_location=get_available_device())
    )

    return model
