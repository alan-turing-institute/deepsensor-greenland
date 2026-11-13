"""Tests for the Hydra entry-point configs in ``src/ds_greenland/configs/``."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import ds_greenland

CONFIG_DIR = Path(ds_greenland.__file__).parent / "configs"
ENTRY_CONFIGS = ["base", "inference", "placement", "placement_eval"]
EXPERIMENTS = sorted(
    p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml") if p.stem != "common"
)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("config_name", ENTRY_CONFIGS)
def test_entry_config_uses_ds_greenland_logging(
    config_name: str, experiment: str
) -> None:
    """Every entry config/experiment pair logs with our format at `logging.log_level`."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=[f"experiment={experiment}", "logging.log_level=DEBUG"],
            return_hydra_config=True,
        )

    job_logging = cfg.hydra.job_logging
    assert job_logging.root.level == "DEBUG"
    assert job_logging.formatters.simple.format == (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


PATH_ROOT = "/tmp/data-root"
PROD_EXPERIMENTS = [e for e in EXPERIMENTS if not e.endswith("_debug")]
DATA_PATH_KEYS = ["data_path", "ancillary_path", "station_locations_file"]
PREPROCESSING_PATH_KEYS = {
    "hirham5_create_basefile": ["interim_path", "output_dir"],
    "preprocess_station_data": ["input_file", "output_file"],
}


@pytest.mark.parametrize("experiment", PROD_EXPERIMENTS)
@pytest.mark.parametrize("config_name", [*ENTRY_CONFIGS, "placement_pareto"])
def test_experiment_data_paths_follow_paths_root(
    config_name: str, experiment: str
) -> None:
    """Every production data path is built from `paths.root`, so overriding it moves them all."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=[f"experiment={experiment}", f"paths.root={PATH_ROOT}"],
        )

    data = OmegaConf.to_container(cfg.data, resolve=True)
    assert isinstance(data, dict)
    for key in DATA_PATH_KEYS:
        assert data[key].startswith(f"{PATH_ROOT}/"), key
    assert "/gws/" not in OmegaConf.to_yaml(cfg, resolve=True)


@pytest.mark.parametrize(("config_name", "keys"), PREPROCESSING_PATH_KEYS.items())
def test_preprocessing_paths_follow_paths_root(
    config_name: str, keys: list[str]
) -> None:
    """The preprocessing scripts' input/output paths are built from `paths.root` too."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=[f"paths.root={PATH_ROOT}"])

    preprocessing = OmegaConf.to_container(cfg.preprocessing, resolve=True)
    assert isinstance(preprocessing, dict)
    for key in keys:
        assert preprocessing[key].startswith(f"{PATH_ROOT}/"), key
    assert "/gws/" not in OmegaConf.to_yaml(cfg, resolve=True)
