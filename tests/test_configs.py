"""Tests for the Hydra entry-point configs in ``src/giant_sp/configs/``."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import giant_sp

CONFIG_DIR = Path(giant_sp.__file__).parent / "configs"
ENTRY_CONFIGS = ["base", "inference", "placement", "placement_eval"]
EXPERIMENTS = sorted(
    p.stem for p in (CONFIG_DIR / "experiment").glob("*.yaml") if p.stem != "common"
)


@pytest.mark.parametrize("experiment", EXPERIMENTS)
@pytest.mark.parametrize("config_name", ENTRY_CONFIGS)
def test_entry_config_uses_giant_sp_logging(config_name: str, experiment: str) -> None:
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
