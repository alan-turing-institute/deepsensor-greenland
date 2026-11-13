from pathlib import Path

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import ds_greenland
import ds_greenland.preprocessing.preprocess_station_data as psd

CONFIG_DIR = Path(ds_greenland.__file__).parent / "configs"
INST = psd.INSTALLATION_COLUMN
VALID = psd.LAST_VALID_COLUMN


@pytest.fixture
def stations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "site_id": ["A", "B", "C", "D", "E"],
            INST: ["1995-06-07", "2007-08-20", "2016-12-31", "2024-05-18", None],
            VALID: ["2026-01-01", "2015-08-13", "2026-01-01", "2025-07-28", None],
        }
    )


class TestFilterByDateRange:
    def test_both_bounds_inclusive(self, stations):
        out = psd.filter_by_date_range(stations, INST, "2007-08-20", "2016-12-31")
        assert out["site_id"].tolist() == ["B", "C"]

    def test_only_start(self, stations):
        out = psd.filter_by_date_range(stations, INST, start_date="2010-01-01")
        assert out["site_id"].tolist() == ["C", "D"]

    def test_only_end(self, stations):
        out = psd.filter_by_date_range(stations, INST, end_date="2010-01-01")
        assert out["site_id"].tolist() == ["A", "B"]

    def test_last_valid_column(self, stations):
        out = psd.filter_by_date_range(stations, VALID, start_date="2020-01-01")
        assert out["site_id"].tolist() == ["A", "C", "D"]

    def test_no_bounds_returns_unchanged(self, stations):
        out = psd.filter_by_date_range(stations, INST)
        assert out["site_id"].tolist() == ["A", "B", "C", "D", "E"]

    def test_start_after_end_raises(self, stations):
        with pytest.raises(ValueError, match="after end_date"):
            psd.filter_by_date_range(stations, INST, "2020-01-01", "2000-01-01")

    def test_missing_column_raises(self, stations):
        with pytest.raises(KeyError, match=INST):
            psd.filter_by_date_range(stations.drop(columns=INST), INST)


def test_preprocess_station_data_filters_both_columns(stations, tmp_path):
    input_file = tmp_path / "in.csv"
    output_file = tmp_path / "out" / "filtered.csv"
    stations.to_csv(input_file, index=False)
    params = OmegaConf.create(
        {
            "input_file": str(input_file),
            "output_file": str(output_file),
            "installation": {"start_date": None, "end_date": "2016-12-31"},
            "last_valid": {"start_date": "2020-01-01", "end_date": None},
        }
    )

    psd.preprocess_station_data(params)

    assert pd.read_csv(output_file)["site_id"].tolist() == ["A", "C"]


def test_config_composes():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="preprocess_station_data",
            overrides=[
                "preprocessing.installation.start_date=2000-01-01",
                "preprocessing.installation.end_date=2016-12-31",
                "preprocessing.last_valid.start_date=2020-01-01",
            ],
        )
    assert cfg.preprocessing.output_file.endswith(
        "promice_locations_inst_2000-01-01_2016-12-31_valid_2020-01-01_None.csv"
    )
