import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from giant_sp.data.dataloader import (
    BaseDataLoader,
    CARRA2DataLoader,
    HIRHAM5DataLoader,
    HIRHAM5ZarrDataLoader,
)


def _make_config(
    bm_ancils_path=None, basin=None, data_path="/nonexistent", data_pattern="*.nc"
):
    config = {
        "data": {
            "data_path": data_path,
            "data_pattern": data_pattern,
            "region": "test_region",
            "basin": basin,
            "context_var_IDs": ["var"],
            "target_var_IDs": ["var"],
            "train_start": "2020-01-01",
            "train_end": "2020-01-03",
            "val_start": "2020-01-04",
            "val_end": "2020-01-05",
        },
        "dataloader": {"lsm_mask": False, "frequency": "D"},
    }
    if bm_ancils_path is not None:
        config["data"]["bedmachine_ancillary_path"] = str(bm_ancils_path)
    return config


def _write_basin_netcdf(path, se_code=1, other_code=2):
    """A synthetic subregion mask matching what CARRA2/HIRHAM5 load_basin expects:
    a 2D (y, x) `subregion` DataArray whose `subregion_labels` attr JSON-encodes
    a {code: label} mapping."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array(
        [
            [other_code, other_code, other_code, other_code],
            [other_code, se_code, se_code, other_code],
            [other_code, se_code, se_code, other_code],
            [other_code, other_code, se_code, other_code],
            [other_code, other_code, other_code, other_code],
        ]
    )
    subregion = xr.DataArray(
        values, coords={"y": y, "x": x}, dims=("y", "x"), name="subregion"
    )
    subregion.attrs["subregion_labels"] = json.dumps(
        {str(se_code): "SE", str(other_code): "NW"}
    )
    subregion.to_dataset().to_netcdf(path)


class TestBaseDataLoaderInit:
    def test_reads_config_into_attributes(self):
        config = _make_config()
        loader = BaseDataLoader(config)
        assert str(loader.source) == "/nonexistent"
        assert loader.region == "test_region"
        assert loader.basin is None
        assert loader.context_vars == ["var"]
        assert loader.target_vars == ["var"]
        assert loader.frequency == "D"
        assert loader.variables == []


class TestBaseDataLoaderAbstractMethods:
    def test_load_ds_not_implemented(self):
        loader = BaseDataLoader(_make_config())
        with pytest.raises(NotImplementedError, match="load_ds"):
            loader.load_ds()

    def test_load_aux_ds_not_implemented(self):
        loader = BaseDataLoader(_make_config())
        with pytest.raises(NotImplementedError, match="load_aux_ds"):
            loader.load_aux_ds()

    def test_get_dataproc_taskload_not_implemented(self):
        loader = BaseDataLoader(_make_config())
        with pytest.raises(NotImplementedError, match="get_dataproc_taskload"):
            loader.get_dataproc_taskload()


class TestInitDataProcessor:
    def test_uses_lat_lon_as_spatial_dims(self):
        loader = BaseDataLoader(_make_config())
        data_processor = loader.init_data_processor()
        assert data_processor.x1_name == "lat"
        assert data_processor.x2_name == "lon"


class TestAddSpatiotemporalAux:
    def test_adds_x1_x2_and_circular_time_variables(self):
        loader = BaseDataLoader(_make_config())

        x1 = np.array([0.0, 1.0])
        x2 = np.array([0.0, 1.0, 2.0])
        aux_ds = xr.Dataset(
            coords={"x1": x1, "x2": x2},
            data_vars={"lsm": (("x1", "x2"), np.ones((2, 3)))},
        )
        ref_ds = xr.Dataset(coords={"time": pd.date_range("2020-01-01", "2020-01-05")})

        result = loader.add_spatiotemporal_aux(aux_ds, ref_ds)

        assert "x1_arr" in result.data_vars
        assert "x2_arr" in result.data_vars
        assert "cos_D" in result.data_vars
        assert "sin_D" in result.data_vars
        np.testing.assert_array_equal(result["x1_arr"].values, [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(result["x2_arr"].values, [[0, 1, 2], [0, 1, 2]])
        assert result.sizes["time"] == 5


@pytest.mark.parametrize(
    ("loader_cls", "basin_filename"),
    [
        (CARRA2DataLoader, "basins_2500m.nc"),
        (HIRHAM5DataLoader, "basins_5500m.nc"),
    ],
)
class TestLoadBasin:
    def test_selects_x_y_range_for_target_basin(
        self, tmp_path, loader_cls, basin_filename
    ):
        _write_basin_netcdf(tmp_path / basin_filename, se_code=1, other_code=2)
        config = _make_config(bm_ancils_path=tmp_path, basin="SE")

        loader = loader_cls(config)

        np.testing.assert_array_equal(loader.basin_x.values, [1.0, 2.0])
        np.testing.assert_array_equal(loader.basin_y.values, [1.0, 2.0, 3.0])

    def test_skips_basin_loading_when_basin_is_none(
        self, tmp_path, loader_cls, basin_filename
    ):
        config = _make_config(bm_ancils_path=tmp_path, basin=None)

        loader = loader_cls(config)

        assert not hasattr(loader, "basin_x")
        assert not hasattr(loader, "basin_y")


class TestHirham5ZarrLoadBasin:
    def test_selects_x_y_range_for_target_basin(self, tmp_path):
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # HIRHAM5ZarrDataLoader.load_basin hardcodes label_to_code = {..., "SE": 4, "NW": 6, ...}
        values = np.array(
            [
                [6, 6, 6, 6],
                [6, 4, 4, 6],
                [6, 4, 4, 6],
                [6, 6, 4, 6],
                [6, 6, 6, 6],
            ]
        )
        maskbas = xr.DataArray(
            values, coords={"y": y, "x": x}, dims=("y", "x"), name="maskbas"
        )
        ds = maskbas.to_dataset().stack(z=("y", "x")).reset_index("z")
        ds.to_zarr(tmp_path / "GRLmask.zarr", zarr_format=2)

        config = _make_config(
            bm_ancils_path=tmp_path,
            basin="SE",
            data_path=str(tmp_path),
            data_pattern="data.zarr",
        )
        loader = HIRHAM5ZarrDataLoader(config)

        np.testing.assert_array_equal(loader.basin_x.values, [1.0, 2.0])
        np.testing.assert_array_equal(loader.basin_y.values, [1.0, 2.0, 3.0])

    def test_skips_basin_loading_when_basin_is_none(self, tmp_path):
        config = _make_config(
            bm_ancils_path=tmp_path,
            basin=None,
            data_path=str(tmp_path),
            data_pattern="data.zarr",
        )

        loader = HIRHAM5ZarrDataLoader(config)

        assert not hasattr(loader, "basin_x")
        assert not hasattr(loader, "basin_y")
