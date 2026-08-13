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
    ancils_path=None, basin=None, data_path="/nonexistent", data_pattern="*.nc"
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
        "dataloader": {
            "lsm_mask": False,
            "frequency": "D",
            "context_delta_t": None,
        },
    }
    if ancils_path is not None:
        config["data"]["ancillary_path"] = str(ancils_path)
    return config


def _write_basin_netcdf(path, se_code=1, other_code=2):
    """Write a synthetic subregion mask matching what load_basin expects.

    A 2D (y, x) `subregion` DataArray whose `subregion_labels` attr JSON-encodes
    a {code: label} mapping.
    """
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
        config = _make_config(ancils_path=tmp_path, basin="SE")

        loader = loader_cls(config)

        np.testing.assert_array_equal(loader.basin_x.values, [1.0, 2.0])
        np.testing.assert_array_equal(loader.basin_y.values, [1.0, 2.0, 3.0])

    def test_skips_basin_loading_when_basin_is_none(
        self, tmp_path, loader_cls, basin_filename
    ):
        config = _make_config(ancils_path=tmp_path, basin=None)

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
            ancils_path=tmp_path,
            basin="SE",
            data_path=str(tmp_path),
            data_pattern="data.zarr",
        )
        loader = HIRHAM5ZarrDataLoader(config)

        np.testing.assert_array_equal(loader.basin_x.values, [1.0, 2.0])
        np.testing.assert_array_equal(loader.basin_y.values, [1.0, 2.0, 3.0])

    def test_skips_basin_loading_when_basin_is_none(self, tmp_path):
        config = _make_config(
            ancils_path=tmp_path,
            basin=None,
            data_path=str(tmp_path),
            data_pattern="data.zarr",
        )

        loader = HIRHAM5ZarrDataLoader(config)

        assert not hasattr(loader, "basin_x")
        assert not hasattr(loader, "basin_y")


def _make_anomaly_da():
    """Build two years of the same three calendar days, offset by +10 in 2021.

    The 2020 climatology (trained on 2020 only) reproduces 2020 exactly
    (anomaly 0) and reveals the +10 offset for 2021 (anomaly 10).
    """
    dates = pd.to_datetime(
        [
            "2020-01-01",
            "2020-01-02",
            "2020-01-03",
            "2021-01-01",
            "2021-01-02",
            "2021-01-03",
        ]
    )
    dayofyear = dates.dayofyear.values
    year_offset = np.where(dates.year == 2020, 0, 10)
    values = (dayofyear + year_offset).astype(float)
    return xr.DataArray(
        np.broadcast_to(values[:, None, None], (6, 2, 2)).copy(),
        coords={"time": dates, "y": [0, 1], "x": [0, 1]},
        dims=("time", "y", "x"),
        name="var",
    )


class TestComputeGriddedAnomalies:
    def test_computes_anomalies_relative_to_training_climatology(self, tmp_path):
        (tmp_path / "test_region").mkdir()
        loader = BaseDataLoader(_make_config(data_path=str(tmp_path)))

        anomalies = loader.compute_gridded_anomalies("var", _make_anomaly_da())

        expected = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        np.testing.assert_allclose(anomalies.mean(("y", "x")).values, expected)

    def test_caches_climatology_to_disk(self, tmp_path):
        (tmp_path / "test_region").mkdir()
        loader = BaseDataLoader(_make_config(data_path=str(tmp_path)))

        loader.compute_gridded_anomalies("var", _make_anomaly_da())

        cached_files = list((tmp_path / "test_region").glob("*.nc"))
        assert len(cached_files) == 1

    def test_reuses_cached_climatology_on_second_call(self, tmp_path):
        (tmp_path / "test_region").mkdir()
        loader = BaseDataLoader(_make_config(data_path=str(tmp_path)))
        da = _make_anomaly_da()
        loader.compute_gridded_anomalies("var", da)

        # Tamper with the cached climatology so it's all zero; if the second
        # call reads the cache (rather than recomputing), the "anomaly" it
        # returns should equal the raw values themselves.
        cached_file = next((tmp_path / "test_region").glob("*.nc"))
        tampered = xr.open_dataset(cached_file).load()
        tampered.close()
        var_name = next(iter(tampered.data_vars))
        tampered[var_name] = xr.zeros_like(tampered[var_name])
        tampered.to_netcdf(cached_file, mode="w")

        anomalies = loader.compute_gridded_anomalies("var", da)

        expected = np.array([1.0, 2.0, 3.0, 11.0, 12.0, 13.0])
        np.testing.assert_allclose(anomalies.mean(("y", "x")).values, expected)


class TestFeaturizeVars:
    def test_produces_chunked_anomaly_variable(self, tmp_path):
        (tmp_path / "test_region").mkdir()
        loader = BaseDataLoader(_make_config(data_path=str(tmp_path)))

        dates = pd.date_range("2020-01-01", "2020-01-03")
        da = xr.DataArray(
            np.zeros((3, 2, 2)),
            coords={"time": dates, "y": [0, 1], "x": [0, 1]},
            dims=("time", "y", "x"),
            name="var",
        )

        result = loader.featurize_vars(
            var="var", dims=("time", "y", "x"), chunks=(2, 1, 1), abs=da
        )

        assert "var_anm" in result.data_vars
        assert dict(result.chunks) == {"time": (2, 1), "y": (1, 1), "x": (1, 1)}
