from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from hydra import compose, initialize_config_dir

import giant_sp
import giant_sp.preprocessing.hirham5_create_basefile as cb

CONFIG_DIR = Path(giant_sp.__file__).parent / "configs"
VARS = ["tas", "ahfl", "ahfs", "dlwrad", "rainfall", "snfall", "dswrad", "tsl"]


def make_daily(times: pd.DatetimeIndex, nx: int = 3, ny: int = 2) -> xr.Dataset:
    """Daily grid where location (x=0, y=0) is NaN at every time step."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(len(times), ny, nx)).astype(np.float32)
    data[:, 0, 0] = np.nan
    return xr.Dataset(
        {var: (("time", "y", "x"), data.copy()) for var in VARS},
        coords={"time": times, "x": np.arange(nx) * 5.0, "y": np.arange(ny) * 5.0},
    )


class TestFlattenSpatial:
    def test_drops_all_nan_locations(self):
        ds = make_daily(pd.date_range("2000-01-01", periods=4))

        flat = cb.flatten_spatial(ds)

        assert flat.sizes["z"] == 5
        assert {"x", "y"} <= set(flat.coords)

    def test_keeps_nan_locations_without_dropna(self):
        flat = cb.flatten_spatial(
            make_daily(pd.date_range("2000-01-01", periods=4)), dropna=False
        )

        assert flat.sizes["z"] == 6


def test_create_coords_dataset_keeps_grid_and_attrs():
    x, y = np.arange(3), np.arange(2)
    src = xr.Dataset(
        {
            "lat": (("y", "x"), np.ones((2, 3))),
            "lon": (("y", "x"), np.zeros((2, 3))),
            "glacGRL": (("y", "x"), np.ones((2, 3))),
        },
        coords={"x": x, "y": y},
        attrs={"title": "mask"},
    )

    meta = cb.create_coords_dataset(src)

    assert set(meta.data_vars) == {"lat", "lon"}
    np.testing.assert_array_equal(meta["x"], x)
    assert meta["lat"].dims == ("y", "x")
    assert meta.attrs == {"title": "mask"}


def test_clip_to_ranges_clips_and_skips_missing():
    ds = xr.Dataset({"ahfs": ("t", np.array([-200.0, -50.0, 10.0]))})

    out = cb.clip_to_ranges(ds, {"ahfs": [-140, None], "missing": [0, 1]})

    np.testing.assert_array_equal(out["ahfs"], [-140.0, -50.0, 10.0])
    np.testing.assert_array_equal(ds["ahfs"], [-200.0, -50.0, 10.0])


def test_build_base_dataset_adds_energyin():
    ds = make_daily(pd.date_range("2000-01-01", periods=4))

    out = cb.build_base_dataset(ds, {"time": 2, "z": -1})

    assert out.sizes["z"] == 5
    np.testing.assert_allclose(
        out["energyin"], out["ahfl"] + out["ahfs"] + out["dlwrad"], rtol=1e-6
    )


def test_rolling_means_only_full_windows_from_start_date():
    times = pd.date_range("2000-01-01", periods=6)
    ds = xr.Dataset({"rainfall": ("time", np.arange(6.0))}, coords={"time": times})

    out = cb.rolling_means(ds, ["rainfall"], 3, "2000-01-02", {"time": -1})

    assert list(out.data_vars) == ["rainfall_10yr-avg"]
    np.testing.assert_array_equal(out["rainfall_10yr-avg"], [np.nan, 1, 2, 3, 4])


def test_config_composes_with_logging():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="hirham5_create_basefile",
            overrides=["logging.log_level=DEBUG"],
            return_hydra_config=True,
        )

    assert cfg.hydra.job_logging.root.level == "DEBUG"
    assert cfg.preprocessing.end_year >= cfg.preprocessing.start_year


@pytest.mark.filterwarnings("ignore:Consolidated metadata:UserWarning")
@pytest.mark.parametrize(
    ("source", "folder"),
    [
        ("erai", "HIRHAM5-ERAInterim"),
        ("cesm2", "HIRHAM5-CESM2"),
        ("ecearth3", "HIRHAM5-ECEARTH3"),
    ],
)
def test_create_basefile_end_to_end(tmp_path, source, folder):
    aux_dir = tmp_path / "interim" / "AuxFiles"
    daily_dir = tmp_path / "interim" / source
    aux_dir.mkdir(parents=True)
    daily_dir.mkdir()

    grid = make_daily(pd.date_range("2000-01-01", periods=1)).isel(time=0)
    xr.Dataset(
        {
            "glacGRL": grid["tas"],
            "maskbas": grid["tas"],
            "lat": grid["tas"].fillna(70.0),
            "lon": grid["tas"].fillna(-40.0),
        }
    ).to_netcdf(aux_dir / "GRLmask.nc")
    xr.Dataset({"zones": grid["tas"]}).to_netcdf(aux_dir / "GRLzones.nc")
    # start_year=2010 also loads the 10 years before it, for the (fixed 3652-day)
    # rolling means, so write full daily years for every rolling window to be full
    for year in range(2000, 2012):
        make_daily(pd.date_range(f"{year}-01-01", f"{year}-12-31")).to_netcdf(
            daily_dir / f"Daily2D_GRL_{year}.nc"
        )

    out_dir = tmp_path / "out"
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="hirham5_create_basefile",
            overrides=[
                f"preprocessing.interim_path={tmp_path / 'interim'}",
                f"preprocessing.output_dir={out_dir}",
                f"preprocessing.daily_dir={source}",
                "preprocessing.version=3",
                "preprocessing.start_year=2010",
                "preprocessing.end_year=2011",
                "preprocessing.drop_vars=[tsl]",
                "preprocessing.rolling.start_date=2010-01-01",
            ],
        )

    cb.create_basefile(cfg.preprocessing)

    out_path = out_dir / folder / "v_03"
    base = xr.open_zarr(out_path / "base_dataset_2010_2011.zarr")
    n_days = len(pd.date_range("2000-01-01", "2011-12-31"))
    assert base.sizes == {"time": n_days, "z": 5}
    assert base["time"].values[0] == np.datetime64("2000-01-01")
    assert "energyin" in base
    assert "tsl" not in base
    rolled = xr.open_zarr(out_path / "base_dataset_2010_2011_10yr.zarr")
    assert set(rolled.data_vars) == {
        f"{v}_10yr-avg" for v in ["rainfall", "snfall", "energyin", "dswrad"]
    }
    assert rolled.sizes["time"] == len(pd.date_range("2010-01-01", "2011-12-31"))
    assert not rolled["rainfall_10yr-avg"].isnull().all(dim="z").any()
    assert xr.open_zarr(out_path / "GRLmask.zarr").sizes["z"] == 5
    assert xr.open_zarr(out_path / "GRLzones.zarr").sizes["z"] == 5
    assert set(xr.open_zarr(out_path / "coords_only.zarr").data_vars) == {"lat", "lon"}
