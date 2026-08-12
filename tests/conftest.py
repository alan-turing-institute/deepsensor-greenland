import deepsensor.torch  # noqa: F401  (registers the torch backend with deepsensor)
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from deepsensor.data import DataProcessor, TaskLoader


@pytest.fixture
def lsm():
    """Build a tiny land-sea mask DataArray (land==1, sea==0) on a 3x3 lat/lon grid."""
    lat = np.array([-10.0, 0.0, 10.0])
    lon = np.array([0.0, 10.0, 20.0])
    values = np.array(
        [
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
        ]
    )
    return xr.DataArray(values, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))


@pytest.fixture
def stn_locs_df():
    """Build a tiny station-location table with lat/lon columns."""
    return pd.DataFrame({"lat": [1.0, 2.0, 3.0], "lon": [10.0, 20.0, 30.0]})


@pytest.fixture
def native_grid_ds():
    """Build a tiny 3x3 native x/y grid with 2D lon/lat coords, lon/lat = 10x the x/y index."""
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 2])
    X, Y = np.meshgrid(x, y, indexing="xy")
    lon2d = X.astype(float) * 10
    lat2d = Y.astype(float) * 10
    return xr.Dataset(
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
        }
    )


@pytest.fixture(scope="session")
def tiny_model_config():
    """Build a minimal ConvNP config small enough to build/train instantly in tests."""
    return {"model": {"unet_channels": (4, 4), "likelihood": "cnp"}}


@pytest.fixture(scope="session")
def tiny_dp_and_task_loader():
    """Build a DataProcessor + TaskLoader over a tiny synthetic 5x5 lat/lon grid.

    Small enough to build/train a real (but fast) ConvNP without needing real
    climate data.
    """
    lat = np.linspace(0, 1, 5)
    lon = np.linspace(0, 1, 5)
    dates = pd.date_range("2020-01-01", periods=3)
    rng = np.random.default_rng(0)
    values = rng.normal(size=(len(dates), len(lat), len(lon)))
    da = xr.DataArray(
        values,
        coords={"time": dates, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="var",
    )
    ds = da.to_dataset()

    data_processor = DataProcessor(x1_name="lat", x2_name="lon", verbose=False)
    ds_norm = data_processor(ds)
    task_loader = TaskLoader(context=ds_norm, target=ds_norm)
    return data_processor, task_loader, dates
