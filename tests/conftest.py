import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def lsm_3x3():
    """A tiny land-sea mask DataArray (land==1, sea==0) on a 3x3 lat/lon grid."""
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
    """A tiny station-location table with lat/lon columns."""
    return pd.DataFrame({"lat": [1.0, 2.0, 3.0], "lon": [10.0, 20.0, 30.0]})


@pytest.fixture
def native_grid_ds():
    """A tiny 3x3 native x/y grid with 2D lon/lat coordinates, lon/lat = 10x the x/y index."""
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
