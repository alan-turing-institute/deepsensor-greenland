import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes

from giant_sp.utils.plots import plot_greenland_only


@pytest.fixture
def greenland_field():
    """Build a tiny lat/lon field spanning Greenland."""
    lat = np.linspace(60.0, 80.0, 5)
    lon = np.linspace(-50.0, -25.0, 6)
    return xr.DataArray(
        np.arange(30.0).reshape(5, 6),
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
    )


@pytest.fixture
def placements():
    """Build (new, existing) placement tables with lat/lon columns."""
    new = pd.DataFrame({"lat": [70.0, 72.0], "lon": [-40.0, -35.0]})
    existing = pd.DataFrame({"lat": [68.0], "lon": [-45.0]})
    return new, existing


@pytest.fixture
def scatter_calls(monkeypatch):
    """Record the keyword arguments of every GeoAxes.scatter call."""
    calls = []
    original = GeoAxes.scatter

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GeoAxes, "scatter", spy)
    yield calls
    plt.close("all")


def _new_placement_kwargs(calls):
    (kwargs,) = [c for c in calls if c.get("label") == "New placements"]
    return kwargs


def test_plot_greenland_only_default_new_placement_style(
    greenland_field, placements, scatter_calls
):
    plot_greenland_only(greenland_field, *placements, pcolormesh_kwargs={})
    kwargs = _new_placement_kwargs(scatter_calls)
    assert (kwargs["marker"], kwargs["c"], kwargs["s"]) == ("+", "r", 9)


def test_plot_greenland_only_newplace_kwargs_override_defaults(
    greenland_field, placements, scatter_calls
):
    plot_greenland_only(
        greenland_field,
        *placements,
        pcolormesh_kwargs={},
        newplace_kwargs={"marker": "o", "c": "#0072B2", "edgecolors": "white"},
    )
    kwargs = _new_placement_kwargs(scatter_calls)
    assert kwargs["marker"] == "o"
    assert kwargs["c"] == "#0072B2"
    assert kwargs["edgecolors"] == "white"
    # Keys not overridden keep their defaults.
    assert (kwargs["s"], kwargs["zorder"]) == (9, 5)
    # Existing stations are unaffected.
    (existing,) = [c for c in scatter_calls if c.get("label") == "Existing placements"]
    assert (existing["marker"], existing["c"]) == ("x", "k")
