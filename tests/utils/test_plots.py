import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.collections import QuadMesh

from ds_greenland.utils.plots import (
    comparison_map_plot,
    grid_metric_maps_plot,
    plot_greenland_only,
)


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


def test_plot_greenland_only_without_placements_draws_only_the_field(
    greenland_field, scatter_calls
):
    ax, pcm = plot_greenland_only(greenland_field)
    assert scatter_calls == []
    assert len(ax.texts) == 0
    assert pcm.get_array() is not None


class TestGridMetricMapsPlot:
    @pytest.fixture
    def maps(self, greenland_field):
        return xr.Dataset(
            {
                "rmse": greenland_field,
                "mae": greenland_field / 2,
                "avgstd": greenland_field / 3,
            }
        )

    @staticmethod
    def _meshes(fig):
        return [
            next(c for c in ax.collections if isinstance(c, QuadMesh))
            for ax in fig.axes[:3]
        ]

    def test_one_panel_per_metric_with_mean_in_title(self, maps):
        fig = grid_metric_maps_plot(maps)
        titles = [ax.get_title() for ax in fig.axes[:3]]
        assert titles == [
            "RMSE (mean 14.50)",
            "MAE (mean 7.25)",
            "Average std (mean 4.83)",
        ]
        plt.close(fig)

    def test_default_limits_run_from_zero_to_max(self, maps):
        fig = grid_metric_maps_plot(maps)
        assert [m.get_clim() for m in self._meshes(fig)] == [
            (0, 29),
            (0, 14.5),
            (0, 29 / 3),
        ]
        assert all(m.colorbar.extend == "neither" for m in self._meshes(fig))
        plt.close(fig)

    def test_value_lims_fix_the_scale_and_extend_the_colorbar(self, maps):
        fig = grid_metric_maps_plot(maps, value_lims={"rmse": (None, 10.0)})
        rmse, mae, _ = self._meshes(fig)
        assert rmse.get_clim() == (0, 10.0)
        assert rmse.colorbar.extend == "max"
        assert mae.get_clim() == (0, 14.5)  # unlisted metric keeps the default
        plt.close(fig)


def _first_mesh(ax):
    return next(c for c in ax.collections if isinstance(c, QuadMesh))


class TestComparisonMapPlot:
    def test_single_date_panels_and_residual_stats(self, greenland_field):
        fig = comparison_map_plot(
            greenland_field, greenland_field + 1, xr.full_like(greenland_field, 0.5)
        )
        maps = fig.axes[:4]
        assert [ax.get_title() for ax in maps] == [
            "True",
            "Predicted",
            "Residual (pred. - true)",
            "Std",
        ]
        assert maps[2].texts[0].get_text() == "RMSE: 1.00\nMAE: 1.00\nMBE: 1.00"
        plt.close(fig)

    def test_time_dimension_maps_totals_and_mean_std(self, greenland_field):
        two_days = xr.concat([greenland_field, greenland_field], dim="time")
        fig = comparison_map_plot(two_days, two_days, two_days)
        true_mesh = _first_mesh(fig.axes[0])
        assert true_mesh.get_clim() == (0, 58)  # 2 x max of 29
        assert true_mesh.colorbar.ax.get_xlabel() == "mm w.e."
        assert fig.axes[3].get_title() == "Average std"
        assert _first_mesh(fig.axes[3]).get_clim() == (0, 29)
        plt.close(fig)

    def test_melt_extent_from_cell_areas(self, greenland_field):
        cell_area = xr.full_like(greenland_field, 1e6)  # 1 km² cells
        fig = comparison_map_plot(
            greenland_field, greenland_field, greenland_field, cell_area=cell_area
        )
        # Values 2..29 are above the 1 mm/day threshold: 28 cells.
        assert fig.axes[0].texts[0].get_text() == "ME: 28 km$^2$"
        plt.close(fig)

    def test_residual_max_clips_the_scale_and_marks_min_and_max(self, greenland_field):
        offset = np.where(np.arange(30).reshape(5, 6) % 2, 3.0, -2.0)
        fig = comparison_map_plot(
            greenland_field,
            greenland_field + offset,
            greenland_field,
            residual_max=1.0,
        )
        cbar = _first_mesh(fig.axes[2]).colorbar
        assert cbar.extend == "both"
        markers = [line.get_color() for line in cbar.ax.lines]
        assert markers == ["blue", "red"]
        plt.close(fig)

    def test_true_and_pred_share_one_scale_and_colorbar_by_default(
        self, greenland_field
    ):
        fig = comparison_map_plot(
            greenland_field, greenland_field + 10, greenland_field
        )
        true_mesh, pred_mesh = _first_mesh(fig.axes[0]), _first_mesh(fig.axes[1])
        assert true_mesh.get_clim() == pred_mesh.get_clim() == (0, 39)
        assert len(fig.axes) == 4 + 3  # maps + true/pred, residual, std bars
        plt.close(fig)

    def test_separate_scales_and_colorbars(self, greenland_field):
        fig = comparison_map_plot(
            greenland_field, greenland_field + 10, greenland_field, join_colorbar=False
        )
        assert _first_mesh(fig.axes[0]).get_clim() == (0, 29)
        assert _first_mesh(fig.axes[1]).get_clim() == (10, 39)
        assert len(fig.axes) == 4 + 4  # maps + true, pred, residual, std bars
        plt.close(fig)
