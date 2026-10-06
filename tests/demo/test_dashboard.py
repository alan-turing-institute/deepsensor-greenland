import numpy as np
import pytest
import zarr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure

from giant_sp.demo.dashboard import (
    EXCLUDED_EXPERIMENTS,
    EXCLUDED_VARIABLES,
    MAP_CMAPS,
    MAP_LOG_DECADES,
    MAP_LOG_UPPER_PERCENTILE,
    Dashboard,
    build_catalog,
    build_series,
    coordinate_names,
    decode_time,
    discover_experiments,
    is_log_scale,
    map_color_range,
    nearest_time_index,
    render_map_png,
    time_array_name,
)


def test_time_array_name(root):
    assert time_array_name(root["baseline"]) == "time"
    assert time_array_name(root["grlmask"]) is None


def test_coordinate_names_include_time_and_cf_coordinates(root):
    assert coordinate_names(root["baseline"]) == {"time", "lat", "lon"}


def test_discover_experiments_skips_static_groups_and_nests_one_level(root):
    assert set(discover_experiments(root)) == {"baseline", "project/run1"}


def test_discover_experiments_falls_back_to_root():
    root = zarr.group()
    root.create_array("time", shape=(2,), dtype="f8")
    assert set(discover_experiments(root)) == {"root"}


def test_build_catalog(root):
    catalog = build_catalog(root)
    assert [e["name"] for e in catalog["experiments"]] == ["baseline", "project/run1"]
    baseline = catalog["experiments"][0]
    assert baseline == {
        "name": "baseline",
        "n_points": 4,
        "has_time": True,
        "variables": ["melt", "total"],
    }
    melt = next(v for v in catalog["variables"] if v["key"] == "melt")
    assert melt == {
        "key": "melt",
        "units": "mm",
        "long_name": "Acc snow melt",
        "spatial_mean": True,
    }


@pytest.mark.parametrize("name", sorted(EXCLUDED_VARIABLES))
def test_excluded_variables_are_hidden(root, name):
    root["baseline"].create_array(name, shape=(4, 3), dtype="f8")
    catalog = build_catalog(root)
    assert name not in catalog["experiments"][0]["variables"]
    assert name not in {v["key"] for v in catalog["variables"]}
    with pytest.raises(KeyError):
        build_series(root, "baseline", name)


@pytest.mark.parametrize("name", sorted(EXCLUDED_EXPERIMENTS))
def test_excluded_experiments_are_hidden(root, add_experiment, name):
    add_experiment(root, name)
    assert name not in discover_experiments(root)
    assert name not in {e["name"] for e in build_catalog(root)["experiments"]}
    with pytest.raises(KeyError):
        build_series(root, name, "melt")


def test_build_series_domain_mean_and_decoded_time(root):
    s = build_series(root, "baseline", "melt")
    assert s["values"] == [1.0, 4.0, 7.0, 10.0]
    assert s["time"][:2] == ["2000-01-01T00:00:00", "2000-01-02T00:00:00"]
    assert s["spatial_mean"] is True


def test_build_series_sanitizes_non_finite_values(root):
    assert build_series(root, "baseline", "total")["values"] == [1.0, None, 3.0, None]


def test_build_series_all_nan_time_step_is_none(root):
    root["baseline"]["melt"][1, :] = np.nan
    assert build_series(root, "baseline", "melt")["values"][1] is None


@pytest.mark.parametrize(
    ("experiment", "variable"), [("nope", "melt"), ("baseline", "nope")]
)
def test_build_series_unknown_raises_key_error(root, experiment, variable):
    with pytest.raises(KeyError):
        build_series(root, experiment, variable)


def test_decode_time_without_units_falls_back_to_index(add_experiment):
    root = zarr.group()
    add_experiment(root, "e", time_units=None)
    assert decode_time(root["e"], "time") == [0, 1, 2, 3]


def test_nearest_time_index(root):
    assert nearest_time_index(root["baseline"], "time", "2000-01-03") == 2
    assert nearest_time_index(root["baseline"], "time", "2099-01-01") == 3


def test_nearest_time_index_without_units_clamps_sample_index(add_experiment):
    root = zarr.group()
    add_experiment(root, "e", time_units=None)
    assert nearest_time_index(root["e"], "time", "10") == 3


def test_render_map_png_rejects_1d_variable(root):
    with pytest.raises(ValueError, match="no time dimension"):
        render_map_png(root, "baseline", "total", "2000-01-01")


@pytest.mark.parametrize(
    ("long_name", "expected"),
    [("Acc snow melt", MAP_CMAPS["acc snow melt"]), ("2-m Temperature", "viridis")],
)
def test_render_map_png_picks_cmap_by_long_name(root, monkeypatch, long_name, expected):
    root["baseline"]["melt"].attrs["long_name"] = long_name
    seen = []
    original = GeoAxes.scatter

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("cmap"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GeoAxes, "scatter", spy)
    render_map_png(root, "baseline", "melt", "2000-01-02")
    assert seen == [expected]


def test_map_color_range_spans_all_dates(root):
    # melt is 0..11 over 4 dates x 3 points; NaNs are ignored.
    root["baseline"]["melt"].attrs["long_name"] = "2-m Temperature"
    root["baseline"]["melt"][0, 0] = np.nan
    vmin, vmax = map_color_range(root, "baseline", "melt")
    finite = np.arange(1, 12, dtype=float)
    assert vmin == pytest.approx(np.percentile(finite, 1))
    assert vmax == pytest.approx(np.percentile(finite, 99))


def test_map_color_range_log_scale_spans_decades_below_upper_percentile(root):
    # "Acc snow melt" is log-scaled; zeros are left out of the percentile.
    assert is_log_scale(root["baseline"]["melt"])
    vmin, vmax = map_color_range(root, "baseline", "melt")
    positive = np.arange(1, 12, dtype=float)
    assert vmax == pytest.approx(np.percentile(positive, MAP_LOG_UPPER_PERCENTILE))
    assert vmin == pytest.approx(vmax / 10**MAP_LOG_DECADES)


def test_map_color_range_log_scale_falls_back_to_linear_without_positives(root):
    root["baseline"]["melt"][:] = 0.0
    assert map_color_range(root, "baseline", "melt") == (0.0, 0.0)


def test_render_map_png_log_scale_lifts_zeros_to_bottom(root, monkeypatch):
    seen = []
    original = GeoAxes.scatter

    def spy(self, *args, **kwargs):
        seen.append((kwargs["norm"], np.asarray(kwargs["c"])))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GeoAxes, "scatter", spy)
    render_map_png(root, "baseline", "melt", "2000-01-01", (0.5, 50.0))
    [(norm, values)] = seen
    assert isinstance(norm, LogNorm)
    assert (norm.vmin, norm.vmax) == (0.5, 50.0)
    assert values.tolist() == [0.5, 1.0, 2.0]


def test_render_map_png_log_scale_colorbar_has_plain_number_labels(root, monkeypatch):
    cbars = []
    original = Figure.colorbar

    def spy(self, *args, **kwargs):
        cbars.append(original(self, *args, **kwargs))
        return cbars[-1]

    monkeypatch.setattr(Figure, "colorbar", spy)
    render_map_png(root, "baseline", "melt", "2000-01-01", (0.05, 50.0))
    [cbar] = cbars
    axis = cbar.ax.xaxis
    ticks = [t for t in axis.get_majorticklocs() if 0.05 <= t <= 50.0]
    assert ticks == pytest.approx([0.1, 0.3, 1, 3, 10, 30])
    assert [axis.get_major_formatter()(t, 0) for t in ticks] == [
        "0.1",
        "0.3",
        "1",
        "3",
        "10",
        "30",
    ]


@pytest.mark.parametrize("long_name", ["Acc snow melt", "2-m Temperature"])
def test_map_color_range_none_without_finite_values(root, long_name):
    root["baseline"]["melt"].attrs["long_name"] = long_name
    root["baseline"]["melt"][:] = np.nan
    assert map_color_range(root, "baseline", "melt") is None


def test_dashboard_maps_share_one_color_range(fake_source, monkeypatch):
    seen = []
    original = GeoAxes.scatter

    def spy(self, *args, **kwargs):
        seen.append((kwargs["norm"].vmin, kwargs["norm"].vmax))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GeoAxes, "scatter", spy)
    dashboard = Dashboard(fake_source)
    dashboard.map_png("baseline", "melt", "2000-01-01")
    dashboard.map_png("baseline", "melt", "2000-01-04")
    expected = map_color_range(fake_source.group, "baseline", "melt")
    assert seen == [expected, expected]


def test_render_map_png_requires_lat_lon():
    root = zarr.group()
    g = root.create_group("e")
    g.create_array("time", shape=(2,), dtype="f8")
    g.create_array("field", shape=(2, 3), dtype="f8")
    with pytest.raises(ValueError, match="lat/lon"):
        render_map_png(root, "e", "field", "0")


def test_dashboard_caches_per_snapshot(fake_source):
    source = fake_source
    dashboard = Dashboard(source)
    first = dashboard.catalog()
    assert dashboard.catalog() is first
    assert dashboard.series("baseline", "melt") is dashboard.series("baseline", "melt")
    assert source.opened == ["snap1"]

    source.snapshot_id = "snap2"
    assert dashboard.catalog() is not first
    assert source.opened == ["snap1", "snap2"]
    assert dashboard.snapshot().id == "snap2"
