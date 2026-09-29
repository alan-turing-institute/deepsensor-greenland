import numpy as np
import pytest
import zarr

from giant_sp.demo.dashboard import (
    Dashboard,
    build_catalog,
    build_series,
    coordinate_names,
    decode_time,
    discover_experiments,
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
