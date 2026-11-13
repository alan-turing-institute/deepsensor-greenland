import numpy as np
import pytest
import zarr

from ds_greenland.demo.source import Snapshot


def _add_experiment(
    root, name, n_time=4, n_points=3, time_units="days since 2000-01-01"
):
    """Add a (time, points) experiment group with CF-style time and lat/lon coords."""
    g = root.create_group(name)
    t = g.create_array("time", shape=(n_time,), dtype="f8")
    t[:] = np.arange(n_time, dtype=float)
    if time_units:
        t.attrs.update({"units": time_units, "calendar": "standard"})
    for coord, vals in {
        "lat": [70.0, 72.0, 75.0],
        "lon": [-45.0, -40.0, -50.0],
    }.items():
        a = g.create_array(coord, shape=(n_points,), dtype="f8")
        a[:] = vals[:n_points]
    melt = g.create_array("melt", shape=(n_time, n_points), dtype="f8")
    melt[:] = np.arange(n_time * n_points, dtype=float).reshape(n_time, n_points)
    melt.attrs.update(
        {"units": "mm", "long_name": "Acc snow melt", "coordinates": "lat lon"}
    )
    total = g.create_array("total", shape=(n_time,), dtype="f8")
    total[:] = [1.0, np.nan, 3.0, np.inf][:n_time]
    return g


@pytest.fixture
def add_experiment():
    """Expose the experiment-group builder to tests that need a custom layout."""
    return _add_experiment


@pytest.fixture
def root():
    """Build an in-memory dataset with two experiments and a static mask group."""
    root = zarr.group()
    _add_experiment(root, "baseline")
    nested = root.create_group("project")
    _add_experiment(nested, "run1")
    mask = root.create_group("grlmask")
    mask.create_array("mask", shape=(3,), dtype="i1")
    return root


class FakeSource:
    """In-memory stand-in for ArraylakeSource with a switchable snapshot."""

    repo = "org/repo"
    branch = "main"
    root_path = ""

    def __init__(self, group):
        self.group = group
        self.snapshot_id = "snap1"
        self.opened: list[str] = []

    def tip_snapshot(self):
        return Snapshot(self.snapshot_id, "2026-01-01T00:00:00", "msg")

    def open_group(self, snapshot_id):
        self.opened.append(snapshot_id)
        return self.group


@pytest.fixture
def fake_source(root):
    """Wrap the in-memory dataset in a FakeSource."""
    return FakeSource(root)
