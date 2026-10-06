import json

import pytest

from giant_sp.demo.dashboard import Dashboard
from giant_sp.demo.snapshot import map_dates, write_snapshot


@pytest.fixture
def dashboard(fake_source, monkeypatch):
    """Build a Dashboard over the in-memory dataset, with map rendering stubbed."""
    dashboard = Dashboard(fake_source)
    monkeypatch.setattr(
        dashboard, "map_png", lambda e, v, d, full: f"{e}|{v}|{d}|{full}".encode()
    )
    return dashboard


def test_map_dates_monthly():
    times = ["2015-01-01T12:00:00", "2015-02-10T12:00:00", "2015-03-31T12:00:00"]
    assert map_dates(times, "MS") == ["2015-01-01", "2015-02-01", "2015-03-01"]


def test_map_dates_needs_real_dates():
    assert map_dates([0, 1, 2], "MS") == []
    assert map_dates([], "MS") == []


def test_write_snapshot(dashboard, tmp_path):
    out = tmp_path / "static-api"
    n_files = write_snapshot(
        dashboard,
        out,
        map_variables=["melt", "total"],
        map_freq="2D",
        plotly_url=None,
    )

    repo = json.loads((out / "repo.json").read_text())
    assert repo["snapshot"]["id"] == "snap1"
    assert "saved_at" in repo
    assert json.loads((out / "catalog.json").read_text()) == dashboard.catalog()
    series = json.loads(
        (out / "series" / "project" / "run1" / "total.json").read_text()
    )
    assert series == dashboard.series("project/run1", "total")

    dates = ["2000-01-01", "2000-01-03"]
    index = json.loads((out / "maps" / "index.json").read_text())
    assert index == {"baseline": {"melt": dates}, "project/run1": {"melt": dates}}
    maps = out / "maps" / "baseline" / "melt"
    assert (maps / "2000-01-03.png").read_bytes() == b"baseline|melt|2000-01-03|False"
    assert (
        maps / "2000-01-03-full.png"
    ).read_bytes() == b"baseline|melt|2000-01-03|True"
    # "total" is 1-D, so it gets no maps even when asked for.
    assert not (out / "maps" / "baseline" / "total").exists()

    # repo + catalog + 2 experiments x 2 series + 2 x 2 x 2 maps + map index
    assert n_files == 2 + 4 + 8 + 1
    assert sum(1 for p in out.rglob("*") if p.is_file()) == n_files


def test_write_snapshot_maps_every_2d_variable_by_default(dashboard, tmp_path):
    out = tmp_path / "static-api"
    write_snapshot(dashboard, out, map_freq="2D", plotly_url=None)
    index = json.loads((out / "maps" / "index.json").read_text())
    assert {exp: set(variables) for exp, variables in index.items()} == {
        "baseline": {"melt"},
        "project/run1": {"melt"},
    }


def test_write_snapshot_replaces_previous_snapshot(dashboard, tmp_path):
    out = tmp_path / "static-api"
    write_snapshot(dashboard, out, map_variables=["melt"], plotly_url=None)
    write_snapshot(dashboard, out, map_variables=[], plotly_url=None)
    assert not (out / "maps" / "baseline").exists()


def test_write_snapshot_refuses_unrelated_directory(dashboard, tmp_path):
    (tmp_path / "keep.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        write_snapshot(dashboard, tmp_path, plotly_url=None)
    assert (tmp_path / "keep.txt").exists()
