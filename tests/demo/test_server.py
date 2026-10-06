import pytest
from fastapi.testclient import TestClient

from giant_sp.demo.dashboard import Dashboard
from giant_sp.demo.server import create_app


@pytest.fixture
def site_dir(tmp_path):
    """Build a minimal static site with a dashboard subpage."""
    (tmp_path / "index.html").write_text("<p>home</p>")
    (tmp_path / "dashboard").mkdir()
    (tmp_path / "dashboard" / "index.html").write_text("<p>dashboard</p>")
    return tmp_path


@pytest.fixture
def client(site_dir, fake_source, monkeypatch):
    """Serve the demo app over an in-memory dataset, with map rendering stubbed."""
    dashboard = Dashboard(fake_source)
    monkeypatch.setattr(dashboard, "map_png", _fake_map_png)
    with TestClient(create_app(dashboard, site_dir=site_dir)) as c:
        yield c


def _fake_map_png(experiment, variable, date, full_range):
    if experiment != "baseline":
        raise KeyError(experiment)
    if variable == "total":
        raise ValueError("no time dimension")
    return b"\x89PNG" + (b" full" if full_range else b"")


def test_serves_static_site_with_no_cache(client):
    r = client.get("/")
    assert r.text == "<p>home</p>"
    assert r.headers["cache-control"] == "no-cache"
    assert client.get("/dashboard/").text == "<p>dashboard</p>"


def test_health(client):
    assert client.get("/dashboard/api/health").json() == {"status": "ok"}


def test_repo(client):
    body = client.get("/dashboard/api/repo").json()
    assert body["repo"] == "org/repo"
    assert body["snapshot"]["id"] == "snap1"


def test_catalog(client):
    body = client.get("/dashboard/api/catalog").json()
    assert [e["name"] for e in body["experiments"]] == ["baseline", "project/run1"]


def test_series(client):
    r = client.get(
        "/dashboard/api/series", params={"experiment": "baseline", "variable": "melt"}
    )
    assert r.json()["values"] == [1.0, 4.0, 7.0, 10.0]


def test_series_unknown_is_404(client):
    r = client.get(
        "/dashboard/api/series", params={"experiment": "nope", "variable": "melt"}
    )
    assert r.status_code == 404


@pytest.mark.parametrize(
    ("experiment", "variable", "status"),
    [("baseline", "melt", 200), ("nope", "melt", 404), ("baseline", "total", 400)],
)
def test_map_status_codes(client, experiment, variable, status):
    r = client.get(
        "/dashboard/api/map",
        params={"experiment": experiment, "variable": variable, "date": "2000-01-01"},
    )
    assert r.status_code == status
    if status == 200:
        assert r.headers["content-type"] == "image/png"


@pytest.mark.parametrize(
    ("params", "body"), [({}, b"\x89PNG"), ({"full_range": "true"}, b"\x89PNG full")]
)
def test_map_full_range(client, params, body):
    r = client.get(
        "/dashboard/api/map",
        params={"experiment": "baseline", "variable": "melt", "date": "2000-01-01"}
        | params,
    )
    assert r.content == body
