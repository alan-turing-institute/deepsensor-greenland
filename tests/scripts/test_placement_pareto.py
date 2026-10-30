import numpy as np
import pytest
import xarray as xr
from deepsensor.data.processor import DataProcessor

import giant_sp.scripts.placement_pareto as pp


@pytest.fixture
def coords_ref():
    """Tiny 5 (y) x 11 (x) integer-indexed grid with 2D lat/lon, ~0.1 deg spacing."""
    nx, ny = 11, 5
    x, y = np.arange(nx), np.arange(ny)
    lon2d, lat2d = np.meshgrid(-40.0 + 0.1 * x, 70.0 + 0.1 * y)
    return xr.Dataset(
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "lon": (("y", "x"), lon2d),
            "lat": (("y", "x"), lat2d),
        }
    )


class TestContextDistKmFactor:
    def test_full_grid_range_matches_full_grid_cells(self, coords_ref):
        # Trained on the whole grid: one normalised unit = max(nx, ny) - 1 cells.
        factor = pp.context_dist_km_factor(coords_ref, (0.0, 10.0))

        assert factor == pytest.approx(pp.km_per_grid_cell(coords_ref) * 10)

    def test_basin_cropped_range_scales_down(self, coords_ref):
        # Trained on a basin covering half the grid's index range.
        full = pp.context_dist_km_factor(coords_ref, (0.0, 10.0))
        cropped = pp.context_dist_km_factor(coords_ref, (3.0, 8.0))

        assert cropped == pytest.approx(full / 2)

    @pytest.mark.parametrize(
        "x1_map",
        [(-2546675.0, -1809675.0), (5.0, 5.0)],
        ids=["projected_metres", "empty_range"],
    )
    def test_rejects_non_grid_index_range(self, coords_ref, x1_map):
        with pytest.raises(ValueError, match="grid-index range"):
            pp.context_dist_km_factor(coords_ref, x1_map)


class TestLoadRunX1Map:
    def test_reads_saved_data_processor_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        files_dir = tmp_path / "wandb" / "run-20260101_000000-abc123" / "files"
        files_dir.mkdir(parents=True)
        DataProcessor(
            x1_name="lat", x2_name="lon", x1_map=(3.0, 8.0), x2_map=(1.0, 6.0)
        ).save(str(files_dir))

        assert pp.load_run_x1_map("abc123") == (3.0, 8.0)

    def test_missing_local_copy_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError, match=r"run placement\.py"):
            pp.load_run_x1_map("abc123")
