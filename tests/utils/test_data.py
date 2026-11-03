import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import Point, Polygon

from giant_sp.utils.data import (
    construct_circ_and_linear_time_ds,
    context_sampling_with_delta,
    find_lon_lat,
    find_xy_for_points,
    mixed_sampling_strategy,
    normalize_lons,
    points_in_polygons,
    sample_dates_by_year_chunks,
    sample_land_locations,
    sample_station_locations,
    ten_year_store_name,
)


class TestSampleLandLocations:
    def test_requires_lsm(self):
        with pytest.raises(ValueError, match="Land-sea mask"):
            sample_land_locations(5)

    def test_shape(self, lsm):
        raw_locs = sample_land_locations(5, lsm=lsm, seed=0)
        assert raw_locs.shape == (2, 5)

    def test_samples_only_land_cells(self, lsm):
        land_cells = {
            (float(lat), float(lon))
            for lat in lsm["lat"].values
            for lon in lsm["lon"].values
            if lsm.sel(lat=lat, lon=lon).item()
        }
        raw_locs = sample_land_locations(50, lsm=lsm, seed=0)
        sampled = set(zip(raw_locs[0], raw_locs[1], strict=True))
        assert sampled <= land_cells

    def test_invert_mask_samples_only_sea_cells(self, lsm):
        sea_cells = {
            (float(lat), float(lon))
            for lat in lsm["lat"].values
            for lon in lsm["lon"].values
            if not lsm.sel(lat=lat, lon=lon).item()
        }
        raw_locs = sample_land_locations(50, lsm=lsm, seed=0, invert_mask=True)
        sampled = set(zip(raw_locs[0], raw_locs[1], strict=True))
        assert sampled <= sea_cells

    def test_seed_reproducible(self, lsm):
        first = sample_land_locations(10, lsm=lsm, seed=123)
        second = sample_land_locations(10, lsm=lsm, seed=123)
        np.testing.assert_array_equal(first, second)


class TestSampleStationLocations:
    def test_shape(self, stn_locs_df):
        raw_locs = sample_station_locations(stn_locs_df, 2, seed=0)
        assert raw_locs.shape == (2, 2)

    def test_samples_without_replacement(self, stn_locs_df):
        raw_locs = sample_station_locations(stn_locs_df, len(stn_locs_df), seed=0)
        sampled_rows = {tuple(row) for row in raw_locs.T}
        assert len(sampled_rows) == len(stn_locs_df)

    def test_seed_reproducible(self, stn_locs_df):
        first = sample_station_locations(stn_locs_df, 2, seed=7)
        second = sample_station_locations(stn_locs_df, 2, seed=7)
        np.testing.assert_array_equal(first, second)


class TestMixedSamplingStrategy:
    def test_shape(self, stn_locs_df, lsm):
        raw_locs = mixed_sampling_strategy(
            stn_locs_df, 10, frac_stn=0.2, lsm=lsm, seed=0
        )
        assert raw_locs.shape == (2, 10)


class TestConstructCircAndLinearTimeDs:
    @pytest.mark.parametrize("freq", ["D", "H", "MS", "ME"])
    def test_known_frequencies_produce_expected_variables(self, freq):
        dates = pd.date_range("2020-01-01", "2021-12-31", freq="MS")
        ds = construct_circ_and_linear_time_ds(dates, freq)
        assert f"cos_{freq}" in ds.data_vars
        assert f"sin_{freq}" in ds.data_vars
        assert "year" in ds.data_vars
        assert ds.sizes["time"] == len(dates)

    def test_unsupported_frequency_raises(self):
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        with pytest.raises(ValueError, match="not implemented"):
            construct_circ_and_linear_time_ds(dates, "bogus")

    @pytest.mark.parametrize(
        ("freq", "sample_dates", "time_var", "mod"),
        [
            (
                "D",
                ["2020-01-01", "2020-04-10", "2020-12-31"],
                np.array([1, 101, 366]),
                365.25,
            ),
            (
                "H",
                ["2020-01-01T00:00", "2020-01-01T06:00", "2020-01-01T18:00"],
                np.array([0, 6, 18]),
                24,
            ),
            (
                "MS",
                ["2020-01-01", "2020-04-01", "2020-07-01"],
                np.array([1, 4, 7]),
                12,
            ),
        ],
    )
    def test_cos_sin_values_for_each_frequency(self, freq, sample_dates, time_var, mod):
        dates = pd.to_datetime(sample_dates)
        ds = construct_circ_and_linear_time_ds(dates, freq)
        expected_cos = np.cos(2 * np.pi * time_var / mod)
        expected_sin = np.sin(2 * np.pi * time_var / mod)
        np.testing.assert_allclose(ds[f"cos_{freq}"].values, expected_cos)
        np.testing.assert_allclose(ds[f"sin_{freq}"].values, expected_sin)

    def test_year_is_normalised_to_minus_one_and_one(self):
        dates = pd.to_datetime(["2020-01-01", "2021-06-01", "2022-12-01"])
        ds = construct_circ_and_linear_time_ds(dates, "MS")
        np.testing.assert_allclose(ds["year"].values[[0, -1]], [-1.0, 1.0])

    def test_ms_and_me_produce_identical_values(self):
        dates = pd.to_datetime(["2020-01-01", "2020-04-01", "2020-07-01"])
        ds_ms = construct_circ_and_linear_time_ds(dates, "MS")
        ds_me = construct_circ_and_linear_time_ds(dates, "ME")
        np.testing.assert_array_equal(ds_ms["cos_MS"].values, ds_me["cos_ME"].values)
        np.testing.assert_array_equal(ds_ms["sin_MS"].values, ds_me["sin_ME"].values)
        np.testing.assert_array_equal(ds_ms["year"].values, ds_me["year"].values)


class TestSampleDatesByYearChunks:
    def test_returns_requested_count_per_year(self):
        weights = np.ones(366)
        sampled = sample_dates_by_year_chunks(weights, "2020-01-01", "2021-12-31", 3)
        assert len(sampled) == 3 * 2

    def test_dates_within_requested_range(self):
        weights = np.ones(366)
        start, end = "2020-01-01", "2021-12-31"
        sampled = sample_dates_by_year_chunks(weights, start, end, 3)
        assert all(
            pd.Timestamp(start) <= pd.Timestamp(d) <= pd.Timestamp(end) for d in sampled
        )


class TestFindLonLat:
    def test_finds_standard_names(self):
        ds = xr.Dataset(coords={"lon": [0, 1], "lat": [0, 1]})
        lon, lat = find_lon_lat(ds)
        assert lon.name == "lon"
        assert lat.name == "lat"

    def test_finds_long_names(self):
        ds = xr.Dataset(coords={"longitude": [0, 1], "latitude": [0, 1]})
        lon, lat = find_lon_lat(ds)
        assert lon.name == "longitude"
        assert lat.name == "latitude"

    def test_missing_coords_raises(self):
        ds = xr.Dataset(coords={"x": [0, 1], "y": [0, 1]})
        with pytest.raises(ValueError, match="Couldn't find lon/lat"):
            find_lon_lat(ds)


class TestFindXyForPoints:
    def test_snaps_to_nearest_grid_cell(self, native_grid_ds):
        points = gpd.GeoDataFrame({"geometry": [Point(11, 9), Point(0, 0)]})
        result = find_xy_for_points(native_grid_ds, points)
        np.testing.assert_array_equal(result["x"].to_numpy(), [1, 0])
        np.testing.assert_array_equal(result["y"].to_numpy(), [1, 0])

    def test_does_not_mutate_input(self, native_grid_ds):
        points = gpd.GeoDataFrame({"geometry": [Point(11, 9)]})
        find_xy_for_points(native_grid_ds, points)
        assert "x" not in points.columns
        assert "y" not in points.columns

    def test_preserves_other_columns(self, native_grid_ds):
        points = gpd.GeoDataFrame({"name": ["station_a"], "geometry": [Point(11, 9)]})
        result = find_xy_for_points(native_grid_ds, points)
        assert result["name"].tolist() == ["station_a"]


class TestNormalizeLons:
    def test_converts_0_360_to_minus_180_180(self):
        result = normalize_lons(np.array([0, 90, 200, 359]))
        np.testing.assert_array_equal(result, [0, 90, -160, -1])

    def test_leaves_already_normalized_untouched(self):
        lons = np.array([-170, 0, 170])
        result = normalize_lons(lons)
        np.testing.assert_array_equal(result, lons)


class TestTenYearStoreName:
    def test_appends_10yr_before_extension(self):
        assert ten_year_store_name("base_dataset.zarr") == "base_dataset_10yr.zarr"

    def test_only_replaces_trailing_extension(self):
        assert ten_year_store_name("v2.zarr/base.zarr") == "v2.zarr/base_10yr.zarr"

    def test_rejects_non_zarr_store(self):
        with pytest.raises(ValueError, match=r"Expected a \.zarr store"):
            ten_year_store_name("base_dataset.nc")


class TestPointsInPolygons:
    def test_marks_inside_and_outside_points(self):
        square = Polygon([(0, 0), (0, 10), (10, 10), (10, 0)])
        lon_grid, lat_grid = np.meshgrid([1.0, 20.0], [1.0, 20.0])
        mask = points_in_polygons(lon_grid, lat_grid, square)
        np.testing.assert_array_equal(mask, [[True, False], [False, False]])

    def test_holes_are_excluded(self):
        outer = [(0, 0), (0, 10), (10, 10), (10, 0)]
        hole = [(4, 4), (4, 6), (6, 6), (6, 4)]
        polygon_with_hole = Polygon(outer, holes=[hole])
        lon_grid, lat_grid = np.meshgrid([1.0, 5.0], [1.0, 5.0])
        mask = points_in_polygons(lon_grid, lat_grid, polygon_with_hole)
        assert mask[0, 0]  # (1, 1) inside outer, outside hole
        assert not mask[1, 1]  # (5, 5) inside the hole


class TestContextSamplingWithDelta:
    def test_single_group_appends_trailing_all(self):
        result = context_sampling_with_delta("X", [1])
        assert result == ["X", "all"]

    def test_repeats_first_group_by_its_delta_factor(self):
        result = context_sampling_with_delta("X", [2, 3])
        assert result == ["X", "X", "all", "all", "all", "all"]

    def test_only_first_group_uses_first_set_obj(self):
        result = context_sampling_with_delta("X", [1, 1])
        assert result == ["X", "all", "all"]

    def test_preserves_object_identity(self):
        stn_locs = object()
        result = context_sampling_with_delta(stn_locs, [2])
        assert result[0] is stn_locs
        assert result[1] is stn_locs

    def test_raises_on_empty_delta_factor_context(self):
        with pytest.raises(ValueError, match="zip"):
            context_sampling_with_delta("X", [])
