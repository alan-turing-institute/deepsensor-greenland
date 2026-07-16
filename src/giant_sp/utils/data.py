import datetime
from random import randrange
import os
import json
import wandb

import numpy as np
import xarray as xr
import pandas as pd

import torch

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from deepsensor.model.convnp import ConvNP
import scipy as sp

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def sample_land_locations(n_obs, lsm=None, seed=None, invert_mask=False):
    """
    Arguments:
        n_obs (int): number of points to sample
        lsm (DataArray): land-sea mask in which sea==0
        seed (int): random seed
        invert_mask (bool): default False. If True, sample from sea instead of land
    """
    if lsm is None:
        lsm = dl.load_aux_ds('global', 'lsm', resolution=0.25)['lsm']
        lsm = lsm.where(lsm > 0.5, other=0)
    lsm = lsm.astype(bool)
    if invert_mask:
        lsm = ~lsm

    lat_weights = np.cos(np.radians(lsm['lat']))
    lsm_weighted = lsm * lat_weights

    lsm_stacked = lsm_weighted.stack(points=('lat', 'lon')).compute()
    land_points = lsm_stacked.where(lsm_stacked, drop=True)

    probs = np.abs(land_points) / np.abs(land_points).sum()

    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(land_points.points.size, n_obs, replace=True, p=probs.values)
    sampled_points = land_points.isel(points=sampled_idx)

    lats = sampled_points['lat'].values
    lons = sampled_points['lon'].values
    raw_locs = np.array(list(zip(lats, lons))).transpose()  # (2, n_obs)

    return raw_locs

def sample_station_locations(stn_locs_df, n_obs, seed=None):
    rng = np.random.default_rng(seed)
    sampled_idx = rng.choice(len(stn_locs_df), n_obs, replace=False)
    sampled_stns = stn_locs_df.iloc[sampled_idx]

    raw_locs = sampled_stns.to_numpy().transpose()

    return raw_locs

def mixed_sampling_strategy(stn_file, n_obs, frac_stn=0.1, frac_land=0.9, lsm=None, seed=None):
    n_stn = int(frac_stn * n_obs)
    n_land = int(frac_land * n_obs)
    stn_locs = sample_station_locations(stn_file, n_stn, seed=seed)
    land_locs = sample_land_locations(n_land, lsm=lsm, seed=seed)

    raw_locs = np.concatenate((stn_locs, land_locs), axis=1)
    return raw_locs

def construct_circ_and_linear_time_ds(dates, freq):
    """
    Return an :class:`xarray.Dataset` containing a circular variable for time.
    The ``freq`` entry dictates the frequency of cycling of the circular
    variable. E.g.:
    Note: Can be applied after preprocessing since outputs are normalised.

        - ``'H'``: cycles once per day at hourly intervals
        - ``'D'``: cycles once per year at daily intervals
        - ``'M'``: cycles once per year at monthly intervals

    Args:
        dates (...):
            ...
        freq (...):
            ...

    Returns:
        :class:`xarray.Dataset`
            ...
    """
    dates = pd.DatetimeIndex(dates)
    if freq == "D":
        time_var = dates.dayofyear
        mod = 365.25
    elif freq == "H":
        time_var = dates.hour
        mod = 24
    elif freq == "MS" or freq == "ME":
        time_var = dates.month
        mod = 12
    else:
        raise ValueError("Circular time variable not implemented for this frequency.")

    cos_time = np.cos(2 * np.pi * time_var / mod)
    sin_time = np.sin(2 * np.pi * time_var / mod)
    # normalised year
    year_time = (((dates.year - dates.year.min()) / (dates.year.max() - dates.year.min()) * 2) - 1)

    ds = xr.Dataset(
        coords={"time": dates},
        data_vars={
            f"cos_{freq}": ("time", cos_time),
            f"sin_{freq}": ("time", sin_time),
            f"year": ("time", year_time),
        },
    )
    return ds

def sample_dates_by_year_chunks(weights, start_date, end_date, n_samples_per_year):
    """
    Source: https://github.com/eschlager/MeltEmulation/
    Sample dates within 12-month chunks starting from `start_date`, using the given weights.

    Parameters:
        dates (array-like): Array or Series of datetime-like objects.
        weights (array-like): Array of weights corresponding to each date.
        start_date (datetime-like): The start of the overall date range.
        end_date (datetime-like): The end of the overall date range.
        n_samples_per_year (int): Number of dates to sample per 12-month period.

    Returns:
        List of sampled dates.

    Source: https://github.com/eschlager/MeltEmulation/
    """
    sampled_dates = []

    # Ensure dates and weights are pandas Series
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    weights = weights[dates.dayofyear - 1]

    current_start = pd.Timestamp(start_date)
    while current_start < pd.Timestamp(end_date):
        # Define the end of the 12-month period
        next_year = current_start + pd.DateOffset(years=1)
        current_end = min(next_year, pd.Timestamp(end_date))

        # Select dates within this period
        mask = (dates >= current_start) & (dates < current_end)
        period_dates = dates[mask]
        period_weights = weights[mask]

        if len(period_dates) < n_samples_per_year:
            current_start = next_year
            continue  # Skip periods with too few dates

        # Normalize weights
        period_weights = period_weights / period_weights.sum()

        # Sample without replacement
        sampled = np.random.choice(
            period_dates,
            size=n_samples_per_year,
            replace=False,
            p=period_weights
        )

        sampled_dates.extend(sampled)
        current_start = next_year

    return sampled_dates

def find_lon_lat(ds):
    # source: https://github.com/eschlager/MeltEmulation/
    # Try common names (coords or data_vars)
    lon_names = [n for n in ds.coords if n.lower() in ('lon', 'longitude', 'long')]
    lat_names = [n for n in ds.coords if n.lower() in ('lat', 'latitude')]
    if not lon_names or not lat_names:
        raise ValueError("Couldn't find lon/lat variables!")
    else:
        lon = ds[lon_names[0]].values
        lat = ds[lat_names[0]].values
        return lon, lat

def find_xy_for_points(ds, gdf_points):
    """
    For each point in gdf_points (lat/lon geometry), find the nearest
    grid cell in ds and return its x,y coordinate values.
    ds must have 2D lon/lat coordinates and 1D x/y dims.
    """
    lon2d, lat2d = find_lon_lat(ds)

    X2d, Y2d = xr.broadcast(ds['x'], ds['y'])
    X2d = X2d.transpose(*ds['lon'].dims).values
    Y2d = Y2d.transpose(*ds['lon'].dims).values

    tree = sp.spatial.cKDTree(np.column_stack([lon2d.ravel(), lat2d.ravel()]))
    query_pts = np.column_stack([gdf_points.geometry.x.values, gdf_points.geometry.y.values])
    _, idx = tree.query(query_pts)

    gdf_points = gdf_points.copy()
    gdf_points['x'] = X2d.ravel()[idx]
    gdf_points['y'] = Y2d.ravel()[idx]
    return gdf_points