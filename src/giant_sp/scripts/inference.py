import os
import xarray as xr
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import deepsensor.torch
from deepsensor.model.convnp import ConvNP
from deepsensor.data.processor import DataProcessor
from giant_sp.data import HIRHAM5ZarrDataLoader
import giant_sp.utils.training as tu
import giant_sp.utils.data as du
import giant_sp.utils.plots as myplt
import json
import datetime

import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy as sp

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_wandb_model(data_processor, task_loader, config_path, weights_path):

    with open(config_path, "r") as f:
        config = json.load(f)

    # Build model from config
    model_args = dict(
        unet_channels = config['unet_channels'],
        unet_kernels = config['unet_kernels'],
        likelihood = config['likelihood'],
    )

    model = ConvNP(
        data_processor,
        task_loader,
        **model_args
    )

    map_location = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model weights
    model.model.load_state_dict(torch.load(weights_path, map_location=map_location))

    return model

def export_zarr(aod_var, pred_ds, truth_ds, files_path, fname):
    pred_ds, truth_da = xr.align(pred_ds[aod_var], truth_ds[aod_var], join='inner')

    pred_final = xr.concat([truth_da, pred_ds['mean'], pred_ds['std']], dim='variable').assign_coords(variable=['snmel_true', 'snmel_pred', 'snmel_std'])
    pred_final = pred_final.to_dataset(dim='variable')

    coords_file = os.path.sep.join([config['data']['data_path'], 'coords_only.zarr'])
    coords_ref = xr.open_dataset(coords_file)
    xy_coords_ref = (coords_ref['x'], coords_ref['y'])
    lonlat_coords_ref = (coords_ref['lon'], coords_ref['lat'])

    rename_map = {
        'lon': 'x',
        'lat': 'y'
    }
    pred_final = pred_final.rename(rename_map)

    ds_xr = pred_final.reindex({'x': xy_coords_ref[0], 'y': xy_coords_ref[1]})
    ds_xr = ds_xr.assign_coords(lon=lonlat_coords_ref[0], lat=lonlat_coords_ref[1])
    ds_xr = ds_xr.chunk({'time': chunk_size, 'x': -1, 'y': -1})

    ds_xr.to_zarr(
                    store=os.path.join(files_path,f'{fname}.zarr'),
                    mode="w",
                    compute=True,
                    consolidated=True,
                    encoding={
                        var: {"chunks": ds_xr[var].data.chunksize, "compressor": None}
                        for var in ds_xr.data_vars
                    }
                )

# Path and files
files_path = 'outputs/4flbw8xy'
config_file = 'src/giant_sp/configs/hirham5zarr.yaml'
savefolder = os.path.join(files_path, 'test_results')
figfolder = os.path.join(savefolder, 'figures')
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
if not os.path.exists(figfolder):
    os.makedirs(figfolder)

# Args
val_set = 'station'
date_split = 'test'
seed = 321
plot = True
config = load_config(config_file)
source = config['data']['source']
chunk_size = 100
aod_var = config['data']['target_var_IDs']

# Initialize data loader
if source == 'carra2':
    data_loader = CARRA2DataLoader(config)
    td = np.timedelta64(0, 'ms')
elif source == 'hirham5':
    data_loader = HIRHAM5DataLoader(config)
    td = np.timedelta64(12, 'h')  # offset 12hours
elif source == 'hirham5zarr':
    data_loader = HIRHAM5ZarrDataLoader(config)
    td = np.timedelta64(12, 'h')
elif source == 'hirham5predzarr':
    data_loader = HIRHAM5ZarrPredictionDataLoader(config)
    td = np.timedelta64(12, 'h')
else:
    raise ValueError(f"Unknown source: {source}")

dates_start = config['data'][f'{date_split}_start']
dates_end = config['data'][f'{date_split}_end']
frequency = config['dataloader']['frequency']
temporal_sampling_method = 'regular'
temporal_sampling_every = 150
n_obs = 400

if temporal_sampling_method == 'seasonal':
    # source: https://github.com/eschlager/MeltEmulation/tree/main
    # temporal sub-sampling of training data
    doy = np.arange(1, 367)  # days 1 to 366
    mu = 205  # 24th of July ~ peak of melt season
    sigma = 60
    min_weight_doy = 0.0
    gauss_weights = np.exp(-0.5 * ((doy - mu) / sigma) ** 2)
    weights = min_weight_doy + (1 - min_weight_doy) * gauss_weights
    weights_doy = gauss_weights / gauss_weights.sum()

    n_samples_per_year = temporal_sampling_every

    np.random.seed(42)
    dates = sample_dates_by_year_chunks(weights_doy, dates_start, dates_end, n_samples_per_year)

    dates = list(np.sort(dates + td).astype('datetime64[ns]'))
elif temporal_sampling_method == 'regular':
    dates = pd.date_range(start=dates_start, end=dates_end, freq=frequency)[::temporal_sampling_every]
    dates = list(np.sort(dates + td).astype('datetime64[ns]'))
else:
    raise ValueError(f"Unknown temporal sampling method: {temporal_sampling_method}")

data_processor = DataProcessor(files_path)
aod_raw_ds, data_processor, task_loader = data_loader.get_dataproc_taskload(dates=dates, data_processor=data_processor)
lsm, _ = data_loader.load_aux_ds(resolution="bedmachine")

config_file = os.path.join(files_path, 'model_config.json')
weights_file = os.path.join(files_path, 'best-weights.pt')
model = load_wandb_model(data_processor, task_loader, config_file, weights_file)

# Validation - random locations
if val_set == 'random':
    # Generate tasks
    task_stream_val = tu.TaskStreamDataset(task_loader, dates, frequency=frequency, sampling=temporal_sampling_every, seed=123,
                                           datewise_deterministic=True, shuffle=False)
    val_tasks = [task for task in task_stream_val]

    # Save given context locations
    n_obs = [task['X_c'][0].shape[1] for task in val_tasks]

    # Predict
    val_pred = model.predict(val_tasks, X_t=aod_raw_ds, X_t_mask=lsm)
    val_mean_da = val_pred[aod_var]['mean']
    val_std_da = val_pred[aod_var]['std']

    fname = f'{aod_var}_{date_split}-{val_set}-e{temporal_sampling_every}_predictions'
    export_zarr(aod_var, val_pred, aod_raw_ds, savefolder, fname)

elif val_set == 'no-obs':
    tasks = task_loader(dates, context_sampling=(0, 'all', 'all'), target_sampling='all')

    # Predict
    print(datetime.datetime.now(), f'Starting prediction of {len(tasks)} tasks')
    stnval_pred = model.predict(tasks, X_t=aod_raw_ds, X_t_mask=lsm)  # pred_params=("mean", "std", "covariance")
    print(datetime.datetime.now(), f'Done')

    fname = f'{aod_var}_{date_split}-{val_set}-e{temporal_sampling_every}_predictions.nc'
    export_zarr(aod_var, val_pred, aod_raw_ds, savefolder, fname)
    print(datetime.datetime.now(), f'Saved')

elif val_set == 'random-land':
    print(datetime.datetime.now(), f'Starting task generation ({len(dates)})')
    val_tasks = [
        task_loader(
            date,
            context_sampling=(
                data_processor.map_coord_array(
                    du.sample_land_locations(
                        n_obs=n_obs,
                        lsm=lsm,
                        seed=(seed, i)
                    )
                ),
                "all", "all"),
            target_sampling="all",
            datewise_deterministic=True
        )
        for i, date in enumerate(dates)
    ]
    print(datetime.datetime.now(), 'Finished task generation')

    # Predict
    print(datetime.datetime.now(), f'Starting prediction of {len(val_tasks)} tasks')
    val_pred = model.predict(val_tasks, X_t=aod_raw_ds, X_t_mask=lsm)  # pred_params=("mean", "std", "covariance")
    val_mean_da = val_pred[aod_var]['mean']
    val_std_da = val_pred[aod_var]['std']
    print(datetime.datetime.now(), f'Done')

    fname = f'{aod_var}_{date_split}-{val_set}-seed{seed}-e{temporal_sampling_every}_predictions'
    export_zarr(aod_var, val_pred, aod_raw_ds, savefolder, fname)
    print(datetime.datetime.now(), f'Saved')

# Validation - station locations
elif val_set == 'station':
    # All station locations ever
    stn_file = config['data']['station_locations_file']
    stn_locs_df = pd.read_csv(stn_file)[['latitude_last_valid', 'longitude_last_valid']]  # get lat/lon only

    coords_file = os.path.sep.join([config['data']['data_path'], 'coords_only.zarr'])
    coords_ref = xr.open_dataset(coords_file)
    xy_coords_ref = (coords_ref['x'], coords_ref['y'])
    lonlat_coords_ref = (coords_ref['lon'], coords_ref['lat'])

    rename_map = {
        'lon': 'x',
        'lat': 'y'
    }

    # subset based on lsm mask
    mask_reference = lsm.rename(rename_map)
    mask_reference = mask_reference.reindex({'x': xy_coords_ref[0], 'y': xy_coords_ref[1]})
    mask_reference = mask_reference.assign_coords(lon=lonlat_coords_ref[0], lat=lonlat_coords_ref[1])

    gdf_points = gpd.GeoDataFrame(stn_locs_df,
                                  crs='epsg:4326',
                                  geometry=gpd.points_from_xy(stn_locs_df['longitude_last_valid'],
                                                              stn_locs_df['latitude_last_valid']))

    gdf_points = du.find_xy_for_points(mask_reference, gdf_points)

    gdf_points['mask_val'] = mask_reference.sel(
        x=xr.DataArray(gdf_points['x'].values, dims='points'),
        y=xr.DataArray(gdf_points['y'].values, dims='points'),
        method='nearest'
    ).values

    stn_locs_df_subset = gdf_points[gdf_points['mask_val'] == 1]

    raw_stn_locs = pd.DataFrame({'lat': stn_locs_df_subset.y, 'lon': stn_locs_df_subset.x})
    stn_locs = data_processor.map_coord_array(raw_stn_locs.to_numpy().transpose())

    # Create set of stnval tasks
    print(datetime.datetime.now(), f'Starting task generation ({len(dates)})')
    stnval_tasks = task_loader(
        dates,
        context_sampling=(stn_locs, "all", "all"),
        target_sampling="all"
    )
    print(datetime.datetime.now(), 'Finished task generation')

    # Predict
    print(datetime.datetime.now(), f'Starting prediction of {len(stnval_tasks)} tasks')
    stnval_pred = model.predict(stnval_tasks, X_t=aod_raw_ds, X_t_mask=lsm)
    print(datetime.datetime.now(), f'Done')

    fname = f'{aod_var}_{date_split}-{val_set}-e{temporal_sampling_every}_predictions'
    export_zarr(aod_var, stnval_pred, aod_raw_ds, savefolder, fname)
    print(datetime.datetime.now(), f'Saved')

    # Averaged prediction plots
    if plot:
        date = pd.Timestamp(dates[0])
        task = [t for t in stnval_tasks if t['time'] == date][0]
        stnval_mean_da, stnval_cams_raw_da = xr.align(stnval_pred[aod_var]['mean'], aod_raw_ds[aod_var], join='inner')
        fig, ax = myplt.averaged_prediction_plot(stnval_cams_raw_da, stnval_pred[aod_var]['mean'], stnval_pred[aod_var]['std'], cmap='viridis', task=task,
                                                 dataproc=data_processor, highlight_negatives=True, plot_xc=True, crs="None")
        fig.savefig(os.path.join(figfolder, f'{fname}_avgpred.png'))

        # PIT
        # Randomly select dates
        test_dates = [stnval_tasks[i]['time'] for i in range(len(stnval_tasks))]
        n_dates = min(len(test_dates), 30)
        np.random.seed(42)
        dates_plt = np.random.choice(test_dates, n_dates, replace=False)

        # Subselect and flatten datasets
        true_plt = stnval_cams_raw_da.sel(time=dates_plt).values.flatten()
        pred_mean_plt = stnval_pred[aod_var]['mean'].sel(time=dates_plt).values.flatten()
        pred_std_plt = stnval_pred[aod_var]['std'].sel(time=dates_plt).values.flatten()

        # PIT histogram (area weighted)
        # Compute PIT values
        pit_values = sp.stats.norm.cdf(true_plt, loc=pred_mean_plt, scale=pred_std_plt)
        # Plot
        fig, ax = plt.subplots()
        ax.hist(pit_values, bins=100, range=(0, 1), density=True,
                label='ConvNP model')
        ax.set_xlabel('Probability integral transform')
        ax.set_ylabel('Density')
        ax.set_title('PIT histogram')
        ax.grid()
        # Plot uniform line
        ax.axhline(1., color='k', linestyle='--')

        fig.savefig(os.path.join(figfolder, f'{fname}_pit_histogram_.png'))

else:
    raise ValueError('Unknown val_set, should be one of [random, station, active, reduced, station-reduced, grid]')

