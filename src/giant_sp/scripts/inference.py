import os
import glob
import argparse

import xarray as xr
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import deepsensor.torch
from deepsensor.model.convnp import ConvNP
from deepsensor.data.processor import DataProcessor
from giant_sp.data import CARRA2DataLoader, HIRHAM5DataLoader, HIRHAM5ZarrDataLoader
import giant_sp.utils.data as du
import giant_sp.utils.plots as myplt
import json
import datetime

import colorcet as cc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy as sp

import wandb

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def download_wandb_file(run_ref, files_path, file):
    api = wandb.Api()
    therun = api.run(run_ref)

    therun.file(file).download(root=os.path.join('wandb', f'run-{run_id}'), exist_ok=True)

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

def export_zarr(target_var, pred_ds, truth_ds, files_path, fname):
    pred_ds, truth_da = xr.align(pred_ds[target_var], truth_ds[target_var], join='inner')

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

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_file', default='src/giant_sp/configs/hirham5zarr_debug.yaml', type=str)
parser.add_argument('--val_set', default='station', type=str) # random, station, active, reduced, station-reduced
parser.add_argument('--n_obs', default=400, type=int)
parser.add_argument('--run_ref', default='acoca/giant-sp/4flbw8xy')
parser.add_argument('--weights_filename', default='best-weights-stnval.pt')
parser.add_argument('--date_split', default='test')
parser.add_argument('--every', default=200, type=int)
parser.add_argument('--seed', default=321, type=int)
parser.add_argument('--plot', action='store_true', default=False)
args = parser.parse_args()

# Args
config_file = args.config_file
run_ref = args.run_ref
weights_fn = args.weights_filename
n_obs = args.n_obs
val_set = args.val_set
date_split = args.date_split
every = args.every
seed = args.seed
plot = args.plot

# Set up output folders
run_id = run_ref.split('/')[-1]
output_path = f'outputs/{run_id}'
savefolder = os.path.join(output_path, 'test_results')
figfolder = os.path.join(savefolder, 'figures')
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
if not os.path.exists(figfolder):
    os.makedirs(figfolder)

config = load_config(config_file)
source = config['data']['source']
chunk_size = 100
target_var = config['data']['target_var_IDs']

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
else:
    raise ValueError(f"Unknown source: {source}")

dates_start = config['data'][f'{date_split}_start']
dates_end = config['data'][f'{date_split}_end']
frequency = config['dataloader']['frequency']

dates = pd.date_range(start=dates_start, end=dates_end, freq=frequency)[::every]
dates = list(np.sort(dates + td).astype('datetime64[ns]'))

# Locate data
target_files = ['model_config.json', 'data_processor_config.json', weights_fn]
try:
    files_path = sorted(glob.glob(f'wandb/run*{run_id}/files'), reverse=True)[0]
    if not all(os.path.exists(os.path.join(files_path, f)) for f in target_files):
        raise IndexError
except IndexError:
    files_path = str(os.path.join('wandb', f'run-{run_id}', 'files'))
    [download_wandb_file(run_ref, files_path, os.path.join('files',i)) for i in target_files]
print('Loading run data from: ', files_path)

data_processor = DataProcessor(files_path)
raw_ds, data_processor, task_loader = data_loader.get_dataproc_taskload(dates=dates, data_processor=data_processor)
lsm = data_loader.load_aux_ds(resolution="bedmachine")

config_file = os.path.join(files_path, 'model_config.json')
weights_file = os.path.join(files_path, weights_fn)

model = load_wandb_model(data_processor, task_loader, config_file, weights_file)

weights_fn = weights_fn.split('.')[0]
if val_set == 'no-obs':
    tasks = task_loader(dates, context_sampling=(0, 'all', 'all'), target_sampling='all')

    # Predict
    print(datetime.datetime.now(), f'Starting prediction of {len(tasks)} tasks')
    val_pred = model.predict(tasks, X_t=raw_ds, X_t_mask=lsm)  # pred_params=("mean", "std", "covariance")
    print(datetime.datetime.now(), f'Done')

    fname = f'{target_var}_{date_split}_{val_set}_e{every}_{weights_fn}_predictions'
    export_zarr(target_var, val_pred, raw_ds, savefolder, fname)
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
    val_pred = model.predict(val_tasks, X_t=raw_ds, X_t_mask=lsm)  # pred_params=("mean", "std", "covariance")
    val_mean_da = val_pred[target_var]['mean']
    val_std_da = val_pred[target_var]['std']
    print(datetime.datetime.now(), f'Done')

    fname = f'{target_var}_{date_split}_{val_set}_seed{seed}_e{every}_{weights_fn}_predictions'
    export_zarr(target_var, val_pred, raw_ds, savefolder, fname)
    print(datetime.datetime.now(), f'Saved')

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
    val_tasks = task_loader(
        dates,
        context_sampling=(stn_locs, "all", "all"),
        target_sampling="all"
    )
    print(datetime.datetime.now(), 'Finished task generation')

    # Predict
    print(datetime.datetime.now(), f'Starting prediction of {len(val_tasks)} tasks')
    val_pred = model.predict(val_tasks, X_t=raw_ds, X_t_mask=lsm)
    print(datetime.datetime.now(), f'Done')

    fname = f'{target_var}_{date_split}_{val_set}_e{every}_{weights_fn}_predictions'
    export_zarr(target_var, val_pred, raw_ds, savefolder, fname)
    print(datetime.datetime.now(), f'Saved')

else:
    raise ValueError('Unknown val_set, should be one of [station, random-land, no-obs]')

# Averaged prediction plots
if args.plot:
    date = pd.Timestamp(dates[0])
    task = [t for t in val_tasks if t['time'] == date][0]
    mean_da, raw_da = xr.align(val_pred[target_var]['mean'], raw_ds[target_var], join='inner')
    fig, ax = myplt.averaged_prediction_plot(raw_da, val_pred[target_var]['mean'], val_pred[target_var]['std'], cmap='viridis', task=task,
                                             dataproc=data_processor, highlight_negatives=True, plot_xc=True, crs="None")
    fig.savefig(os.path.join(figfolder, f'{fname}_avgpred.png'))

    # PIT
    # Randomly select dates
    test_dates = [val_tasks[i]['time'] for i in range(len(val_tasks))]
    n_dates = min(len(test_dates), 30)
    np.random.seed(42)
    dates_plt = np.random.choice(test_dates, n_dates, replace=False)

    # Subselect and flatten datasets
    true_plt = raw_ds[target_var].sel(time=dates_plt).values.flatten()
    pred_mean_plt = val_pred[target_var]['mean'].sel(time=dates_plt).values.flatten()
    pred_std_plt = val_pred[target_var]['std'].sel(time=dates_plt).values.flatten()

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