import os
import pandas as pd
import xarray as xr
import argparse
import yaml
import glob
import numpy as np
import json

import torch
import deepsensor.torch as deepsensor
from deepsensor.model.convnp import ConvNP
from deepsensor.active_learning.algorithms import GreedyAlgorithm
import deepsensor.active_learning.acquisition_fns as acq
from deepsensor.train.train import set_gpu_default_device
from deepsensor.data.processor import DataProcessor
import geopandas as gpd
import scipy as sp

# src
import sys

from giant_sp.data import HIRHAM5ZarrDataLoader
import giant_sp.utils.plots as myplt
from giant_sp.utils.data import sample_dates_by_year_chunks, find_xy_for_points

set_gpu_default_device() if torch.cuda.is_available() else torch.device('cpu')


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
        unet_channels=config['unet_channels'],
        unet_kernels=config['unet_kernels'],
        likelihood=config['likelihood'],
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


def main():
    # Trained model args
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_ref', default='y5f2wnx5')
    parser.add_argument('--weights_filename', default='best-weights-stnval.pt')
    # Expt settings
    parser.add_argument('--acq_fn', default='MeanStddev', type=str)
    parser.add_argument('--n_new', default=3, type=int)

    parser.add_argument('--search_start_date', default='2016-01-01', type=str)
    parser.add_argument('--search_end_date', default='2016-12-31', type=str)
    parser.add_argument('--every', default=1, type=int)
    parser.add_argument('--pred_resolution', default=0, type=int)

    parser.add_argument('--search_mask_file', default='')
    parser.add_argument('--search_resolution', default=20, type=int)
    parser.add_argument('--min_land', default=0.25, type=float)
    parser.add_argument('--mask_ant', action='store_true', default=False)

    parser.add_argument('--tag', default='')

    # args = parser.parse_args(args=[])
    args = parser.parse_args()
    tag = args.tag

    # Args
    run_id = '4flbw8xy'

    # Path and files
    files_path = f'outputs/{run_id}'
    config_file = 'src/giant_sp/configs/hirham5zarr.yaml'
    coords_path = config['data']['data_path']
    savefolder = os.path.join(files_path, 'demo')
    os.makedirs(savefolder, exist_ok=True)

    # Args
    config = load_config(config_file)
    source = config['data']['source']
    chunk_size = 100
    tvar = 'snmel_abs'

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

    target_var = config['data']['target_var_IDs']
    dates_start = args.search_start_date
    dates_end = args.search_end_date
    frequency = config['dataloader']['frequency']
    temporal_sampling_method = config['dataloader']['temporal_sampling_method']
    temporal_sampling_every = args.every

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
    truth_ds, data_processor, task_loader = data_loader.get_dataproc_taskload(dates=dates,
                                                                              data_processor=data_processor)
    config_file = os.path.join(files_path, 'model_config.json')
    weights_file = os.path.join(files_path, 'best-weights.pt')
    model = load_wandb_model(data_processor, task_loader, config_file, weights_file)

    # Load raw/GT data
    aod_raw_ds = data_loader.load_ds()
    aod_ds = data_processor(aod_raw_ds)

    # Define target grid and gridbox weights
    if args.pred_resolution == 0:
        # Default: AOD original grid
        target_grid = aod_raw_ds
    else:
        # Load a 0.25degree resolution dataset as starting point and coarsen to desired resolution
        _, lsm = data_loader.load_aux_ds(resolution="bedmachine")
        coarsen_factor = int(args.pred_resolution / 1.0)
        target_grid = lsm.coarsen({'lat': coarsen_factor, 'lon': coarsen_factor}, boundary='trim').mean()

    # Load Aeronet sensor locations
    ## station locations
    mask, _ = data_loader.load_aux_ds(resolution="bedmachine")
    coords_file = os.path.sep.join([coords_path, 'coords_only.zarr'])
    coords_ref = xr.open_dataset(coords_file)
    xy_coords_ref = (coords_ref['x'], coords_ref['y'])
    lonlat_coords_ref = (coords_ref['lon'], coords_ref['lat'])

    rename_map = {
        'lon': 'x',
        'lat': 'y'
    }

    mask_reference = mask.rename(rename_map)

    ds = mask_reference.reindex({'x': xy_coords_ref[0], 'y': xy_coords_ref[1]})
    ds = ds.assign_coords(lon=lonlat_coords_ref[0], lat=lonlat_coords_ref[1])

    stn_file = config['data']['station_locations_file']
    stn_locs_df = pd.read_csv(stn_file)[['latitude_last_valid', 'longitude_last_valid']]  # get lat/lon only
    gdf_points = gpd.GeoDataFrame(stn_locs_df,
                                  crs='epsg:4326',
                                  geometry=gpd.points_from_xy(stn_locs_df['longitude_last_valid'],
                                                              stn_locs_df['latitude_last_valid']))

    gdf_points = find_xy_for_points(ds, gdf_points)

    gdf_points['mask_val'] = ds.sel(
        x=xr.DataArray(gdf_points['x'].values, dims='points'),
        y=xr.DataArray(gdf_points['y'].values, dims='points'),
        method='nearest'
    ).values
    stn_locs_df_subset = gdf_points[gdf_points['mask_val'] == 1]
    raw_stn_locs = pd.DataFrame({'lat': stn_locs_df_subset.y, 'lon': stn_locs_df_subset.x})
    X_stn = data_processor.map_coord_array(raw_stn_locs.to_numpy().transpose())

    # Search space definition
    if args.search_mask_file == '':
        coarsen_factor = int(args.search_resolution / 1.0)
        _, lsm = data_loader.load_aux_ds(resolution="bedmachine")
        aux_raw_ds = lsm.coarsen({'lat': coarsen_factor, 'lon': coarsen_factor}, boundary='trim').mean()
        search_mask = aux_raw_ds.where(aux_raw_ds > args.min_land, other=0)  # search where True
    else:
        search_mask = xr.open_dataarray(args.search_mask_file)

    # Generate tasks
    tasks = task_loader(dates, (X_stn, 'all', 'all'))

    # Acquisition functions
    # Parallel
    if args.acq_fn == 'Random':
        acq_fn = acq.Random()
        tasks = tasks[:1]
    elif args.acq_fn == 'ContextDist':
        acq_fn = acq.ContextDist(context_set_idx=0)
    elif args.acq_fn == 'Stddev':
        acq_fn = acq.Stddev(model)
    # Sequential
    elif args.acq_fn == 'MeanStddev':
        acq_fn = acq.MeanStddev(model)
    elif args.acq_fn == 'MeanVariance':
        acq_fn = acq.MeanVariance(model)
    elif args.acq_fn == 'MeanMarginalEntropy':
        acq_fn = acq.MeanMarginalEntropy(model)
    elif args.acq_fn == 'JointEntropy':
        acq_fn = acq.JointEntropy(model)
    elif args.acq_fn == 'ExpectedImprovement':
        acq_fn = acq.ExpectedImprovement(model)
    # Oracle
    elif args.acq_fn == 'OracleRMSE':
        acq_fn = acq.OracleRMSE(model)
    elif args.acq_fn == 'OracleMAE':
        acq_fn = acq.OracleMAE(model)
    elif args.acq_fn == 'OracleMarginalNLL':
        acq_fn = acq.OracleMarginalNLL(model)
    elif args.acq_fn == 'OracleJointNLL':
        acq_fn = acq.OracleJointNLL(model)
    # Custom
    elif args.acq_fn == 'GreatCircleContextDist':
        acq_fn = myacqfns.GreatCircleContextDist(context_set_idx=0, data_processor=data_processor)
        tasks = tasks[:1]
    # Lognormal
    elif args.acq_fn == 'StddevLognormal':
        acq_fn = myacqfns.StddevLognormal(model, target_var_ID=aod_var)
    else:
        raise NotImplementedError(f'No deepsensor acquisition function named {args.acq_fn}')

    if args.acq_fn[:6] == 'Oracle':
        aod_ds.load()
        # Args for fns needing ground truth
        greedy_gt_args = {
            'query_infill': aod_ds[aod_var],
            'proposed_infill': aod_ds[aod_var],
            'task_loader': task_loader
        }
    else:
        greedy_gt_args = {}

    # Greedy algorithm
    greedy_alg = GreedyAlgorithm(
        model=model,
        X_t=target_grid,
        X_s=search_mask,
        X_s_mask=search_mask.astype(bool),
        N_new_context=args.n_new,
        progress_bar=True,
        **greedy_gt_args
    )

    # Checkpointing parameters
    savedir = os.path.join(files_path, 'placement', args.acq_fn)
    figdir = os.path.join(files_path, 'placement', 'figures')
    if not os.path.exists(savedir): os.makedirs(savedir)
    if not os.path.exists(figdir): os.makedirs(figdir)

    # New placements
    X_new_df, acq_fn_da = greedy_alg(acq_fn, tasks)

    # Save
    acq_fn_da.to_netcdf(os.path.join(savedir, f'AcqFn_{args.acq_fn}_{args.n_new}-iter_{tag}.nc'))
    X_new_df.to_csv(os.path.join(savedir, f'NewPlace_{args.acq_fn}_{args.n_new}-iter_{tag}.csv'))

    # Plot
    fig, ax = myplt.new_placements_plot_xy(acq_fn_da, X_new_df, raw_stn_locs[['lat', 'lon']], title=args.acq_fn)
    fig.savefig(os.path.join(figdir, f'newplacements_{args.acq_fn}_{args.n_new}-iter_{tag}.jpg'))


if __name__ == "__main__":
    main()