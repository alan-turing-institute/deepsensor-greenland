#!/usr/bin/env python
"""
Main training script for DeepSensor spatial interpolation
"""
import os
import logging
import argparse
import yaml
from pathlib import Path
import wandb

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

import torch

import deepsensor.torch
from deepsensor.train import set_gpu_default_device

from giant_sp.data import CARRA2DataLoader, HIRHAM5DataLoader, HIRHAM5ZarrDataLoader
from giant_sp.models import DeepSensorModel
from giant_sp.training import Trainer
import giant_sp.utils.training as tu
from giant_sp.utils.data import sample_land_locations, sample_dates_by_year_chunks, find_xy_for_points


def setup_logging(config):
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['log_level'])

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    """Main training function"""
    # Load configuration
    config = load_config(args.config)
    source = config['data']['source']

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    # Initialize W&B early to track the full run (data loading, model build, training)
    wandb_run = None
    use_wandb = config['logging'].get('wandb', {}).get('enabled', False)
    resume = config['training'].get('resume', False)
    if use_wandb and not resume:
        wandb_cfg = config['logging']['wandb']
        wandb_run = wandb.init(
            project=wandb_cfg.get('project', 'deepsensor-spatial-interpolation'),
            entity=wandb_cfg.get('entity'),
            name=wandb_cfg.get('name'),
            tags=wandb_cfg.get('tags', []),
            notes=wandb_cfg.get('docs', ''),
            config={
                'model': config['model'],
                'training': config['training'],
                'data': config['data'],
            }
        )
        logger.info(f"Initialized W&B run: {wandb_run.name}")
        logger.info(f"W&B URL: {wandb_run.url}")

    logger.info(f"Active devices before updating default: {torch.get_default_device()}")
    set_gpu_default_device() if torch.cuda.is_available() else torch.device('cpu')
    logger.info(f"Active devices after updating default: {torch.get_default_device()}")
    logger.info(f"Configuration: {args.config}")

    # Initialize data loader
    if source == 'carra2':
        data_loader = CARRA2DataLoader(config)
        td =  np.timedelta64(0, 'ms')
    elif source == 'hirham5':
        data_loader = HIRHAM5DataLoader(config)
        td = np.timedelta64(12, 'h') #offset 12hours
    elif source == 'hirham5zarr':
        data_loader = HIRHAM5ZarrDataLoader(config)
        td = np.timedelta64(12, 'h')
    else:
        raise ValueError(f"Unknown source: {source}")

    # Define data partition
    train_start = config['data']['train_start']
    train_end = config['data']['train_end']
    val_start = config['data']['val_start']
    val_end = config['data']['val_end']

    # Define data loader parameters
    train_sampling = config['dataloader']['train_sampling']
    target_sampling = config['dataloader']['target_sampling']
    frequency = config['dataloader']['frequency']
    change_dates_every_epoch = config['dataloader']['change_dates_every_epoch']
    temporal_sampling_method = config['dataloader']['temporal_sampling_method']
    temporal_sampling_every = config['dataloader']['temporal_sampling_every']
    lsm_mask = config['dataloader']['lsm_mask']

    if temporal_sampling_method == 'seasonal':
        # source: https://github.com/eschlager/MeltEmulation/tree/main
        # temporal sub-sampling of training data
        doy = np.arange(1, 367)   # days 1 to 366
        mu = 205    # 24th of July ~ peak of melt season
        sigma = 60
        min_weight_doy = 0.0
        logger.info(f'Subsample training dates using seasonal weighting following gaussian distribution with mu {mu} and sigma {sigma}')
        gauss_weights = np.exp(-0.5 * ((doy - mu) / sigma) ** 2)
        weights = min_weight_doy + (1 - min_weight_doy) * gauss_weights
        weights_doy = gauss_weights / gauss_weights.sum()

        n_samples_per_year = temporal_sampling_every
        logger.info(f'Sample {n_samples_per_year} dates per year')

        np.random.seed(42)
        train_sampled_dates = sample_dates_by_year_chunks(weights_doy, train_start, train_end, n_samples_per_year)
        np.random.seed(42)
        val_sampled_dates = sample_dates_by_year_chunks(weights_doy, val_start, val_end, n_samples_per_year)

        train_dates = list(np.sort(train_sampled_dates + td).astype('datetime64[ns]'))
        val_dates = list(np.sort(val_sampled_dates + td).astype('datetime64[ns]'))
    elif temporal_sampling_method == 'regular':
        train_dates = pd.date_range(start=train_start, end=train_end, freq=frequency)[::temporal_sampling_every]
        val_dates = pd.date_range(start=val_start, end=val_end, freq=frequency)[::temporal_sampling_every]
        train_dates = list(np.sort(train_dates + td).astype('datetime64[ns]'))
        val_dates = list(np.sort(val_dates + td).astype('datetime64[ns]'))
    else:
        raise ValueError(f"Unknown temporal sampling method: {temporal_sampling_method}")

    # Load or create data
    logger.info("Loading data...")
    truth_ds, data_processor, task_loader = data_loader.get_dataproc_taskload(dates=train_dates+val_dates)

    # Load station locations
    stn_file = config['data']['station_locations_file']
    stn_locs_df = pd.read_csv(stn_file)[['latitude_last_valid', 'longitude_last_valid']]  # get lat/lon only

    if source == 'hirham5zarr':
        coords_file = os.path.join(config['data']['data_path'], 'coords_only.zarr')
        coords_ref = xr.open_zarr(coords_file, chunks='auto')
        xy_coords_ref = (coords_ref['x'], coords_ref['y'])
        lonlat_coords_ref = (coords_ref['lon'], coords_ref['lat'])

        rename_map = {
            'lon': 'x',
            'lat': 'y'
        }

        # subset based on lsm mask
        lsm = data_loader.load_aux_ds(resolution="bedmachine")
        mask_reference = lsm.rename(rename_map)
        mask_reference = mask_reference.reindex({'x': xy_coords_ref[0], 'y': xy_coords_ref[1]})
        mask_reference = mask_reference.assign_coords(lon=lonlat_coords_ref[0], lat=lonlat_coords_ref[1])

        gdf_points = gpd.GeoDataFrame(stn_locs_df,
                                      crs='epsg:4326',
                                      geometry=gpd.points_from_xy(stn_locs_df['longitude_last_valid'],
                                                                  stn_locs_df['latitude_last_valid']))

        gdf_points = find_xy_for_points(mask_reference, gdf_points)

        gdf_points['mask_val'] = mask_reference.sel(
            x=xr.DataArray(gdf_points['x'].values, dims='points'),
            y=xr.DataArray(gdf_points['y'].values, dims='points'),
            method='nearest'
        ).values

        stn_locs_df_subset = gdf_points[gdf_points['mask_val'] == 1]

        raw_stn_locs = pd.DataFrame({'lat': stn_locs_df_subset.y, 'lon': stn_locs_df_subset.x})
        stn_locs = data_processor.map_coord_array(raw_stn_locs.to_numpy().transpose())  # normalise to x1/x2 using data_processor

    else:
        gdf_points = gpd.GeoDataFrame(stn_locs_df,
                                      crs='epsg:4326',
                                      geometry=gpd.points_from_xy(stn_locs_df['longitude_last_valid'],
                                                                  stn_locs_df['latitude_last_valid'])).to_crs('epsg:3413')
        stn_locs_df_subset = gdf_points.cx[
            truth_ds.lon[0].values:truth_ds.lon[-1].values, truth_ds.lat[0].values:truth_ds.lat[-1].values]

        raw_stn_locs = pd.DataFrame({'lat': stn_locs_df_subset.geometry.y, 'lon': stn_locs_df_subset.geometry.x})
        stn_locs = data_processor.map_coord_array(raw_stn_locs.to_numpy().transpose())  # normalise to x1/x2 using data_processor

    # Create training tasks
    logger.info("Generating training tasks...")
    # Initialise training task generator
    # Train loader is unseeded so different context sampling is used each epoch
    if train_sampling == 'stn_obs':
        task_stream_tr = tu.TaskStreamDataset(
            task_loader,
            train_dates,
            frequency=frequency,
            sampling=temporal_sampling_every, change_dates_every_epoch=change_dates_every_epoch,
            aod_sampling_strat=stn_locs,
            target_sampling=target_sampling
        )
    elif train_sampling == 'land_obs':
        lsm = data_loader.load_aux_ds(resolution="bedmachine")
        task_stream_tr = tu.TaskStreamDataset(
            task_loader,
            train_dates,
            min_obs=100, max_obs=400,
            frequency=frequency,
            sampling=temporal_sampling_every, change_dates_every_epoch=change_dates_every_epoch,
            aod_sampling_strat='land_obs', data_processor=data_processor, lsm=lsm,
            target_sampling=target_sampling,
        )
    elif train_sampling == 'mixed_obs':
        lsm = data_loader.load_aux_ds(resolution="bedmachine")
        task_stream_tr = tu.TaskStreamDataset(
            task_loader,
            train_dates,
            min_obs=100, max_obs=400,
            frequency=frequency,
            sampling=temporal_sampling_every, change_dates_every_epoch=change_dates_every_epoch,
            aod_sampling_strat='mixed_obs', data_processor=data_processor, lsm=lsm,
            target_sampling=target_sampling, stations_location=raw_stn_locs
        )
    elif train_sampling == 'default':  # default random sampling
        task_stream_tr = tu.TaskStreamDataset(
            task_loader,
            train_dates,
            min_obs=100, max_obs=400,
            frequency=frequency,
            sampling=temporal_sampling_every, change_dates_every_epoch=change_dates_every_epoch,
            target_sampling=target_sampling
        )
    else:
        raise ValueError('Unknown train_sampling mode. Must be one of [default, stn_obs, land_obs, mixed_obs]')

    # Initialise validation task generator
    # with seeded rng and datewise_deterministic so tasks remain the same every epoch + for reproducibility
    rng = np.random.default_rng(123)
    if train_sampling == 'land_obs':
        val_tasks = [
            task_loader(
                date,
                context_sampling=(
                    data_processor.map_coord_array(
                        sample_land_locations(
                            n_obs=rng.integers(100, 400),
                            lsm=lsm,
                            seed=int(date)
                        )
                    ),
                    "all", "all"),
                target_sampling="all",
                datewise_deterministic=True
            ).remove_context_nans().remove_target_nans()
            for date in val_dates
        ]
    else:  # default
        val_tasks = [
            task_loader(
                date,
                context_sampling=(rng.integers(100, 400), "all", "all"),
                target_sampling="all",
                datewise_deterministic=True
            ).remove_context_nans().remove_target_nans()
            for date in val_dates
        ]
    logger.info(f'Finished val task generation ({len(val_dates)})')

    logger.info(f'Starting stnval task generation ({len(val_dates)})')
    stnval_tasks = [
        task_loader(
            date,
            context_sampling=(stn_locs, "all", "all"),
            target_sampling="all"
        ).remove_context_nans().remove_target_nans()
        for date in val_dates
    ]
    logger.info(f'Finished stnval task generation ({len(val_dates)})')

    # Build model
    logger.info("Building model...")
    model_wrapper = DeepSensorModel(config)
    model_wrapper.build_model(data_processor, task_loader)

    logger.info(f"Model parameters: {model_wrapper.count_parameters():,}")
    
    # Create trainer
    trainer = Trainer(model_wrapper, config, wandb_run=wandb_run)

    # Check mask
    if lsm_mask:
        lsm = data_loader.load_aux_ds(resolution="bedmachine")
    else:
        lsm = None

    # Train model
    logger.info("Starting training...")
    trainer.train(truth_ds, task_stream_tr, val_tasks, mask=lsm, stnval_tasks=stnval_tasks)

    logger.info("Training complete!")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DeepSensor model')
    parser.add_argument('--config', type=str, default='configs/carra2_debug.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    main(args)
