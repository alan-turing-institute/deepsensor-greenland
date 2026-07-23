"""
Training manager for DeepSensor models
"""
import logging
from typing import Optional, Callable
import os
import glob
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import wandb
import random

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import matplotlib.colors as mcolors
matplotlib.use('Agg')

import pandas as pd
import xarray as xr

from deepsensor.model import ConvNP
import deepsensor.plot as dsplt
from deepsensor.data.processor import DataProcessor

from giant_sp.utils.model import compute_pred_ds
from giant_sp.utils.data import NumpyFloatValuesEncoder
import giant_sp.utils.training as tu
from giant_sp.utils.model import get_pred_DAs_taskstream
import giant_sp.utils.plots as myplt

logger = logging.getLogger(__name__)


class Trainer:
    """Training manager for DeepSensor models"""
    
    def __init__(self, model, config: dict, wandb_run=None):
        """
        Initialize trainer
        
        Args:
            model: DeepSensorModel instance
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        self.train_config = config['training']

        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        # Best model tracking
        self.best_val_loss = np.inf
        self.epochs_without_improvement = 0
        self.best_stnval_loss = np.inf
        
        # Initialize wandb if enabled
        self.use_wandb = config['logging'].get('wandb', {}).get('enabled', False)
        self.wandb_run = wandb_run  # may be pre-initialized from the training script

        # Plotting settings
        self.plot_dates = config['visualization']['plot_dates']
        self.res_factor_maps = config['visualization']['res_factor_maps']
        self.crs = config['visualization']['crs']
        self.cmap = config['visualization']['cmap']
        self.fig_size = config['visualization']['fig_size']

        if self.crs == "north_polar":
            self.projection = ccrs.NorthPolarStereo(central_longitude=-45)
        else:
            self.projection = ccrs.PlateCarree()

        if self.use_wandb & self.config['training']['resume']:
            self._restore_wandb()
        elif self.use_wandb & (not self.config['training']['resume']):
            if self.wandb_run is not None:
                self._setup_existing_wandb_run()
            else:
                self._init_wandb()

    def _create_optimizer(self):
        """Create optimizer based on config"""
        lr = self.train_config['learning_rate']
        weight_decay = self.train_config.get('weight_decay', 0.0)
        
        if self.train_config['optimizer'].lower() == 'adam':
            optimizer = optim.Adam(
                self.model.model.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif self.train_config['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.model.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif self.train_config['optimizer'].lower() == 'sgd':
            optimizer = optim.SGD(
                self.model.model.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.train_config['optimizer']}")
        
        logger.info(f"Created {self.train_config['optimizer']} optimizer with lr={lr}")
        return optimizer
    
    def _restore_wandb(self):
        """Initialize Weights & Biases logging"""
        run_id = self.config['training']['wandb_run_id']
        project_id = self.config['logging']['wandb']['project']

        # Restore wandb run
        self.wandb_run = wandb.init(
            project=project_id,
            id=run_id,
            resume="auto",
        )

        self.local_save_dir = os.path.join(self.wandb_run.dir, 'files')
        os.makedirs(self.local_save_dir)

        latest_model_epoch = self.wandb_run.summary['epoch']

        # Load model weights
        logger.info(f"Loading model weights of epoch {latest_model_epoch:02}...")
        weights_file = wandb.restore(f'files/last-weights.pt')
        self.model.model.model.load_state_dict(torch.load(weights_file.name))

        logger.info(f"Restore W&B run: {self.wandb_run.name}")
        logger.info(f"W&B URL: {self.wandb_run.url}")

    def _setup_existing_wandb_run(self):
        """Configure a W&B run that was already initialized before Trainer construction"""
        wandb_config = self.config['logging']['wandb']

        # Add model parameters now that the model is built
        self.wandb_run.config.update({'model_parameters': self.model.count_parameters()})

        if wandb_config.get('watch_model', True):
            wandb.watch(
                self.model.model.model,
                log='all',
                log_freq=wandb_config.get('watch_freq', 100)
            )

        self.local_save_dir = os.path.join(self.wandb_run.dir, 'files')
        os.makedirs(self.local_save_dir)

        logger.info(f"Saving data processor config")
        with open(os.path.join(self.local_save_dir, 'data_processor_config.json'), 'w') as f:
            json.dump(self.model.model.data_processor.config, f, indent=4, sort_keys=False)
        wandb.save(os.path.join(self.local_save_dir, 'data_processor_config.json'), base_path=self.wandb_run.dir)

        logger.info(f"Saving model config")
        config_fpath = os.path.join(self.local_save_dir, 'model_config.json')
        with open(config_fpath, "w") as f:
            json.dump(self.model.model.config, f, cls=NumpyFloatValuesEncoder, indent=4, sort_keys=False)
        wandb.save(os.path.join(self.local_save_dir, 'model_config.json'), base_path=self.wandb_run.dir)

    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb_config = self.config['logging']['wandb']

        # Initialize wandb run
        self.wandb_run = wandb.init(
            project=wandb_config.get('project', 'deepsensor-spatial-interpolation'),
            entity=wandb_config.get('entity'),
            name=wandb_config.get('name'),
            tags=wandb_config.get('tags', []),
            notes=wandb_config.get('docs', ''),
            config={
                'model': self.config['model'],
                'training': self.train_config,
                'data': self.config['data'],
                'model_parameters': self.model.count_parameters()
            }
        )

        # Watch model if enabled
        if wandb_config.get('watch_model', True):
            wandb.watch(
                self.model.model.model,
                log='all',
                log_freq=wandb_config.get('watch_freq', 100)
            )

        self.local_save_dir = os.path.join(self.wandb_run.dir, 'files')
        os.makedirs(self.local_save_dir)

        logger.info(f"Initialized W&B run: {self.wandb_run.name}")
        logger.info(f"W&B URL: {self.wandb_run.url}")

        # Save norm params
        logger.info(f"Saving data processor config")
        with open(os.path.join(self.local_save_dir, 'data_processor_config.json'), 'w') as f:
            json.dump(self.model.model.data_processor.config, f, indent=4, sort_keys=False)
        wandb.save(os.path.join(self.local_save_dir, 'data_processor_config.json'), base_path=self.wandb_run.dir)


    def train(self, truth_ds, task_stream_tr, val_tasks, mask=None, stnval_tasks=None,
              callback: Optional[Callable] = None):
        """
        Full training loop

        Args:
            truth_ds: Xarray Dataset containing the ground truth data for error computation
            task_stream_tr: List of training tasks
            val_tasks: List of validation tasks
            mask: Optional mask
            stnval_tasks: Optional list of station validation tasks
            callback: Optional callback function called after each epoch
        """
        num_epochs = self.train_config['num_epochs']
        model = self.model.model
        lr = self.train_config['learning_rate']
        target_var = self.config['data']['target_var_IDs']
        batch_size = self.config['training']['batch_size']
        log_stnval = self.config['logging']['log_stnval']
    
        self.pred_params = ['mean', 'std']
        if self.model.config['model']['likelihood'] == 'bernoulli-gamma':
            self.pred_params += ['mixture_probs', 'k', 'scale']

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(task_stream_tr.date_list)}")
        logger.info(f"Validation samples: {len(val_tasks)}")

        # Plot encoding and log it to wandb
        fig = dsplt.context_encoding(model, val_tasks[0], model.task_loader)
        wandb.log({'encoding': wandb.Image(fig)})

        # Wrap with torch DataLoader for training loop
        task_dataloader_tr = DataLoader(task_stream_tr, batch_size=batch_size, collate_fn=tu.task_collate_fn)

        if self.train_config['resume']:
            n0 = self.wandb_run.summary['epoch'] + 1
            nend = self.train_config['num_epochs']
        else:
            n0 = 0
            nend = self.train_config['num_epochs']
        for ep in tqdm(range(n0, nend), desc='Training'):
            # Train
            batch_losses = tu.train_epoch_with_dataloader(model, task_dataloader_tr, self.optimizer)
            epoch_loss = np.mean(batch_losses)
            logger.info(f'Starting train set prediction')
            _, train_std_da, train_err_da = get_pred_DAs_taskstream(model, task_stream_tr, 120, truth_ds, target_var, mask=mask)
            logger.info(f'Done, computing metrics')
            train_nll = (np.log(train_std_da) + np.power(train_err_da, 2) / (2 * np.power(train_std_da, 2))).mean().values.item()
            train_mae = np.abs(train_err_da).mean().values.item()
            train_rmse = np.sqrt(np.power(train_err_da, 2).mean()).values.item()

            # Val
            # Compute validation loss
            logger.info(f'Starting val loss computation')
            val_task_losses = []
            for task in val_tasks:
                val_task_losses.append(model.loss_fn(task, normalise=True).detach().cpu().numpy())
            epoch_loss_val = np.mean(val_task_losses)

            # Update history
            self.history['train_loss'].append(epoch_loss)
            self.history['val_loss'].append(epoch_loss_val)
            self.history['epoch'].append(ep + 1)

            # Save best val loss weights
            if epoch_loss_val < self.best_val_loss:
                self.best_val_loss = epoch_loss_val
                self.epochs_without_improvement = 0

                if self.train_config.get('save_best_only', True):
                    wandb.run.summary['best_val_loss'] =  self.best_val_loss
                    self.model.save_model(os.path.join(self.local_save_dir, 'best-weights.pt'))
                    wandb.save(os.path.join(self.local_save_dir, 'best-weights.pt'), base_path=self.wandb_run.dir)
                else:
                    self.epochs_without_improvement += 1

            # Early stopping
            if (self.train_config.get('early_stopping', False) and
                    self.epochs_without_improvement >= self.train_config['patience']):
                logger.info(f"Early stopping triggered after {ep + 1} epochs")
                break

            # Periodic checkpointing
            if ((ep + 1) % self.train_config.get('save_checkpoint_every', 10) == 0):
                torch.save(model.model.state_dict(), os.path.join(self.local_save_dir, f'weights-ep{ep:02}.pt'))
                wandb.save(os.path.join(self.local_save_dir, f'weights-ep{ep:02}.pt'), base_path=self.wandb_run.dir)

            # Track val metrics
            logger.info(f'Starting val set prediction')
            val_pred, val_err_ds = compute_pred_ds(model, val_tasks, truth_ds, target_var, self.pred_params, mask=mask)
            val_mean_ds = val_pred[target_var]['mean']
            val_std_ds = val_pred[target_var]['std']
            logger.info(f'Done, computing metrics')

            val_nll = (np.log(val_std_ds) + np.power(val_err_ds, 2) / (2 * np.power(val_std_ds, 2))).mean().values.item()
            val_mae = np.abs(val_err_ds).mean().values.item()
            val_rmse = np.sqrt(np.power(val_err_ds, 2).mean()).values.item()
            if 'lead_time' in val_mean_ds.dims:
                val_rmse_lt = np.power(val_err_ds, 2).mean(['init_time', 'lat', 'lon']) ** 0.5

            # Stnval
            if log_stnval and ((ep % 2 == 0) or ep == nend - 1):
                # Compute validation loss
                logger.info('Starting stnval loss computation')
                stnval_task_losses = []
                for task in stnval_tasks:
                    stnval_task_losses.append(model.loss_fn(task, normalise=True).detach().cpu().numpy())
                epoch_loss_stnval = np.mean(stnval_task_losses)
                # Save best val loss weights
                if epoch_loss_stnval < self.best_stnval_loss:
                    self.best_stnval_loss = epoch_loss_stnval
                    wandb.run.summary['best_stnval_loss'] = self.best_stnval_loss
                    torch.save(model.model.state_dict(), os.path.join(self.local_save_dir, 'best-weights-stnval.pt'))
                    wandb.save(os.path.join(self.local_save_dir, 'best-weights-stnval.pt'), base_path=self.wandb_run.dir)
                # Track val metrics
                logger.info(f'Starting stnval set prediction')
                stnval_pred, stnval_err_ds = compute_pred_ds(model, stnval_tasks, truth_ds, target_var, self.pred_params)
                stnval_std_ds = stnval_pred[target_var]['std']
                logger.info(f'Done, computing metrics')
                stnval_nll = (np.log(stnval_std_ds) + np.power(stnval_err_ds, 2) / (
                            2 * np.power(stnval_std_ds, 2))).mean().values.item()
                stnval_mae = np.abs(stnval_err_ds).mean().values.item()
                stnval_rmse = np.sqrt(np.power(stnval_err_ds, 2).mean()).values.item()

                if self.use_wandb:
                    ep_log = ep + 1
                    wandb.log({f'val/loss_station/{target_var}': epoch_loss_stnval}, step=ep_log)
                    wandb.log({f'val/nll_stn/{target_var}': stnval_nll}, step=ep_log)
                    wandb.log({f'val/mae_stn/{target_var}': stnval_mae}, step=ep_log)
                    wandb.log({f'val/rmse_stn/{target_var}': stnval_rmse}, step=ep_log)

            if self.use_wandb:
                ep_log = ep + 1
                self._log_wandb_metrics(ep_log, epoch_loss, epoch_loss_val)
                wandb.log({f'train/nll/{target_var}': train_nll}, step=ep_log)
                wandb.log({f'train/mae/{target_var}': train_mae}, step=ep_log)
                wandb.log({f'train/rmse/{target_var}': train_rmse}, step=ep_log)
                wandb.log({f'val/nll/{target_var}': val_nll}, step=ep_log)
                wandb.log({f'val/mae/{target_var}': val_mae}, step=ep_log)
                wandb.log({f'val/rmse/{target_var}': val_rmse}, step=ep_log)

                # Log images periodically
                wandb_config = self.config['logging']['wandb']
                if (wandb_config.get('log_maps', True) and
                    (ep + 1) % wandb_config.get('log_maps_frequency') == 0):
                    self._plot_additional_maps('val', val_tasks, model, truth_ds, mask, ep_log)

                    if log_stnval and ((ep % 2 == 0) or ep == nend - 1):
                        self._plot_additional_maps('stnval', stnval_tasks, model, truth_ds, mask, ep_log)

                if (wandb_config.get('log_charts', True) and
                    (ep + 1) % wandb_config.get('log_charts_frequency') == 0):
                    # self._log_wandb_predictions(truth_ds, ep)
                    self._plot_additional_charts('val', val_tasks, truth_ds, val_pred, ep_log)
                    if 'lead_time' in val_mean_ds.dims:
                        self._plot_additional_charts_forecast(val_rmse_lt, ep_log)

                    if log_stnval and ((ep % 2 == 0) or ep == nend - 1):
                        self._plot_additional_charts('stnval', stnval_tasks, truth_ds, stnval_pred, ep_log)

                # Save last checkpoint weights
                torch.save(model.model.state_dict(), os.path.join(self.local_save_dir, 'last-weights.pt'))
                wandb.save(os.path.join(self.local_save_dir, 'last-weights.pt'), base_path=self.wandb_run.dir)

        # Save final model weights
        torch.save(model.model.state_dict(), os.path.join(self.local_save_dir, 'final-weights.pt'))

        if self.use_wandb:
            logger.info(f"Saved model config")
            wandb.save(os.path.join(self.local_save_dir, 'final-weights.pt'), base_path=self.wandb_run.dir)
            self.wandb_run.finish()
            logger.info("W&B run finished")

    def _plot_additional_maps(self, partition, tasks, model, truth_ds, mask_ds, epoch):
        target_var = self.model.config["data"]["target_var_IDs"]
        pred, _ = compute_pred_ds(model, tasks, truth_ds, target_var, self.pred_params, mask=mask_ds, res_factor=self.res_factor_maps)
        mean_ds = pred[target_var]['mean']
        std_ds = pred[target_var]['std']
        # Multiple dates prediction plots
        self._log_wandb_predictions_task(partition, tasks, truth_ds[target_var], mean_ds, std_ds, epoch, cmap=self.cmap, crs=self.projection, plot_xc=False)
        # Averaged prediction plots
        fig, ax = myplt.averaged_prediction_plot(truth_ds[target_var], mean_ds, std_ds, cmap=self.cmap, crs=self.projection, highlight_negatives=False)
        wandb.log({f'predictions/plot_{partition}_average': wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    def _plot_additional_charts(self, partition, tasks, truth_ds, val_pred, epoch):
        target_var = self.model.config["data"]["target_var_IDs"]
        val_dates = [tasks[i]['time'] for i in range(len(tasks))]
        # Randomly select dates
        n_dates = min(len(val_dates), 30)
        np.random.seed(42)
        dates_plt = np.random.choice(val_dates, n_dates, replace=False)
        # Subselect and flatten datasets
        true_plt = truth_ds[target_var].sel(time=dates_plt).values.flatten()
        pred_plt = val_pred[target_var].sel(time=dates_plt)
        pred_mean_plt = pred_plt['mean'].values.flatten()

        # Parity plot (hexbin)
        fig, _ = myplt.hexbin_parity_plot(true_plt, pred_mean_plt)
        wandb.log({f'predictions/hexbin_{partition}_{n_dates}_dates': wandb.Image(fig)}, step=epoch)
        plt.close(fig)
        # PIT histogram (uncertainty calibration)
        fig, _ = myplt.PIT_histogram(true_plt, pred_plt, likelihood=self.model.config['model']['likelihood'])
        wandb.log({f'predictions/pit_{partition}_{n_dates}_dates': wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    def _log_wandb_metrics(self, epoch: int, train_loss: float, val_loss: float):
        """
        Log metrics to Weights & Biases
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
        """
        # Get current learning rate from optimizer
        current_lr = self.optimizer.param_groups[0]['lr']
        
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'val/loss': val_loss,
            'learning_rate': current_lr
        }, step=epoch)

    def _log_wandb_predictions_task(self, partition, tasks, truth_ds, mean_da, std_da, epoch: int, cmap="seismic", crs=ccrs.PlateCarree(), plot_xc=True):
        """
        Generate and log prediction visualizations to W&B

        Args:
            truth_ds: Xarray Dataset containing the ground truth data for error computation
            epoch: Current epoch number
        """

        try:
            # Make predictions
            ### Prepare tasks =============================================================
            if self.plot_dates is None:
                random.seed(0)
                dates_all = [t['time'] for t in tasks]
                dates = [
                    dates_all[i] for i in sorted(random.sample(range(len(dates_all)), 3))
                ]
            else:
                dates = self.plot_dates
                dates = [pd.Timestamp(d) for d in dates]

            mean_da, truth_da = xr.align(mean_da, truth_ds, join='inner')
            # Calculate prediction errors
            err_da = mean_da - truth_da

            fig1, axes = plt.subplots(nrows=len(dates), ncols=4, figsize=(4 * self.fig_size, len(dates) * self.fig_size),
                                      subplot_kw=dict(projection=crs))

            fig1.suptitle(f'{truth_da.name} predictions multiple dates - Epoch: {epoch}')

            for row_i, date in enumerate(dates):
                task = [t for t in tasks if t['time'] == date][0]
                N_c = task['X_c'][0].shape[1]

                vmin = min(mean_da.mean(dim='time').min().compute(),
                           truth_da.mean(dim='time').min().compute())
                vmax = max(mean_da.mean(dim='time').max().compute(),
                           truth_da.mean(dim='time').max().compute())
                if self.model.config['data']['target_var_IDs'][-3:] == 'anm':
                    maxabs = np.max(np.abs([vmin, vmax]))
                    var_clim = (-maxabs, maxabs)
                else:
                    var_clim = (vmin, vmax)

                ax = axes[row_i, 0]
                truth_da.sel(time=date).plot(ax=ax, center=False, cmap=cmap, vmin=var_clim[0], vmax=var_clim[1])
                ax.coastlines(linewidth=0.5)
                ax.set_title(f"Ground Truth \n Date: {date}'")

                ax = axes[row_i, 1]
                mean_da.sel(time=date).plot(ax=ax, center=False, cmap=cmap, vmin=var_clim[0], vmax=var_clim[1])
                ax.coastlines(linewidth=0.5)
                ax.set_title(f"ConvNP mean \n 'N_context: {N_c}")

                ax = axes[row_i, 2]
                std_da.sel(time=date).plot(ax=ax, center=False, cmap='Greys')
                ax.coastlines(linewidth=0.5)
                ax.set_title('ConvNP uncertainty (std)')

                ax = axes[row_i, 3]
                err_da.sel(time=date).plot(ax=ax, center=False, cmap='RdBu_r')
                ax.coastlines(linewidth=0.5)
                ax.set_title("ConvNP error")

                # Plot context locations on ground truth plot
                if plot_xc == True:
                    dsplt.offgrid_context(axes[row_i, 0], task, self.model.model.data_processor, s=3, markers="x", linewidths=0.5, colors='green',
                                          add_legend=False)

            plt.tight_layout()

            # Log to wandb
            wandb.log({
                f'predictions/plot_{partition}_dates': wandb.Image(fig1),
            }, step=epoch)

            plt.close(fig1)

        except Exception as e:
            logger.warning(f"Failed to log predictions to W&B: {e}")

