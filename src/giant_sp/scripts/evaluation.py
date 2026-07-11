import os
import xarray as xr
import yaml
import numpy as np
import pandas as pd

import torch
import deepsensor.torch
from deepsensor.model.convnp import ConvNP
from deepsensor.data.processor import DataProcessor
from giant_sp.data import HIRHAM5ZarrPredictionDataLoader, HIRHAM5ZarrDataLoader
import json
import matplotlib.pyplot as plt

import colorcet as cc
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

# Path and files
files_path = '/gws/pw/j07/aria_giant/wip/draft/acocac/outputs/present/hirham5/1h622vp9'
config_file = 'src/giant_sp/configs/hirham5zarr_prediction.yaml'
coords_path = '/gws/pw/j07/aria_giant/wip/draft/acocac/data/hirham5/processed/HIRHAM5-ERAInterim/v_02'

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
elif source == 'hirham5predzarr':
    data_loader = HIRHAM5ZarrPredictionDataLoader(config)
    td = np.timedelta64(12, 'h')
else:
    raise ValueError(f"Unknown source: {source}")

test_start = config['data']['test_start']
test_end = config['data']['test_end']
frequency = config['dataloader']['frequency']
temporal_sampling_every = 1

dates = pd.date_range(start=test_start, end=test_end, freq=frequency)[::temporal_sampling_every]
dates = list(np.sort(dates + td).astype('datetime64[ns]'))

data_processor = DataProcessor(files_path)
truth_ds, data_processor, task_loader = data_loader.get_dataproc_taskload(dates=dates, data_processor=data_processor)

config_file = os.path.join(files_path, 'model_config.json')
weights_file = os.path.join(files_path, 'best-weights.pt')
model = load_wandb_model(data_processor, task_loader, config_file, weights_file)

true_ds = data_loader.load_ds()
mask, _ = data_loader.load_aux_ds(resolution="bedmachine")

# predict
tasks = []
for date in dates:
    task = task_loader(date, ("all", "all"), "all", seed_override=42)
    tasks.append(task)

pred_ds = model.predict(tasks, X_t=true_ds, X_t_mask=mask)[tvar]
mean_da, truth_da = xr.align(pred_ds['mean'], truth_ds[tvar], join='inner')

# Randomly select dates
test_dates = [tasks[i]['time'] for i in range(len(tasks))]
n_dates = min(len(test_dates), 30)
np.random.seed(42)
dates_plt = np.random.choice(test_dates, n_dates, replace=False)

# Subselect and flatten datasets
true_plt      = truth_da.sel(time=dates_plt).values.flatten()
pred_mean_plt = pred_ds['mean'].sel(time=dates_plt).values.flatten()
pred_std_plt  = pred_ds['std'].sel(time=dates_plt).values.flatten()

# PIT histogram (area weighted)
# Compute PIT values
pit_values = sp.stats.norm.cdf(true_plt, loc=pred_mean_plt, scale=pred_std_plt)
# Plot
fig, ax = plt.subplots()
ax.hist(pit_values, bins=100, range=(0,1), density=True,
    label='ConvNP model')
ax.set_xlabel('Probability integral transform')
ax.set_ylabel('Density')
ax.set_title('PIT histogram')
ax.grid()
# Plot uniform line
ax.axhline(1., color='k', linestyle='--')

fig.savefig(os.path.join(files_path, 'test_figures', 'pit_histogram.png'))