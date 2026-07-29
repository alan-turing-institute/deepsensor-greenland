import itertools

import deepsensor.torch as deepsensor  # noqa: F401
import numpy as np
import pandas as pd
import torch
from deepsensor import backend
from deepsensor.data.loader import TaskLoader
from deepsensor.data.task import Task, concat_tasks
from deepsensor.model.convnp import ConvNP
from deepsensor.model.nps import convert_task_to_nps_args
from torch.utils.data import IterableDataset
from tqdm import tqdm

from giant_sp.utils.data import mixed_sampling_strategy, sample_land_locations


# Based on DeepSensor issue #43
class TaskStreamDataset(IterableDataset):
    def __init__(
        self,
        task_loader: TaskLoader,
        date_list,
        frequency="D",
        sampling=1,
        aod_sampling_strat="n_obs",
        min_obs=50,
        max_obs=100,
        target_sampling="all",
        min_targets=5000,
        max_targets=10000,
        stations_location=None,
        data_processor=None,
        lsm=None,
        datewise_deterministic=False,
        seed=None,
        shuffle=True,
        change_dates_every_epoch=False,
    ):
        super().__init__()

        self.task_loader = task_loader

        # Compute list of dates to generate tasks for
        # date_list = list(pd.date_range(min_date, max_date, freq=frequency))[::sampling]
        # Shuffle
        if shuffle:
            self.date_list = np.random.permutation(date_list)
        else:
            self.date_list = date_list
        # self.min_date = min_date
        # self.max_date = max_date
        self.freq = frequency
        self.sampling = sampling

        self.aod_sampling_strat = aod_sampling_strat
        self.target_sampling = target_sampling
        self.min_targets = min_targets
        self.max_targets = max_targets

        if aod_sampling_strat in ["land_obs", "mixed_obs"] and (
            data_processor is None or lsm is None
        ):
            raise ValueError(
                "The chosen aod_sampling_strat requires data_processor and lsm to be passed!"
            )
        self.lsm = lsm
        self.data_processor = data_processor
        self.stations_location = stations_location

        self.seed = seed
        self.min_obs = min_obs
        self.max_obs = max_obs
        self.datewise_deterministic = datewise_deterministic
        self.change_dates_every_epoch = change_dates_every_epoch

    def __len__(self):
        return len(self.date_list)

    def __iter__(self):
        if self.change_dates_every_epoch:
            self.date_list = list(np.random.permutation(self.date_list))[
                :: self.sampling
            ]

        rng = np.random.default_rng(self.seed)
        for date in self.date_list:
            # Context sampling
            if (
                isinstance(self.aod_sampling_strat, str)
                and self.aod_sampling_strat == "n_obs"
            ):
                # Number of off-grid AOD context observations to sample
                aod_context = rng.integers(self.min_obs, self.max_obs)
            elif self.aod_sampling_strat == "land_obs":
                # Randomly sample observations from land only
                n_obs = rng.integers(self.min_obs, self.max_obs)
                dateseed = (
                    int(date.strftime("%Y%m%d"))
                    if self.datewise_deterministic
                    else None
                )
                raw_locs = sample_land_locations(n_obs, lsm=self.lsm, seed=dateseed)
                obs_locs = self.data_processor.map_coord_array(raw_locs)
                aod_context = obs_locs
            elif self.aod_sampling_strat == "mixed_obs":
                n_obs = rng.integers(self.min_obs, self.max_obs)
                dateseed = (
                    int(date.strftime("%Y%m%d"))
                    if self.datewise_deterministic
                    else None
                )
                raw_locs = mixed_sampling_strategy(
                    self.stations_location, n_obs, lsm=self.lsm, seed=dateseed
                )
                obs_locs = self.data_processor.map_coord_array(raw_locs)
                aod_context = obs_locs
            else:
                aod_context = self.aod_sampling_strat
            # Target sampling
            if self.target_sampling == "random":
                aod_target = rng.integers(self.min_targets, self.max_targets)
            else:
                aod_target = self.target_sampling

            task = (
                self.task_loader(
                    date,
                    context_sampling=(aod_context, "all", "all"),
                    target_sampling=aod_target,
                    datewise_deterministic=self.datewise_deterministic,
                )
                .remove_context_nans()
                .remove_target_nans()
            )
            yield task


class GapFillTaskStreamDataset(IterableDataset):
    def __init__(
        self,
        task_loader: TaskLoader,
        min_date,
        max_date,
        frequency="D",
        sampling=1,
        context_sampling=("all", "gapfill"),
        target_sampling="gapfill",
        data_processor=None,
        datewise_deterministic=False,
        seed=None,
        shuffle=True,
        change_dates_every_epoch=False,
    ):
        super().__init__()

        self.task_loader = task_loader

        # Compute list of dates to generate tasks for
        date_list = list(pd.date_range(min_date, max_date, freq=frequency))[::sampling]
        # Shuffle
        if shuffle:
            self.date_list = np.random.permutation(date_list)
        else:
            self.date_list = np.array(date_list)
        self.min_date = min_date
        self.max_date = max_date
        self.freq = frequency
        self.sampling = sampling

        self.context_sampling = context_sampling
        self.target_sampling = target_sampling
        self.data_processor = data_processor

        self.seed = seed
        self.datewise_deterministic = datewise_deterministic
        self.change_dates_every_epoch = change_dates_every_epoch

    def __len__(self):
        return len(self.date_list)

    def __iter__(self):
        if self.change_dates_every_epoch:
            self.date_list = list(
                np.random.permutation(
                    pd.date_range(self.min_date, self.max_date, freq=self.freq)
                )
            )[:: self.sampling]

        for date in self.date_list:
            task = (
                self.task_loader(
                    date,
                    context_sampling=self.context_sampling,
                    target_sampling=self.target_sampling,
                    datewise_deterministic=self.datewise_deterministic,
                )
                .remove_context_nans()
                .remove_target_nans()
            )
            yield task


def task_collate_fn(batch: list[Task]) -> dict:
    """Create a batch of samples.

    Args:
        batch: list of tasks that we want to batch
    """
    # Combine/batch tasks to a single task
    batched_and_padded = concat_tasks(batch)
    # Cast to tensor, add batch dim, mask NaNs
    batched_and_padded = ConvNP.modify_task(batched_and_padded)
    # Convert task into args to be passed to the nps model
    context_data, xt, yt, model_kwargs = convert_task_to_nps_args(batched_and_padded)
    return {
        "context_data": context_data,
        "xt": xt,
        "yt": yt,
        "model_kwargs": model_kwargs,
    }


# Train model for one epoch from DataLoader
def train_epoch_with_dataloader(
    model,
    task_dataloader,
    optimizer,
):

    def train_step(batch):
        optimizer.zero_grad()

        logpdfs = backend.nps.loglik(
            model.model,
            batch["context_data"],
            batch["xt"],
            batch["yt"],
            **batch["model_kwargs"],
            fix_noise=None,
            num_samples=8,
            normalise=True,
        )

        train_loss = -torch.mean(logpdfs)

        train_loss.backward()
        optimizer.step()

        return train_loss.detach().cpu().numpy()

    batch_losses = []
    for batch in tqdm(task_dataloader, desc="Batches"):
        batch_loss = train_step(batch)
        batch_losses.append(batch_loss)

    return batch_losses


# itertools.batched in Python 3.12
def batched(iterable, n):
    # batched('ABCDEFG', 3) → ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch
