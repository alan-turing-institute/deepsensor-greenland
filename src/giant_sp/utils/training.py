"""Utilities for streaming DeepSensor training tasks and running training epochs.

This module provides ``IterableDataset`` implementations that lazily build
DeepSensor :class:`~deepsensor.data.task.Task` objects for a sequence of
dates (with configurable context/target sampling strategies), a collate
function that batches those tasks for the ``neuralprocesses`` backend, and a
helper to run a single training epoch over a ``DataLoader`` of such batches.
"""

import itertools
from collections.abc import Iterable, Iterator, Sequence
from typing import Any, TypeVar, cast

import deepsensor.torch as deepsensor  # noqa: F401
import numpy as np
import pandas as pd
import torch
import xarray as xr
from deepsensor import backend
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.data.task import Task, concat_tasks
from deepsensor.model.convnp import ConvNP
from deepsensor.model.nps import convert_task_to_nps_args
from numpy.typing import NDArray
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from giant_sp.utils.data import mixed_sampling_strategy, sample_land_locations

T = TypeVar("T")

#: A DeepSensor context/target sampling specification, mirroring
#: ``TaskLoader.__call__``: a named strategy (e.g. ``"all"``), a fixed
#: count/fraction, an array of coordinates, or a list mixing any of these.
SamplingSpec = str | int | float | NDArray[Any] | list[str | int | float | NDArray[Any]]


# Based on DeepSensor issue #43
class TaskStreamDataset(IterableDataset[Task]):
    """Iterable dataset that lazily generates :class:`Task` objects for a set of dates.

    For each date, off-grid context observations are (re)sampled on the
    fly according to ``sampling_strat``, and targets are (re)sampled
    according to ``target_sampling``. This avoids materialising every task
    up front, which is useful for large date ranges.

    Attributes
    ----------
    task_loader : TaskLoader
        DeepSensor task loader used to build a :class:`Task` for each date.
    date_list : NDArray[Any] | Sequence[pd.Timestamp]
        Dates to generate tasks for, optionally shuffled.
    freq : str
        Nominal frequency of ``date_list`` (kept for reference; not used to
        regenerate dates in this class).
    sampling : int
        Step used to subsample ``date_list`` when ``change_dates_every_epoch``
        is enabled.
    sampling_strat : SamplingSpec
        Strategy used to choose the number/location of context
        observations. One of ``"n_obs"``, ``"land_obs"``, ``"mixed_obs"``, or
        any value accepted directly as DeepSensor context sampling.
    target_sampling : SamplingSpec
        Strategy used to choose targets. ``"random"`` samples a random
        count between ``min_targets`` and ``max_targets``; any other value
        is passed through to the task loader unchanged.
    min_targets : int
        Lower bound (inclusive) for random target counts.
    max_targets : int
        Upper bound (exclusive) for random target counts.
    lsm : xr.DataArray | None
        Land-sea mask (sea == 0) used by the ``"land_obs"``/``"mixed_obs"``
        sampling strategies.
    data_processor : DataProcessor | None
        DeepSensor data processor used to normalise raw sampled coordinates.
    stations_location : pd.DataFrame | None
        Station locations used by the ``"mixed_obs"`` sampling strategy.
    seed : int | None
        Base random seed. If ``None``, sampling is unseeded (non-reproducible).
    min_obs : int
        Lower bound (inclusive) for random context observation counts.
    max_obs : int
        Upper bound (exclusive) for random context observation counts.
    datewise_deterministic : bool
        If True, per-date sampling (for ``"land_obs"``/``"mixed_obs"``) is
        seeded deterministically from the date itself.
    change_dates_every_epoch : bool
        If True, ``date_list`` is reshuffled and resubsampled at the start of
        every call to :meth:`__iter__`.
    """

    def __init__(
        self,
        task_loader: TaskLoader,
        date_list: Sequence[pd.Timestamp],
        frequency: str = "D",
        sampling: int = 1,
        sampling_strat: SamplingSpec = "n_obs",
        min_obs: int = 50,
        max_obs: int = 100,
        target_sampling: SamplingSpec = "all",
        min_targets: int = 5000,
        max_targets: int = 10000,
        stations_location: pd.DataFrame | None = None,
        data_processor: DataProcessor | None = None,
        lsm: xr.DataArray | None = None,
        datewise_deterministic: bool = False,
        seed: int | None = None,
        shuffle: bool = True,
        change_dates_every_epoch: bool = False,
    ) -> None:
        """Initialise the dataset.

        Parameters
        ----------
        task_loader : TaskLoader
            DeepSensor task loader used to build a :class:`Task` for each date.
        date_list : Sequence[pd.Timestamp]
            Dates to generate tasks for.
        frequency : str, default "D"
            Nominal frequency of ``date_list``, kept for reference.
        sampling : int, default 1
            Step used to subsample the (re)shuffled date list when
            ``change_dates_every_epoch`` is enabled.
        sampling_strat : SamplingSpec, default "n_obs"
            Strategy for sampling context observations. ``"n_obs"``
            samples a random count between ``min_obs`` and ``max_obs``;
            ``"land_obs"`` samples that many land-only locations from
            ``lsm``; ``"mixed_obs"`` mixes station and land locations; any
            other value is used directly as the context sampling spec.
        min_obs : int, default 50
            Lower bound (inclusive) for random context observation counts.
        max_obs : int, default 100
            Upper bound (exclusive) for random context observation counts.
        target_sampling : SamplingSpec, default "all"
            Strategy for sampling targets. ``"random"`` samples a random
            count between ``min_targets`` and ``max_targets``; any other
            value is passed through to the task loader unchanged.
        min_targets : int, default 5000
            Lower bound (inclusive) for random target counts.
        max_targets : int, default 10000
            Upper bound (exclusive) for random target counts.
        stations_location : pd.DataFrame | None, default None
            Station locations, required when ``sampling_strat`` is
            ``"mixed_obs"``.
        data_processor : DataProcessor | None, default None
            DeepSensor data processor used to normalise raw sampled
            coordinates. Required when ``sampling_strat`` is
            ``"land_obs"`` or ``"mixed_obs"``.
        lsm : xr.DataArray | None, default None
            Land-sea mask (sea == 0). Required when ``sampling_strat`` is
            ``"land_obs"`` or ``"mixed_obs"``.
        datewise_deterministic : bool, default False
            If True, per-date sampling is seeded deterministically from the
            date itself (for ``"land_obs"``/``"mixed_obs"``).
        seed : int | None, default None
            Base random seed for the per-iteration random generator.
        shuffle : bool, default True
            If True, ``date_list`` is randomly permuted once at construction
            time.
        change_dates_every_epoch : bool, default False
            If True, ``date_list`` is reshuffled and resubsampled (by
            ``sampling``) at the start of every call to :meth:`__iter__`.

        Raises
        ------
        ValueError
            If ``sampling_strat`` is ``"land_obs"`` or ``"mixed_obs"``
            and either ``data_processor`` or ``lsm`` is not provided.
        """
        super().__init__()

        self.task_loader = task_loader

        # Compute list of dates to generate tasks for
        # date_list = list(pd.date_range(min_date, max_date, freq=frequency))[::sampling]
        # Shuffle
        if shuffle:
            self.date_list: NDArray[Any] | Sequence[pd.Timestamp] = (
                np.random.permutation(date_list)
            )
        else:
            self.date_list = date_list
        # self.min_date = min_date
        # self.max_date = max_date
        self.freq = frequency
        self.sampling = sampling

        self.sampling_strat = sampling_strat
        self.target_sampling = target_sampling
        self.min_targets = min_targets
        self.max_targets = max_targets

        if sampling_strat in ["land_obs", "mixed_obs"] and (
            data_processor is None or lsm is None
        ):
            raise ValueError(
                "The chosen sampling_strat requires data_processor and lsm to be passed!"
            )
        self.lsm = lsm
        self.data_processor = data_processor
        self.stations_location = stations_location

        self.seed = seed
        self.min_obs = min_obs
        self.max_obs = max_obs
        self.datewise_deterministic = datewise_deterministic
        self.change_dates_every_epoch = change_dates_every_epoch

    def __len__(self) -> int:
        """Return the number of dates (and thus tasks) in the dataset.

        Returns
        -------
        int
            Length of ``date_list``.
        """
        return len(self.date_list)

    def __iter__(self) -> Iterator[Task]:
        """Yield one :class:`Task` per date in ``date_list``.

        Yields
        ------
        Task
            A task built by ``task_loader`` for each date, with context and
            target NaNs removed.
        """
        if self.change_dates_every_epoch:
            self.date_list = list(np.random.permutation(self.date_list))[
                :: self.sampling
            ]

        rng = np.random.default_rng(self.seed)
        for date in self.date_list:
            # Context sampling
            context: SamplingSpec | np.integer[Any]
            if (
                isinstance(self.sampling_strat, str)
                and self.sampling_strat == "n_obs"
            ):
                # Number of off-grid context observations to sample
                context = rng.integers(self.min_obs, self.max_obs)
            elif self.sampling_strat == "land_obs":
                # Randomly sample observations from land only
                n_obs = rng.integers(self.min_obs, self.max_obs)
                dateseed = (
                    int(date.strftime("%Y%m%d"))
                    if self.datewise_deterministic
                    else None
                )
                raw_locs = sample_land_locations(n_obs, lsm=self.lsm, seed=dateseed)
                obs_locs = cast(DataProcessor, self.data_processor).map_coord_array(
                    raw_locs
                )
                context = obs_locs
            elif self.sampling_strat == "mixed_obs":
                n_obs = rng.integers(self.min_obs, self.max_obs)
                dateseed = (
                    int(date.strftime("%Y%m%d"))
                    if self.datewise_deterministic
                    else None
                )
                raw_locs = mixed_sampling_strategy(
                    self.stations_location, n_obs, lsm=self.lsm, seed=dateseed
                )
                obs_locs = cast(DataProcessor, self.data_processor).map_coord_array(
                    raw_locs
                )
                context = obs_locs
            else:
                context = self.sampling_strat
            # Target sampling
            target: SamplingSpec | np.integer[Any]
            if self.target_sampling == "random":
                target = rng.integers(self.min_targets, self.max_targets)
            else:
                target = self.target_sampling

            task = (
                self.task_loader(
                    date,
                    context_sampling=(context, "all", "all"),
                    target_sampling=target,
                    datewise_deterministic=self.datewise_deterministic,
                )
                .remove_context_nans()
                .remove_target_nans()
            )
            yield task


class GapFillTaskStreamDataset(IterableDataset[Task]):
    """Iterable dataset that generates gap-filling :class:`Task` objects for a date range.

    Unlike :class:`TaskStreamDataset`, the date list is computed internally
    from ``min_date``/``max_date``/``frequency``, and context/target
    sampling strategies are fixed (typically DeepSensor's ``"gapfill"``
    sampling) rather than randomly (re)drawn per date.

    Attributes
    ----------
    task_loader : TaskLoader
        DeepSensor task loader used to build a :class:`Task` for each date.
    date_list : NDArray[Any] | list[pd.Timestamp]
        Dates to generate tasks for, optionally shuffled.
    min_date : str | pd.Timestamp
        Start of the date range (inclusive), as passed to
        :func:`pandas.date_range`.
    max_date : str | pd.Timestamp
        End of the date range (inclusive), as passed to
        :func:`pandas.date_range`.
    freq : str
        Frequency string passed to :func:`pandas.date_range`.
    sampling : int
        Step used to subsample the date range.
    context_sampling : SamplingSpec | tuple[SamplingSpec, ...]
        Context sampling spec passed to the task loader for every date.
    target_sampling : SamplingSpec
        Target sampling spec passed to the task loader for every date.
    data_processor : DataProcessor | None
        DeepSensor data processor (stored for reference; not otherwise used
        in this class).
    seed : int | None
        Base random seed (stored for reference; not otherwise used in this
        class).
    datewise_deterministic : bool
        Passed through to the task loader for each date.
    change_dates_every_epoch : bool
        If True, the date range is recomputed, reshuffled, and resubsampled
        at the start of every call to :meth:`__iter__`.
    """

    def __init__(
        self,
        task_loader: TaskLoader,
        min_date: str | pd.Timestamp,
        max_date: str | pd.Timestamp,
        frequency: str = "D",
        sampling: int = 1,
        context_sampling: SamplingSpec | tuple[SamplingSpec, ...] = (
            "all",
            "gapfill",
        ),
        target_sampling: SamplingSpec = "gapfill",
        data_processor: DataProcessor | None = None,
        datewise_deterministic: bool = False,
        seed: int | None = None,
        shuffle: bool = True,
        change_dates_every_epoch: bool = False,
    ) -> None:
        """Initialise the dataset.

        Parameters
        ----------
        task_loader : TaskLoader
            DeepSensor task loader used to build a :class:`Task` for each date.
        min_date : str | pd.Timestamp
            Start of the date range (inclusive).
        max_date : str | pd.Timestamp
            End of the date range (inclusive).
        frequency : str, default "D"
            Frequency string passed to :func:`pandas.date_range`.
        sampling : int, default 1
            Step used to subsample the date range.
        context_sampling : SamplingSpec | tuple[SamplingSpec, ...], default ("all", "gapfill")
            Context sampling spec passed to the task loader for every date.
        target_sampling : SamplingSpec, default "gapfill"
            Target sampling spec passed to the task loader for every date.
        data_processor : DataProcessor | None, default None
            DeepSensor data processor (stored for reference).
        datewise_deterministic : bool, default False
            Passed through to the task loader for each date.
        seed : int | None, default None
            Base random seed (stored for reference).
        shuffle : bool, default True
            If True, the computed date range is randomly permuted once at
            construction time.
        change_dates_every_epoch : bool, default False
            If True, the date range is recomputed, reshuffled, and
            resubsampled (by ``sampling``) at the start of every call to
            :meth:`__iter__`.
        """
        super().__init__()

        self.task_loader = task_loader

        # Compute list of dates to generate tasks for
        date_list = list(pd.date_range(min_date, max_date, freq=frequency))[::sampling]
        # Shuffle
        if shuffle:
            self.date_list: NDArray[Any] | list[pd.Timestamp] = np.random.permutation(
                date_list
            )
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

    def __len__(self) -> int:
        """Return the number of dates (and thus tasks) in the dataset.

        Returns
        -------
        int
            Length of ``date_list``.
        """
        return len(self.date_list)

    def __iter__(self) -> Iterator[Task]:
        """Yield one :class:`Task` per date in ``date_list``.

        Yields
        ------
        Task
            A task built by ``task_loader`` for each date, with context and
            target NaNs removed.
        """
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


def task_collate_fn(batch: list[Task]) -> dict[str, Any]:
    """Create a single batched, model-ready sample from a list of tasks.

    Parameters
    ----------
    batch : list[Task]
        List of tasks to combine into a single batch.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``"context_data"``, ``"xt"``, ``"yt"`` and
        ``"model_kwargs"``, as produced by
        :func:`deepsensor.model.nps.convert_task_to_nps_args`, ready to be
        passed to the ``neuralprocesses`` backend.
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
    model: ConvNP,
    task_dataloader: DataLoader[dict[str, Any]] | Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
) -> list[NDArray[np.floating[Any]]]:
    """Run one training epoch over a dataloader of collated task batches.

    For each batch (as produced by :func:`task_collate_fn`), computes the
    negative mean log-likelihood under ``model``, backpropagates, and takes
    an optimizer step.

    Parameters
    ----------
    model : ConvNP
        DeepSensor ConvNP model being trained.
    task_dataloader : DataLoader[dict[str, Any]] | Iterable[dict[str, Any]]
        Iterable of batches, each a dict with keys ``"context_data"``,
        ``"xt"``, ``"yt"`` and ``"model_kwargs"`` (see :func:`task_collate_fn`).
    optimizer : torch.optim.Optimizer
        Optimizer used to update ``model``'s parameters.

    Returns
    -------
    list[NDArray[np.floating[Any]]]
        Per-batch training losses (negative mean log-likelihood), detached
        and moved to CPU as NumPy scalars.
    """

    def train_step(batch: dict[str, Any]) -> NDArray[np.floating[Any]]:
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

    batch_losses: list[NDArray[np.floating[Any]]] = []
    for batch in tqdm(task_dataloader, desc="Batches"):
        batch_loss = train_step(batch)
        batch_losses.append(batch_loss)

    return batch_losses


# itertools.batched in Python 3.12
def batched(iterable: Iterable[T], n: int) -> Iterator[tuple[T, ...]]:
    """Batch an iterable into tuples of length ``n``.

    Backport of :func:`itertools.batched` (added in Python 3.12).
    ``batched('ABCDEFG', 3)`` yields ``ABC``, ``DEF``, ``G``.

    Parameters
    ----------
    iterable : Iterable[T]
        Items to group into batches.
    n : int
        Batch size. The final batch may be shorter if ``iterable``'s length
        is not a multiple of ``n``.

    Yields
    ------
    tuple[T, ...]
        Tuples of up to ``n`` consecutive items from ``iterable``.

    Raises
    ------
    ValueError
        If ``n`` is less than 1.
    """
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(itertools.islice(iterator, n)):
        yield batch
