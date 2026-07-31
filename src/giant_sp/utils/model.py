"""Utility functions for model evaluation and prediction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import xarray as xr

from giant_sp.utils.training import batched

if TYPE_CHECKING:
    from deepsensor.data.task import Task
    from deepsensor.model.convnp import ConvNP
    from deepsensor.model.pred import Prediction


def compute_pred_ds(
    model: ConvNP,
    tasks: Task | list[Task],
    truth_ds: xr.Dataset,
    varname: str,
    pred_params: Sequence[str],
    mask: xr.DataArray | xr.Dataset | None = None,
    res_factor: int = 1,
) -> tuple[Prediction, xr.DataArray]:
    """Predict on a set of tasks and compute the error against ground truth.

    Runs `model.predict` for `tasks` and aligns the resulting mean
    prediction with the corresponding truth values (selecting the
    prediction's time/init_time coordinate from `truth_ds`) to compute the
    prediction error.

    Parameters
    ----------
    model : deepsensor.model.convnp.ConvNP
        Trained DeepSensor model used to generate predictions.
    tasks : deepsensor.data.task.Task or list[deepsensor.data.task.Task]
        Task(s) to predict on.
    truth_ds : xarray.Dataset
        Dataset holding the ground truth values, with ``lat``/``lon``
        coordinates and a variable named `varname`.
    varname : str
        Name of the target variable to predict and evaluate.
    pred_params : Sequence[str]
        Prediction parameters to compute (e.g. ``("mean", "std")``), passed
        through to `model.predict`.
    mask : xarray.DataArray, xarray.Dataset, or None, default=None
        Mask passed to `model.predict` as `X_t_mask`.
    res_factor : int, default=1
        Resolution factor passed to `model.predict`.

    Returns
    -------
    tuple[deepsensor.model.pred.Prediction, xarray.DataArray]
        The full `Prediction` object returned by `model.predict`, and the
        prediction error (predicted mean minus truth) as a DataArray aligned
        on the overlapping coordinates.
    """
    pred = model.predict(
        tasks,
        X_t=truth_ds[varname],
        X_t_mask=mask,
        resolution_factor=res_factor,
        pred_params=pred_params,
    )
    mean_ds = pred[varname]["mean"].assign_coords(
        {"latitude": truth_ds.lat, "longitude": truth_ds.lon}
    )
    cutout = pred[varname].init_time if pred.forecasting_mode else pred[varname].time
    mean_ds, truth_aligned = xr.align(
        mean_ds, truth_ds[varname].sel(time=cutout), join="inner"
    )
    err_ds = mean_ds - truth_aligned
    return pred, err_ds


def get_pred_DAs_taskstream(
    model: ConvNP,
    task_stream: Iterable[Task],
    batch_size: int,
    truth_ds: xr.Dataset,
    varname: str,
    mask: xr.DataArray | xr.Dataset | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Predict over a stream of tasks in batches and compute the error.

    Consumes `task_stream` in batches of `batch_size`, predicting on each
    batch and concatenating the resulting mean and standard deviation
    DataArrays along the time (or init_time, in forecasting mode) dimension.
    The concatenated mean is then aligned with `truth_ds` to compute the
    prediction error.

    Parameters
    ----------
    model : deepsensor.model.convnp.ConvNP
        Trained DeepSensor model used to generate predictions.
    task_stream : Iterable[deepsensor.data.task.Task]
        Iterable of individual tasks to predict on, consumed in batches of
        `batch_size`.
    batch_size : int
        Number of tasks to pass to `model.predict` at a time.
    truth_ds : xarray.Dataset
        Dataset holding the ground truth values, with ``lat``/``lon``
        coordinates and a variable named `varname`.
    varname : str
        Name of the target variable to predict and evaluate.
    mask : xarray.DataArray, xarray.Dataset, or None, default=None
        Mask passed to `model.predict` as `X_t_mask`.

    Returns
    -------
    tuple[xarray.DataArray, xarray.DataArray, xarray.DataArray]
        The concatenated predicted mean (`mean_da`, aligned with
        `truth_ds`), the concatenated predicted standard deviation
        (`std_da`, not aligned with `truth_ds`), and the prediction error
        (`err_da`, predicted mean minus truth).

    Notes
    -----
    Whether results are concatenated along ``"init_time"`` or ``"time"`` is
    determined by ``pred.forecasting_mode`` of the *last* batch's
    prediction; if `task_stream` yields no batches, `pred` is undefined and
    this raises `UnboundLocalError`.
    """
    mean_arrays = []
    std_arrays = []
    for tasks in batched(task_stream, batch_size):
        pred = model.predict(list(tasks), X_t=truth_ds, X_t_mask=mask)
        batch_mean_da = pred[varname]["mean"]
        batch_std_da = pred[varname]["std"]
        mean_arrays.append(batch_mean_da)
        std_arrays.append(batch_std_da)
    dim = "init_time" if pred.forecasting_mode else "time"
    mean_da = xr.concat(mean_arrays, dim=dim)
    std_da = xr.concat(std_arrays, dim=dim)
    mean_da = mean_da.assign_coords(
        {"latitude": truth_ds.lat, "longitude": truth_ds.lon}
    )
    mean_da, truth_aligned = xr.align(mean_da, truth_ds[varname], join="inner")
    err_da = mean_da - truth_aligned

    return mean_da, std_da, err_da
