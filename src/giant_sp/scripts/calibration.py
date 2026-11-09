"""Bernoulli-Gamma diagnostics for a trained ConvNP model used for inference.

Description: loads the model referenced by an inference config, predicts on a sample
of dates from one or more date splits (e.g. ``train``/``val``/``test``) and compares
the predictive distribution against the gridded truth. The Bernoulli part of the
likelihood is read as melt occurrence (melt / no melt) and the Gamma part as melt
amount.

Diagnostics written to ``<output_dir>/<run_id>/diagnostics/``:

- summary tables per split (melt probability, Gamma parameters, NLL, spread);
- NLL split by melt / no-melt targets, and the worst melt points by NLL;
- randomized PIT of the full mixture and PIT of the melt-only conditional Gamma;
- melt-occurrence calibration (reliability) table and plot;
- mean absolute calibration error (MACE, from ``uncertainty-toolbox``) per split,
  for the Gaussian approximation of the predictive distribution and for the
  Bernoulli-Gamma mixture itself (see `mace_table`);
- ensemble coverage per split and coverage level, from samples of each point's
  marginal predictive distribution (see `coverage_table`), with its average
  calibration error and a reliability plot;
- melt-only Gamma PIT summaries and P-P curves by regime (predicted Gamma mean,
  calendar month, daily context drivers, observed melt magnitude);
- val-minus-train deltas of the summary table, when both splits are evaluated.

Run with the inference config of the model to diagnose, e.g.::

    uv run src/giant_sp/scripts/calibration.py --config-name inference.future.local

Optional settings live under ``diagnostics`` (see `DIAGNOSTICS_DEFAULTS`) and are
added from the CLI with a ``+`` prefix unless the config already has the section,
e.g. ``+diagnostics.splits=[train,val,test] +diagnostics.max_dates=20``
(``splits`` defaults to ``[train,val]``). ``+diagnostics.date_sampling=seasonal``
draws dates weighted towards the melt season, as for training, instead of
taking every ``inference.every``-th date.
"""

import glob
import logging
import os
from collections.abc import Sequence
from typing import Any, cast

import deepsensor.torch  # noqa: F401
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import uncertainty_toolbox as uct
import xarray as xr
from deepsensor.data.loader import TaskLoader
from deepsensor.data.processor import DataProcessor
from deepsensor.data.task import Task
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from scipy.special import gammainc

import giant_sp.utils.data as du
from giant_sp.data import CARRA2DataLoader, HIRHAM5DataLoader, HIRHAM5ZarrDataLoader
from giant_sp.scripts.inference import download_wandb_file, load_wandb_model
from giant_sp.utils.data import context_sampling_with_delta

logger = logging.getLogger(__name__)

DIAGNOSTICS_DEFAULTS: dict[str, Any] = {
    # Date splits to evaluate; each needs `data.<split>_start`/`data.<split>_end`.
    # Train vs val (as in the source notebook) checks for overfitting; None uses
    # `inference.date_split` only.
    "splits": ["train", "val"],
    # How candidate dates are drawn from each split before `max_dates` subsampling:
    # "regular" takes every `inference.every`-th date at `dataloader.frequency`;
    # "seasonal" draws `seasonal_samples_per_year` dates per year weighted towards
    # the melt season, as `interpolation.py` does for training (with the default
    # seed, the train split reproduces the training dates exactly).
    "date_sampling": "regular",
    # Dates per year for "seasonal"; None uses `dataloader.temporal_sampling_every`
    # when the model was trained with seasonal sampling.
    "seasonal_samples_per_year": None,
    # Dates sampled per split (None keeps every date of the split).
    "max_dates": 30,
    # Grid points sampled per date (None keeps every land point).
    "max_points_per_date": 5000,
    # Observations strictly above this (raw units) count as melt.
    "melt_threshold": 0.0,
    # Context variables to bin melt-only PIT by; None uses the daily driver group
    # (`data.context_var_IDs[1]`).
    "regime_vars": None,
    # Quantile bins used for continuous regime variables and calibration.
    "n_regime_bins": 5,
    "n_calibration_bins": 50,
    # MACE: number of expected proportions in [0, 1], and whether observed
    # proportions are taken in centred prediction intervals ("interval") or
    # below predictive quantiles ("quantile").
    "mace_num_bins": 100,
    "mace_prop_type": "interval",
    # Coverage: ensemble members sampled per point from its marginal predictive
    # distribution, and nominal coverage levels (None: 0.05 to 0.95 by 0.05, as
    # autocast's MultiCoverage).
    "coverage_n_samples": 100,
    "coverage_levels": None,
    "seed": 42,
}

PRED_PARAMS = ["mean", "std", "mixture_probs", "k", "scale"]
OCCURRENCE_LABELS = {0: "no melt", 1: "melt"}


def locate_run_files(run_ref: str, files: Sequence[str]) -> str:
    """Return a local folder holding `files` for a W&B run, downloading if needed.

    Parameters
    ----------
    run_ref : str
        Fully qualified W&B run reference, e.g. ``"entity/project/run_id"``.
    files : Sequence[str]
        Filenames that must be present in the run's ``files/`` directory.

    Returns
    -------
    str
        Path to the local ``files/`` directory of the run.

    """
    run_id = run_ref.split("/")[-1]
    cached = sorted(glob.glob(f"wandb/run*{run_id}/files"), reverse=True)
    if cached and all(os.path.exists(os.path.join(cached[0], f)) for f in files):
        return cached[0]
    root_path = os.path.join("wandb", f"run-{run_id}")
    os.makedirs(root_path, exist_ok=True)
    for f in files:
        download_wandb_file(run_ref, root_path, f)
    return os.path.join(root_path, "files")


def candidate_split_dates(
    start: str,
    end: str,
    method: str,
    td: np.timedelta64,
    frequency: str = "D",
    every: int = 1,
    samples_per_year: int | None = None,
    seed: int = 42,
) -> list[np.datetime64]:
    """Candidate dates of one split, drawn as for training or inference.

    Parameters
    ----------
    start, end : str
        Split bounds (inclusive, as in ``data.<split>_start``/``_end``).
    method : {"regular", "seasonal"}
        ``"regular"``: every `every`-th date of a `frequency` date range, as in
        ``inference.py``. ``"seasonal"``: `samples_per_year` daily dates per
        12-month chunk weighted by `giant_sp.utils.data.seasonal_doy_weights`,
        as in ``interpolation.py``.
    td : numpy.timedelta64
        Offset added to every date (e.g. 12 h for HIRHAM5 daily means).
    frequency : str, default="D"
        Date-range frequency for ``"regular"``.
    every : int, default=1
        Stride for ``"regular"``.
    samples_per_year : int or None, default=None
        Dates per year for ``"seasonal"``; required for that method.
    seed : int, default=42
        Seed for ``"seasonal"``. It seeds NumPy's global generator the way
        ``interpolation.py`` does, so the same seed reproduces its dates.

    Returns
    -------
    list[numpy.datetime64]
        Sorted, offset dates (``datetime64[ns]``).

    Raises
    ------
    ValueError
        If `method` is unknown, or `samples_per_year` is missing for
        ``"seasonal"``.

    """
    if method == "regular":
        dates = np.asarray(pd.date_range(start=start, end=end, freq=frequency)[::every])
    elif method == "seasonal":
        if samples_per_year is None:
            raise ValueError("'seasonal' date sampling needs samples_per_year")
        np.random.seed(seed)
        dates = np.asarray(
            du.sample_dates_by_year_chunks(
                du.seasonal_doy_weights(), start, end, samples_per_year
            )
        )
    else:
        raise ValueError(
            f"Unknown date sampling {method!r}; use 'regular' or 'seasonal'."
        )
    return list(np.sort(dates + td).astype("datetime64[ns]"))


def seasonal_samples_per_year(configured: int | None, dataloader_config: Any) -> int:
    """Resolve ``diagnostics.seasonal_samples_per_year``.

    Parameters
    ----------
    configured : int or None
        The diagnostics value; returned as is when set.
    dataloader_config : Mapping
        The ``dataloader`` config section, used as fallback when the model was
        trained with ``temporal_sampling_method: seasonal``.

    Returns
    -------
    int
        Dates per year to draw.

    Raises
    ------
    ValueError
        If `configured` is None and the training config isn't seasonal.

    """
    if configured is not None:
        return int(configured)
    if dataloader_config.get("temporal_sampling_method") == "seasonal":
        return int(dataloader_config["temporal_sampling_every"])
    raise ValueError(
        "diagnostics.date_sampling=seasonal needs "
        "diagnostics.seasonal_samples_per_year, since the training config "
        "doesn't use seasonal sampling."
    )


def sample_dates(
    dates: Sequence[np.datetime64], n: int | None, seed: int
) -> list[np.datetime64]:
    """Sample `n` dates without replacement, keeping chronological order.

    Parameters
    ----------
    dates : Sequence[numpy.datetime64]
        Candidate dates.
    n : int or None
        Number of dates to keep; ``None`` or ``n >= len(dates)`` keeps all.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    list[numpy.datetime64]
        Sorted sampled dates.

    """
    dates = list(np.sort(np.asarray(dates)))
    if n is None or n >= len(dates):
        return dates
    idx = np.sort(np.random.default_rng(seed).choice(len(dates), size=n, replace=False))
    return [dates[i] for i in idx]


def prediction_frame(
    pred_ds: xr.Dataset,
    truth_da: xr.DataArray,
    regime_ds: xr.Dataset | None = None,
    max_points_per_date: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Flatten gridded predictions and truth into one row per (time, point).

    Points where the truth or any prediction parameter is non-finite (e.g.
    outside the land mask) are dropped, then at most `max_points_per_date`
    points are sampled per date.

    Parameters
    ----------
    pred_ds : xarray.Dataset
        Predictions for the target variable (``pred[target_var]``) with
        ``mixture_probs_0``/``mixture_probs_1``, ``k``, ``scale`` and ``std``.
    truth_da : xarray.DataArray
        Ground truth for the target variable, in raw units.
    regime_ds : xarray.Dataset or None, default=None
        Extra raw variables to carry along (e.g. context drivers for regime
        analysis), on the same grid.
    max_points_per_date : int or None, default=None
        Maximum points kept per date; ``None`` keeps all.
    seed : int, default=42
        Seed for the point subsampling.

    Returns
    -------
    pandas.DataFrame
        One row per sampled point with a ``time`` column, the spatial
        coordinates, ``y_true`` and every variable of `pred_ds`/`regime_ds`.

    """
    pred_aligned, truth_aligned = xr.align(pred_ds, truth_da, join="inner")
    ds = pred_aligned.assign(y_true=truth_aligned)
    if regime_ds is not None:
        ds, regime_aligned = xr.align(ds, regime_ds, join="inner")
        ds = ds.merge(regime_aligned)

    required = ["y_true", "mixture_probs_1", "k", "scale"]
    rng = np.random.default_rng(seed)
    frames = []
    for t in ds["time"].values:
        df = ds.sel(time=t).to_dataframe().reset_index()
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required)
        if max_points_per_date is not None and len(df) > max_points_per_date:
            df = df.iloc[
                np.sort(rng.choice(len(df), max_points_per_date, replace=False))
            ]
        df["time"] = pd.Timestamp(t)
        frames.append(df)
    if not frames:
        raise ValueError("No predictions to diagnose")
    return pd.concat(frames, ignore_index=True)


def enrich_prediction_columns(
    df: pd.DataFrame,
    data_processor: DataProcessor,
    target_var: str,
    melt_threshold: float = 0.0,
) -> pd.DataFrame:
    """Derive melt-occurrence, Gamma and NLL columns from raw prediction columns.

    DeepSensor unnormalises only ``mean``/``std``: the Gamma ``k``/``scale``
    stay in normalised units, so the truth is normalised (``y_true_norm``)
    before evaluating the Gamma density, and Gamma moments are reported both
    normalised (``*_norm``) and in raw units.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `prediction_frame`.
    data_processor : deepsensor.data.processor.DataProcessor
        Data processor the model was trained with.
    target_var : str
        Name of the target variable.
    melt_threshold : float, default=0.0
        Observations strictly above this (raw units) count as melt.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with ``melt_prob``, ``gamma_shape``, ``gamma_scale_norm``,
        ``gamma_rate``, ``gamma_mean[_norm]``, ``gamma_std[_norm]``,
        ``predictive_std``, ``y_true_norm``, ``observed_melt``,
        ``target_group`` and ``nll`` (in normalised units) columns.

    """
    df = df.copy()
    df["melt_prob"] = df["mixture_probs_1"].clip(1e-7, 1 - 1e-7)
    df["gamma_shape"] = df["k"]
    df["gamma_scale_norm"] = df["scale"].clip(lower=1e-12)
    df["gamma_rate"] = 1.0 / df["gamma_scale_norm"]
    df["gamma_mean_norm"] = df["gamma_shape"] * df["gamma_scale_norm"]
    df["gamma_std_norm"] = np.sqrt(df["gamma_shape"]) * df["gamma_scale_norm"]
    for col in ["gamma_mean", "gamma_std"]:
        df[col] = data_processor.map_array(
            df[f"{col}_norm"].to_numpy(), target_var, unnorm=True, add_offset=False
        )
    df["predictive_std"] = df["std"]
    df["y_true_norm"] = data_processor.map_array(
        df["y_true"].to_numpy(),
        target_var,
        method=data_processor.config[target_var]["method"],
    )

    melt = (df["y_true"] > melt_threshold).to_numpy()
    df["observed_melt"] = melt.astype(float)
    df["target_group"] = np.where(melt, OCCURRENCE_LABELS[1], OCCURRENCE_LABELS[0])

    nll = -np.log1p(-df["melt_prob"].to_numpy())
    nll[melt] = -np.log(df.loc[melt, "melt_prob"]) - sp.stats.gamma.logpdf(
        df.loc[melt, "y_true_norm"].clip(lower=1e-12),
        a=df.loc[melt, "gamma_shape"],
        scale=df.loc[melt, "gamma_scale_norm"],
    )
    df["nll"] = nll
    return df


def add_pit_columns(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Add the full-mixture PIT and the melt-only conditional Gamma PIT.

    No-melt observations sit on the point mass at zero, so their PIT is drawn
    uniformly across the CDF jump ``[0, P(no melt)]``; melt observations use
    the Gamma CDF above it. ``gamma_pit`` is the conditional Gamma CDF of melt
    observations only (NaN for no-melt rows).

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `enrich_prediction_columns`.
    seed : int, default=42
        Seed for the randomized PIT of no-melt observations.

    Returns
    -------
    pandas.DataFrame
        Copy of `df` with ``pit`` and ``gamma_pit`` columns.

    """
    df = df.copy()
    melt = df["observed_melt"].to_numpy() == 1
    p_no_melt = 1.0 - df["melt_prob"].to_numpy()
    gamma_cdf = gammainc(
        df["gamma_shape"].to_numpy(),
        df["y_true_norm"].to_numpy() / df["gamma_scale_norm"].to_numpy(),
    )
    rng = np.random.default_rng(seed)
    pit = rng.uniform(size=len(df)) * p_no_melt
    pit[melt] = p_no_melt[melt] + (1.0 - p_no_melt[melt]) * gamma_cdf[melt]
    df["pit"] = np.clip(pit, 0.0, 1.0)
    df["gamma_pit"] = np.where(melt, np.clip(gamma_cdf, 0.0, 1.0), np.nan)
    return df


def summary_table(diag: pd.DataFrame) -> pd.DataFrame:
    """Per-split summary of melt occurrence, Gamma parameters, spread and NLL."""
    return diag.groupby("split").agg(
        n=("y_true", "size"),
        observed_melt_freq=("observed_melt", "mean"),
        mean_pred_melt_prob=("melt_prob", "mean"),
        gamma_shape_mean=("gamma_shape", "mean"),
        gamma_shape_std=("gamma_shape", "std"),
        gamma_rate_mean=("gamma_rate", "mean"),
        gamma_rate_std=("gamma_rate", "std"),
        gamma_mean_mean=("gamma_mean", "mean"),
        gamma_mean_std=("gamma_mean", "std"),
        gamma_std_mean=("gamma_std", "mean"),
        gamma_std_std=("gamma_std", "std"),
        predictive_std_mean=("predictive_std", "mean"),
        predictive_std_median=("predictive_std", "median"),
        predictive_std_p90=("predictive_std", lambda s: s.quantile(0.90)),
        nll_mean=("nll", "mean"),
    )


def nll_by_target_group(diag: pd.DataFrame) -> pd.DataFrame:
    """Per-split NLL statistics for melt and no-melt targets."""
    return (
        diag.groupby(["split", "target_group"])
        .agg(
            n=("nll", "size"),
            observed_mean=("y_true", "mean"),
            pred_melt_prob_mean=("melt_prob", "mean"),
            nll_mean=("nll", "mean"),
            nll_median=("nll", "median"),
            nll_p90=("nll", lambda s: s.quantile(0.90)),
        )
        .reset_index()
    )


def melt_calibration(diag: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table of predicted melt probability vs observed melt frequency.

    Predicted probabilities are binned into (up to) `n_bins` equal-count bins
    within each split.
    """
    frames = []
    for split, split_df in diag.groupby("split"):
        bins = pd.qcut(
            split_df["melt_prob"].rank(method="first"),
            q=min(n_bins, len(split_df)),
            labels=False,
            duplicates="drop",
        )
        frames.append(
            split_df.groupby(bins)
            .agg(
                n=("observed_melt", "size"),
                mean_pred_melt_prob=("melt_prob", "mean"),
                observed_melt_freq=("observed_melt", "mean"),
            )
            .rename_axis("prob_bin")
            .reset_index()
            .assign(split=split)
        )
    return pd.concat(frames, ignore_index=True)


def pit_mace(
    pit: np.ndarray, num_bins: int = 100, prop_type: str = "interval"
) -> float:
    """Mean absolute calibration error of a sample of PIT values.

    ``uncertainty_toolbox`` assumes a Gaussian predictive distribution. Mapping
    each PIT value ``u`` to ``z = Phi^-1(u)`` and scoring ``z`` as a residual of a
    standard-normal prediction makes its Gaussian intervals/quantiles exactly the
    intervals/quantiles of the original predictive distribution, so the result
    is the MACE of that distribution (e.g. the Bernoulli-Gamma mixture).

    Parameters
    ----------
    pit : numpy.ndarray
        PIT values in ``[0, 1]``; NaNs are dropped.
    num_bins : int, default=100
        Number of expected proportions in ``[0, 1]``.
    prop_type : str, default="interval"
        ``"interval"`` (centred prediction intervals) or ``"quantile"``.

    Returns
    -------
    float
        MACE in ``[0, 0.5]``; NaN if there are no PIT values.

    """
    pit = np.asarray(pit, dtype=float)
    pit = pit[np.isfinite(pit)]
    if pit.size == 0:
        return float("nan")
    z = sp.stats.norm.ppf(np.clip(pit, 1e-12, 1 - 1e-12))
    return float(
        uct.mean_absolute_calibration_error(
            y_pred=z,
            y_std=np.ones_like(z),
            y_true=np.zeros_like(z),
            num_bins=num_bins,
            vectorized=True,
            prop_type=prop_type,
        )
    )


def mace_table(
    diag: pd.DataFrame, num_bins: int = 100, prop_type: str = "interval"
) -> pd.DataFrame:
    """Per-split mean absolute calibration error (MACE) of the predictions.

    MACE is the mean absolute difference between expected and observed
    proportions of targets inside the predictive intervals (``0`` is perfectly
    calibrated, ``0.5`` the worst). Three variants are reported:

    - ``mace_gaussian``: ``uncertainty_toolbox``'s MACE on the predictive
      ``mean``/``std`` (raw units), i.e. assuming a Gaussian predictive
      distribution - comparable with Gaussian-likelihood models but
      misspecified for the zero-inflated Bernoulli-Gamma mixture;
    - ``mace_mixture``: MACE of the Bernoulli-Gamma mixture itself, from the
      randomized ``pit`` (see `pit_mace`);
    - ``mace_melt_gamma``: MACE of the melt-only conditional Gamma, from
      ``gamma_pit`` (melt targets only).

    Parameters
    ----------
    diag : pandas.DataFrame
        Output of `add_pit_columns`, with a ``split`` column.
    num_bins : int, default=100
        Number of expected proportions in ``[0, 1]``.
    prop_type : str, default="interval"
        ``"interval"`` (centred prediction intervals) or ``"quantile"``.

    Returns
    -------
    pandas.DataFrame
        One row per split (index ``split``) with ``n``, ``n_melt`` and the
        three MACE columns.

    """
    rows = []
    for split, split_df in diag.groupby("split", sort=False):
        gauss = split_df[["mean", "std", "y_true"]].replace([np.inf, -np.inf], np.nan)
        gauss = gauss.dropna()
        gauss = gauss[gauss["std"] > 0]
        mace_gaussian = (
            float(
                uct.mean_absolute_calibration_error(
                    y_pred=gauss["mean"].to_numpy(dtype=float),
                    y_std=gauss["std"].to_numpy(dtype=float),
                    y_true=gauss["y_true"].to_numpy(dtype=float),
                    num_bins=num_bins,
                    vectorized=True,
                    prop_type=prop_type,
                )
            )
            if len(gauss)
            else float("nan")
        )
        rows.append(
            {
                "split": split,
                "n": len(split_df),
                "n_melt": int(split_df["gamma_pit"].notna().sum()),
                "mace_gaussian": mace_gaussian,
                "mace_mixture": pit_mace(
                    split_df["pit"].to_numpy(), num_bins, prop_type
                ),
                "mace_melt_gamma": pit_mace(
                    split_df["gamma_pit"].to_numpy(), num_bins, prop_type
                ),
            }
        )
    return pd.DataFrame(rows).set_index("split")


def default_coverage_levels() -> list[float]:
    """Nominal coverage levels 0.05, 0.10, ..., 0.95 (autocast's defaults)."""
    return [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]


def sample_marginal_ensemble(
    melt_prob: np.ndarray,
    gamma_shape: np.ndarray,
    gamma_scale: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw an ensemble from each point's Bernoulli-Gamma marginal prediction.

    Each member is ``0`` with probability ``1 - melt_prob`` and otherwise a
    ``Gamma(gamma_shape, gamma_scale)`` draw. Points are sampled independently
    (marginals only, no spatial correlation).

    Parameters
    ----------
    melt_prob, gamma_shape, gamma_scale : numpy.ndarray
        Per-point parameters, shape ``(N,)``.
    n_samples : int
        Ensemble members per point.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Samples of shape ``(N, n_samples)``, in the units of `gamma_scale`.

    """
    size = (len(melt_prob), n_samples)
    melt = rng.uniform(size=size) < np.asarray(melt_prob)[:, None]
    amount = rng.gamma(
        np.asarray(gamma_shape)[:, None], np.asarray(gamma_scale)[:, None], size=size
    )
    return np.where(melt, amount, 0.0)


def ensemble_coverage(
    samples: np.ndarray, y_true: np.ndarray, coverage_level: float
) -> np.ndarray:
    """Per-point coverage indicator of the central ensemble interval.

    Port of ``autocast.metrics.coverage.Coverage._score``
    (https://github.com/alan-turing-institute/autocast): ``y_true`` is covered
    when it lies within the ``0.5 -/+ coverage_level / 2`` empirical quantiles
    of the ensemble (linear interpolation, as ``torch.quantile``), bounds
    included.

    Parameters
    ----------
    samples : numpy.ndarray
        Ensemble of shape ``(N, M)``.
    y_true : numpy.ndarray
        Observations of shape ``(N,)``.
    coverage_level : float
        Nominal coverage in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(N,)``: ``1.0`` where covered, else ``0.0``.

    Raises
    ------
    ValueError
        If `coverage_level` is not in ``(0, 1)``.

    """
    if not 0 < coverage_level < 1:
        raise ValueError(f"coverage_level must be in (0, 1), got {coverage_level}")
    q_low, q_high = np.quantile(
        samples, [0.5 - coverage_level / 2, 0.5 + coverage_level / 2], axis=-1
    )
    y_true = np.asarray(y_true)
    return ((y_true >= q_low) & (y_true <= q_high)).astype(float)


def randomized_ensemble_pit(
    samples: np.ndarray, y_true: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Randomized PIT of `y_true` from its rank in the ensemble, ties broken at random.

    With ``n_less``/``n_equal`` members below/equal to the observation, its rank
    among the ``M + 1`` values is drawn uniformly from ``n_less, ...,
    n_less + n_equal`` and jittered within its slot, giving
    ``(rank + U) / (M + 1)``. For an exchangeable (calibrated) ensemble this is
    uniform on ``[0, 1]`` even with point masses, e.g. observed zeros against
    many zero members.

    Parameters
    ----------
    samples : numpy.ndarray
        Ensemble of shape ``(N, M)``.
    y_true : numpy.ndarray
        Observations of shape ``(N,)``.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        PIT values in ``[0, 1]``, shape ``(N,)``.

    """
    y = np.asarray(y_true)[:, None]
    n_less = (samples < y).sum(axis=-1)
    n_equal = (samples == y).sum(axis=-1)
    rank = n_less + rng.integers(0, n_equal + 1)
    return (rank + rng.uniform(size=len(rank))) / (samples.shape[-1] + 1)


def coverage_table(
    diag: pd.DataFrame,
    coverage_levels: Sequence[float] | None = None,
    n_samples: int = 100,
    seed: int = 42,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Per-split observed coverage of ensembles sampled from the predictions.

    For every point, `n_samples` members are drawn from its marginal
    Bernoulli-Gamma prediction (normalised units, see
    `sample_marginal_ensemble`) and scored against ``y_true_norm`` with
    `ensemble_coverage`. A calibrated model has observed coverage equal to the
    nominal level. No-melt targets (``0``) are covered whenever the lower
    ensemble quantile is ``0``, so dry points inflate coverage at low levels
    when ``P(no melt)`` is large. ``observed_coverage_randomized`` removes that
    by breaking ties at random: a point is covered when its
    `randomized_ensemble_pit` lies in ``[0.5 - level / 2, 0.5 + level / 2]``.

    Parameters
    ----------
    diag : pandas.DataFrame
        Output of `enrich_prediction_columns`, with a ``split`` column.
    coverage_levels : Sequence[float] or None, default=None
        Nominal levels; None uses `default_coverage_levels`.
    n_samples : int, default=100
        Ensemble members per point.
    seed : int, default=42
        Seed for the ensemble sampling.
    chunk_size : int, default=50_000
        Points sampled at once, to bound memory.

    Returns
    -------
    pandas.DataFrame
        One row per split and level with ``split``, ``coverage_level``, ``n``,
        ``observed_coverage``, ``observed_coverage_melt``,
        ``observed_coverage_no_melt``, ``observed_coverage_randomized``,
        ``abs_error`` (``|observed_coverage - coverage_level|``) and
        ``abs_error_randomized``.

    """
    levels = list(coverage_levels or default_coverage_levels())
    rng = np.random.default_rng(seed)
    rows = []
    for split, split_df in diag.groupby("split", sort=False):
        covered = np.empty((len(levels), len(split_df)))
        pit = np.empty(len(split_df))
        for start in range(0, len(split_df), chunk_size):
            chunk = split_df.iloc[start : start + chunk_size]
            samples = sample_marginal_ensemble(
                chunk["melt_prob"].to_numpy(),
                chunk["gamma_shape"].to_numpy(),
                chunk["gamma_scale_norm"].to_numpy(),
                n_samples,
                rng,
            )
            y_true = chunk["y_true_norm"].to_numpy()
            pit[start : start + len(chunk)] = randomized_ensemble_pit(
                samples, y_true, rng
            )
            for i, level in enumerate(levels):
                covered[i, start : start + len(chunk)] = ensemble_coverage(
                    samples, y_true, level
                )
        melt = split_df["observed_melt"].to_numpy() == 1
        for i, level in enumerate(levels):
            rows.append(
                {
                    "split": split,
                    "coverage_level": level,
                    "n": len(split_df),
                    "observed_coverage": covered[i].mean(),
                    "observed_coverage_melt": (
                        covered[i, melt].mean() if melt.any() else np.nan
                    ),
                    "observed_coverage_no_melt": (
                        covered[i, ~melt].mean() if (~melt).any() else np.nan
                    ),
                    "observed_coverage_randomized": np.mean(
                        np.abs(pit - 0.5) <= level / 2
                    ),
                }
            )
    table = pd.DataFrame(rows)
    table["abs_error"] = (table["observed_coverage"] - table["coverage_level"]).abs()
    table["abs_error_randomized"] = (
        table["observed_coverage_randomized"] - table["coverage_level"]
    ).abs()
    return table


def coverage_calibration_error(coverage: pd.DataFrame) -> pd.DataFrame:
    """Per-split mean ``|observed - nominal|`` coverage (autocast's MultiCoverage).

    ``coverage_calibration_error`` uses autocast's coverage;
    ``coverage_calibration_error_randomized`` the tie-broken coverage (see
    `coverage_table`).
    """
    return (
        coverage.groupby("split", sort=False)[["abs_error", "abs_error_randomized"]]
        .mean()
        .rename(
            columns={
                "abs_error": "coverage_calibration_error",
                "abs_error_randomized": "coverage_calibration_error_randomized",
            }
        )
    )


def plot_coverage(coverage: pd.DataFrame) -> Figure:
    """Reliability diagram of observed vs nominal ensemble coverage per split."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k:", linewidth=2, label="Expected")
    for i, (split, split_df) in enumerate(coverage.groupby("split", sort=False)):
        ax.plot(
            split_df["coverage_level"],
            split_df["observed_coverage"],
            color=f"C{i}",
            linewidth=2,
            label=split,
        )
        ax.plot(
            split_df["coverage_level"],
            split_df["observed_coverage_randomized"],
            color=f"C{i}",
            linestyle="-.",
            linewidth=2,
            label=f"{split} (randomized ties)",
        )
        for col, style in [
            ("observed_coverage_melt", "--"),
            ("observed_coverage_no_melt", ":"),
        ]:
            ax.plot(
                split_df["coverage_level"],
                split_df[col],
                color=f"C{i}",
                linestyle=style,
                alpha=0.6,
                label=f"{split} ({col.removeprefix('observed_coverage_')})",
            )
    ax.set(
        xlabel=r"Coverage level, $\alpha$",
        ylabel="Observed coverage",
        title="Ensemble coverage",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=8)
    return fig


def add_within_split_quantiles(
    df: pd.DataFrame, source_col: str, output_col: str, q: int = 5
) -> pd.DataFrame:
    """Label each row with its within-split quantile bin of `source_col`.

    Binning within each split keeps bins populated under train/test
    distribution shift. Constant columns get a single ``"constant (...)"`` bin.
    """
    df = df.copy()
    df[output_col] = pd.Series(index=df.index, dtype="object")
    for _, split_df in df.groupby("split"):
        values = split_df[source_col]
        bins = pd.qcut(values, q=q, duplicates="drop")
        if bins.notna().any():
            labels = bins.map(lambda iv: f"{iv.left:.3g} to {iv.right:.3g}")
            df.loc[split_df.index, output_col] = labels.astype(str)
        else:
            df.loc[split_df.index, output_col] = f"constant ({values.iloc[0]:.3g})"
    return df


def pit_group_summary(
    df: pd.DataFrame, source_col: str, group_col: str, label: str
) -> pd.DataFrame:
    """Uniformity statistics of melt-only Gamma PIT per split and regime group.

    For calibrated forecasts ``pit_mean`` is about 0.5 and each tail fraction
    about 0.05; excess ``pit_below_005`` means melt amounts are overpredicted,
    excess ``pit_above_095`` underpredicted. ``ks_distance`` is the maximum
    empirical-CDF departure from uniformity.
    """
    rows = []
    for (split, group), group_df in df.groupby(["split", group_col], observed=True):
        values = np.sort(group_df["gamma_pit"].to_numpy(dtype=float))
        n = len(values)
        rows.append(
            {
                "diagnostic": label,
                "split": split,
                "group": group,
                "n": n,
                "n_unique_times": group_df["time"].nunique(),
                "source_min": group_df[source_col].min(),
                "source_max": group_df[source_col].max(),
                "pit_mean": values.mean(),
                "pit_below_005": np.mean(values < 0.05),
                "pit_above_095": np.mean(values > 0.95),
                "ks_distance": max(
                    np.max(np.arange(1, n + 1) / n - values),
                    np.max(values - np.arange(n) / n),
                ),
            }
        )
    return pd.DataFrame(rows)


def plot_pit_by_split(diag: pd.DataFrame, col: str, xlabel: str, title: str) -> Figure:
    """Histogram of PIT column `col` per split against the uniform density."""
    splits = list(diag["split"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(6 * len(splits), 4.5),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, split in zip(axes.flat, splits, strict=True):
        values = diag.loc[diag["split"] == split, col].dropna()
        ax.hist(
            values,
            bins=20,
            range=(0.0, 1.0),
            density=True,
            color="tab:blue",
            alpha=0.7,
            edgecolor="black",
        )
        ax.axhline(
            1.0, color="black", linewidth=1, linestyle="--", label="Uniform reference"
        )
        ax.set(xlim=(0, 1), xlabel=xlabel, ylabel="Density", title=f"{title} - {split}")
        ax.legend()
    return fig


def plot_melt_calibration(
    calibration: pd.DataFrame, diag: pd.DataFrame, n_hist_bins: int = 50
) -> Figure:
    """Reliability diagram of melt occurrence, with a histogram of predictions.

    Parameters
    ----------
    calibration : pd.DataFrame
        Reliability table from `melt_calibration`.
    diag : pd.DataFrame
        Diagnostics frame with `split` and `melt_prob` columns; its predicted
        melt probabilities are histogrammed (equal-width bins over [0, 1], log
        count axis) under the reliability curve, showing how sparsely each part
        of the probability range is populated.
    n_hist_bins : int, optional
        Number of equal-width histogram bins over [0, 1].

    Returns
    -------
    Figure
        Reliability curve (top) and prediction histogram (bottom).
    """
    fig, (ax, ax_hist) = plt.subplots(
        2,
        1,
        figsize=(6, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    for split, split_df in calibration.groupby("split", sort=False):
        ax.plot(
            split_df["mean_pred_melt_prob"],
            split_df["observed_melt_freq"],
            marker="o",
            label=split,
        )
    ax.legend(title="split")
    ax.plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    ax.set(
        ylabel="Observed melt frequency",
        title="Melt occurrence calibration",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    hist_bins = np.linspace(0, 1, n_hist_bins + 1)
    for split in calibration["split"].unique():
        ax_hist.hist(
            diag.loc[diag["split"] == split, "melt_prob"],
            bins=hist_bins,
            histtype="step",
            label=split,
        )
    ax_hist.set(
        xlabel="Predicted melt probability",
        ylabel="Count",
        yscale="log",
        xlim=(0, 1),
    )
    return fig


def plot_distributions(diag: pd.DataFrame) -> Figure:
    """Per-split distributions of predicted parameters and NLL."""
    plot_cols = [
        "melt_prob",
        "gamma_shape",
        "gamma_rate",
        "gamma_mean",
        "gamma_std",
        "predictive_std",
        "nll",
    ]
    available = [c for c in plot_cols if np.isfinite(diag[c]).any()]
    fig, axes = plt.subplots(
        len(available),
        1,
        figsize=(9, 3 * len(available)),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, col in zip(axes.flat, available, strict=True):
        values = diag[col].replace([np.inf, -np.inf], np.nan)
        bins = np.histogram_bin_edges(values.dropna(), bins=50)
        for split in diag["split"].drop_duplicates():
            ax.hist(
                values[diag["split"] == split].dropna(),
                bins=bins,
                density=True,
                histtype="step",
                linewidth=1.5,
                label=split,
            )
        ax.set(title=col, xlabel=col, ylabel="Density")
        ax.legend(title="split")
    return fig


def plot_nll_by_target_group(diag: pd.DataFrame) -> Figure:
    """Box plot of point NLL for melt vs no-melt targets per split."""
    fig, ax = plt.subplots(figsize=(8, 4))
    finite = diag[np.isfinite(diag["nll"])]
    groups = list(OCCURRENCE_LABELS.values())
    splits = list(finite["split"].drop_duplicates())
    width = 0.8 / len(splits)
    for i, split in enumerate(splits):
        split_df = finite[finite["split"] == split]
        box = ax.boxplot(
            [split_df.loc[split_df["target_group"] == g, "nll"] for g in groups],
            positions=np.arange(len(groups)) + (i - (len(splits) - 1) / 2) * width,
            widths=width * 0.9,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black"},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(f"C{i}")
        box["boxes"][0].set_label(split)
    ax.set_xticks(np.arange(len(groups)), groups)
    ax.legend(title="split")
    ax.set(
        title="NLL split by melt vs no-melt targets",
        xlabel="Observed melt occurrence",
        ylabel="Point NLL",
    )
    return fig


def plot_regime_pp(df: pd.DataFrame, group_col: str, title: str) -> Figure:
    """P-P curves of melt-only Gamma PIT per regime group and split.

    Curves above the diagonal mean melt amounts are overpredicted, below it
    underpredicted; S-shapes indicate wrong spread or shape.
    """
    splits = list(df["split"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(splits),
        figsize=(7 * len(splits), 5),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, split in zip(axes.flat, splits, strict=True):
        split_df = df[df["split"] == split]
        for group, values in split_df.groupby(group_col, observed=True)["gamma_pit"]:
            values = np.sort(values.to_numpy(dtype=float))
            empirical = np.arange(1, len(values) + 1) / len(values)
            ax.plot(
                values, empirical, linewidth=1.5, label=f"{group} (n={len(values):,})"
            )
        ax.plot(
            [0, 1], [0, 1], color="black", linewidth=1, linestyle="--", label="Uniform"
        )
        ax.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Melt-only Gamma PIT",
            ylabel="Empirical CDF",
            title=f"{title} - {split}",
        )
        ax.legend(fontsize=8, loc="best")
    return fig


def build_tasks(
    task_loader: TaskLoader,
    dates: Sequence[np.datetime64],
    val_set: str,
    delta_factor_context: list[int],
    data_processor: DataProcessor,
    lsm: xr.DataArray,
    n_obs: int,
    seed: int,
) -> list[Task]:
    """Build one task per date with the same context sampling as ``inference.py``.

    Raises
    ------
    NotImplementedError
        If `val_set` is not ``"no-obs"`` or ``"random-land"``.

    """
    if val_set == "no-obs":
        return task_loader(
            dates,
            context_sampling=context_sampling_with_delta(0, delta_factor_context),
            target_sampling="all",
        )
    if val_set == "random-land":
        return [
            task_loader(
                date,
                context_sampling=context_sampling_with_delta(
                    data_processor.map_coord_array(
                        du.sample_land_locations(n_obs=n_obs, lsm=lsm, seed=(seed, i))
                    ),
                    delta_factor_context,
                ),
                target_sampling="all",
                datewise_deterministic=True,
            )
            for i, date in enumerate(dates)
        ]
    raise NotImplementedError(
        f"inference.val_set={val_set!r} is not supported for diagnostics; "
        "use 'no-obs' or 'random-land'."
    )


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(config: DictConfig) -> None:
    """Compute Bernoulli-Gamma (melt occurrence + amount) diagnostics for a model.

    Uses ``config.inference`` (``run_ref``, ``weights_filename``, ``val_set``,
    ``n_obs``, ``date_split``, ``every``, ``seed``, ``output_dir``) exactly as
    ``inference.py`` does to load the model and build tasks, plus the optional
    ``config.diagnostics`` section (see `DIAGNOSTICS_DEFAULTS`).

    Raises
    ------
    ValueError
        If the data source is unknown or the model likelihood is not
        ``"bernoulli-gamma"``.

    """
    inference_config = config.inference
    diag_config = dict(DIAGNOSTICS_DEFAULTS)
    if config.get("diagnostics") is not None:
        diag_config.update(
            cast(dict[str, Any], OmegaConf.to_container(config.diagnostics))
        )
    run_ref = inference_config.run_ref
    weights_fn = inference_config.weights_filename
    every = inference_config.every
    splits = list(diag_config["splits"] or [inference_config.date_split])
    seed = diag_config["seed"]
    melt_threshold = diag_config["melt_threshold"]
    n_bins = diag_config["n_regime_bins"]

    run_id = run_ref.split("/")[-1]
    savefolder = os.path.join(
        inference_config.get("output_dir", "outputs"), run_id, "diagnostics"
    )
    figfolder = os.path.join(savefolder, "figures")
    os.makedirs(figfolder, exist_ok=True)

    source = config["data"]["source"]
    target_var = config["data"]["target_var_IDs"]
    context_delta_t = config["dataloader"]["context_delta_t"]

    data_loader: CARRA2DataLoader | HIRHAM5DataLoader | HIRHAM5ZarrDataLoader
    if source == "carra2":
        data_loader = CARRA2DataLoader(config)
        td = np.timedelta64(0, "ms")
    elif source == "hirham5":
        data_loader = HIRHAM5DataLoader(config)
        td = np.timedelta64(12, "h")
    elif source == "hirham5zarr":
        data_loader = HIRHAM5ZarrDataLoader(config)
        td = np.timedelta64(12, "h")
    else:
        raise ValueError(f"Unknown source: {source}")

    # Sample dates per split
    split_dates: dict[str, list[np.datetime64]] = {}
    date_sampling = diag_config["date_sampling"]
    n_per_year = (
        seasonal_samples_per_year(
            diag_config["seasonal_samples_per_year"], config["dataloader"]
        )
        if date_sampling == "seasonal"
        else None
    )
    logger.info(
        "Date sampling: %s (%s)",
        date_sampling,
        f"{n_per_year} dates/year" if n_per_year else f"every {every}",
    )
    seeds = np.random.SeedSequence(seed).spawn(len(splits))
    for split, split_seed in zip(splits, seeds, strict=True):
        dates = candidate_split_dates(
            config["data"][f"{split}_start"],
            config["data"][f"{split}_end"],
            date_sampling,
            td,
            frequency=config["dataloader"]["frequency"],
            every=every,
            samples_per_year=n_per_year,
            seed=seed,
        )
        split_dates[split] = sample_dates(
            dates, diag_config["max_dates"], int(split_seed.generate_state(1)[0])
        )
        logger.info("%s: %d dates sampled", split, len(split_dates[split]))
    all_dates = sorted({d for dates in split_dates.values() for d in dates})

    # Load model
    files_path = locate_run_files(
        run_ref, ["model_config.json", "data_processor_config.json", weights_fn]
    )
    logger.info("Loading run data from: %s", files_path)
    data_processor = DataProcessor(files_path)
    raw_ds, data_processor, task_loader = data_loader.get_dataproc_taskload(
        dates=all_dates, data_processor=data_processor
    )
    lsm = data_loader.load_aux_ds(resolution="bedmachine")
    model = load_wandb_model(
        data_processor,
        task_loader,
        os.path.join(files_path, "model_config.json"),
        os.path.join(files_path, weights_fn),
    )
    likelihood = model.config["likelihood"]
    if likelihood != "bernoulli-gamma":
        raise ValueError(
            f"Diagnostics need a 'bernoulli-gamma' model, got {likelihood!r}"
        )

    if context_delta_t is None:
        delta_factor_context = [1] * len(config["data"]["context_var_IDs"])
    else:
        delta_factor_context = [len(group) for group in context_delta_t]

    # Predict
    tasks = build_tasks(
        task_loader,
        all_dates,
        inference_config.val_set,
        delta_factor_context,
        data_processor,
        lsm,
        inference_config.n_obs,
        inference_config.seed,
    )
    logger.info("Starting prediction of %d tasks", len(tasks))
    # A fresh list: deepsensor's Model.predict mutates pred_params in place.
    pred = model.predict(tasks, X_t=raw_ds, X_t_mask=lsm, pred_params=list(PRED_PARAMS))
    logger.info("Done")

    regime_vars = diag_config["regime_vars"]
    if regime_vars is None:
        context_groups = config["data"]["context_var_IDs"]
        regime_vars = list(context_groups[1]) if len(context_groups) > 1 else []
    regime_vars = [v for v in regime_vars if v in raw_ds.data_vars and v != target_var]

    frame = prediction_frame(
        pred[target_var],
        raw_ds[target_var],
        raw_ds[regime_vars] if regime_vars else None,
        max_points_per_date=diag_config["max_points_per_date"],
        seed=seed,
    )
    date_to_split = {
        pd.Timestamp(d): s for s, dates in split_dates.items() for d in dates
    }
    frame["split"] = frame["time"].map(date_to_split)
    diag = add_pit_columns(
        enrich_prediction_columns(frame, data_processor, target_var, melt_threshold),
        seed=seed,
    )
    logger.info("Diagnosing %d points", len(diag))

    def save_table(df: pd.DataFrame, name: str, index: bool = False) -> None:
        df.to_csv(os.path.join(savefolder, f"{name}.csv"), index=index)
        logger.info("%s:\n%s", name, df.to_string(index=index))

    def save_fig(fig: Figure, name: str) -> None:
        fig.savefig(
            os.path.join(figfolder, f"{name}.png"), dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

    # Summary tables
    summary = summary_table(diag)
    save_table(summary, "summary", index=True)
    save_table(nll_by_target_group(diag), "nll_by_target_group")
    worst_cols = [
        "split",
        "time",
        "lat",
        "lon",
        "y_true",
        "y_true_norm",
        "melt_prob",
        "gamma_mean",
        "gamma_std",
        "gamma_shape",
        "gamma_rate",
        "nll",
    ]
    save_table(
        diag.loc[diag["observed_melt"] == 1].nlargest(100, "nll")[
            [c for c in worst_cols if c in diag.columns]
        ],
        "worst_melt_nll",
    )
    if {"train", "val"}.issubset(summary.index):
        save_table(
            (summary.loc["val"] - summary.loc["train"]).to_frame(
                name="val_minus_train"
            ),
            "overfit_deltas",
            index=True,
        )

    # PIT
    save_fig(
        plot_pit_by_split(
            diag, "pit", "PIT value", "Probability Integral Transform (PIT)"
        ),
        "pit",
    )
    save_fig(
        plot_pit_by_split(
            diag, "gamma_pit", "Conditional Gamma PIT value", "Melt-only Gamma PIT"
        ),
        "melt_gamma_pit",
    )

    # Melt occurrence calibration
    calibration = melt_calibration(diag, diag_config["n_calibration_bins"])
    save_table(calibration, "melt_calibration")
    save_fig(plot_melt_calibration(calibration, diag), "melt_calibration")

    # Mean absolute calibration error
    save_table(
        mace_table(diag, diag_config["mace_num_bins"], diag_config["mace_prop_type"]),
        "mace",
        index=True,
    )

    # Ensemble coverage
    coverage = coverage_table(
        diag,
        diag_config["coverage_levels"],
        n_samples=diag_config["coverage_n_samples"],
        seed=seed,
    )
    save_table(coverage, "coverage")
    save_table(coverage_calibration_error(coverage), "coverage_error", index=True)
    save_fig(plot_coverage(coverage), "coverage")

    # Distribution checks
    save_fig(plot_distributions(diag), "distributions")
    save_fig(plot_nll_by_target_group(diag), "nll_by_target_group")

    # Melt-only Gamma PIT by regime
    melt_regime = diag.loc[diag["observed_melt"] == 1].dropna(subset=["gamma_pit"])
    melt_regime = melt_regime.assign(
        month=melt_regime["time"].dt.month,
        month_group=melt_regime["time"].dt.month.map(lambda m: f"{m:02d}"),
    )
    regime_specs = [
        ("gamma_mean", "predicted_mean_group", "Predicted Gamma mean"),
        ("month", "month_group", "Calendar month"),
        *[(v, f"{v}_group", v) for v in regime_vars],
        ("y_true", "observed_magnitude_group", "Observed melt magnitude (descriptive)"),
    ]
    for source_col, group_col, _ in regime_specs:
        if source_col != "month":
            melt_regime = add_within_split_quantiles(
                melt_regime.dropna(subset=[source_col]), source_col, group_col, q=n_bins
            )
    regime_summary = pd.concat(
        [pit_group_summary(melt_regime, s, g, label) for s, g, label in regime_specs],
        ignore_index=True,
    ).sort_values(["diagnostic", "split", "source_min"])
    save_table(regime_summary, "melt_gamma_pit_by_regime")
    for _, group_col, title in regime_specs:
        save_fig(
            plot_regime_pp(melt_regime, group_col, title),
            f"melt_gamma_pit_pp_{group_col}",
        )

    logger.info("Diagnostics saved to %s", savefolder)


if __name__ == "__main__":
    main()
