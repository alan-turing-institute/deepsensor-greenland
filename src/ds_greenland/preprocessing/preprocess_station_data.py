"""Filter station locations by installation and last-valid dates.

Reads a station-locations CSV (e.g. ``promice_locations.csv``) and keeps only the
stations whose ``date_installation`` and ``date_last_valid`` fall within
user-defined date ranges.
"""

import logging
import os

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

INSTALLATION_COLUMN = "date_installation"
LAST_VALID_COLUMN = "date_last_valid"


def filter_by_date_range(
    df: pd.DataFrame,
    column: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Keep rows whose ``column`` date lies within ``[start_date, end_date]``.

    Parameters
    ----------
    df : pd.DataFrame
        Station table with a ``column`` of dates (strings or datetimes).
    column : str
        Name of the date column to filter on.
    start_date : str or None, optional
        Earliest date to keep (inclusive). ``None`` for no lower bound.
    end_date : str or None, optional
        Latest date to keep (inclusive). ``None`` for no upper bound.

    Returns
    -------
    pd.DataFrame
        The filtered rows, in their original order. If both bounds are ``None``,
        ``df`` is returned unchanged; otherwise rows with a missing or
        unparseable date are dropped.

    Raises
    ------
    KeyError
        If ``column`` is not in ``df``.
    ValueError
        If both bounds are given and ``start_date`` is after ``end_date``.

    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in station table")
    if start_date is None and end_date is None:
        return df

    start = pd.Timestamp(start_date) if start_date is not None else None
    end = pd.Timestamp(end_date) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError(
            f"{column}: start_date ({start_date}) is after end_date ({end_date})"
        )

    dates = pd.to_datetime(df[column], errors="coerce")
    mask = dates.notna()
    if start is not None:
        mask &= dates >= start
    if end is not None:
        mask &= dates <= end
    return df.loc[mask]


def preprocess_station_data(params: DictConfig) -> pd.DataFrame:
    """Filter ``params.input_file`` by date and write ``params.output_file``.

    Parameters
    ----------
    params : DictConfig
        The ``preprocessing`` section of the composed config, with ``input_file``,
        ``output_file``, and ``installation`` / ``last_valid`` sections, each with
        ``start_date`` and ``end_date`` (either may be null), applied to the
        ``date_installation`` / ``date_last_valid`` columns respectively.

    Returns
    -------
    pd.DataFrame
        The filtered station table that was written.

    """
    df = pd.read_csv(params.input_file)
    filtered = df
    for column, bounds in (
        (INSTALLATION_COLUMN, params.installation),
        (LAST_VALID_COLUMN, params.last_valid),
    ):
        n_before = len(filtered)
        filtered = filter_by_date_range(
            filtered, column, bounds.start_date, bounds.end_date
        )
        logger.info(
            "%s between %s and %s: kept %d of %d stations",
            column,
            bounds.start_date or "-inf",
            bounds.end_date or "+inf",
            len(filtered),
            n_before,
        )

    out_dir = os.path.dirname(params.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    filtered.to_csv(params.output_file, index=False)
    logger.info(
        "Saved %d of %d stations to %s", len(filtered), len(df), params.output_file
    )
    return filtered


@hydra.main(
    version_base=None, config_path="../configs", config_name="preprocess_station_data"
)
def main(config: DictConfig) -> None:
    """Run the station-data preprocessing."""
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))
    preprocess_station_data(config.preprocessing)


if __name__ == "__main__":
    main()
