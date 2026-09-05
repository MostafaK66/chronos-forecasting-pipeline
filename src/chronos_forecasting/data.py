"""CSV validation and row-aligned time-series splitting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from chronos_forecasting.config import DataConfig, ForecastConfig
from chronos_forecasting.errors import DataValidationError
from chronos_forecasting.models import ForecastSplit

CsvReader = Callable[..., pd.DataFrame]


def load_time_series(
    config: DataConfig, *, reader: CsvReader = pd.read_csv
) -> pd.DataFrame:
    """Read and normalize source columns into AutoGluon's index convention."""
    try:
        raw = reader(config.path, sep=config.delimiter)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataValidationError(f"cannot read input data {config.path}: {exc}") from exc
    required = (config.item_column, config.timestamp_column, config.target_column)
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise DataValidationError(f"input data is missing columns: {', '.join(missing)}")
    if raw.empty:
        raise DataValidationError("input data contains no rows")
    frame = raw.drop(columns=list(config.drop_columns), errors="ignore").copy()
    frame = frame.rename(
        columns={
            config.item_column: "item_id",
            config.timestamp_column: "timestamp",
            config.target_column: "target",
        }
    )
    if frame[["item_id", "timestamp", "target"]].isnull().any().any():
        raise DataValidationError("item_id, timestamp, and target must not contain nulls")
    try:
        frame["timestamp"] = pd.to_datetime(
            frame["timestamp"], errors="raise", format="mixed"
        )
        frame["target"] = pd.to_numeric(frame["target"], errors="raise").astype("float64")
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"cannot parse timestamp or target: {exc}") from exc
    if not np.isfinite(frame["target"].to_numpy()).all():
        raise DataValidationError("target contains non-finite values")
    if frame.duplicated(subset=["item_id", "timestamp"]).any():
        raise DataValidationError("item_id and timestamp pairs must be unique")
    return frame.set_index(["item_id", "timestamp"]).sort_index()


def validate_history(frame: pd.DataFrame, config: ForecastConfig) -> None:
    """Ensure every series survives AutoGluon's backtest window requirement."""
    if list(frame.index.names) != ["item_id", "timestamp"]:
        raise DataValidationError("time-series index must be [item_id, timestamp]")
    counts = frame.groupby(level="item_id", sort=False).size()
    short = counts[counts < config.minimum_series_length]
    if not short.empty:
        details = ", ".join(f"{item}: {count}" for item, count in short.items())
        raise DataValidationError(
            f"series shorter than required {config.minimum_series_length}: {details}"
        )


def split_time_series(frame: pd.DataFrame, prediction_length: int) -> ForecastSplit:
    """Mirror AutoGluon splitting while retaining full test history."""
    if prediction_length < 1:
        raise DataValidationError("prediction_length must be positive")
    if list(frame.index.names) != ["item_id", "timestamp"]:
        raise DataValidationError("time-series index must be [item_id, timestamp]")
    counts = frame.groupby(level="item_id", sort=False).size()
    if counts.empty or (counts <= prediction_length).any():
        raise DataValidationError(
            "every series must be longer than the prediction length"
        )
    train_parts = [
        group.iloc[:-prediction_length]
        for _, group in frame.groupby(level="item_id", sort=False)
    ]
    train = pd.concat(train_parts).sort_index()
    return ForecastSplit(train=train, test=frame.copy())
