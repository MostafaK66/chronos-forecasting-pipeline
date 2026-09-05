from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos_forecasting.config import DataConfig, ForecastConfig
from chronos_forecasting.data import load_time_series, split_time_series, validate_history
from chronos_forecasting.errors import DataValidationError


def test_load_time_series_normalizes_and_sorts(
    raw_frame: pd.DataFrame, tmp_path: Path
) -> None:
    shuffled = raw_frame.sample(frac=1, random_state=2)
    calls: list[tuple[object, object]] = []

    def reader(path: object, *, sep: object) -> pd.DataFrame:
        calls.append((path, sep))
        return shuffled

    config = DataConfig(path=tmp_path / "data.csv", delimiter=";")
    result = load_time_series(config, reader=reader)
    assert calls == [(config.path, ";")]
    assert result.index.names == ["item_id", "timestamp"]
    assert result.index.is_monotonic_increasing
    assert result["target"].dtype == np.float64
    assert "published" not in result.columns
    assert "covariate" in result.columns


@pytest.mark.parametrize("error", [OSError("denied"), pd.errors.ParserError("bad")])
def test_load_wraps_reader_errors(error: Exception) -> None:
    def reader(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        raise error

    with pytest.raises(DataValidationError, match="cannot read"):
        load_time_series(DataConfig(), reader=reader)


def test_load_rejects_missing_columns(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="y"):
        load_time_series(DataConfig(), reader=lambda *a, **k: raw_frame.drop(columns="y"))


def test_load_rejects_empty_frame(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="no rows"):
        load_time_series(DataConfig(), reader=lambda *a, **k: raw_frame.iloc[:0])


@pytest.mark.parametrize("column", ["unique_id", "ds", "y"])
def test_load_rejects_required_nulls(raw_frame: pd.DataFrame, column: str) -> None:
    frame = raw_frame.copy()
    frame.loc[0, column] = None
    with pytest.raises(DataValidationError, match="must not contain nulls"):
        load_time_series(DataConfig(), reader=lambda *a, **k: frame)


@pytest.mark.parametrize(
    ("column", "value"), [("ds", "not-a-date"), ("y", "not-a-number")]
)
def test_load_rejects_parse_errors(
    raw_frame: pd.DataFrame, column: str, value: object
) -> None:
    frame = raw_frame.copy()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(DataValidationError, match="cannot parse"):
        load_time_series(DataConfig(), reader=lambda *a, **k: frame)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_load_rejects_nonfinite_target(raw_frame: pd.DataFrame, value: float) -> None:
    frame = raw_frame.copy()
    frame.loc[0, "y"] = value
    with pytest.raises(DataValidationError, match="non-finite"):
        load_time_series(DataConfig(), reader=lambda *a, **k: frame)


def test_load_rejects_duplicate_keys(raw_frame: pd.DataFrame) -> None:
    frame = pd.concat([raw_frame, raw_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="unique"):
        load_time_series(DataConfig(), reader=lambda *a, **k: frame)


def test_validate_history_accepts_sufficient_series(series_frame: pd.DataFrame) -> None:
    validate_history(series_frame, ForecastConfig(prediction_length=1, num_val_windows=1))


def test_validate_history_reports_each_short_series(series_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="required 9") as raised:
        validate_history(
            series_frame, ForecastConfig(prediction_length=2, num_val_windows=2)
        )
    assert "a: 7" in str(raised.value)
    assert "b: 7" in str(raised.value)


def test_validate_history_rejects_wrong_index(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="index"):
        validate_history(raw_frame, ForecastConfig())


def test_split_time_series_preserves_full_test(series_frame: pd.DataFrame) -> None:
    split = split_time_series(series_frame, 2)
    assert len(split.train) == 10
    assert len(split.test) == 14
    assert split.train.groupby(level="item_id").size().tolist() == [5, 5]
    assert split.test.equals(series_frame)
    split.test.iloc[0, 0] = 999
    assert series_frame.iloc[0, 0] != 999


@pytest.mark.parametrize("prediction_length", [0, -1])
def test_split_rejects_invalid_prediction_length(
    series_frame: pd.DataFrame, prediction_length: int
) -> None:
    with pytest.raises(DataValidationError, match="positive"):
        split_time_series(series_frame, prediction_length)


def test_split_rejects_wrong_index(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="index"):
        split_time_series(raw_frame, 2)


def test_split_rejects_short_or_empty_series(series_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="longer"):
        split_time_series(series_frame.groupby(level="item_id").head(2), 2)
    empty = series_frame.iloc[:0]
    with pytest.raises(DataValidationError, match="longer"):
        split_time_series(empty, 2)
