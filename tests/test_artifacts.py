from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chronos_forecasting.artifacts import calculate_mae, save_artifacts
from chronos_forecasting.config import OutputConfig
from chronos_forecasting.errors import ArtifactError
from chronos_forecasting.models import ForecastRun


def forecast_for(series_frame: pd.DataFrame, delta: float = 1.0) -> pd.DataFrame:
    result = (
        series_frame.groupby(level="item_id")
        .tail(2)[["target"]]
        .rename(columns={"target": "mean"})
    )
    result["mean"] += delta
    return result


def run_for(series_frame: pd.DataFrame, tmp_path: Path) -> ForecastRun:
    return ForecastRun(
        forecast_for(series_frame),
        pd.DataFrame({"model": ["Chronos"], "score_val": [-1.0]}),
        "Chronos[bolt_tiny]",
        tmp_path / "models",
    )


def test_calculate_mae(series_frame: pd.DataFrame) -> None:
    assert calculate_mae(series_frame, forecast_for(series_frame), 2) == 1.0


@pytest.mark.parametrize(
    ("data_drop", "forecast_drop"),
    [("target", None), (None, "mean")],
)
def test_calculate_mae_requires_columns(
    series_frame: pd.DataFrame,
    data_drop: str | None,
    forecast_drop: str | None,
) -> None:
    data = series_frame.drop(columns=data_drop) if data_drop else series_frame
    forecast = forecast_for(series_frame)
    if forecast_drop:
        forecast = forecast.drop(columns=forecast_drop)
    with pytest.raises(ArtifactError, match="requires"):
        calculate_mae(data, forecast, 2)


def test_calculate_mae_requires_index(raw_frame: pd.DataFrame) -> None:
    with pytest.raises(ArtifactError, match="index"):
        calculate_mae(
            raw_frame.rename(columns={"y": "target"}),
            pd.DataFrame({"mean": [1.0]}),
            2,
        )


def test_calculate_mae_rejects_duplicate_indices(series_frame: pd.DataFrame) -> None:
    forecast = pd.concat([forecast_for(series_frame), forecast_for(series_frame)])
    with pytest.raises(ArtifactError, match="unique"):
        calculate_mae(series_frame, forecast, 2)


@pytest.mark.parametrize("change", ["short", "reverse"])
def test_calculate_mae_requires_exact_horizon(
    series_frame: pd.DataFrame, change: str
) -> None:
    forecast = forecast_for(series_frame)
    forecast = forecast.iloc[:-1] if change == "short" else forecast.iloc[::-1]
    with pytest.raises(ArtifactError, match="exactly"):
        calculate_mae(series_frame, forecast, 2)


@pytest.mark.parametrize("column", ["target", "mean"])
def test_calculate_mae_rejects_nonfinite_values(
    series_frame: pd.DataFrame, column: str
) -> None:
    data = series_frame.copy()
    forecast = forecast_for(series_frame)
    if column == "target":
        data.iloc[-1, data.columns.get_loc("target")] = np.inf
    else:
        forecast.iloc[-1, forecast.columns.get_loc("mean")] = np.nan
    with pytest.raises(ArtifactError, match="non-finite"):
        calculate_mae(data, forecast, 2)


def test_save_artifacts_writes_files(series_frame: pd.DataFrame, tmp_path: Path) -> None:
    output = OutputConfig(
        model_dir=tmp_path / "models", artifact_dir=tmp_path / "nested" / "output"
    )
    result = save_artifacts(
        run_for(series_frame, tmp_path), output, 1.25, plot=tmp_path / "plot.png"
    )
    assert result.forecast.exists()
    assert result.leaderboard.exists()
    assert result.metrics.exists()
    assert result.plot == tmp_path / "plot.png"
    metrics = json.loads(result.metrics.read_text(encoding="utf-8"))
    assert metrics["best_model"] == "Chronos[bolt_tiny]"
    assert metrics["mean_absolute_error"] == 1.25
    assert pd.read_csv(result.forecast).columns.tolist()[:2] == [
        "item_id",
        "timestamp",
    ]


def test_save_artifacts_wraps_filesystem_failure(
    series_frame: pd.DataFrame, tmp_path: Path
) -> None:
    occupied = tmp_path / "occupied"
    occupied.write_text("file", encoding="utf-8")
    with pytest.raises(ArtifactError, match="cannot write"):
        save_artifacts(
            run_for(series_frame, tmp_path),
            OutputConfig(artifact_dir=occupied),
            1.0,
        )
