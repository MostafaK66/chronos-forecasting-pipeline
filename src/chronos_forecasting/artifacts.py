"""Forecast evaluation and filesystem persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chronos_forecasting.config import OutputConfig
from chronos_forecasting.errors import ArtifactError
from chronos_forecasting.models import ForecastArtifacts, ForecastRun


def calculate_mae(
    complete_data: pd.DataFrame,
    forecast: pd.DataFrame,
    prediction_length: int,
) -> float:
    """Calculate MAE only when every expected holdout timestamp is predicted."""
    if "target" not in complete_data.columns or "mean" not in forecast.columns:
        raise ArtifactError("evaluation requires target and mean columns")
    if list(complete_data.index.names) != ["item_id", "timestamp"]:
        raise ArtifactError("evaluation data must use the time-series index")
    expected = pd.concat(
        [
            group.iloc[-prediction_length:]
            for _, group in complete_data.groupby(level="item_id", sort=False)
        ]
    )
    if not expected.index.is_unique or not forecast.index.is_unique:
        raise ArtifactError("evaluation indices must be unique")
    if len(forecast) != len(expected) or not forecast.index.equals(expected.index):
        raise ArtifactError("forecast index does not exactly match the holdout horizon")
    actual = expected["target"].to_numpy(dtype=np.float64)
    predicted = forecast["mean"].to_numpy(dtype=np.float64)
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ArtifactError("evaluation contains non-finite values")
    return float(np.mean(np.abs(actual - predicted)))


def save_artifacts(
    run: ForecastRun,
    output: OutputConfig,
    mae: float,
    *,
    plot: Path | None = None,
) -> ForecastArtifacts:
    """Persist forecast, leaderboard, and a compact metrics manifest."""
    forecast_path = output.artifact_dir / output.forecast_file
    leaderboard_path = output.artifact_dir / output.leaderboard_file
    metrics_path = output.artifact_dir / output.metrics_file
    metrics = {
        "best_model": run.best_model,
        "mean_absolute_error": mae,
        "model_path": str(run.model_path),
    }
    try:
        output.artifact_dir.mkdir(parents=True, exist_ok=True)
        run.forecast.reset_index().to_csv(forecast_path, index=False)
        run.leaderboard.to_csv(leaderboard_path, index=False)
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError) as exc:
        raise ArtifactError(f"cannot write forecast artifacts: {exc}") from exc
    return ForecastArtifacts(
        forecast=forecast_path,
        leaderboard=leaderboard_path,
        metrics=metrics_path,
        model_path=run.model_path,
        plot=plot,
        mean_absolute_error=mae,
    )
