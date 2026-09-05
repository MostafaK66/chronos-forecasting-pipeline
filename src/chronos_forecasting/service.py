"""End-to-end forecasting orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd

from chronos_forecasting.artifacts import calculate_mae, save_artifacts
from chronos_forecasting.backend import AutoGluonBackend
from chronos_forecasting.config import AppConfig, ForecastConfig, ModelConfig
from chronos_forecasting.data import (
    CsvReader,
    load_time_series,
    split_time_series,
    validate_history,
)
from chronos_forecasting.models import ForecastArtifacts, ForecastRun
from chronos_forecasting.plotting import save_forecast_plot


class ForecastBackend(Protocol):
    def fit_predict(
        self,
        train: pd.DataFrame,
        forecast: ForecastConfig,
        model: ModelConfig,
        model_dir: Path,
    ) -> ForecastRun: ...


def run_forecast(
    config: AppConfig,
    *,
    backend: ForecastBackend | None = None,
    reader: CsvReader | None = None,
    plot: bool = False,
) -> ForecastArtifacts:
    """Validate, train, evaluate, and persist one forecasting run."""
    data = load_time_series(config.data, reader=reader or pd.read_csv)
    validate_history(data, config.forecast)
    split = split_time_series(data, config.forecast.prediction_length)
    selected_backend = backend or AutoGluonBackend()
    run = selected_backend.fit_predict(
        split.train,
        config.forecast,
        config.model,
        config.output.model_dir,
    )
    mae = calculate_mae(split.test, run.forecast, config.forecast.prediction_length)
    plot_path = None
    if plot:
        plot_path = save_forecast_plot(
            split.train,
            split.test,
            run.forecast,
            config.output.artifact_dir / config.output.plot_file,
        )
    return save_artifacts(run, config.output, mae, plot=plot_path)
