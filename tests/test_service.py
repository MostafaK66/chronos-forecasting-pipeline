from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chronos_forecasting.config import AppConfig
from chronos_forecasting.models import ForecastRun
from chronos_forecasting.service import run_forecast


class FakeBackend:
    def __init__(self) -> None:
        self.train: pd.DataFrame | None = None
        self.options: tuple[object, object, Path] | None = None

    def fit_predict(
        self,
        train: pd.DataFrame,
        forecast: object,
        model: object,
        model_dir: Path,
    ) -> ForecastRun:
        self.train = train
        self.options = (forecast, model, model_dir)
        complete = self.complete
        prediction = (
            complete.groupby(level="item_id")
            .tail(2)[["target"]]
            .rename(columns={"target": "mean"})
        )
        prediction["mean"] += 0.5
        return ForecastRun(
            prediction,
            pd.DataFrame({"model": ["Chronos"], "score_val": [-0.5]}),
            "Chronos[bolt_tiny]",
            model_dir,
        )

    complete: pd.DataFrame


def reader_for(frame: pd.DataFrame):
    def reader(path: object, *, sep: object) -> pd.DataFrame:
        del path, sep
        return frame

    return reader


def test_run_forecast_end_to_end_with_fake_backend(
    app_config: AppConfig, raw_frame: pd.DataFrame, series_frame: pd.DataFrame
) -> None:
    backend = FakeBackend()
    backend.complete = series_frame
    artifacts = run_forecast(
        app_config,
        backend=backend,
        reader=reader_for(raw_frame),
    )
    assert backend.train is not None and len(backend.train) == 10
    assert backend.options is not None
    assert backend.options[2] == app_config.output.model_dir
    assert artifacts.mean_absolute_error == 0.5
    assert artifacts.forecast.exists()
    assert artifacts.plot is None


def test_run_forecast_can_save_plot(
    app_config: AppConfig,
    raw_frame: pd.DataFrame,
    series_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    backend.complete = series_frame
    calls: list[Path] = []

    def plot(
        train: pd.DataFrame,
        test: pd.DataFrame,
        forecast: pd.DataFrame,
        path: Path,
    ) -> Path:
        del train, test, forecast
        calls.append(path)
        return path

    monkeypatch.setattr("chronos_forecasting.service.save_forecast_plot", plot)
    artifacts = run_forecast(
        app_config,
        backend=backend,
        reader=reader_for(raw_frame),
        plot=True,
    )
    assert calls == [app_config.output.artifact_dir / "forecast.png"]
    assert artifacts.plot == calls[0]


def test_run_forecast_uses_default_backend_and_reader(
    app_config: AppConfig,
    raw_frame: pd.DataFrame,
    series_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend()
    backend.complete = series_frame
    monkeypatch.setattr("chronos_forecasting.service.AutoGluonBackend", lambda: backend)
    monkeypatch.setattr("chronos_forecasting.service.pd.read_csv", reader_for(raw_frame))
    artifacts = run_forecast(app_config)
    assert artifacts.mean_absolute_error == 0.5
