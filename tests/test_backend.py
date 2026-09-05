from __future__ import annotations

import builtins
from pathlib import Path

import pandas as pd
import pytest

from chronos_forecasting.backend import AutoGluonBackend
from chronos_forecasting.config import ForecastConfig, ModelConfig
from chronos_forecasting.errors import BackendError, DependencyUnavailableError


class FakePredictor:
    def __init__(
        self,
        forecast: object,
        leaderboard: object,
        *,
        best_model: object = "Chronos[bolt_tiny]",
        error: Exception | None = None,
    ) -> None:
        self.forecast = forecast
        self.board = leaderboard
        self.model_best = best_model
        self.error = error
        self.fit_options: dict[str, object] = {}
        self.train_data: object = None

    def fit(self, train_data: object, **kwargs: object) -> FakePredictor:
        if self.error:
            raise self.error
        self.train_data = train_data
        self.fit_options = kwargs
        return self

    def predict(self, data: object) -> object:
        assert data is self.train_data
        return self.forecast

    def leaderboard(self, *, display: bool) -> object:
        assert display is False
        return self.board


def valid_frames(series_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecast = (
        series_frame.groupby(level="item_id")
        .tail(2)[["target"]]
        .rename(columns={"target": "mean"})
    )
    board = pd.DataFrame({"model": ["Chronos"], "score_val": [-1.0]})
    return forecast, board


def test_backend_forwards_validated_options(
    series_frame: pd.DataFrame, tmp_path: Path
) -> None:
    forecast, board = valid_frames(series_frame)
    predictor = FakePredictor(forecast, board)
    constructor_options: list[dict[str, object]] = []

    def factory(**kwargs: object) -> FakePredictor:
        constructor_options.append(kwargs)
        return predictor

    frame_calls: list[pd.DataFrame] = []

    def frame_factory(frame: pd.DataFrame) -> object:
        frame_calls.append(frame)
        return object()

    forecast_config = ForecastConfig(
        prediction_length=2,
        num_val_windows=1,
        refit_every_n_windows=1,
        time_limit_seconds=30,
    )
    model = ModelConfig(preset="bolt_tiny", batch_size=4, device="cpu")
    result = AutoGluonBackend(
        predictor_factory=factory, frame_factory=frame_factory
    ).fit_predict(series_frame, forecast_config, model, tmp_path / "model")
    assert constructor_options == [
        {
            "prediction_length": 2,
            "target": "target",
            "eval_metric": "WQL",
            "path": str(tmp_path / "model"),
        }
    ]
    assert frame_calls == [series_frame]
    assert predictor.fit_options["presets"] == "bolt_tiny"
    assert predictor.fit_options["time_limit"] == 30
    assert predictor.fit_options["hyperparameters"] == model.hyperparameters()
    assert result.best_model == "Chronos[bolt_tiny]"
    assert result.forecast is not forecast
    assert result.model_path == tmp_path / "model"


def test_backend_omits_optional_fit_options(
    series_frame: pd.DataFrame, tmp_path: Path
) -> None:
    forecast, board = valid_frames(series_frame)
    predictor = FakePredictor(forecast, board)
    backend = AutoGluonBackend(
        predictor_factory=lambda **kwargs: predictor,
        frame_factory=lambda frame: frame,
    )
    backend.fit_predict(
        series_frame,
        ForecastConfig(prediction_length=2),
        ModelConfig(),
        tmp_path / "model",
    )
    assert "presets" not in predictor.fit_options
    assert "time_limit" not in predictor.fit_options


def test_backend_wraps_predictor_failure(
    series_frame: pd.DataFrame, tmp_path: Path
) -> None:
    forecast, board = valid_frames(series_frame)
    backend = AutoGluonBackend(
        predictor_factory=lambda **kwargs: FakePredictor(
            forecast, board, error=RuntimeError("training failed")
        ),
        frame_factory=lambda frame: frame,
    )
    with pytest.raises(BackendError, match="training failed"):
        backend.fit_predict(
            series_frame,
            ForecastConfig(prediction_length=2),
            ModelConfig(),
            tmp_path / "model",
        )


@pytest.mark.parametrize(
    ("forecast", "board", "best", "message"),
    [
        (None, pd.DataFrame({"model": ["x"]}), "x", "forecast"),
        (pd.DataFrame({"mean": [1]}), pd.DataFrame(), "x", "leaderboard"),
        (pd.DataFrame({"mean": [1]}), pd.DataFrame({"model": ["x"]}), "", "best"),
        (pd.DataFrame({"mean": [1]}), pd.DataFrame({"model": ["x"]}), 3, "best"),
    ],
)
def test_backend_rejects_malformed_outputs(
    series_frame: pd.DataFrame,
    tmp_path: Path,
    forecast: object,
    board: object,
    best: object,
    message: str,
) -> None:
    backend = AutoGluonBackend(
        predictor_factory=lambda **kwargs: FakePredictor(
            forecast, board, best_model=best
        ),
        frame_factory=lambda frame: frame,
    )
    with pytest.raises(BackendError, match=message):
        backend.fit_predict(
            series_frame,
            ForecastConfig(prediction_length=2),
            ModelConfig(),
            tmp_path / "model",
        )


def test_backend_explains_missing_autogluon(
    series_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("autogluon"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DependencyUnavailableError, match="autogluon"):
        AutoGluonBackend().fit_predict(
            series_frame,
            ForecastConfig(prediction_length=2),
            ModelConfig(),
            tmp_path / "model",
        )
