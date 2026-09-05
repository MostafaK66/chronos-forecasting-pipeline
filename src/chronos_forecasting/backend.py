"""AutoGluon boundary isolated from the offline application core."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

import pandas as pd

from chronos_forecasting.config import ForecastConfig, ModelConfig
from chronos_forecasting.errors import BackendError, DependencyUnavailableError
from chronos_forecasting.models import ForecastRun


class Predictor(Protocol):
    model_best: str

    def fit(self, train_data: object, **kwargs: object) -> object: ...

    def predict(self, data: object) -> object: ...

    def leaderboard(self, *, display: bool) -> object: ...


PredictorFactory = Callable[..., Predictor]
FrameFactory = Callable[[pd.DataFrame], object]


class AutoGluonBackend:
    """Fit and invoke AutoGluon without importing it during local-only work."""

    def __init__(
        self,
        *,
        predictor_factory: PredictorFactory | None = None,
        frame_factory: FrameFactory | None = None,
    ) -> None:
        self._predictor_factory = predictor_factory
        self._frame_factory = frame_factory

    def fit_predict(
        self,
        train: pd.DataFrame,
        forecast: ForecastConfig,
        model: ModelConfig,
        model_dir: Path,
    ) -> ForecastRun:
        """Train, forecast, and return a validated leaderboard."""
        predictor_factory, frame_factory = self._factories()
        try:
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            train_data = frame_factory(train)
            predictor = predictor_factory(
                prediction_length=forecast.prediction_length,
                target="target",
                eval_metric=forecast.eval_metric,
                path=str(model_dir),
            )
            fit_options: dict[str, object] = {
                "hyperparameters": model.hyperparameters(),
                "num_val_windows": forecast.num_val_windows,
                "refit_every_n_windows": forecast.refit_every_n_windows,
                "refit_full": False,
                "skip_model_selection": False,
                "random_seed": forecast.random_seed,
            }
            if model.preset is not None:
                fit_options["presets"] = model.preset
            if forecast.time_limit_seconds is not None:
                fit_options["time_limit"] = forecast.time_limit_seconds
            fitted = cast(Predictor, predictor.fit(train_data, **fit_options))
            prediction = fitted.predict(train_data)
            leaderboard = fitted.leaderboard(display=False)
        except Exception as exc:
            raise BackendError(f"AutoGluon forecasting failed: {exc}") from exc
        forecast_frame = _dataframe(prediction, "forecast")
        leaderboard_frame = _dataframe(leaderboard, "leaderboard")
        best_model = fitted.model_best
        if not isinstance(best_model, str) or not best_model:
            raise BackendError("AutoGluon did not identify a best model")
        return ForecastRun(
            forecast=forecast_frame,
            leaderboard=leaderboard_frame,
            best_model=best_model,
            model_path=model_dir,
        )

    def _factories(self) -> tuple[PredictorFactory, FrameFactory]:
        if self._predictor_factory is not None and self._frame_factory is not None:
            return self._predictor_factory, self._frame_factory
        try:
            from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        except ImportError as exc:
            raise DependencyUnavailableError(
                "forecasting requires: pip install "
                "'chronos-forecasting-pipeline[autogluon]'"
            ) from exc
        return (
            self._predictor_factory or cast(PredictorFactory, TimeSeriesPredictor),
            self._frame_factory or cast(FrameFactory, TimeSeriesDataFrame),
        )


def _dataframe(value: object, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame) or value.empty:
        raise BackendError(f"AutoGluon returned an empty or invalid {name}")
    return value.copy()
