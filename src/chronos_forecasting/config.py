"""Immutable TOML configuration with early validation."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chronos_forecasting.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class DataConfig:
    path: Path = Path("input/medium_views_published_holidays.csv")
    delimiter: str = ","
    item_column: str = "unique_id"
    timestamp_column: str = "ds"
    target_column: str = "y"
    drop_columns: tuple[str, ...] = ("published", "is_holiday")

    def __post_init__(self) -> None:
        names = (self.item_column, self.timestamp_column, self.target_column)
        if not self.delimiter:
            raise ConfigurationError("data.delimiter must not be empty")
        if any(not name.strip() for name in names) or len(set(names)) != 3:
            raise ConfigurationError("data column names must be non-empty and unique")


@dataclass(frozen=True, slots=True)
class ForecastConfig:
    prediction_length: int = 100
    num_val_windows: int = 6
    refit_every_n_windows: int = 2
    random_seed: int = 123
    eval_metric: str = "WQL"
    time_limit_seconds: int | None = None

    def __post_init__(self) -> None:
        if self.prediction_length < 1:
            raise ConfigurationError("forecast.prediction_length must be positive")
        if self.num_val_windows < 1 or self.refit_every_n_windows < 1:
            raise ConfigurationError("validation and refit windows must be positive")
        if not self.eval_metric.strip():
            raise ConfigurationError("forecast.eval_metric must not be empty")
        if self.random_seed < 0:
            raise ConfigurationError("forecast.random_seed must not be negative")
        if self.time_limit_seconds is not None and self.time_limit_seconds < 1:
            raise ConfigurationError("forecast.time_limit_seconds must be positive")

    @property
    def minimum_series_length(self) -> int:
        """Smallest full series that leaves enough training data for backtests."""
        return (self.num_val_windows + 2) * self.prediction_length + 1


@dataclass(frozen=True, slots=True)
class ModelConfig:
    model_type: str = "Chronos"
    model_path: str = "bolt_tiny"
    batch_size: int = 32
    device: str = "auto"
    preset: str | None = None

    def __post_init__(self) -> None:
        if not self.model_type.strip() or not self.model_path.strip():
            raise ConfigurationError("model type and path must not be empty")
        if self.batch_size < 1 or not self.device.strip():
            raise ConfigurationError("model batch_size and device must be valid")
        if self.preset is not None and not self.preset.strip():
            raise ConfigurationError("model.preset must not be empty")

    def hyperparameters(self) -> dict[str, dict[str, object]]:
        """Build a fresh backend dictionary without mutable configuration state."""
        return {
            self.model_type: {
                "model_path": self.model_path,
                "batch_size": self.batch_size,
                "device": self.device,
            }
        }


@dataclass(frozen=True, slots=True)
class OutputConfig:
    model_dir: Path = Path("models/chronos")
    artifact_dir: Path = Path("output")
    forecast_file: str = "forecast.csv"
    leaderboard_file: str = "leaderboard.csv"
    metrics_file: str = "metrics.json"
    plot_file: str = "forecast.png"

    def __post_init__(self) -> None:
        for name, value in (
            ("forecast_file", self.forecast_file),
            ("leaderboard_file", self.leaderboard_file),
            ("metrics_file", self.metrics_file),
            ("plot_file", self.plot_file),
        ):
            if not value or Path(value).name != value:
                raise ConfigurationError(f"output.{name} must be a simple filename")


@dataclass(frozen=True, slots=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_toml(cls, path: Path) -> AppConfig:
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
        base = path.resolve().parent
        data = _table(raw, "data")
        forecast = _table(raw, "forecast")
        model = _table(raw, "model")
        output = _table(raw, "output")
        return cls(
            data=DataConfig(
                path=_resolve(
                    base,
                    Path(
                        _string(
                            data.get("path", "input/medium_views_published_holidays.csv"),
                            "data.path",
                        )
                    ),
                ),
                delimiter=_string(data.get("delimiter", ","), "data.delimiter"),
                item_column=_string(
                    data.get("item_column", "unique_id"), "data.item_column"
                ),
                timestamp_column=_string(
                    data.get("timestamp_column", "ds"), "data.timestamp_column"
                ),
                target_column=_string(
                    data.get("target_column", "y"), "data.target_column"
                ),
                drop_columns=_strings(
                    data.get("drop_columns", ("published", "is_holiday")),
                    "data.drop_columns",
                ),
            ),
            forecast=ForecastConfig(
                prediction_length=_integer(
                    forecast.get("prediction_length", 100),
                    "forecast.prediction_length",
                ),
                num_val_windows=_integer(
                    forecast.get("num_val_windows", 6), "forecast.num_val_windows"
                ),
                refit_every_n_windows=_integer(
                    forecast.get("refit_every_n_windows", 2),
                    "forecast.refit_every_n_windows",
                ),
                random_seed=_integer(
                    forecast.get("random_seed", 123), "forecast.random_seed"
                ),
                eval_metric=_string(
                    forecast.get("eval_metric", "WQL"), "forecast.eval_metric"
                ),
                time_limit_seconds=_optional_int(forecast.get("time_limit_seconds")),
            ),
            model=ModelConfig(
                model_type=_string(model.get("model_type", "Chronos"), "model.type"),
                model_path=_string(
                    model.get("model_path", "bolt_tiny"), "model.model_path"
                ),
                batch_size=_integer(model.get("batch_size", 32), "model.batch_size"),
                device=_string(model.get("device", "auto"), "model.device"),
                preset=_optional_string(model.get("preset")),
            ),
            output=OutputConfig(
                model_dir=_resolve(
                    base,
                    Path(
                        _string(
                            output.get("model_dir", "models/chronos"),
                            "output.model_dir",
                        )
                    ),
                ),
                artifact_dir=_resolve(
                    base,
                    Path(
                        _string(
                            output.get("artifact_dir", "output"),
                            "output.artifact_dir",
                        )
                    ),
                ),
                forecast_file=_string(
                    output.get("forecast_file", "forecast.csv"),
                    "output.forecast_file",
                ),
                leaderboard_file=_string(
                    output.get("leaderboard_file", "leaderboard.csv"),
                    "output.leaderboard_file",
                ),
                metrics_file=_string(
                    output.get("metrics_file", "metrics.json"), "output.metrics_file"
                ),
                plot_file=_string(
                    output.get("plot_file", "forecast.png"), "output.plot_file"
                ),
            ),
        )


def _table(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a TOML table")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) for item in value
    ):
        raise ConfigurationError(f"{name} must be an array of strings")
    return tuple(value)


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be a string")
    return value


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError(f"{name} must be an integer")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigurationError("forecast.time_limit_seconds must be an integer")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ConfigurationError("model.preset must be a string")
    return value


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else base / value
