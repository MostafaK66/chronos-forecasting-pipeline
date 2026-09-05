from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from chronos_forecasting.config import (
    AppConfig,
    DataConfig,
    ForecastConfig,
    ModelConfig,
    OutputConfig,
)
from chronos_forecasting.errors import ConfigurationError


def test_default_configuration_and_immutability() -> None:
    config = AppConfig()
    assert config.model.model_path == "bolt_tiny"
    assert config.forecast.minimum_series_length == 801
    with pytest.raises(FrozenInstanceError):
        config.model.batch_size = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delimiter": ""},
        {"item_column": ""},
        {"item_column": "same", "timestamp_column": "same"},
    ],
)
def test_data_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        DataConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prediction_length": 0}, "prediction_length"),
        ({"num_val_windows": 0}, "windows"),
        ({"refit_every_n_windows": 0}, "windows"),
        ({"eval_metric": " "}, "eval_metric"),
        ({"random_seed": -1}, "random_seed"),
        ({"time_limit_seconds": 0}, "time_limit"),
    ],
)
def test_forecast_config_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        ForecastConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model_type": ""},
        {"model_path": " "},
        {"batch_size": 0},
        {"device": ""},
        {"preset": " "},
    ],
)
def test_model_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        ModelConfig(**kwargs)  # type: ignore[arg-type]


def test_model_hyperparameters_are_fresh() -> None:
    config = ModelConfig(model_type="Chronos2", model_path="local", batch_size=8)
    first = config.hyperparameters()
    first["Chronos2"]["batch_size"] = 99
    assert config.hyperparameters()["Chronos2"]["batch_size"] == 8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"forecast_file": ""},
        {"leaderboard_file": "nested/file.csv"},
        {"metrics_file": "../metrics.json"},
        {"plot_file": "plots/view.png"},
    ],
)
def test_output_config_rejects_unsafe_names(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError, match="simple filename"):
        OutputConfig(**kwargs)  # type: ignore[arg-type]


def test_load_complete_toml_resolves_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[data]
path = "source.csv"
delimiter = ";"
item_column = "series"
timestamp_column = "date"
target_column = "value"
drop_columns = ["ignore"]
[forecast]
prediction_length = 4
num_val_windows = 2
refit_every_n_windows = 1
random_seed = 9
eval_metric = "MASE"
time_limit_seconds = 30
[model]
model_type = "Chronos2"
model_path = "local/model"
batch_size = 8
device = "cpu"
preset = "fast"
[output]
model_dir = "saved/model"
artifact_dir = "artifacts"
forecast_file = "values.csv"
leaderboard_file = "board.csv"
metrics_file = "score.json"
plot_file = "view.png"
""",
        encoding="utf-8",
    )
    config = AppConfig.from_toml(path)
    assert config.data.path == tmp_path / "source.csv"
    assert config.data.drop_columns == ("ignore",)
    assert config.forecast.minimum_series_length == 17
    assert config.forecast.time_limit_seconds == 30
    assert config.model.model_type == "Chronos2"
    assert config.model.preset == "fast"
    assert config.output.model_dir == tmp_path / "saved/model"


def test_load_minimal_toml_and_absolute_paths(tmp_path: Path) -> None:
    absolute = tmp_path / "input.csv"
    path = tmp_path / "config.toml"
    path.write_text(f'[data]\npath = "{absolute}"\n', encoding="utf-8")
    config = AppConfig.from_toml(path)
    assert config.data.path == absolute
    assert config.model.preset is None
    assert config.forecast.time_limit_seconds is None


@pytest.mark.parametrize(
    "content",
    [
        "invalid = [",
        "data = 1",
        "[data]\ndrop_columns = 2",
        "[forecast]\ntime_limit_seconds = true",
        '[forecast]\nprediction_length = "100"',
        "[forecast]\nrandom_seed = true",
        "[data]\npath = 1",
        "[model]\nbatch_size = 3.5",
        "[output]\nforecast_file = 1",
        "[model]\npreset = 3",
    ],
)
def test_load_rejects_invalid_toml(tmp_path: Path, content: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        AppConfig.from_toml(path)


def test_load_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        AppConfig.from_toml(tmp_path / "missing.toml")
