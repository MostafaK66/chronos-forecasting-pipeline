from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from chronos_forecasting.config import (
    AppConfig,
    DataConfig,
    ForecastConfig,
    ModelConfig,
    OutputConfig,
)
from chronos_forecasting.data import load_time_series


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item, offset in (("a", 0.0), ("b", 10.0)):
        for day in range(7):
            rows.append(
                {
                    "unique_id": item,
                    "ds": f"2026-01-{day + 1:02d}",
                    "y": offset + day,
                    "published": True,
                    "is_holiday": False,
                    "covariate": day * 2,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def series_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    return load_time_series(DataConfig(), reader=lambda *args, **kwargs: raw_frame)


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data=DataConfig(path=tmp_path / "input.csv"),
        forecast=ForecastConfig(
            prediction_length=2,
            num_val_windows=1,
            refit_every_n_windows=1,
            time_limit_seconds=60,
        ),
        model=ModelConfig(batch_size=4),
        output=OutputConfig(
            model_dir=tmp_path / "models" / "chronos",
            artifact_dir=tmp_path / "output",
        ),
    )
