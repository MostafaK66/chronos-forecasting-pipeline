"""Optional non-interactive forecast visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from chronos_forecasting.errors import ArtifactError, DependencyUnavailableError


def save_forecast_plot(
    train: pd.DataFrame,
    test: pd.DataFrame,
    forecast: pd.DataFrame,
    path: Path,
) -> Path:
    """Save history, holdout, and mean forecast for every item."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DependencyUnavailableError(
            "plotting requires: pip install 'chronos-forecasting-pipeline[plots]'"
        ) from exc
    if "target" not in train.columns or "target" not in test.columns:
        raise ArtifactError("plot data is missing the target column")
    if "mean" not in forecast.columns:
        raise ArtifactError("plot forecast is missing the mean column")
    figure: Any = plt.figure(figsize=(11, 7))
    axis = figure.add_subplot(111)
    for item in train.index.get_level_values("item_id").unique():
        train_item = train.xs(item, level="item_id")
        test_item = test.xs(item, level="item_id")
        forecast_item = forecast.xs(item, level="item_id")
        axis.plot(train_item.index, train_item["target"], label=f"train {item}")
        axis.plot(
            test_item.index[-len(forecast_item) :],
            test_item["target"].iloc[-len(forecast_item) :],
            linestyle="--",
            label=f"actual {item}",
        )
        axis.plot(
            forecast_item.index,
            forecast_item["mean"],
            linestyle=":",
            label=f"forecast {item}",
        )
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Target")
    axis.set_title("Chronos forecast")
    axis.grid(True)
    axis.legend()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
    except OSError as exc:
        raise ArtifactError(f"cannot save forecast plot {path}: {exc}") from exc
    finally:
        plt.close(figure)
    return path
