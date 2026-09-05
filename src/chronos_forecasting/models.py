"""Immutable values exchanged between pipeline stages."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class ForecastSplit:
    """Training history and complete evaluation frame."""

    train: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ForecastRun:
    """Backend outputs before filesystem persistence."""

    forecast: pd.DataFrame
    leaderboard: pd.DataFrame
    best_model: str
    model_path: Path


@dataclass(frozen=True, slots=True)
class ForecastArtifacts:
    """Files and model directory produced by one run."""

    forecast: Path
    leaderboard: Path
    metrics: Path
    model_path: Path
    plot: Path | None
    mean_absolute_error: float
