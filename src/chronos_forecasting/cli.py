"""Thin command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from chronos_forecasting.config import AppConfig
from chronos_forecasting.errors import ForecastingError
from chronos_forecasting.service import run_forecast


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chronos-forecast",
        description="Train and evaluate an AutoGluon Chronos forecast",
    )
    parser.add_argument("--config", type=Path, help="TOML configuration path")
    parser.add_argument("--plot", action="store_true", help="save a forecast PNG")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = AppConfig.from_toml(args.config) if args.config else AppConfig()
        artifacts = run_forecast(config, plot=args.plot)
    except ForecastingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"forecast: {artifacts.forecast}")
    print(f"leaderboard: {artifacts.leaderboard}")
    print(f"metrics: {artifacts.metrics}")
    print(f"model: {artifacts.model_path}")
    print(f"MAE: {artifacts.mean_absolute_error:.6f}")
    if artifacts.plot is not None:
        print(f"plot: {artifacts.plot}")
    return 0
