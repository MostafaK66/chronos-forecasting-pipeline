from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from chronos_forecasting.cli import build_parser, main
from chronos_forecasting.errors import DataValidationError
from chronos_forecasting.models import ForecastArtifacts


def artifacts(tmp_path: Path, *, plot: bool = False) -> ForecastArtifacts:
    return ForecastArtifacts(
        forecast=tmp_path / "forecast.csv",
        leaderboard=tmp_path / "leaderboard.csv",
        metrics=tmp_path / "metrics.json",
        model_path=tmp_path / "model",
        plot=tmp_path / "plot.png" if plot else None,
        mean_absolute_error=1.25,
    )


def test_build_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.config is None
    assert args.plot is False


def test_main_runs_default_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[tuple[object, bool]] = []

    def run(config: object, *, plot: bool) -> ForecastArtifacts:
        seen.append((config, plot))
        return artifacts(tmp_path)

    monkeypatch.setattr("chronos_forecasting.cli.run_forecast", run)
    assert main([]) == 0
    assert seen[0][1] is False
    output = capsys.readouterr().out
    assert "forecast:" in output
    assert "MAE: 1.250000" in output
    assert "plot:" not in output


def test_main_loads_config_and_prints_plot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[forecast]\nprediction_length = 2\n", encoding="utf-8")

    def run(config: object, *, plot: bool) -> ForecastArtifacts:
        assert config.forecast.prediction_length == 2  # type: ignore[attr-defined]
        assert plot is True
        return artifacts(tmp_path, plot=True)

    monkeypatch.setattr("chronos_forecasting.cli.run_forecast", run)
    assert main(["--config", str(config_path), "--plot"]) == 0
    assert "plot:" in capsys.readouterr().out


def test_main_translates_domain_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(config: object, *, plot: bool) -> ForecastArtifacts:
        del config, plot
        raise DataValidationError("bad history")

    monkeypatch.setattr("chronos_forecasting.cli.run_forecast", broken)
    assert main([]) == 2
    assert "error: bad history" in capsys.readouterr().err


def test_module_entrypoint_returns_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chronos_forecasting.cli.main", lambda: 9)
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("chronos_forecasting.__main__", run_name="__main__")
    assert raised.value.code == 9
