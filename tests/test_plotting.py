from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from chronos_forecasting.errors import ArtifactError, DependencyUnavailableError
from chronos_forecasting.plotting import save_forecast_plot


class FakeAxis:
    def __init__(self) -> None:
        self.lines = 0

    def plot(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.lines += 1

    def set_xlabel(self, value: str) -> None:
        del value

    def set_ylabel(self, value: str) -> None:
        del value

    def set_title(self, value: str) -> None:
        del value

    def grid(self, value: bool) -> None:
        del value

    def legend(self) -> None:
        pass


class FakeFigure:
    def __init__(self, *, error: bool = False) -> None:
        self.axis = FakeAxis()
        self.error = error

    def add_subplot(self, value: int) -> FakeAxis:
        del value
        return self.axis

    def tight_layout(self) -> None:
        pass

    def savefig(self, path: Path) -> None:
        if self.error:
            raise OSError("disk full")
        path.write_text("plot", encoding="utf-8")


def install_matplotlib(
    monkeypatch: pytest.MonkeyPatch, *, error: bool = False
) -> FakeFigure:
    figure = FakeFigure(error=error)
    package = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = lambda **kwargs: figure  # type: ignore[attr-defined]
    pyplot.close = lambda value: None  # type: ignore[attr-defined]
    package.pyplot = pyplot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", package)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return figure


def forecast_for(series_frame: pd.DataFrame) -> pd.DataFrame:
    return (
        series_frame.groupby(level="item_id")
        .tail(2)[["target"]]
        .rename(columns={"target": "mean"})
    )


def test_save_forecast_plot(
    series_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figure = install_matplotlib(monkeypatch)
    train = series_frame.groupby(level="item_id").head(4)
    result = save_forecast_plot(
        train, series_frame, forecast_for(series_frame), tmp_path / "nested" / "plot.png"
    )
    assert result.exists()
    assert figure.axis.lines == 6


@pytest.mark.parametrize("missing", ["train", "test", "forecast"])
def test_plot_rejects_missing_columns(
    series_frame: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    install_matplotlib(monkeypatch)
    train = series_frame.groupby(level="item_id").head(4)
    test = series_frame
    forecast = forecast_for(series_frame)
    if missing == "train":
        train = train.drop(columns="target")
    elif missing == "test":
        test = test.drop(columns="target")
    else:
        forecast = forecast.drop(columns="mean")
    with pytest.raises(ArtifactError, match="missing"):
        save_forecast_plot(train, test, forecast, tmp_path / "plot.png")


def test_plot_wraps_write_error(
    series_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_matplotlib(monkeypatch, error=True)
    with pytest.raises(ArtifactError, match="disk full"):
        save_forecast_plot(
            series_frame.groupby(level="item_id").head(4),
            series_frame,
            forecast_for(series_frame),
            tmp_path / "plot.png",
        )


def test_plot_explains_missing_dependency(
    series_frame: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("matplotlib"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DependencyUnavailableError, match="plotting"):
        save_forecast_plot(
            series_frame,
            series_frame,
            forecast_for(series_frame),
            tmp_path / "plot.png",
        )
