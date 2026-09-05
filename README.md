# Chronos Forecasting Pipeline

A validated, reproducible pipeline for training and evaluating pretrained Chronos
time-series models through AutoGluon. It supports Python 3.11 and 3.12 and keeps
large model downloads, GPU execution, and AutoGluon itself outside the offline test
path.

## What it does

- reads multi-series CSV data and normalizes configurable source columns;
- validates timestamps, numeric targets, unique item/timestamp pairs, and history;
- creates a row-aligned holdout horizon for every time series;
- trains a fixed Chronos-Bolt model through `TimeSeriesPredictor`;
- generates forecasts and an AutoGluon leaderboard;
- calculates holdout mean absolute error with exact index matching;
- writes the forecast, leaderboard, metrics manifest, and optional static plot.

The default `bolt_tiny` model is a current counterpart to the original tiny Chronos
experiment. AutoGluon documents Chronos presets and fixed hyperparameters in its
[TimeSeriesPredictor API](https://auto.gluon.ai/1.4.0/api/autogluon.timeseries.TimeSeriesPredictor.fit.html)
and its current [Chronos tutorial](https://auto.gluon.ai/stable/tutorials/timeseries/forecasting-chronos.html).

## Installation

Linux and macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
cp config.example.toml config.toml
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
Copy-Item config.example.toml config.toml
```

The base install omits AutoGluon and Matplotlib so data validation, configuration,
and artifact tooling stay lightweight. Use `.[autogluon]` or `.[plots]` separately
when appropriate.

## Input contract

By default the CSV must contain:

```text
unique_id, ds, y
```

These become AutoGluon's `item_id`, `timestamp`, and `target`. Optional `published`
and `is_holiday` columns are dropped. All names and the delimiter are configurable.
Input data is intentionally ignored by Git.

Each series must contain at least
`(num_val_windows + 2) * prediction_length + 1` rows: one horizon is reserved for
local evaluation, while AutoGluon uses the remaining history for backtesting.
AutoGluon otherwise discards the series during training; this repository reports
every short item before any model is downloaded. The split follows AutoGluon's documented
[`train_test_split` behavior](https://auto.gluon.ai/dev/api/autogluon.timeseries.TimeSeriesDataFrame.train_test_split.html).

## Usage

```bash
chronos-forecast --config config.toml
chronos-forecast --config config.toml --plot
```

Model files are written beneath `output.model_dir`. Forecast, leaderboard, metrics,
and optional plot artifacts are written beneath `output.artifact_dir`. Existing
AutoGluon model directories should be moved or configured to a new path before a
new experiment.

The first real run can download model weights and may take significant time. GPU
hardware is recommended for larger models, while Chronos-Bolt tiny can also run on
CPU. No API key is required.

## Architecture

The package uses frozen configuration objects and a `src/chronos_forecasting`
layout. CSV access is injected, AutoGluon imports are lazy, and the predictor and
time-series frame factories are replaceable. Forecast evaluation rejects partial,
duplicated, reordered, or non-finite results instead of producing misleading
metrics. CLI code only handles arguments and error presentation.

The original hyperparameter search was removed because it mixed a preset with
unvalidated search spaces and did not enforce the history required for six
backtests. Configuration now uses fixed, reproducible model parameters. Advanced
tuning can be added through the backend after validating a bounded search space.

## Development

```bash
python -m pip install -e '.[dev]'
python -m ruff check src tests
python -m pytest --cov=chronos_forecasting --cov-report=term-missing
python -m mypy
python -m compileall -q src tests
```

GitHub Actions runs the same gates on Python 3.11 and 3.12. Tests use synthetic
series and fake predictor factories; they never download Chronos, contact Hugging
Face, train a real model, or require a GPU.

## License and attribution

See `NOTICE` for project lineage and `LICENSE` for the MIT terms. Downloaded model
weights and third-party dependencies retain their own licenses.
