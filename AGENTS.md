# Engineering contract

- Support Python 3.11 and 3.12 with the `src/` package layout.
- Keep configuration immutable and validate all external data before training.
- Keep AutoGluon, model downloads, GPU use, plotting, and filesystem access isolated.
- Unit tests must be deterministic and require no network, GPU, model, or private data.
- Preserve item/timestamp alignment across splitting, forecasting, and evaluation.
- Run Ruff, strict mypy, branch-aware pytest coverage, and compile checks before merging.
- Never commit datasets, generated forecasts, model checkpoints, caches, or credentials.
