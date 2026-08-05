# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`giant-sp` — sensor placement to aid location planning in GIANT, built on top of
[DeepSensor](https://github.com/alan-turing-institute/deepsensor) (ConvNP models) and PyTorch.
It trains spatial-interpolation models on gridded climate reanalysis/RCM data (CARRA2, HIRHAM5)
plus in-situ station observations, then uses the trained model to run active-learning searches
for new sensor placements and to run inference/export.

## Commands

Dependency management uses [uv](https://docs.astral.sh/uv/); the project is Python 3.11+.

```bash
uv sync                              # install package + dev/test/lint groups, editable
uv run pytest                        # run tests
uv run pytest --cov=giant_sp         # run tests with coverage
uv run pytest tests/path/to/test.py::[TestClass::]test_name   # run a single test
uv run mypy                          # type check src/giant_sp and tests (see [tool.mypy] in pyproject.toml)
uv run pre-commit run --all-files    # ruff lint + format + generic hooks (same as CI)
uv run ruff check src --fix          # lint only
uv run ruff format src               # format only
uv build                             # verify the package builds
```

Adding dependencies: `uv add <package>` (runtime), `uv add --group dev|test|lint <package>`.
`uv.lock` is committed; run `uv lock` after changing dependencies (CI checks it's up to date via
`uv sync --locked`).

CI (`.github/workflows/ci.yml`) runs three jobs on every push/PR: `pre-commit run --all-files`,
`mypy`, and `pytest` across Python 3.11/3.12/3.13 (coverage uploaded from the 3.13 run). Tests live
under `tests/`, mirroring the `src/giant_sp/` layout (`tests/data/`, `tests/models/`,
`tests/utils/`) — new code should come with tests there.

Running the actual training/inference/placement scripts requires a YAML config from
`src/giant_sp/configs/` (e.g. `uv run python -m giant_sp.scripts.interpolation --config
src/giant_sp/configs/carra2_debug.yaml`) and access to the data referenced in that config (see
`data.data_path`, `data.station_locations_file`, etc.) — these are not part of the repo and are
not available in most dev environments.

## Architecture

The package (`src/giant_sp/`) is organized as a pipeline: **data loading → DeepSensor
model/training → scripts that drive training, inference, and sensor placement.**

### Data loading (`data/dataloader.py`)

`BaseDataLoader` and its subclasses `CARRA2DataLoader`, `HIRHAM5DataLoader`,
`HIRHAM5ZarrDataLoader` (one per data source) are the entry point for every script. Each subclass
implements `load_ds()` (raw gridded dataset), `load_aux_ds()` (auxiliary/land-sea-mask data,
e.g. BedMachine at a given resolution), `load_basin()`, and `get_dataproc_taskload()`, which wires
a loaded dataset into a DeepSensor `DataProcessor` (normalization) and `TaskLoader` (context/target
sampling) — this triple (`dataset, data_processor, task_loader`) is what every downstream script
consumes. Which subclass to instantiate is selected at runtime from `config["data"]["source"]`
(`"carra2"`, `"hirham5"`, `"hirham5zarr"`) — this three-way branch is duplicated verbatim across
`scripts/interpolation.py`, `scripts/inference.py`, and `scripts/placement.py`.

### Model (`models/model.py`)

`DeepSensorModel` wraps a DeepSensor `ConvNP`, built from a `DataProcessor` + `TaskLoader` and a
`config["model"]` dict (architecture: `unet_channels`, `unet_kernels`, `likelihood`,
`num_basis_functions`, ...). It only handles model construction/parameter counting; actual
training and prediction logic live elsewhere (`training/trainer.py`, `utils/model.py`) and largely
bypass this wrapper by calling the underlying `ConvNP` directly.

### Training (`training/trainer.py`, `utils/training.py`)

`Trainer` owns the training loop: optimizer creation, W&B run setup/resume, the `train()` loop
(loss computation, checkpointing to `local_save_dir`, LR scheduling, early stopping, periodic
validation against both the gridded truth and held-out stations, and W&B logging of maps/charts).
`utils/training.py` provides the `TaskStreamDataset` / `GapFillTaskStreamDataset`
(`IterableDataset[Task]`) that turn a date range + sampling strategy (`train_sampling`:
`"default"`, `"stn_obs"`, `"land_obs"`, `"mixed_obs"`) into a stream of DeepSensor `Task`s, plus
`task_collate_fn` for batching them.

### Sampling & geospatial helpers (`utils/data.py`)

Pure functions used across data loading, training, and scripts: sampling context-set locations
(`sample_land_locations`, `sample_station_locations`, `mixed_sampling_strategy`), time-covariate
construction (`construct_circ_and_linear_time_ds`), and geospatial matching (`find_lon_lat`,
`find_xy_for_points`, `points_in_polygons`, `load_greenland_shape`). Several of these (marked with
a `# source:` comment) are ported from
[eschlager/MeltEmulation](https://github.com/eschlager/MeltEmulation).

### Scripts (`scripts/`) — the runnable entry points

- **`interpolation.py`**: trains a spatial-interpolation model end-to-end (build data
  loader → sample train/val dates → build task streams → build model → `Trainer.train()`).
- **`inference.py`**: loads a trained model from a W&B run (`download_wandb_file` /
  `load_wandb_model`) and runs/export predictions (`export_zarr`).
- **`placement.py`**: loads a trained model from a W&B run, then runs DeepSensor's
  `GreedyAlgorithm` with a chosen acquisition function (`ContextDist`, `Stddev`,
  `MeanMarginalEntropy`, `OracleRMSE`, ...) to propose new sensor locations, saving results and a
  summary plot per acquisition function.

All three scripts follow the same shape: parse config/CLI args → build the source-specific data
loader → sample dates (`"seasonal"`, a Gaussian-weighted-by-day-of-year subsample, or
`"regular"`, a fixed-stride `pd.date_range`) → build station locations (snapping to nearest grid
cell via `find_xy_for_points`, filtered by land-sea mask for `hirham5zarr` or by domain bounding
box otherwise) → hand off to model/training/placement logic. When editing one of these code paths,
check whether the same fix is needed in the other two scripts.

`scripts/preprocessing/hirham5_create_basefile.py` is a standalone preprocessing script (not part
of the train/infer/place pipeline above) that builds the HIRHAM5 base NetCDF file.

### Plotting & metrics (`utils/plots.py`, `utils/metrics.py`)

`utils/plots.py` holds all matplotlib/cartopy figure-building code (prediction maps, PIT
histograms, new-placement maps in both lon/lat and native x/y grids). `utils/metrics.py` has
`compute_pit_bernoulli_gamma`, the PIT calculation for the zero-inflated Bernoulli-Gamma
predictive distribution used to evaluate precipitation-like targets.

### Configuration

Every script is driven by a YAML file under `src/giant_sp/configs/` (`carra2*.yaml`,
`hirham5*.yaml`, `hirham5zarr*.yaml`, each with a `_debug` variant for quick runs). Config
sections map directly onto the pipeline stages above: `data` (source + paths/dates), `dataloader`
(sampling strategy/frequency), `model` (ConvNP architecture), `training` (optimizer/scheduler/
checkpointing), `logging` (W&B), `visualization`.

## Typing & docs conventions

- Type hints use PEP 604 unions (`X | None`, not `Optional[X]`/`Union`).
- Docstrings are NumPy-style (`Parameters`/`Returns`/`Raises`/`Attributes` sections).
- `mypy` config (`[tool.mypy]` in `pyproject.toml`) type-checks all of `src/giant_sp` and `tests`;
  `deepsensor` is exempted from import-following (`follow_imports = "skip"`) since it ships no
  stubs, and `tests.*` is exempted from `disallow_untyped_defs`/`disallow_untyped_decorators` since
  pytest fixtures/decorators are commonly untyped.
