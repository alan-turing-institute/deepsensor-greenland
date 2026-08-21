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

Running the actual training/inference/placement scripts requires access to the data referenced in
the composed config (see `data.data_path`, `data.station_locations_file`, etc.) — these are not
part of the repo and are not available in most dev environments. Each script is a
[Hydra](https://hydra.cc/) entry point; run it with no arguments for the default experiment
(`uv run src/giant_sp/scripts/interpolation.py`), override the experiment with
`experiment=<name>` (e.g. `experiment=hirham5zarr`), or override any individual value with
`key=value` (e.g. `training.num_epochs=5`). See "Configuration" below for how the config tree is
organized, `--help` for the full composed config and available experiments, or `--cfg job` to
print the resolved config without running anything.

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

Every script is a Hydra entry point composing its config from `src/giant_sp/configs/`:
`base.yaml`/`inference.yaml`/`placement.yaml` are the per-script entry configs (each defaulting
to one `experiment`). `configs/experiment/{carra2,hirham5}{,_debug}.yaml` each hold their own
`data`/`dataloader` section plus the `model`/`training`/`logging`/`visualization` values that
differ for that source; scalars identical across every source (optimizer/scheduler settings,
most of `logging`/`visualization`, ...) live once in `configs/experiment/common.yaml`, which
`carra2`/`hirham5` inherit from via `defaults: [common, _self_]`. `hirham5zarr.yaml` doesn't repeat
any of that — since it's identical to `hirham5.yaml` outside of the data backend, it inherits from
`hirham5.yaml` directly and overrides only `data`/`dataloader` fields (paths,
`context_var_IDs`/`context_delta_t` shape). `_debug` variants compose on top of their own prod counterpart the same way (`carra2_debug` on
`carra2`, `hirham5_debug` on `hirham5`, `hirham5zarr_debug` on `hirham5zarr`), overriding only the
fields that differ (paths, date ranges, epoch counts, ...). Because `hirham5zarr_debug` inherits
from `hirham5zarr` rather than from `hirham5_debug`, its debug-scale overrides (dates,
`model`/`training` sizing, W&B log frequency) are a near-verbatim copy of `hirham5_debug.yaml`'s
rather than inherited from it — keep the two in sync by hand if debug scale changes.
`inference.yaml`/`placement.yaml` additionally carry an
`inference:`/`placement:` section for that script's run-specific parameters (`run_ref`, `acq_fn`,
`n_new`, ...). Config sections map directly onto the pipeline stages above: `data` (source +
paths/dates), `dataloader` (sampling strategy/frequency), `model` (ConvNP architecture), `training`
(optimizer/scheduler/checkpointing), `logging` (W&B), `visualization`.

To add a new experiment variant or a personal override without touching shared files, add a new
YAML under `configs/experiment/` that lists the base experiment in its own `defaults:` and overrides
only the changed keys (same pattern as the `_debug`/`hirham5zarr` files), or point `--config-name`
at an entirely separate top-level config (e.g. a gitignored `<name>.local.yaml` — see the README).

## Typing & docs conventions

- Type hints use PEP 604 unions (`X | None`, not `Optional[X]`/`Union`).
- Docstrings are NumPy-style (`Parameters`/`Returns`/`Raises`/`Attributes` sections).
- `mypy` config (`[tool.mypy]` in `pyproject.toml`) type-checks all of `src/giant_sp` and `tests`;
  `deepsensor` is exempted from import-following (`follow_imports = "skip"`) since it ships no
  stubs, and `tests.*` is exempted from `disallow_untyped_defs`/`disallow_untyped_decorators` since
  pytest fixtures/decorators are commonly untyped.
