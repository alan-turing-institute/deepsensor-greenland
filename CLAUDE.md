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

### Demo site (`demo/`, `src/giant_sp/demo/`)

`demo/` is a static presentation site (plain HTML + `nav.js`'s `<site-shell>` element). Its Data
page iframes `demo/dashboard/`, a Plotly dashboard that reads `./api/*` from the local server in
`src/giant_sp/demo/` (`uv run python -m giant_sp.demo`, Python 3.12+ — see the README): `source.py`
opens the Arraylake/Icechunk repo, `dashboard.py` discovers experiments/variables and builds
domain-mean series and on-demand cartopy maps (cached per snapshot), and `server.py` is the FastAPI
app serving `/dashboard/api/*` plus the static site at `/`. Its deps are the `demo` dependency
group; tests (`tests/demo/`) use an in-memory zarr group instead of Arraylake.

Run the demo (and anything else needing Arraylake) with `UV_PROJECT_ENVIRONMENT=.venv-demo uv run
--python 3.12 ...`, never a bare `uv run --python 3.12`: the main `.venv` is Python 3.11, and the
next plain `uv run` rebuilds it — pulling the environment out from under a running demo server
(its periodic Arraylake credential refresh then fails with `FileNotFoundError`).

Data the dashboard should hide is excluded server-side, in `dashboard.py`'s
`EXCLUDED_VARIABLES` (runoff `rogl`, albedo `albedom`) and `EXCLUDED_EXPERIMENTS`
(`baseline_10year`), which both the catalog and the series/map endpoints honour. Map colormaps
are also chosen server-side, in `dashboard.py`'s `MAP_CMAPS` (keyed on lowercased `long_name`;
surface melt uses the Results figure's reversed colorcet `linear_kryw_5_100_c67`, everything else
`DEFAULT_MAP_CMAP`, viridis). By default every date of a variable's map shares one colour scale:
the `MAP_PERCENTILES` (1st to 99th) over all dates and points, cached per snapshot; skewed
variables in `LOG_SCALE_VARIABLES` use a log scale instead. The sidebar's "Map colour scale"
choice (`scale` on `/map`, a `MapScale`) picks between that (`shared`), the data's min to max over
all dates (`full`), or the mapped date's own min to max, giving each date its own colour bar
(`date`; always linear, even for `LOG_SCALE_VARIABLES`). Display-side
choices live at the top of `demo/dashboard/app.js`: `DEFAULT_VARIABLE_NAMES` (shown on first load)
and `OUTPUT_VARIABLES` (long_name → display title for the sidebar's "Output variable" section;
everything else is an input variable). Variables are matched on `long_name`, not key. There's no
experiment picker — every experiment the server exposes is shown.

`snapshot.py` (`python -m giant_sp.demo.snapshot`, run by hand before a demo, while online) saves an
offline copy of the API to `demo/dashboard/static-api/` (git-ignored): `repo.json`, `catalog.json`,
every series, map PNGs for `--map-variables` (default: every 2-D variable in the catalog) at `--map-freq` (default `MS`,
the 1st of each month), each in all three colour scales (`<date>.png`, `<date>-full.png` and
`<date>-date.png`; keep `snapshot.py`'s `MAP_SCALE_SUFFIXES` and `app.js`'s `SAVED_MAP_SUFFIXES` in sync), listed in
`maps/index.json`, and a copy of Plotly (keep `PLOTLY_URL` in sync
with the `<script>` in `dashboard/index.html`). If the live catalog fails or times out — or the
footer's "Use saved copy" button (shown only when a saved copy exists) is clicked — `app.js`
switches every request to that copy (noting its save date in the footer; there's deliberately no
banner), and snaps map dates to the nearest saved one.
The copy mirrors the catalog at save time, so it needs regenerating after the Arraylake data or the
server-side exclusions change — but not after front-end-only changes.

The Results tab's Pareto figure is generated by `demo/results/placement_pareto_demo.py`
(`uv run python demo/results/placement_pareto_demo.py`): it plots every complete result set under
`demo/results/pareto/` (discovered from `mask/<run>_<acq_fn>_<cost>.npy`, with matching `acq/`,
`cost/`, `rank/`, `placements/` files) to `demo/images/pareto/`, plus a `manifest.json` that
`demo/results.js` reads to show them (with a picker when there's more than one). The Overview page
loads the same script to show the first figure, without the picker.

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
