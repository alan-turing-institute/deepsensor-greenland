# DeepSensor Greenland

[![Actions Status](https://github.com/alan-turing-institute/deepsensor-greenland/workflows/CI/badge.svg)](https://github.com/alan-turing-institute/deepsensor-greenland/actions)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Sensor placement to aid location planning in Greenland, based on the [DeepSensor package](https://github.com/alan-turing-institute/deepsensor).

## Installation

From source:
```bash
git clone https://github.com/alan-turing-institute/deepsensor-greenland
cd deepsensor-greenland
```

In this package, we use `uv` for our dependency management. Installation instructions can be found here: https://docs.astral.sh/uv/

## Usage
There are a set of scripts in the `src/ds_greenland/scripts/` folder for training and running the
model. Configuration is managed by [Hydra](https://hydra.cc/): each script composes its config
from YAML files under `src/ds_greenland/configs/`, with a `data`/`hirham5`-style *experiment* selectable
(and any individual value overridable) from the command line.

The `interpolation.py` script trains a spatial interpolation model. It can be run using:

```bash
uv run src/ds_greenland/scripts/interpolation.py experiment=<experiment name>
```
where `<experiment name>` is one of the configs under `src/ds_greenland/configs/experiment/` (e.g.
`carra2`, `hirham5zarr_debug`) — see `src/ds_greenland/configs/base.yaml` for the default. Any config
value can also be overridden directly, e.g. `training.num_epochs=5`. This will save a model to
WandB, and produce a path to the model run, for example: `turing-giant/present-train/runs/5p4odkbr`.

This model can then be used to run `inference.py` to run / export predictions, or `placement.py` to
propose new sensor locations, using the following format:

```bash
uv run <path to script> experiment=<experiment name> <inference|placement>.run_ref=<path to wandb run>
```

Run any script with `--help` to see the available experiments and the full composed config, or
`--cfg job` to print the resolved config without running anything. See [CLAUDE.md](CLAUDE.md) for
more on how the config tree is organized.

### Data sources

Three data sources are available as `experiment=` choices, each with a `_debug` variant:

- `carra2` / `carra2_debug` — CARRA2 reanalysis data, temperature (`2t_abs`) target.
- `hirham5` / `hirham5_debug` — HIRHAM5 RCM data (NetCDF), snowmelt (`snmel_abs`) target.
- `hirham5zarr` / `hirham5zarr_debug` — the same HIRHAM5 source as above, but reading from a
  pre-processed Zarr store instead of NetCDF (adds a rolling 10-year-average covariate, and filters
  station locations by land-sea mask instead of a bounding box).

The `_debug` variants point at small local data samples under `data/` (rather than the JASMIN GWS
paths used in production) and run for only a handful of epochs, so they can run on a laptop
without access to the full dataset — see `src/ds_greenland/configs/experiment/*_debug.yaml`. Use the
non-`_debug` name for an actual training run.

The [github wiki](https://github.com/alan-turing-institute/deepsensor-greenland/wiki) has some information about the datasets and the infrastructure used in this repo if you are part of the project.

### Local config overrides

For overrides you want to reuse across runs without editing a shared config file — e.g. always
tagging your own W&B runs, or defaulting to a particular debug experiment — add a
`src/ds_greenland/configs/<name>.local.yaml` file. Any `*.local.yaml` file is gitignored, so it's safe
to keep personal tweaks there without them ending up in a commit. It should compose on top of
`base`/`inference`/`placement` the same way those files do, e.g.:

```yaml
defaults:
  - base
  - override /experiment: carra2_debug
  - _self_

logging:
  wandb:
    tags: ["spatial-interpolation", "convgnp", "t2m", "carra2", "debug", "<your-name>"]
```

Run it by pointing Hydra at it with `--config-name` (omit the `.yaml` extension):

```bash
uv run src/ds_greenland/scripts/interpolation.py --config-name=<name>.local
```

Any config value can also be overridden from the command line, e.g.
`--config-name=<name>.local training.num_epochs=5`.

### Hyperparameter sweeps

[W&B Sweeps](https://docs.wandb.ai/models/sweeps/) can be used to search over hyperparameters for
`interpolation.py`. A sweep config lives under `src/ds_greenland/configs/sweep/` (see
`example.yaml`); its `parameters:` keys are dotted Hydra config keys, applied as ordinary CLI
overrides via the `${args_no_hyphens}` command macro — no code changes are needed, since
`interpolation.py` already calls `wandb.init()`, which joins the sweep automatically when run
under `wandb agent`. Launch a sweep with:

```bash
uv run wandb sweep src/ds_greenland/configs/sweep/example.yaml
uv run wandb agent <entity/project/sweep_id printed by the command above>
```

Use the full `wandb agent ...` command that `wandb sweep` prints (or its
`<entity>/<project>/<sweep_id>` argument) rather than the bare 8-character sweep ID — without an
`entity`/`project` prefix, `wandb agent` can only resolve the sweep if a default entity/project is
already configured locally (e.g. via `wandb init` or `WANDB_ENTITY`/`WANDB_PROJECT`), otherwise it
fails with `Error: name required for project query`.

Run multiple agents (locally or on separate machines/jobs) against the same sweep to parallelize
the search. Copy `example.yaml` to sweep a different experiment or parameter set.

## Demo site

`demo/` is a static site presenting the project. Its Data page embeds a dashboard that streams the
`aria-giant/hirham5-erai-modis-test` dataset live from Earthmover (Arraylake): domain-mean time
series per experiment/variable, plus a map for any chosen date, rendered on demand. The data isn't
stored locally, so the site is served by a small local server (`ds_greenland.demo`) rather than opened
as files.

The Arraylake client needs Python 3.12+, so run the demo in its own environment (this leaves your
main `.venv` alone):

```bash
uvx --python 3.12 arraylake auth login   # one-time; caches a token in ~/.arraylake
UV_PROJECT_ENVIRONMENT=.venv-demo uv run --python 3.12 python -m ds_greenland.demo
```

Then open <http://127.0.0.1:8000>. Use `--port`/`--host` to change where it listens. Instead of the
cached login you can set `ARRAYLAKE_TOKEN`, and `ARRAYLAKE_REPO`/`ARRAYLAKE_BRANCH`/
`ARRAYLAKE_ROOT_PATH` point it at a different repo.

### Offline copy of the dashboard

In case Arraylake or the internet is unreachable during a demo, save a copy of the dashboard's data
beforehand, while online:

```bash
UV_PROJECT_ENVIRONMENT=.venv-demo uv run --python 3.12 python -m ds_greenland.demo.snapshot
```

This writes every time series, maps for every variable on the first of each month (~200 files, ~23 MB),
and a copy of Plotly to `demo/dashboard/static-api/` (git-ignored). Use `--map-variables` and
`--map-freq` (a pandas offset alias, e.g. `W` for weekly) to save more maps. When the live data
can't be loaded, the dashboard switches to this copy automatically (the footer shows when it was saved), and map dates snap
to the nearest saved one. You can also switch by hand with the "Use saved copy" / "Use live data"
button at the bottom of the dashboard (shown once a saved copy exists). If the demo server can't run at all, any static server will do:
`python -m http.server -d demo`, then open <http://127.0.0.1:8000>.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).
