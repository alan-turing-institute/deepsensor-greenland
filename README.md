# giant-sp

[![Actions Status](https://github.com/alan-turing-institute/giant-sp/workflows/CI/badge.svg)](https://github.com/alan-turing-institute/giant-sp/actions)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Sensor placement to aid location planning in GIANT

## Installation

From source:
```bash
git clone https://github.com/alan-turing-institute/giant-sp
cd giant-sp
```

In this package, we use `uv` for our dependency management. Installation instructions can be found here: https://docs.astral.sh/uv/

## Usage
There are a set of scripts in the `src/giant_sp/scripts/` folder for training and running the
model. Configuration is managed by [Hydra](https://hydra.cc/): each script composes its config
from YAML files under `src/giant_sp/configs/`, with a `data`/`hirham5`-style *experiment* selectable
(and any individual value overridable) from the command line.

The `interpolation.py` script trains a spatial interpolation model. It can be run using:

```bash
uv run src/giant_sp/scripts/interpolation.py experiment=<experiment name>
```
where `<experiment name>` is one of the configs under `src/giant_sp/configs/experiment/` (e.g.
`carra2`, `hirham5zarr_debug`) — see `src/giant_sp/configs/base.yaml` for the default. Any config
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
without access to the full dataset — see `src/giant_sp/configs/experiment/*_debug.yaml`. Use the
non-`_debug` name for an actual training run.

The [github wiki](https://github.com/alan-turing-institute/giant-sp/wiki) has some information about the datasets and the infrastructure used in this repo.

### Local config overrides

For overrides you want to reuse across runs without editing a shared config file — e.g. always
tagging your own W&B runs, or defaulting to a particular debug experiment — add a
`src/giant_sp/configs/<name>.local.yaml` file. Any `*.local.yaml` file is gitignored, so it's safe
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
uv run src/giant_sp/scripts/interpolation.py --config-name=<name>.local
```

Any config value can also be overridden from the command line, e.g.
`--config-name=<name>.local training.num_epochs=5`.

## Demo site

`demo/` is a static site presenting the project. Its Data page embeds a dashboard that streams the
`aria-giant/hirham5-erai-modis-test` dataset live from Earthmover (Arraylake): domain-mean time
series per experiment/variable, plus a map for any chosen date, rendered on demand. The data isn't
stored locally, so the site is served by a small local server (`giant_sp.demo`) rather than opened
as files.

The Arraylake client needs Python 3.12+, so run the demo in its own environment (this leaves your
main `.venv` alone):

```bash
uvx --python 3.12 arraylake auth login   # one-time; caches a token in ~/.arraylake
UV_PROJECT_ENVIRONMENT=.venv-demo uv run --python 3.12 python -m giant_sp.demo
```

Then open <http://127.0.0.1:8000>. Use `--port`/`--host` to change where it listens. Instead of the
cached login you can set `ARRAYLAKE_TOKEN`, and `ARRAYLAKE_REPO`/`ARRAYLAKE_BRANCH`/
`ARRAYLAKE_ROOT_PATH` point it at a different repo.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).
