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

The [github wiki](https://github.com/alan-turing-institute/giant-sp/wiki) has some information about the datasets and the infrastructure used in this repo.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).
