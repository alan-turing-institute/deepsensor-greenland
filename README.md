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
There are a set of scripts in the `/scripts` folder for training and running the model.

The `interpolation.py ` script trains a spatial interpolation model. It can be run using:

```bash
uv run src/giant_sp/scripts/interpolation.py --config <path to data config>
```
This will save a model to WandB, and produce a path to the model run, for example: `turing-giant/present-train/runs/5p4odkbr`.

This model can then be used to run `inference.py` to run / export predictions, or `placement.py` to propose new sensor locations, using the following format:

```bash
uv run <path to script> --config <path to data config> --run_ref <path to wandb run>
```

The [github wiki](https://github.com/alan-turing-institute/giant-sp/wiki) has some information about the datasets and the infrastructure used in this repo.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).
