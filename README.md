# giant-sp

[![Actions Status](https://github.com/alan-turing-institute/giant-sp/workflows/CI/badge.svg)](https://github.com/alan-turing-institute/giant-sp/actions)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-green)](LICENSE)

Sensor placement to aid location planning in GIANT

## Installation

```bash
python -m pip install giant_sp
```

From source:
```bash
git clone https://github.com/alan-turing-institute/giant-sp
cd giant-sp
python -m pip install .
```

## Usage
There are a set of scripts in the `/scripts` folder, which can be run as:

```bash
uv run <path to script> --config <path to config>
```
The `interpolation.py` script trains a spatial interpolation model. This model can then be used to run `inference.py` to run / export predictions, or `placement.py` to propose new sensor locations.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            
[actions-link]:             
<!-- prettier-ignore-end -->
