"""Utilities for converting Hydra/OmegaConf configuration values to plain Python."""

from typing import Any

from omegaconf import OmegaConf


def to_plain(value: Any) -> Any:
    """Convert an OmegaConf container to plain Python, unresolved scalars pass through.

    `OmegaConf.to_container` requires an actual `DictConfig`/`ListConfig`
    and raises `ValueError` on anything else, but indexing into a
    `DictConfig` already returns plain scalars (`str`, `int`, `bool`,
    `None`) unwrapped. This checks first, so a scalar leaf is returned
    unchanged instead of raising.

    Parameters
    ----------
    value : Any
        A config value, as returned by indexing into a `DictConfig`. May be
        a `DictConfig`/`ListConfig` or an already-unwrapped scalar.

    Returns
    -------
    Any
        `value` converted to a plain `dict`/`list` (recursively, with
        interpolations resolved) if it was a config container, otherwise
        `value` unchanged.

    """
    return (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )
