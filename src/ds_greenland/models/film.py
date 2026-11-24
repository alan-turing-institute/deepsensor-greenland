"""FiLM adapters for fine-tuning a pretrained ConvNP's UNet.

FiLM (feature-wise linear modulation, Perez et al. 2018,
https://arxiv.org/abs/1709.07871) scales and shifts every channel of a layer's
output. Adding one after each convolution of a pretrained UNet, starting at the
identity, and training only those (with every pretrained weight frozen) adapts
the model to a new dataset with a small fraction of its parameters.

A FiLM layer directly after a convolution is itself absorbed by that convolution
(``scale * conv(x; W, b) + shift == conv(x; scale * W, scale * b + shift)``), so
`folded_state_dict` turns a model with FiLM layers back into a plain ``ConvNP``
state dict: checkpoints stay loadable by a ``ConvNP`` rebuilt from the run's
``model_config.json``, with no knowledge of FiLM.
"""

import torch
from neuralprocesses.torch import UNet
from torch import nn
from torch.nn.modules.conv import _ConvNd

FILM_ATTR = "film"


class FiLM(nn.Module):
    """Per-channel affine transform ``(1 + scale_delta) * x + shift``.

    Starts at the identity. The scale is stored as an offset from 1 so that
    weight decay pulls the layer toward the identity rather than toward zero.

    Attributes
    ----------
    scale_delta : torch.nn.Parameter
        Per-channel offset of the scale from 1, shape ``(num_channels,)``.
    shift : torch.nn.Parameter
        Per-channel shift, shape ``(num_channels,)``.

    """

    def __init__(
        self,
        num_channels: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Create an identity FiLM layer.

        Parameters
        ----------
        num_channels : int
            Number of channels (dimension 1 of the input) to modulate.
        device : torch.device | None, optional
            Device of the parameters. Default is torch's default device.
        dtype : torch.dtype | None, optional
            Data type of the parameters. Default is torch's default dtype.

        """
        super().__init__()
        self.scale_delta = nn.Parameter(
            torch.zeros(num_channels, device=device, dtype=dtype)
        )
        self.shift = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))

    @property
    def scale(self) -> torch.Tensor:
        """Per-channel scale, ``1 + scale_delta``."""
        return 1 + self.scale_delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale and shift every channel of `x`.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, num_channels, *spatial)``.

        Returns
        -------
        torch.Tensor
            ``scale * x + shift``, broadcast over the batch and spatial
            dimensions.

        """
        shape = (1, -1) + (1,) * (x.dim() - 2)
        return x * self.scale.view(shape) + self.shift.view(shape)


def _apply_film(
    module: nn.Module, args: tuple[object, ...], output: torch.Tensor
) -> torch.Tensor:
    """Forward hook passing a convolution's output through its FiLM layer."""
    film: FiLM = getattr(module, FILM_ATTR)
    return film(output)


def _film_convs(module: nn.Module) -> list[tuple[str, _ConvNd, FiLM]]:
    """List ``(name, conv, film)`` for every convolution in `module` that has a FiLM layer."""
    return [
        (name, conv, film)
        for name, conv in module.named_modules()
        if isinstance(conv, _ConvNd)
        and isinstance(film := getattr(conv, FILM_ATTR, None), FiLM)
    ]


def add_film_layers(module: nn.Module) -> list[str]:
    """Add an identity FiLM layer after every convolution of each UNet in `module`.

    Each layer of a UNet's downsampling (``before_turn_layers``) and
    upsampling (``after_turn_layers``) paths gets a `FiLM` layer applied to
    the output of its convolution (before any pooling and the activation).
    The UNet's ``final_linear`` output layer is left as it is. The FiLM
    layer is registered on the convolution as its ``film`` submodule, so its
    parameters are named e.g.
    ``decoder.links.0.before_turn_layers.0.film.scale_delta``.

    The model's output is unchanged until the FiLM parameters are trained.
    The pretrained parameters are not frozen here; call
    ``module.requires_grad_(False)`` first to train only the FiLM layers.

    Parameters
    ----------
    module : torch.nn.Module
        The model's torch module (``ConvNP.model``).

    Returns
    -------
    list[str]
        Names of the convolutions that were given a FiLM layer.

    Raises
    ------
    ValueError
        If `module` has no UNet (e.g. ``conv_arch="conv"``), already has
        FiLM layers, or a UNet layer isn't a single biased convolution (as
        with the ``"unet-res"``/``"unet-sep"`` architectures), which a FiLM
        layer couldn't be folded into.

    """
    if _film_convs(module):
        raise ValueError("The model already has FiLM layers")

    names = {id(m): name for name, m in module.named_modules()}
    unets = [m for m in module.modules() if isinstance(m, UNet)]
    if not unets:
        raise ValueError(
            "FiLM adaptation needs a UNet, but the model has none (is its "
            "conv_arch 'conv'?)"
        )

    convs = []
    for unet in unets:
        for layer in [*unet.before_turn_layers, *unet.after_turn_layers]:
            layer_convs = [m for m in layer.modules() if isinstance(m, _ConvNd)]
            if len(layer_convs) != 1:
                raise ValueError(
                    f"UNet layer {names[id(layer)]} has {len(layer_convs)} "
                    "convolutions; FiLM adaptation only supports plain UNets "
                    "(conv_arch 'unet'), with one convolution per layer"
                )
            conv = layer_convs[0]
            if conv.bias is None or (conv.transposed and conv.groups != 1):
                raise ValueError(
                    f"Can't fold a FiLM layer into convolution {names[id(conv)]} "
                    "(it has no bias, or is a grouped transposed convolution)"
                )
            convs.append(conv)

    for conv in convs:
        film = FiLM(
            conv.out_channels, device=conv.weight.device, dtype=conv.weight.dtype
        )
        conv.add_module(FILM_ATTR, film)
        conv.register_forward_hook(_apply_film)
    return [names[id(conv)] for conv in convs]


def folded_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Return `module`'s state dict with each FiLM layer folded into its convolution.

    The result has exactly the keys of the model without FiLM layers, and
    computes the same function as `module`: it can be loaded into a plain
    ``ConvNP`` built from the same config. For a module without FiLM layers
    it is just ``module.state_dict()``. `module` itself is left unchanged.

    Parameters
    ----------
    module : torch.nn.Module
        The model's torch module (``ConvNP.model``), with or without FiLM
        layers (see `add_film_layers`).

    Returns
    -------
    dict[str, torch.Tensor]
        The folded state dict.

    """
    state = module.state_dict()
    with torch.no_grad():
        for name, conv, film in _film_convs(module):
            # Output channels are dimension 0 of a convolution's weight and
            # dimension 1 of a transposed convolution's.
            shape = [1] * conv.weight.dim()
            shape[1 if conv.transposed else 0] = -1
            assert conv.bias is not None  # checked by `add_film_layers`
            state[f"{name}.weight"] = conv.weight * film.scale.view(shape)
            state[f"{name}.bias"] = conv.bias * film.scale + film.shift
            for param_name, _ in film.named_parameters():
                del state[f"{name}.{FILM_ATTR}.{param_name}"]
    return state


def load_folded_state_dict(
    module: nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    """Load a plain ``ConvNP`` state dict into `module`, resetting its FiLM layers.

    The pretrained weights are loaded into the convolutions themselves and
    every FiLM layer of `module` is reset to the identity, so `module`
    computes exactly what the state dict's model does. FiLM parameters
    present in `state_dict` (an unfolded checkpoint) are loaded instead of
    being reset. For a module without FiLM layers this is just a strict
    ``module.load_state_dict(state_dict)``.

    Parameters
    ----------
    module : torch.nn.Module
        The model's torch module (``ConvNP.model``), with or without FiLM
        layers (see `add_film_layers`).
    state_dict : dict[str, torch.Tensor]
        State dict to load, e.g. one saved from `folded_state_dict`.

    """
    state = dict(state_dict)
    for name, _, film in _film_convs(module):
        for param_name, param in film.named_parameters():
            state.setdefault(
                f"{name}.{FILM_ATTR}.{param_name}", torch.zeros_like(param)
            )
    module.load_state_dict(state)
