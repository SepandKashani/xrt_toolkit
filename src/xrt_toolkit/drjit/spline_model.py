"""
Helper utilities to build and load the 3D spline network in Dr.Jit.

The Dr.Jit network mirrors the PyTorch architecture defined in
``paper_xtk_latex/xp_paper/3D_boxsplines.ipynb``::

    Linear(4 -> 16) + ReLU repeated four times, followed by Linear(16 -> 1)

The :func:`load_spline_network` helper instantiates the Dr.Jit model and can
optionally import weights from the reference PyTorch checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import numpy as np

import drjit as dr
import drjit.nn as dnn
from drjit.llvm.ad import Float16, TensorXf16

PathLike = Union[str, Path]

_ROOT_DIR = Path(__file__).resolve().parents[3]
_DEFAULT_TORCH_CKPT = _ROOT_DIR / "src" / "3D_boxsplines_model.pth"
_DEFAULT_DRJIT_CKPT = _ROOT_DIR / "src" / "3D_spline_weights.npz"


def _build_spline_sequential() -> dnn.Sequential:
    """Return the canonical 4×16 MLP used for the spline approximation."""

    return dnn.Sequential(
        dnn.Linear(4, 16),
        dnn.ReLU(),
        dnn.Linear(16, 16),
        dnn.ReLU(),
        dnn.Linear(16, 16),
        dnn.ReLU(),
        dnn.Linear(16, 16),
        dnn.ReLU(),
        dnn.Linear(16, 1),
    )


def _ensure_torch():
    try:
        import torch  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyTorch is required to import spline weights from a checkpoint."
        ) from exc
    return torch


def _copy_linear_weights(
    torch_layers: Iterable["torch.nn.Linear"],
    dr_layers: Iterable[dnn.Linear],
) -> None:
    for torch_layer, dr_layer in zip(torch_layers, dr_layers, strict=True):
        weight_np = torch_layer.weight.detach().cpu().numpy().astype(np.float32)
        bias_np = torch_layer.bias.detach().cpu().numpy().astype(np.float32)

        dr_layer.weights[:] = TensorXf16(weight_np)
        dr_layer.bias[:] = TensorXf16(bias_np)


def load_spline_network(
    *,
    dtype: type = TensorXf16,
    seed: int = 0,
    torch_checkpoint: Optional[PathLike] = None,
    drjit_checkpoint: Optional[PathLike] = None,
    require_torch_checkpoint: bool = False,
) -> Tuple[Float16, dnn.Sequential]:
    """
    Instantiate the Dr.Jit spline network, optionally restoring weights.

    Parameters
    ----------
    dtype:
        Tensor type used to allocate the network weights. Defaults to
        :class:`drjit.llvm.ad.TensorXf16`, matching the notebook setup.
    seed:
        Seed used when allocating the network.
    torch_checkpoint:
        Optional path (``.pth``) to a PyTorch checkpoint whose parameters will
        be copied into the Dr.Jit network prior to packing. When omitted, the
        function falls back to ``src/3D_boxsplines_model.pth`` if available.
    drjit_checkpoint:
        Optional path (``.npz``) containing Dr.Jit weights saved via
        ``np.savez(..., weights=...)``. If provided, this overwrites the packed
        weight buffer after copying any PyTorch parameters.
    require_torch_checkpoint:
        If ``True`` and no PyTorch checkpoint path is available, raise an
        exception instead of leaving the network randomly initialised.

    Returns
    -------
    weights, net:
        The packed training weights (``Float16``) and the packed network object.
        Both outputs can be fed directly into optimisation/evaluation code.
    """

    net = _build_spline_sequential()
    net = net.alloc(dtype=dtype, size=4, rng=dr.rng(seed=seed))

    ckpt_path: Optional[Path] = Path(torch_checkpoint) if torch_checkpoint else None
    if ckpt_path is None and _DEFAULT_TORCH_CKPT.exists():
        ckpt_path = _DEFAULT_TORCH_CKPT

    if ckpt_path is not None:
        torch = _ensure_torch()

        class _TorchSplineNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(4, 16)
                self.fc2 = torch.nn.Linear(16, 16)
                self.fc3 = torch.nn.Linear(16, 16)
                self.fc4 = torch.nn.Linear(16, 16)
                self.fc5 = torch.nn.Linear(16, 1)

            def forward(self, x):  # pragma: no cover - inference only
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                x = torch.relu(self.fc4(x))
                return self.fc5(x)

        model = _TorchSplineNN()
        state_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state_dict)

        torch_layers = [
            module for module in model.modules() if isinstance(module, torch.nn.Linear)
        ]
        dr_layers = [layer for layer in net.layers if isinstance(layer, dnn.Linear)]

        if len(torch_layers) != len(dr_layers):
            raise RuntimeError(
                "Mismatch between PyTorch and Dr.Jit layer counts "
                f"({len(torch_layers)} vs {len(dr_layers)})."
            )

        _copy_linear_weights(torch_layers, dr_layers)
    elif require_torch_checkpoint:
        raise FileNotFoundError(
            "No PyTorch checkpoint provided or found at the default location."
        )

    weights, net = dnn.pack(net, layout="training")

    ckpt_npz: Optional[Path] = Path(drjit_checkpoint) if drjit_checkpoint else None
    if ckpt_npz is None and _DEFAULT_DRJIT_CKPT.exists():
        ckpt_npz = _DEFAULT_DRJIT_CKPT

    if ckpt_npz is not None and ckpt_npz.exists():
        weights_np = np.load(ckpt_npz)["weights"].astype(np.float32)
        weights[:] = Float16(weights_np)

    return weights, net


__all__ = ["load_spline_network"]
