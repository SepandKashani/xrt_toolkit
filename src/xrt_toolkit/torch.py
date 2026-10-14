r"""
PyTorch bindings.

:py:func:`xrt_torch` is :py:func:`~xrt_toolkit.xrt_apply` as a differentiable PyTorch function:
the volume and the rays are tensors, and gradients reach all three,

* the volume, through the exact adjoint :py:func:`~xrt_toolkit.xrt_adjoint`;
* the ray anchors and directions, through :py:func:`~xrt_toolkit.xrt_grad`.

Example: calibrate the rays together with the volume.

.. code-block:: python

   import torch, xrt_toolkit as xtk
   from xrt_toolkit.torch import xrt_torch

   x = torch.zeros(math.prod(knot_spec.num), device="cuda", requires_grad=True)
   t = torch.nn.Parameter(t0)  # (D, L) ray anchors
   n = torch.nn.Parameter(n0)  # (D, L) ray directions
   loss = ((xrt_torch(x, t, n, knot_spec, order) - y) ** 2).sum()
   loss.backward()  # fills x.grad, t.grad, n.grad
"""

import math

import drjit as dr
import torch

import xrt_toolkit.util as xrtu

from .drjit.ray_grad import xrt_grad
from .drjit.ray_xrt import xrt_adjoint, xrt_apply


def _drjit_types(x: torch.Tensor, D: int) -> tuple:
    # (ArrayNf, Float) Dr.Jit types on the tensor's device (CUDA or CPU) and precision
    if x.is_cuda:
        import drjit.cuda as backend
    else:
        import drjit.llvm as backend
    suffix = {torch.float32: "", torch.float64: "64"}[x.dtype]
    return getattr(backend, f"Array{D}f{suffix}"), getattr(backend, f"Float{suffix}")


def _rays(t: torch.Tensor, n: torch.Tensor) -> tuple:
    ArrayNf, Float = _drjit_types(t, t.shape[0])
    return tuple(
        ArrayNf(*[Float(v[d].contiguous()) for d in range(len(v))]) for v in (t, n)
    )


class _XRT(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, t, n, knot_spec, order):
        _, Float = _drjit_types(data, t.shape[0])
        y = xrt_apply(_rays(t, n), knot_spec, order, Float(data.contiguous()))
        ctx.save_for_backward(data, t, n)
        ctx.knot_spec, ctx.order = knot_spec, order
        return y.torch()

    @staticmethod
    def backward(ctx, grad_y):
        data, t, n = ctx.saved_tensors
        _, Float = _drjit_types(data, t.shape[0])
        rays = _rays(t, n)
        g = Float(grad_y.contiguous())

        need_data, need_t, need_n = ctx.needs_input_grad[:3]
        out = [None, None, None]
        if need_data:  # A^T g
            out[0] = xrt_adjoint(rays, ctx.knot_spec, ctx.order, g)
        if need_t or need_n:
            # each projection depends only on its own ray: g_l times the ray's gradient
            g_t, g_n = xrt_grad(
                rays, ctx.knot_spec, ctx.order, Float(data.contiguous())
            )
            out[1] = (g_t * g) if need_t else None
            out[2] = (g_n * g) if need_n else None
        dr.eval(*[o for o in out if o is not None])  # one launch for all of them
        grad_data, grad_t, grad_n = (None if (o is None) else o.torch() for o in out)
        return grad_data, grad_t, grad_n, None, None


def xrt_torch(
    data: torch.Tensor,
    t: torch.Tensor,
    n: torch.Tensor,
    knot_spec: xrtu.UniformSpec,
    order: int,
) -> torch.Tensor:
    r"""
    Projections of a volume along rays, differentiable in all three.

    Parameters
    ----------
    data: torch.Tensor
        (Q1*...*QD,) flattened C-ordered volume coefficients.
    t, n: torch.Tensor
        (D, L) ray anchors and directions, one column per ray, on the same device (CUDA or CPU)
        and in the same precision (float32 or float64) as `data`.
    knot_spec: UniformSpec
        Volume lattice. (See :py:func:`~xrt_toolkit.xrt_apply`.)
    order: 0 | 1 | 2
        Basis order.

    Returns
    -------
    proj: torch.Tensor
        (L,) projections.
    """
    D = knot_spec.ndim
    assert (t.shape == n.shape) and (t.shape[0] == D)
    assert data.shape == (math.prod(knot_spec.num),)
    assert (data.device == t.device == n.device) and (data.dtype == t.dtype == n.dtype)
    return _XRT.apply(data, t, n, knot_spec, order)


class XRTProjector(torch.nn.Module):
    r"""
    :py:func:`xrt_torch` as a ``torch.nn.Module``, for a fixed volume lattice and basis order.
    """

    def __init__(self, knot_spec: xrtu.UniformSpec, order: int):
        super().__init__()
        self.knot_spec, self.order = knot_spec, order

    def forward(
        self, data: torch.Tensor, t: torch.Tensor, n: torch.Tensor
    ) -> torch.Tensor:
        return xrt_torch(data, t, n, self.knot_spec, self.order)
