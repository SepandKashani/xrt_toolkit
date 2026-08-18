r"""
PyTorch bindings for the XTK X-ray transform.

This module exposes the projector as a :class:`torch.autograd.Function` so
that it composes with PyTorch's autograd. Gradients flow to three groups of
inputs:

* the image coefficients (through the exact adjoint),
* the ray anchors :math:`\bbt` (through :py:func:`xrt_ad_t_x` / ``_t_y``),
* the ray directions :math:`\bbn` (through :py:func:`xrt_ad_n_x` / ``_n_y``).

The kernels run in Dr.Jit's symbolic mode, so each call compiles to a single
fused kernel rather than one launch per traversal step.

The geometry gradients are available for 2D geometries, which is what the
library validates; image gradients are available in 2D and 3D. Tensors stay
on the GPU: conversions to and from Dr.Jit are zero-copy.

Example
-------
>>> import torch, xrt_toolkit as xtk
>>> from xrt_toolkit.torch import XRTProjector
>>> knot = xtk.UniformSpec(start=0, step=1, num=(128, 128))
>>> proj = XRTProjector(knot, order=2)
>>> x = torch.rand(128 * 128, device="cuda", requires_grad=True)
>>> t = torch.nn.Parameter(t0)              # (2, M) ray anchors
>>> n = torch.nn.Parameter(n0)              # (2, M) ray directions
>>> y = proj(x, t, n)                        # (M,) projections
>>> loss = torch.mean((y - y_meas) ** 2)
>>> loss.backward()                          # fills x.grad, t.grad, n.grad
"""

import math

import torch

from .util import UniformSpec


def _dr():
    import drjit as dr
    from drjit.cuda.ad import Array2f, Array3f, Float

    from .drjit import ray_xrt as _xrt

    return dr, Array2f, Array3f, Float, _xrt


class _XRTFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, t, n, knot_spec, order, mode):
        dr, Array2f, Array3f, Float, _xrt = _dr()
        D = t.shape[0]
        ArrayNf = Array2f if D == 2 else Array3f
        ray = (ArrayNf(*[Float(t[d].contiguous()) for d in range(D)]),
               ArrayNf(*[Float(n[d].contiguous()) for d in range(D)]))
        data_dr = Float(data.contiguous())
        y = _xrt.xrt_apply(ray, knot_spec, order, data_dr, mode=mode)
        ctx.save_for_backward(data, t, n)
        ctx.knot_spec, ctx.order, ctx.D, ctx.mode = knot_spec, order, D, mode
        return y.torch()

    @staticmethod
    def backward(ctx, grad_out):
        dr, Array2f, Array3f, Float, _xrt = _dr()
        data, t, n = ctx.saved_tensors
        knot, order, D = ctx.knot_spec, ctx.order, ctx.D
        mode = ctx.mode
        ArrayNf = Array2f if D == 2 else Array3f
        ray = (ArrayNf(*[Float(t[d].contiguous()) for d in range(D)]),
               ArrayNf(*[Float(n[d].contiguous()) for d in range(D)]))
        g = Float(grad_out.contiguous())

        grad_data = grad_t = grad_n = None
        if ctx.needs_input_grad[0]:  # d loss / d image = A^T (grad_out)
            grad_data = _xrt.xrt_adjoint(ray, knot, order, g, mode=mode).torch()

        want_geom = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        if want_geom:
            if D != 2:
                raise NotImplementedError(
                    "Geometry gradients through the PyTorch wrapper are "
                    "implemented for 2D. For 3D, use the drjit operators "
                    "in xrt_toolkit.drjit.ray_xrt_new directly.")
            data_dr = Float(data.contiguous())
            if ctx.needs_input_grad[1]:
                gx = g * _xrt.xrt_ad_t_x(ray, knot, order, data_dr, mode=mode)
                gy = g * _xrt.xrt_ad_t_y(ray, knot, order, data_dr, mode=mode)
                grad_t = torch.stack([gx.torch(), gy.torch()], dim=0)
            if ctx.needs_input_grad[2]:
                gx = g * _xrt.xrt_ad_n_x(ray, knot, order, data_dr, mode=mode)
                gy = g * _xrt.xrt_ad_n_y(ray, knot, order, data_dr, mode=mode)
                grad_n = torch.stack([gx.torch(), gy.torch()], dim=0)

        return grad_data, grad_t, grad_n, None, None, None


def xrt_torch(data: torch.Tensor, t: torch.Tensor, n: torch.Tensor,
              knot_spec: UniformSpec, order: int,
              mode: str = "symbolic") -> torch.Tensor:
    r"""
    Differentiable X-ray projection for PyTorch.

    Parameters
    ----------
    data: torch.Tensor
        (Q1*...*QD,) flattened C-ordered image coefficients (CUDA tensor).
    t: torch.Tensor
        (D, M) ray anchors, one column per ray.
    n: torch.Tensor
        (D, M) ray directions.
    knot_spec: UniformSpec
        Image lattice.
    order: 0 | 1 | 2
        Basis order.
    mode: "symbolic" | "evaluated"
        Traversal mode of the underlying kernels. ``"symbolic"`` compiles one
        fused kernel and is what you want: it is two orders of magnitude faster
        than ``"evaluated"``, which launches a kernel per loop iteration
        (measured on 200k rays through a 256^2 lattice: 6 ms versus 814 ms for
        the forward, 8 ms versus 1.4 s for a geometry derivative, with
        identical results). ``"evaluated"`` is kept as an escape hatch.

    Returns
    -------
    proj: torch.Tensor
        (M,) projections, differentiable w.r.t. `data`, `t`, and `n`.
    """
    assert data.is_cuda and t.is_cuda and n.is_cuda, "inputs must be CUDA tensors"
    assert len(data) == math.prod(knot_spec.num)
    return _XRTFunction.apply(data, t, n, knot_spec, order, mode)


class XRTProjector(torch.nn.Module):
    r"""
    ``torch.nn.Module`` wrapper around :py:func:`xrt_torch`.

    Parameters
    ----------
    knot_spec: UniformSpec
        Image lattice.
    order: 0 | 1 | 2
        Basis order.
    mode: "symbolic" | "evaluated"
        See :py:func:`xrt_torch`; the default fuses the traversal into a single
        kernel.
    """

    def __init__(self, knot_spec: UniformSpec, order: int = 2,
                 mode: str = "symbolic"):
        super().__init__()
        self.knot_spec = knot_spec
        self.order = order
        self.mode = mode

    def forward(self, data: torch.Tensor, t: torch.Tensor,
                n: torch.Tensor) -> torch.Tensor:
        return xrt_torch(data, t, n, self.knot_spec, self.order, self.mode)
