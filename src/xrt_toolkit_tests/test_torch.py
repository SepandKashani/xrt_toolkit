import math

import numpy as np
import pytest

import xrt_toolkit as xtk

torch = pytest.importorskip("torch")
from xrt_toolkit.torch import XRTProjector, xrt_torch  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def problem(D: int, device: str, dtype=torch.float32, L: int = 500) -> tuple:
    # random volume, rays through it (random anchors, directions of random norm)
    rng = np.random.default_rng(0)
    knot_spec = xtk.UniformSpec(
        start=(-1.1, 0.3, 2.2)[:D], step=(0.5, 1.0, 0.75)[:D], num=(14, 11, 9)[:D]
    )
    lo = np.array(knot_spec.start)
    hi = lo + np.array(knot_spec.step) * (np.array(knot_spec.num) - 1)
    x = rng.uniform(lo[:, None], hi[:, None], (D, L))
    n = rng.standard_normal((D, L)) * rng.uniform(0.5, 2, L)
    t = x - rng.uniform(-15, 15, L) * n / np.linalg.norm(n, axis=0)
    f = rng.standard_normal(math.prod(knot_spec.num))
    T = lambda a: torch.tensor(a, dtype=dtype, device=device)  # noqa: E731
    return knot_spec, T(f), T(t), T(n)


def as_drjit(knot_spec, f, t, n):
    # the same inputs as Dr.Jit arrays, through numpy
    import drjit.cuda as cuda
    import drjit.llvm as llvm

    backend = cuda if f.is_cuda else llvm
    D = knot_spec.ndim
    ArrayNf, Float = getattr(backend, f"Array{D}f"), backend.Float
    c = lambda v: v.detach().cpu().numpy()  # noqa: E731
    return (ArrayNf(c(t)), ArrayNf(c(n))), Float(c(f))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("D", [2, 3])
def test_xrt_torch(D, order, device):
    # forward = xrt_apply; backward = xrt_adjoint for the volume, xrt_grad for the rays
    knot_spec, f, t, n = problem(D, device)
    rays, f_dr = as_drjit(knot_spec, f, t, n)
    w = torch.tensor(
        np.random.default_rng(1).standard_normal(t.shape[1]),
        dtype=f.dtype,
        device=device,
    )

    f, t, n = (v.clone().requires_grad_() for v in (f, t, n))
    y = xrt_torch(f, t, n, knot_spec, order)
    (w * y).sum().backward()

    y_gt = np.asarray(xtk.xrt_apply(rays, knot_spec, order, f_dr))
    f_gt = np.asarray(
        xtk.xrt_adjoint(rays, knot_spec, order, type(f_dr)(w.cpu().numpy()))
    )
    g_t, g_n = (
        np.asarray(g) * w.cpu().numpy()
        for g in xtk.xrt_grad(rays, knot_spec, order, f_dr)
    )
    for a, b in ((y, y_gt), (f.grad, f_gt), (t.grad, g_t), (n.grad, g_n)):
        a = a.detach().cpu().numpy()
        assert np.abs(a - b).max() <= 1e-5 * np.abs(b).max()


@pytest.mark.parametrize("device", DEVICES)
def test_gradcheck(device):
    # torch's own check against finite differences (float64): volume and anchors
    knot_spec, f, t, n = problem(2, device, torch.float64, L=20)
    f, t = f.requires_grad_(), t.requires_grad_()
    fn = lambda f, t: xrt_torch(f, t, n, knot_spec, 2)  # noqa: E731
    assert torch.autograd.gradcheck(fn, (f, t), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_module():
    knot_spec, f, t, n = problem(3, DEVICES[-1])
    proj = XRTProjector(knot_spec, order=1)
    assert torch.equal(proj(f, t, n), xrt_torch(f, t, n, knot_spec, 1))
