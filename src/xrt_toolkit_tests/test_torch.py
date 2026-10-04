"""
Tests for the PyTorch bindings in ``xrt_toolkit.torch``.

Requires a CUDA device and PyTorch.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
drjit = pytest.importorskip("drjit")
import drjit as dr  # noqa: E402

try:
    from drjit.cuda.ad import Float  # noqa: E402

    dr.width(Float(0.0))
    CUDA_OK = torch.cuda.is_available()
except Exception:
    CUDA_OK = False

pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA backend unavailable")

import xrt_toolkit as xtk  # noqa: E402

N, M = 64, 2000


def _setup(order=1, mode="symbolic"):
    from xrt_toolkit.torch import XRTProjector

    yy, xx = np.mgrid[:N, :N]
    img = ((((xx - N/2) / 18) ** 2 + ((yy - N/2) / 24) ** 2) < 1).astype(np.float32)
    knot = xtk.UniformSpec(start=(-N/2 + 0.5,) * 2, step=1, num=(N, N))
    rng = np.random.default_rng(0)
    t = torch.tensor(rng.uniform(-24, 24, (2, M)), dtype=torch.float32, device="cuda")
    d = rng.normal(size=(2, M))
    d /= np.linalg.norm(d, axis=0)
    n = torch.tensor(d, dtype=torch.float32, device="cuda")
    return XRTProjector(knot, order=order, mode=mode), img, t, n


def _round_trip(proj, img, t0, n0):
    x = torch.tensor(img.reshape(-1), device="cuda", requires_grad=True)
    t = torch.nn.Parameter(t0.clone())
    n = torch.nn.Parameter(n0.clone())
    y = proj(x, t, n)
    (y ** 2).mean().backward()
    return (y.detach().cpu().numpy(), x.grad.cpu().numpy(),
            t.grad.cpu().numpy(), n.grad.cpu().numpy())


def test_gradients_populated():
    proj, img, t, n = _setup()
    y, gx, gt, gn = _round_trip(proj, img, t, n)
    assert y.shape == (M,)
    for g in (gx, gt, gn):
        assert np.isfinite(g).all()
        assert np.abs(g).max() > 0


def test_default_mode_is_symbolic():
    # symbolic fuses the traversal into one kernel; evaluated launches one per
    # step and is ~100x slower, so the default must not regress to it
    from xrt_toolkit.torch import XRTProjector, xrt_torch
    import inspect

    assert XRTProjector(xtk.UniformSpec(start=0, step=1, num=(4, 4))).mode == "symbolic"
    assert inspect.signature(xrt_torch).parameters["mode"].default == "symbolic"


def test_modes_agree():
    ref = _round_trip(*(_setup(mode="symbolic")))
    alt = _round_trip(*(_setup(mode="evaluated")))
    for a, b, name in zip(ref, alt, ("y", "grad_image", "grad_t", "grad_n")):
        dev = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
        assert dev < 1e-4, f"{name}: {dev:.2e}"


def test_image_gradient_matches_adjoint():
    # d loss / d image must be exactly A^T applied to the upstream gradient
    proj, img, t, n = _setup(order=1)
    x = torch.tensor(img.reshape(-1), device="cuda", requires_grad=True)
    y = proj(x, t, n)
    g_up = torch.rand(M, device="cuda")
    y.backward(g_up)
    rays = (dr.cuda.ad.Array2f(np.ascontiguousarray(t.cpu().numpy())),
            dr.cuda.ad.Array2f(np.ascontiguousarray(n.cpu().numpy())))
    want = np.asarray(xtk.xrt_adjoint(rays, proj.knot_spec, 1,
                                      Float(g_up.cpu().numpy())))
    got = x.grad.cpu().numpy()
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-4
