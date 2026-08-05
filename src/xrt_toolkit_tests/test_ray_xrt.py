"""
Tests for the core scalar operators in ``xrt_toolkit.drjit.ray_xrt`` and the
geometry derivatives in ``xrt_toolkit.drjit.ray_xrt_new``.

Requires a CUDA device (the drjit backend of the library).
"""

import numpy as np
import pytest

drjit = pytest.importorskip("drjit")
import drjit as dr  # noqa: E402

try:
    from drjit.cuda.ad import Array2f, Array3f, Float  # noqa: E402

    dr.width(Float(0.0))  # touch the backend
    CUDA_OK = True
except Exception:
    CUDA_OK = False

pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA backend unavailable")

import xrt_toolkit as xtk  # noqa: E402


def _rays(D, N, L, rng):
    t = rng.uniform(4, N - 4, (D, L)).astype(np.float32)
    n = rng.normal(size=(D, L)).astype(np.float32)
    n /= np.linalg.norm(n, axis=0)
    A = Array2f if D == 2 else Array3f
    return (A(t), A(n))


def _maxrel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() / (np.abs(a).max() + 1e-12)


# ------------------------------------------------------------- forward -------
@pytest.mark.parametrize("D,N", [(2, 64), (3, 24)])
def test_constant_volume_chord(D, N):
    # On a constant volume the order-0 transform returns the chord length of
    # the ray inside the lattice bounding box.  Check axis-aligned rays whose
    # chord is known exactly.
    knot = xtk.UniformSpec(start=0, step=1, num=(N,) * D)
    L = 32
    t = np.full((D, L), N / 2, dtype=np.float32)
    t[1] = np.linspace(8, N - 8, L)
    n = np.zeros((D, L), dtype=np.float32)
    n[0] = 1.0
    A = Array2f if D == 2 else Array3f
    vol = Float(np.ones(N**D, dtype=np.float32))
    y = np.asarray(xtk.xrt_apply((A(t), A(n)), knot, 0, vol, mode="evaluated"))
    assert np.abs(y - N).max() / N < 1e-5


@pytest.mark.parametrize("D,N", [(2, 64), (3, 24)])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_forward_linear(D, N, order):
    # A(f + a g) == A(f) + a A(g).
    rng = np.random.default_rng(3)
    knot = xtk.UniformSpec(start=0, step=1, num=(N,) * D)
    ray = _rays(D, N, 200, rng)
    f = Float(rng.random(N**D, dtype=np.float32))
    g = Float(rng.random(N**D, dtype=np.float32))

    y_sum = xtk.xrt_apply(ray, knot, order, f + 2.5 * g, mode="evaluated")
    y_lin = (np.asarray(xtk.xrt_apply(ray, knot, order, f, mode="evaluated"))
             + 2.5 * np.asarray(xtk.xrt_apply(ray, knot, order, g, mode="evaluated")))
    assert _maxrel(y_sum, y_lin) < 1e-5


@pytest.mark.parametrize("D,N", [(2, 64), (3, 24)])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_adjoint_dot(D, N, order):
    # <A f, y> == <f, A^T y> (non-negative operands keep fp32 cancellation
    # benign; tolerance covers atomic-order variation).
    rng = np.random.default_rng(1)
    L = 500
    knot = xtk.UniformSpec(start=0, step=1, num=(N,) * D)
    ray = _rays(D, N, L, rng)
    f = Float(rng.random(N**D, dtype=np.float32))
    y = Float(rng.random(L, dtype=np.float32))

    Af = np.asarray(xtk.xrt_apply(ray, knot, order, f, mode="evaluated"), np.float64)
    Aty = np.asarray(xtk.xrt_adjoint(ray, knot, order, y, mode="evaluated"), np.float64)
    d1 = float(Af @ np.asarray(y, np.float64))
    d2 = float(np.asarray(f, np.float64) @ Aty)
    assert abs(d1 - d2) / abs(d1) < 5e-4


@pytest.mark.parametrize("order", [0, 1, 2])
def test_symbolic_matches_evaluated(order):
    rng = np.random.default_rng(2)
    N, L = 64, 300
    knot = xtk.UniformSpec(start=0, step=1, num=(N, N))
    ray = _rays(2, N, L, rng)
    f = Float(rng.random(N * N, dtype=np.float32))
    y_e = xtk.xrt_apply(ray, knot, order, f, mode="evaluated")
    y_s = xtk.xrt_apply(ray, knot, order, f, mode="symbolic")
    assert _maxrel(y_e, y_s) < 1e-6


# ------------------------------------------- geometry derivatives (AD) -------
def _fd_median_rel(fn_val, fn_grad, h=1e-2):
    """Median relative deviation between the AD gradient and central FD."""
    g_ad = np.asarray(fn_grad(), np.float64)
    g_fd = (np.asarray(fn_val(+h), np.float64)
            - np.asarray(fn_val(-h), np.float64)) / (2 * h)
    keep = np.abs(g_fd) > 1e-3 * np.abs(g_fd).max()
    rel = np.abs(g_ad[keep] - g_fd[keep]) / np.abs(g_fd[keep])
    return float(np.median(rel))


def _smooth_volume(D, N):
    ax = np.stack(np.meshgrid(*(np.arange(N),) * D, indexing="ij"))
    r2 = sum((a - N / 2) ** 2 for a in ax)
    return Float(np.exp(-r2 / (0.1 * N * N)).reshape(-1).astype(np.float32))


@pytest.mark.parametrize("D,N,order", [(2, 64, 1), (2, 64, 2), (3, 24, 0)])
def test_ad_t_matches_fd(D, N, order):
    # d/d(t_x): AD against central finite differences on a smooth volume.
    # fp32 + FD noise: require median agreement within 5%.
    rng = np.random.default_rng(4)
    L = 400
    knot = xtk.UniformSpec(start=0, step=1, num=(N,) * D)
    vol = _smooth_volume(D, N)
    t0 = rng.uniform(4, N - 4, (D, L)).astype(np.float32)
    n = rng.normal(size=(D, L)).astype(np.float32)
    n /= np.linalg.norm(n, axis=0)
    A = Array2f if D == 2 else Array3f

    def val(eps):
        t = t0.copy()
        t[0] += eps
        return xtk.xrt_apply((A(t), A(n)), knot, order, vol, mode="evaluated")

    def grad():
        return xtk.xrt_ad_t_x((A(t0), A(n)), knot, order, vol, mode="evaluated")

    assert _fd_median_rel(val, grad) < 0.05


@pytest.mark.parametrize("order", [1, 2])
def test_ad_n_matches_fd(order):
    # d/d(n_x) in 2D.  This is the derivative fixed in v2: the AD callback
    # must differentiate local copies of the loop state.
    rng = np.random.default_rng(5)
    N, L = 64, 400
    knot = xtk.UniformSpec(start=0, step=1, num=(N, N))
    vol = _smooth_volume(2, N)
    t = rng.uniform(4, N - 4, (2, L)).astype(np.float32)
    n0 = rng.normal(size=(2, L)).astype(np.float32)
    n0 /= np.linalg.norm(n0, axis=0)

    def val(eps):
        n = n0.copy()
        n[0] += eps
        return xtk.xrt_apply((Array2f(t), Array2f(n)), knot, order, vol,
                             mode="evaluated")

    def grad():
        return xtk.xrt_ad_n_x((Array2f(t), Array2f(n0)), knot, order, vol,
                              mode="evaluated")

    assert _fd_median_rel(val, grad) < 0.05
