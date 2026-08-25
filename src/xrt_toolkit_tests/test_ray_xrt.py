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


@pytest.mark.parametrize("surface", ["diagonal", "axis"])
@pytest.mark.parametrize("order", [1, 2])
def test_ad_n_tie_rays(order, surface):
    # d/dn for rays exactly on the model's branch surfaces: the lattice
    # diagonal (|n_x| == |n_y|, rays at 45 deg) and the lattice axis
    # (min(|n_x|, |n_y|) ~ 0, e.g. vertical rays -- note cos(pi/2) is 6e-17,
    # not 0, in f64).  The projection model is continuous but kinked across
    # both surfaces, so the derivative must return the two-sided
    # (branch-averaged) slope; the one-sided slope it used to return is
    # wrong by up to O(1) (~0.1-0.2 median relative on this geometry).
    # f64: in fp32 both the FD reference and near-surface evaluations are
    # noise-limited at the ~10% level, which hides exactly this defect.
    from drjit.cuda.ad import Array2f64, Float64

    rng = np.random.default_rng(2)
    N, L = 64, 200
    knot = xtk.UniformSpec.centered(step=1.0, num=(N, N))
    ax = np.stack(np.meshgrid(np.arange(N), np.arange(N), indexing="ij"))
    ax = ax - (N - 1) / 2
    vol = np.zeros((N, N))
    for _ in range(6):
        cx, cy = rng.uniform(-N / 4, N / 4, 2)
        sg = rng.uniform(N / 14, N / 8)
        vol += rng.uniform(0.5, 1.5) * np.exp(
            -((ax[0] - cx) ** 2 + (ax[1] - cy) ** 2) / (2 * sg**2))
    V = Float64(vol.ravel())
    # perturb the direction component transverse to the ray (the rotational
    # part -- the radial part is exactly zero for this scale-invariant model)
    th, comp, ad_fn = {
        "diagonal": (np.deg2rad(45.0), 1, xtk.xrt_ad_n_y),
        "axis": (np.pi / 2, 0, xtk.xrt_ad_n_x),
    }[surface]
    off = rng.uniform(-N / 2.5, N / 2.5, L)
    t = np.stack([-N * np.cos(th) - off * np.sin(th),
                  -N * np.sin(th) + off * np.cos(th)])
    n0 = np.stack([np.full(L, np.cos(th)), np.full(L, np.sin(th))])

    def val(eps):
        n = n0.copy()
        n[comp] += eps
        return np.asarray(xtk.xrt_apply(
            (Array2f64(t), Array2f64(n)), knot, order, V, mode="evaluated"))

    g_ad = np.asarray(ad_fn(
        (Array2f64(t), Array2f64(n0)), knot, order, V, mode="evaluated"))

    # Central FD reference at the kink.  The order-2 forward additionally
    # carries a small value jump J across the diagonal (a separate forward
    # defect); FD picks it up as J/(2h), so estimate J by Richardson
    # extrapolation (dP(h) = J + 2h*slope) and remove it.  (J ~ 0 across
    # the axis; the correction is then a no-op.)
    h, h1, h2 = 1e-5, 1e-8, 1e-7
    J = (h2 * (val(+h1) - val(-h1)) - h1 * (val(+h2) - val(-h2))) / (h2 - h1)
    g_fd = (val(+h) - val(-h) - J) / (2 * h)

    keep = np.abs(g_fd) > 1e-3 * np.abs(g_fd).max()
    rel = np.abs(g_ad[keep] - g_fd[keep]) / np.abs(g_fd[keep])
    assert float(np.median(rel)) < 0.05

def test_ad_n_3d_voxel_matches_fd():
    # d/dn for the 3D order-0 closed form (vox2) against f64 central FD, at
    # a non-unit |n| (cone_beam emits |n| ~ sdd).  Locks two contracts: the
    # derivative matches the shipped geometric (scale-invariant) forward --
    # no alpha-parameterization radial term -P n / |n|^2 -- and the
    # face-transfer terms carry the correct |n| scaling.
    from drjit.cuda.ad import Array3f64, Float64

    rng = np.random.default_rng(7)
    N, L = 24, 200
    knot = xtk.UniformSpec.centered(step=1.0, num=(N, N, N))
    ax = np.stack(np.meshgrid(*(np.arange(N) - (N - 1) / 2,) * 3,
                              indexing="ij"))
    vol = np.zeros((N, N, N))
    for _ in range(5):
        c = rng.uniform(-N / 5, N / 5, 3)
        sg = rng.uniform(N / 12, N / 7)
        vol += rng.uniform(0.5, 1.5) * np.exp(
            -(((ax - c[:, None, None, None]) ** 2).sum(0)) / (2 * sg**2))
    V = Float64(vol.ravel())
    nv = np.array([0.31, 0.53, 0.79])
    nv /= np.linalg.norm(nv)
    u = np.cross(nv, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    w = np.cross(nv, u)
    t = (-1.5 * N) * nv[:, None] \
        + u[:, None] * (rng.uniform(-N / 3, N / 3, L) + 0.17) \
        + w[:, None] * (rng.uniform(-N / 3, N / 3, L) + 0.29)
    n0 = np.repeat(2.0 * nv[:, None], L, axis=1)  # |n| = 2, deliberately

    def val(eps):
        n = n0.copy()
        n[0] += eps
        return np.asarray(xtk.xrt_apply(
            (Array3f64(t), Array3f64(n)), knot, 0, V, mode="evaluated"))

    g_ad = np.asarray(xtk.xrt_ad_n_x(
        (Array3f64(t), Array3f64(n0)), knot, 0, V, mode="evaluated"))
    h = 1e-5
    g_fd = (val(+h) - val(-h)) / (2 * h)
    keep = np.abs(g_fd) > 1e-3 * np.abs(g_fd).max()
    rel = np.abs(g_ad[keep] - g_fd[keep]) / np.abs(g_fd[keep])
    assert float(np.median(rel)) < 0.05


def test_spline3d_fallback_matches_coopvec():
    # The portable (no cooperative vectors) evaluation of the 3D spline
    # network must agree with the tensor-core path to fp16 accuracy.
    from drjit.cuda import Array2f
    from drjit.cuda import Float as Fl

    import xrt_toolkit.drjit.box_spline as bs

    if not bs.coop_vec_available(__import__(
            "xrt_toolkit.drjit.ray_xrt", fromlist=["net"]).net):
        pytest.skip("cooperative vectors unavailable on this system")

    rng = np.random.default_rng(7)
    x, z = rng.uniform(-1.5, 1.5, (2, 4096)).astype(np.float32)
    th = rng.uniform(0, np.pi, 4096).astype(np.float32)
    n = Array2f(np.cos(th), np.sin(th))
    net = __import__("xrt_toolkit.drjit.ray_xrt", fromlist=["net"]).net
    a = np.asarray(bs.nn_project(net, Fl(x), Fl(z), n))
    b = np.asarray(bs.nn_project_plain(Fl(x), Fl(z), n))
    assert np.abs(a - b).max() < 5e-3
