import drjit as dr
import numpy as np
import pytest

import xrt_toolkit as xtk

try:
    from drjit.cuda import Array2f, Array2f64, Array3f, Array3f64, Float, Float64

    dr.width(Float(0))
except Exception:
    from drjit.llvm import Array2f, Array2f64, Array3f, Array3f64, Float, Float64


def projections(f, knot_spec, t, n) -> np.ndarray:
    # Order-0 projections in float64, independently of XTK: each line is split at every voxel
    # plane, the field is constant on each segment.
    start, step, num = map(np.array, (knot_spec.start, knot_spec.step, knot_spec.num))
    ll = start - step / 2
    out = np.zeros(t.shape[1])
    for l in range(t.shape[1]):
        u = n[:, l] / np.linalg.norm(n[:, l])
        a = [
            (ll[d] + np.arange(num[d] + 1) * step[d] - t[d, l]) / u[d]
            for d in range(len(step))
            if u[d] != 0
        ]
        a = np.sort(np.concatenate(a))
        mid = 0.5 * (a[:-1] + a[1:])
        q = np.floor((t[:, l, None] + mid * u[:, None] - ll[:, None]) / step[:, None])
        inside = np.all((q >= 0) & (q < num[:, None]), axis=0)
        val = np.zeros(len(mid))
        val[inside] = f[tuple(q[:, inside].astype(int))] / np.prod(step)
        out[l] = np.sum(val * np.diff(a))
    return out


GL_X, GL_W = np.polynomial.legendre.leggauss(16)


def beta(k: int, x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    if k == 1:
        return np.clip(1 - a, 0, None)
    return np.where(a <= 0.5, 0.75 - a**2, np.where(a <= 1.5, 0.5 * (1.5 - a) ** 2, 0))


def spline_projections(f, knot_spec, order, t, n) -> np.ndarray:
    # 3D order-1/2 projections in float64, independently of XTK: each line is split at the
    # knot planes of all basis functions, and each polynomial piece is integrated exactly
    # (16-node Gauss-Legendre).
    start, step, num = map(np.array, (knot_spec.start, knot_spec.step, knot_spec.num))
    shift = 0.0 if (order == 1) else 0.5  # knot planes: at the knots, or between them
    out = np.zeros(t.shape[1])
    for l in range(t.shape[1]):
        u = n[:, l] / np.linalg.norm(n[:, l])
        a = [
            (start[d] + (np.arange(-2, num[d] + 2) - shift) * step[d] - t[d, l]) / u[d]
            for d in range(3)
            if u[d] != 0
        ]
        a = np.sort(np.concatenate(a))
        lo, hi = a[:-1], a[1:]
        aq = (0.5 * (lo + hi))[:, None] + (0.5 * (hi - lo))[:, None] * GL_X
        x = (t[:, l, None, None] + aq[None] * u[:, None, None]).reshape(3, -1)

        # field at x: sum over the 4**3 nearest basis functions
        v = (x - start[:, None]) / step[:, None]
        q0 = np.floor(v).astype(int) - 1
        F = np.zeros(x.shape[1])
        for c in np.ndindex(4, 4, 4):
            q = q0 + np.array(c)[:, None]
            w = np.prod([beta(order, v[d] - q[d]) for d in range(3)], axis=0)
            inside = np.all((q >= 0) & (q < num[:, None]), axis=0)
            qc = np.clip(q, 0, (num - 1)[:, None])
            F += np.where(inside, w * f[qc[0], qc[1], qc[2]], 0)
        F = F.reshape(aq.shape) / np.prod(step)
        out[l] = np.sum(F * GL_W * (0.5 * (hi - lo))[:, None])
    return out


def finite_differences(f, knot_spec, t, n, h, order=0) -> tuple:
    # central differences of the projections in each anchor/direction component
    D = t.shape[0]
    if order == 0:
        P = lambda t_, n_: projections(f, knot_spec, t_, n_)  # noqa: E731
    else:
        P = lambda t_, n_: spline_projections(f, knot_spec, order, t_, n_)  # noqa: E731
    fd_t, fd_n = np.zeros(t.shape), np.zeros(n.shape)
    for k in range(D):
        e = h * np.eye(D)[:, [k]]
        fd_t[k] = P(t + e, n) - P(t - e, n)
        fd_n[k] = P(t, n + e) - P(t, n - e)
    return fd_t / (2 * h), fd_n / (2 * h)


def grad(f, knot_spec, t, n, precision: str, order: int = 0) -> tuple:
    D = t.shape[0]
    if precision == "float64":
        ArrayNf, Float_ = (Array2f64 if (D == 2) else Array3f64), Float64
    else:
        ArrayNf, Float_ = (Array2f if (D == 2) else Array3f), Float
    dtype = np.float64 if (precision == "float64") else np.float32
    g_t, g_n = xtk.xrt_grad(
        (ArrayNf(t.astype(dtype)), ArrayNf(n.astype(dtype))),
        knot_spec,
        order,
        Float_(f.ravel().astype(dtype)),
    )
    return np.asarray(g_t, np.float64), np.asarray(g_n, np.float64)


@pytest.fixture(params=[2, 3])
def knot_spec(request) -> xtk.UniformSpec:
    D = request.param
    return xtk.UniformSpec(
        start=(-1.1, 0.3, 2.2)[:D],
        step=(0.5, 1.0, 0.75)[:D],
        num=(14, 11, 9)[:D],
    )


def random_rays(knot_spec, L: int, rng, precision: str) -> tuple:
    # Rays through random points of the volume, anchored up to 15 before or after them, with
    # directions of random norm.
    D = knot_spec.ndim
    lo = np.array(knot_spec.start)
    hi = lo + np.array(knot_spec.step) * (np.array(knot_spec.num) - 1)
    x = rng.uniform(lo[:, None], hi[:, None], (D, L))
    n = rng.standard_normal((D, L)) * rng.uniform(0.5, 2, L)
    t = x - rng.uniform(-15, 15, L) * n / np.linalg.norm(n, axis=0)
    if precision == "float32":
        t, n = t.astype(np.float32), n.astype(np.float32)
    return t.astype(np.float64), n.astype(np.float64)


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_grad(knot_spec, precision):
    # order 0: gradients match central differences.
    rng = np.random.default_rng(0)
    t, n = random_rays(knot_spec, 300, rng, precision)
    f = rng.standard_normal(knot_spec.num)

    g_t, g_n = grad(f, knot_spec, t, n, precision)
    fd_t, fd_n = finite_differences(f, knot_spec, t, n, 1e-6)
    tol = 1e-5 if (precision == "float64") else 3e-5
    assert np.abs(g_t - fd_t).max() < tol * np.abs(fd_t).max()
    assert np.abs(g_n - fd_n).max() < tol * np.abs(fd_n).max()

    # the projection does not change along the line, nor with the norm of n
    assert np.abs(np.sum(g_t * n, axis=0)).max() < tol * np.abs(fd_t).max()
    assert np.abs(np.sum(g_n * n, axis=0)).max() < tol * np.abs(fd_n).max()


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_grad_splines3d(order, precision):
    # 3D orders 1, 2: gradients match central differences.
    knot_spec = xtk.UniformSpec(
        start=(-1.1, 0.3, 2.2), step=(0.5, 1.0, 0.75), num=(9, 8, 7)
    )
    rng = np.random.default_rng(2)
    t, n = random_rays(knot_spec, 60, rng, precision)
    f = rng.standard_normal(knot_spec.num)

    g_t, g_n = grad(f, knot_spec, t, n, precision, order)
    fd_t, fd_n = finite_differences(f, knot_spec, t, n, 1e-6, order)
    assert np.abs(g_t - fd_t).max() < 2e-5 * np.abs(fd_t).max()
    assert np.abs(g_n - fd_n).max() < 2e-5 * np.abs(fd_n).max()
    assert np.abs(np.sum(g_t * n, axis=0)).max() < 2e-5 * np.abs(fd_t).max()
    assert np.abs(np.sum(g_n * n, axis=0)).max() < 2e-5 * np.abs(fd_n).max()


@pytest.mark.parametrize("order", [1, 2])
def test_grad_splines3d_planes(order):
    # 3D orders 1, 2, rays lying in knot planes (as in parallel beam with detector rows on the
    # knots): at order 1 the normal derivative of f jumps there, and the gradient is the
    # average of both sides, as central differences.
    knot_spec = xtk.UniformSpec(
        start=(-3.0, -2.0, -3.0), step=(1.0, 1.0, 1.0), num=(7, 5, 7)
    )
    rng = np.random.default_rng(6)
    L = 40
    th = rng.uniform(0, 2 * np.pi, L)
    n = np.stack([np.cos(th), np.sin(th), np.zeros(L)])  # in z = const planes
    n[:, : L // 4] = [[1], [0], [0]]  # and in y = const planes
    t = np.stack([rng.uniform(-2, 2, L), rng.uniform(-1, 1, L), rng.integers(-3, 4, L)])
    t[1, : L // 4] = rng.integers(-2, 3, L // 4)
    t[:2] -= rng.uniform(-10, 10, L) * n[:2]
    f = rng.standard_normal(knot_spec.num)

    g_t, g_n = grad(f, knot_spec, t, n, "float64", order)
    fd_t, fd_n = finite_differences(f, knot_spec, t, n, 1e-7, order)
    assert np.abs(g_t - fd_t).max() < 1e-5 * np.abs(fd_t).max()
    assert np.abs(g_n - fd_n).max() < 1e-5 * np.abs(fd_n).max()


@pytest.mark.parametrize(
    "n",
    [(1, 1), (1, -1), (2, 1), (1, 1, 1), (1, -1, 2), (2, 0, 1)],
)
@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_grad_ties(n, precision):
    # Rays through lattice corners, crossing several faces at once: gradients match central
    # differences, i.e. the average of the one-sided derivatives.
    D = len(n)
    rng = np.random.default_rng(1)
    num = (6, 5, 7)[:D]
    knot_spec = xtk.UniformSpec(
        start=(0.5,) * D, step=(1.0,) * D, num=num
    )  # faces on integers
    L = 64
    n = np.tile(np.array(n, dtype=float)[:, None], (1, L))
    t = rng.integers(1, min(num), (D, L)) - rng.integers(-3, 4, L) * n
    t[n == 0] += 0.3  # not on a face the ray runs along
    f = rng.standard_normal(num)

    g_t, g_n = grad(f, knot_spec, t, n, precision)
    fd_t, fd_n = finite_differences(f, knot_spec, t, n, 1e-7)
    assert np.abs(g_t - fd_t).max() < 1e-5 * np.abs(fd_t).max()
    assert np.abs(g_n - fd_n).max() < 1e-5 * np.abs(fd_n).max()


def test_grad_miss():
    # rays missing the volume, also within one cell of it, have zero gradients
    knot_spec = xtk.UniformSpec(start=(0.5, 0.5), step=(1.0, 1.0), num=(4, 4))
    t = np.array([[-2.0, -0.5, 0.0, 9.0], [4.5, 0.0, -1.0, 9.0]])
    n = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, -1.0, 0.0, -0.2]])
    f = np.ones((4, 4))
    g_t, g_n = grad(f, knot_spec, t, n, "float32")
    assert np.all(g_t == 0) and np.all(g_n == 0)
