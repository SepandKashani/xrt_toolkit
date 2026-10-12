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


def finite_differences(f, knot_spec, t, n, h) -> tuple:
    # central differences of the projections in each anchor/direction component
    D = t.shape[0]
    fd_t, fd_n = np.zeros(t.shape), np.zeros(n.shape)
    for k in range(D):
        e = h * np.eye(D)[:, [k]]
        fd_t[k] = projections(f, knot_spec, t + e, n) - projections(
            f, knot_spec, t - e, n
        )
        fd_n[k] = projections(f, knot_spec, t, n + e) - projections(
            f, knot_spec, t, n - e
        )
    return fd_t / (2 * h), fd_n / (2 * h)


def grad(f, knot_spec, t, n, precision: str) -> tuple:
    D = t.shape[0]
    if precision == "float64":
        ArrayNf, Float_ = (Array2f64 if (D == 2) else Array3f64), Float64
    else:
        ArrayNf, Float_ = (Array2f if (D == 2) else Array3f), Float
    dtype = np.float64 if (precision == "float64") else np.float32
    g_t, g_n = xtk.xrt_grad(
        (ArrayNf(t.astype(dtype)), ArrayNf(n.astype(dtype))),
        knot_spec,
        0,
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


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_grad(knot_spec, precision):
    # Rays through random points of the volume, anchored up to 15 before or after them, with
    # directions of random norm: gradients match central differences.
    D = knot_spec.ndim
    rng = np.random.default_rng(0)
    L = 300
    lo = np.array(knot_spec.start)
    hi = lo + np.array(knot_spec.step) * (np.array(knot_spec.num) - 1)
    x = rng.uniform(lo[:, None], hi[:, None], (D, L))
    n = rng.standard_normal((D, L)) * rng.uniform(0.5, 2, L)
    t = x - rng.uniform(-15, 15, L) * n / np.linalg.norm(n, axis=0)
    if precision == "float32":
        t, n = (
            t.astype(np.float32).astype(np.float64),
            n.astype(np.float32).astype(np.float64),
        )
    f = rng.standard_normal(knot_spec.num)

    g_t, g_n = grad(f, knot_spec, t, n, precision)
    fd_t, fd_n = finite_differences(f, knot_spec, t, n, 1e-6)
    tol = 1e-5 if (precision == "float64") else 3e-5
    assert np.abs(g_t - fd_t).max() < tol * np.abs(fd_t).max()
    assert np.abs(g_n - fd_n).max() < tol * np.abs(fd_n).max()

    # the projection does not change along the line, nor with the norm of n
    assert np.abs(np.sum(g_t * n, axis=0)).max() < tol * np.abs(fd_t).max()
    assert np.abs(np.sum(g_n * n, axis=0)).max() < tol * np.abs(fd_n).max()


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
