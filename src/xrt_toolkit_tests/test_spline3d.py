import itertools

import drjit as dr
import numpy as np
import pytest

import xrt_toolkit as xtk

try:
    from drjit.cuda import Array3f, Float

    dr.width(Float(0))
except Exception:
    from drjit.llvm import Array3f, Float

KNOTS = {  # knot positions of beta^k
    0: [-0.5, 0.5],
    1: [-1.0, 0.0, 1.0],
    2: [-1.5, -0.5, 0.5, 1.5],
}
GL_X, GL_W = np.polynomial.legendre.leggauss(8)


def beta(k: int, x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    if k == 0:
        return (a <= 0.5) * 1.0
    elif k == 1:
        return np.clip(1 - a, 0, None)
    elif k == 2:
        return np.where(
            a <= 0.5, 0.75 - a**2, np.where(a <= 1.5, 0.5 * (1.5 - a) ** 2, 0)
        )


def exact(
    k: int, t: np.ndarray, n: np.ndarray, center: np.ndarray, step: np.ndarray
) -> np.ndarray:
    # Line integrals / prod(step) of the basis function at `center`, in float64:
    # rays are split at the knot planes and each polynomial piece is integrated exactly.
    m = (
        n / np.linalg.norm(n, axis=1, keepdims=True) / step
    )  # direction in lattice units
    m_norm = np.linalg.norm(m, axis=1, keepdims=True)
    d = m / m_norm
    p = (t - center) / step
    h = KNOTS[k][-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ta, tb = (-h - p) / d, (h - p) / d
        lo = np.where(d != 0, np.minimum(ta, tb), -np.inf).max(1)
        hi = np.where(d != 0, np.maximum(ta, tb), np.inf).min(1)
        hit = np.all((d != 0) | (np.abs(p) <= h), 1) & (hi > lo)
        lo, hi = np.where(hit, lo, 0), np.where(hit, hi, 0)
        tk = (np.array(KNOTS[k])[None, None] - p[..., None]) / d[..., None]
    tk = np.where(d[..., None] != 0, tk, lo[:, None, None]).reshape(len(p), -1)
    tt = np.sort(np.c_[lo, np.clip(tk, lo[:, None], hi[:, None]), hi], 1)
    a, b = tt[:, :-1], tt[:, 1:]  # (R, pieces)
    tq = (0.5 * (a + b))[..., None] + (0.5 * (b - a))[..., None] * GL_X
    x = p[:, None, None, :] + tq[..., None] * d[:, None, None, :]
    f = beta(k, x[..., 0]) * beta(k, x[..., 1]) * beta(k, x[..., 2])
    return np.einsum("rmq,q,rm->r", f, GL_W, 0.5 * (b - a)) / (
        m_norm[:, 0] * step.prod()
    )


def directions(rng) -> np.ndarray:
    # Fibonacci sphere, the 26 axes / face / body diagonals, and their perturbations.
    N = 2000
    i = np.arange(N) + 0.5
    z = 1 - 2 * i / N
    phi = np.pi * (3 - np.sqrt(5)) * i
    fib = np.stack(
        [np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z], 1
    )
    diag = np.array(
        [v for v in itertools.product((-1, 0, 1), repeat=3) if any(v)], float
    )
    diag /= np.linalg.norm(diag, axis=1, keepdims=True)
    pert = [diag + eps * rng.standard_normal(diag.shape) for eps in (1e-2, 1e-4, 1e-6)]
    dirs = np.concatenate([fib, diag, *pert])
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def rays(
    dirs: np.ndarray, center: np.ndarray, step: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # 8 rays per direction, offset from `center` on a spiral covering the basis support.
    a = np.where(np.abs(dirs[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
    u = a - (a * dirs).sum(1, keepdims=True) * dirs
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.cross(dirs, u)
    j = np.arange(8)
    r = (0.15 + 0.3 * j) * step.max()
    phi = 2.4 * j[None, :] + 0.37 * np.arange(len(dirs))[:, None]
    s = (r * np.cos(phi))[..., None] * u[:, None] + (r * np.sin(phi))[..., None] * v[
        :, None
    ]
    n = np.repeat(dirs, 8, 0)
    t = center + s.reshape(-1, 3) - 30 * step.max() * n  # anchors outside the support
    return t, n


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize(
    "start, step, num, q",
    [
        ((-4, -4, -4), (1, 1, 1), (9, 9, 9), (4, 4, 4)),  # unit lattice
        ((-3, -1.5, -8), (1, 0.5, 2), (7, 8, 9), (3, 3, 4)),  # anisotropic, non-cubic
        (
            (0, 0, 0),
            (1, 1, 1),
            (5, 6, 7),
            (0, 0, 0),
        ),  # corner: support past the lattice bbox
    ],
)
def test_single_basis(order, start, step, num, q):
    # one basis function, all directions on the sphere: matches the exact line integral.
    start, step = np.array(start, float), np.array(step, float)
    center = start + np.array(q) * step
    t, n = rays(directions(np.random.default_rng(0)), center, step)

    f = np.zeros(num, np.float32)
    f[q] = 1
    knot_spec = xtk.UniformSpec(start=tuple(start), step=tuple(step), num=num)
    ray_spec = (Array3f(t.T.astype(np.float32)), Array3f(n.T.astype(np.float32)))
    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, Float(f.ravel())))

    p_gt = exact(order, t, n, center, step)
    assert np.abs(p - p_gt).max() < 2e-5 * p_gt.max()


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("step", [(1, 1, 1), (1, 0.5, 2)])
def test_integral(order, step):
    # sum of parallel projections * pixel area = integral of the basis function / prod(step) = 1.
    step = np.array(step, float)
    knot_spec = xtk.UniformSpec(start=tuple(-4 * step), step=tuple(step), num=(9, 9, 9))
    f = np.zeros((9, 9, 9), np.float32)
    f[4, 4, 4] = 1

    rng = np.random.default_rng(order)
    dirs = np.r_[rng.standard_normal((4, 3)), [[1, 0, 0], [1, 1, 0], [1, 1, 1]]]
    for d in dirs / np.linalg.norm(dirs, axis=1, keepdims=True):
        a = np.array([1.0, 0, 0]) if abs(d[0]) < 0.9 else np.array([0, 1.0, 0])
        u = a - (a @ d) * d
        u /= np.linalg.norm(u)
        v = np.cross(d, u)
        h = 0.04 * step.min()
        g = np.arange(-2 * np.linalg.norm(step), 2 * np.linalg.norm(step), h) + 0.0123
        A, B = np.meshgrid(g, g, indexing="ij")
        t = A.reshape(-1, 1) * u + B.reshape(-1, 1) * v - 30 * step.max() * d
        n = np.broadcast_to(d, t.shape)
        ray_spec = (Array3f(t.T.astype(np.float32)), Array3f(n.T.astype(np.float32)))
        p = np.asarray(
            xtk.xrt_apply(ray_spec, knot_spec, order, Float(f.ravel())), np.float64
        )
        assert abs(p.sum() * h * h - 1) < 1e-4


@pytest.mark.parametrize("order", [1, 2])
def test_adjoint_transpose(order):
    # The adjoint equals the transpose of the forward, built column by column.
    # Half the rays go through lattice points along axes / face / body diagonals:
    # they cross cell edges and corners, where the walk steps along several axes at once.
    rng = np.random.default_rng(order)
    start, step, num = np.array([-1.0, 0.5, 2.0]), np.array([1.0, 0.5, 2.0]), (5, 6, 7)
    knot_spec = xtk.UniformSpec(start=tuple(start), step=tuple(step), num=num)
    hi = start + step * (np.array(num) - 1)

    t = rng.uniform(start - 2, hi + 2, (100, 3))
    n = rng.standard_normal((100, 3))
    d = rng.choice([-1.0, 0.0, 1.0], (100, 3))
    d[~d.any(1)] = 1
    t = np.r_[t, start + step * rng.integers(0, num, (100, 3)) - 50 * d * step]
    n = np.r_[n, d * step]
    ray_spec = (Array3f(t.T.astype(np.float32)), Array3f(n.T.astype(np.float32)))

    Q = int(np.prod(num))
    A = np.stack(
        [
            np.asarray(
                xtk.xrt_apply(
                    ray_spec, knot_spec, order, Float(np.eye(1, Q, q, np.float32)[0])
                )
            )
            for q in range(Q)
        ],
        axis=1,
    )
    p = rng.standard_normal(len(t)).astype(np.float32)
    f = np.asarray(xtk.xrt_adjoint(ray_spec, knot_spec, order, Float(p)))
    assert np.abs(f - A.T @ p).max() < 1e-5 * np.abs(A.T @ p).max()
