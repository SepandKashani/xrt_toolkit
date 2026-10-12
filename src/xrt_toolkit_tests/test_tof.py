import math

import drjit as dr
import numpy as np
import pytest

import xrt_toolkit as xtk

try:
    from drjit.cuda import Array2f, Array3f, Float

    dr.width(Float(0))
except Exception:
    from drjit.llvm import Array2f, Array3f, Float

erf = np.vectorize(math.erf)


def tof_reference(f, knot_spec, t, n, mu, sigma) -> np.ndarray:
    # Exact TOF projections of the order-0 field `f`, independently of XTK: each ray is
    # split at every voxel plane; on each segment the field is constant and the Gaussian
    # integrates to a difference of erf.
    start, step, num = map(np.array, (knot_spec.start, knot_spec.step, knot_spec.num))
    ll = start - step / 2
    out = np.zeros(t.shape[1])
    for l in range(t.shape[1]):
        nh = n[:, l] / np.linalg.norm(n[:, l])
        cuts = [
            (ll[d] + np.arange(num[d] + 1) * step[d] - t[d, l]) / nh[d]
            for d in range(len(step))
            if nh[d] != 0
        ]
        a = np.sort(np.concatenate(cuts))
        mid = 0.5 * (a[:-1] + a[1:])
        q = np.floor((t[:, l, None] + mid * nh[:, None] - ll[:, None]) / step[:, None])
        inside = np.all((q >= 0) & (q < num[:, None]), axis=0)
        val = np.zeros(len(mid))
        val[inside] = f[tuple(q[:, inside].astype(int))] / np.prod(step)
        z = (a - mu[l]) / (sigma[l] * math.sqrt(2))
        out[l] = np.sum(val * 0.5 * (erf(z[1:]) - erf(z[:-1])))
    return out


@pytest.fixture(params=[2, 3])
def knot_spec(request) -> xtk.UniformSpec:
    D = request.param
    return xtk.UniformSpec(
        start=(-1.1, 0.3, 2.2)[:D],
        step=(0.5, 1.0, 0.75)[:D],
        num=(14, 11, 9)[:D],
    )


def rays(knot_spec, L, rng) -> tuple:
    # Rays through random points x of the volume, anchored up to 15 before or after x,
    # random directions (a few axis-aligned), and TOF centers near x.
    D = knot_spec.ndim
    lo = np.array(knot_spec.start)
    hi = lo + np.array(knot_spec.step) * (np.array(knot_spec.num) - 1)
    x = rng.uniform(lo[:, None], hi[:, None], (D, L))
    n = rng.standard_normal((D, L))
    n[:, : L // 10] = np.eye(D)[:, :1]
    n /= np.linalg.norm(n, axis=0)
    s = rng.uniform(-15, 15, L)
    t = x - s * n
    mu = s + rng.uniform(-2, 2, L)
    return t.astype(np.float32), n.astype(np.float32), mu.astype(np.float32)


def test_tof_reference(knot_spec):
    D = knot_spec.ndim
    rng = np.random.default_rng(0)
    L = 300
    t, n, mu = rays(knot_spec, L, rng)
    sigma = rng.choice([0.3, 3.0, 30.0], L).astype(np.float32)
    f = rng.random(knot_spec.num).astype(np.float32)

    ArrayNf = Array2f if (D == 2) else Array3f
    tof = xtk.TOFSpec(center=Float(mu), sigma=Float(sigma))
    p = xtk.xrt_apply((ArrayNf(t), ArrayNf(n)), knot_spec, 0, Float(f.ravel()), tof=tof)
    p_gt = tof_reference(f, knot_spec, t.astype(np.float64), n, mu, sigma)
    assert np.abs(np.asarray(p) - p_gt).max() < 1e-4 * np.abs(p_gt).max()


@pytest.mark.parametrize("order", [0, 1, 2])
def test_tof_adjoint(knot_spec, order):
    # <A f, p> = <f, A^T p> with TOF
    D = knot_spec.ndim
    rng = np.random.default_rng(1)
    L = 500
    t, n, mu = rays(knot_spec, L, rng)
    tof = xtk.TOFSpec(
        center=Float(mu),
        sigma=Float(rng.uniform(0.5, 10, L).astype(np.float32)),
    )
    f = rng.standard_normal(math.prod(knot_spec.num)).astype(np.float32)
    p = rng.standard_normal(L).astype(np.float32)

    ArrayNf = Array2f if (D == 2) else Array3f
    ray_spec = (ArrayNf(t), ArrayNf(n))
    Af = np.asarray(
        xtk.xrt_apply(ray_spec, knot_spec, order, Float(f), tof=tof), np.float64
    )
    Atp = np.asarray(
        xtk.xrt_adjoint(ray_spec, knot_spec, order, Float(p), tof=tof), np.float64
    )
    lhs, rhs = Af @ p.astype(np.float64), f.astype(np.float64) @ Atp
    assert abs(lhs - rhs) < 1e-4 * np.linalg.norm(Af) * np.linalg.norm(p)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_tof_anchor(knot_spec, order):
    # The arc length is measured from the anchor: moving the anchor by s along the ray
    # and the TOF center by -s gives the same projections.
    D = knot_spec.ndim
    rng = np.random.default_rng(2)
    L = 300
    t, n, mu = rays(knot_spec, L, rng)
    f = Float(rng.random(math.prod(knot_spec.num)).astype(np.float32))

    ArrayNf = Array2f if (D == 2) else Array3f
    out = []
    for s in (0.0, 7.5):
        tof = xtk.TOFSpec(center=Float(mu - s), sigma=2.0)
        ray_spec = (ArrayNf(t + s * n), ArrayNf(n))
        out.append(np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, f, tof=tof)))
    assert np.abs(out[1] - out[0]).max() < 1e-4 * np.abs(out[0]).max()


KNOTS = {1: [-1.0, 0.0, 1.0], 2: [-1.5, -0.5, 0.5, 1.5]}  # knot positions of beta^k
GL_X, GL_W = np.polynomial.legendre.leggauss(16)


def beta(k: int, x: np.ndarray) -> np.ndarray:
    a = np.abs(x)
    if k == 1:
        return np.clip(1 - a, 0, None)
    return np.where(a <= 0.5, 0.75 - a**2, np.where(a <= 1.5, 0.5 * (1.5 - a) ** 2, 0))


@pytest.mark.parametrize("sigma", [1.0, 3.0, 30.0])
@pytest.mark.parametrize("order", [1, 2])
def test_tof_3d_splines(order, sigma):
    # One 3D basis function: matches its line integral against the TOF window, computed
    # by splitting each ray at the knot planes (16-node Gauss-Legendre per piece).
    rng = np.random.default_rng(3)
    step, num, q = np.array([1.0, 0.5, 2.0]), (9, 9, 9), (4, 4, 4)
    start = -4 * step
    center = start + np.array(q) * step
    knot_spec = xtk.UniformSpec(start=tuple(start), step=tuple(step), num=num)

    L = 500
    n = rng.standard_normal((3, L))
    n /= np.linalg.norm(n, axis=0)
    x = rng.standard_normal((3, L)) * 1.2 * step[:, None]
    x -= np.sum(x * n, axis=0) * n  # rays passing near the basis center
    s = rng.uniform(-20, 20, L)
    t = center[:, None] + x - s * n
    mu = s + rng.uniform(-2, 2, L) * sigma

    p_gt = np.zeros(L)
    for l in range(L):
        cuts = [
            (center[d] + np.array(KNOTS[order]) * step[d] - t[d, l]) / n[d, l]
            for d in range(3)
        ]
        a = np.unique(np.concatenate(cuts))
        lo, hi = a[:-1], a[1:]
        aq = 0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * GL_X
        xq = t[:, l, None, None] + aq * n[:, l, None, None]
        f = np.prod(
            [beta(order, (xq[d] - center[d]) / step[d]) for d in range(3)], axis=0
        )
        g = np.exp(-0.5 * ((aq - mu[l]) / sigma) ** 2) / (
            sigma * math.sqrt(2 * math.pi)
        )
        p_gt[l] = np.sum(f * g * GL_W * 0.5 * (hi - lo)[:, None]) / np.prod(step)

    vol = np.zeros(num, np.float32)
    vol[q] = 1
    ray_spec = (Array3f(t.astype(np.float32)), Array3f(n.astype(np.float32)))
    tof = xtk.TOFSpec(center=Float(mu.astype(np.float32)), sigma=sigma)
    p = np.asarray(
        xtk.xrt_apply(ray_spec, knot_spec, order, Float(vol.ravel()), tof=tof)
    )
    assert np.abs(p - p_gt).max() < 1e-4 * np.abs(p_gt).max()


@pytest.mark.parametrize("order", [1, 2])
def test_tof_2d_splines(order):
    # 2D orders 1, 2 weight each basis function's projection by the TOF window at its
    # center.
    rng = np.random.default_rng(4)
    knot_spec = xtk.UniformSpec(start=(-1.1, 0.3), step=(0.5, 1.0), num=(8, 7))
    L = 300
    t, n, mu = rays(knot_spec, L, rng)
    sigma = 3.0
    Q = math.prod(knot_spec.num)
    ray_spec = (Array2f(t), Array2f(n))

    # projections of each basis function (no TOF), and the window at each basis center
    P = np.stack(
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
    axes = [s + d * np.arange(m) for (s, d, m) in knot_spec]
    centers = np.stack([c.ravel() for c in np.meshgrid(*axes, indexing="ij")])  # (2, Q)
    alpha = np.einsum("dlq,dl->lq", centers[:, None, :] - t[:, :, None], n)
    g = np.exp(-0.5 * ((alpha - mu[:, None]) / sigma) ** 2) / (
        sigma * math.sqrt(2 * math.pi)
    )

    f = rng.random(Q).astype(np.float32)
    tof = xtk.TOFSpec(center=Float(mu), sigma=sigma)
    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, Float(f), tof=tof))
    p_gt = (P * g) @ f
    assert np.abs(p - p_gt).max() < 1e-5 * np.abs(p_gt).max()
