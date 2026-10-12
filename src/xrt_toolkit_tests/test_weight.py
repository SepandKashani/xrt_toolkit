import math

import drjit as dr
import numpy as np
import pytest

import xrt_toolkit as xtk
from xrt_toolkit_tests.test_tof import GL_W, GL_X, KNOTS, beta, rays

try:
    from drjit.cuda import Array2f, Array3f, Float

    dr.width(Float(0))
except Exception:
    from drjit.llvm import Array2f, Array3f, Float


def poly(coeff, origin, a):
    # w(a) per ray: coeff (K, L), origin (L,), a (L, ...)
    u = a - origin[..., None]
    return sum(c[..., None] * u**k for k, c in enumerate(coeff))


def table(values, num, start, step, a):
    # linear interpolation per ray, constant beyond the ends: values (L, num), a (L, ...)
    s = np.clip((a - start[:, None]) / step[:, None], 0, num - 1)
    j = np.minimum(np.floor(s).astype(int), num - 2)
    f = s - j
    v0 = np.take_along_axis(values, j, axis=1)
    v1 = np.take_along_axis(values, j + 1, axis=1)
    return (1 - f) * v0 + f * v1


def reference(f, knot_spec, t, n, w, cuts=None) -> np.ndarray:
    # Weighted order-0 projections, independently of XTK: each ray is split at every voxel
    # plane (and at `cuts`); on each segment the field is constant, and the weight is
    # integrated with 16-node Gauss-Legendre.
    start, step, num = map(np.array, (knot_spec.start, knot_spec.step, knot_spec.num))
    ll = start - step / 2
    out = np.zeros(t.shape[1])
    for l in range(t.shape[1]):
        nh = n[:, l] / np.linalg.norm(n[:, l])
        a = [
            (ll[d] + np.arange(num[d] + 1) * step[d] - t[d, l]) / nh[d]
            for d in range(len(step))
            if nh[d] != 0
        ]
        a = np.sort(np.concatenate(a + ([] if cuts is None else [cuts[l]])))
        mid = 0.5 * (a[:-1] + a[1:])
        q = np.floor((t[:, l, None] + mid * nh[:, None] - ll[:, None]) / step[:, None])
        inside = np.all((q >= 0) & (q < num[:, None]), axis=0)
        val = np.zeros(len(mid))
        val[inside] = f[tuple(q[:, inside].astype(int))] / np.prod(step)
        aq = mid[:, None] + 0.5 * (a[1:] - a[:-1])[:, None] * GL_X
        wq = w(l, aq)
        out[l] = np.sum(val * np.sum(wq * GL_W, axis=1) * 0.5 * (a[1:] - a[:-1]))
    return out


@pytest.fixture(params=[2, 3])
def knot_spec(request) -> xtk.UniformSpec:
    D = request.param
    return xtk.UniformSpec(
        start=(-1.1, 0.3, 2.2)[:D],
        step=(0.5, 1.0, 0.75)[:D],
        num=(14, 11, 9)[:D],
    )


def polynomial_weight(L, rng) -> tuple:
    # degree-3, per-ray coefficients and origins; changes sign along most rays
    coeff = rng.uniform(-1, 1, (4, L)) * np.array([1, 0.3, 0.03, 0.003])[:, None]
    origin = rng.uniform(-5, 5, L)
    spec = xtk.PolynomialWeightSpec(
        coeff=[Float(c.astype(np.float32)) for c in coeff],
        origin=Float(origin.astype(np.float32)),
    )
    return spec, coeff, origin


def tabulated_weight(L, rng) -> tuple:
    # per-ray grids covering part of each ray: constant beyond the ends
    num = 12
    values = rng.uniform(-1, 2, (L, num))
    start = rng.uniform(-10, 0, L)
    step = rng.uniform(0.3, 1.5, L)
    spec = xtk.TabulatedWeightSpec(
        values=Float(values.ravel().astype(np.float32)),
        num=num,
        start=Float(start.astype(np.float32)),
        step=Float(step.astype(np.float32)),
    )
    cuts = start[:, None] + step[:, None] * np.arange(num)
    return spec, (values, num, start, step), cuts


def array_t(D: int):
    return Array2f if (D == 2) else Array3f


def test_polynomial_reference(knot_spec):
    D = knot_spec.ndim
    rng = np.random.default_rng(0)
    L = 300
    t, n, _ = rays(knot_spec, L, rng)
    spec, coeff, origin = polynomial_weight(L, rng)
    f = rng.random(knot_spec.num).astype(np.float32)

    ArrayNf = array_t(D)
    ray_spec = (ArrayNf(t), ArrayNf(n))
    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, 0, Float(f.ravel()), weight=spec))
    w = lambda l, a: poly(coeff[:, l : l + 1], origin[l : l + 1], a[None])[0]  # noqa: E731
    p_gt = reference(f, knot_spec, t.astype(np.float64), n, w)
    assert np.abs(p - p_gt).max() < 1e-4 * np.abs(p_gt).max()


def test_tabulated_reference(knot_spec):
    D = knot_spec.ndim
    rng = np.random.default_rng(1)
    L = 300
    t, n, _ = rays(knot_spec, L, rng)
    spec, (values, num, start, step), cuts = tabulated_weight(L, rng)
    f = rng.random(knot_spec.num).astype(np.float32)

    ArrayNf = array_t(D)
    ray_spec = (ArrayNf(t), ArrayNf(n))
    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, 0, Float(f.ravel()), weight=spec))
    w = (
        lambda l, a: table(  # noqa: E731
            values[l : l + 1], num, start[l : l + 1], step[l : l + 1], a.reshape(1, -1)
        ).reshape(a.shape)
    )
    p_gt = reference(f, knot_spec, t.astype(np.float64), n, w, cuts)
    assert np.abs(p - p_gt).max() < 1e-4 * np.abs(p_gt).max()


@pytest.mark.parametrize("order", [0, 1, 2])
def test_constant_weight(knot_spec, order):
    # w = 1 gives the unweighted projections.
    D = knot_spec.ndim
    rng = np.random.default_rng(2)
    L = 300
    t, n, _ = rays(knot_spec, L, rng)
    f = Float(rng.random(math.prod(knot_spec.num)).astype(np.float32))
    ray_spec = (array_t(D)(t), array_t(D)(n))

    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, f))
    for spec in (
        xtk.PolynomialWeightSpec([1.0]),
        xtk.TabulatedWeightSpec(Float(np.ones(2 * L, np.float32)), num=2),
    ):
        p_w = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, f, weight=spec))
        assert np.abs(p_w - p).max() < 1e-5 * np.abs(p).max()


@pytest.mark.parametrize("order", [0, 1, 2])
def test_tabulated_matches_polynomial(knot_spec, order):
    # A table sampling w(a) = 2 - 0.1 a, extended linearly far beyond the volume, equals
    # the polynomial weight.
    D = knot_spec.ndim
    rng = np.random.default_rng(3)
    L = 300
    t, n, _ = rays(knot_spec, L, rng)
    f = Float(rng.random(math.prod(knot_spec.num)).astype(np.float32))
    ray_spec = (array_t(D)(t), array_t(D)(n))

    a = np.linspace(-60, 60, 41)
    tab = xtk.TabulatedWeightSpec(
        Float(np.tile(2 - 0.1 * a, L).astype(np.float32)),
        num=len(a),
        start=-60.0,
        step=3.0,
    )
    pol = xtk.PolynomialWeightSpec([2.0, -0.1])
    p_tab = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, f, weight=tab))
    p_pol = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, f, weight=pol))
    assert np.abs(p_tab - p_pol).max() < 1e-4 * np.abs(p_pol).max()


@pytest.mark.parametrize("order", [1, 2])
def test_polynomial_3d_splines(order):
    # One 3D basis function with a quadratic weight: matches the line integral computed by
    # splitting each ray at the knot planes (16-node Gauss-Legendre per piece).
    rng = np.random.default_rng(4)
    step, num, q = np.array([1.0, 0.5, 2.0]), (9, 9, 9), (4, 4, 4)
    start = -4 * step
    center = start + np.array(q) * step
    knot_spec = xtk.UniformSpec(start=tuple(start), step=tuple(step), num=num)

    L = 300
    n = rng.standard_normal((3, L))
    n /= np.linalg.norm(n, axis=0)
    x = rng.standard_normal((3, L)) * 1.2 * step[:, None]
    x -= np.sum(x * n, axis=0) * n  # rays passing near the basis center
    s = rng.uniform(-20, 20, L)
    t = center[:, None] + x - s * n
    coeff = np.array([1.0, -0.2, 0.05])

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
        w = coeff[0] + coeff[1] * aq + coeff[2] * aq**2
        p_gt[l] = np.sum(f * w * GL_W * 0.5 * (hi - lo)[:, None]) / np.prod(step)

    vol = np.zeros(num, np.float32)
    vol[q] = 1
    ray_spec = (Array3f(t.astype(np.float32)), Array3f(n.astype(np.float32)))
    spec = xtk.PolynomialWeightSpec(list(coeff))
    p = np.asarray(
        xtk.xrt_apply(ray_spec, knot_spec, order, Float(vol.ravel()), weight=spec)
    )
    assert np.abs(p - p_gt).max() < 1e-4 * np.abs(p_gt).max()


@pytest.mark.parametrize("kind", ["polynomial", "tabulated"])
@pytest.mark.parametrize("order", [1, 2])
def test_weight_2d_splines(order, kind):
    # 2D orders 1, 2 weight each basis function's projection by the weight at its center.
    rng = np.random.default_rng(6)
    knot_spec = xtk.UniformSpec(start=(-1.1, 0.3), step=(0.5, 1.0), num=(8, 7))
    L = 300
    t, n, _ = rays(knot_spec, L, rng)
    Q = math.prod(knot_spec.num)
    ray_spec = (Array2f(t), Array2f(n))

    # projections of each basis function (no weight), and the weight at each basis center
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
    if kind == "polynomial":
        spec, coeff, origin = polynomial_weight(L, rng)
        g = poly(coeff, origin, alpha)
    else:
        spec, (values, num, start, step), _ = tabulated_weight(L, rng)
        g = table(values, num, start, step, alpha)

    f = rng.random(Q).astype(np.float32)
    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, Float(f), weight=spec))
    p_gt = (P * g) @ f
    assert np.abs(p - p_gt).max() < 1e-5 * np.abs(p_gt).max()


@pytest.mark.parametrize("kind", ["polynomial", "tabulated"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_weight_adjoint(knot_spec, order, kind):
    # <A f, p> = <f, A^T p> with a weight
    D = knot_spec.ndim
    rng = np.random.default_rng(5)
    L = 500
    t, n, _ = rays(knot_spec, L, rng)
    if kind == "polynomial":
        spec = polynomial_weight(L, rng)[0]
    else:
        spec = tabulated_weight(L, rng)[0]
    f = rng.standard_normal(math.prod(knot_spec.num)).astype(np.float32)
    p = rng.standard_normal(L).astype(np.float32)

    ray_spec = (array_t(D)(t), array_t(D)(n))
    Af = xtk.xrt_apply(ray_spec, knot_spec, order, Float(f), weight=spec)
    Atp = xtk.xrt_adjoint(ray_spec, knot_spec, order, Float(p), weight=spec)
    Af, Atp = np.asarray(Af, np.float64), np.asarray(Atp, np.float64)
    lhs, rhs = Af @ p.astype(np.float64), f.astype(np.float64) @ Atp
    assert abs(lhs - rhs) < 1e-4 * np.linalg.norm(Af) * np.linalg.norm(p)
