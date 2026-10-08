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


def array_t(D: int):
    return Array2f if (D == 2) else Array3f


def random_rays(knot_spec: xtk.UniformSpec, L: int, rng) -> tuple:
    # Rays anchored inside the volume, with random directions.
    D = knot_spec.ndim
    ll = np.array(knot_spec.start)
    ur = ll + np.array(knot_spec.step) * (np.array(knot_spec.num) - 1)
    t = rng.uniform(ll[:, None], ur[:, None], (D, L))
    n = rng.standard_normal((D, L))
    n /= np.linalg.norm(n, axis=0)
    ArrayNf = array_t(D)
    return ArrayNf(t.astype(np.float32)), ArrayNf(n.astype(np.float32))


def inner_product(a, b) -> float:
    return float(np.dot(np.asarray(a, np.float64), np.asarray(b, np.float64)))


@pytest.fixture(params=[2, 3])
def knot_spec(request) -> xtk.UniformSpec:
    D = request.param
    return xtk.UniformSpec(
        start=(-1.1, 0.3, 2.2)[:D],  # no lattice boundary on the axes
        step=(0.5, 1.0, 0.75)[:D],
        num=(20, 17, 15)[:D],
    )


def test_value_apply(knot_spec):
    # order 0, axis-aligned rays through voxel centers.
    # psi is the normalized box-spline (unit integral):
    # each voxel contributes step[axis] / prod(step).
    D = knot_spec.ndim
    rng = np.random.default_rng(0)
    f = rng.standard_normal(knot_spec.num).astype(np.float32)

    start, step, num = map(np.array, (knot_spec.start, knot_spec.step, knot_spec.num))
    t, n, p_gt = [], [], []
    for axis in range(D):
        centers = [start[d] + step[d] * np.arange(num[d]) for d in range(D)]
        # anchor outside the volume
        centers[axis] = np.array([start[axis] - 3 * step[axis]])
        _t = np.stack([c.ravel() for c in np.meshgrid(*centers, indexing="ij")])
        _n = np.zeros_like(_t)
        _n[axis] = 1
        t.append(_t)
        n.append(_n)
        p_gt.append(np.sum(f, axis=axis).ravel() * step[axis] / np.prod(step))
    t, n, p_gt = map(np.concatenate, (t, n, p_gt), (1, 1, 0))

    ArrayNf = array_t(D)
    ray_spec = (ArrayNf(t.astype(np.float32)), ArrayNf(n.astype(np.float32)))
    p = xtk.xrt_apply(ray_spec, knot_spec, 0, Float(f.ravel()))
    assert np.allclose(np.asarray(p), p_gt, atol=1e-4)


@pytest.mark.parametrize("order", [0, 1, 2])
def test_math_adjoint(knot_spec, order):
    # <A f, p> = <f, A^T p>
    rng = np.random.default_rng(1)
    L = 1_000
    ray_spec = random_rays(knot_spec, L, rng)
    f = Float(rng.standard_normal(math.prod(knot_spec.num)).astype(np.float32))
    p = Float(rng.standard_normal(L).astype(np.float32))

    lhs = inner_product(xtk.xrt_apply(ray_spec, knot_spec, order, f), p)
    rhs = inner_product(f, xtk.xrt_adjoint(ray_spec, knot_spec, order, p))
    assert np.isclose(lhs, rhs, rtol=1e-4)


def struct_spec(D: int, beam: str) -> tuple:
    angles = Float(np.linspace(0, np.pi, 7, endpoint=False, dtype=np.float32))
    detector_spec = xtk.util.DetectorSpec(size=40, num_cell=(11, 9)[: D - 1])
    if beam == "parallel":
        return xtk.parallel_beam(angles, detector_spec)
    else:
        return xtk.cone_beam(30, 60, angles, detector_spec)


def explicit_rays(ray_spec: tuple, D: int) -> tuple:
    # the rays a structured spec encodes, built independently with NumPy
    ray_t_spec, ray_n_spec, ray_u_spec = ray_spec
    u = [start + step * np.arange(num) for (start, step, num) in ray_u_spec]
    uu = np.stack([c.ravel() for c in np.meshgrid(*u, indexing="ij")])  # (D, L_proj)
    H_t = np.asarray(ray_t_spec)  # (D, D, N_proj)
    H_n = np.asarray(ray_n_spec)
    t = np.einsum("ijk,jl->ikl", H_t, uu).reshape(D, -1)  # (D, N_proj * L_proj)
    n = np.einsum("ijk,jl->ikl", H_n, uu).reshape(D, -1)
    return t, n


@pytest.mark.parametrize("beam", ["parallel", "cone"])
def test_struct_rays(knot_spec, beam):
    D = knot_spec.ndim
    ray_spec = struct_spec(D, beam)
    t, n = xtk.struct_rays(ray_spec)
    t_gt, n_gt = explicit_rays(ray_spec, D)
    assert np.allclose(np.asarray(t), t_gt, atol=1e-5)
    assert np.allclose(np.asarray(n), n_gt, atol=1e-5)


@pytest.mark.parametrize("beam", ["parallel", "cone"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_struct_apply(knot_spec, beam, order):
    # xrt_struct_apply() = xrt_apply() on the rays the structured spec encodes.
    D = knot_spec.ndim
    rng = np.random.default_rng(2)
    ray_spec = struct_spec(D, beam)
    t, n = explicit_rays(ray_spec, D)
    ArrayNf = array_t(D)

    f = Float(rng.standard_normal(math.prod(knot_spec.num)).astype(np.float32))
    p = xtk.xrt_struct_apply(ray_spec, knot_spec, order, f)
    p_gt = xtk.xrt_apply((ArrayNf(t), ArrayNf(n)), knot_spec, order, f)
    assert np.allclose(np.asarray(p), np.asarray(p_gt), atol=1e-4)


@pytest.mark.parametrize("beam", ["parallel", "cone"])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_struct_math_adjoint(knot_spec, beam, order):
    D = knot_spec.ndim
    rng = np.random.default_rng(3)
    ray_spec = struct_spec(D, beam)
    L = 7 * math.prod(ray_spec[2].num)

    f = Float(rng.standard_normal(math.prod(knot_spec.num)).astype(np.float32))
    p = Float(rng.standard_normal(L).astype(np.float32))

    lhs = inner_product(xtk.xrt_struct_apply(ray_spec, knot_spec, order, f), p)
    rhs = inner_product(f, xtk.xrt_struct_adjoint(ray_spec, knot_spec, order, p))
    assert np.isclose(lhs, rhs, rtol=1e-4)
