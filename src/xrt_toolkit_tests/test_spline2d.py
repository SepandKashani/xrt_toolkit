import math

import drjit as dr
import numpy as np
import pytest

import xrt_toolkit as xtk
from xrt_toolkit.drjit.box_spline import box_spline_1d_np

try:
    from drjit.cuda import Array2f, Float

    dr.width(Float(0))
except Exception:
    from drjit.llvm import Array2f, Float


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("step", [(1, 1), (0.7, 1.3)])
@pytest.mark.parametrize("q", [(5, 6), (0, 6), (5, 12), (10, 0)])
def test_single_basis(order, step, q):
    # One basis function (interior, on an edge, in a corner), rays in all directions,
    # also outside the lattice: matches the 1D box-spline of its projected directions.
    # Directions within 1e-2 rad of the lattice axes/diagonals are excluded (E_mask
    # approximation).
    rng = np.random.default_rng(0)
    step, start, num = np.array(step), np.array([-2.0, 1.0]), (11, 13)
    center = start + np.array(q) * step

    th = rng.uniform(0, 2 * np.pi, 3000)
    n = np.stack([np.cos(th), np.sin(th)], 1)
    g = n / step
    ang = np.arctan2(g[:, 1], g[:, 0]) % (np.pi / 4)  # angle in lattice units
    n = n[(1e-2 < ang) & (ang < np.pi / 4 - 1e-2)]
    u = np.stack([-n[:, 1], n[:, 0]], 1)
    x = rng.uniform(-3, 3, len(n)) * step.max()  # lateral offset from the basis center
    t = center + x[:, None] * u - 50 * n  # anchors outside the lattice

    f = np.zeros(num, np.float32)
    f[q] = 1
    knot_spec = xtk.UniformSpec(start=tuple(start), step=tuple(step), num=num)
    ray_spec = (Array2f(t.T.astype(np.float32)), Array2f(n.T.astype(np.float32)))
    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, Float(f.ravel())))

    E = [(1, 0), (0, 1), (1, 1), (1, -1)][: order + 2]
    # (L, order + 2) projected directions
    E = np.abs(u @ (step[:, None] * np.array(E).T))
    p_gt = np.array([box_spline_1d_np(E[i], x[i : i + 1])[0] for i in range(len(x))])
    assert np.abs(p - p_gt).max() < 3e-5 * p_gt.max()


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("base", ["axis", "diagonal"])
@pytest.mark.parametrize("E_0", [1e-5, 1e-4, 5e-4])
def test_near_axes(order, base, E_0):
    # Rays close to a lattice axis or diagonal, where one projected direction E_0 <= 1e-3 is
    # dropped: still matches the exact box-spline.
    rng = np.random.default_rng(1)
    step, start, num, q = np.array([0.7, 1.3]), np.array([-2.0, 1.0]), (11, 13), (5, 6)
    center = start + np.array(q) * step

    w = np.array([step[0], 0]) if (base == "axis") else step  # direction dropped
    th = math.atan2(w[1], w[0]) + E_0 / np.linalg.norm(w)
    n = np.tile([math.cos(th), math.sin(th)], (500, 1))
    u = np.stack([-n[:, 1], n[:, 0]], 1)
    x = rng.uniform(-3, 3, len(n)) * step.max()
    t = center + x[:, None] * u - 50 * n

    f = np.zeros(num, np.float32)
    f[q] = 1
    knot_spec = xtk.UniformSpec(start=tuple(start), step=tuple(step), num=num)
    ray_spec = (Array2f(t.T.astype(np.float32)), Array2f(n.T.astype(np.float32)))
    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, order, Float(f.ravel())))

    # exact box-spline, all directions kept (float64 is accurate enough for E >= 1e-6)
    E = [(1, 0), (0, 1), (1, 1), (1, -1)][: order + 2]
    E = np.abs(u[0] @ (step[:, None] * np.array(E).T))
    p_gt = sum(
        (-1) ** sum(S) * np.maximum(x + E.sum() / 2 - np.dot(S, E), 0) ** (len(E) - 1)
        for S in np.ndindex(*(2,) * len(E))
    ) / (math.factorial(len(E) - 1) * np.prod(E))
    assert np.abs(p - p_gt).max() < 2e-5 * p_gt.max()
