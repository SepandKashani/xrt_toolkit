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
def test_single_basis(order, step):
    # One basis function, rays in all directions: matches the 1D box-spline of its
    # projected directions. Directions within 1e-2 rad of the lattice axes/diagonals
    # are excluded (E_mask approximation).
    rng = np.random.default_rng(0)
    step, start, num, q = np.array(step), np.array([-2.0, 1.0]), (11, 13), (5, 6)
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
