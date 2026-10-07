r"""
Exact 3D projections of the order-1 and order-2 basis functions.

In 3D, the basis function of order :math:`k \in \{1, 2\}` is the tensor-product B-spline

.. math::

   \psi(\bbx)
   =
   \beta^{k}(x_{1} / \Delta_{1})
   \beta^{k}(x_{2} / \Delta_{2})
   \beta^{k}(x_{3} / \Delta_{3}),
   \qquad
   \beta^{k} = \mathbb{1}_{[-1/2, 1/2]}^{\ast (k+1)},

i.e. the voxel convolved with itself :math:`k` times (trilinear, then triquadratic).

Along a ray, :math:`\psi` is a polynomial of degree :math:`3k` between consecutive crossings of its knot
planes. The knot planes of all basis functions form one Cartesian grid: the dual grid (corners at the basis
centers) for :math:`k = 1`, the primal grid for :math:`k = 2`. A DDA walk over that grid therefore visits
the polynomial pieces, and Gauss-Legendre quadrature with :math:`\lceil (3k+1)/2 \rceil` nodes per piece (2 for
:math:`k = 1`, 4 for :math:`k = 2`) integrates each piece exactly. Each visited cell contributes to the :math:`(k+1)^{3}`
basis functions whose support covers it.

Normalization matches order 0: each entry is :math:`\int \psi(\bbt + \alpha \bbn) d\alpha / \prod_{d}
\Delta_{d}`.
"""

import typing as typ

import drjit as dr
import numpy as np

BoolT = typ.TypeVar("BoolT", bound=dr.AnyArray)
ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
ArrayNuT = typ.TypeVar("ArrayNuT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
UIntT = typ.TypeVar("UIntT", bound=dr.AnyArray)


def _gauss_legendre(m: int) -> tuple[list[float], list[float]]:
    # m-node Gauss-Legendre rule on [0, 1].
    x, w = np.polynomial.legendre.leggauss(m)
    return [float(_) for _ in 0.5 * (x + 1)], [float(_) for _ in 0.5 * w]


# Exact for the degree-3k pieces of order k.
NODES = {
    1: _gauss_legendre(2),
    2: _gauss_legendre(4),
}


def _blend(order: int, u: ArrayNfT) -> list[ArrayNfT]:
    # Uniform B-spline blending functions of one cell, u \in [0, 1].
    if order == 1:
        return [1 - u, u]
    elif order == 2:
        return [0.5 * dr.square(1 - u), dr.fma(u, 1 - u, 0.5), 0.5 * dr.square(u)]


def spline3d_grid(
    knot_start: ArrayNfT,
    knot_step: ArrayNfT,
    knot_num: ArrayNuT,
    order: int,
) -> tuple[ArrayNfT, ArrayNfT, ArrayNuT]:
    r"""
    Grid of polynomial pieces of the order-`order` basis functions.

    It covers the support of every basis function: the dual grid (one more cell per axis) at order 1, the primal
    grid padded by one cell on each side at order 2.

    Returns
    -------
    grid_ll, grid_ur: ArrayNfT
        (3,) lower-left/upper-right grid corners.
    grid_res: ArrayNuT
        (3,) grid size.
    """
    grid_ll = knot_start - (0.5 * (order + 1)) * knot_step
    grid_res = knot_num + order
    grid_ur = grid_ll + grid_res * knot_step
    return grid_ll, grid_ur, grid_res


def spline3d_stencil(
    order: int,
    index: ArrayNuT,
    p_a: ArrayNfT,
    p_b: ArrayNfT,
    knot_step: ArrayNfT,
    knot_num: ArrayNuT,
    stride: ArrayNuT,
) -> list[tuple[UIntT, BoolT, FloatT]]:
    r"""
    Contributions of one traversed cell to its :math:`(k+1)^{3}` basis functions.

    Parameters
    ----------
    order: 1 | 2
    index: ArrayNuT
        (3,) cell index on the grid of :py:func:`spline3d_grid`.
    p_a, p_b: ArrayNfT
        (3,) fractional entry/exit positions in the cell.
    knot_step: ArrayNfT
        (3,) lattice step.
    knot_num: ArrayNuT
        (3,) lattice size.
    stride: ArrayNuT
        (3,) C-order strides of the lattice.

    Returns
    -------
    stencil: list[tuple[UIntT, BoolT, FloatT]]
        (offset, valid, weight) per basis function:

        * offset: flat lattice offset (clipped to stay in range);
        * valid: basis function lies inside the lattice;
        * weight: line integral of the basis function over the segment, divided by :math:`\prod_{d} \Delta_{d}`.
    """
    ArrayNf = type(p_a)
    ArrayNi = dr.int32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    UInt = dr.uint32_array_t(Float)

    nodes, weights = NODES[order]
    K = order + 1

    seg = p_b - p_a
    scale = dr.norm(seg * knot_step) * dr.rcp(dr.prod(knot_step))
    B = [_blend(order, dr.fma(seg, s, p_a)) for s in nodes]  # B[node][c][axis]

    # traversal cell i <-> basis functions i - k + c, c = 0..k
    base = ArrayNi(index) - order
    num = ArrayNi(knot_num)
    ax_valid, ax_offset = [], []
    for d in range(3):
        q = [base[d] + c for c in range(K)]
        ax_valid.append([(_q >= 0) & (_q < num[d]) for _q in q])
        ax_offset.append([UInt(dr.clip(_q, 0, num[d] - 1)) * stride[d] for _q in q])

    stencil = []
    for c1 in range(K):
        w1 = [weights[j] * B[j][c1].x for j in range(len(nodes))]
        for c2 in range(K):
            w12 = [w1[j] * B[j][c2].y for j in range(len(nodes))]
            for c3 in range(K):
                w = w12[0] * B[0][c3].z
                for j in range(1, len(nodes)):
                    w = dr.fma(w12[j], B[j][c3].z, w)
                offset = ax_offset[0][c1] + ax_offset[1][c2] + ax_offset[2][c3]
                valid = ax_valid[0][c1] & ax_valid[1][c2] & ax_valid[2][c3]
                stencil.append((offset, valid, w * scale))
    return stencil
