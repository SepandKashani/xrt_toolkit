import itertools
import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

from .bbox import ray_bbox_intersect
from .dda import dda
from .spline3d import (
    spline3d_cell_basis,
    spline3d_coeffs,
    spline3d_grad,
    spline3d_grid,
)

BoolT = typ.TypeVar("BoolT", bound=dr.AnyArray)
ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
ArrayNiT = typ.TypeVar("ArrayNiT", bound=dr.AnyArray)
ArrayNuT = typ.TypeVar("ArrayNuT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNfT, ArrayNfT]


def xrt_grad(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
) -> tuple[ArrayNfT, ArrayNfT]:
    r"""
    Gradients of 2D/3D projections with respect to the ray geometry.

    For each ray :math:`l`, the gradients of its projection :math:`P_{l}` (see
    :py:func:`~xrt_toolkit.xrt_apply`) with respect to its anchor :math:`\bbt_{l}` and its direction
    :math:`\bbn_{l}`:

    .. math::

       \nabla_{\bbt} P_{l} = \int_{\bR} \nabla f(\bbt_{l} + \alpha \hat{\bbn}_{l}) \, d\alpha,
       \qquad
       \nabla_{\bbn} P_{l} = \frac{\bbI - \hat{\bbn}_{l} \hat{\bbn}_{l}^{T}}{\|\bbn_{l}\|}
       \int_{\bR} \alpha \nabla f(\bbt_{l} + \alpha \hat{\bbn}_{l}) \, d\alpha,

    with :math:`\hat{\bbn}_{l} = \bbn_{l} / \|\bbn_{l}\|`. All D components come from one walk along
    the ray.

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt` and directions :math:`\bbn`. (See
        :py:func:`~xrt_toolkit.xrt_apply`.)
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Interpolation order (orders 1, 2: 3D only).
    data: FloatT
        (Q1,...,QD) flattened C-ordered volume weights :math:`f_{\bbq} \in \bR`.

    Returns
    -------
    grad_t, grad_n: ArrayNfT
        (L,) gradients :math:`\nabla_{\bbt} P_{l}`, :math:`\nabla_{\bbn} P_{l}`, in the units of
        `knot_spec`.

    Notes
    -----
    * :math:`P_{l}` depends neither on the anchor's position along the line nor on
      :math:`\|\bbn_{l}\|`: :math:`\nabla_{\bbt} P_{l}` and :math:`\nabla_{\bbn} P_{l}` are orthogonal
      to :math:`\bbn_{l}`. Rotations pivot at the anchor.
    * Order 0: :math:`\nabla f` lives on the cell faces. Each face the ray crosses contributes the
      jump of :math:`f` across it, divided by :math:`\hat{n}_{k}` (face normal to axis :math:`k`),
      times :math:`\alpha` for :math:`\nabla_{\bbn}`.
    * A ray through a lattice edge or corner (several faces crossed at once) is a kink of
      :math:`P_{l}`. There, the jump across face :math:`k` is averaged over face :math:`k` crossed
      first and crossed last: the gradient is what central finite differences converge to.
    """
    ray_t, ray_n = ray_spec

    ArrayNf = type(ray_t)
    Float = dr.value_t(ArrayNf)

    # type checking ---------------------------------------
    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D in (2, 3))
    assert type(ray_n) is ArrayNf

    assert knot_spec.ndim == D
    assert (order == 0) or ((D == 3) and (order in (1, 2)))

    assert type(data) is Float
    assert len(data) == math.prod(knot_spec.num)
    # -----------------------------------------------------

    if order == 0:
        return _grad_voxel(ray_t, ray_n, knot_spec, data)
    else:
        return _grad_spline3d(ray_t, ray_n, knot_spec, order, data)


def _grad_voxel(
    ray_t: ArrayNfT,
    ray_n: ArrayNfT,
    knot_spec: xrtu.UniformSpec,
    data: FloatT,
) -> tuple[ArrayNfT, ArrayNfT]:
    # order 0: jumps of f across the crossed faces.
    ArrayNf = type(ray_t)
    ArrayNi = dr.int32_array_t(ArrayNf)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Int = dr.value_t(ArrayNi)
    Bool = dr.mask_t(Float)
    D = dr.size_v(ArrayNf)
    L = max(ray_t.shape[1], ray_n.shape[1])

    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = knot_start - (knot_step / 2) + (knot_num * knot_step)
    if D == 2:
        stride = ArrayNi(knot_num.y, 1)
    elif D == 3:
        stride = ArrayNi(knot_num.y * knot_num.z, knot_num.z, 1)
    rcp_vol = dr.rcp(dr.prod(knot_step))

    u = dr.normalize(ray_n)
    rcp_u = dr.select(u != 0, dr.rcp(u), 0)  # axes with u = 0 are never crossed

    # walk the bbox padded by one (empty) cell: entering and leaving the lattice are
    # crossings like any other.
    grid_ll = bbox_ll - knot_step
    grid_ur = bbox_ur + knot_step
    grid_res = knot_num + 2
    step = ArrayNi(dr.select(ray_n >= 0, 1, -1))

    # as xrt_apply(): move the anchor just before the (padded) bbox entry.
    active, t1, t2 = ray_bbox_intersect(grid_ll, grid_ur, ray_t, ray_n)
    t_min = dr.minimum(t1, t2)
    ray_o = dr.select(active, ray_t + (t_min - 1) * ray_n, ray_t)

    def value(q: ArrayNiT, valid: BoolT) -> FloatT:
        # f in (padded) cell q, 0 outside the lattice
        q = q - 1
        valid = valid & dr.all((0 <= q) & (q < ArrayNi(knot_num)))
        return dr.gather(Float, data, dr.dot(q, stride), valid) * rcp_vol

    def crossing(q_a: ArrayNiT, q_b: ArrayNiT, f_a: FloatT, f_b: FloatT) -> ArrayNfT:
        # Jump of f across each face crossed from cell q_a into cell q_b. Through an edge or a
        # corner (several axes K at once), the jump across face k is averaged over face k
        # crossed first and crossed last, as a central finite difference in t_k or n_k sees it.
        K = q_b != q_a
        J = dr.select(K, f_b - f_a, 0) * dr.ones(ArrayNf, 1)

        def tie_jump(J: ArrayNfT, q_a: ArrayNiT, K: BoolT) -> ArrayNfT:
            for k in range(D):
                e_k = ArrayNi(*[step[k] if (j == k) else 0 for j in range(D)])
                f_first = value(q_a + e_k, K[k])
                f_last = value(q_a + dr.select(K, step, 0) - e_k, K[k])
                J[k] = dr.select(K[k], 0.5 * (J[k] + f_first - f_last), 0)
            return J

        tie = sum(Int(K[k]) for k in range(D)) > 1
        return dr.if_stmt((J, q_a, K), tie, tie_jump, lambda J, q_a, K: J)

    def project(
        state: tuple,
        index: ArrayNuT,
        p_a: ArrayNfT,
        p_b: ArrayNfT,
        active: BoolT,
    ) -> tuple[tuple, BoolT]:
        # crossing into this cell (none into the first one: it is outside the lattice)
        g_t, g_u, r, q_a, f_a, started = state
        q = ArrayNi(index)
        f_q = value(q, active)
        dg = crossing(q_a, q, f_a, f_q) * rcp_u
        alpha_a = dr.dot(grid_ll + (ArrayNf(q) + p_a) * knot_step - ray_t, u)
        on = active & started
        g_t = dr.select(on, g_t + dg, g_t)
        g_u = dr.select(on, dr.fma(alpha_a, dg, g_u), g_u)
        r = dr.select(on, dr.fma(alpha_a, f_q - f_a, r), r)
        q_a = dr.select(active, q, q_a)
        f_a = dr.select(active, f_q, f_a)
        return (g_t, g_u, r, q_a, f_a, started | active), Bool(True)

    state = (
        dr.zeros(ArrayNf, L),  # grad_t
        dr.zeros(ArrayNf, L),  # integral of alpha grad f
        dr.zeros(Float, L),  # integral of alpha df/dalpha (= -P)
        dr.zeros(ArrayNi, L),  # previous cell
        dr.zeros(Float, L),  # f there
        dr.full(Bool, False, L),  # a cell was visited
    )
    g_t, g_u, r, *_ = dda(
        ray_o=ray_o,
        ray_d=ray_n,
        ray_max=Float(dr.inf),
        grid_res=grid_res,
        grid_min=grid_ll,
        grid_max=grid_ur,
        func=project,
        state=state,
        active=active,
        mode="symbolic",
        max_iterations=-1,
    )

    grad_t = dr.select(active, g_t, 0)
    grad_n = dr.select(active, (g_u - u * r) * dr.rcp(dr.norm(ray_n)), 0)
    return grad_t, grad_n


def _grad_spline3d(
    ray_t: ArrayNfT,
    ray_n: ArrayNfT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
) -> tuple[ArrayNfT, ArrayNfT]:
    # 3D orders 1, 2: gradient of the field integrated over each polynomial piece.
    ArrayNf = type(ray_t)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)
    L = max(ray_t.shape[1], ray_n.shape[1])

    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    stride = ArrayNu(knot_num.y * knot_num.z, knot_num.z, 1)
    u = dr.normalize(ray_n)

    # as xrt_apply(): walk the grid of polynomial pieces, from just before its entry.
    grid_ll, grid_ur, grid_res = spline3d_grid(knot_start, knot_step, knot_num, order)
    active, t1, t2 = ray_bbox_intersect(grid_ll, grid_ur, ray_t, ray_n)
    t_min = dr.minimum(t1, t2)
    ray_o = dr.select(active, ray_t + (t_min - 1) * ray_n, ray_t)

    # Order 1: a ray lying in a knot plane (n_k = 0, on a grid plane, as dda() sees it) is a
    # kink of the projection: the normal derivative of f jumps across the plane. Average the
    # cells on both sides, as central differences do.
    plane = None
    if order == 1:
        grid_scale = ArrayNf(grid_res) / (grid_ur - grid_ll)
        o = dr.fma(ray_o, grid_scale, -grid_ll * grid_scale)
        plane = (ray_n == 0) & (o == dr.floor(o))

    def other_side(G0, G1, f, index, p_a, p_b):
        # Order 1: across plane k the normal derivative of f is (f_1 - f_0) on one side and
        # (f_0 - f_-1) on the other (layers of coefficients along k, bilinear in the others);
        # their average is half that of the cell with layer 0 replaced by layer -1 (ray on
        # the lower face; layer 1 by layer 2 on the upper face): 4 more coefficients.
        ArrayNi = dr.int32_array_t(ArrayNf)
        for k in range(3):
            e_k = ArrayNi(*[1 if (j == k) else 0 for j in range(3)])
            below = p_a[k] == 0
            index_k = ArrayNu(ArrayNi(index) + dr.select(below, -e_k, e_k))
            f_k = []
            for f_c, c, (offset, valid) in zip(
                f,
                itertools.product(range(2), repeat=3),
                spline3d_cell_basis(order, index_k, knot_num, stride),
            ):
                swap = plane[k] & (below if (c[k] == 0) else ~below)
                f_k.append(
                    dr.select(swap, dr.gather(Float, data, offset, valid & swap), f_c)
                )
            H0, H1 = spline3d_grad(order, p_a, p_b, knot_step, f_k)
            G0[k] = dr.select(plane[k], 0.5 * H0[k], G0[k])
            G1[k] = dr.select(plane[k], 0.5 * H1[k], G1[k])
        return G0, G1

    def project(
        state: tuple,
        index: ArrayNuT,
        p_a: ArrayNfT,
        p_b: ArrayNfT,
        active: BoolT,
    ) -> tuple[tuple, BoolT]:
        g_t, g_u = state
        f = spline3d_coeffs(order, index, knot_num, stride, data, active)
        G0, G1 = spline3d_grad(order, p_a, p_b, knot_step, f)
        if plane is not None:
            G0, G1 = dr.if_stmt(
                (G0, G1, f, index, p_a, p_b),
                active & dr.any(plane),
                other_side,
                lambda G0, G1, *_: (G0, G1),
            )
        alpha_a = dr.dot(grid_ll + (ArrayNf(index) + p_a) * knot_step - ray_t, u)
        g_t = dr.select(active, g_t + G0, g_t)
        g_u = dr.select(active, dr.fma(alpha_a, G0, g_u) + G1, g_u)
        return (g_t, g_u), Bool(True)

    g_t, g_u = dda(
        ray_o=ray_o,
        ray_d=ray_n,
        ray_max=Float(dr.inf),
        grid_res=grid_res,
        grid_min=grid_ll,
        grid_max=grid_ur,
        func=project,
        state=(dr.zeros(ArrayNf, L), dr.zeros(ArrayNf, L)),
        active=active,
        mode="symbolic",
        max_iterations=-1,
    )

    grad_t = dr.select(active, g_t, 0)
    grad_n = dr.select(active, (g_u - u * dr.dot(u, g_u)) * dr.rcp(dr.norm(ray_n)), 0)
    return grad_t, grad_n
