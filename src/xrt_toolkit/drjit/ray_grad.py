import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

from .bbox import ray_bbox_intersect
from .dda import dda

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
    order: 0
        Interpolation order.
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
    ArrayNi = dr.int32_array_t(ArrayNf)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Int = dr.value_t(ArrayNi)
    Bool = dr.mask_t(Float)

    # type checking ---------------------------------------
    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D in (2, 3))
    assert type(ray_n) is ArrayNf

    assert knot_spec.ndim == D
    assert order in (0,)

    assert type(data) is Float
    assert len(data) == math.prod(knot_spec.num)

    L = max(ray_t.shape[1], ray_n.shape[1])
    # -----------------------------------------------------

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
