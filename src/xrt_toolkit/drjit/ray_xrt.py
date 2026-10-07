import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

from .bbox import bbox_contains, ray_bbox_intersect
from .dda import dda
from .spline2d import spline2d_adjoint, spline2d_apply
from .spline3d import (
    NO_CELL,
    spline3d_advance,
    spline3d_flush,
    spline3d_grid,
    spline3d_mirror,
    spline3d_stencil,
    spline3d_weights,
)

BoolT = typ.TypeVar("BoolT", bound=dr.AnyArray)
ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
ArrayNiT = typ.TypeVar("ArrayNiT", bound=dr.AnyArray)
ArrayNuT = typ.TypeVar("ArrayNuT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNfT, ArrayNfT]


def xrt_apply(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero, \bbQ-1}}
       f_{\bbq} \psi_(\bbx - \bbx_{\bbq}),

    with

    .. math::

       f_{\bbq} \in \bR,
       \bbx_{q} = \bbx_{0} + \bbq \odot \bbDelta,
           \bbx_{0} \in \bR^{D},
           \bbDelta \in \bR_{+}^{D},
       \psi(\bbx; \bbE \in \bR^{D \times N}) =
           box-spline with direction vectors
           \{ \bbe_{l} \in \bR^{D} \}_{l=1..N}

    Then ``xrt_apply()`` computes samples of

    .. math::

       \xrt[f](\bbn, \bbt) = \int_{\bR} f(\bbt + \alpha \bbn) d\alpha

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt \in \bR^{D}` and directions :math:`\bbn \in \bR^{D}`.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Data interpolation order.

        This parameter sets which :math:`\psi` is used to interpolate data values:

        * order = 0 (2D, 3D):

          .. math::

             \bbE = \diag(\bbDelta)

        * order = 1 (2D):

          .. math::

             \bbE = [\bbDelta_{1}           0  \bbDelta_{1}
                               0  \bbDelta_{2} \bbDelta_{2}]

        * order = 2 (2D):

          .. math::

             \bbE = [\bbDelta_{1}           0  \bbDelta_{1}  \bbDelta_{1}
                               0  \bbDelta_{2} \bbDelta_{2} -\bbDelta_{2}]

        * order = 1, 2 (3D): tensor-product B-splines (trilinear, triquadratic)

          .. math::

             \psi(\bbx)
             =
             \beta^{k}(x_{1} / \Delta_{1})
             \beta^{k}(x_{2} / \Delta_{2})
             \beta^{k}(x_{3} / \Delta_{3}),
             \quad
             \beta^{k} = \mathbb{1}_{[-1/2, 1/2]}^{\ast (k+1)},
             \quad
             k = \text{order}

        The support of :math:`\psi` and its projections can be viewed using :func:`~xrt_toolkit.drjit.diagnostics.plot_2d_basis`.

    data: FloatT
        (Q1,...,QD) flattened C-ordered volume weights :math:`f_{\bbq} \in \bR`.
    buffer: FloatT
        (L,) buffer in which to accumulate projections.

    Returns
    -------
    proj: FloatT
        (L,) projections :math:`\xrt[f] \in \bR`.
    """
    ray_t, ray_n = ray_spec

    ArrayNf = type(ray_t)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)

    # type checking ---------------------------------------
    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D in (2, 3))
    assert type(ray_n) is ArrayNf

    assert knot_spec.ndim == D
    assert order in (0, 1, 2)

    assert type(data) is Float
    assert len(data) == math.prod(knot_spec.num)

    L = max(ray_t.shape[1], ray_n.shape[1])
    if buffer is None:
        buffer = dr.zeros(Float, L)
    else:
        assert type(buffer) is Float
        assert len(buffer) == L
    # -----------------------------------------------------

    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = knot_start - (knot_step / 2) + (knot_num * knot_step)
    grid_res = knot_num
    if D == 2:
        stride = ArrayNu(knot_num.y, 1)
    elif D == 3:
        stride = ArrayNu(knot_num.y * knot_num.z, knot_num.z, 1)

    if order == 0:
        state = (buffer,)

        def project(
            state: tuple[FloatT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT], BoolT]:
            # compute analytic ray/cell projection.
            (accum,) = state

            offset = dr.dot(index, stride)
            fq = dr.gather(Float, data, offset, active)
            L = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            accum += fq * L

            return (accum,), Bool(True)

    elif D == 2:
        return spline2d_apply((ray_t, ray_n), knot_spec, order, data, buffer)

    elif (D == 3) and (order > 0):
        state = (buffer,)

        def project(
            state: tuple[FloatT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT], BoolT]:
            # exact ray/cell projections of the (order + 1)**3 basis functions.
            (accum,) = state

            for offset, valid, L in spline3d_stencil(
                order, index, p_a, p_b, knot_step, knot_num, stride
            ):
                fq = dr.gather(Float, data, offset, valid & active)
                accum += fq * L

            return (accum,), Bool(True)

        # walk the grid of polynomial pieces (covers every basis support).
        bbox_ll, bbox_ur, grid_res = spline3d_grid(
            knot_start, knot_step, knot_num, order
        )

    # dda() starts the walk from `ray_t`, but we want to start from the bbox boundary.
    # -> rewind `ray_t` for it to lie outside the bbox boundary.
    active, t1, t2 = ray_bbox_intersect(bbox_ll, bbox_ur, ray_t, ray_n)
    t_min = dr.minimum(t1, t2)
    ray_t = dr.select(
        active & bbox_contains(bbox_ll, bbox_ur, ray_t),
        ray_t + (t_min - 1) * ray_n,  # go a bit further to be truly outside bbox
        ray_t,
    )

    state = dda(
        ray_o=ray_t,
        ray_d=ray_n,
        ray_max=Float(dr.inf),
        grid_res=grid_res,
        grid_min=bbox_ll,
        grid_max=bbox_ur,
        func=project,
        state=state,
        active=active,
        mode="symbolic",
        max_iterations=-1,
    )

    return buffer


def xrt_adjoint(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
) -> FloatT:
    r"""
    Compute 2D/3D back-projections.

    Adjoint of ``xrt_apply()``: maps projection weights to volume expansion coefficients.

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt \in \bR^{D}` and directions :math:`\bbn \in \bR^{D}`.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Data interpolation order.

        This parameter sets which :math:`\psi` is used to interpolate data values.
        The support of :math:`\psi` and its projections can be viewed using :func:`~xrt_toolkit.drjit.diagnostics.plot_2d_basis`.
    data: FloatT
        (L,) projections :math:`g_{l} \in \bR`.
    buffer: FloatT
        (Q1,...,QD) flattened buffer in which to accumulate back-projected weights :math:`f_{\bbq} \in \bR`.

    Returns
    -------
    b_proj: FloatT
        (Q1,...,QD) flattened C-ordered back-projected weights :math:`f_{\bbq} \in \bR`.
    """
    ray_t, ray_n = ray_spec

    ArrayNf = type(ray_t)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)

    # type checking ---------------------------------------
    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D in (2, 3))
    assert type(ray_n) is ArrayNf

    assert knot_spec.ndim == D
    assert order in (0, 1, 2)

    L = max(ray_t.shape[1], ray_n.shape[1])
    assert type(data) is Float
    assert len(data) == L

    if buffer is None:
        buffer = dr.zeros(Float, math.prod(knot_spec.num))
    else:
        assert type(buffer) is Float
        assert len(buffer) == math.prod(knot_spec.num)
    # -----------------------------------------------------

    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = knot_start - (knot_step / 2) + (knot_num * knot_step)
    grid_res = knot_num
    if D == 2:
        stride = ArrayNu(knot_num.y, 1)
    elif D == 3:
        stride = ArrayNu(knot_num.y * knot_num.z, knot_num.z, 1)

    if order == 0:
        state = (buffer,)

        def back_project(
            state: tuple[FloatT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT], BoolT]:
            # compute analytic ray/cell back-projection.
            (accum,) = state

            offset = dr.dot(index, stride)
            L = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            dr.scatter_add(accum, L * data, offset, active, mode=dr.ReduceMode.Direct)

            return (accum,), Bool(True)

    elif D == 2:
        return spline2d_adjoint((ray_t, ray_n), knot_spec, order, data, buffer)

    elif (D == 3) and (order > 0):
        # running sums of the (order + 1)**3 basis functions overlapping the current
        # cell: each is written once, when the ray leaves its support (see spline3d.py).
        fwd = ray_n >= 0  # travel direction
        sums = [dr.zeros(Float, L) for _ in range((order + 1) ** 3)]
        cell = dr.full(dr.int32_array_t(ArrayNf), NO_CELL, L)
        state = (sums, cell)

        def back_project(
            state: tuple[list[FloatT], ArrayNiT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[list[FloatT], ArrayNiT], BoolT]:
            # exact ray/cell projections of the (order + 1)**3 basis functions.
            (sums, cell) = state

            sums, cell = spline3d_advance(
                order,
                sums,
                cell,
                type(cell)(index),
                active,
                fwd,
                knot_num,
                stride,
                data,
                buffer,
            )
            L = spline3d_weights(
                order,
                spline3d_mirror(fwd, p_a),
                spline3d_mirror(fwd, p_b),
                knot_step,
            )
            sums = [dr.select(active, s + l, s) for (s, l) in zip(sums, L)]

            return (sums, cell), Bool(True)

        # walk the grid of polynomial pieces (covers every basis support).
        bbox_ll, bbox_ur, grid_res = spline3d_grid(
            knot_start, knot_step, knot_num, order
        )

    # dda() starts the walk from `ray_t`, but we want to start from the bbox boundary.
    # -> rewind `ray_t` for it to lie outside the bbox boundary.
    active, t1, t2 = ray_bbox_intersect(bbox_ll, bbox_ur, ray_t, ray_n)
    t_min = dr.minimum(t1, t2)
    ray_t = dr.select(
        active & bbox_contains(bbox_ll, bbox_ur, ray_t),
        ray_t + (t_min - 1) * ray_n,  # go a bit further to be truly outside bbox
        ray_t,
    )

    state = dda(
        ray_o=ray_t,
        ray_d=ray_n,
        ray_max=Float(dr.inf),
        grid_res=grid_res,
        grid_min=bbox_ll,
        grid_max=bbox_ur,
        func=back_project,
        state=state,
        active=active,
        mode="symbolic",
        max_iterations=-1,
    )

    if (D == 3) and (order > 0):
        # write the running sums left at the end of the walk.
        (sums, cell) = state
        spline3d_flush(order, sums, cell, fwd, knot_num, stride, data, buffer)

    return buffer
