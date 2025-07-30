import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

from .bbox import bbox_contains, ray_bbox_intersect
from .box_spline import box_spline_1d_dr
from .dda import dda

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
    ArrayNi = dr.int32_array_t(ArrayNf)
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
    if D == 2:
        stride = ArrayNu(knot_num.y, 1)
    elif D == 3:
        stride = ArrayNu(knot_num.y * knot_num.z, knot_num.z, 1)

    state = (buffer,)
    if order == 0:

        def project(
            state: tuple[FloatT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT], BoolT]:
            # compute analytic ray/cell projection.
            (accum,) = state

            offset = index @ stride
            fq = dr.gather(Float, data, offset, active)
            L = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            accum += fq * L

            return (accum,), Bool(True)

    elif D == 2:
        # compute (E, E_mask) for box_spline_1d_dr()
        Array4f = xrtu.float_array_t(Float, 4)
        ray_n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
        to_1d = lambda _: dr.abs(ray_n_perp @ (knot_step * _))
        E = Array4f(
            to_1d(ArrayNf(+1, +0)),
            to_1d(ArrayNf(+0, +1)),
            to_1d(ArrayNf(+1, +1)),
            to_1d(ArrayNf(+1, -1)),
        )
        if order == 1:
            E = E.xyz

        E_mask_t = dr.int_array_t(E)
        E_mask = dr.select(E <= 1e-3, E_mask_t(0), E_mask_t(1))

        def project(
            state: tuple[FloatT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT], BoolT]:
            # compute analytic ray<>box-spline projection.
            (accum,) = state

            ray_n = p_b - p_a  # local coordinates
            n_perp = dr.normalize(  # global coordinates
                dr.reverse(knot_step * ray_n * ArrayNf(1, -1))
            )

            direction = dr.abs(ray_n.x) >= dr.abs(ray_n.y)
            shift_l = dr.select(direction, ArrayNi(0, -1), ArrayNi(-1, 0))
            shift_m = ArrayNi(0, 0)
            shift_r = dr.select(direction, ArrayNi(0, +1), ArrayNi(+1, 0))

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
                index_s = index + shift  # "_s" = shifted
                offset = index_s @ stride
                active = dr.all((0 <= index_s) & (index_s < knot_num))
                fq = dr.gather(Float, data, offset, active)

                cell_center = ArrayNf(0.5, 0.5) + shift
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)

                return (fq, L)

            (fq_l, L_l) = process_shift(shift_l)
            (fq_m, L_m) = process_shift(shift_m)
            (fq_r, L_r) = process_shift(shift_r)
            accum += (fq_l * L_l) + (fq_m * L_m) + (fq_r * L_r)

            return (accum,), Bool(True)

    elif (D == 3) and (order > 0):
        raise NotImplementedError  # todo

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
        grid_res=knot_num,
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
    ArrayNi = dr.int32_array_t(ArrayNf)
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
    if D == 2:
        stride = ArrayNu(knot_num.y, 1)
    elif D == 3:
        stride = ArrayNu(knot_num.y * knot_num.z, knot_num.z, 1)

    state = (buffer,)
    if order == 0:

        def back_project(
            state: tuple[FloatT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT], BoolT]:
            # compute analytic ray/cell back-projection.
            (accum,) = state

            offset = index @ stride
            L = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            dr.scatter_add(accum, L * data, offset, active)

            return (accum,), Bool(True)

    elif D == 2:
        # compute (E, E_mask) for box_spline_1d_dr()
        Array4f = xrtu.float_array_t(Float, 4)
        ray_n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
        to_1d = lambda _: dr.abs(ray_n_perp @ (knot_step * _))
        E = Array4f(
            to_1d(ArrayNf(+1, +0)),
            to_1d(ArrayNf(+0, +1)),
            to_1d(ArrayNf(+1, +1)),
            to_1d(ArrayNf(+1, -1)),
        )
        if order == 1:
            E = E.xyz

        E_mask_t = dr.int_array_t(E)
        E_mask = dr.select(E <= 1e-3, E_mask_t(0), E_mask_t(1))

        def back_project(
            state: tuple[FloatT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT], BoolT]:
            # compute analytic ray<>box-spline back-projection.
            (accum,) = state

            ray_n = p_b - p_a  # local coordinates
            n_perp = dr.normalize(  # global coordinates
                dr.reverse(knot_step * ray_n * ArrayNf(1, -1))
            )

            direction = dr.abs(ray_n.x) >= dr.abs(ray_n.y)
            shift_l = dr.select(direction, ArrayNi(0, -1), ArrayNi(-1, 0))
            shift_m = ArrayNi(0, 0)
            shift_r = dr.select(direction, ArrayNi(0, +1), ArrayNi(+1, 0))

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNuT, BoolT]:
                index_s = index + shift  # "_s" = shifted
                offset = index_s @ stride
                active = dr.all((0 <= index_s) & (index_s < knot_num))

                cell_center = ArrayNf(0.5, 0.5) + shift
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)

                return (L, offset, active)

            (L_l, offset_l, active_l) = process_shift(shift_l)
            (L_m, offset_m, active_m) = process_shift(shift_m)
            (L_r, offset_r, active_r) = process_shift(shift_r)
            dr.scatter_add(accum, L_l * data, offset_l, active_l)
            dr.scatter_add(accum, L_m * data, offset_m, active_m)
            dr.scatter_add(accum, L_r * data, offset_r, active_r)

            return (accum,), Bool(True)

    elif (D == 3) and (order > 0):
        raise NotImplementedError  # todo

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
        grid_res=knot_num,
        grid_min=bbox_ll,
        grid_max=bbox_ur,
        func=back_project,
        state=state,
        active=active,
        mode="symbolic",
        max_iterations=-1,
    )

    return buffer
