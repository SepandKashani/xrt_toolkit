import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

from .bbox import bbox_contains, ray_bbox_intersect
from .box_spline import box_spline_1d_dr, box_spline_1d_E, spline_3d_dr
from .dda import dda

BoolT = typ.TypeVar("BoolT", bound=dr.AnyArray)
ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
ArrayNiT = typ.TypeVar("ArrayNiT", bound=dr.AnyArray)
ArrayNuT = typ.TypeVar("ArrayNuT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNfT, ArrayNfT]

# Neural network for 3D spline projection
from pathlib import Path

import numpy as np
from drjit import nn as dnn
from drjit.cuda.ad import Float16, Float32, TensorXf16

# The 3D spline network needs fp16 support; construction can fail on
# exotic setups, in which case the portable fallback path is used
# (see box_spline.nn_project_plain) and `net` stays None.
try:
    rng_net = dr.rng(seed=0)
    net = dnn.Sequential(
        dnn.Linear(4, 16),
        dnn.ReLU(),
        dnn.Linear(16, 16),
        dnn.ReLU(),
        dnn.Linear(16, 16),
        dnn.ReLU(),
        dnn.Linear(16, 16),
        dnn.ReLU(),
        dnn.Linear(16, 1),
    )
    net = net.alloc(dtype=TensorXf16, size=4, rng=rng_net)
    _packed = dnn.pack(net, layout='training')
    if isinstance(_packed, tuple):  # drjit < 1.4 returns (weights, net)
        weights, net = _packed
    else:  # drjit >= 1.4 returns the packed module; the buffer is shared
        net = _packed
        weights = net.layers[0].weights.buffer

    weights_path = Path(__file__).parent / 'gpu_3D_spline_weights_drjit.npz'

    saved = np.load(weights_path)['weights']
    weights[:] = Float16(saved)
except Exception:
    net = None


eps = 1e-5


def xrt_apply(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode: str = "symbolic",
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
       f_{\bbq} \psi(\bbx - \bbx_{\bbq}),

    with

    .. math::

       f_{\bbq} \in \bR, \qquad
       \bbx_{\bbq} = \bbx_{0} + \bbq \odot \bbDelta, \qquad
       \bbx_{0} \in \bR^{D}, \qquad
       \bbDelta \in \bR_{+}^{D},

    where :math:`\psi` is a separable spline of degree `order` on the lattice.

    Then ``xrt_apply()`` computes samples of

    .. math::

       \xrt[f](\bbn, \bbt) = \int_{\bR} f(\bbt + \alpha \bbn) \, \mathrm{d}\alpha

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt \in \bR^{D}` and directions :math:`\bbn \in \bR^{D}`.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Degree of the spline :math:`\psi` on the lattice:

        * 0: voxels, so the matrix entries are chord lengths;
        * 1: linear;
        * 2: quadratic.

        Higher degrees are smoother and their projections are wider. In 3D,
        any degree above 0 is much slower than 0.

        :py:func:`~xrt_toolkit.plot_2d_basis` draws the support of
        :math:`\psi` and its projections.

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

    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = knot_start - (knot_step / 2) + (knot_num * knot_step)
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
            (accum,) = state
            offset = dr.dot(index, stride)
            fq = dr.gather(Float, data, offset, active)
            L = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            accum += fq * L
            return (accum,), Bool(True)

    elif D == 2:
        index_prev = ArrayNi(-1)
        state = (buffer, index_prev)

        n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
        E, E_mask = box_spline_1d_E(order, knot_step, n_perp)

        direction = ray_n * dr.rcp(knot_step)
        look_lr = dr.abs(direction.x) >= dr.abs(direction.y)
        mv_dir = dr.select(look_lr, ArrayNi(+1, 0), ArrayNi(0, +1))
        shift_l = dr.reverse(-mv_dir)
        shift_m = ArrayNi(0, 0)
        shift_r = dr.reverse(+mv_dir)

        def project(
            state: tuple[FloatT, ArrayNiT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT, ArrayNiT], BoolT]:
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
                index_s = index + shift
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))
                fq = dr.gather(Float, data, offset, active)
                cell_center = ArrayNf(0.5, 0.5) + shift
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)
                return fq, L

            fq_l, L_l = process_shift(shift_l)
            fq_m, L_m = process_shift(shift_m)
            fq_r, L_r = process_shift(shift_r)

            Array3f = xrtu.float_array_t(Float, 3)
            displacement = ArrayNi(index) - index_prev
            fq_lmr = dr.if_stmt(
                (fq_l, fq_m, fq_r),
                dr.dot(displacement, mv_dir) != 0,
                lambda l, m, r: Array3f(l, m, r),
                lambda l, m, r: dr.select(
                    dr.dot(displacement, dr.reverse(mv_dir)) == -1,
                    Array3f(l, 0, 0),
                    Array3f(0, 0, r),
                ),
            )
            accum += dr.dot(fq_lmr, Array3f(L_l, L_m, L_r))
            return (accum, ArrayNi(index)), Bool(True)

    elif (D == 3) and (order > 0):
        # TODO case where ray touches an edge : conditions are stricter

        n = dr.normalize(ray_n)
        index_prev = ArrayNi(-1)
        state = (buffer, index_prev)
        direction = dr.abs(ray_n * dr.rcp(knot_step))

        dominant_axis = dr.floor(direction / dr.max(direction) + 1e-6)
        shift_x = ArrayNf(dominant_axis.y, dominant_axis.z, dominant_axis.x)
        shift_y = dr.cross(dominant_axis, shift_x)

        n_perp_x = dr.normalize(shift_x - dr.dot(shift_x, n) * n)
        n_perp_y = dr.cross(n, n_perp_x)

        shift_m = ArrayNi(0, 0, 0)

        def project(
            state: tuple[FloatT, ArrayNiT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT, ArrayNiT], BoolT]:
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
                index_s = index + shift
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))
                fq = dr.gather(Float, data, offset, active)
                cell_center = ArrayNf(0.5, 0.5, 0.5) + shift
                x = dr.dot(n_perp_y, knot_step * (cell_center - p_a))
                y = dr.dot(n_perp_x, knot_step * (cell_center - p_a))
                L = spline_3d_dr(net, x, y, n)
                return fq, L

            fq_mlr, L_mlr = process_shift(shift_m)
            fq_mud, L_mud = process_shift(shift_m)
            fq_l, L_l = process_shift(shift_x)
            fq_r, L_r = process_shift(-shift_x)
            fq_d, L_d = process_shift(shift_y)
            fq_u, L_u = process_shift(-shift_y)
            fq_ll, L_ll = process_shift(shift_x + shift_y)
            fq_rr, L_rr = process_shift(-shift_x - shift_y)
            fq_lr, L_lr = process_shift(shift_x - shift_y)
            fq_rl, L_rl = process_shift(-shift_x + shift_y)

            Array3f = xrtu.float_array_t(Float, 3)
            displacement = ArrayNi(index) - index_prev

            fq_lmr = dr.if_stmt(
                (fq_l, fq_mlr, fq_r),
                dr.dot(displacement, shift_x) != 0,
                lambda l, m, r: dr.select(
                    dr.dot(displacement, shift_x) == -1,
                    Array3f(0, 0, r),
                    Array3f(l, 0, 0),
                ),
                lambda l, m, r: Array3f(l, m, r),
            )
            L_mlr = dr.select(dr.dot(displacement, shift_x) == 0, 0, L_mlr)
            L_l = dr.select(dr.dot(displacement, shift_y) != 0, 0, L_l)
            L_r = dr.select(dr.dot(displacement, shift_y) != 0, 0, L_r)
            L_lmr = Array3f(L_l, L_mlr, L_r)

            fq_dmu = dr.if_stmt(
                (fq_d, fq_mud, fq_u),
                dr.dot(displacement, shift_y) != 0,
                lambda d, m, u: dr.select(
                    dr.dot(displacement, shift_y) == -1,
                    Array3f(0, 0, u),
                    Array3f(d, 0, 0),
                ),
                lambda d, m, u: Array3f(d, m, u),
            )
            L_mud = L_mud - L_mlr
            L_d = dr.select(dr.dot(displacement, shift_x) != 0, 0, L_d)
            L_u = dr.select(dr.dot(displacement, shift_x) != 0, 0, L_u)
            L_dmu = Array3f(L_d, L_mud, L_u)

            accum += dr.dot(fq_dmu, L_dmu)
            accum += dr.dot(fq_lmr, L_lmr)

            fq_ll = dr.select(dr.dot(displacement, shift_y) != 0, 0, fq_ll)
            fq_ll = dr.select(dr.dot(displacement, shift_x) != 0, 0, fq_ll)
            fq_rr = dr.select(dr.dot(displacement, shift_y) != 0, 0, fq_rr)
            fq_rr = dr.select(dr.dot(displacement, shift_x) != 0, 0, fq_rr)
            fq_lr = dr.select(dr.dot(displacement, shift_y) != 0, 0, fq_lr)
            fq_lr = dr.select(dr.dot(displacement, shift_x) != 0, 0, fq_lr)
            fq_rl = dr.select(dr.dot(displacement, shift_y) != 0, 0, fq_rl)
            fq_rl = dr.select(dr.dot(displacement, shift_x) != 0, 0, fq_rl)

            accum += fq_ll * L_ll
            accum += fq_rr * L_rr
            accum += fq_lr * L_lr
            accum += fq_rl * L_rl

            return (accum, ArrayNi(index)), Bool(True)

    active, t1, t2 = ray_bbox_intersect(bbox_ll, bbox_ur, ray_t, ray_n)
    t_min = dr.minimum(t1, t2)
    ray_t = dr.select(
        active & bbox_contains(bbox_ll, bbox_ur, ray_t),
        ray_t + (t_min - 1) * ray_n,
        ray_t,
    )
    dda(
        ray_o=ray_t, ray_d=ray_n, ray_max=Float(dr.inf),
        grid_res=knot_num, grid_min=bbox_ll, grid_max=bbox_ur,
        func=project, state=state, active=active, mode=mode, max_iterations=-1,
    )
    return buffer


def xrt_adjoint(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode: str = "symbolic",
) -> FloatT:
    r"""
    Compute 2D/3D back-projections.

    Exact adjoint of ``xrt_apply()``: it maps projection weights back to volume
    expansion coefficients. Writing :math:`\bbA` for the matrix ``xrt_apply()``
    applies, with

    .. math::

       [\bbA]_{l \bbq}
       =
       \int_{\bR} \psi(\bbt_{l} + \alpha \bbn_{l} - \bbx_{\bbq})
       \, \mathrm{d}\alpha,

    this routine computes :math:`\bbA^{\top} y`, that is

    .. math::

       (\bbA^{\top} y)_{\bbq} = \sum_{l} [\bbA]_{l \bbq} \, y_{l}.

    The same traversal visits the same cells with the same weights as the
    forward pass and scatters instead of gathering, so
    :math:`\langle \bbA f, y \rangle = \langle f, \bbA^{\top} y \rangle` holds
    to machine precision.

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

    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = knot_start - (knot_step / 2) + (knot_num * knot_step)
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
            (accum,) = state
            offset = dr.dot(index, stride)
            L = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            dr.scatter_add(accum, L * data, offset, active)
            return (accum,), Bool(True)

    elif D == 2:
        index_prev = ArrayNi(-1)
        state = (buffer, index_prev)

        n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
        E, E_mask = box_spline_1d_E(order, knot_step, n_perp)

        direction = ray_n * dr.rcp(knot_step)
        look_lr = dr.abs(direction.x) >= dr.abs(direction.y)
        mv_dir = dr.select(look_lr, ArrayNi(+1, 0), ArrayNi(0, +1))
        shift_l = dr.reverse(-mv_dir)
        shift_m = ArrayNi(0, 0)
        shift_r = dr.reverse(+mv_dir)

        def back_project(
            state: tuple[FloatT, ArrayNiT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT, ArrayNiT], BoolT]:
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNuT, BoolT]:
                index_s = index + shift
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))
                cell_center = ArrayNf(0.5, 0.5) + shift
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)
                return L, offset, active

            L_l, offset_l, active_l = process_shift(shift_l)
            L_m, offset_m, active_m = process_shift(shift_m)
            L_r, offset_r, active_r = process_shift(shift_r)

            Array3b = dr.mask_t(xrtu.float_array_t(Float, 3))
            displacement = ArrayNi(index) - index_prev
            active_lmr = dr.if_stmt(
                (active_l, active_m, active_r),
                dr.dot(displacement, mv_dir) != 0,
                lambda l, m, r: Array3b(l, m, r),
                lambda l, m, r: dr.select(
                    dr.dot(displacement, dr.reverse(mv_dir)) == -1,
                    Array3b(l, False, False),
                    Array3b(False, False, r),
                ),
            )
            dr.scatter_add(accum, L_l * data, offset_l, active_lmr.x)
            dr.scatter_add(accum, L_m * data, offset_m, active_lmr.y)
            dr.scatter_add(accum, L_r * data, offset_r, active_lmr.z)
            return (accum, ArrayNi(index)), Bool(True)

    elif (D == 3) and (order > 0):
        # TODO case where ray touches an edge : conditions are stricter

        n = dr.normalize(ray_n)
        index_prev = ArrayNi(-1)
        state = (buffer, index_prev)
        direction = dr.abs(ray_n * dr.rcp(knot_step))

        dominant_axis = dr.floor(direction / dr.max(direction) + 1e-6)
        shift_x = ArrayNf(dominant_axis.y, dominant_axis.z, dominant_axis.x)
        shift_y = dr.cross(dominant_axis, shift_x)

        n_perp_x = dr.normalize(shift_x - dr.dot(shift_x, n) * n)
        n_perp_y = dr.cross(n, n_perp_x)

        shift_m = ArrayNi(0, 0, 0)

        def back_project(
            state: tuple[FloatT, ArrayNiT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT, ArrayNiT], BoolT]:
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNuT, BoolT]:
                index_s = index + shift
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))
                cell_center = ArrayNf(0.5, 0.5, 0.5) + shift
                x = dr.dot(n_perp_y, knot_step * (cell_center - p_a))
                y = dr.dot(n_perp_x, knot_step * (cell_center - p_a))
                L = spline_3d_dr(net, x, y, n)
                return L, offset, active

            L_mlr, offset_mlr, active_mlr = process_shift(shift_m)
            L_mud, offset_mud, active_mud = process_shift(shift_m)
            L_l, offset_l, active_l = process_shift(shift_x)
            L_r, offset_r, active_r = process_shift(-shift_x)
            L_d, offset_d, active_d = process_shift(shift_y)
            L_u, offset_u, active_u = process_shift(-shift_y)
            L_ll, offset_ll, active_ll = process_shift(shift_x + shift_y)
            L_rr, offset_rr, active_rr = process_shift(-shift_x - shift_y)
            L_lr, offset_lr, active_lr = process_shift(shift_x - shift_y)
            L_rl, offset_rl, active_rl = process_shift(-shift_x + shift_y)

            Array3b = dr.mask_t(xrtu.float_array_t(Float, 3))
            displacement = ArrayNi(index) - index_prev

            active_lmr = dr.if_stmt(
                (active_l, active_mlr, active_r),
                dr.dot(displacement, shift_x) != 0,
                lambda l, m, r: dr.select(
                    dr.dot(displacement, shift_x) == -1,
                    Array3b(False, False, r),
                    Array3b(l, False, False),
                ),
                lambda l, m, r: Array3b(l, m, r),
            )
            active_mlr = dr.select(dr.dot(displacement, shift_x) == 0, False, active_mlr)
            active_l = dr.select(dr.dot(displacement, shift_y) != 0, False, active_l)
            active_r = dr.select(dr.dot(displacement, shift_y) != 0, False, active_r)

            dr.scatter_add(accum, L_l * data, offset_l, active_lmr.x & active_l)
            dr.scatter_add(accum, L_mlr * data, offset_mlr, active_lmr.y & active_mlr)
            dr.scatter_add(accum, L_r * data, offset_r, active_lmr.z & active_r)

            active_dmu = dr.if_stmt(
                (active_d, active_mud, active_u),
                dr.dot(displacement, shift_y) != 0,
                lambda d, m, u: dr.select(
                    dr.dot(displacement, shift_y) == -1,
                    Array3b(False, False, u),
                    Array3b(d, False, False),
                ),
                lambda d, m, u: Array3b(d, m, u),
            )
            active_mud = active_mud & ~active_mlr
            active_d = dr.select(dr.dot(displacement, shift_x) != 0, False, active_d)
            active_u = dr.select(dr.dot(displacement, shift_x) != 0, False, active_u)

            dr.scatter_add(accum, L_d * data, offset_d, active_dmu.x & active_d)
            dr.scatter_add(accum, L_mud * data, offset_mud, active_dmu.y & active_mud)
            dr.scatter_add(accum, L_u * data, offset_u, active_dmu.z & active_u)

            active_ll = dr.select(dr.dot(displacement, shift_y) != 0, False, active_ll)
            active_ll = dr.select(dr.dot(displacement, shift_x) != 0, False, active_ll)
            active_rr = dr.select(dr.dot(displacement, shift_y) != 0, False, active_rr)
            active_rr = dr.select(dr.dot(displacement, shift_x) != 0, False, active_rr)
            active_lr = dr.select(dr.dot(displacement, shift_y) != 0, False, active_lr)
            active_lr = dr.select(dr.dot(displacement, shift_x) != 0, False, active_lr)
            active_rl = dr.select(dr.dot(displacement, shift_y) != 0, False, active_rl)
            active_rl = dr.select(dr.dot(displacement, shift_x) != 0, False, active_rl)

            dr.scatter_add(accum, L_ll * data, offset_ll, active_ll)
            dr.scatter_add(accum, L_rr * data, offset_rr, active_rr)
            dr.scatter_add(accum, L_lr * data, offset_lr, active_lr)
            dr.scatter_add(accum, L_rl * data, offset_rl, active_rl)

            return (accum, ArrayNi(index)), Bool(True)

    active, t1, t2 = ray_bbox_intersect(bbox_ll, bbox_ur, ray_t, ray_n)
    t_min = dr.minimum(t1, t2)
    ray_t = dr.select(
        active & bbox_contains(bbox_ll, bbox_ur, ray_t),
        ray_t + (t_min - 1) * ray_n,
        ray_t,
    )
    dda(
        ray_o=ray_t, ray_d=ray_n, ray_max=Float(dr.inf),
        grid_res=knot_num, grid_min=bbox_ll, grid_max=bbox_ur,
        func=back_project, state=state, active=active, mode=mode, max_iterations=-1,
    )
    return buffer


# ---------------------------------------------------------------------------
# Geometry gradients (2D only)
# ---------------------------------------------------------------------------
# Both xrt_ad_t_x/y and xrt_ad_n_x/y share the same DDA structure.
# The only per-component difference is which gradient of the projection weight
# L = box_spline_1d_dr(...) is extracted after a drjit backward pass.
#
# _xrt_ad_t(idx): ∂P/∂t_{idx}  where t is the ray anchor (offset)
#   grad = p_a.grad[idx]
#
# _xrt_ad_n(idx): ∂P/∂n_{idx}  where n is the ray direction
#   grad = ray_n.grad[idx] + fac_t * p_a.grad[idx]
#   (fac_t accounts for the implicit dependence of the entry point p_a on n)
# ---------------------------------------------------------------------------

def _xrt_ad_t(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT,
    mode: str,
    idx: int,
) -> FloatT:
    ray_t, ray_n = ray_spec

    ArrayNf = type(ray_t)
    ArrayNi = dr.int32_array_t(ArrayNf)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)

    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D == 2 or (D == 3 and order == 0))
    assert type(ray_n) is ArrayNf
    assert knot_spec.ndim == D
    assert order in (0, 1, 2)
    assert type(data) is Float
    assert len(data) == math.prod(knot_spec.num)

    L_size = max(ray_t.shape[1], ray_n.shape[1])
    if buffer is None:
        buffer = dr.zeros(Float, L_size)
    else:
        assert type(buffer) is Float
        assert len(buffer) == L_size

    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = knot_start - (knot_step / 2) + (knot_num * knot_step)
    if D == 2:
        stride = ArrayNu(knot_num.y, 1)
    elif D == 3:
        stride = ArrayNu(knot_num.y * knot_num.z, knot_num.z, 1)

    if D == 3 and order == 0:
        # 3D voxel case: closed-form gradient (paper Algorithm vox1).
        # sigma[idx] = (p_in on left face) + (p_out on left face)
        #            - (p_in on right face) - (p_out on right face)
        # g_s[idx]  = sigma[idx] / |d[idx]|
        state = (buffer,)

        def project(
            state: tuple[FloatT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT], BoolT]:
            (accum,) = state
            offset = dr.dot(index, stride)
            fq = dr.gather(Float, data, offset, active)

            sigma = (
                dr.select(p_a[idx] < eps, Float(1), Float(0))
                + dr.select(p_b[idx] < eps, Float(1), Float(0))
                - dr.select(p_a[idx] > 1.0 - eps, Float(1), Float(0))
                - dr.select(p_b[idx] > 1.0 - eps, Float(1), Float(0))
            )
            n_abs = dr.abs(ray_n[idx])
            # Face-transfer terms scale with |n| / n_idx (crossing positions
            # sit at ray parameter (face - t_idx) / n_idx and transfer
            # geometric length |n| d-alpha); with unit directions the |n|
            # factor is invisible, but the forward accepts any scale.
            # When n[idx]=0 the ray never crosses idx-faces; sigma=0 too → return 0
            n_norm = dr.norm(ray_n)
            g = dr.select(n_abs > eps, sigma * n_norm / n_abs, Float(0))
            accum += fq * g
            return (accum,), Bool(True)

    else:
        # 2D (any order): use drjit AD on the box-spline projection weight.
        index_prev = ArrayNi(-1)
        state = (buffer, index_prev)

        direction = ray_n * dr.rcp(knot_step)
        look_lr = dr.abs(direction.x) >= dr.abs(direction.y)
        mv_dir = dr.select(look_lr, ArrayNi(+1, 0), ArrayNi(0, +1))
        shift_l = dr.reverse(-mv_dir)
        shift_m = ArrayNi(0, 0)
        shift_r = dr.reverse(+mv_dir)

        def project(
            state: tuple[FloatT, ArrayNiT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT, ArrayNiT], BoolT]:
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
                index_s = index + shift
                offset = dr.dot(index_s, stride)
                in_bounds = dr.all((0 <= index_s) & (index_s < knot_num))
                fq = dr.gather(Float, data, offset, in_bounds)

                cell_center = ArrayNf(0.5, 0.5) + shift
                p_local = ArrayNf(p_a)
                dr.enable_grad(p_local)
                def contribution(p_a_, ray_n_):
                    n_perp = dr.normalize(ArrayNf(-ray_n_.y, ray_n_.x))
                    E, E_mask = box_spline_1d_E(order, knot_step, n_perp)
                    x = dr.dot(n_perp, knot_step * (cell_center - p_a_))
                    return box_spline_1d_dr(E, E_mask, x)
                dr.backward(contribution(p_local, ray_n),
                            flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                g = p_local.grad[idx]
                dr.disable_grad(p_local)
                return fq, g

            fq_l, L_l = process_shift(shift_l)
            fq_m, L_m = process_shift(shift_m)
            fq_r, L_r = process_shift(shift_r)

            Array3f = xrtu.float_array_t(Float, 3)
            displacement = ArrayNi(index) - index_prev
            fq_lmr = dr.if_stmt(
                (fq_l, fq_m, fq_r),
                dr.dot(displacement, mv_dir) != 0,
                lambda l, m, r: Array3f(l, m, r),
                lambda l, m, r: dr.select(
                    dr.dot(displacement, dr.reverse(mv_dir)) == -1,
                    Array3f(l, 0, 0),
                    Array3f(0, 0, r),
                ),
            )
            accum += dr.dot(fq_lmr, Array3f(L_l, L_m, L_r))
            return (accum, ArrayNi(index)), Bool(True)

    active, t1, t2 = ray_bbox_intersect(bbox_ll, bbox_ur, ray_t, ray_n)
    t_min = dr.minimum(t1, t2)
    ray_t = dr.select(
        active & bbox_contains(bbox_ll, bbox_ur, ray_t),
        ray_t + (t_min - 1) * ray_n,
        ray_t,
    )
    dda(
        ray_o=ray_t, ray_d=ray_n, ray_max=Float(dr.inf),
        grid_res=knot_num, grid_min=bbox_ll, grid_max=bbox_ur,
        func=project, state=state, active=active, mode=mode, max_iterations=-1,
    )
    return buffer


def _xrt_ad_n(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT,
    mode: str,
    idx: int,
    _handle_ties: bool = True,
) -> FloatT:
    ray_t, ray_n = ray_spec
    ray_t_init = dr.copy(ray_t)

    ArrayNf = type(ray_t)
    ArrayNi = dr.int32_array_t(ArrayNf)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)

    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D == 2 or (D == 3 and order == 0))
    assert type(ray_n) is ArrayNf
    assert knot_spec.ndim == D
    assert order in (0, 1, 2)
    assert type(data) is Float
    assert len(data) == math.prod(knot_spec.num)

    if (D == 2) and _handle_ties:
        # The 2D projection model is continuous but kinked in `ray_n` on two
        # surfaces (d = n / step): the lattice diagonals |d.x| == |d.y| and
        # the lattice axes min(|d.x|, |d.y|) == 0.  On both, a |dot(n_perp,
        # .)| entry of the box-spline generator E crosses its |.| kink and
        # the E-mask degeneracy cut engages (the diagonal also flips
        # `look_lr`).  Differentiating there returns one arbitrary one-sided
        # slope, wrong by up to O(1) for orders >= 1.  (Rays built from
        # cos/sin are microscopically off the vertical axis -- cos(pi/2) is
        # 6e-17, not 0 -- which puts them on one side of the axis kink;
        # exactly-zero components hit dr.abs' sign(0) = 0 convention, which
        # happens to drop the kink term and give the right answer.  The tie
        # handling makes both cases exact by construction.)
        # On tie lanes, evaluate the derivative on both sides of the surface
        # (directions rotated by -/+ delta) and average: the two-sided
        # (Clarke) derivative of the kinked forward, which is what a central
        # finite difference converges to.  Off-tie lanes are bit-identical to
        # the plain evaluation, and tie-free calls take the plain path.
        # (xrt_ad_t_* needs no such handling: perturbing the anchor cannot
        # move the direction across either surface.)
        d = ray_n * dr.rcp(ArrayNf(*knot_spec.step))
        abs_dx = dr.abs(d.x)
        abs_dy = dr.abs(d.y)
        tol = max(1e-9, 8 * dr.epsilon(Float))
        big = dr.maximum(abs_dx, abs_dy)
        tie_diag = dr.abs(abs_dx - abs_dy) <= tol * big
        tie_axis = dr.minimum(abs_dx, abs_dy) <= tol * big
        tie = tie_diag | tie_axis
        if bool(dr.any(tie)):  # decided per call, before any kernel is traced
            delta = max(1e-11, 32 * dr.epsilon(Float))  # rotation angle [rad]
            n_p = dr.select(
                tie,
                ArrayNf(ray_n.x - delta * ray_n.y, ray_n.y + delta * ray_n.x),
                ray_n,
            )
            n_m = dr.select(
                tie,
                ArrayNf(ray_n.x + delta * ray_n.y, ray_n.y - delta * ray_n.x),
                ray_n,
            )
            g_p = _xrt_ad_n(
                (ray_t, n_p), knot_spec, order, data, None, mode, idx,
                _handle_ties=False,
            )
            g_m = _xrt_ad_n(
                (ray_t, n_m), knot_spec, order, data, None, mode, idx,
                _handle_ties=False,
            )
            g = 0.5 * (g_p + g_m)
            return g if buffer is None else buffer + g

    L_size = max(ray_t.shape[1], ray_n.shape[1])
    if buffer is None:
        buffer = dr.zeros(Float, L_size)
    else:
        assert type(buffer) is Float
        assert len(buffer) == L_size

    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = knot_start - (knot_step / 2) + (knot_num * knot_step)
    if D == 2:
        stride = ArrayNu(knot_num.y, 1)
    elif D == 3:
        stride = ArrayNu(knot_num.y * knot_num.z, knot_num.z, 1)

    if D == 3 and order == 0:
        # 3D voxel case: closed-form gradient (paper Algorithm vox2).
        # g_s[i]       = sigma[i] / |d[i]|          (same as Algorithm vox1)
        # sigma_out[i] = indicator of exit face for component i
        # local_g_d[i] = L_fw * sigma_out[i] / |d[i]|
        # g[i]         = local_g_d[i] + fac_t * g_s[i]
        state = (buffer,)

        def project(
            state: tuple[FloatT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT], BoolT]:
            (accum,) = state
            offset = dr.dot(index, stride)
            fq = dr.gather(Float, data, offset, active)
            L_fw = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            sigma = (
                dr.select(p_a[idx] < eps, Float(1), Float(0))
                + dr.select(p_b[idx] < eps, Float(1), Float(0))
                - dr.select(p_a[idx] > 1.0 - eps, Float(1), Float(0))
                - dr.select(p_b[idx] > 1.0 - eps, Float(1), Float(0))
            )
            sigma_out = (
                dr.select(p_b[idx] < eps, Float(1), Float(0))
                - dr.select(p_b[idx] > 1.0 - eps, Float(1), Float(0))
            )
            n_abs = dr.abs(ray_n[idx])
            # Face-transfer terms scale with |n| / n_idx, as in _xrt_ad_t.
            # When n[idx]=0 the ray never crosses idx-faces; sigma=0 too → return 0
            n_norm = dr.norm(ray_n)
            # sigma encodes sign(n_idx) (1_in - 1_out); the crossing-motion
            # term is |n| alpha_in (1_in - 1_out) / n_idx, while the exit
            # chord term -L_fw 1_out / n_idx carries no |n| factor.
            g_s       = dr.select(n_abs > eps, sigma * n_norm / n_abs, Float(0))
            local_g_d = dr.select(n_abs > eps, L_fw * sigma_out / n_abs, Float(0))
            p_a_phys = bbox_ll + (ArrayNf(index) + p_a) * knot_step
            # Dot product is robust for any direction (including axis-aligned).
            # For unit ray_n: fac_t = (p_a_phys - t_init) · n  (ray parameter at p_a).
            n_norm_sq = dr.squared_norm(ray_n)
            fac_t = dr.select(
                n_norm_sq > eps * eps,
                dr.dot(p_a_phys - ray_t_init, ray_n) / n_norm_sq,
                Float(0),
            )
            # The face-counting terms above differentiate the
            # alpha-parameterized transform (per-cell weights 1/|n|), whose
            # gradient carries a radial component -P n / |n|^2.  The shipped
            # forward is the geometric (scale-invariant) one -- its radial
            # derivative is exactly zero -- so add the per-cell counterpart
            # of that term back (it telescopes to +P n_idx / |n|^2).
            radial = dr.select(
                n_norm_sq > eps * eps,
                L_fw * ray_n[idx] / n_norm_sq,
                Float(0),
            )
            g = local_g_d + fac_t * g_s + radial
            accum += fq * g
            return (accum,), Bool(True)

    else:
        index_prev = ArrayNi(-1)
        state = (buffer, index_prev)

        direction = ray_n * dr.rcp(knot_step)
        look_lr = dr.abs(direction.x) >= dr.abs(direction.y)
        mv_dir = dr.select(look_lr, ArrayNi(+1, 0), ArrayNi(0, +1))
        shift_l = dr.reverse(-mv_dir)
        shift_m = ArrayNi(0, 0)
        shift_r = dr.reverse(+mv_dir)

        def project(
            state: tuple[FloatT, ArrayNiT],
            index: ArrayNuT,
            p_a: ArrayNfT,
            p_b: ArrayNfT,
            active: BoolT,
        ) -> tuple[tuple[FloatT, ArrayNiT], BoolT]:
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
                index_s = index + shift
                offset = dr.dot(index_s, stride)
                in_bounds = dr.all((0 <= index_s) & (index_s < knot_num))
                fq = dr.gather(Float, data, offset, in_bounds)

                cell_center = ArrayNf(0.5, 0.5) + shift
                # ray parameter of the cell entry point (robust for any direction)
                p_a_phys = bbox_ll + (ArrayNf(index) + p_a) * knot_step
                n_norm_sq = dr.squared_norm(ray_n)
                fac_t = dr.select(
                    n_norm_sq > eps * eps,
                    dr.dot(p_a_phys - ray_t_init, ray_n) / n_norm_sq,
                    Float(0),
                )

                # differentiate LOCAL copies: enabling grad on the captured
                # loop-external ray_n breaks symbolic tracing
                n_local = ArrayNf(ray_n)
                p_local = ArrayNf(p_a)
                dr.enable_grad(n_local, p_local)
                def contribution(p_a_, ray_n_):
                    n_perp = dr.normalize(ArrayNf(-ray_n_.y, ray_n_.x))
                    E, E_mask = box_spline_1d_E(order, knot_step, n_perp)
                    x = dr.dot(n_perp, knot_step * (cell_center - p_a_))
                    return box_spline_1d_dr(E, E_mask, x)
                dr.backward(contribution(p_local, n_local),
                            flags=dr.ADFlag.Default | dr.ADFlag.AllowNoGrad)
                g = n_local.grad[idx] + fac_t * p_local.grad[idx]
                dr.disable_grad(n_local, p_local)
                return fq, g

            fq_l, L_l = process_shift(shift_l)
            fq_m, L_m = process_shift(shift_m)
            fq_r, L_r = process_shift(shift_r)

            Array3f = xrtu.float_array_t(Float, 3)
            displacement = ArrayNi(index) - index_prev
            fq_lmr = dr.if_stmt(
                (fq_l, fq_m, fq_r),
                dr.dot(displacement, mv_dir) != 0,
                lambda l, m, r: Array3f(l, m, r),
                lambda l, m, r: dr.select(
                    dr.dot(displacement, dr.reverse(mv_dir)) == -1,
                    Array3f(l, 0, 0),
                    Array3f(0, 0, r),
                ),
            )
            accum += dr.dot(fq_lmr, Array3f(L_l, L_m, L_r))
            return (accum, ArrayNi(index)), Bool(True)

    active, t1, t2 = ray_bbox_intersect(bbox_ll, bbox_ur, ray_t, ray_n)
    t_min = dr.minimum(t1, t2)
    ray_t = dr.select(
        active & bbox_contains(bbox_ll, bbox_ur, ray_t),
        ray_t + (t_min - 1) * ray_n,
        ray_t,
    )
    dda(
        ray_o=ray_t, ray_d=ray_n, ray_max=Float(dr.inf),
        grid_res=knot_num, grid_min=bbox_ll, grid_max=bbox_ur,
        func=project, state=state, active=active, mode=mode, max_iterations=-1,
    )
    return buffer


def xrt_ad_t_x(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode: str = "symbolic",
) -> FloatT:
    r"""
    Compute per-ray :math:`\partial \mathcal{P} / \partial t_x` for 2D projections.

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt` and directions :math:`\bbn`.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Data interpolation order.
    data: FloatT
        (Q1, Q2) flattened volume weights.
    buffer: FloatT
        (L,) output buffer.

    Returns
    -------
    FloatT
        (L,) values of :math:`\partial \mathcal{P} / \partial t_x`.
    """
    return _xrt_ad_t(ray_spec, knot_spec, order, data, buffer, mode, idx=0)


def xrt_ad_t_y(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode: str = "symbolic",
) -> FloatT:
    r"""
    Compute per-ray :math:`\partial \mathcal{P} / \partial t_y` for 2D projections.

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt` and directions :math:`\bbn`.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Data interpolation order.
    data: FloatT
        (Q1, Q2) flattened volume weights.
    buffer: FloatT
        (L,) output buffer.

    Returns
    -------
    FloatT
        (L,) values of :math:`\partial \mathcal{P} / \partial t_y`.
    """
    return _xrt_ad_t(ray_spec, knot_spec, order, data, buffer, mode, idx=1)


def xrt_ad_n_x(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode: str = "symbolic",
) -> FloatT:
    r"""
    Compute per-ray :math:`\partial \mathcal{P} / \partial n_x` for 2D projections.

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt` and directions :math:`\bbn`.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Data interpolation order.
    data: FloatT
        (Q1, Q2) flattened volume weights.
    buffer: FloatT
        (L,) output buffer.

    Returns
    -------
    FloatT
        (L,) values of :math:`\partial \mathcal{P} / \partial n_x`.
    """
    return _xrt_ad_n(ray_spec, knot_spec, order, data, buffer, mode, idx=0)


def xrt_ad_n_y(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode: str = "symbolic",
) -> FloatT:
    r"""
    Compute per-ray :math:`\partial \mathcal{P} / \partial n_y` for 2D projections.

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt` and directions :math:`\bbn`.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Data interpolation order.
    data: FloatT
        (Q1, Q2) flattened volume weights.
    buffer: FloatT
        (L,) output buffer.

    Returns
    -------
    FloatT
        (L,) values of :math:`\partial \mathcal{P} / \partial n_y`.
    """
    return _xrt_ad_n(ray_spec, knot_spec, order, data, buffer, mode, idx=1)


def xrt_ad_t_z(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode: str = "symbolic",
) -> FloatT:
    r"""
    Compute per-ray :math:`\partial \mathcal{P} / \partial t_z` for 3D projections (order=0 only).

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt` and directions :math:`\bbn`.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0
        Data interpolation order (only 0 supported for 3D).
    data: FloatT
        (Q1, Q2, Q3) flattened volume weights.
    buffer: FloatT
        (L,) output buffer.

    Returns
    -------
    FloatT
        (L,) values of :math:`\partial \mathcal{P} / \partial t_z`.
    """
    return _xrt_ad_t(ray_spec, knot_spec, order, data, buffer, mode, idx=2)


def xrt_ad_n_z(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode: str = "symbolic",
) -> FloatT:
    r"""
    Compute per-ray :math:`\partial \mathcal{P} / \partial n_z` for 3D projections (order=0 only).

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt` and directions :math:`\bbn`.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0
        Data interpolation order (only 0 supported for 3D).
    data: FloatT
        (Q1, Q2, Q3) flattened volume weights.
    buffer: FloatT
        (L,) output buffer.

    Returns
    -------
    FloatT
        (L,) values of :math:`\partial \mathcal{P} / \partial n_z`.
    """
    return _xrt_ad_n(ray_spec, knot_spec, order, data, buffer, mode, idx=2)
