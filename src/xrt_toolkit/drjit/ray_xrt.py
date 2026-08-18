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

# ----- Neural network for 3D spline projection -----
from drjit import nn as dnn
from pathlib import Path
from drjit.cuda.ad import Float32 , Float16 , TensorXf16
import numpy as np

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
        dnn.Linear(16, 1)
    )
    net = net.alloc(dtype=TensorXf16, size=4, rng=rng_net)
    _packed = dnn.pack(net, layout='training')
    if isinstance(_packed, tuple):  # drjit < 1.4 returns (weights, net)
        weights, net = _packed
    else:  # drjit >= 1.4 returns the packed module; the buffer is shared
        net = _packed
        weights = net.layers[0].weights.buffer

    weights_path = Path(__file__).parent / 'gpu_3D_spline_weights_drjit.npz'
    if not weights_path.exists():  # legacy fallback: current working directory
        weights_path = Path('gpu_3D_spline_weights_drjit.npz')
    saved = np.load(weights_path)['weights']
    weights[:] = Float16(saved)
except Exception:
    net = None
# -----------------------------------------------------

eps = 1e-5

_TOF_INV_SQRT2 = 0.7071067811865476
_TOF_INV_SQRT2PI = 0.3989422804014327


def _tof_setup(tof, ray_t, ray_n, bbox_ll, knot_step):
    r"""
    Build per-ray TOF weight evaluators from a :py:class:`~xrt_toolkit.util.TOFSpec`.

    Returns ``(chord_weight, center_weight)``:

    * ``chord_weight(index, p_a, p_b)``: integral of the normalized Gaussian
      TOF kernel over the chord ``[p_a, p_b]`` of cell ``index``
      (exact; used for order 0).
    * ``center_weight(center)``: Gaussian TOF density evaluated at the
      absolute grid position ``center`` of a basis function (orders >= 1).

    Arc lengths are measured from the user-supplied ray anchor along the
    normalized ray direction, in the length unit of `knot_spec`.
    """
    ArrayNf = type(ray_t)
    Float = dr.value_t(ArrayNf)

    center, sigma = (tof.center, tof.sigma) if hasattr(tof, "center") else tof
    mu = Float(center)
    inv_sigma = dr.rcp(Float(sigma))
    n_hat = dr.normalize(ray_n)
    t_user = ArrayNf(ray_t)  # snapshot: anchors before the bbox rewind

    def alpha(p):  # arc length of grid position `p` from the user anchor
        x = bbox_ll + p * knot_step
        return dr.dot(x - t_user, n_hat)

    def chord_weight(index, p_a, p_b):
        z_a = (alpha(index + p_a) - mu) * inv_sigma
        z_b = (alpha(index + p_b) - mu) * inv_sigma
        return 0.5 * dr.abs(
            dr.erf(z_b * _TOF_INV_SQRT2) - dr.erf(z_a * _TOF_INV_SQRT2)
        )

    def center_weight(center):
        z = (alpha(center) - mu) * inv_sigma
        return dr.exp(-0.5 * dr.square(z)) * inv_sigma * _TOF_INV_SQRT2PI

    return chord_weight, center_weight


def xrt_apply(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode="symbolic",
    tof=None,
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
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
    tof: TOFSpec | tuple[FloatT, FloatT] | None
        Optional time-of-flight kernel ``(center, sigma)``.

        When given, each line integral is weighted by a normalized Gaussian
        of standard deviation `sigma` centered at arc length `center`,
        measured from the anchor :math:`\bbt` along the normalized direction
        :math:`\hat{\bbn}`. See :py:class:`~xrt_toolkit.util.TOFSpec`.

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

    if tof is not None:
        tof_chord_w, tof_center_w = _tof_setup(tof, ray_t, ray_n, bbox_ll, knot_step)

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
            if tof is None:
                L = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            else:
                L = tof_chord_w(ArrayNf(index), p_a, p_b) * dr.rcp(dr.prod(knot_step))
            accum += fq * L

            return (accum,), Bool(True)

    elif D == 2:
        index_prev = ArrayNi(-1)  # previous visited cell
        state = (buffer, index_prev)

        # compute (E, E_mask) for box_spline_1d_dr()
        n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
        E, E_mask = box_spline_1d_E(order, knot_step, n_perp)

        # (main, lateral) movement direction
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
            # compute analytic ray<>box-spline projection.
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
                index_s = index + shift  # "_s" = shifted
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))
                fq = dr.gather(Float, data, offset, active)

                cell_center = ArrayNf(0.5, 0.5) + shift
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)
                if tof is not None:
                    L = L * tof_center_w(ArrayNf(index) + cell_center)

                return (fq, L)

            (fq_l, L_l) = process_shift(shift_l)
            (fq_m, L_m) = process_shift(shift_m)
            (fq_r, L_r) = process_shift(shift_r)

            # mask updates depending on inter-cell displacement
            Array3f = xrtu.float_array_t(Float, 3)
            displacement = ArrayNi(index) - index_prev
            fq_lmr = dr.if_stmt(
                (fq_l, fq_m, fq_r),
                dr.dot(displacement, mv_dir) != 0,  # going in mv_dir
                lambda l, m, r: Array3f(l, m, r),
                lambda l, m, r: dr.select(
                    dr.dot(displacement, dr.reverse(mv_dir)) == -1,  # going left
                    Array3f(l, 0, 0),
                    Array3f(0, 0, r),
                ),
            )
            L_lmr = Array3f(L_l, L_m, L_r)

            accum += dr.dot(fq_lmr, L_lmr)

            return (accum, ArrayNi(index)), Bool(True)

    elif (D == 3) and (order > 0):
        
        # TODO case where ray touches an edge : conditions are stricter
        
        n = dr.normalize(ray_n) # cos theta, sin theta, 0 (for parallel beam)
        index_prev = ArrayNi(-1)  # previous visited cell
        state = (buffer, index_prev)
        direction = dr.abs(ray_n * dr.rcp(knot_step))

        dominant_axis = dr.floor(direction / dr.max(direction) + 1e-6) # floating precision issues TODO case 110, 011, 101 is diagonal (choose one randomly)
        shift_x = ArrayNf(dominant_axis.y, dominant_axis.z, dominant_axis.x)
        shift_y = dr.cross(dominant_axis, shift_x)

        # gram-schmidt to get n_perp_x and n_perp_y
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
            # compute analytic ray<>box-spline projection.
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
                index_s = index + shift  # "_s" = shifted
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))
                fq = dr.gather(Float, data, offset, active)

                cell_center = ArrayNf(0.5, 0.5, 0.5) + shift
                x = dr.dot(n_perp_y, knot_step * (cell_center - p_a)) # horizontal position of detector
                y = dr.dot(n_perp_x, knot_step * (cell_center - p_a)) # vertical position of detector

                L = spline_3d_dr(net, x, y, n)
                if tof is not None:
                    L = L * tof_center_w(ArrayNf(index) + cell_center)

                return (fq, L)

            (fq_mlr, L_mlr) = process_shift(shift_m)
            (fq_mud, L_mud) = process_shift(shift_m)
            (fq_l, L_l) = process_shift(shift_x)
            (fq_r, L_r) = process_shift(-shift_x)
            (fq_d, L_d) = process_shift(shift_y)
            (fq_u, L_u) = process_shift(-shift_y)

            (fq_ll, L_ll) = process_shift(shift_x + shift_y)
            (fq_rr, L_rr) = process_shift(-shift_x - shift_y)
            (fq_lr, L_lr) = process_shift(shift_x - shift_y)
            (fq_rl, L_rl) = process_shift(-shift_x + shift_y)

            Array3f = xrtu.float_array_t(Float, 3)
            displacement = ArrayNi(index) - index_prev
            
            fq_lmr = dr.if_stmt(
                (fq_l, fq_mlr, fq_r),
                dr.dot(displacement, shift_x) != 0,  # going in mv_dir
                lambda l, m, r: dr.select(
                    dr.dot(displacement, shift_x) == -1,  # going left
                    Array3f(0, 0, r),
                    Array3f(l, 0, 0),
                ),
                lambda l, m, r: Array3f(l, m, r),
            )

            L_mlr = dr.select(
                dr.dot(displacement, shift_x) == 0,
                0,
                L_mlr,
            )
            L_l = dr.select(
                dr.dot(displacement, shift_y) != 0,
                0,
                L_l,
            )
            L_r = dr.select(
                dr.dot(displacement, shift_y) != 0,
                0,
                L_r,
            )
            L_lmr = Array3f(L_l, L_mlr, L_r)

            fq_dmu = dr.if_stmt(
                (fq_d, fq_mud, fq_u),

                dr.dot(displacement, shift_y) != 0,  # going in mv_dir
                lambda d, m, u: dr.select(
                    dr.dot(displacement, shift_y) == -1,  # going down
                    Array3f(0, 0, u),
                    Array3f(d, 0, 0),
                ),
                lambda d, m, u: Array3f(d, m, u),
            )

            L_mud = L_mud - L_mlr
            L_d = dr.select(
                dr.dot(displacement, shift_x) != 0,
                0,
                L_d,
            )
            L_u = dr.select(
                dr.dot(displacement, shift_x) != 0,
                0,
                L_u,
            )

            L_dmu = Array3f(L_d, L_mud, L_u)


            accum += dr.dot(fq_dmu, L_dmu) 
            accum += dr.dot(fq_lmr, L_lmr)

            fq_ll = dr.select(
                dr.dot(displacement, shift_y) != 0,
                0,
                fq_ll,
            )
            fq_ll = dr.select(
                dr.dot(displacement, shift_x) != 0,
                0,
                fq_ll,
            )
            fq_rr = dr.select(
                dr.dot(displacement, shift_y) != 0,
                0,
                fq_rr,
            )
            fq_rr = dr.select(
                dr.dot(displacement, shift_x) != 0,
                0,
                fq_rr,
            )
            fq_lr = dr.select(
                dr.dot(displacement, shift_y) != 0,
                0,
                fq_lr,
            )
            fq_lr = dr.select(
                dr.dot(displacement, shift_x) != 0,
                0,
                fq_lr,
            )
            fq_rl = dr.select(
                dr.dot(displacement, shift_y) != 0,
                0,
                fq_rl,
            )
            fq_rl = dr.select(
                dr.dot(displacement, shift_x) != 0,
                0,
                fq_rl,
            )

            accum += fq_ll * L_ll 
            accum += fq_rr * L_rr
            accum += fq_lr * L_lr
            accum += fq_rl * L_rl

            return (accum, ArrayNi(index)), Bool(True)

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
        mode=mode,
        max_iterations=-1,
    )

    return buffer


def xrt_adjoint(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
    mode="symbolic",
    tof=None,
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
    tof: TOFSpec | tuple[FloatT, FloatT] | None
        Optional time-of-flight kernel ``(center, sigma)``, identical to the
        `tof` parameter of :py:func:`xrt_apply`. When given, the operator is
        the exact adjoint of the TOF-weighted forward projection.

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

    if tof is not None:
        tof_chord_w, tof_center_w = _tof_setup(tof, ray_t, ray_n, bbox_ll, knot_step)

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
            if tof is None:
                L = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            else:
                L = tof_chord_w(ArrayNf(index), p_a, p_b) * dr.rcp(dr.prod(knot_step))
            dr.scatter_add(accum, L * data, offset, active)

            return (accum,), Bool(True)

    elif D == 2:
        index_prev = ArrayNi(-1)  # previous visited cell
        state = (buffer, index_prev)

        # compute (E, E_mask) for box_spline_1d_dr()
        n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
        E, E_mask = box_spline_1d_E(order, knot_step, n_perp)

        # (main, lateral) movement direction
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
            # compute analytic ray<>box-spline back-projection.
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNuT, BoolT]:
                index_s = index + shift  # "_s" = shifted
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))

                cell_center = ArrayNf(0.5, 0.5) + shift
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)
                if tof is not None:
                    L = L * tof_center_w(ArrayNf(index) + cell_center)

                return (L, offset, active)

            (L_l, offset_l, active_l) = process_shift(shift_l)
            (L_m, offset_m, active_m) = process_shift(shift_m)
            (L_r, offset_r, active_r) = process_shift(shift_r)

            # mask updates depending on inter-cell displacement
            Array3b = dr.mask_t(xrtu.float_array_t(Float, 3))
            displacement = ArrayNi(index) - index_prev
            active_lmr = dr.if_stmt(
                (active_l, active_m, active_r),
                dr.dot(displacement, mv_dir) != 0,  # going in mv_dir
                lambda l, m, r: Array3b(l, m, r),
                lambda l, m, r: dr.select(
                    dr.dot(displacement, dr.reverse(mv_dir)) == -1,  # going left
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

        n = dr.normalize(ray_n) # cos theta, sin theta, 0 (for parallel beam)
        index_prev = ArrayNi(-1)  # previous visited cell
        state = (buffer, index_prev)
        direction = dr.abs(ray_n * dr.rcp(knot_step))

        dominant_axis = dr.floor(direction / dr.max(direction) + 1e-6) # floating precision issues TODO case 110, 011, 101 is diagonal (choose one randomly)
        shift_x = ArrayNf(dominant_axis.y, dominant_axis.z, dominant_axis.x)
        shift_y = dr.cross(dominant_axis, shift_x)

        # gram-schmidt to get n_perp_x and n_perp_y
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
            # compute analytic ray<>box-spline back-projection.
            (accum, index_prev) = state

            def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNuT, BoolT]:
                index_s = index + shift  # "_s" = shifted
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))

                cell_center = ArrayNf(0.5, 0.5, 0.5) + shift
                x = dr.dot(n_perp_y, knot_step * (cell_center - p_a)) # horizontal position of detector
                y = dr.dot(n_perp_x, knot_step * (cell_center - p_a)) # vertical position of detector

                L = spline_3d_dr(net, x, y, n)
                if tof is not None:
                    L = L * tof_center_w(ArrayNf(index) + cell_center)

                return (L, offset, active)

            (L_mlr, offset_mlr, active_mlr) = process_shift(shift_m)
            (L_mud, offset_mud, active_mud) = process_shift(shift_m)
            (L_l, offset_l, active_l) = process_shift(shift_x)
            (L_r, offset_r, active_r) = process_shift(-shift_x)
            (L_d, offset_d, active_d) = process_shift(shift_y)
            (L_u, offset_u, active_u) = process_shift(-shift_y)
            (L_ll, offset_ll, active_ll) = process_shift(shift_x + shift_y)
            (L_rr, offset_rr, active_rr) = process_shift(-shift_x - shift_y)
            (L_lr, offset_lr, active_lr) = process_shift(shift_x - shift_y)
            (L_rl, offset_rl, active_rl) = process_shift(-shift_x + shift_y)
            # mask updates depending on inter-cell displacement
            Array3b = dr.mask_t(xrtu.float_array_t(Float, 3))
            displacement = ArrayNi(index) - index_prev
            active_lmr = dr.if_stmt(
                (active_l, active_mlr, active_r),
                dr.dot(displacement, shift_x) != 0,  # going in mv_dir
                lambda l, m, r: dr.select(
                    dr.dot(displacement, shift_x) == -1,  # going left
                    Array3b(False, False, r),
                    Array3b(l, False, False),
                ),
                lambda l, m, r: Array3b(l, m, r),
            )
            active_mlr = dr.select(
                dr.dot(displacement, shift_x) == 0,
                False,
                active_mlr,
            )
            active_l = dr.select(
                dr.dot(displacement, shift_y) != 0,
                False,
                active_l,
            )
            active_r = dr.select(
                dr.dot(displacement, shift_y) != 0,
                False,
                active_r,
            )
            dr.scatter_add(accum, L_l * data, offset_l, active_lmr.x & active_l)
            dr.scatter_add(accum, L_mlr * data, offset_mlr, active_lmr.y & active_mlr)
            dr.scatter_add(accum, L_r * data, offset_r, active_lmr.z & active_r)
            active_dmu = dr.if_stmt(
                (active_d, active_mud, active_u),
                dr.dot(displacement, shift_y) != 0,  # going in mv_dir
                lambda d, m, u: dr.select(
                    dr.dot(displacement, shift_y) == -1,  # going down
                    Array3b(False, False, u),
                    Array3b(d, False, False),
                ),
                lambda d, m, u: Array3b(d, m, u),
            )
            active_mud = active_mud & ~active_mlr
            active_d = dr.select(
                dr.dot(displacement, shift_x) != 0,
                False,
                active_d,
            )
            active_u = dr.select(
                dr.dot(displacement, shift_x) != 0,
                False,
                active_u,
            )
            dr.scatter_add(accum, L_d * data, offset_d, active_dmu.x & active_d)
            dr.scatter_add(accum, L_mud * data, offset_mud, active_dmu.y & active_mud)
            dr.scatter_add(accum, L_u * data, offset_u, active_dmu.z & active_u)
            active_ll = dr.select(
                dr.dot(displacement, shift_y) != 0,
                False,
                active_ll,
            )
            active_ll = dr.select(
                dr.dot(displacement, shift_x) != 0,
                False,
                active_ll,
            )
            active_rr = dr.select(
                dr.dot(displacement, shift_y) != 0,
                False,
                active_rr,
            )
            active_rr = dr.select(
                dr.dot(displacement, shift_x) != 0,
                False,
                active_rr,
            )
            active_lr = dr.select(
                dr.dot(displacement, shift_y) != 0,
                False,
                active_lr,
            )
            active_lr = dr.select(
                dr.dot(displacement, shift_x) != 0,
                False,
                active_lr,
            )
            active_rl = dr.select(
                dr.dot(displacement, shift_y) != 0,
                False,
                active_rl,
            )
            active_rl = dr.select(
                dr.dot(displacement, shift_x) != 0,
                False,
                active_rl,
            )
            dr.scatter_add(accum, L_ll * data, offset_ll, active_ll)
            dr.scatter_add(accum, L_rr * data, offset_rr, active_rr)
            dr.scatter_add(accum, L_lr * data, offset_lr, active_lr)
            dr.scatter_add(accum, L_rl * data, offset_rl, active_rl)
            return (accum, ArrayNi(index)), Bool(True)

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
        mode=mode,
        max_iterations=-1,
    )

    return buffer


def xrt_ad_t_x_(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
mode = "symbolic"
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
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

            L = 1/ray_n[0] 
            L_hor = dr.select(
                dr.abs(p_b[1] - p_a[1]) > 1 - eps,
                0,
                L,
            )
            L_hor_vert = dr.select(
                dr.abs(p_b[0] - p_a[0]) > 1 - eps,
                0,
                L_hor,
            )

            L_hor_vert_signed = dr.select(
                dr.abs(p_b[0] - 0.5) > 0.5 - eps,
                -L_hor_vert,
                L_hor_vert,
            )
            accum += fq * (L_hor_vert_signed)

            return (accum,), Bool(True)


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
        mode=mode,
        max_iterations=-1,
    )

    return buffer

def xrt_ad_t_x(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
mode = "symbolic"
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
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


    index_prev = ArrayNi(-1)  # previous visited cell
    state = (buffer, index_prev)

    # (main, lateral) movement direction
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
        # compute analytic ray<>box-spline projection.
        (accum, index_prev) = state

        def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
            index_s = index + shift  # "_s" = shifted
            offset = dr.dot(index_s, stride)
            active = dr.all((0 <= index_s) & (index_s < knot_num))
            fq = dr.gather(Float, data, offset, active)

            cell_center = ArrayNf(0.5, 0.5) + shift

            dr.enable_grad(p_a)
            def contribution(p_a, ray_n):
                n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
                E, E_mask = box_spline_1d_E(order, knot_step, n_perp)
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)
                return L
            
            L = contribution(p_a, ray_n)
            dr.backward(L)
            grad_contribution = p_a.grad[0]
            dr.disable_grad(p_a)

            return (fq, grad_contribution)

        (fq_l, L_l) = process_shift(shift_l)
        (fq_m, L_m) = process_shift(shift_m)
        (fq_r, L_r) = process_shift(shift_r)

        # mask updates depending on inter-cell displacement
        Array3f = xrtu.float_array_t(Float, 3)
        displacement = ArrayNi(index) - index_prev
        fq_lmr = dr.if_stmt(
            (fq_l, fq_m, fq_r),
            dr.dot(displacement, mv_dir) != 0,  # going in mv_dir
            lambda l, m, r: Array3f(l, m, r),
            lambda l, m, r: dr.select(
                dr.dot(displacement, dr.reverse(mv_dir)) == -1,  # going left
                Array3f(l, 0, 0),
                Array3f(0, 0, r),
            ),
        )
        L_lmr = Array3f(L_l, L_m, L_r)

        accum += dr.dot(fq_lmr, L_lmr)

        return (accum, ArrayNi(index)), Bool(True)

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
        mode=mode,
        max_iterations=-1,
    )

    return buffer



def xrt_ad_t_y_(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
mode = "symbolic"
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
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

            L = 1/ray_n[1] 
            L_hor = dr.select(
                dr.abs(p_b[1] - p_a[1]) > 1 - eps,
                0,
                L,
            )
            L_hor_vert = dr.select(
                dr.abs(p_b[0] - p_a[0]) > 1 - eps,
                0,
                L_hor,
            )

            L_hor_vert_signed = dr.select(
                dr.abs(p_b[1] - 0.5) > 0.5 - eps,
                -L_hor_vert,
                L_hor_vert,
            )
            accum += fq * (L_hor_vert_signed)

            return (accum,), Bool(True)


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
        mode=mode,
        max_iterations=-1,
    )

    return buffer


def xrt_ad_t_y(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
mode = "symbolic"
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
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


    index_prev = ArrayNi(-1)  # previous visited cell
    state = (buffer, index_prev)

    # (main, lateral) movement direction
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
        # compute analytic ray<>box-spline projection.
        (accum, index_prev) = state

        def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
            index_s = index + shift  # "_s" = shifted
            offset = dr.dot(index_s, stride)
            active = dr.all((0 <= index_s) & (index_s < knot_num))
            fq = dr.gather(Float, data, offset, active)

            cell_center = ArrayNf(0.5, 0.5) + shift

            dr.enable_grad(p_a)
            def contribution(p_a, ray_n):
                n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
                E, E_mask = box_spline_1d_E(order, knot_step, n_perp)
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)
                return L
            
            L = contribution(p_a, ray_n)
            dr.backward(L)
            grad_contribution = p_a.grad[1]
            dr.disable_grad(p_a)

            return (fq, grad_contribution)

        (fq_l, L_l) = process_shift(shift_l)
        (fq_m, L_m) = process_shift(shift_m)
        (fq_r, L_r) = process_shift(shift_r)

        # mask updates depending on inter-cell displacement
        Array3f = xrtu.float_array_t(Float, 3)
        displacement = ArrayNi(index) - index_prev
        fq_lmr = dr.if_stmt(
            (fq_l, fq_m, fq_r),
            dr.dot(displacement, mv_dir) != 0,  # going in mv_dir
            lambda l, m, r: Array3f(l, m, r),
            lambda l, m, r: dr.select(
                dr.dot(displacement, dr.reverse(mv_dir)) == -1,  # going left
                Array3f(l, 0, 0),
                Array3f(0, 0, r),
            ),
        )
        L_lmr = Array3f(L_l, L_m, L_r)

        accum += dr.dot(fq_lmr, L_lmr)

        return (accum, ArrayNi(index)), Bool(True)

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
        mode=mode,
        max_iterations=-1,
    )

    return buffer










def xrt_ad_n_x_(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
mode = "symbolic"
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
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
    ray_t_init = dr.copy(ray_t)
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
            p_a_global = p_a + index -0.5

            fac_t = (p_a_global[0] - ray_t_init[0])/ray_n[0]  # p_a = o + t*n
            fac_n = dr.select(
                dr.abs(p_b[1] - 0.5) > 0.5 - eps,
                1-p_a[1],
                1-p_a[0],
            )
            fac_n = dr.select(
                dr.abs(p_b[0]) < eps,
                fac_n-1,
                fac_n,
            )

            residual = ray_n[0] / ray_n[1]
            residual = dr.select(
                dr.abs(p_b[1] - 0.5) > 0.5 - eps,
                residual,
                -1/residual**2,
            )

            L = 1/ray_n[0]
            L_hor = dr.select(
                dr.abs(p_b[1] - p_a[1]) > 1 - eps,
                0,
                L,
            )
            L_hor_vert = dr.select(
                dr.abs(p_b[0] - p_a[0]) > 1 - eps,
                0,
                L_hor,
            )

            L_hor_vert_signed = dr.select(
                dr.abs(p_b[0] - 0.5) > 0.5 - eps,
                -L_hor_vert,
                L_hor_vert,
            )
            accum += fq * (fac_t * L_hor_vert_signed + fac_n * residual)
            return (accum,), Bool(True)


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
        mode=mode,
        max_iterations=-1,
    )

    return buffer

def xrt_ad_n_x(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
mode = "symbolic"
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
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
    ray_t_init = dr.copy(ray_t)
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


    index_prev = ArrayNi(-1)  # previous visited cell
    state = (buffer, index_prev)

    # (main, lateral) movement direction
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
        # compute analytic ray<>box-spline projection.
        (accum, index_prev) = state

        def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
            index_s = index + shift  # "_s" = shifted
            offset = dr.dot(index_s, stride)
            active = dr.all((0 <= index_s) & (index_s < knot_num))
            fq = dr.gather(Float, data, offset, active)

            cell_center = ArrayNf(0.5, 0.5) + shift

            p_a_global = p_a + index -0.5
            fac_t = (p_a_global[0] - ray_t_init[0])/ray_n[0]

            dr.enable_grad(ray_n, p_a)
            def contribution(p_a, ray_n):
                n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
                E, E_mask = box_spline_1d_E(order, knot_step, n_perp)
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)
                return L
            
            L = contribution(p_a, ray_n)
            dr.backward(L)
            grad_contribution = ray_n.grad[0] + fac_t * p_a.grad[0]
            dr.disable_grad(ray_n, p_a) 

            return (fq, grad_contribution)

        (fq_l, L_l) = process_shift(shift_l)
        (fq_m, L_m) = process_shift(shift_m)
        (fq_r, L_r) = process_shift(shift_r)

        # mask updates depending on inter-cell displacement
        Array3f = xrtu.float_array_t(Float, 3)
        displacement = ArrayNi(index) - index_prev
        fq_lmr = dr.if_stmt(
            (fq_l, fq_m, fq_r),
            dr.dot(displacement, mv_dir) != 0,  # going in mv_dir
            lambda l, m, r: Array3f(l, m, r),
            lambda l, m, r: dr.select(
                dr.dot(displacement, dr.reverse(mv_dir)) == -1,  # going left
                Array3f(l, 0, 0),
                Array3f(0, 0, r),
            ),
        )
        L_lmr = Array3f(L_l, L_m, L_r)

        accum += dr.dot(fq_lmr, L_lmr)

        return (accum, ArrayNi(index)), Bool(True)

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
        mode=mode,
        max_iterations=-1,
    )

    return buffer

def xrt_ad_n_y_(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
mode = "symbolic"
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
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
    ray_t_init = dr.copy(ray_t)
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

            p_a_global = p_a + index -0.5
            fac_t = (p_a_global[0] - ray_t_init[0])/ray_n[0]
            fac_n = dr.select(
                dr.abs(p_b[1] - 0.5) > 0.5 - eps,
                1-p_a[1],
                1-p_a[0],
            )
            fac_n = dr.select(
                dr.abs(p_b[0]) < eps,
                fac_n-1,
                fac_n,
            )

            residual = -(ray_n[0] / ray_n[1])**2
            residual = dr.select(
                dr.abs(p_b[1] - 0.5) > 0.5 - eps,
                residual,
                ray_n[1] / ray_n[0],
            )
            L = 1/ray_n[1] 
            L_hor = dr.select(
                dr.abs(p_b[1] - p_a[1]) > 1 - eps,
                0,
                L,
            )
            L_hor_vert = dr.select(
                dr.abs(p_b[0] - p_a[0]) > 1 - eps,
                0,
                L_hor,
            )

            L_hor_vert_signed = dr.select(
                dr.abs(p_b[1] - 0.5) > 0.5 - eps,
                -L_hor_vert,
                L_hor_vert,
            )
            accum += fq * (fac_t * L_hor_vert_signed + fac_n * residual)

            return (accum,), Bool(True)


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
        mode=mode,
        max_iterations=-1,
    )

    return buffer

def xrt_ad_n_y(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
mode = "symbolic"
) -> FloatT:
    r"""
    Compute 2D/3D projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
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
    ray_t_init = dr.copy(ray_t)
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


    index_prev = ArrayNi(-1)  # previous visited cell
    state = (buffer, index_prev)

    # (main, lateral) movement direction
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
        # compute analytic ray<>box-spline projection.
        (accum, index_prev) = state

        def process_shift(shift: ArrayNiT) -> tuple[ArrayNfT, ArrayNfT]:
            index_s = index + shift  # "_s" = shifted
            offset = dr.dot(index_s, stride)
            active = dr.all((0 <= index_s) & (index_s < knot_num))
            fq = dr.gather(Float, data, offset, active)

            cell_center = ArrayNf(0.5, 0.5) + shift

            p_a_global = p_a + index -0.5
            fac_t = (p_a_global[0] - ray_t_init[0])/ray_n[0]

            dr.enable_grad(ray_n, p_a)
            def contribution(p_a, ray_n):
                n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
                E, E_mask = box_spline_1d_E(order, knot_step, n_perp)
                x = dr.dot(n_perp, knot_step * (cell_center - p_a))
                L = box_spline_1d_dr(E, E_mask, x)
                return L
            
            L = contribution(p_a, ray_n)
            dr.backward(L)
            grad_contribution = ray_n.grad[1] + fac_t * p_a.grad[1]
            dr.disable_grad(ray_n, p_a) 

            return (fq, grad_contribution)

        (fq_l, L_l) = process_shift(shift_l)
        (fq_m, L_m) = process_shift(shift_m)
        (fq_r, L_r) = process_shift(shift_r)

        # mask updates depending on inter-cell displacement
        Array3f = xrtu.float_array_t(Float, 3)
        displacement = ArrayNi(index) - index_prev
        fq_lmr = dr.if_stmt(
            (fq_l, fq_m, fq_r),
            dr.dot(displacement, mv_dir) != 0,  # going in mv_dir
            lambda l, m, r: Array3f(l, m, r),
            lambda l, m, r: dr.select(
                dr.dot(displacement, dr.reverse(mv_dir)) == -1,  # going left
                Array3f(l, 0, 0),
                Array3f(0, 0, r),
            ),
        )
        L_lmr = Array3f(L_l, L_m, L_r)

        accum += dr.dot(fq_lmr, L_lmr)

        return (accum, ArrayNi(index)), Bool(True)

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
        mode=mode,
        max_iterations=-1,
    )

    return buffer
