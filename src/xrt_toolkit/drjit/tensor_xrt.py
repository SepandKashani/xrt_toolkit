r"""
Fused multi-channel (tensor) X-ray transforms.

Tensor tomography reconstructs a tensor field :math:`\mathbf{f} = (f_1, ..., f_C)`
from contracted line integrals. In every standard linear modality (Doppler /
vector tomography, strain tomography, photoelasticity, SAXS / dark-field tensor
tomography), the contraction weights depend only on the ray -- its direction
:math:`\mathbf{n}` and possibly a per-ray probe direction -- so the measurement
factorizes through the scalar transform:

.. math::

   \mathcal{P}_{\text{tens}}[\mathbf{f}](\mathbf{t}, \mathbf{n})
   = \sum_{c=1}^{C} w_c \, \mathcal{P}[f_c](\mathbf{t}, \mathbf{n}).

One *could* evaluate this with C calls to ``xrt_apply``, but that traverses the
lattice C times and recomputes the projected-basis weight -- the dominant cost
for spline orders in 3D -- C times. The operators here instead run a *single*
DDA traversal per ray: the basis weight is computed once per visited cell and
contracted against the C channel values, fetched with one packet gather from a
channels-interleaved volume. The adjoint scatters the C weighted residuals with
one packet scatter. The scalar operators in ``ray_xrt`` are not modified.

Data layout: channels-last (AoS). The volume buffer is flat with
``buffer[q * C + c]`` holding channel ``c`` of cell ``q`` (C-ordered cells, as
in the scalar operators). Use :func:`pack_channels` / :func:`unpack_channels`
to convert from per-channel flat volumes.

Weights: a list of C per-ray arrays (or literals). Builders for the common
contractions are provided (:func:`doppler_weights`, :func:`lrt_weights`,
:func:`trt_weights`); arbitrary per-ray weights (e.g. spherical-harmonic
weights for SAXS tensor tomography) can be passed directly.

Reference implementations (``*_ref``) compose the scalar operators channel by
channel; they produce the same numbers (to float precision) and serve as the
validation baseline.
"""

import importlib
import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

from .bbox import bbox_contains, ray_bbox_intersect
from .box_spline import box_spline_1d_dr, box_spline_1d_E, spline_3d_dr
from .dda import dda
from .ray_xrt import _tof_setup, net, xrt_adjoint, xrt_apply

BoolT = typ.TypeVar("BoolT", bound=dr.AnyArray)
ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
ArrayNiT = typ.TypeVar("ArrayNiT", bound=dr.AnyArray)
ArrayNuT = typ.TypeVar("ArrayNuT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNfT, ArrayNfT]
WeightsT = typ.Sequence


def _dyn_array_t(Float):
    """Dynamic-size array type (ArrayXf) from the backend of `Float`."""
    return importlib.import_module(Float.__module__).ArrayXf


# ------------------------------------------------------------------ weights --
def doppler_weights(ray_n: ArrayNfT) -> list:
    r"""
    Longitudinal weights for a vector field: :math:`w = \hat{\mathbf{n}}`.

    Channels: ``(x, y[, z])``. Model: :math:`\int \langle \mathbf{v}, \hat{\mathbf{n}} \rangle`.
    """
    n = dr.normalize(ray_n)
    return [n[d] for d in range(dr.size_v(type(ray_n)))]


def lrt_weights(ray_n: ArrayNfT) -> list:
    r"""
    Longitudinal ray transform weights for a symmetric rank-2 field:
    :math:`y = \int \hat{\mathbf{n}}^\top \mathbf{f} \, \hat{\mathbf{n}}`.

    Channels: 2D ``(xx, yy, xy)``; 3D ``(xx, yy, zz, xy, xz, yz)``
    (off-diagonal channels carry the factor 2).
    """
    n = dr.normalize(ray_n)
    D = dr.size_v(type(ray_n))
    if D == 2:
        return [n.x * n.x, n.y * n.y, 2 * n.x * n.y]
    return [n.x * n.x, n.y * n.y, n.z * n.z,
            2 * n.x * n.y, 2 * n.x * n.z, 2 * n.y * n.z]


def trt_weights(ray_n: ArrayNfT, eta: ArrayNfT) -> list:
    r"""
    Transverse ray transform weights: the rank-2 contraction of
    :func:`lrt_weights` evaluated at a per-ray probe direction
    :math:`\boldsymbol{\eta} \perp \mathbf{n}` (e.g. a grating sensitivity
    direction in dark-field tensor tomography).
    """
    e = dr.normalize(eta - dr.normalize(ray_n) * dr.dot(dr.normalize(ray_n), eta))
    return lrt_weights(e)


# ------------------------------------------------------------------- layout --
def pack_channels(channels: typ.Sequence) -> FloatT:
    """(C,) list of flat (Q,) volumes -> channels-interleaved flat (Q*C,) buffer."""
    C = len(channels)
    Float = type(channels[0])
    UInt32 = dr.uint32_array_t(Float)
    Q = dr.width(channels[0])
    out = dr.zeros(Float, Q * C)
    q = dr.arange(UInt32, Q)
    for c, ch in enumerate(channels):
        assert type(ch) is Float and dr.width(ch) == Q
        dr.scatter(out, ch, q * C + c)
    dr.eval(out)
    return out


def unpack_channels(buffer: FloatT, C: int) -> list:
    """Channels-interleaved flat (Q*C,) buffer -> (C,) list of flat (Q,) volumes."""
    Float = type(buffer)
    UInt32 = dr.uint32_array_t(Float)
    Q = dr.width(buffer) // C
    q = dr.arange(UInt32, Q)
    return [dr.gather(Float, buffer, q * C + c) for c in range(C)]


# ----------------------------------------------------------------- operators --
def xrt_tensor_apply(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    weights: WeightsT,
    data: FloatT,
    buffer: FloatT = None,
    mode="symbolic",
    tof=None,
) -> FloatT:
    r"""
    Contracted multi-channel projection
    :math:`y_l = \sum_c w_{c,l} \int f_c(\mathbf{t}_l + \alpha \mathbf{n}_l) \, \mathrm{d}\alpha`
    in a single fused traversal.

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors and directions, as in ``xrt_apply``.
    knot_spec: UniformSpec
        Volume properties :math:`(\mathbf{x}_0, \boldsymbol{\Delta}, \mathbf{Q})`.
    order: 0 | 1 | 2
        Basis order, as in ``xrt_apply``.
    weights: sequence of C per-ray weight arrays (L,) or literals
        Contraction weights :math:`w_c`; constant along each ray.
    data: FloatT
        (Q1*...*QD*C,) channels-interleaved coefficients (see :func:`pack_channels`).
    buffer: FloatT
        Optional (L,) output buffer.
    tof: TOFSpec | None
        Optional time-of-flight kernel, as in ``xrt_apply``.

    Returns
    -------
    proj: FloatT
        (L,) contracted projections.
    """
    ray_t, ray_n = ray_spec

    ArrayNf = type(ray_t)
    ArrayNi = dr.int32_array_t(ArrayNf)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)
    ArrayXf = _dyn_array_t(Float)

    # type checking ---------------------------------------
    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D in (2, 3))
    assert type(ray_n) is ArrayNf
    assert knot_spec.ndim == D
    assert order in (0, 1, 2)

    C = len(weights)
    assert C >= 1
    w = [wc if type(wc) is Float else Float(wc) for wc in weights]

    assert type(data) is Float
    assert len(data) == math.prod(knot_spec.num) * C

    L_rays = max(ray_t.shape[1], ray_n.shape[1])
    if buffer is None:
        buffer = dr.zeros(Float, L_rays)
    else:
        assert type(buffer) is Float
        assert len(buffer) == L_rays
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

    def cgather(offset, active):
        # one packet gather of the C channel values of a cell, contracted with
        # the per-ray weights: the "virtual scalar volume" seen by this ray.
        # Offsets are sanitized because packet memory ops on masked-off lanes
        # still compute addresses (Dr.Jit 1.2): every lane must stay in range.
        fq = dr.gather(ArrayXf, data, dr.select(active, offset, 0), active,
                       shape=(C, L_rays))
        acc = w[0] * fq[0]
        for c in range(1, C):
            acc = dr.fma(w[c], fq[c], acc)
        return acc

    if order == 0:
        state = (buffer,)

        def project(state, index, p_a, p_b, active):
            (accum,) = state

            offset = dr.dot(index, stride)
            fq = cgather(offset, active)
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

        def project(state, index, p_a, p_b, active):
            # identical to the scalar callback, with the contracted gather.
            (accum, index_prev) = state

            def process_shift(shift):
                index_s = index + shift  # "_s" = shifted
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))
                fq = cgather(offset, active)

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
        n = dr.normalize(ray_n)
        index_prev = ArrayNi(-1)  # previous visited cell
        state = (buffer, index_prev)
        direction = dr.abs(ray_n * dr.rcp(knot_step))

        dominant_axis = dr.floor(direction / dr.max(direction) + 1e-6)
        shift_x = ArrayNf(dominant_axis.y, dominant_axis.z, dominant_axis.x)
        shift_y = dr.cross(dominant_axis, shift_x)

        # gram-schmidt to get n_perp_x and n_perp_y
        n_perp_x = dr.normalize(shift_x - dr.dot(shift_x, n) * n)
        n_perp_y = dr.cross(n, n_perp_x)

        shift_m = ArrayNi(0, 0, 0)

        def project(state, index, p_a, p_b, active):
            # identical to the scalar callback, with the contracted gather.
            (accum, index_prev) = state

            def process_shift(shift):
                index_s = index + shift  # "_s" = shifted
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))
                fq = cgather(offset, active)

                cell_center = ArrayNf(0.5, 0.5, 0.5) + shift
                x = dr.dot(n_perp_y, knot_step * (cell_center - p_a))
                y = dr.dot(n_perp_x, knot_step * (cell_center - p_a))

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

            corner_off = dr.dot(displacement, shift_y) != 0
            corner_off = corner_off | (dr.dot(displacement, shift_x) != 0)
            fq_ll = dr.select(corner_off, 0, fq_ll)
            fq_rr = dr.select(corner_off, 0, fq_rr)
            fq_lr = dr.select(corner_off, 0, fq_lr)
            fq_rl = dr.select(corner_off, 0, fq_rl)

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

    return state[0]


def xrt_tensor_adjoint(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    weights: WeightsT,
    data: FloatT,
    buffer: FloatT = None,
    mode="symbolic",
    tof=None,
) -> FloatT:
    r"""
    Exact adjoint of :func:`xrt_tensor_apply`: maps (L,) residuals to
    channels-interleaved (Q*C,) coefficients,
    :math:`(\mathbf{A}^* \mathbf{r})_{q,c} = \sum_l w_{c,l} \, r_l \, \Psi_{l,q}`,
    in a single fused traversal (one packet scatter per visited cell).

    Parameters mirror :func:`xrt_tensor_apply`; `data` is the (L,) projection
    residual and the return value is the (Q1*...*QD*C,) interleaved volume.
    """
    ray_t, ray_n = ray_spec

    ArrayNf = type(ray_t)
    ArrayNi = dr.int32_array_t(ArrayNf)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)
    ArrayXf = _dyn_array_t(Float)

    # type checking ---------------------------------------
    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D in (2, 3))
    assert type(ray_n) is ArrayNf
    assert knot_spec.ndim == D
    assert order in (0, 1, 2)

    C = len(weights)
    assert C >= 1
    w = [wc if type(wc) is Float else Float(wc) for wc in weights]

    L_rays = max(ray_t.shape[1], ray_n.shape[1])
    assert type(data) is Float
    assert len(data) == L_rays

    if buffer is None:
        buffer = dr.zeros(Float, math.prod(knot_spec.num) * C)
    else:
        assert type(buffer) is Float
        assert len(buffer) == math.prod(knot_spec.num) * C
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

    # per-ray weighted residuals: one scatter per channel writes C consecutive
    # addresses of the interleaved volume.
    wdata = [w[c] * data for c in range(C)]

    def cscatter(accum, Lw, offset, active):
        # Per-channel *scalar* scatters: Dr.Jit 1.2 miscompiles packet
        # scatters inside evaluated-mode loops for some widths, so we keep
        # the code path of the validated scalar operators. The C consecutive
        # addresses coalesce in hardware; the traversal is still fused.
        base = offset * C
        for c in range(C):
            dr.scatter_add(accum, Lw * wdata[c], base + c, active)

    if order == 0:
        state = (buffer,)

        def back_project(state, index, p_a, p_b, active):
            (accum,) = state

            offset = dr.dot(index, stride)
            if tof is None:
                L = dr.norm((p_b - p_a) * knot_step) * dr.rcp(dr.prod(knot_step))
            else:
                L = tof_chord_w(ArrayNf(index), p_a, p_b) * dr.rcp(dr.prod(knot_step))
            cscatter(accum, L, offset, active)

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

        def back_project(state, index, p_a, p_b, active):
            # identical to the scalar callback, with the packet scatter.
            (accum, index_prev) = state

            def process_shift(shift):
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

            cscatter(accum, L_l, offset_l, active_lmr.x)
            cscatter(accum, L_m, offset_m, active_lmr.y)
            cscatter(accum, L_r, offset_r, active_lmr.z)

            return (accum, ArrayNi(index)), Bool(True)

    elif (D == 3) and (order > 0):
        n = dr.normalize(ray_n)
        index_prev = ArrayNi(-1)  # previous visited cell
        state = (buffer, index_prev)
        direction = dr.abs(ray_n * dr.rcp(knot_step))

        dominant_axis = dr.floor(direction / dr.max(direction) + 1e-6)
        shift_x = ArrayNf(dominant_axis.y, dominant_axis.z, dominant_axis.x)
        shift_y = dr.cross(dominant_axis, shift_x)

        # gram-schmidt to get n_perp_x and n_perp_y
        n_perp_x = dr.normalize(shift_x - dr.dot(shift_x, n) * n)
        n_perp_y = dr.cross(n, n_perp_x)

        shift_m = ArrayNi(0, 0, 0)

        def back_project(state, index, p_a, p_b, active):
            # identical to the scalar callback, with the packet scatter.
            (accum, index_prev) = state

            def process_shift(shift):
                index_s = index + shift  # "_s" = shifted
                offset = dr.dot(index_s, stride)
                active = dr.all((0 <= index_s) & (index_s < knot_num))

                cell_center = ArrayNf(0.5, 0.5, 0.5) + shift
                x = dr.dot(n_perp_y, knot_step * (cell_center - p_a))
                y = dr.dot(n_perp_x, knot_step * (cell_center - p_a))

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
            cscatter(accum, L_l, offset_l, active_lmr.x & active_l)
            cscatter(accum, L_mlr, offset_mlr, active_lmr.y & active_mlr)
            cscatter(accum, L_r, offset_r, active_lmr.z & active_r)
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
            cscatter(accum, L_d, offset_d, active_dmu.x & active_d)
            cscatter(accum, L_mud, offset_mud, active_dmu.y & active_mud)
            cscatter(accum, L_u, offset_u, active_dmu.z & active_u)
            corner_off = dr.dot(displacement, shift_y) != 0
            corner_off = corner_off | (dr.dot(displacement, shift_x) != 0)
            cscatter(accum, L_ll, offset_ll, active_ll & ~corner_off)
            cscatter(accum, L_rr, offset_rr, active_rr & ~corner_off)
            cscatter(accum, L_lr, offset_lr, active_lr & ~corner_off)
            cscatter(accum, L_rl, offset_rl, active_rl & ~corner_off)
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

    return state[0]


# ---------------------------------------------------------------- reference --
def xrt_tensor_apply_ref(ray_spec, knot_spec, order, weights, data,
                         mode="symbolic", tof=None):
    """Channel-by-channel composition of ``xrt_apply`` (validation baseline)."""
    C = len(weights)
    chans = unpack_channels(data, C)
    Float = type(chans[0])
    out = None
    for c in range(C):
        y = xrt_apply(ray_spec, knot_spec, order, chans[c], mode=mode, tof=tof)
        wc = weights[c] if type(weights[c]) is Float else Float(weights[c])
        out = wc * y if out is None else dr.fma(wc, y, out)
    return out


def xrt_tensor_adjoint_ref(ray_spec, knot_spec, order, weights, data,
                           mode="symbolic", tof=None):
    """Channel-by-channel composition of ``xrt_adjoint`` (validation baseline)."""
    Float = type(data)
    bp = []
    for c in range(len(weights)):
        wc = weights[c] if type(weights[c]) is Float else Float(weights[c])
        bp.append(xrt_adjoint(ray_spec, knot_spec, order, wc * data,
                              mode=mode, tof=tof))
    return pack_channels(bp)
