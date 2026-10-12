r"""
2D order-1/2 projections by column walk.

Same basis, closed form and (ray, basis function) pairs as a cell-by-cell DDA, computed differently:

1. Column walk. Each ray is walked along its main axis (:math:`|n_{x} / \Delta_{x}| \ge |n_{y} / \Delta_{y}|`),
   one lattice column per iteration. In each column, the basis functions whose support meets the ray are
   consecutive lateral indices: the first is the floor of an affine function of the column, and there are at
   most 4. Each basis function is visited once per ray, with no per-step branch and no previous-cell state.

2. Closed form folded once per ray. The projected box-spline is symmetric, so only the truncated powers with
   positive knots remain: :math:`B(u) = \sum_{t} c_{t} (k_{t} - u)_{+}^{m}` with :math:`u = |x|`. For
   order 2 the 16 knots reduce to 4; for order 1 the 8 knots reduce to 3. Knots, coefficients and degree are
   per-ray constants.

3. Drift-free lateral position. Per column, the lateral coordinate of the ray is split into integer + fraction
   with an exact two-product, so float32 distances stay O(1) on any grid size.

4. Degree split. Rays with a masked direction (axis or diagonal within ~1e-3 rad) have a lower degree. They run
   in a second loop, so the degree is fixed at trace time.

5. Weight along the ray (e.g. TOF). Each basis function's projection is multiplied by the weight
   at the basis function's center: an approximation, accurate when the weight varies slowly
   compared to the basis support.

6. Coalesced adjoint. The adjoint accumulates into [layout, transposed layout], so the lateral axis is contiguous
   for every ray, and the lanes of a warp start at the warp's first column, so their atomics hit adjacent
   addresses.
"""

import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

from .bbox import ray_bbox_intersect
from .box_spline import box_spline_1d_E
from .weight import WeightT, ray_weight

ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNfT, ArrayNfT]

K_MAX = 4  # max basis functions per column


def _closed_form(order: int, knot_step: ArrayNfT, n_perp: ArrayNfT) -> tuple:
    # Per-ray constants (knots, coeffs, k_sup, masked) such that
    #     box_spline_1d_dr(E, E_mask, x)
    #     = sum_t coeffs[t] * max(knots[t] - |x|, 0) ** m,
    # with m = order + 1 if no direction is masked, else m = order.
    # k_sup is the support half-width.
    E, E_mask = box_spline_1d_E(order, knot_step, n_perp)
    Float = type(E[0])
    N = order + 2
    m_f = [Float(E_mask[k]) for k in range(N)]
    masked = dr.zeros(dr.mask_t(Float), dr.width(E[0]))
    for k in range(N):
        masked |= E_mask[k] == 0

    # normalization of box_spline_1d_dr(): 1 / ((N_nz - 1)! prod(E + 1 - E_mask))
    prod = Float(1)
    for k in range(N):
        prod *= E[k] + 1 - m_f[k]
    fact = dr.select(masked, float(math.factorial(N - 2)), float(math.factorial(N - 1)))
    norm = dr.rcp(fact * prod)

    # E without its masked entry (at most one entry is masked)
    m_idx = dr.full(dr.int32_array_t(Float), N - 1, dr.width(E[0]))
    for k in reversed(range(N)):
        m_idx = dr.select(E_mask[k] == 0, k, m_idx)
    E_m = [dr.select(m_idx <= p, E[p + 1], E[p]) for p in range(N - 1)]

    def half_set(E_d: list) -> list:
        # positive knots 1/2 sigma.E_d of the symmetric box-spline with 2 or 3
        # directions E_d, one per pair (sigma, -sigma), with signed coefficients.
        S = 0.5 * sum(E_d)
        out = [(S, Float(1))]
        if len(E_d) == 2:
            out.append((dr.abs(0.5 * (E_d[0] - E_d[1])), Float(-1)))
        elif len(E_d) == 3:
            for k in range(3):
                kap = S - E_d[k]
                out.append((dr.abs(kap), -dr.sign(kap)))
        return out

    if order == 1:
        # 3 directions with E_max = sum of the other two: knots S, S - E_lo, S - E_mid.
        S = 0.5 * (E[0] + E[1] + E[2])
        e_lo = dr.minimum(dr.minimum(E[0], E[1]), E[2])
        e_hi = dr.maximum(dr.maximum(E[0], E[1]), E[2])
        e_mid = (E[0] + E[1] + E[2]) - e_lo - e_hi
        full = [(S, Float(1)), (S - e_lo, Float(-1)), (S - e_mid, Float(-1))]
        part = half_set(E_m) + [(Float(0), Float(0))]
    elif order == 2:
        # Zwart-Powell element, {E3, E4} = {a + b, |a - b|}: of the 16 knots only
        # 1/2 (+-3A +- B) and 1/2 (+-A +- 3B) carry weight.
        A = dr.maximum(E[0], E[1])
        B = dr.minimum(E[0], E[1])
        S = 0.5 * (E[0] + E[1] + E[2] + E[3])
        k3 = S - dr.minimum(E[2], E[3])
        full = [
            (S, Float(1)),
            (S - B, Float(-1)),
            (k3, Float(-1)),
            (dr.abs(k3 - A), Float(1)),
        ]
        part = half_set(E_m)
    knots = [dr.select(masked, k_p, k_f) for (k_f, _), (k_p, _) in zip(full, part)]
    coeffs = [
        dr.select(masked, c_p, c_f) * norm for (_, c_f), (_, c_p) in zip(full, part)
    ]
    return knots, coeffs, knots[0], masked


def _eval(knots: list, coeffs: list, x: FloatT, degree: int) -> FloatT:
    # sum_t coeffs[t] * max(knots[t] - |x|, 0) ** degree
    u = dr.abs(x)
    y = None
    for k_t, c_t in zip(knots, coeffs):
        z = dr.maximum(k_t - u, 0)
        p = z if degree == 1 else (z * z if degree == 2 else z * z * z)
        y = c_t * p if y is None else dr.fma(c_t, p, y)
    return y


def _setup(
    ray_t: ArrayNfT,
    ray_n: ArrayNfT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    transpose: bool,
    weight: WeightT = None,
    closed: bool = False,
) -> dict:
    # Per-ray walk constants. The walk covers the support of every basis function: the
    # bbox padded by one cell on each side.
    # closed: also visit the basis functions whose support the ray only touches, where the
    # order-1 limit profile (degree 1) has a kink.
    ArrayNf = type(ray_t)
    ArrayNi = dr.int32_array_t(ArrayNf)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Int = dr.value_t(ArrayNi)
    UInt = dr.value_t(ArrayNu)

    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = knot_start - (knot_step / 2) + (knot_num * knot_step)

    # move the anchor just before the padded bbox entry
    active, t1, t2 = ray_bbox_intersect(
        bbox_ll - knot_step, bbox_ur + knot_step, ray_t, ray_n
    )
    t_min = dr.minimum(t1, t2)
    ray_o = dr.select(active, ray_t + (t_min - 1) * ray_n, ray_t)

    # grid coordinates (cell q spans [q, q + 1]), entry/exit of the padded grid [-1, Q + 1]
    grid_res = ArrayNf(knot_num)
    grid_scale = grid_res / (bbox_ur - bbox_ll)
    o = dr.fma(ray_o, grid_scale, -bbox_ll * grid_scale)
    d = dr.select(active, ray_n * grid_scale, ray_n)
    rcp_d = dr.rcp(d)
    inf_t = d == 0
    t_a = (-1 - o) * rcp_d
    t_b = (grid_res + 1 - o) * rcp_d
    t_lo = dr.minimum(t_a, t_b)
    t_hi = dr.maximum(t_a, t_b)
    t_hi[inf_t] = dr.inf
    t_lo[inf_t] = -dr.inf
    t_in = dr.maximum(dr.max(t_lo), 0)
    t_out = dr.min(t_hi)
    active &= (t_out > t_in) & dr.isfinite(t_out)
    active &= dr.all(~inf_t | ((-1 <= o) & (o <= grid_res + 1)))
    o_in = dr.fma(d, t_in, o)
    o_out = dr.fma(d, t_out, o)

    # main (a) / lateral (b) axis, as xrt_apply
    direction = ray_n * dr.rcp(knot_step)
    x_main = dr.abs(direction.x) >= dr.abs(direction.y)
    ab = lambda v: (dr.select(x_main, v.x, v.y), dr.select(x_main, v.y, v.x))
    da, db = ab(d)
    oa, ob = ab(o_in)
    oa_out, _ = ab(o_out)
    Qa, Qb = ab(ArrayNi(knot_num))
    sa, sb = ab(ArrayNi(knot_num.y, 1))

    # columns whose basis functions can meet the ray: one beyond the crossed cells on
    # each side (supports span 3 columns)
    pos = da >= 0
    step_a = dr.select(pos, Int(1), Int(-1))
    i_first = Int(dr.select(pos, dr.floor(oa) - 1, dr.ceil(oa)))
    i_last = Int(dr.select(pos, dr.ceil(oa_out), dr.floor(oa_out) - 1))
    i_first = dr.clip(i_first, 0, Qa - 1)
    i_last = dr.clip(i_last, 0, Qa - 1)

    # loop schedule: column i = (start + c) * step_a, c = 0..n_iter-1
    key = i_first * step_a
    key_end = dr.maximum(i_last * step_a, key)
    if transpose:
        # lanes of a warp start at the warp's first column (at most 64 idle iterations),
        # so that they visit the same column at the same time.
        L = dr.width(key)
        w_idx = dr.arange(UInt, L) // 32
        w_min = dr.full(Int, 2**30, (L + 31) // 32)
        dr.scatter_reduce(dr.ReduceOp.Min, w_min, dr.select(active, key, 2**30), w_idx)
        start = dr.clip(dr.gather(Int, w_min, w_idx), key - 64, key)
    else:
        start = key
    n_iter = dr.select(active, key_end - start + 1, 0)

    # slope as a double-float; entry point split in integer + fraction
    s_hi = db / da
    s_lo = dr.fma(-s_hi, da, db) / da
    ia0 = dr.floor(oa)
    fa0 = oa - ia0
    jb0 = dr.floor(ob)
    fb0 = ob - jb0
    # lateral coordinate of the ray at the center of column ia0 + k, minus jb0:
    #     yc(k) = C + k * slope,  C = fb0 + (0.5 - fa0) * slope
    C = dr.fma(0.5 - fa0, s_hi, fb0)

    # closed form and support
    n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
    knots, coeffs, k_sup, masked = _closed_form(order, knot_step, n_perp)
    _, beta = ab(n_perp * knot_step)  # x(i, j) = beta * (j + 1/2 - yc(i))
    # first lateral index inside the support: |beta (j + 1/2 - yc)| < k_sup
    r_beta = dr.rcp(beta)
    w_lo = -k_sup * dr.abs(r_beta)
    # basis functions per column: ceil(2 k_sup / |beta|); 2 for an unmasked order-1 ray
    # on the (1, 1) diagonal side (support exactly 2 lateral steps wide).
    K = Int(dr.ceil(2 * k_sup * dr.abs(r_beta)))
    if order == 1:
        K = dr.select(~masked & (s_hi >= 0), 2, K)
    closed = (masked & (order == 1)) if closed else None
    if closed is not None:
        K = dr.select(closed, K + 1, K)
    K = dr.clip(K, 1, K_MAX)

    if transpose:  # lateral axis contiguous for every ray
        sa, sb = Qb, Int(1)
        base = dr.select(x_main, 0, Int(math.prod(knot_spec.num)))
    else:
        base = Int(0)

    w_density = None
    if weight is not None:
        _, w_density = ray_weight(weight, ray_t, ray_n)

    return dict(
        w_density=w_density,
        x_main=x_main,
        knot_start=knot_start,
        knot_step=knot_step,
        active=active,
        n_iter=n_iter,
        start=start,
        key=key,
        step_a=step_a,
        ia0=Int(ia0),
        jb0=Int(jb0),
        s_hi=s_hi,
        s_lo=s_lo,
        C=C,
        hs=0.5 * dr.abs(s_hi),
        W0=w_lo - 0.5,
        K=K,
        beta=beta,
        knots=knots,
        coeffs=coeffs,
        Qb=Qb,
        sa=sa,
        sb=sb,
        base=base,
        masked=masked,
        closed=closed,
    )


def _column(st: dict, c, n_iter) -> dict:
    # Per-column quantities for loop counter c.
    Float = type(st["C"])
    Int = type(st["K"])
    i_key = st["start"] + c
    # the lane has reached its first column and not passed its last
    on = (i_key >= st["key"]) & (c < n_iter)
    i = i_key * st["step_a"]
    k = Float(i - st["ia0"])
    p = k * st["s_hi"]
    e = dr.fma(k, st["s_hi"], -p) + k * st["s_lo"]
    jp = dr.floor(p)
    fr = (p - jp) + (e + st["C"])  # yc - (jb0 + jp), O(1)
    J0 = st["jb0"] + Int(jp)
    jlo = Int(dr.floor(fr - st["hs"])) - 1  # dda() lateral range, relative to J0
    jhi = Int(dr.floor(fr + st["hs"])) + 1
    jb = Int(dr.floor(fr + st["W0"])) + 1  # first index inside the support
    if st["closed"] is not None:  # first index on the closed support
        jb = dr.select(st["closed"], Int(dr.ceil(fr + st["W0"])), jb)
    x0 = st["beta"] * (Float(jb) + 0.5 - fr)
    ioff = st["base"] + i * st["sa"]
    return dict(on=on, i=i, J0=J0, jlo=jlo, jhi=jhi, jb=jb, x0=x0, ioff=ioff)


def _basis(st: dict, col: dict, kk: int) -> tuple:
    # (offset, valid, x, center) of the basis function in lateral slot kk of a column:
    # x is the closed form's argument, center the basis function's position.
    jr = col["jb"] + kk
    j = col["J0"] + jr
    valid = (
        col["on"] & (jr <= col["jhi"]) & (jr >= col["jlo"]) & (j >= 0) & (j < st["Qb"])
    )
    if kk >= 2:
        valid &= kk < st["K"]
    offset = dr.uint32_array_t(type(col["x0"]))(col["ioff"] + j * st["sb"])
    x = col["x0"] if kk == 0 else dr.fma(st["beta"], float(kk), col["x0"])

    ArrayNf = type(st["knot_start"])
    Float = type(x)
    qx = Float(dr.select(st["x_main"], col["i"], j))
    qy = Float(dr.select(st["x_main"], j, col["i"]))
    center = st["knot_start"] + ArrayNf(qx, qy) * st["knot_step"]
    return offset, valid, x, center


def _slot(st: dict, col: dict, kk: int, degree: int) -> tuple:
    # (offset, valid, weight) of lateral slot kk of a column.
    offset, valid, x, center = _basis(st, col, kk)
    w = _eval(st["knots"], st["coeffs"], x, degree)
    if st["w_density"] is not None:  # weight at the basis function's center
        w *= st["w_density"](center)
    return offset, valid, w


def _walk(
    st: dict, order: int, slot_fn: typ.Callable, carry, slot: typ.Callable = None
):
    # Visit every (ray, basis function) pair:
    #     carry = slot_fn(carry, *slot(st, column, kk, degree))
    # with slot = _slot: carry = slot_fn(carry, offset, valid, weight).
    slot = _slot if (slot is None) else slot
    Int = type(st["K"])
    L = dr.width(st["n_iter"])
    skip = order == 1  # slots 2-3 behind a (warp-uniform) branch on K

    # unmasked rays at degree order + 1, then masked rays at degree order
    for degree, n_iter in (
        (order + 1, dr.select(st["masked"], 0, st["n_iter"])),
        (order, dr.select(st["masked"], st["n_iter"], 0)),
    ):

        def rest(carry, c, degree=degree, n_iter=n_iter):
            col = _column(st, c, n_iter)
            for kk in range(2, K_MAX):
                carry = slot_fn(carry, *slot(st, col, kk, degree))
            return carry

        def body(c, carry, degree=degree, n_iter=n_iter, rest=rest):
            col = _column(st, c, n_iter)
            for kk in range(2):
                carry = slot_fn(carry, *slot(st, col, kk, degree))
            if skip:
                carry = dr.if_stmt(
                    (carry, c), st["K"] > 2, rest, lambda carry, c: carry
                )
            else:
                carry = rest(carry, c)
            return c + 1, carry

        _, carry = dr.while_loop(
            state=(dr.zeros(Int, L), carry),
            cond=lambda c, carry, n_iter=n_iter: c < n_iter,
            body=body,
            mode="symbolic",
            labels=("c", "carry"),
        )
    return carry


def spline2d_apply(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT,
    weight: WeightT = None,
) -> FloatT:
    r"""
    2D projections at order 1 or 2.

    Called by :py:func:`~xrt_toolkit.drjit.ray_xrt.xrt_apply`, which documents the parameters.
    """
    ray_t, ray_n = ray_spec
    Float = dr.value_t(type(ray_t))
    st = _setup(ray_t, ray_n, knot_spec, order, transpose=False, weight=weight)

    def project(accum, offset, valid, w):
        fq = dr.gather(Float, data, offset, valid)
        return dr.fma(fq, w, accum)

    buffer += _walk(st, order, project, dr.zeros(Float, dr.width(buffer)))
    return buffer


def spline2d_adjoint(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT,
    weight: WeightT = None,
) -> FloatT:
    r"""
    Adjoint of :py:func:`spline2d_apply`.

    Called by :py:func:`~xrt_toolkit.drjit.ray_xrt.xrt_adjoint`, which documents the parameters.
    """
    ray_t, ray_n = ray_spec
    Float = dr.value_t(type(ray_t))
    UInt = dr.uint32_array_t(Float)

    # Outside symbolic loops: accumulate into [layout, transposed layout], then fold.
    # Inside one (e.g. xrt_struct_adjoint()), the fold cannot read the buffer back:
    # accumulate in place.
    transpose = not dr.flag(dr.JitFlag.SymbolicScope)
    st = _setup(ray_t, ray_n, knot_spec, order, transpose, weight)

    if transpose:
        Q = math.prod(knot_spec.num)
        target = dr.zeros(Float, 2 * Q)
    else:
        target = buffer

    def back_project(carry, offset, valid, w):
        # `target` is captured, not carried: it is scattered to by both degree passes.
        valid &= w != 0  # outside the support: no atomic
        dr.scatter_add(target, w * data, offset, valid, mode=dr.ReduceMode.Direct)
        return carry

    _walk(st, order, back_project, ())

    if transpose:
        Qx, Qy = knot_spec.num
        k = dr.arange(UInt, Q)
        buffer += dr.gather(Float, target, k)
        buffer += dr.gather(Float, target, Q + (k % Qy) * Qx + k // Qy)
    return buffer


def _closed_form_grad(
    order: int, knot_step: ArrayNfT, n_perp: ArrayNfT, n_hat: ArrayNfT
) -> tuple:
    # Derivatives of _closed_form()'s knots and coeffs in the ray angle theta
    # (d n_perp / d theta = -n_hat), by forward-mode AD outside the walk. On an axis or a
    # diagonal, |E_k| has zero derivative: the average of both sides.
    ArrayNf = type(n_perp)
    Float = dr.value_t(ArrayNf)
    ArrayNf_ad = dr.diff_array_t(ArrayNf)
    Float_ad = dr.value_t(ArrayNf_ad)

    theta = dr.zeros(Float_ad, dr.width(n_perp))
    dr.enable_grad(theta)
    knots, coeffs, _, _ = _closed_form(
        order, ArrayNf_ad(knot_step), ArrayNf_ad(n_perp) - theta * ArrayNf_ad(n_hat)
    )
    dr.forward_from(theta)
    d = lambda v: Float(dr.detach(dr.grad(v)))  # noqa: E731
    return [d(_) for _ in knots], [d(_) for _ in coeffs]


def _slot_grad(st: dict, col: dict, kk: int, degree: int) -> tuple:
    # (offset, valid, dw/dx, dw/dtheta at fixed x, alpha) of lateral slot kk of a column,
    # with x the signed distance of the basis center to the line and alpha its arc length
    # along it.
    offset, valid, x, center = _basis(st, col, kk)
    Float = type(x)
    u = dr.abs(x)
    sgn = dr.select(
        x > 0, Float(1), dr.select(x < 0, Float(-1), 0)
    )  # 0 at x = 0: average
    w_u, w_th = Float(0), Float(0)
    for k_t, c_t, dk_t, dc_t in zip(
        st["knots"], st["coeffs"], st["dknots"], st["dcoeffs"]
    ):
        z = dr.maximum(k_t - u, 0)
        if degree == 1:  # on the kink z = 0: average of both sides
            p, dp = z, dr.select(z > 0, Float(1), dr.select(k_t == u, Float(0.5), 0))
        elif degree == 2:
            p, dp = z * z, 2 * z
        else:
            p, dp = z * z * z, 3 * z * z
        w_u = dr.fma(c_t, dp, w_u)  # d/dk_t of the term, = -d/du
        w_th += dc_t * p + c_t * dp * dk_t
    w_x = -sgn * w_u
    alpha = dr.dot(center - st["ray_t"], st["n_hat"])
    return offset, valid, w_x, w_th, alpha


def spline2d_grad(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
) -> tuple[ArrayNfT, ArrayNfT]:
    r"""
    Gradients of the 2D projections at order 1 or 2 with respect to the rays.

    Called by :py:func:`~xrt_toolkit.drjit.ray_grad.xrt_grad`, which documents the parameters.

    The projection of a basis function is its closed form :math:`w(\rho; \theta)`, with
    :math:`\rho` the signed distance of its center to the line and :math:`\theta` the ray angle.
    The anchor moves :math:`\rho` only; the direction moves :math:`\rho` (by the arc length of the
    center, about the anchor) and the closed form itself.
    """
    ray_t, ray_n = ray_spec
    ArrayNf = type(ray_t)
    Float = dr.value_t(ArrayNf)
    L = max(ray_t.shape[1], ray_n.shape[1])

    st = _setup(ray_t, ray_n, knot_spec, order, transpose=False, closed=True)
    n_hat = dr.normalize(ray_n)
    n_perp = dr.normalize(ArrayNf(-ray_n.y, ray_n.x))
    dknots, dcoeffs = _closed_form_grad(order, ArrayNf(*knot_spec.step), n_perp, n_hat)
    st.update(dknots=dknots, dcoeffs=dcoeffs, ray_t=ray_t, n_hat=n_hat)

    def accumulate(carry, offset, valid, w_x, w_th, alpha):
        S, A, B = carry
        fq = dr.gather(Float, data, offset, valid)
        return dr.fma(fq, w_x, S), dr.fma(fq * w_x, alpha, A), dr.fma(fq, w_th, B)

    carry = tuple(dr.zeros(Float, L) for _ in range(3))
    S, A, B = _walk(st, order, accumulate, carry, slot=_slot_grad)

    # d rho / d t = -n_perp, d rho / d theta = -alpha, d theta / d n = n_perp / |n|
    grad_t = -n_perp * S
    grad_n = n_perp * ((B - A) * dr.rcp(dr.norm(ray_n)))
    return grad_t, grad_n
