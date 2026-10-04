r"""
Refractive (curved-ray) transforms for bent-ray travel-time tomography.

The straight-ray operators in :mod:`ray_xrt` integrate a field along a fixed
line :math:`\mathbf{x}(s) = \mathbf{t} + s\,\mathbf{n}`. In a medium with a
spatially varying wave speed :math:`c(\mathbf{x})` the ray *bends*: to first
order in wavelength (the geometrical-acoustics / eikonal limit) it follows the
characteristic of the eikonal equation :math:`|\nabla T| = u`, with slowness
:math:`u = 1/c`. In arc-length form the ray obeys

.. math::

   \frac{d\mathbf{x}}{ds} = \boldsymbol{\tau}, \qquad
   \frac{d\boldsymbol{\tau}}{ds}
     = \frac{1}{u}\bigl(\nabla u - (\nabla u\cdot\boldsymbol{\tau})\,\boldsymbol{\tau}\bigr)
     = \frac{1}{u}\,\nabla_{\!\perp} u,

i.e. the unit tangent :math:`\boldsymbol{\tau}` turns toward higher slowness
(lower speed). The first-arrival travel time is the slowness integrated along
that bent path,

.. math::

   T = \int_{\text{ray}} u(\mathbf{x})\,ds .

This is the forward model of ultrasound computed tomography (USCT) and seismic
travel-time tomography. Compared to the two standard families:

* **Eikonal / fast-marching solvers** compute first arrivals by a *sequential*
  causal sweep (a priority queue), which parallelises poorly. Here every ray is
  an independent SPMD lane -- thousands march concurrently on the GPU.
* **Full-waveform inversion** (e.g. k-Wave) solves the full wave equation and is
  the fidelity gold standard, but two to three orders of magnitude costlier. The
  ray model is first-arrival only (no diffraction / finite-frequency effects).

**Cost.** One march step is a straight DDA step plus a single slowness-gradient
sample, so a refractive pass costs roughly twice an ``order``-0 straight pass and
scales the same way (rays x steps), fully on the GPU.

**Matched adjoint (Fermat).** The travel time is *stationary* with respect to
first-order perturbations of the path (Fermat's principle), so the sensitivity
of :math:`T` to the slowness field is the reconstruction footprint integrated
along the frozen ray,

.. math::

   \frac{\partial T}{\partial u_{\mathbf{q}}}
     = \int_{\text{ray}} \varphi_{\mathbf{q}}(\mathbf{x})\,ds ,

which is exactly the transpose of the (frozen-path) forward operator. This is the
standard bent-ray tomography linearisation; Fermat's principle guarantees it is
first-order accurate. :func:`refract_apply` and :func:`refract_adjoint` are an
exact transpose pair (validated by a dot test), so they drop straight into the
CG / Gauss-Newton loops already used for the straight-ray operators.

Design notes. The slowness field is reconstructed with continuous multilinear
interpolation of the coefficient grid so that :math:`\nabla u` exists (this is
what refraction requires; the piecewise-constant ``order``-0 field cannot bend a
ray). The path integral uses the arc-length midpoint rule with an RK2 (midpoint)
step for the coupled position/direction ODE, matched to :math:`O(ds^2)`. The
straight-ray operators in :mod:`ray_xrt` are **not** modified; setting
``bend=False`` reduces this module to a straight quadrature integrator.
"""

import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

from .bbox import bbox_contains, ray_bbox_intersect

BoolT = typ.TypeVar("BoolT", bound=dr.AnyArray)
ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
ArrayNiT = typ.TypeVar("ArrayNiT", bound=dr.AnyArray)
ArrayNuT = typ.TypeVar("ArrayNuT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNfT, ArrayNfT]

_EPS = 1e-8


def _default_ds(knot_step) -> float:
    """One march step per shortest cell edge."""
    return float(min(knot_step))


def _grid_maps(knot_spec, ArrayNf, ArrayNu):
    """Grid geometry shared by the forward and adjoint passes."""
    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = bbox_ll + ArrayNf(knot_num) * knot_step
    D = dr.size_v(ArrayNf)
    if D == 2:
        stride = ArrayNu(knot_num.y, 1)
    else:
        stride = ArrayNu(knot_num.y * knot_num.z, knot_num.z, 1)
    return knot_start, knot_step, knot_num, bbox_ll, bbox_ur, stride


def _corner_bits(D: int):
    """Enumerate the 2**D corner offsets of a grid cell as bit tuples."""
    out = []
    for k in range(2 ** D):
        out.append(tuple((k >> d) & 1 for d in range(D)))
    return out


# ---------------------------------------------------------------- operators --
def refract_apply(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    coeff: FloatT,
    coeff_geom: FloatT = None,
    *,
    ds: float = None,
    max_steps: int = None,
    bend: bool = True,
    smax: FloatT = None,
    buffer: FloatT = None,
    mode: str = "symbolic",
    return_path: bool = False,
):
    r"""
    Travel time along refractive rays,
    :math:`T_l = \int_{\text{ray}_l} u(\mathbf{x})\,ds`, marched through the
    slowness field with a per-step direction update.

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\mathbf{t}` (source positions) and initial
        directions :math:`\mathbf{n}` (launch directions), as in ``xrt_apply``.
    knot_spec: UniformSpec
        Volume properties :math:`(\mathbf{x}_0, \boldsymbol{\Delta}, \mathbf{Q})`.
    coeff: FloatT
        (Q1*...*QD,) flattened slowness field :math:`u = 1/c`. This field is
        both bent through and integrated (the nonlinear forward), unless
        ``coeff_geom`` is given.
    coeff_geom: FloatT | None
        Optional separate field that drives the bending while ``coeff`` is the
        integrand. Passing the reference slowness here and a perturbation as
        ``coeff`` yields the *linearised* (frozen-path) forward operator whose
        transpose is :func:`refract_adjoint`.
    ds: float | None
        Arc-length step (world units). Default: shortest cell edge.
    max_steps: int | None
        March-step cap. Default: bounding-box diagonal / ``ds`` + 4.
    bend: bool
        If ``False`` the direction is held fixed (straight quadrature); useful
        as a sanity baseline.
    smax: FloatT | float | None
        Optional per-ray arc-length limit measured from the anchor. For
        transmission tomography set it to the source--receiver distance: the
        integral then stops at the receiver instead of the volume boundary
        (the final step is shortened to land exactly on ``smax``). By Fermat's
        principle the residual error from the small lateral miss of a shot ray
        is second order.
    return_path: bool
        If ``True`` also return the per-ray exit position and exit direction.

    Returns
    -------
    time: FloatT
        (L,) travel times. If ``return_path``: ``(time, x_exit, tau_exit)``.
    """
    ray_t, ray_n = ray_spec

    ArrayNf = type(ray_t)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)
    UInt = dr.value_t(ArrayNu)

    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D in (2, 3))
    assert type(ray_n) is ArrayNf
    assert knot_spec.ndim == D
    assert type(coeff) is Float
    assert len(coeff) == math.prod(knot_spec.num)
    if coeff_geom is None:
        coeff_geom = coeff
    else:
        assert type(coeff_geom) is Float and len(coeff_geom) == len(coeff)

    L = max(ray_t.shape[1], ray_n.shape[1])
    if buffer is None:
        buffer = dr.zeros(Float, L)
    else:
        assert type(buffer) is Float and len(buffer) == L

    (knot_start, knot_step, knot_num,
     bbox_ll, bbox_ur, stride) = _grid_maps(knot_spec, ArrayNf, ArrayNu)

    if ds is None:
        ds = _default_ds(knot_spec.step)
    ds = Float(ds)
    if max_steps is None:
        diag = float(dr.norm(bbox_ur - bbox_ll)[0])
        max_steps = int(diag / float(ds[0])) + 4

    sample = _make_sampler(coeff_geom, coeff, knot_start, knot_step, knot_num,
                           stride, ArrayNf, ArrayNu, Float, D)

    if smax is not None:
        smax = smax if type(smax) is Float else Float(smax)

    # March all rays from the bounding-box entry point along the straight launch
    # direction (the medium outside the grid is field-free, hence unbent).
    # The entry point lies on the box face by construction; fp32 rounding can
    # push it marginally outside, so clamp it back (exact, unbiased).
    n0 = dr.normalize(ray_n)
    active, t1, t2 = ray_bbox_intersect(bbox_ll, bbox_ur, ray_t, n0)
    t_enter = dr.maximum(dr.minimum(t1, t2), 0)
    x = dr.clip(ray_t + t_enter * n0, bbox_ll, bbox_ur)
    tau = ArrayNf(n0)
    T = Float(buffer)
    step = UInt(0)
    s_arc = Float(t_enter) if smax is not None else Float(0)
    active = Bool(active)
    if smax is not None:
        active = active & (s_arc < smax)

    def body(active, x, tau, T, step, s_arc):
        # per-lane step: shorten the final step to land exactly on smax.
        if smax is not None:
            ds_l = dr.minimum(ds, dr.maximum(smax - s_arc, 0))
        else:
            ds_l = ds
        u0, g0 = sample.geom(x, active)
        # RK2 (midpoint) on (x, tau); direction only turns via the perpendicular
        # slowness gradient so |tau| is preserved to O(ds^2), re-normalised each step.
        a0 = _bend_accel(g0, tau, u0, bend, Float)
        x_m = dr.fma(0.5 * ds_l, tau, x)
        tau_m = dr.normalize(dr.fma(0.5 * ds_l, a0, tau))
        u_m, g_m = sample.geom(x_m, active)
        a_m = _bend_accel(g_m, tau_m, u_m, bend, Float)
        x_next = dr.fma(ds_l, tau_m, x)
        tau_next = dr.normalize(dr.fma(ds_l, a_m, tau))

        # Midpoint-rule slowness sample of the integrand field. The sample is
        # gated by whether the midpoint lies in the box so the last (partial)
        # step never counts a cell outside the domain; the adjoint gates the
        # scatter identically, keeping the pair an exact transpose.
        inside_m = active & bbox_contains(bbox_ll, bbox_ur, x_m)
        u_int = sample.integrand(x_m, inside_m)
        T = T + dr.select(inside_m, ds_l * u_int, 0)

        s_next = s_arc + ds_l
        inside = bbox_contains(bbox_ll, bbox_ur, x_next)
        active_next = active & inside & (step + 1 < max_steps)
        if smax is not None:
            active_next = active_next & (s_next < smax)
        return (active_next, x_next, tau_next, T, step + 1, s_next)

    _, x, tau, T, _, _ = dr.while_loop(
        state=(active, x, tau, T, step, s_arc),
        cond=lambda active, *_: active,
        body=body,
        mode=mode,
        labels=("active", "x", "tau", "T", "step", "s_arc"),
        max_iterations=-1,
    )

    if return_path:
        return T, dr.clip(x, bbox_ll, bbox_ur), tau
    return T


def refract_adjoint(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    residual: FloatT,
    coeff_geom: FloatT,
    *,
    ds: float = None,
    max_steps: int = None,
    bend: bool = True,
    smax: FloatT = None,
    buffer: FloatT = None,
    mode: str = "symbolic",
) -> FloatT:
    r"""
    Exact transpose of the frozen-path forward operator: scatters per-ray
    residuals along the bent rays into the slowness grid,
    :math:`(\mathbf{A}^*\mathbf{r})_{\mathbf{q}} = \sum_l r_l
    \int_{\text{ray}_l}\varphi_{\mathbf{q}}\,ds`.

    The rays are bent through ``coeff_geom`` (the reference slowness); the path
    is thus frozen and the operator is linear in the residual. By Fermat's
    principle this is also the gradient of :math:`\tfrac12\|T(u)-T^{\text{obs}}\|^2`
    with respect to the slowness field.

    Parameters mirror :func:`refract_apply`; ``residual`` is the (L,) per-ray
    quantity to back-project and the return value is the (Q1*...*QD,) grid.
    """
    ray_t, ray_n = ray_spec

    ArrayNf = type(ray_t)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)
    UInt = dr.value_t(ArrayNu)

    D = dr.size_v(ArrayNf)
    assert (ray_t.ndim == 2) and (D in (2, 3))
    assert type(ray_n) is ArrayNf
    assert knot_spec.ndim == D
    assert type(coeff_geom) is Float
    assert len(coeff_geom) == math.prod(knot_spec.num)

    L = max(ray_t.shape[1], ray_n.shape[1])
    assert type(residual) is Float and len(residual) == L

    if buffer is None:
        buffer = dr.zeros(Float, math.prod(knot_spec.num))
    else:
        assert type(buffer) is Float and len(buffer) == math.prod(knot_spec.num)

    (knot_start, knot_step, knot_num,
     bbox_ll, bbox_ur, stride) = _grid_maps(knot_spec, ArrayNf, ArrayNu)

    if ds is None:
        ds = _default_ds(knot_spec.step)
    ds = Float(ds)
    if max_steps is None:
        diag = float(dr.norm(bbox_ur - bbox_ll)[0])
        max_steps = int(diag / float(ds[0])) + 4

    sample = _make_sampler(coeff_geom, None, knot_start, knot_step, knot_num,
                           stride, ArrayNf, ArrayNu, Float, D)

    if smax is not None:
        smax = smax if type(smax) is Float else Float(smax)

    # Same entry-point construction as the forward (see comment there).
    n0 = dr.normalize(ray_n)
    active, t1, t2 = ray_bbox_intersect(bbox_ll, bbox_ur, ray_t, n0)
    t_enter = dr.maximum(dr.minimum(t1, t2), 0)
    x = dr.clip(ray_t + t_enter * n0, bbox_ll, bbox_ur)
    tau = ArrayNf(n0)
    step = UInt(0)
    s_arc = Float(t_enter) if smax is not None else Float(0)
    active = Bool(active)
    if smax is not None:
        active = active & (s_arc < smax)
    accum = Float(buffer)

    def body(active, x, tau, accum, step, s_arc):
        if smax is not None:
            ds_l = dr.minimum(ds, dr.maximum(smax - s_arc, 0))
        else:
            ds_l = ds
        u0, g0 = sample.geom(x, active)
        a0 = _bend_accel(g0, tau, u0, bend, Float)
        x_m = dr.fma(0.5 * ds_l, tau, x)
        tau_m = dr.normalize(dr.fma(0.5 * ds_l, a0, tau))
        u_m, g_m = sample.geom(x_m, active)
        a_m = _bend_accel(g_m, tau_m, u_m, bend, Float)
        x_next = dr.fma(ds_l, tau_m, x)
        tau_next = dr.normalize(dr.fma(ds_l, a_m, tau))

        # Transpose of T += ds_l * interp(x_m): scatter ds_l * residual with
        # the multilinear weights of x_m into the corner cells. Same gating as
        # the forward, so the pair is exact.
        inside_m = active & bbox_contains(bbox_ll, bbox_ur, x_m)
        sample.scatter(accum, x_m, ds_l * residual, inside_m)

        s_next = s_arc + ds_l
        inside = bbox_contains(bbox_ll, bbox_ur, x_next)
        active_next = active & inside & (step + 1 < max_steps)
        if smax is not None:
            active_next = active_next & (s_next < smax)
        return (active_next, x_next, tau_next, accum, step + 1, s_next)

    _, _, _, accum, _, _ = dr.while_loop(
        state=(active, x, tau, accum, step, s_arc),
        cond=lambda active, *_: active,
        body=body,
        mode=mode,
        labels=("active", "x", "tau", "accum", "step", "s_arc"),
        max_iterations=-1,
    )
    return accum


# ----------------------------------------------------------- field sampling --
def _bend_accel(grad, tau, u, bend, Float):
    """Perpendicular slowness acceleration ``(1/u)(grad - (grad.tau) tau)``."""
    if not bend:
        return dr.zeros_like(tau)
    g_perp = grad - dr.dot(grad, tau) * tau
    inv_u = dr.rcp(dr.maximum(u, Float(_EPS)))
    return g_perp * inv_u


class _Sampler(typ.NamedTuple):
    geom: typ.Callable      # x, active -> (value, grad) of the bending field
    integrand: typ.Callable  # x, active -> value of the integrand field
    scatter: typ.Callable    # accum, x, amount, active -> None


def _make_sampler(coeff_geom, coeff_int, knot_start, knot_step, knot_num,
                  stride, ArrayNf, ArrayNu, Float, D):
    r"""
    Continuous multilinear reconstruction of a coefficient grid and its
    transpose. The coefficient :math:`u_{\mathbf{q}}` sits at
    :math:`\mathbf{x}_0 + \mathbf{q}\odot\boldsymbol{\Delta}` (cell centres), so
    the grid coordinate of a world point is
    :math:`\mathbf{g} = (\mathbf{x}-\mathbf{x}_0)/\boldsymbol{\Delta}`.
    """
    ArrayNi = dr.int32_array_t(ArrayNf)
    UInt = dr.value_t(ArrayNu)
    stride_i = ArrayNi(stride)
    inv_step = dr.rcp(knot_step)
    q_max = ArrayNi(knot_num) - 1
    corners = _corner_bits(D)

    def _base_frac(x):
        g = (x - knot_start) * inv_step
        i0 = ArrayNi(dr.floor(g))
        f = g - ArrayNf(i0)
        return i0, f

    def _corner_terms(i0, f):
        """Yield (offset, weight, grad_weight) for each cell corner."""
        one_minus = ArrayNf(1.0) - f
        for bits in corners:
            idx = i0 + ArrayNi(*bits)
            idx_c = dr.clip(idx, ArrayNi(0), q_max)
            offset = UInt(dr.dot(idx_c, stride_i))
            # multilinear weight and its per-axis derivative wrt world coords
            w = Float(1.0)
            comp = []
            for d in range(D):
                wd = f[d] if bits[d] else one_minus[d]
                comp.append(wd)
                w = w * wd
            grad = ArrayNf(0.0)
            for d in range(D):
                gd = Float(1.0)
                for e in range(D):
                    if e == d:
                        gd = gd * (Float(1.0) if bits[d] else Float(-1.0))
                    else:
                        gd = gd * comp[e]
                grad[d] = gd * inv_step[d]
            yield offset, w, grad

    def value_grad(coeff, x, active):
        i0, f = _base_frac(x)
        val = Float(0.0)
        grad = ArrayNf(0.0)
        for offset, w, gw in _corner_terms(i0, f):
            cq = dr.gather(Float, coeff, offset, active)
            val = dr.fma(w, cq, val)
            grad = grad + gw * cq
        return val, grad

    def value(coeff, x, active):
        i0, f = _base_frac(x)
        val = Float(0.0)
        for offset, w, _ in _corner_terms(i0, f):
            cq = dr.gather(Float, coeff, offset, active)
            val = dr.fma(w, cq, val)
        return val

    def scatter(accum, x, amount, active):
        i0, f = _base_frac(x)
        for offset, w, _ in _corner_terms(i0, f):
            dr.scatter_add(accum, w * amount, offset, active)

    def geom(x, active):
        return value_grad(coeff_geom, x, active)

    def integrand(x, active):
        return value(coeff_int, x, active)

    return _Sampler(geom=geom, integrand=integrand, scatter=scatter)


# ---------------------------------------------------------------- utilities --
def refract_time(ray_spec, knot_spec, coeff, **kw):
    """Convenience: nonlinear first-arrival travel time (bend and integrate the
    same slowness field). Accepts the keyword arguments of :func:`refract_apply`.
    """
    return refract_apply(ray_spec, knot_spec, coeff, **kw)
