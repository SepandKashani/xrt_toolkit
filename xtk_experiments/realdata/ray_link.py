"""Two-point ray linking by shooting (2D), on top of XTK's refractive marcher.

For a fixed source and receiver the ray is an unknown: we shoot a candidate
ray, track its closest approach to the receiver, and run a secant iteration on
the SIGNED lateral miss to adjust the launch angle until the ray lands on the
receiver. All pairs are linked simultaneously (one lane per pair).

Returns the travel time at closest approach together with the launch direction
and the arc length, which is exactly what `refract_adjoint(..., smax=...)`
needs to backproject a residual along the very same path.
"""
import drjit as dr
from drjit.cuda.ad import Array2f, Float

from xrt_toolkit.drjit.bbox import bbox_contains, ray_bbox_intersect
from xrt_toolkit.drjit.curved_xrt import (_bend_accel, _default_ds, _grid_maps,
                                          _make_sampler)


def _shoot_track(source, ldir, recv, knot_spec, coeff_geom, coeff_int,
                 ds, max_steps, bend=True):
    """March from `source` along `ldir`; report the state at closest approach
    to `recv`: (T, d_min, signed_miss, arc_length)."""
    ArrayNf = Array2f
    ArrayNu = dr.uint32_array_t(ArrayNf)
    UInt = dr.value_t(ArrayNu)

    (knot_start, knot_step, knot_num,
     bbox_ll, bbox_ur, stride) = _grid_maps(knot_spec, ArrayNf, ArrayNu)
    sample = _make_sampler(coeff_geom, coeff_int, knot_start, knot_step,
                           knot_num, stride, ArrayNf, ArrayNu, Float, 2)
    ds = Float(ds)

    n0 = dr.normalize(ldir)
    active, t1, t2 = ray_bbox_intersect(bbox_ll, bbox_ur, source, n0)
    t_enter = dr.maximum(dr.minimum(t1, t2), 0)
    x = dr.clip(source + t_enter * n0, bbox_ll, bbox_ur)
    tau = ArrayNf(n0)
    T = dr.zeros(Float, dr.width(source))
    s = Float(t_enter)
    step = UInt(0)
    d_min = dr.norm(x - recv)
    T_min = Float(T)
    s_min = Float(s)
    x_min = ArrayNf(x)
    active = dr.mask_t(Float)(active)

    def body(active, x, tau, T, s, d_min, T_min, s_min, x_min, step):
        u0, g0 = sample.geom(x, active)
        a0 = _bend_accel(g0, tau, u0, bend, Float)
        x_m = dr.fma(0.5 * ds, tau, x)
        tau_m = dr.normalize(dr.fma(0.5 * ds, a0, tau))
        u_m, g_m = sample.geom(x_m, active)
        a_m = _bend_accel(g_m, tau_m, u_m, bend, Float)
        x_next = dr.fma(ds, tau_m, x)
        tau_next = dr.normalize(dr.fma(ds, a_m, tau))

        inside_m = active & bbox_contains(bbox_ll, bbox_ur, x_m)
        u_int = sample.integrand(x_m, inside_m)
        T_next = T + dr.select(inside_m, ds * u_int, 0)
        s_next = s + ds

        d = dr.norm(x_next - recv)
        upd = active & (d < d_min)
        d_min2 = dr.select(upd, d, d_min)
        T_min2 = dr.select(upd, T_next, T_min)
        s_min2 = dr.select(upd, s_next, s_min)
        x_min2 = ArrayNf(dr.select(upd, x_next.x, x_min.x),
                         dr.select(upd, x_next.y, x_min.y))

        inside = bbox_contains(bbox_ll, bbox_ur, x_next)
        active_next = active & inside & (step + 1 < max_steps)
        return (active_next, x_next, tau_next, T_next, s_next,
                d_min2, T_min2, s_min2, x_min2, step + 1)

    _, _, _, _, _, d_min, T_min, s_min, x_min, _ = dr.while_loop(
        state=(active, x, tau, T, s, d_min, T_min, s_min, x_min, step),
        cond=lambda a, *_: a, body=body, mode="symbolic",
        labels=("active", "x", "tau", "T", "s", "d_min", "T_min", "s_min",
                "x_min", "step"),
        max_iterations=-1)

    miss = n0.x * (x_min.y - recv.y) - n0.y * (x_min.x - recv.x)
    return T_min, d_min, miss, s_min


def _rotate(n0, delta):
    c, s = dr.cos(delta), dr.sin(delta)
    return Array2f(c * n0.x - s * n0.y, s * n0.x + c * n0.y)


def link_time(source, recv, knot_spec, coeff, ds=None, max_steps=None,
              n_iter=8, bend=True):
    """Two-point bent-ray travel time source->recv by secant ray linking.

    Returns (T, d_min, launch_direction, arc_length).
    """
    ArrayNf = Array2f
    ArrayNu = dr.uint32_array_t(ArrayNf)
    (_, knot_step, _, bbox_ll, bbox_ur, _) = _grid_maps(knot_spec, ArrayNf, ArrayNu)
    if ds is None:
        ds = _default_ds(knot_spec.step)
    if max_steps is None:
        diag = float(dr.norm(bbox_ur - bbox_ll)[0])
        max_steps = int(3 * diag / ds) + 8

    n0 = dr.normalize(recv - source)

    def shoot(delta):
        T, d, m, smax = _shoot_track(source, _rotate(n0, delta), recv, knot_spec,
                                     coeff, coeff, ds, max_steps, bend)
        dr.eval(T, d, m, smax)
        return T, d, m, smax

    d0 = Float(0.0)
    d1 = Float(0.02)                        # ~1.1 deg probe
    _, _, m0, _ = shoot(d0)
    T1, dmin1, m1, smax1 = shoot(d1)
    for _ in range(n_iter):
        denom = m1 - m0
        step = dr.select(dr.abs(denom) > 1e-9, m1 * (d1 - d0) / denom, Float(0.0))
        d2 = d1 - dr.clip(step, -0.5, 0.5)
        T2, dmin2, m2, smax2 = shoot(d2)
        d0, m0 = d1, m1
        d1, m1, T1, dmin1, smax1 = d2, m2, T2, dmin2, smax2
    return T1, dmin1, _rotate(n0, d1), smax1
