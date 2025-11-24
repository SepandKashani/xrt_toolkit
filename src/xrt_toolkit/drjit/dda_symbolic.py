import collections.abc as cabc
import typing as typ

import drjit as dr

from .dda import (
    ArrayNfT,
    ArrayNuT,
    ArrayNiT,
    BoolT,
    StateT,
)


def dda(
    ray_o: ArrayNfT,
    ray_d: ArrayNfT,
    ray_max: object,
    grid_res: ArrayNuT,
    grid_min: ArrayNfT,
    grid_max: ArrayNfT,
    func: cabc.Callable[
        [StateT, ArrayNuT, ArrayNfT, ArrayNfT, BoolT],
        tuple[StateT, BoolT],
    ],
    state: StateT,
    active: BoolT,
    mode: typ.Literal["scalar", "symbolic", "evaluated", None] = None,
    max_iterations: typ.Optional[int] = None,
) -> StateT:
    """
    Variant of :func:`xrt_toolkit.drjit.dda.dda` that mirrors the update scheme
    needed for Dr.Jit's symbolic mode. The main loop avoids masked in-place
    updates on differentiable variables and instead builds updated values via
    ``dr.select``. This sidesteps LoopOp's restriction that a differentiable
    variable cannot simultaneously be treated as both input and output.
    """

    ArrayNf = type(ray_o)
    ArrayNi = dr.int32_array_t(ArrayNf)
    ArrayNu = dr.uint32_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Bool = dr.mask_t(Float)

    assert type(ray_d) is ArrayNf
    assert type(ray_max) is Float
    assert type(grid_res) is ArrayNu
    assert type(grid_min) is ArrayNf
    assert type(grid_max) is ArrayNf
    assert type(active) is Bool

    grid_res_f = ArrayNf(grid_res)
    grid_scale = grid_res_f / (grid_max - grid_min)
    grid_offset = -grid_min * grid_scale

    ray_o = dr.fma(ray_o, grid_scale, grid_offset)
    ray_d = dr.select(active, ray_d * grid_scale, ray_d)
    rcp_d = dr.rcp(ray_d)
    inf_t = ray_d == 0

    t_min_v = -ray_o * rcp_d
    t_max_v = (grid_res_f - ray_o) * rcp_d
    t_min_v2 = dr.minimum(t_min_v, t_max_v)
    t_max_v2 = dr.maximum(t_min_v, t_max_v)

    t_max_v2[inf_t] = dr.inf
    t_min_v2[inf_t] = -dr.inf

    t_min = dr.maximum(dr.max(t_min_v2), 0)
    t_max = dr.minimum(dr.min(t_max_v2), ray_max)

    active = active & (t_max > t_min) & dr.isfinite(t_max)  # type: ignore
    active = active & dr.all(~inf_t | ((0 <= ray_o) & (ray_o <= grid_res_f)))

    ray_o = dr.fma(ray_d, t_min, ray_o)
    t_min, t_max = 0, t_max - t_min  # type: ignore

    step = ArrayNi(dr.select(ray_d >= 0, 1, -1))
    abs_rcp_d = abs(rcp_d)

    pi = dr.clip(ArrayNi(ray_o), 0, ArrayNi(grid_res - 1))
    p0 = ray_o - ArrayNf(pi)

    dt_v = dr.select(ray_d >= 0, dr.fma(-p0, rcp_d, rcp_d), -p0 * rcp_d)
    dt_v[inf_t] = dr.inf

    def body_fn(
        active: BoolT,
        state: StateT,
        dt_v: ArrayNfT,
        p0: ArrayNfT,
        pi: ArrayNiT,
        t_rem: typ.Any,
    ) -> tuple[BoolT, StateT, ArrayNfT, ArrayNfT, ArrayNiT, typ.Any]:
        dt = dr.minimum(dr.min(dt_v), t_rem)
        mask = dt_v == dt

        p1 = dr.fma(ray_d, dt, p0)
        state, cont = func(state, ArrayNu(pi), p0, p1, active & (dt > 0))  # type: ignore

        dt_v_new = dr.select(
            active,
            dr.select(mask, abs_rcp_d, dt_v - dt),
            dt_v,
        )

        p0_step = dr.fma(ray_d, dt, p0)
        p0_reset = dr.select(ray_d >= 0, Float(0), Float(1))
        p0_new = dr.select(active, p0_step, p0)
        p0_new = dr.select(mask & active, p0_reset, p0_new)

        pi_new = dr.select(mask & active, pi + step, pi)

        t_rem_new = dr.select(active, t_rem - dt, t_rem)

        active_new = active & dr.all((pi_new >= 0) & (pi_new < ArrayNi(grid_res))) & (t_rem_new > 0) & cont

        return active_new, state, dt_v_new, p0_new, pi_new, t_rem_new

    return dr.while_loop(
        state=(active, state, dt_v, p0, pi, t_max),
        body=body_fn,
        cond=lambda *args: args[0],
        mode=mode or "symbolic",
        labels=("active", "state", "dt_v", "p0", "pi", "t_rem"),
        max_iterations=max_iterations,
    )[1]
