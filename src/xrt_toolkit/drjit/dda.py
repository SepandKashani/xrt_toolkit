import collections.abc as cabc
import typing as typ

import drjit as dr
from drjit.cuda.ad import UInt32

ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
ArrayNuT = typ.TypeVar("ArrayNuT", bound=dr.AnyArray)
ArrayNiT = typ.TypeVar("ArrayNiT", bound=dr.AnyArray)
BoolT = typ.TypeVar("BoolT", bound=dr.ArrayBase | bool)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
BoolT = typ.TypeVar("BoolT", bound=dr.AnyArray)
TensorXfT = typ.TypeVar("TensorXfT", bound=dr.AnyArray)
StateT = typ.TypeVar("StateT")

from drjit.auto import Bool, UInt

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
    r"""
    N-dimensional digital differential analyzer (DDA).

    This function traverses the intersection of a Cartesian coordinate grid and
    a specified ray or ray segment.  The following snippet shows how to use it
    to enumerate the intersection of a grid with a single ray.

    .. code-block:: python

       from drjit.scalar import Array3f, Array3u, Float, Bool

       def dda_fun(state: list, index: Array3u,
                   pt_in: Array3f, pt_out: Array3f,
                   active: Bool) -> tuple[list, bool]:
           # Entered a grid cell, stash it in the 'state' variable
           state.append(index)
           return state, Bool(True)

       result = dda(
            ray_o    = Array3f(-.1),
            ray_d    = Array3f(.1, .2, .3),
            ray_max  = Float(float('inf')),
            grid_res = Array3u(10),
            grid_min = Array3f(0),
            grid_max = Array3f(1),
            func     = dda_fun,
            state    = [],
            active   = Bool(True)
       )

       print(result)

    Since all input elements are Dr.Jit arrays, everything works analogously
    when processing ``N`` rays and ``N`` potentially different grid
    configurations. The entire process can be captured symbolically.

    The function takes the following arguments. Note that many of them are
    generic `type variables
    <https://mypy.readthedocs.io/en/stable/generics.html>`__ (signaled by ending
    with a capital ``T``). To support different dimensions and precisions, the
    implementation must be able to deal with various input types, which is
    communicated by these type variables.

    Parameters
    ----------
    ray_o: ArrayNfT
        Ray origin, where the ``ArrayNfT`` type variable refers to an
        n-dimensional scalar or Jit-compiled floating point array.

    ray_d: ArrayNfT
        Ray direction. Does not need to be normalized.

    ray_max: object
        Maximum extent along the ray, which is permitted to be infinite. The
        value is specfied as a multiple of the norm of ``ray_d``, which is not
        necessarily unit-length. Must be of type :py:func:`dr.value_t(ArrayNfT)
        <drjit.value_t>`.

    grid_res: ArrayNuT
        Grid resolution, where the ``ArrayNuT`` type variable refers to a
        matched 32-bit unsigned integer array (i.e., :py:func:`ArrayNuT =
        dr.uint32_array_t(ArrayNfT) <drjit.uint32_array_t>`).

    grid_min: ArrayNfT
        Bottom-left corner of the grid bounding box.

    grid_max: ArrayNfT
        Upper-right corner of the grid bounding box.

    func: Callable[[StateT, ArrayNuT, ArrayNfT, ArrayNfT, BoolT], tuple[StateT, BoolT]]
        Callback invoked when the DDA traverses a grid cell. It must take the
        following five positional arguments:

        1. ``arg0: StateT``: Arbitrary state value.

        2. ``arg1: ArrayNuT``: Integer array specifying the cell index along
           each dimension.

        3. ``arg2: ArrayNfT``: Fractional position (:math:`\in [0, 1]^n`) where
           the ray *enters* the current cell.

        4. ``arg3: ArrayNfT``: Fractional position (:math:`\in [0, 1]^n`) where
           the ray *leaves* the current cell.

        5. ``arg4: BoolT``: Boolean array specifying which elements are active.

        The callback should then return a tuple of type ``tuple[StateT, BoolT]``
        containing

        1. An updated state value.

        2. A boolean array that can be used to exit the loop prematurely for
           some or all rays. The iteration stops if the associated entry of the
           return value equals ``False``.

    state: StateT
        Arbitrary *initial* state that will be passed to the callback.

    active: BoolT
        Array specifying which elements of the input are active, where the
        ``BoolT`` type variable refers to a matched boolean array (i.e.,
        :py:func:`BoolT = dr.mask_t(ray_o.x) <drjit.mask_t>`).

    mode: str | None
        The operation can operate in scalar, symbolic, or evaluated modes ---
        see the ``mode`` argument and the documentation of
        :py:func:`drjit.while_loop` for details.

    max_iterations: int | None
        Bound on the iteration count that is needed for reverse-mode
        differentiation. Forwarded to the ``max_iterations`` parameter of
        :py:func:`drjit.while_loop`.

    Returns
    -------
    StateT
        Final state value of the callback upon termination.

    .. note::

       Unlike Dr.Jit's built-in :py:func:`drjit.dda.dda`, all coordinates are
       provided in ``(X, Y, Z)`` order.
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

    # Linear map to grid coordinates (likely optimized away if
    # 'grid_*' are literal constants)
    grid_res_f = ArrayNf(grid_res)
    grid_scale = grid_res_f / (grid_max - grid_min)
    grid_offset = -grid_min * grid_scale

    # Transform the ray into the grid coordinate system
    ray_o = dr.fma(ray_o, grid_scale, grid_offset)
    ray_d[active] = ray_d * grid_scale
    rcp_d = dr.rcp(ray_d)
    inf_t = ray_d == 0

    # Per-axis intersection of the ray with the grid bounds
    t_min_v = -ray_o * rcp_d
    t_max_v = (grid_res_f - ray_o) * rcp_d
    t_min_v2 = dr.minimum(t_min_v, t_max_v)
    t_max_v2 = dr.maximum(t_min_v, t_max_v)

    # Disable extent computation for dims where the ray direction is zero
    t_max_v2[inf_t] = dr.inf
    t_min_v2[inf_t] = -dr.inf

    # Reduce constraints to a single ray interval
    t_min = dr.maximum(dr.max(t_min_v2), 0)
    t_max = dr.minimum(dr.min(t_max_v2), ray_max)

    # Only run the DDA algorithm if the interval is nonempty
    active = active & (t_max > t_min) & dr.isfinite(t_max)  # type: ignore

    # Deactivate rays that have zero direction along any axis
    # and whose origin along that axis is outside the grid bounds
    active = active & dr.all(~inf_t | ((0 <= ray_o) & (ray_o <= grid_res_f)))

    # Advance the ray to the start of the interval
    ray_o = dr.fma(ray_d, t_min, ray_o)
    t_min, t_max = 0, t_max - t_min  # type: ignore

    # Compute the integer step direction
    step = ArrayNi(dr.select(ray_d >= 0, 1, -1))
    abs_rcp_d = abs(rcp_d)

    # Integer grid coordinates
    pi = dr.clip(ArrayNi(ray_o), 0, ArrayNi(grid_res - 1))

    # Fractional entry position
    p0 = ray_o - ArrayNf(pi)

    # Step size to next interaction
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
        # Select the smallest step. It's possible that dt == 0 when starting
        # directly on a grid line.

        dt = dr.minimum(dr.min(dt_v), t_rem)
        mask = dt_v == dt

        # Compute an updated position
        ray_d2 = ray_d#dr.gather(ArrayNf, ray_d, dr.arange(UInt32, dr.width(ray_d)), shape=(3,))
        p1 = dr.fma(ray_d2, dt, p0)

        # Invoke the user-provided callback
        state, cont = func(state, ArrayNu(pi), p0, p1, active & (dt > 0))  # type: ignore

        # Advance
        abs_rcp_d2 = abs_rcp_d#dr.gather(ArrayNf, abs_rcp_d, dr.arange(UInt32, dr.width(abs_rcp_d)), shape=(3,))

        dt_v[active] = dr.select(mask, abs_rcp_d2, dt_v - dt)

        p0[active] = dr.fma(ray_d2, dt, p0)
        p0[mask & active] = dr.select(ray_d2 >= 0, Float(0), Float(1))

        pi[mask & active] += step
        t_rem[active] = t_rem - dt

        active[active] &= dr.all((pi >= 0) & (pi < ArrayNi(grid_res))) & (t_rem > 0) & cont

        return active, state, dt_v, p0, pi, t_rem
        
    return dr.while_loop(
        state=(active, state, dt_v, p0, pi, t_max),
        body=body_fn,
        cond=lambda *args: args[0],
        mode=mode,
        labels=("active", "state", "dt_v", "p1", "pi", "t_rem"),
        max_iterations=-1,
    )[1]

