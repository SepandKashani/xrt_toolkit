import typing as typ

import drjit as dr
import numpy as np

import xrt_toolkit.util as xrtu

ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
ArrayNiT = typ.TypeVar("ArrayNiT", bound=dr.AnyArray)
ArrayNbT = typ.TypeVar("ArrayNbT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
IntT = typ.TypeVar("IntT", bound=dr.AnyArray)
BoolT = typ.TypeVar("BoolT", bound=dr.AnyArray)


def box_spline_1d_np(
    E: np.ndarray,
    x: np.ndarray,
) -> np.ndarray:
    r"""
    NumPy implementation to compute 1D box-splines :math:`\psi(x; \bbE \in \bR^{N})`.

    Parameters
    ----------
    E: NDArray
        (N,) box-spline directions :math:`\bbE`, assumed non-negative.
    x: NDArray
        (Q,) evaluation points :math:`x \in \bR`.

    Returns
    -------
    y: NDArray
        (Q,) box-spline values :math:`\psi(x; \bbE)`.
    """
    assert E.shape == (E.size,)
    assert x.shape == (x.size,)

    def factorial(n: int) -> int:
        assert n >= 0
        if n == 0:
            return 1
        else:
            return np.arange(1, n + 1).prod()

    def A(n: int) -> np.ndarray:
        assert n >= 1

        A_prev = np.c_[1.0, -1].T  # (2, 1)
        if n == 1:
            A = A_prev
        else:
            for i in range(2, n + 1):
                A = np.zeros((2**i, i))
                A[:, 0] = np.kron([1, -1], np.ones(2 ** (i - 1)))
                A[:, 1:] = np.kron(np.c_[1, 1].T, A_prev)
                A_prev = A

        return A  # (2**n, n)

    def B(E_mask: np.ndarray) -> np.ndarray:
        n = E_mask.size
        assert E_mask.shape == (n,)

        B = np.r_[1.0]  # (1,)
        for i in range(n):
            if E_mask[i]:
                b = np.r_[1, -1]
            else:
                b = np.r_[1, 0]
            B = np.kron(B, b)

        return B  # (2**n,)

    E_mask = abs(E) >= 1e-3
    N_tot = len(E_mask)
    N_nz = E_mask.sum().item()

    if N_nz == 1:
        y = np.where(abs(x) <= E.sum() / 2, 1.0, 0)
    else:
        lhs = (x.reshape(-1, 1) + (0.5 * A(N_tot) @ E)).clip(0, None) ** (N_nz - 1)
        rhs = B(E_mask)
        y = lhs @ rhs
    y /= factorial(N_nz - 1) * (E + (1 - E_mask)).prod()

    return y


def box_spline_1d_dr(
    E: ArrayNfT,
    E_mask: ArrayNiT,
    x: FloatT,
) -> FloatT:
    r"""
    DrJit implementation to compute 1D box-splines :math:`\psi(x; \bbE \in \bR^{N})`.

    Parameters
    ----------
    E: ArrayNfT
        (N,) box-spline directions :math:`\bbE`, assumed non-negative.
    E_mask: ArrayNiT
        (N,) mask with non-zero entries in :math:`\bbE`. (Integer-valued, not binary.)
    x: FloatT
        Evaluation points :math:`x \in \bR`.

    Returns
    -------
    y: FloatT
        Box-spline values :math:`\psi(x; \bbE)`.

    Notes
    -----
    This implementation follows closely that of ``box_spline_1d_np()``.
    """

    ArrayNf = type(E)
    ArrayNi = dr.int_array_t(ArrayNf)
    Float = dr.value_t(ArrayNf)
    Int = dr.value_t(ArrayNi)

    # type checking ---------------------------------------
    assert type(E_mask) is ArrayNi
    assert type(x) is Float
    # -----------------------------------------------------

    def factorial(n: IntT) -> IntT:
        return dr.gather(
            Int,
            Int(1, 1, 2, 6, 24),  # don't need larger factorials here
            n,
        )

    def bs_E1(E: ArrayNfT, E_mask: ArrayNiT, x: FloatT) -> FloatT:
        y = dr.select(
            dr.abs(x) <= 0.5 * dr.sum(E),
            Float(1),
            Float(0),
        )
        return y

    def bs_Ek(E: ArrayNfT, E_mask: ArrayNiT, x: FloatT) -> FloatT:
        N_tot = dr.size_v(ArrayNf)
        assert 2 <= N_tot <= 4

        y = dr.zeros(Float, dr.shape(x))
        N_nz = dr.sum(E_mask)
        lhs_expr = lambda sign: dr.maximum(0, x + 0.5 * sign @ E) ** (N_nz - 1)
        if N_tot == 2:
            lhs = lhs_expr(ArrayNf(+1, +1))
            rhs = 1 * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, -1))
            rhs = 1 * (-E_mask[1])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, +1))
            rhs = (-E_mask[0]) * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, -1))
            rhs = (-E_mask[0]) * (-E_mask[1])
            y += lhs * rhs
        elif N_tot == 3:
            lhs = lhs_expr(ArrayNf(+1, +1, +1))
            rhs = 1 * 1 * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, +1, -1))
            rhs = 1 * 1 * (-E_mask[2])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, -1, +1))
            rhs = 1 * (-E_mask[1]) * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, -1, -1))
            rhs = 1 * (-E_mask[1]) * (-E_mask[2])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, +1, +1))
            rhs = (-E_mask[0]) * 1 * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, +1, -1))
            rhs = (-E_mask[0]) * 1 * (-E_mask[2])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, -1, +1))
            rhs = (-E_mask[0]) * (-E_mask[1]) * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, -1, -1))
            rhs = (-E_mask[0]) * (-E_mask[1]) * (-E_mask[2])
            y += lhs * rhs
        elif N_tot == 4:
            lhs = lhs_expr(ArrayNf(+1, +1, +1, +1))
            rhs = 1 * 1 * 1 * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, +1, +1, -1))
            rhs = 1 * 1 * 1 * (-E_mask[3])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, +1, -1, +1))
            rhs = 1 * 1 * (-E_mask[2]) * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, +1, -1, -1))
            rhs = 1 * 1 * (-E_mask[2]) * (-E_mask[3])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, -1, +1, +1))
            rhs = 1 * (-E_mask[1]) * 1 * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, -1, +1, -1))
            rhs = 1 * (-E_mask[1]) * 1 * (-E_mask[3])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, -1, -1, +1))
            rhs = 1 * (-E_mask[1]) * (-E_mask[2]) * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(+1, -1, -1, -1))
            rhs = 1 * (-E_mask[1]) * (-E_mask[2]) * (-E_mask[3])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, +1, +1, +1))
            rhs = (-E_mask[0]) * 1 * 1 * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, +1, +1, -1))
            rhs = (-E_mask[0]) * 1 * 1 * (-E_mask[3])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, +1, -1, +1))
            rhs = (-E_mask[0]) * 1 * (-E_mask[2]) * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, +1, -1, -1))
            rhs = (-E_mask[0]) * 1 * (-E_mask[2]) * (-E_mask[3])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, -1, +1, +1))
            rhs = (-E_mask[0]) * (-E_mask[1]) * 1 * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, -1, +1, -1))
            rhs = (-E_mask[0]) * (-E_mask[1]) * 1 * (-E_mask[3])
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, -1, -1, +1))
            rhs = (-E_mask[0]) * (-E_mask[1]) * (-E_mask[2]) * 1
            y += lhs * rhs

            lhs = lhs_expr(ArrayNf(-1, -1, -1, -1))
            rhs = (-E_mask[0]) * (-E_mask[1]) * (-E_mask[2]) * (-E_mask[3])
            y += lhs * rhs

        return y

    N_nz = dr.sum(E_mask)
    y = dr.if_stmt(
        args=(E, E_mask, x),
        cond=N_nz == 1,
        true_fn=bs_E1,
        false_fn=bs_Ek,
    )
    y *= dr.rcp(factorial(N_nz - 1) * dr.prod(E + 1 - E_mask))

    return y


def box_spline_1d_E(
    order: int,
    scale: ArrayNfT,
    proj_dir: ArrayNfT = None,
):
    r"""
    Compute 1D box-spline parameters `\bbE \in \bR^{N}` given 2D projection directions.

    Parameters
    ----------
    order: 0 | 1 | 2
        Data interpolation order.
        (See :py:func:`~xrt_toolkit.drjit.ray_xrt.xrt_apply`.)
    scale: ArrayNfT
        (2,) axial scaling vector applied to :math:`\bbE`.
        (This can be assimilated to `knot_spec.step` from :py:func:`~xrt_toolkit.drjit.ray_xrt.xrt_apply`.)
    proj_dir: ArrayNfT, None
        (2,) projection direction, not necessarily normalized.

    Returns
    -------
    - If `proj_dir` is unspecified:

    E: TensorfT
        (2, order+2) 2D box-spline generator.

    - If `proj_dir` is given:

    E: ArrayNfT
        1D box-spline direction vectors :math:`\bbE`.
    E_mask: ArrayNiT
        0/1-mask of non-zero E entries.
    """
    assert order in (0, 1, 2)

    ArrayNf = type(scale)
    assert dr.size_v(scale) == 2
    assert dr.shape(scale)[1] == 1, "scale must be shared across dimensions."
    Float = dr.value_t(ArrayNf)

    if proj_dir is None:
        Tensor = dr.tensor_t(Float)
        E = Tensor(
            [
                [1, 0, 1, 1],
                [0, 1, 1, -1],
            ]
        ) * Tensor(scale)
        return E[:, : order + 2]
    else:
        assert type(proj_dir) is ArrayNf
        Array4f = xrtu.float_array_t(Float, 4)

        n = dr.normalize(proj_dir)
        to_1d = lambda _: dr.abs_dot(n, scale * _)
        E = Array4f(
            to_1d(ArrayNf(+1, +0)),
            to_1d(ArrayNf(+0, +1)),
            to_1d(ArrayNf(+1, +1)),
            to_1d(ArrayNf(+1, -1)),
        )[: order + 2]

        E_mask_t = dr.int_array_t(E)
        E_mask = dr.select(E <= 1e-3, E_mask_t(0), E_mask_t(1))

        return E, E_mask
