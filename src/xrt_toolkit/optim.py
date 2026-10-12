r"""
Reconstruction algorithms.

The iterative solvers take the forward and adjoint operators as callables, so they work with any
operator pair, e.g. :py:func:`~xrt_toolkit.xrt_apply` / :py:func:`~xrt_toolkit.xrt_adjoint` or
:py:func:`~xrt_toolkit.xrt_struct_apply` / :py:func:`~xrt_toolkit.xrt_struct_adjoint`:

.. code-block:: python

   A = lambda f: xtk.xrt_apply(ray_spec, knot_spec, order, f)
   At = lambda y: xtk.xrt_adjoint(ray_spec, knot_spec, order, y)
   f = xtk.cg(A, At, y, math.prod(knot_spec.num))
"""

import typing as typ

import drjit as dr
import numpy as np

FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
OperatorT = typ.Callable[[FloatT], FloatT]


def cg(
    A: OperatorT,
    At: OperatorT,
    y: FloatT,
    n_unknowns: int,
    n_iter: int = 30,
    x0: FloatT = None,
) -> FloatT:
    r"""
    Least squares :math:`\min_{f} \|A f - y\|_{2}^{2}`, by conjugate gradient on the normal
    equations :math:`A^{T} A f = A^{T} y`.

    Parameters
    ----------
    A, At: callable
        Forward and adjoint operators.
    y: FloatT
        Measurements.
    n_unknowns: int
        Size of :math:`f`.
    n_iter: int
        Number of iterations. With noisy or incomplete data, stopping early regularizes.
    x0: FloatT
        Initial :math:`f` (default: 0).

    Returns
    -------
    f: FloatT
        (n_unknowns,) solution.
    """
    Float = type(y)
    if x0 is None:
        f = dr.zeros(Float, n_unknowns)
        r = Float(At(y))
    else:
        f = Float(x0)
        r = At(y - A(f))
    p = Float(r)
    rs = dr.sum(r * r)
    for _ in range(n_iter):
        Ap = At(A(p))
        alpha = rs / (dr.sum(p * Ap) + 1e-30)
        f = dr.fma(alpha, p, f)
        r = dr.fma(-alpha, Ap, r)
        rs_new = dr.sum(r * r)
        p = dr.fma(rs_new / (rs + 1e-30), p, r)
        rs = rs_new
        dr.eval(f, r, p, rs)
    return f


def gd(
    A: OperatorT,
    At: OperatorT,
    y: FloatT,
    n_unknowns: int,
    n_iter: int = 100,
    step: float = None,
    x0: FloatT = None,
) -> FloatT:
    r"""
    Least squares :math:`\min_{f} \|A f - y\|_{2}^{2}`, by gradient descent (Landweber
    iteration): :math:`f \leftarrow f - \tau A^{T} (A f - y)`.

    Parameters
    ----------
    A, At: callable
        Forward and adjoint operators.
    y: FloatT
        Measurements.
    n_unknowns: int
        Size of :math:`f`.
    n_iter: int
        Number of iterations.
    step: float
        Step :math:`\tau`. Default: :math:`1.9 / \|A^{T} A\|`, with the norm estimated by 15
        power iterations (convergence needs :math:`\tau < 2 / \|A^{T} A\|`).
    x0: FloatT
        Initial :math:`f` (default: 0).

    Returns
    -------
    f: FloatT
        (n_unknowns,) solution.
    """
    Float = type(y)
    if step is None:
        v = Float(np.random.default_rng(0).random(n_unknowns))
        for _ in range(15):
            v = At(A(v))
            v_norm = dr.norm(v)
            v = v / (v_norm + 1e-30)
            dr.eval(v, v_norm)
        step = 1.9 / v_norm[0]
    f = dr.zeros(Float, n_unknowns) if (x0 is None) else Float(x0)
    for _ in range(n_iter):
        f = dr.fma(-step, At(A(f) - y), f)
        dr.eval(f)
    return f
