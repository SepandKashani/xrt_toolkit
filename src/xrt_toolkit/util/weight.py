from dataclasses import dataclass


@dataclass
class PolynomialWeightSpec:
    r"""
    Polynomial weight along each ray,

    .. math::

       w(\alpha) = \sum_{k=0}^{K-1} c_{k} (\alpha - \alpha_{0})^{k},

    where :math:`\alpha` is the signed arc length from the ray anchor along the unit direction.
    Pass it as `weight` of :py:func:`~xrt_toolkit.xrt_apply` and :py:func:`~xrt_toolkit.xrt_adjoint`.

    Parameters
    ----------
    coeff: list[FloatT | float]
        Coefficients :math:`c_{0}, \ldots, c_{K-1}`, in increasing degree. Each is a scalar shared
        by all rays, or (L,) one per ray.
    origin: FloatT | float
        Expansion point :math:`\alpha_{0}`, scalar or (L,).

    Notes
    -----
    * Order 0 is exact.
    * 3D orders 1, 2 are exact for degrees up to 8 (order 1) and 5 (order 2).
    * 2D orders 1, 2 weight each basis function by :math:`w` at its center (an approximation).

    Examples
    --------
    Multiple-scattering moments of a ray of length :math:`H`, anchored at its entry point
    (muon scattering tomography):

    >>> m0 = PolynomialWeightSpec([1.0])  # int f
    >>> m1 = PolynomialWeightSpec([H, -1.0])  # int f (H - alpha)
    >>> m2 = PolynomialWeightSpec([H * H, -2.0 * H, 1.0])  # int f (H - alpha)^2
    """

    coeff: list
    origin: object = 0.0


@dataclass
class TabulatedWeightSpec:
    r"""
    Weight along each ray, tabulated on a uniform grid of arc lengths and interpolated linearly.

    Sample :math:`j` of ray :math:`l` sits at arc length :math:`\text{start}_{l} + j \,
    \text{step}_{l}` from the ray anchor. Beyond the ends of the table, the weight is constant.
    Pass it as `weight` of :py:func:`~xrt_toolkit.xrt_apply` and :py:func:`~xrt_toolkit.xrt_adjoint`.

    Use it when the weight has no closed form, e.g. the attenuation factor
    :math:`\exp(-\int_{\alpha}^{\infty} \mu)` of an attenuated transform with diverging rays.

    Parameters
    ----------
    values: FloatT
        (L * num,) weights, ray-major: ray :math:`l` holds ``values[l * num : (l + 1) * num]``.
    num: int
        Samples per ray, at least 2.
    start: FloatT | float
        Arc length of sample 0, scalar or (L,).
    step: FloatT | float
        Sample spacing > 0, scalar or (L,).

    Notes
    -----
    * Order 0 is exact: the interpolant is integrated over each voxel chord.
    * 3D orders 1, 2 integrate it with the basis functions by Gauss quadrature, which does not
      see the kinks of the interpolant: accurate when the table is smooth over a knot step
      (relative error 2e-4 for random values every 4 knot steps, 2e-2 every quarter step).
    * 2D orders 1, 2 weight each basis function by the interpolant at its center (an
      approximation).
    """

    values: object
    num: int
    start: object = 0.0
    step: object = 1.0
