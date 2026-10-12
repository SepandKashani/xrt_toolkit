from dataclasses import dataclass


@dataclass
class TOFSpec:
    r"""
    Time-of-flight (TOF) weighting along each ray.

    With TOF, :py:func:`~xrt_toolkit.xrt_apply` computes

    .. math::

       \int_{\bR} f(\bbt + \alpha \hat{\bbn}) \, g_{\sigma}(\alpha - \mu) \, d\alpha,
       \qquad
       g_{\sigma}(\alpha) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{\alpha^{2}}{2\sigma^{2}}\right),

    where :math:`\alpha` is the signed arc length from the ray anchor :math:`\bbt` along the unit
    direction :math:`\hat{\bbn}`. In positron emission tomography, :math:`\mu` is the most likely
    annihilation position along the line of response and :math:`\sigma` the timing resolution of
    the scanner, both in the length unit of `knot_spec`.

    Parameters
    ----------
    center: FloatT | float
        (L,) TOF centers :math:`\mu`, one per ray, or a scalar shared by all rays.
    sigma: FloatT | float
        (L,) TOF standard deviations :math:`\sigma > 0`, one per ray, or a scalar shared by all
        rays.

    Notes
    -----
    Binned TOF data is modelled by repeating each ray once per TOF bin, with that bin's `center`.
    """

    center: object
    sigma: object
