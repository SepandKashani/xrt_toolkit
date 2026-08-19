from dataclasses import dataclass


@dataclass
class TOFSpec:
    r"""
    Time-of-flight (TOF) kernel specifier.

    Restricts the line integral of :py:func:`~xrt_toolkit.drjit.xrt_apply` to a
    Gaussian window along each ray:

    .. math::

       \xrt_{\text{TOF}}[f](\bbn, \bbt)
       =
       \int_{\bR} f(\bbt + \alpha \hat{\bbn}) \,
       g_{\sigma}(\alpha - \mu) \, \mathrm{d}\alpha,
       \qquad
       g_{\sigma}(\alpha) = \frac{1}{\sigma\sqrt{2\pi}}
       \exp\!\left(-\frac{\alpha^{2}}{2\sigma^{2}}\right),

    where :math:`\alpha` is the (signed) arc length measured from the ray
    anchor :math:`\bbt` along the unit direction :math:`\hat{\bbn}`.
    In positron emission tomography, :math:`\mu` encodes the most likely
    annihilation position along the line of response and :math:`\sigma` the
    timing resolution of the scanner (both in the length unit of `knot_spec`).

    Parameters
    ----------
    center: FloatT
        (L,) TOF centers :math:`\mu` (one per ray), or a scalar broadcast to
        all rays.
    sigma: FloatT | float
        (L,) TOF standard deviations :math:`\sigma > 0`, or a scalar broadcast
        to all rays.

    Notes
    -----
    * ``TOFSpec(center=0, sigma=inf)`` recovers the plain x-ray transform up
      to the vanishing weight; pass ``tof=None`` instead to disable TOF.
    * Binned-TOF sinograms are obtained by replicating each ray with one
      `center` per TOF bin.
    """

    center: object
    sigma: object
