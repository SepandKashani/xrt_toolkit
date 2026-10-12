import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)


def tof_window(tof: xrtu.TOFSpec, ray_t: ArrayNfT, ray_n: ArrayNfT) -> tuple:
    r"""
    TOF window along each ray, as functions of world positions :math:`\bbx`.

    Parameters
    ----------
    tof: TOFSpec
    ray_t, ray_n: ArrayNfT
        (L,) ray anchors and directions, as given by the user: the arc length is measured from
        `ray_t` along the unit direction.

    Returns
    -------
    chord: callable
        ``chord(x_a, x_b)``: integral of the window between positions `x_a` and `x_b`.
    density: callable
        ``density(x)``: window density at position `x`.
    """
    Float = dr.value_t(type(ray_t))
    mu = Float(tof.center)
    r_sigma = dr.rcp(Float(tof.sigma) * math.sqrt(2))
    n_hat = dr.normalize(ray_n)

    def z(x: ArrayNfT) -> FloatT:
        return (dr.dot(x - ray_t, n_hat) - mu) * r_sigma

    def chord(x_a: ArrayNfT, x_b: ArrayNfT) -> FloatT:
        return 0.5 * dr.abs(dr.erf(z(x_b)) - dr.erf(z(x_a)))

    def density(x: ArrayNfT) -> FloatT:
        return dr.exp(-dr.square(z(x))) * r_sigma * (1 / math.sqrt(math.pi))

    return chord, density
