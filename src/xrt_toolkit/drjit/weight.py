import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)


def ray_weight(weight: xrtu.TOFSpec, ray_t: ArrayNfT, ray_n: ArrayNfT) -> tuple:
    r"""
    Weight along each ray, as functions of world positions :math:`\bbx`.

    Parameters
    ----------
    weight: TOFSpec
    ray_t, ray_n: ArrayNfT
        (L,) ray anchors and directions, as given by the user: the arc length is measured from
        `ray_t` along the unit direction.

    Returns
    -------
    chord: callable
        ``chord(x_a, x_b)``: integral of the weight between positions `x_a` and `x_b`, with
        `x_b` after `x_a` along the ray.
    density: callable
        ``density(x)``: weight at position `x`.
    """
    Float = dr.value_t(type(ray_t))
    mu = Float(weight.center)
    r_sigma = dr.rcp(Float(weight.sigma) * math.sqrt(2))
    n_hat = dr.normalize(ray_n)

    def z(x: ArrayNfT) -> FloatT:
        return (dr.dot(x - ray_t, n_hat) - mu) * r_sigma

    def chord(x_a: ArrayNfT, x_b: ArrayNfT) -> FloatT:
        return 0.5 * dr.abs(dr.erf(z(x_b)) - dr.erf(z(x_a)))

    def density(x: ArrayNfT) -> FloatT:
        return dr.exp(-dr.square(z(x))) * r_sigma * (1 / math.sqrt(math.pi))

    return chord, density
