import math
import typing as typ

import drjit as dr
import numpy as np

import xrt_toolkit.util as xrtu

ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)


WeightT = typ.Union[xrtu.TOFSpec, xrtu.PolynomialWeightSpec, xrtu.TabulatedWeightSpec]


def ray_weight(weight: WeightT, ray_t: ArrayNfT, ray_n: ArrayNfT) -> tuple:
    r"""
    Weight along each ray, as functions of world positions :math:`\bbx`.

    Parameters
    ----------
    weight: TOFSpec | PolynomialWeightSpec | TabulatedWeightSpec
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
    n_hat = dr.normalize(ray_n)

    def alpha(x: ArrayNfT) -> FloatT:  # arc length of position x
        return dr.dot(x - ray_t, n_hat)

    if isinstance(weight, xrtu.TOFSpec):
        chord, density = _tof(weight, type(ray_t))
    elif isinstance(weight, xrtu.PolynomialWeightSpec):
        chord, density = _polynomial(weight, type(ray_t))
    elif isinstance(weight, xrtu.TabulatedWeightSpec):
        chord, density = _tabulated(weight, type(ray_t))
    else:
        raise TypeError(f"Unsupported weight: {type(weight).__name__}.")

    return (
        lambda x_a, x_b: chord(alpha(x_a), alpha(x_b)),
        lambda x: density(alpha(x)),
    )


# Each weight below is defined on arc lengths: chord(a, b) integrates it over [a, b] (a <= b),
# density(a) evaluates it.


def _tof(weight: xrtu.TOFSpec, ArrayNf) -> tuple:
    Float = dr.value_t(ArrayNf)
    mu = Float(weight.center)
    r_sigma = dr.rcp(Float(weight.sigma) * math.sqrt(2))

    def chord(a: FloatT, b: FloatT) -> FloatT:
        return 0.5 * (dr.erf((b - mu) * r_sigma) - dr.erf((a - mu) * r_sigma))

    def density(a: FloatT) -> FloatT:
        return (
            dr.exp(-dr.square((a - mu) * r_sigma)) * r_sigma * (1 / math.sqrt(math.pi))
        )

    return chord, density


def _polynomial(weight: xrtu.PolynomialWeightSpec, ArrayNf) -> tuple:
    Float = dr.value_t(ArrayNf)
    coeff = [Float(c) for c in weight.coeff]
    origin = Float(weight.origin)
    # Gauss-Legendre on the chord, exact for the polynomial's degree (no cancellation)
    x, w = np.polynomial.legendre.leggauss(max(1, math.ceil(len(coeff) / 2)))

    def density(a: FloatT) -> FloatT:
        u = a - origin
        y = Float(0)
        for c in reversed(coeff):  # Horner
            y = dr.fma(y, u, c)
        return y

    def chord(a: FloatT, b: FloatT) -> FloatT:
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        return half * sum(
            float(_w) * density(dr.fma(half, float(_x), mid)) for _x, _w in zip(x, w)
        )

    return chord, density


def _tabulated(weight: xrtu.TabulatedWeightSpec, ArrayNf) -> tuple:
    Float = dr.value_t(ArrayNf)
    UInt = dr.uint32_array_t(Float)
    Float64 = dr.float64_array_t(Float)

    num = int(weight.num)
    assert num >= 2
    values = Float(weight.values)
    start, step = Float(weight.start), Float(weight.step)
    L = dr.width(values) // num
    assert dr.width(values) == L * num
    row = dr.arange(UInt, L) * num  # first sample of each ray

    # cumulative integral of the interpolant at the samples, per ray (float64 prefix sum)
    k = dr.arange(UInt, L * num)
    v_next = dr.gather(Float, values, dr.minimum(k + 1, L * num - 1))
    step_k = step if dr.width(step) == 1 else dr.gather(Float, step, k // num)
    area = dr.select(k % num == num - 1, 0, 0.5 * (values + v_next) * step_k)
    P = dr.prefix_sum(Float64(area))  # exclusive
    C = Float(P - dr.gather(Float64, P, (k // num) * num))

    def locate(a: FloatT) -> tuple:
        s = (a - start) * dr.rcp(step)
        j = UInt(dr.clip(dr.floor(s), 0, num - 2))
        f = s - Float(j)
        v0 = dr.gather(Float, values, row + j)
        v1 = dr.gather(Float, values, row + j + 1)
        return j, f, v0, v1

    def density(a: FloatT) -> FloatT:
        _, f, v0, v1 = locate(a)
        return dr.lerp(v0, v1, dr.clip(f, 0, 1))

    # integral from sample 0 to a; the weight is constant beyond the ends
    def integral(a: FloatT) -> FloatT:
        j, f, v0, v1 = locate(a)
        fc = dr.clip(f, 0, 1)
        inside = fc * v0 + 0.5 * dr.square(fc) * (v1 - v0)
        outside = dr.minimum(f, 0) * v0 + dr.maximum(f - 1, 0) * v1
        return dr.gather(Float, C, row + j) + step * (inside + outside)

    def chord(a: FloatT, b: FloatT) -> FloatT:
        return integral(b) - integral(a)

    return chord, density
