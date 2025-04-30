import typing as typ

import drjit as dr

BoolT = typ.TypeVar("BoolT", bound=dr.AnyArray)
ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)


def bbox_contains(bbox_ll: ArrayNfT, bbox_ur: ArrayNfT, x: ArrayNfT) -> BoolT:
    """
    Is `x` inside a bbox?

    Parameters
    ----------
    bbox_[ll,ur]: ArrayNf
        (D,) bbox (lower-left, upper-right) coordinate.
    x: ArrayNf
        (D,) point.

    Returns
    -------
    active: Bool
        True if within bbox.
    """
    active = dr.all((bbox_ll <= x) & (x <= bbox_ur))
    return active


def ray_bbox_intersect(
    bbox_ll: ArrayNfT,
    bbox_ur: ArrayNfT,
    ray_o: ArrayNfT,
    ray_d: ArrayNfT,
) -> tuple[BoolT, FloatT, FloatT]:
    """
    Compute ray/bbox intersection parameters.

    Parameters
    ----------
    bbox_[ll,ur]: ArrayNf
        (D,) bbox (lower-left, upper-right) coordinate.
    ray_[o, d]: ArrayNf
        (D,) ray (anchor, direction).

    Returns
    -------
    active: Bool
        True if intersection occurs.
    mint, maxt: Float
        Scale factor ``t`` such that ``ray_o + t * ray_d`` intersects the bbox.
        The value only makes sense if `active` is enabled.
    """

    # Ensure
    # - ray has a nonzero slope on each axis; or
    # - ray origin on a 0-valued axis is within the bbox bounds.
    active = dr.all((ray_d != 0) | (bbox_ll < ray_o) | (ray_o < bbox_ur))

    # Compute intersection intervals for each axis
    d_rcp = dr.rcp(ray_d)
    t1 = (bbox_ll - ray_o) * d_rcp
    t2 = (bbox_ur - ray_o) * d_rcp

    # Ensure proper ordering
    t1p = dr.minimum(t1, t2)
    t2p = dr.maximum(t1, t2)

    # Intersect intervals
    mint = dr.max(t1p)
    maxt = dr.min(t2p)
    active &= mint <= maxt

    return active, mint, maxt
