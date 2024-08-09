# Helper classes/functions to interact with DrJit

import types

import drjit as dr
import numpy.typing as npt

from xrt_toolkit.array_module import NDArrayInfo

DrJitBackend = types.ModuleType  # (drjit.llvm, drjit.cuda)


def xp2dr(x: npt.ArrayLike):
    """
    Transform NP/CP inputs to format allowing zero-copy casts to {drb.Float, drb.Array[23]f}
    """
    ndi = NDArrayInfo.from_obj(x)
    if ndi == NDArrayInfo.NUMPY:
        return x
    elif ndi == NDArrayInfo.CUPY:
        return x.__dlpack__()
    else:
        raise NotImplementedError


def load_dr_variant(ndi: NDArrayInfo) -> DrJitBackend:
    """
    Load right computation backend.
    """
    if ndi == NDArrayInfo.NUMPY:
        import drjit.llvm as drb
    elif ndi == NDArrayInfo.CUPY:
        import drjit.cuda as drb
    else:
        raise NotImplementedError
    return drb


def Arrayf_Factory(drb: DrJitBackend, D: int):
    """
    Load right Array[23]f class.
    """
    if D == 2:
        Arrayf = drb.Array2f
    elif D == 3:
        Arrayf = drb.Array3f
    else:
        raise NotImplementedError
    return Arrayf


def Arrayu_Factory(drb: DrJitBackend, D: int):
    """
    Load right Array[23]u class.
    """
    if D == 2:
        Arrayu = drb.Array2u
    elif D == 3:
        Arrayu = drb.Array3u
    else:
        raise NotImplementedError
    return Arrayu


def Rayf_Factory(drb: DrJitBackend, D: int):
    """
    Create a Ray[23]f class associated with a compute backend.
    """
    Arrayf = Arrayf_Factory(drb, D)

    class Rayf:
        # Dr.JIT-backed ray.
        #
        # Rays take the parametric form
        #     r(t) = o + d * t,
        # where {o, d} \in \bR^{D} and t \in \bR.
        DRJIT_STRUCT = dict(
            o=Arrayf,
            d=Arrayf,
        )

        def __init__(self, o=Arrayf(), d=Arrayf()):
            # Parameters
            # ----------
            # o: Array3f
            #    (D,) ray origin.
            # d: Array3f
            #    (D,) ray direction.
            #
            # [2023.10.19 Sepand/Miguel]
            # Use C++'s `operator=()` semantics instead of Python's `=` to safely reference inputs.
            self.o = Arrayf()
            self.o.assign(o)

            self.d = Arrayf()
            self.d.assign(d)

        def __call__(self, t: drb.Float) -> Arrayf:
            # Compute r(t).
            #
            # Parameters
            # ----------
            # t: Float
            #
            # Returns
            # -------
            # p: Arrayf
            #    p = r(t)
            return dr.fma(self.d, t, self.o)

        def assign(self, r: "Rayf"):
            # See __init__'s docstring for more info.
            self.o.assign(r.o)
            self.d.assign(r.d)

        def __repr__(self) -> str:
            return f"Rayf(o={dr.shape(self.o)}, d={dr.shape(self.d)})"

    return Rayf


def BoundingBoxf_Factory(drb: DrJitBackend, D: int):
    """
    Create a BoundingBox[23]f class associated with a compute backend.
    """
    Arrayf = Arrayf_Factory(drb, D)
    Rayf = Rayf_Factory(drb, D)

    class BoundingBoxf:
        # Dr.JIT-backed bounding box.
        #
        # A bounding box is described by coordinates {pMin, pMax} of two of its diagonal corners.
        #
        # Important
        # ---------
        # We do NOT check if (pMin < pMax) when constructing the BBox: users are left responsible of their actions.
        DRJIT_STRUCT = dict(
            pMin=Arrayf,
            pMax=Arrayf,
        )

        def __init__(self, pMin=Arrayf(), pMax=Arrayf()):
            # Parameters
            # ----------
            # pMin: Arrayf
            #     (D,) corner coordinate.
            # pMax: Arrayf
            #     (D,) corner coordinate.
            #
            # [2023.10.19 Sepand/Miguel]
            # Use C++'s `operator=()` semantics instead of Python's `=` to safely reference inputs.
            self.pMin = Arrayf()
            self.pMin.assign(pMin)

            self.pMax = Arrayf()
            self.pMax.assign(pMax)

        def contains(self, p: Arrayf) -> drb.Bool:
            # Check if point `p` lies in/on the BBox.
            return dr.all((self.pMin <= p) & (p <= self.pMax))

        def ray_intersect(self, r: Rayf) -> tuple[drb.Bool, drb.Float, drb.Float]:
            # Compute ray/bbox intersection parameters. [Adapted from Mitsuba3's ray_intersect().]
            #
            # Parameters
            # ----------
            # r: Rayf
            #
            # Returns
            # -------
            # active: Bool
            #     True if intersection occurs.
            # mint, maxt: Float
            #     Ray parameters `t` such that r(t) touches a BBox border.
            #     The value only makes sense if `active` is enabled.

            # Ensure ray has a nonzero slope on each axis, or that its origin on a 0-valued axis is within the box
            # bounds.
            active = dr.all(dr.neq(r.d, 0) | (self.pMin < r.o) | (r.o < self.pMax))

            # Compute intersection intervals for each axis
            d_rcp = dr.rcp(r.d)
            t1 = (self.pMin - r.o) * d_rcp
            t2 = (self.pMax - r.o) * d_rcp

            # Ensure proper ordering
            t1p = dr.minimum(t1, t2)
            t2p = dr.maximum(t1, t2)

            # Intersect intervals
            mint = dr.max(t1p)
            maxt = dr.min(t2p)
            active &= mint <= maxt

            return active, mint, maxt

        def assign(self, bbox: "BoundingBoxf"):
            # See __init__'s docstring for more info.
            self.pMin.assign(bbox.pMin)
            self.pMax.assign(bbox.pMax)

        def __repr__(self) -> str:
            return f"BoundingBoxf(pMin={dr.shape(self.pMin)}, pMax={dr.shape(self.pMax)})"

    return BoundingBoxf


def get_ray_step(drb: DrJitBackend, D: int):
    """
    Create ray_step() function associated with a compute backend.
    """
    Rayf = Rayf_Factory(drb, D)
    BoundingBoxf = BoundingBoxf_Factory(drb, D)

    def ray_step(r: Rayf) -> Rayf:
        # Advance ray until next unit-step lattice intersection.
        #
        # Parameters
        #   r(o, d): ray to move. (`d` assumed normalized.)
        # Returns
        #   r_next(o_next, d): next ray position on unit-step lattice intersection.
        eps = 1e-4  # threshold for proximity tests with 0

        # Go to local coordinate system.
        o_ref = dr.floor(r.o)
        r_local = Rayf(o=r.o - o_ref, d=r.d)

        # Find bounding box for ray-intersection tests.
        on_boundary = r_local.o <= eps
        bbox_border = dr.select(on_boundary, dr.sign(r.d), 1)
        bbox = BoundingBoxf(
            dr.minimum(0, bbox_border),
            dr.maximum(0, bbox_border),
        )

        # Compute step size to closest bounding box wall.
        #   (a1, a2) may contain negative values or Infs.
        #   In any case, we must always choose min(a1, a2) > 0.
        _, a1, a2 = bbox.ray_intersect(r_local)
        a_min = dr.minimum(a1, a2)
        a_max = dr.maximum(a1, a2)
        a = dr.select(a_min >= eps, a_min, a_max)

        # Move ray to new position in global coordinates.
        # r_next located on lattice intersection (up to FP error).
        r_next = Rayf(o=o_ref + r_local(a), d=r.d)
        return r_next

    return ray_step
