"""Bent-ray refraction tomography on the real Koenigsee field dataset.

Data: pyGIMLi example data (gimli-org/example-data, traveltime/koenigsee.sgt) --
63 shot/geophone positions along a ~56 m surface line and 714 PICKED
first-arrival travel times from a real refraction seismic survey.

Physics: sources and receivers are both at the surface, so the first arrival is
a DIVING wave that refracts down through the increasing-velocity subsurface and
turns back up. A straight ray between two surface points would skim the surface
and is meaningless here -- bending is mandatory.

Method: for each shot we fan-shoot rays with XTK's refractive marcher (one
launch angle per lane, thousands of rays in parallel), record where each ray
re-emerges at the surface and its accumulated travel time, then read the
FIRST-ARRIVAL time at each geophone as the minimum over the emerging fan. This
is the fan + min-time construction that recovers first arrivals with a
Lagrangian ray tracer.

Coordinates: x along the line, z = DEPTH (positive downwards). The free surface
is the top of the bounding box, so a ray that returns to z = 0 leaves the box
and `refract_apply(..., return_path=True)` reports its exit point and time.
"""
import pathlib

import numpy as np
import drjit as dr
from drjit.cuda.ad import Array2f, Float

from xrt_toolkit.util import UniformSpec
from xrt_toolkit.drjit.curved_xrt import refract_apply, refract_adjoint

# ---------------------------------------------------------------- model grid
X0, X1 = -6.0, 53.0        # m along the line
ZMAX = 28.0                # m depth (turning depth ~22 m at max offset)
DX = 0.10                  # cell size [m] (fine enough to resolve shallow turning rays)
NX = int(round((X1 - X0) / DX))
NZ = int(round(ZMAX / DX))
DS = DX / 2                # ray marching step [m]

# knot centres so that the bbox starts exactly at x=X0 and z=0
KNOT = UniformSpec(start=(X0 + DX / 2, DX / 2), step=(DX, DX), num=(NX, NZ))

XC = X0 + DX / 2 + np.arange(NX) * DX
ZC = DX / 2 + np.arange(NZ) * DX
XX, ZZ = np.meshgrid(XC, ZC, indexing="ij")

V_LO, V_HI = 300.0, 9000.0     # velocity bounds [m/s]


HERE = pathlib.Path(__file__).resolve().parent


def read_sgt(path=None):
    """pyGIMLi .sgt: N sensor (x,y) then M measurements (s, g, t)."""
    path = path or (HERE / "data" / "koenigsee.sgt")
    lines = [l.strip() for l in open(path) if l.strip()]
    i = 0
    ns = int(lines[i].split("#")[0]);  i += 1
    i += 1                                          # '#x y' header
    pos = np.array([[float(v) for v in lines[i + k].split()[:2]]
                    for k in range(ns)]);  i += ns
    nm = int(lines[i].split("#")[0]);  i += 1
    i += 1                                          # '#s g t' header
    mm = np.array([[float(v) for v in lines[i + k].split()[:3]]
                   for k in range(nm)])
    s = mm[:, 0].astype(int) - 1
    g = mm[:, 1].astype(int) - 1
    t = mm[:, 2]
    return pos, s, g, t


def gradient_model(v0=701.5, k=195.4):
    """Starting model: velocity increasing linearly with depth."""
    return (v0 + k * ZZ).astype(np.float64)


def fan_shoot(u_flat, src_x, n_fan=1200, ang_lo=0.15, ang_hi=89.5, zstart=1e-3):
    """Shoot a fan of rays from one surface source; return (exit_x, T) per ray.

    Rays are launched in both +x and -x directions over `n_fan` angles below
    horizontal. Each lane is one ray; the marcher runs them all in parallel.
    """
    ang = np.deg2rad(np.linspace(ang_lo, ang_hi, n_fan))
    dirs = np.concatenate([np.stack([np.cos(ang), np.sin(ang)], 0),
                           np.stack([-np.cos(ang), np.sin(ang)], 0)], axis=1)
    L = dirs.shape[1]
    tx = np.full(L, src_x, np.float32)
    tz = np.full(L, zstart, np.float32)
    ray = (Array2f(tx, tz),
           Array2f(dirs[0].astype(np.float32).copy(),
                   dirs[1].astype(np.float32).copy()))
    T, xe, _ = refract_apply(ray, KNOT, u_flat, ds=DS,
                             max_steps=int(6 * (X1 - X0) / DS),
                             bend=True, return_path=True)
    dr.eval(T, xe)
    return np.array(xe.x), np.array(xe.y), np.array(T), dirs


def first_arrival(exit_x, exit_z, T, targets, surf_tol=DX):
    """First-arrival time at each target x: minimum time over fan rays that
    re-emerged at the surface near that x (linear blend of the two bracketing
    rays)."""
    ok = (exit_z < surf_tol) & np.isfinite(T)
    ex, tt = exit_x[ok], T[ok]
    out = np.full(len(targets), np.nan)
    if len(ex) < 3:
        return out
    order = np.argsort(ex)
    ex, tt = ex[order], tt[order]
    for i, xt in enumerate(targets):
        j = np.searchsorted(ex, xt)
        lo, hi = max(j - 2, 0), min(j + 2, len(ex))
        w = np.abs(ex[lo:hi] - xt)
        m = w < 1.5
        if m.any():
            out[i] = tt[lo:hi][m].min()
    return out
