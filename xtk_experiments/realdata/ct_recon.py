"""Cone-beam CT reconstruction of REAL walnut data (FIPS, Zenodo 6986012) with XTK.

Every ray is built explicitly (one per detector pixel per view), so the
reconstruction uses the arbitrary-ray path of the library rather than a
structured-geometry helper. Solver: conjugate gradient on the normal equations.
"""
import os
import time

import numpy as np
import drjit as dr
from drjit.cuda.ad import Array3f, Float

from xrt_toolkit.util import UniformSpec
from xrt_toolkit.drjit.ray_xrt import xrt_apply, xrt_adjoint

ORDER = 0
NVOX = 192
ITERS = 30
LAM = 1e-2


def build_rays(nrow, ncol, pixel, sod, sdd, angles_deg, sense=-1.0):
    """Explicit cone-beam rays: source position and unit direction per pixel.

    `sense` sets the rotation handedness. In this scanner the SAMPLE rotates on
    the stage while source and detector stay fixed, so in the sample frame the
    source travels the OPPOSITE way: sense = -1. Getting this wrong is not a
    harmless mirror -- negating the angle is equivalent to mirroring the object
    AND flipping the detector column axis, so the data can tell the two apart
    (held-out view RMSE 0.046 with sense=-1 vs 0.070 with +1).
    """
    a = sense * np.deg2rad(angles_deg).astype(np.float64)
    r = np.arange(nrow) - (nrow - 1) / 2
    c = np.arange(ncol) - (ncol - 1) / 2
    u2 = (-r * pixel)[:, None]                    # vertical (rotation axis, +z up)
    u1 = (c * pixel)[None, :]                     # horizontal (fan direction)
    u1 = np.broadcast_to(u1, (nrow, ncol)).ravel()
    u2 = np.broadcast_to(u2, (nrow, ncol)).ravel()

    ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
    # source: R(a) @ (-sod, 0, 0)
    tx = np.broadcast_to(-sod * ca, (len(a), u1.size)).ravel()
    ty = np.broadcast_to(-sod * sa, (len(a), u1.size)).ravel()
    tz = np.zeros_like(tx)
    # direction: R(a) @ (sdd, u1, u2), normalised to unit length
    nx = (sdd * ca - u1[None, :] * sa).ravel()
    ny = (sdd * sa + u1[None, :] * ca).ravel()
    nz = np.broadcast_to(u2[None, :], (len(a), u1.size)).ravel()
    nn = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = nx / nn, ny / nn, nz / nn
    t = Array3f(*(v.astype(np.float32) for v in (tx, ty, tz)))
    n = Array3f(*(v.astype(np.float32) for v in (nx, ny, nz)))
    return t, n


HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    d = np.load(os.path.join(HERE, "data", "ct_walnut_prep.npz"))
    g, angles = d["g"], d["angles"]
    pixel, sod, sdd = float(d["pixel_mm"]), float(d["sod"]), float(d["sdd"])
    nviews, nrow, ncol = g.shape
    print(f"data {g.shape}, pixel {pixel} mm, SOD {sod}, SDD {sdd}")

    M = sdd / sod
    fov = ncol * pixel / M
    voxel = fov / NVOX
    print(f"magnification {M:.4f}, isocenter FOV {fov:.2f} mm, voxel {voxel:.4f} mm")

    knot = UniformSpec.centered(step=voxel, num=(NVOX,) * 3)
    t0 = time.time()
    t, n = build_rays(nrow, ncol, pixel, sod, sdd, angles)
    L = dr.width(t)
    print(f"{L/1e6:.1f} M rays built in {time.time()-t0:.1f}s")

    b_data = Float(g.ravel().astype(np.float32))
    ray = (t, n)

    # XTK's basis has unit integral, so xrt_apply returns chord/prod(step).
    # Multiplying by prod(step) makes the operator return true arc-length line
    # integrals [mm], hence the unknown is mu in 1/mm.
    S = float(np.prod(knot.step))

    # Cylindrical support: the circular trajectory only constrains the cylinder
    # inscribed in the detector FOV. Leaving the corners free makes the system
    # underdetermined and CG diverges into that null space.
    ax = (np.arange(NVOX) - (NVOX - 1) / 2) * voxel
    XX, YY = np.meshgrid(ax, ax, indexing="ij")
    rad = np.sqrt(XX**2 + YY**2) < (0.98 * fov / 2)
    mask3 = np.broadcast_to(rad[:, :, None], (NVOX,) * 3).astype(np.float32)
    print(f"support: {mask3.sum()/1e6:.1f} M voxels vs {L/1e6:.1f} M measurements")
    Mask = Float(mask3.ravel().copy())

    def A(x):
        return S * xrt_apply(ray, knot, ORDER, x)

    def At(y):
        return S * xrt_adjoint(ray, knot, ORDER, y)

    dr.sync_thread(); tt = time.time()
    probe = A(Float(np.ones(NVOX**3, np.float32))); dr.eval(probe); dr.sync_thread()
    print(f"one forward: {(time.time()-tt)*1e3:.0f} ms "
          f"(mean chord {float(dr.mean(probe)[0]):.2f} mm)")

    # CGLS with Tikhonov: min ||A x - b||^2 + lam ||x||^2 on the masked support.
    # CGLS applies A and A^T separately and is far better conditioned in fp32
    # than explicitly forming the normal equations.
    t0 = time.time()
    x = dr.zeros(Float, NVOX**3)
    rr = Float(b_data)                       # residual b - A x
    s = At(rr) * Mask
    p = Float(s)
    gam = float(dr.sum(s * s)[0])
    gam0 = gam
    best = (np.inf, None)
    for k in range(ITERS):
        q = A(p)
        denom = float(dr.sum(q * q)[0]) + LAM * float(dr.sum(p * p)[0]) + 1e-30
        alpha = gam / denom
        x = x + alpha * p
        rr = rr - alpha * q
        s = (At(rr) - LAM * x) * Mask
        gam_new = float(dr.sum(s * s)[0])
        p = s + (gam_new / (gam + 1e-30)) * p
        gam = gam_new
        dr.eval(x, rr, p, s)
        dm = float(dr.sqrt(dr.mean(rr * rr))[0])
        if dm < best[0]:
            best = (dm, np.array(x))
        if k % 5 == 0 or k == ITERS - 1:
            print(f"  CGLS {k:3d}  ||A^T r||/||A^T r0|| = {np.sqrt(gam/gam0):.4e}   "
                  f"data RMSE = {dm:.4e}", flush=True)
    dr.sync_thread()
    print(f"CGLS {ITERS} iters in {time.time()-t0:.1f}s; best data RMSE {best[0]:.4e}")
    x = Float(best[1])

    vol = np.array(x).reshape(NVOX, NVOX, NVOX)
    print(f"mu: min {vol.min():.4f} max {vol.max():.4f} (1/mm)")
    np.save(os.path.join(HERE, "results", "ct_walnut_vol.npy"), vol.astype(np.float32))
    np.savez(os.path.join(HERE, "results", "ct_walnut_meta.npz"), voxel=voxel, order=ORDER, iters=ITERS,
             nviews=nviews, L=L)
    print("saved ct_walnut_vol.npy")


if __name__ == "__main__":
    main()
