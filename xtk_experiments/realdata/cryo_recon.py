"""Cryo-electron tomography reconstruction of a REAL tilt series with XTK.

Data: EMPIAR-10045 (Bharat & Scheres, RELION subtomogram-averaging tutorial),
tomogram 11, S. cerevisiae 80S ribosomes. The deposited `.mrcs` stack is
ALREADY ALIGNED (fiducial alignment applied by IMOD `newstack`), and the
refined tilt angles ship as `.tlt`, so no alignment or CTF work is done here --
only the tomographic reconstruction.

Geometry: single-axis tilt, parallel beam. For tilt angle t about the y axis
the beam direction in specimen coordinates is n = (sin t, 0, cos t) and the
detector axes are (cos t, 0, -sin t) (foreshortening direction) and (0, 1, 0)
(the tilt axis). The angular range is -58 deg to +29 deg, so the reconstruction
carries the usual MISSING WEDGE: elongation along the beam axis is expected and
is a property of the acquisition, not of the solver.
"""
import pathlib
import time

import numpy as np
import mrcfile
import drjit as dr
from drjit.cuda.ad import Array3f, Float

from xrt_toolkit.util import UniformSpec
from xrt_toolkit.drjit.ray_xrt import xrt_apply, xrt_adjoint

HERE = pathlib.Path(__file__).resolve().parent
# 883 MB aligned stack, downloaded on demand (see README)
STACK = str(HERE / "data" / "IS002_291013_011.mrcs")
TLT = str(HERE / "data" / "IS002_291013_011.tlt")
BIN = 8
CROP = 320          # binned pixels kept in each direction
NZ = 96             # reconstructed thickness in binned voxels
ITERS = 12
APIX = 2.2764       # A/px unbinned


def bin2(a, b):
    h, w = a.shape
    a = a[: h // b * b, : w // b * b]
    return a.reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def main():
    ang = np.array([float(x) for x in open(TLT).read().split()])
    with mrcfile.mmap(STACK, permissive=True, mode="r") as m:
        raw = m.data
        n_t = raw.shape[0]
        print(f"stack {raw.shape} {raw.dtype}, {n_t} tilts "
              f"{ang.min():.1f}..{ang.max():.1f} deg")
        imgs = []
        for i in range(n_t):
            a = bin2(np.asarray(raw[i], dtype=np.float32), BIN)
            h, w = a.shape
            a = a[h // 2 - CROP // 2: h // 2 + CROP // 2,
                  w // 2 - CROP // 2: w // 2 + CROP // 2]
            a = (a - a.mean()) / (a.std() + 1e-9)
            imgs.append(-a)          # dense material attenuates -> positive
        g = np.stack(imgs)
    print(f"binned/cropped projections {g.shape}, pixel {APIX*BIN/10:.2f} nm")

    px = APIX * BIN / 10.0            # nm per binned pixel
    knot = UniformSpec.centered(step=px, num=(CROP, CROP, NZ))

    # explicit parallel-beam rays, one per (tilt, detector pixel)
    uu = (np.arange(CROP) - (CROP - 1) / 2) * px
    vv = (np.arange(CROP) - (CROP - 1) / 2) * px
    U, V = np.meshgrid(uu, vv, indexing="ij")     # U along image x, V along y
    U, V = U.ravel(), V.ravel()
    a = np.deg2rad(ang)
    tx, ty, tz, nx, ny, nz = [], [], [], [], [], []
    for th in a:
        c, s = np.cos(th), np.sin(th)
        tx.append(U * c);      ty.append(V);            tz.append(-U * s)
        nx.append(np.full(U.size, s)); ny.append(np.zeros(U.size))
        nz.append(np.full(U.size, c))
    cat = lambda L: np.concatenate(L).astype(np.float32)
    t = Array3f(cat(tx), cat(ty), cat(tz))
    n = Array3f(cat(nx), cat(ny), cat(nz))
    L = dr.width(t)
    print(f"{L/1e6:.2f} M rays, volume {CROP}x{CROP}x{NZ}")

    S = float(np.prod(knot.step))
    A = lambda x: S * xrt_apply((t, n), knot, 0, x)
    At = lambda y: S * xrt_adjoint((t, n), knot, 0, y)

    b = Float(g.ravel().astype(np.float32))
    t0 = time.time()
    x = dr.zeros(Float, CROP * CROP * NZ)
    rr = Float(b)
    s_ = At(rr)
    p = Float(s_)
    gam = float(dr.sum(s_ * s_)[0]); gam0 = gam
    for k in range(ITERS):
        q = A(p)
        alpha = gam / (float(dr.sum(q * q)[0]) + 1e-30)
        x = x + alpha * p
        rr = rr - alpha * q
        s_ = At(rr)
        gn = float(dr.sum(s_ * s_)[0])
        p = s_ + (gn / (gam + 1e-30)) * p
        gam = gn
        dr.eval(x, rr, p, s_)
        if k % 4 == 0 or k == ITERS - 1:
            print(f"  CGLS {k:2d}  rel {np.sqrt(gam/gam0):.4e}  "
                  f"data RMSE {float(dr.sqrt(dr.mean(rr*rr))[0]):.4f}", flush=True)
    dr.sync_thread()
    print(f"CGLS in {time.time()-t0:.1f}s")
    vol = np.array(x).reshape(CROP, CROP, NZ)
    np.save(HERE / "results" / "cryo_vol.npy", vol.astype(np.float32))
    print("saved cryo_vol.npy", vol.shape, vol.min(), vol.max())


if __name__ == "__main__":
    main()
