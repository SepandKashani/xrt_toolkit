"""Cryo-ET reconstruction of a REAL whole-bacterium tilt series with XTK.

Data: CZ CryoET Data Portal dataset 10489, run Vibrio_pilT_pilU_131 --
whole *Vibrio cholerae* cells with a sheathed flagellum (MotorBench; Owens,
Webb, Jensen, Kaplan & Hart, doi:10.1101/2025.04.23.650258).
  https://cryoetdataportal.czscience.com/datasets/10489
  files.cryoetdataportal.cziscience.com/10489/Vibrio_pilT_pilU_131/...

1023 x 1440 x 41, float32, 13.328 A/px, tilts -53.24 deg .. +66.62 deg (~3 deg,
dose-symmetric). The deposited stack is ALREADY ALIGNED -- the accompanying
`.xf` is NOT identity but its transform is already baked into the pixels, so
applying it again would corrupt the reconstruction. Only the `.tlt` is used.
The portal also ships a reference key photo of the depositors' own tomogram,
which is what this reconstruction is compared against.

Geometry: single-axis tilt, parallel beam, tilt axis VERTICAL (image y).
Reconstructed thickness 400 unbinned px, matching the deposited tomogram.

Solver: WEIGHTED BACKPROJECTION, the standard cryo-ET reconstruction. Each tilt
image is ramp-filtered along the direction perpendicular to the tilt axis before
a single adjoint (backprojection) pass. The ramp is essential: an unfiltered
backprojection -- and equally a few iterations of CGLS on this heavily
underdetermined problem -- returns a low-frequency blur in which the granules
and flagellum are invisible.
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
SCRATCH = pathlib.Path("/tmp/claude-275574/-scratch-haouchat/"
                       "9cfc19ba-5b2c-4521-91ff-f93b41e97041/scratchpad/cryo2")
STACK, TLT = SCRATCH / "vibrio_ts.mrc", SCRATCH / "vibrio.tlt"
BIN = 2
NZ = 200                      # 400 unbinned px, as in the deposited tomogram
RADIAL = (0.35, 0.035)        # ramp cutoff / Gaussian falloff (IMOD `tilt` default)
APIX = 13.328                 # A per unbinned pixel


def bin2(a, b):
    h, w = a.shape
    a = a[: h // b * b, : w // b * b]
    return a.reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def main():
    ang = np.array([float(x) for x in open(TLT).read().split()])
    with mrcfile.mmap(str(STACK), permissive=True, mode="r") as m:
        raw = m.data
        print(f"stack {raw.shape} {raw.dtype}, {len(ang)} tilts "
              f"{ang.min():.2f}..{ang.max():.2f} deg")
        imgs = []
        for i in range(raw.shape[0]):
            a = bin2(np.asarray(raw[i], dtype=np.float32), BIN)
            imgs.append(-(a - a.mean()) / (a.std() + 1e-9))   # dense -> positive
        g = np.stack(imgs)
    n_t, NV, NU = g.shape            # V = image y (tilt axis), U = image x
    px = APIX * BIN / 10.0           # nm per binned voxel
    print(f"binned projections {g.shape}, {px:.2f} nm/voxel, "
          f"field {NU*px/1000:.2f} x {NV*px/1000:.2f} um, thickness {NZ*px:.0f} nm")

    knot = UniformSpec.centered(step=px, num=(NU, NV, NZ))
    S = float(np.prod(knot.step))
    uu = (np.arange(NU) - (NU - 1) / 2) * px
    vv = (np.arange(NV) - (NV - 1) / 2) * px
    U, V = np.meshgrid(uu, vv, indexing="ij")     # (NU,NV) matches volume axes
    U, V = U.ravel(), V.ravel()

    tx, ty, tz, nx, ny, nz = [], [], [], [], [], []
    for th in np.deg2rad(ang):
        c, s = np.cos(th), np.sin(th)
        tx.append(U * c); ty.append(V); tz.append(-U * s)
        nx.append(np.full(U.size, s)); ny.append(np.zeros(U.size))
        nz.append(np.full(U.size, c))
    cat = lambda L: np.concatenate(L).astype(np.float32)
    ray = (Array3f(cat(tx), cat(ty), cat(tz)), Array3f(cat(nx), cat(ny), cat(nz)))
    print(f"{dr.width(ray[0])/1e6:.2f} M rays, volume {NU}x{NV}x{NZ}")

    # ---- ramp (R-weighted) filter along U, then one backprojection pass ----
    gf = np.transpose(g, (0, 2, 1)).copy()            # (tilt, U, V)
    fk = np.fft.fftfreq(NU)                           # cycles per binned pixel
    r = np.abs(fk) / 0.5                              # fraction of Nyquist
    cut, fall = RADIAL
    win = np.where(r <= cut, 1.0, np.exp(-0.5 * ((r - cut) / fall) ** 2))
    filt = (np.abs(fk) * win).astype(np.float32)
    G = np.fft.fft(gf, axis=1) * filt[None, :, None]
    gf = np.real(np.fft.ifft(G, axis=1)).astype(np.float32)
    print(f"ramp filter applied (cutoff {cut} Nyq, falloff {fall})")

    t0 = time.time()
    b = Float(gf.ravel())
    vol = np.array(S * xrt_adjoint(ray, knot, 0, b)).reshape(NU, NV, NZ)
    dr.sync_thread()
    print(f"weighted backprojection in {time.time()-t0:.1f}s")
    np.save(HERE / "results" / "cryo_vibrio_vol.npy", vol.astype(np.float32))
    np.savez(HERE / "results" / "cryo_vibrio_meta.npz", px_nm=px, ang=ang,
             bin=BIN, apix=APIX)
    print("saved results/cryo_vibrio_vol.npy", vol.shape)


if __name__ == "__main__":
    main()
