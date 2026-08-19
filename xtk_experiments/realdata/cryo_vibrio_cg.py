"""Cryo-ET of a whole Vibrio cholerae cell: structured parallel beam, plain CGLS.

Data: CZ CryoET Data Portal dataset 10489, run Vibrio_pilT_pilU_131.
41 tilts, -53.24 .. +66.62 deg, 1023 x 1440, 13.328 A/px.

No filtering anywhere. `parallel_beam` builds the single-axis tilt series, and
CGLS solves the least-squares problem directly. CGLS keeps its residual in data
space, which matters: CG on the normal equations breaks down in single
precision on this problem within a few iterations. CGLS survives to about 25,
which is where it is stopped -- past that the fp32 iteration also degrades, and
early stopping is the usual regulariser for an underdetermined tilt series.
"""
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

import drjit as dr
from drjit.cuda.ad import Float

import xrt_toolkit as xtk

HERE = pathlib.Path(__file__).resolve().parent
D = HERE / "data" / "cryo"
BIN, NZ, APIX, N_ITER = 2, 200, 13.328, 25


def bin2(a, b):
    h, w = a.shape
    a = a[: h // b * b, : w // b * b]
    return a.reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def main():
    ang = np.array([float(x) for x in
                    open(D / "Vibrio_pilT_pilU_131.rawtlt").read().split()])
    with mrcfile.mmap(str(D / "Vibrio_pilT_pilU_131.mrc"),
                      permissive=True, mode="r") as m:
        g = np.stack([-(lambda a: (a - a.mean()) / (a.std() + 1e-9))(
            bin2(np.asarray(m.data[i], np.float32), BIN))
            for i in range(m.data.shape[0])])
    n_t, NV, NU = g.shape
    px = APIX * BIN / 10.0                       # nm per binned voxel

    # parallel_beam rotates about the third lattice axis; u1 spans the second
    # axis and u2 the third. So put the tilt axis on axis 3, and the thin
    # specimen direction on axis 1, where the beam points at zero tilt.
    knot = xtk.UniformSpec.centered(step=px, num=(NZ, NU, NV))
    det = xtk.DetectorSpec(size=(NU * px, NV * px), num_cell=(NU, NV))
    rays = xtk.parallel_beam(
        Float(np.ascontiguousarray(np.deg2rad(ang), np.float32)), det)
    re = xtk.struct_rays(rays)
    print(f"volume {knot.num}, {px:.3f} nm/voxel, {dr.width(re[0])/1e6:.1f} M rays")

    y = Float(np.ascontiguousarray(np.transpose(g, (0, 2, 1)).ravel()))
    A = lambda f: xtk.xrt_apply(re, knot, 0, f)
    At = lambda d: xtk.xrt_adjoint(re, knot, 0, d)

    t0 = time.time()
    x = dr.zeros(Float, int(np.prod(knot.num)))
    r = Float(y)
    s = At(r)
    p = Float(s)
    gam = dr.sum(s * s)
    dr.eval(x, r, p, gam)
    for _ in range(N_ITER):
        q = A(p)
        al = gam / (dr.sum(q * q) + 1e-30)
        x = dr.fma(al, p, x)
        r = dr.fma(-al, q, r)
        s = At(r)
        gnew = dr.sum(s * s)
        p = dr.fma(gnew / gam, p, s)
        gam = gnew
        dr.eval(x, r, p, gam)
    dr.sync_thread()
    print(f"CGLS, {N_ITER} iterations: {time.time() - t0:.1f}s")

    vol = np.asarray(x).reshape(NZ, NU, NV)
    np.save(HERE / "results" / "cryo_vibrio_cgls.npy", vol.astype(np.float32))

    z, half = 88, 5
    sl = vol[z - half:z + half].mean(0).T
    lo, hi = np.percentile(sl, 0.5), np.percentile(sl, 99.5)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.imshow(sl, cmap="gray_r", vmin=lo, vmax=hi, origin="lower",
              extent=[0, NU * px / 1000, 0, NV * px / 1000])
    ax.set_xlabel("µm")
    ax.set_ylabel("µm")
    ax.set_title(f"CGLS, {N_ITER} iterations, {2 * half * px:.0f} nm slab")
    plt.tight_layout()
    fig.savefig(HERE / "figs" / "fig_cryoet_vibrio.png", dpi=110,
                bbox_inches="tight")
    print("saved figs/fig_cryoet_vibrio.png")


if __name__ == "__main__":
    main()
