"""Cryo-ET of a whole Vibrio cholerae cell: parallel_beam + fbp, nothing else.

Data: CZ CryoET Data Portal dataset 10489, run Vibrio_pilT_pilU_131.
41 tilts, -53.24 .. +66.62 deg, 1023 x 1440, 13.328 A/px.

A single-axis tilt series is a parallel-beam scan, so the whole reconstruction
is two library calls: parallel_beam builds the geometry and fbp filters and
backprojects. On this data it beats CGLS: the background is flat, where 25
CGLS iterations still carry a low-frequency gradient across the field.
"""
import pathlib
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
BIN, NZ, APIX, WINDOW = 2, 200, 13.328, "shepp-logan"


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
    px = APIX * BIN / 10.0                     # nm per binned voxel

    # parallel_beam rotates about the third lattice axis, and its first
    # detector axis spans the second. So the tilt axis goes on axis 3, and the
    # thin specimen direction on axis 1, where the beam points at zero tilt.
    knot = xtk.UniformSpec.centered(step=px, num=(NZ, NU, NV))
    det = xtk.DetectorSpec(size=(NU * px, NV * px), num_cell=(NU, NV))
    rays = xtk.parallel_beam(
        Float(np.ascontiguousarray(np.deg2rad(ang), np.float32)), det)

    y = Float(np.ascontiguousarray(np.transpose(g, (0, 2, 1)).ravel()))
    t0 = time.time()
    vol = np.asarray(xtk.fbp(rays, knot, y, window=WINDOW)).reshape(NZ, NU, NV)
    dr.sync_thread()
    print(f"volume {knot.num}, {px:.3f} nm/voxel, {n_t} tilts")
    print(f"fbp ({WINDOW}): {time.time() - t0:.1f}s")

    np.save(HERE / "results" / "cryo_vibrio_fbp.npy", vol.astype(np.float32))
    z, half = 88, 5
    sl = vol[z - half:z + half].mean(0).T
    lo, hi = np.percentile(sl, 0.5), np.percentile(sl, 99.5)
    fig, ax = plt.subplots(figsize=(6.6, 6.6))
    ax.imshow(sl, cmap="gray_r", vmin=lo, vmax=hi, origin="lower",
              extent=[0, NU * px / 1000, 0, NV * px / 1000])
    ax.set_xlabel("µm")
    ax.set_ylabel("µm")
    ax.set_title(f"fbp, {2 * half * px:.0f} nm slab")
    plt.tight_layout()
    fig.savefig(HERE / "figs" / "fig_cryoet_vibrio.png", dpi=110,
                bbox_inches="tight")
    print("saved figs/fig_cryoet_vibrio.png")


if __name__ == "__main__":
    main()
