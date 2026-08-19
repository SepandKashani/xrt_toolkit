"""Preprocess the FIPS walnut cone-beam CT dataset (Zenodo 6986012, CC-BY-4.0).

Real measured projections: 721 views over 360 deg, 2368x2240 uint16 raw
transmission images, 0.05 mm detector pixels, SOD 210.66 mm, SDD 553.74 mm.

No flat/dark fields ship with the dataset, so the free-beam intensity I0 is
estimated per detector ROW from the outer columns (the walnut occupies only the
central ~70% of the detector width). Line integrals are -log(I/I0).
"""
import glob
import io
import os
import zipfile

import numpy as np
from PIL import Image

BIN = 8                     # detector binning
STRIDE = 4                  # use every STRIDE-th projection
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ct_walnut_prep.npz")
HERE = os.path.dirname(os.path.abspath(__file__))
ZDIR = os.path.join(HERE, "walnut")


def bin2d(a, b):
    h, w = a.shape
    a = a[: h // b * b, : w // b * b]
    return a.reshape(h // b, b, w // b, b).mean(axis=(1, 3))


def main():
    zips = sorted(glob.glob(os.path.join(ZDIR, "*_projections_*.zip")))
    print(f"{len(zips)} zips")
    # map projection index -> (zip, name)
    entries = {}
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if n.endswith(".tif"):
                    idx = int(n.split("_")[-1].split(".")[0])
                    entries[idx] = (zp, n)
    idxs = sorted(entries)
    print(f"{len(idxs)} projections found, range {idxs[0]}..{idxs[-1]}")
    # angles: AngleFirst=0, AngleInterval=0.5 deg ; index 1 -> 0 deg
    use = idxs[::STRIDE]
    # drop the duplicate 360deg view if both 0 and 360 present
    ang_all = np.array([(i - 1) * 0.5 for i in use])
    keep = ang_all < 359.999
    use = [u for u, k in zip(use, keep) if k]
    angles = ang_all[keep]
    print(f"using {len(use)} projections, angles {angles[0]}..{angles[-1]} deg")

    proj = None
    zcache = {}
    for j, i in enumerate(use):
        zp, n = entries[i]
        if zp not in zcache:
            zcache = {zp: zipfile.ZipFile(zp)}   # one open zip at a time
        a = np.array(Image.open(io.BytesIO(zcache[zp].read(n))), dtype=np.float32)
        ab = bin2d(a, BIN)
        if proj is None:
            proj = np.empty((len(use),) + ab.shape, np.float32)
            print("binned projection shape", ab.shape)
        proj[j] = ab
        if j % 30 == 0:
            print(f"  {j}/{len(use)}", flush=True)

    # I0 per row from the outer columns (free beam), per projection
    ncol = proj.shape[2]
    edge = max(ncol // 20, 4)
    I0 = 0.5 * (proj[:, :, :edge].mean(axis=2) + proj[:, :, -edge:].mean(axis=2))
    I0 = np.maximum(I0, 1.0)[:, :, None]
    g = -np.log(np.clip(proj / I0, 1e-4, None)).astype(np.float32)
    g = np.maximum(g, 0.0)     # transmission cannot exceed the free beam
    print(f"line integrals: min {g.min():.3f} max {g.max():.3f} mean {g.mean():.3f}")

    np.savez_compressed(OUT, g=g, angles=angles.astype(np.float32),
                        binning=BIN, pixel_mm=0.05 * BIN,
                        sod=210.66, sdd=553.74)
    print("saved", OUT, os.path.getsize(OUT) / 1e6, "MB")


if __name__ == "__main__":
    main()
