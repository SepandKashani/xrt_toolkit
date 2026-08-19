"""Cryo-ET of a whole Vibrio cholerae cell, reconstructed by conjugate gradients.

Plain CG on A^T A returns a low-frequency blur: the problem is badly
underdetermined (41 tilts over a 120 degree range) and A^T A behaves like
1/|k|. Weighting the normal equations by the ramp filter W fixes exactly that,
because A^T W A is close to the identity for tomography. That is why one
backprojection of ramp-filtered data already looks right, and why CG on the
weighted equations converges in a few iterations instead of hundreds.
"""
import sys, time, pathlib; sys.path.insert(0, "src")
import numpy as np, mrcfile, drjit as dr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drjit.cuda.ad import Array3f, Float
from xrt_toolkit.util import UniformSpec
from xrt_toolkit.drjit.ray_xrt import xrt_apply, xrt_adjoint

D = pathlib.Path("xtk_experiments/realdata/data/cryo")
BIN, NZ, APIX, RADIAL = 2, 200, 13.328, (0.35, 0.035)

def bin2(a, b):
    h, w = a.shape
    a = a[: h // b * b, : w // b * b]
    return a.reshape(h // b, b, w // b, b).mean(axis=(1, 3))

ang = np.array([float(x) for x in open(D / "Vibrio_pilT_pilU_131.rawtlt").read().split()])
with mrcfile.mmap(str(D / "Vibrio_pilT_pilU_131.mrc"), permissive=True, mode="r") as m:
    g = np.stack([-(lambda a: (a - a.mean()) / (a.std() + 1e-9))(
        bin2(np.asarray(m.data[i], np.float32), BIN)) for i in range(m.data.shape[0])])
n_t, NV, NU = g.shape
px = APIX * BIN / 10.0
print(f"{g.shape} binned, {px:.3f} nm/voxel, {len(ang)} tilts "
      f"{ang.min():.1f}..{ang.max():.1f} deg")

knot = UniformSpec.centered(step=px, num=(NU, NV, NZ))
S = float(np.prod(knot.step))
uu = (np.arange(NU) - (NU - 1) / 2) * px
vv = (np.arange(NV) - (NV - 1) / 2) * px
U, V = np.meshgrid(uu, vv, indexing="ij"); U, V = U.ravel(), V.ravel()
tx, ty, tz, nx, ny, nz = [], [], [], [], [], []
for th in np.deg2rad(ang):
    c, s = np.cos(th), np.sin(th)
    tx.append(U * c); ty.append(V); tz.append(-U * s)
    nx.append(np.full(U.size, s)); ny.append(np.zeros(U.size)); nz.append(np.full(U.size, c))
cat = lambda L: np.concatenate(L).astype(np.float32)
ray = (Array3f(cat(tx), cat(ty), cat(tz)), Array3f(cat(nx), cat(ny), cat(nz)))
print(f"{dr.width(ray[0])/1e6:.1f} M rays, volume {NU}x{NV}x{NZ}")

fk = np.fft.fftfreq(NU); r = np.abs(fk) / 0.5
cut, fall = RADIAL
W = (np.abs(fk) * np.where(r <= cut, 1.0, np.exp(-0.5 * ((r - cut) / fall) ** 2))
     ).astype(np.float32)

def ramp(vec):                       # W applied along U, in data space
    a = np.asarray(vec).reshape(n_t, NU, NV)
    return Float(np.ascontiguousarray(
        np.real(np.fft.ifft(np.fft.fft(a, axis=1) * W[None, :, None], axis=1)
                ).astype(np.float32).ravel()))

A  = lambda f: xrt_apply(ray, knot, 0, f)
At = lambda d: S * xrt_adjoint(ray, knot, 0, d)
y  = Float(np.ascontiguousarray(np.transpose(g, (0, 2, 1)).ravel()))

t0 = time.time()
b = At(ramp(y)); dr.eval(b)
f = dr.zeros(Float, NU * NV * NZ); rr = Float(b); p = Float(rr)
rs = dr.sum(rr * rr)
for it in range(8):
    Ap = At(ramp(A(p))); dr.eval(Ap)
    al = rs / (dr.sum(p * Ap) + 1e-30)
    f = dr.fma(al, p, f); rr = dr.fma(-al, Ap, rr)
    rn = dr.sum(rr * rr); p = dr.fma(rn / rs, p, rr); rs = rn
    dr.eval(f, rr, p, rs)
dr.sync_thread()
print(f"filtered CGLS, 8 iterations: {time.time()-t0:.1f}s")
vol = np.asarray(f).reshape(NU, NV, NZ)
np.save("/tmp/cryo_vibrio_cg.npy", vol.astype(np.float32))
z, half = 88, 5
sl = vol[:, :, z-half:z+half].mean(2).T
lo, hi = np.percentile(sl, 0.5), np.percentile(sl, 99.5)
fig, ax = plt.subplots(figsize=(6.4, 6.4))
ax.imshow(sl, cmap="gray", vmin=lo, vmax=hi, origin="lower",
          extent=[0, NU*px/1000, 0, NV*px/1000])
ax.set_xlabel("µm"); ax.set_ylabel("µm")
ax.set_title(f"conjugate gradients, {2*half*px:.0f} nm slab at z={z}")
plt.tight_layout(); plt.savefig("/tmp/cryo_cg.png", dpi=110, bbox_inches="tight")
print("saved")
