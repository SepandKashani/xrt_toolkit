"""Bent-ray crosshole travel-time tomography of REAL GPR field data.

Data: Arrenaes, Denmark (Looms, Hansen, Cordua, Nielsen, Jensen & Binley,
Geophysics 75(6) J29-J41, 2010), distributed with the SIPPI toolbox as
data/crosshole/AM13_data.eas -- 702 PICKED first-arrival travel times between
two boreholes 5 m apart, transmitter and receiver depths 1-12 m.
(The .eas column header in the file is wrong: the six columns are really
Sx, Sz, Rx, Rz, traveltime, sigma, and the times are in NANOSECONDS.)

Each source-receiver pair is a genuine TWO-POINT problem, so the ray is linked
by shooting: XTK's refractive marcher traces every candidate ray in parallel
and a secant iteration on the signed miss distance adjusts the launch angle
until the ray lands on the receiver. The travel-time gradient is then the
Fermat (frozen-path) adjoint along those linked rays.
"""
import pathlib
import sys
import time

import numpy as np
import drjit as dr
from drjit.cuda.ad import Array2f, Float
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from xrt_toolkit.util import UniformSpec
from xrt_toolkit.drjit.curved_xrt import refract_apply, refract_adjoint
from ray_link import link_time

DX = 0.10                     # m
X0, X1 = 0.0, 5.0
Z0, Z1 = 0.5, 12.5
NX = int(round((X1 - X0) / DX))
NZ = int(round((Z1 - Z0) / DX))
KNOT = UniformSpec(start=(X0 + DX / 2, Z0 + DX / 2), step=(DX, DX), num=(NX, NZ))
V_LO, V_HI = 0.09, 0.20       # m/ns bounds
ITERS = 25
SMOOTH = (1.2, 1.2)
HERE = pathlib.Path(__file__).resolve().parent


def load():
    L = open(HERE / "data" / "AM13_data.eas", encoding="latin-1").read().splitlines()
    n = int(L[1].split()[0])
    a = np.array([[float(v) for v in l.split()] for l in L[2 + n:] if l.strip()])
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]


def main():
    sx, sz, rx, rz, tobs, sig = load()
    L = len(tobs)
    # nudge sources/receivers just inside the model box
    sxi = np.clip(sx, X0 + 1e-3, X1 - 1e-3).astype(np.float32)
    rxi = np.clip(rx, X0 + 1e-3, X1 - 1e-3).astype(np.float32)
    src = Array2f(sxi.copy(), sz.astype(np.float32).copy())
    rcv = Array2f(rxi.copy(), rz.astype(np.float32).copy())
    print(f"{L} picks, grid {NX}x{NZ} @ {DX} m")

    v0 = float(np.mean(np.hypot(rx - sx, rz - sz) / tobs))
    v = np.full((NX, NZ), v0)
    u = Float((1.0 / v).ravel().astype(np.float32))
    print(f"homogeneous start v0 = {v0:.4f} m/ns")

    def forward(u_):
        T, dmin, ldir, smax = link_time(src, rcv, KNOT, u_, ds=DX / 2,
                                        n_iter=9, bend=True)
        dr.eval(T, dmin, ldir, smax)
        return np.array(T), np.array(dmin), ldir, smax

    hist = []
    for it in range(ITERS):
        T, dmin, ldir, smax = forward(u)
        ok = np.isfinite(T) & (dmin < 0.25)
        res = tobs[ok] - T[ok]
        rmse = np.sqrt(np.mean(res**2))
        hist.append(rmse)
        # discrepancy principle: stop once the model explains the data to
        # within the quoted picking uncertainty (sigma = 0.8 ns)
        if rmse <= float(sig[0]):
            print(f"iter {it:2d}: RMSE {rmse:.3f} ns <= sigma {sig[0]} ns -> stop", flush=True)
            break
        if it % 2 == 0 or it == ITERS - 1:
            print(f"iter {it:2d}: linked {ok.sum()}/{L} (max miss {dmin[ok].max():.3f} m)"
                  f"  RMSE {rmse:.3f} ns", flush=True)

        ray = (Array2f(sxi[ok].copy(), sz[ok].astype(np.float32).copy()),
               Array2f(np.array(ldir.x)[ok].copy(), np.array(ldir.y)[ok].copy()))
        sm = Float(np.array(smax)[ok].copy())
        rr = Float(res.astype(np.float32))
        gu = refract_adjoint(ray, KNOT, rr, u, ds=DX / 2, bend=True, smax=sm)
        dr.eval(gu)
        gm = np.array(gu) * np.array(u)                    # log-slowness
        gm = gaussian_filter(gm.reshape(NX, NZ), SMOOTH).ravel()
        gm /= (np.abs(gm).max() + 1e-30)

        du = Float((gm * np.array(u)).astype(np.float32))
        Jg = refract_apply(ray, KNOT, du, coeff_geom=u, ds=DX / 2,
                           bend=True, smax=sm)
        dr.eval(Jg)
        Jn = np.array(Jg)
        alpha = float(np.sum(res * Jn)) / (float(np.sum(Jn**2)) + 1e-30)
        if not np.isfinite(alpha) or alpha <= 0:
            print("  no descent, stop"); break

        un = np.array(u)
        ok_step = False
        for _ in range(14):
            ut = Float(np.clip(un * np.exp(alpha * gm),
                               1.0 / V_HI, 1.0 / V_LO).astype(np.float32))
            T2, d2, _, _ = forward(ut)
            o2 = np.isfinite(T2) & (d2 < 0.25)
            r2 = np.sqrt(np.mean((tobs[o2] - T2[o2])**2))
            if r2 < rmse:
                u = ut; ok_step = True; break
            alpha *= 0.5
        if not ok_step:
            print("  line search failed, stop"); break

    T, dmin, _, _ = forward(u)
    ok = np.isfinite(T) & (dmin < 0.25)
    rmse = np.sqrt(np.mean((tobs[ok] - T[ok])**2))
    vel = (1.0 / np.array(u)).reshape(NX, NZ)
    print(f"FINAL RMSE {rmse:.3f} ns (start {hist[0]:.3f} ns; pick sigma {sig[0]} ns)")
    print(f"v range {vel.min():.4f}..{vel.max():.4f} m/ns")
    np.savez(HERE / "results" / "gpr_result.npz", vel=vel, T=T, tobs=tobs, ok=ok, hist=np.array(hist),
             sz=sz, rz=rz, X0=X0, X1=X1, Z0=Z0, Z1=Z1, DX=DX, sigma=sig[0])
    print("saved gpr_result.npz")


if __name__ == "__main__":
    main()
