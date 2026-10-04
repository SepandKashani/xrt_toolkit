"""Straight-ray BASELINE vs bent-ray, on the same real crosshole GPR picks.

Identical solver, regularisation, stopping rule and bounds for both runs -- the
only difference is whether the ray is allowed to refract:

  straight : launch direction = straight line source->receiver, `bend=False`,
             integrated to the receiver via `smax = |r - s|`.
  bent     : two-point ray linking (secant on the launch angle) with `bend=True`.

Data: Arrenaes crosshole GPR, 702 picked first arrivals, sigma = 0.8 ns
(Looms et al., Geophysics 75(6) J29-J41, 2010; via SIPPI).
"""
import pathlib
import sys
import time

import numpy as np
import drjit as dr
from drjit.cuda.ad import Array2f, Float
from scipy.ndimage import gaussian_filter

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from xrt_toolkit.util import UniformSpec                       # noqa: E402
from xrt_toolkit.drjit.curved_xrt import (refract_adjoint,     # noqa: E402
                                          refract_apply)
from ray_link import link_time                                 # noqa: E402

DX = 0.10
X0, X1, Z0, Z1 = 0.0, 5.0, 0.5, 12.5
NX, NZ = int(round((X1 - X0) / DX)), int(round((Z1 - Z0) / DX))
KNOT = UniformSpec(start=(X0 + DX / 2, Z0 + DX / 2), step=(DX, DX), num=(NX, NZ))
V_LO, V_HI = 0.09, 0.20
ITERS, SMOOTH = 25, (1.2, 1.2)


def load():
    L = open(HERE / "data" / "AM13_data.eas", encoding="latin-1").read().splitlines()
    n = int(L[1].split()[0])
    a = np.array([[float(v) for v in l.split()] for l in L[2 + n:] if l.strip()])
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4], a[:, 5]


def solve(bend, sx, sz, rx, rz, tobs, sigma):
    src = Array2f(sx.copy(), sz.copy())
    rcv = Array2f(rx.copy(), rz.copy())
    dist = np.hypot(rx - sx, rz - sz)
    # straight-ray geometry: fixed launch direction and arc length
    nd = np.stack([(rx - sx) / dist, (rz - sz) / dist])
    ldir_str = Array2f(nd[0].astype(np.float32).copy(), nd[1].astype(np.float32).copy())
    smax_str = Float(dist.astype(np.float32))

    v0 = float(np.mean(dist / tobs))
    u = Float(np.full(NX * NZ, 1.0 / v0, np.float32))

    def forward(u_):
        if bend:
            T, dmin, ldir, smax = link_time(src, rcv, KNOT, u_, ds=DX / 2,
                                            n_iter=9, bend=True)
            dr.eval(T, dmin, ldir, smax)
            return np.array(T), np.array(dmin), ldir, smax
        T = refract_apply((src, ldir_str), KNOT, u_, ds=DX / 2,
                          bend=False, smax=smax_str)
        dr.eval(T)
        return np.array(T), np.zeros(len(tobs)), ldir_str, smax_str

    hist = []
    for it in range(ITERS):
        T, dmin, ldir, smax = forward(u)
        ok = np.isfinite(T) & (dmin < 0.25)
        res = tobs[ok] - T[ok]
        rmse = float(np.sqrt(np.mean(res ** 2)))
        hist.append(rmse)
        if rmse <= sigma:
            print(f"    iter {it}: RMSE {rmse:.3f} ns <= sigma -> stop", flush=True)
            break
        ray = (Array2f(sx[ok].copy(), sz[ok].copy()),
               Array2f(np.array(ldir.x)[ok].copy(), np.array(ldir.y)[ok].copy()))
        sm = Float(np.array(smax)[ok].copy())
        gu = refract_adjoint(ray, KNOT, Float(res.astype(np.float32)), u,
                             ds=DX / 2, bend=bend, smax=sm)
        dr.eval(gu)
        gm = np.array(gu) * np.array(u)
        gm = gaussian_filter(gm.reshape(NX, NZ), SMOOTH).ravel()
        gm /= (np.abs(gm).max() + 1e-30)
        du = Float((gm * np.array(u)).astype(np.float32))
        Jg = refract_apply(ray, KNOT, du, coeff_geom=u, ds=DX / 2,
                           bend=bend, smax=sm)
        dr.eval(Jg)
        Jn = np.array(Jg)
        alpha = float(np.sum(res * Jn)) / (float(np.sum(Jn ** 2)) + 1e-30)
        if not np.isfinite(alpha) or alpha <= 0:
            break
        un = np.array(u)
        stepped = False
        for _ in range(14):
            ut = Float(np.clip(un * np.exp(alpha * gm), 1 / V_HI, 1 / V_LO).astype(np.float32))
            T2, d2, _, _ = forward(ut)
            o2 = np.isfinite(T2) & (d2 < 0.25)
            if float(np.sqrt(np.mean((tobs[o2] - T2[o2]) ** 2))) < rmse:
                u = ut; stepped = True; break
            alpha *= 0.5
        if not stepped:
            print(f"    iter {it}: line search failed -> stop (RMSE {rmse:.3f})", flush=True)
            break
    T, dmin, _, _ = forward(u)
    ok = np.isfinite(T) & (dmin < 0.25)
    return (1.0 / np.array(u)).reshape(NX, NZ), float(np.sqrt(np.mean((tobs[ok] - T[ok]) ** 2))), hist


def main():
    sx, sz, rx, rz, tobs, sig = load()
    sx = np.clip(sx, X0 + 1e-3, X1 - 1e-3).astype(np.float32)
    rx = np.clip(rx, X0 + 1e-3, X1 - 1e-3).astype(np.float32)
    sz, rz = sz.astype(np.float32), rz.astype(np.float32)
    sigma = float(sig[0])
    out = {}
    for bend in (False, True):
        name = "bent" if bend else "straight"
        print(f"  [{name}]")
        t0 = time.time()
        vel, rmse, hist = solve(bend, sx, sz, rx, rz, tobs, sigma)
        out[name] = dict(vel=vel, rmse=rmse, hist=np.array(hist), wall=time.time() - t0)
        print(f"    -> RMSE {rmse:.3f} ns, v {vel.min():.4f}-{vel.max():.4f} m/ns, "
              f"{out[name]['wall']:.1f}s")
    np.savez(HERE / "results" / "gpr_straight_vs_bent.npz",
             vel_straight=out["straight"]["vel"], vel_bent=out["bent"]["vel"],
             rmse_straight=out["straight"]["rmse"], rmse_bent=out["bent"]["rmse"],
             hist_straight=out["straight"]["hist"], hist_bent=out["bent"]["hist"],
             sigma=sigma, X0=X0, X1=X1, Z0=Z0, Z1=Z1)
    print("saved results/gpr_straight_vs_bent.npz")


if __name__ == "__main__":
    main()
