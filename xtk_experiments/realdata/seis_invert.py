"""Bent-ray first-arrival tomography of the REAL Koenigsee refraction dataset.

Forward: fan-shoot with XTK's refractive marcher, first arrival = min time over
the emerging fan (per shot).  Gradient: Fermat / frozen-path adjoint --
`refract_adjoint` scatters each residual along the very ray that produced it.
Step: scale-free Cauchy step from the linearised forward, plus backtracking.
"""
import time

import numpy as np
import drjit as dr
from drjit.cuda.ad import Array2f, Float
from scipy.ndimage import gaussian_filter

import seis_common as sc
from xrt_toolkit.drjit.curved_xrt import refract_apply, refract_adjoint

N_FAN = 4000
ITERS = 14
SMOOTH = (2.0, 1.5)          # gaussian smoothing of the gradient (cells)


def forward_all(u_flat, pos, s, g, n_fan=N_FAN, bin_w=0.12):
    """Predicted first arrivals + the launch direction of the selected ray.

    The emerging fan is reduced to a FIRST-ARRIVAL curve by taking, in each
    narrow bin of emergence position, the minimum-time ray (the lower envelope
    of the multivalued traveltime branches), then interpolating that curve at
    the geophone positions. Taking a plain minimum over a wide neighbourhood
    would bias the times low.
    """
    pred = np.full(len(s), np.nan)
    ldir = np.zeros((2, len(s)), np.float32)
    for si in np.unique(s):
        ex, ez, T, dirs = sc.fan_shoot(u_flat, pos[si, 0], n_fan=n_fan)
        ok = (ez < sc.DX) & np.isfinite(T)
        if ok.sum() < 5:
            continue
        exo, tto, do = ex[ok], T[ok], dirs[:, ok]
        # lower envelope on a fine grid of emergence positions
        b = np.floor((exo - sc.X0) / bin_w).astype(int)
        order = np.lexsort((tto, b))
        b_s, t_s, i_s = b[order], tto[order], np.arange(len(b))[order]
        first = np.ones(len(b_s), bool)
        first[1:] = b_s[1:] != b_s[:-1]
        bx = sc.X0 + (b_s[first] + 0.5) * bin_w
        bt = t_s[first]
        bidx = i_s[first]
        if len(bx) < 2:
            continue
        m = np.where(s == si)[0]
        xt = pos[g[m], 0]
        inrange = (xt >= bx.min() - bin_w) & (xt <= bx.max() + bin_w)
        pv = np.interp(xt, bx, bt, left=np.nan, right=np.nan)
        pv = np.where(inrange, pv, np.nan)

        # Near field: a finite grid cannot resolve turning depths below one
        # cell, so the fan has a minimum reachable offset. The theta -> 0 limit
        # of the diving ray is the surface direct wave, T = offset / v_surface;
        # include it as a candidate so short offsets are modelled too.
        vsurf = 1.0 / float(np.array(u_flat)[0::sc.NZ][0]) if False else None
        u_np = np.array(u_flat).reshape(sc.NX, sc.NZ)
        ix = np.clip(((xt - sc.X0) / sc.DX).astype(int), 0, sc.NX - 1)
        isx = int(np.clip((pos[si, 0] - sc.X0) / sc.DX, 0, sc.NX - 1))
        u_top = u_np[:, 0]
        t_dir = np.array([np.mean(u_top[min(isx, j):max(isx, j) + 1])
                          * abs(pos[si, 0] - xv) for j, xv in zip(ix, xt)])
        use_dir = ~np.isfinite(pv) | (t_dir < pv)
        pred[m] = np.where(use_dir, t_dir, pv)

        # launch direction: the envelope ray, or horizontal for direct arrivals
        near = np.clip(np.searchsorted(bx, xt), 0, len(bx) - 1)
        ld = do[:, bidx[near]].copy()
        sgn = np.sign(xt - pos[si, 0]); sgn[sgn == 0] = 1.0
        ld[0] = np.where(use_dir, sgn, ld[0])
        ld[1] = np.where(use_dir, 1e-3, ld[1])
        ldir[:, m] = ld
    return pred, ldir


def make_rays(pos, s, ldir, sel):
    sx = pos[s[sel], 0].astype(np.float32)
    sz = np.full(sel.sum(), 1e-3, np.float32)
    return (Array2f(sx.copy(), sz),
            Array2f(ldir[0, sel].copy(), ldir[1, sel].copy()))


def main():
    pos, s, g, t = sc.read_sgt()
    v = sc.gradient_model()
    u = Float((1.0 / v).ravel().astype(np.float32))
    print(f"grid {sc.NX}x{sc.NZ}, {len(t)} picks, {len(np.unique(s))} shots")

    hist = []
    for it in range(ITERS):
        t0 = time.time()
        pred, ldir = forward_all(u, pos, s, g)
        ok = np.isfinite(pred)
        res = t[ok] - pred[ok]
        rmse = np.sqrt(np.mean(res**2))
        hist.append(rmse)
        print(f"iter {it:2d}: coverage {ok.sum()}/{len(t)}  RMSE {rmse*1000:6.3f} ms"
              f"  ({time.time()-t0:.1f}s)", flush=True)

        ray = make_rays(pos, s, ldir, ok)
        rr = Float(res.astype(np.float32))
        # Fermat adjoint: backproject each residual along its own bent ray.
        # Parameterise in LOG-slowness m = log(u): the chain rule gives
        # dT/dm = u * dT/du, which equalises the sensitivity between the slow
        # near surface and the fast deep part (slowness spans one decade).
        grad_u = refract_adjoint(ray, sc.KNOT, rr, u, ds=sc.DS, bend=True)
        dr.eval(grad_u)
        gm = np.array(grad_u) * np.array(u)
        gm = gaussian_filter(gm.reshape(sc.NX, sc.NZ), SMOOTH).ravel()
        gm /= (np.abs(gm).max() + 1e-30)          # unit-scale direction

        # Cauchy step from the linearised (frozen-path) forward: du = u * dm
        du = Float((gm * np.array(u)).astype(np.float32))
        Jg = refract_apply(ray, sc.KNOT, du, coeff_geom=u, ds=sc.DS, bend=True)
        dr.eval(Jg)
        Jg_np = np.array(Jg)
        alpha = float(np.sum(res * Jg_np)) / (float(np.sum(Jg_np**2)) + 1e-30)
        if not np.isfinite(alpha) or alpha <= 0:
            print("  no descent direction, stop"); break

        u_np = np.array(u)
        improved = False
        for _ in range(14):
            u_try_np = np.clip(u_np * np.exp(alpha * gm),
                               1.0 / sc.V_HI, 1.0 / sc.V_LO)
            u_try = Float(u_try_np.astype(np.float32))
            p2, _ = forward_all(u_try, pos, s, g, n_fan=N_FAN)
            o2 = np.isfinite(p2)
            r2 = np.sqrt(np.mean((t[o2] - p2[o2])**2))
            if r2 < rmse:
                u = u_try; improved = True
                print(f"      step {alpha:.4f} -> RMSE {r2*1000:.3f} ms", flush=True)
                break
            alpha *= 0.5
        if not improved:
            print("  line search failed, stop"); break

    pred, _ = forward_all(u, pos, s, g)
    ok = np.isfinite(pred)
    rmse = np.sqrt(np.mean((t[ok] - pred[ok])**2))
    print(f"FINAL RMSE {rmse*1000:.3f} ms (start {hist[0]*1000:.3f} ms)")
    vel = (1.0 / np.array(u)).reshape(sc.NX, sc.NZ)
    np.savez(sc.HERE / "results" / "seis_result.npz", vel=vel, pred=pred, obs=t, s=s, g=g, pos=pos,
             hist=np.array(hist), XC=sc.XC, ZC=sc.ZC)
    print("saved seis_result.npz; v range", vel.min(), vel.max())


if __name__ == "__main__":
    main()
