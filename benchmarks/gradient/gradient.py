"""
Gradient benchmarks: exactness and speed of xrt_grad(); prints the Markdown tables of
gradient.md.

    python benchmarks/gradient/gradient.py

Needs an otherwise idle CUDA GPU.
"""

import sys
import time
from pathlib import Path

import drjit as dr
import numpy as np
from drjit.cuda import Array2f, Array3f, Float

import xrt_toolkit as xtk
from xrt_toolkit_tests.test_grad import (
    box_spline_projections,
    projections,
    spline_projections,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "libs")]
import run_xtk  # noqa: E402


def rays(kind: str, D: int, N: int) -> tuple:
    # (D, L) float32 anchors and directions of a benchmark geometry
    r = run_xtk.geometry(kind, D, N)
    if len(r) == 3:
        r = xtk.struct_rays(r)
    return tuple(np.array(a, dtype=np.float32) for a in r)


def array_t(D: int):
    return Array2f if (D == 2) else Array3f


def reference(f, knot_spec, order: int, D: int, t, n) -> list:
    # Central, left and right differences of an exact float64 forward, in every component of
    # the anchors (first) and the directions (second).
    if order == 0:
        P = lambda t_, n_: projections(f, knot_spec, t_, n_)  # noqa: E731
    elif D == 2:
        P = lambda t_, n_: box_spline_projections(f, knot_spec, order, t_, n_)  # noqa: E731
    else:
        P = lambda t_, n_: spline_projections(f, knot_spec, order, t_, n_)  # noqa: E731
    # small, but above the round-off of the reference (2D splines: truncated powers)
    h = 1e-8 if (order == 0) else (1e-5 if (D == 2) else 1e-7)
    P0 = P(t, n)
    out = []
    for i in range(2):
        c, lft, rgt = np.zeros(t.shape), np.zeros(t.shape), np.zeros(t.shape)
        for k in range(D):
            e = h * np.eye(D)[:, [k]]
            Pp = P(t + e, n) if (i == 0) else P(t, n + e)
            Pm = P(t - e, n) if (i == 0) else P(t, n - e)
            c[k], lft[k], rgt[k] = (Pp - Pm) / (2 * h), (P0 - Pm) / h, (Pp - P0) / h
        out.append((c, lft, rgt))
    return out


def exactness(kind: str, D: int, N: int, L: int) -> dict:
    # On L rays of the geometry, the max error, relative to the largest gradient entry, against
    # central differences of an exact float64 forward (at a kink of the projection: the
    # average of both sides). Returns, for each order: xrt_grad() on all rays, xrt_grad() and
    # finite differences of xrt_apply() (float32) on the rays away from kinks, the number of
    # rays at kinks (where the one-sided derivatives differ).
    rng = np.random.default_rng(0)
    knot_spec = xtk.UniformSpec.centered(step=1, num=(N,) * D)
    t, n = rays(kind, D, N)
    keep = rng.choice(t.shape[1], L, replace=False)
    t, n = t[:, keep], n[:, keep]
    f = rng.standard_normal(knot_spec.num)
    ArrayNf, f32 = array_t(D), Float(f.ravel().astype(np.float32))

    out = {}
    for order in (0, 1, 2):
        ref = reference(f, knot_spec, order, D, t.astype(float), n.astype(float))
        smooth = np.ones(L, bool)
        for c, lft, rgt in ref:
            smooth &= np.all(np.abs(rgt - lft) < 1e-2 * np.abs(c).max(), axis=0)
        g = xtk.xrt_grad((ArrayNf(t), ArrayNf(n)), knot_spec, order, f32)

        h = 1e-3  # finite differences of the float32 forward: about the best step
        P = lambda t_, n_: np.asarray(  # noqa: E731
            xtk.xrt_apply((ArrayNf(t_), ArrayNf(n_)), knot_spec, order, f32), np.float64
        )
        fd = [np.zeros(t.shape), np.zeros(n.shape)]
        for k in range(D):
            e = (h * np.eye(D)[:, [k]]).astype(np.float32)
            for i, (a, b) in enumerate(((t + e, t - e), (n + e, n - e))):
                P_a = P(a, n) if (i == 0) else P(t, a)
                P_b = P(b, n) if (i == 0) else P(t, b)
                fd[i][k] = (P_a - P_b) / (
                    a[k].astype(float) - b[k]
                )  # actual float32 step

        def err(x, i, rays):
            x = np.asarray(x, np.float64)[:, rays]
            c = ref[i][0][:, rays]
            return np.abs(x - c).max() / np.abs(c).max()

        every = np.ones(L, bool)
        out[order] = (
            max(err(g[0], 0, every), err(g[1], 1, every)),
            max(err(g[0], 0, smooth), err(g[1], 1, smooth)),
            max(err(fd[0], 0, smooth), err(fd[1], 1, smooth)),
            int(L - smooth.sum()),
        )
    return out


def speed(kind: str, D: int, N: int, order: int, reps: int) -> dict:
    # median times [ms] of the forward, xrt_grad(), and central differences of the forward
    # in every anchor/direction component (4 D forwards)
    knot_spec = xtk.UniformSpec.centered(step=1, num=(N,) * D)
    ArrayNf = array_t(D)
    t, n = (ArrayNf(a) for a in rays(kind, D, N))
    dr.eval(t, n)
    f = Float(np.random.default_rng(0).random(N**D, dtype=np.float32))
    dr.eval(f)

    def fd():
        out = []
        for k in range(D):
            e = ArrayNf(*[1e-3 if (j == k) else 0 for j in range(D)])
            for t_, n_ in ((t + e, n), (t - e, n), (t, n + e), (t, n - e)):
                out.append(xtk.xrt_apply((t_, n_), knot_spec, order, f))
                dr.eval(out[-1])
        return out

    fns = {
        "fwd": lambda: xtk.xrt_apply((t, n), knot_spec, order, f),
        "grad": lambda: xtk.xrt_grad((t, n), knot_spec, order, f),
        "fd": fd,
    }
    times = {}
    for name, fn in fns.items():
        dr.eval(fn())
        dr.sync_thread()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            dr.eval(fn())
            dr.sync_thread()
            ts.append(time.perf_counter() - t0)
        times[name] = 1e3 * float(np.median(ts))
    return times


def fmt(t):
    return f"{t:.1f}" if t < 100 else f"{t:,.0f}"


GEOMETRIES = (("parallel", "parallel"), ("cone", "fan / cone"), ("random", "random"))

if __name__ == "__main__":
    for D, N, L in ((2, 65, 200), (3, 33, 100)):
        print(f"\n{D}D, {N}^{D} volume, {L} rays\n")
        print(
            "| geometry | order | xrt_grad | xrt_grad, no kinks | finite diff., no kinks | rays at kinks |"
        )
        print("|---|---|---:|---:|---:|---:|")
        for kind, name in GEOMETRIES:
            e = exactness(kind, D, N, L)
            for order in (0, 1, 2):
                a, b, c, k = e[order]
                label = name if order == 0 else ""
                print(f"| {label} | {order} | {a:.0e} | {b:.0e} | {c:.0e} | {k} |")

    # warm-up: let the GPU reach its steady clocks before timing
    t_end = time.perf_counter() + 5
    while time.perf_counter() < t_end:
        speed("parallel", 2, 1024, 0, 1)

    print("\nTime [ms]\n")
    print(
        "| geometry | order | 2D fwd | 2D xrt_grad | 2D finite diff. "
        "| 3D fwd | 3D xrt_grad | 3D finite diff. |"
    )
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for kind, name in GEOMETRIES:
        for order in (0, 1, 2):
            s = [
                speed(kind, D, N, order, reps)
                for D, N, reps in ((2, 1024, 10), (3, 256, 3))
            ]
            s = [fmt(d[k]) for d in s for k in ("fwd", "grad", "fd")]
            print(f"| {name if order == 0 else ''} | {order} | " + " | ".join(s) + " |")
