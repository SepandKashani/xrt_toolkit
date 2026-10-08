"""
Speed benchmarks; prints the Markdown tables of speed.md.

    python benchmarks/speed/speed.py

Needs an otherwise idle CUDA GPU. Each library runs in the interpreter given by
ASTRA_PYTHON, TIGRE_PYTHON or PARALLELPROJ_PYTHON (default: this one); a library that does
not run there is shown as "-".
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "libs")]
import common  # noqa: E402
import run_xtk  # noqa: E402


def fmt(t):
    return "-" if t is None else (f"{t:.1f}" if t < 100 else f"{t:,.0f}")


if __name__ == "__main__":
    # warm-up: let the GPU reach its steady clocks before timing
    t_end = time.perf_counter() + 5
    while time.perf_counter() < t_end:
        common.speed(run_xtk.operators("parallel", 2, 1024), 1)

    print("| geometry | order | 2D fwd | 2D adj | 3D fwd | 3D adj |")
    print("|---|---|---:|---:|---:|---:|")
    for kind, name in (
        ("parallel", "parallel"),
        ("cone", "fan / cone"),
        ("random", "random"),
    ):
        for order in (0, 1, 2):
            t = [
                common.speed(run_xtk.operators(kind, D, N, order), reps)
                for D, N, reps in ((2, 1024, 10), (3, 256, 5))
            ]
            t = [fmt(d[k]) for d in t for k in ("fwd", "adj")]
            print(f"| {name if order == 0 else ''} | {order} | " + " | ".join(t) + " |")

    print()
    print("| | order | struct fwd | stored fwd | struct adj | stored adj |")
    print("|---|---|---:|---:|---:|---:|")
    for D, N in ((2, 1024), (3, 256)):
        for order in (0, 1, 2):
            s = common.speed(run_xtk.operators("parallel", D, N, order), 5)
            r = common.speed(run_xtk.operators("parallel", D, N, order, stored=True), 5)
            t = [fmt(t) for t in (s["fwd"], r["fwd"], s["adj"], r["adj"])]
            label = f"{D}D" if order == 0 else ""
            print(f"| {label} | {order} | " + " | ".join(t) + " |")

    cases = [
        (N, kind, reps)
        for N, reps in ((256, 10), (1024, 3))
        for kind in ("parallel", "cone")
    ]
    res = {
        (lib, N, kind): common.run_case(lib, "speed", kind, 3, N, reps)
        for lib in common.LIBRARIES
        for N, kind, reps in cases
    }
    for key, title in (("fwd", "projection"), ("adj", "back-projection")):
        print(f"\n3D {title}\n")
        print("| volume | geometry | " + " | ".join(common.LIBRARIES) + " |")
        print("|---|---|" + "---:|" * len(common.LIBRARIES))
        for N, kind, _ in cases:
            t = [
                fmt(r[key] if (r := res[lib, N, kind]) else None)
                for lib in common.LIBRARIES
            ]
            print(f"| {N}³ | {kind} | " + " | ".join(t) + " |")
