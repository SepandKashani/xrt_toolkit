"""
Adjoint mismatch benchmarks; prints the Markdown tables of adjoint.md.

    python benchmarks/adjoint/adjoint.py

Mismatch = ||B - A^T||_F / ||A||_F of a library's back-projection B against the exact
adjoint of its own projection A, estimated from 20 random dot-product tests
(see common.adjoint_mismatch). 0 means B is the exact adjoint; float32 rounding gives ~1e-7.
Each library runs in its own interpreter, as in speed/speed.py.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "libs")]
import common  # noqa: E402
import run_xtk  # noqa: E402


def fmt(e):
    return "-" if e is None else f"{e:.0e}" if e < 1e-3 else f"{e:.2f}"


if __name__ == "__main__":
    print("| geometry | order | 2D | 3D |")
    print("|---|---|---:|---:|")
    for kind, name in (
        ("parallel", "parallel"),
        ("cone", "fan / cone"),
        ("random", "random"),
    ):
        for order in (0, 1, 2):
            e = [
                common.adjoint_mismatch(run_xtk.operators(kind, D, N, order))[
                    "mismatch"
                ]
                for D, N in ((2, 512), (3, 128))
            ]
            print(
                f"| {name if order == 0 else ''} | {order} | {fmt(e[0])} | {fmt(e[1])} |"
            )

    cases = [
        (D, N, kind) for D, N in ((2, 512), (3, 128)) for kind in ("parallel", "cone")
    ]
    res = {
        (lib, D, kind): common.run_case(lib, "adjoint", kind, D, N)
        for lib in common.LIBRARIES
        for D, N, kind in cases
    }
    print()
    names = {"parallel": "parallel", "cone": "fan"}
    labels = [f"{D}D {names[kind] if D == 2 else kind}" for D, _, kind in cases]
    print("| library | " + " | ".join(labels) + " |")
    print("|---|" + "---:|" * len(cases))
    for lib in common.LIBRARIES:
        e = []
        for D, _, kind in cases:
            r = res[lib, D, kind]
            e.append(
                "-" if r is None else f"{fmt(r['mismatch'])} ({fmt(r['rescaled'])})"
            )
        print(f"| {lib} | " + " | ".join(e) + " |")
