"""xrt_toolkit wrapper for the benchmark cases of common.py (any order)."""

import math
import sys
from pathlib import Path

import drjit as dr
import numpy as np
from drjit.cuda import Array2f, Array3f, Float

import xrt_toolkit as xtk
from xrt_toolkit.drjit.struct_xrt import _rays

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def geometry(kind: str, D: int, N: int):
    det = (N,) * (D - 1)
    if kind == "parallel":
        ang = Float(common.angles(kind, N).astype(np.float32))
        return xtk.parallel_beam(ang, xtk.util.DetectorSpec(size=det, num_cell=det))
    if kind == "cone":
        ang = Float(common.angles(kind, N).astype(np.float32))
        size = tuple(common.pitch(kind) * d for d in det)
        det_spec = xtk.util.DetectorSpec(size=size, num_cell=det)
        return xtk.cone_beam(common.sod(N), common.sdd(N), ang, det_spec)
    if kind == "random":  # parallel beam, every ray shifted and tilted at random
        rng = np.random.default_rng(0)
        t, n = (
            np.array(a, dtype=np.float32) for a in _rays(geometry("parallel", D, N))
        )
        t += rng.uniform(-0.5, 0.5, t.shape).astype(np.float32)  # +- half a voxel
        n += rng.normal(0, 0.01, n.shape).astype(np.float32)  # ~0.6 deg
        n /= np.linalg.norm(n, axis=0)
        ArrayNf = Array2f if D == 2 else Array3f
        return ArrayNf(t), ArrayNf(n)


def operators(kind: str, D: int, N: int, order: int = 0, stored: bool = False) -> dict:
    # stored=True: compute the rays once and pass them to xrt_apply/xrt_adjoint
    knot_spec = xtk.UniformSpec.centered(step=1, num=(N,) * D)
    rays = geometry(kind, D, N)
    if stored and len(rays) == 3:
        rays = _rays(rays)
        dr.eval(rays)
    struct = len(rays) == 3
    apply = xtk.xrt_struct_apply if struct else xtk.xrt_apply
    adjoint = xtk.xrt_struct_adjoint if struct else xtk.xrt_adjoint
    L = dr.width(rays[0]) * (math.prod(rays[2].num) if struct else 1)

    def run(fn, arg):
        out = fn(rays, knot_spec, order, arg)
        dr.eval(out)
        return out

    def to_lib(a):
        out = Float(a)
        dr.eval(out)
        return out

    return dict(
        fwd=lambda x: run(apply, x),
        adj=lambda y: run(adjoint, y),
        x_size=N**D,
        y_size=L,
        to_lib=to_lib,
        to_np=np.asarray,
        sync=dr.sync_thread,
    )


if __name__ == "__main__":
    common.main(operators)
