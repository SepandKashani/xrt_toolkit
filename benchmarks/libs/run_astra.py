"""
ASTRA wrapper for the benchmark cases of common.py.

3D: FP3D_CUDA / BP3D_CUDA on GPU-resident PyTorch arrays (astra.data3d.GPULink), so no
host <-> device copy is timed. 2D: FP_CUDA / BP_CUDA on host arrays (ASTRA has no GPU link
in 2D); used for the adjoint test only.
"""

import sys
from pathlib import Path

import astra
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def operators(kind: str, D: int, N: int) -> dict:
    ang = common.angles(kind, N)
    p = common.pitch(kind)
    if D == 3:
        return _operators_3d(kind, N, ang, p)
    vg = astra.create_vol_geom(N, N)
    if kind == "parallel":
        pg = astra.create_proj_geom("parallel", p, N, ang)
    else:
        pg = astra.create_proj_geom(
            "fanflat", p, N, ang, common.sod(N), common.sdd(N) - common.sod(N)
        )
    pid = astra.create_projector("cuda", pg, vg)

    def fwd(x):
        _, y = astra.create_sino(x.reshape(N, N), pid)
        return y

    def adj(y):
        _, x = astra.create_backprojection(y.reshape(N, N), pid)
        return x

    return dict(
        fwd=fwd,
        adj=adj,
        x_size=N * N,
        y_size=N * N,
        to_lib=lambda a: a,
        to_np=np.asarray,
        sync=lambda: None,
    )


def _operators_3d(kind, N, ang, p):
    import torch

    vg = astra.create_vol_geom(N, N, N)
    if kind == "parallel":
        pg = astra.create_proj_geom("parallel3d", p, p, N, N, ang)
    else:
        pg = astra.create_proj_geom(
            "cone", p, p, N, N, ang, common.sod(N), common.sdd(N) - common.sod(N)
        )
    vol = torch.zeros((N, N, N), device="cuda")
    sino = torch.zeros((N, N, N), device="cuda")  # (rows, angles, cols)
    vid = astra.data3d.link(
        "-vol", vg, astra.data3d.GPULink(vol.data_ptr(), N, N, N, 4 * N)
    )
    sid = astra.data3d.link(
        "-sino", pg, astra.data3d.GPULink(sino.data_ptr(), N, N, N, 4 * N)
    )
    cfg = astra.astra_dict("FP3D_CUDA")
    cfg.update(VolumeDataId=vid, ProjectionDataId=sid)
    fp = astra.algorithm.create(cfg)
    cfg = astra.astra_dict("BP3D_CUDA")
    cfg.update(ReconstructionDataId=vid, ProjectionDataId=sid)
    bp = astra.algorithm.create(cfg)

    def fwd(x):
        vol.view(-1).copy_(x)
        astra.algorithm.run(fp)
        return sino

    def adj(y):
        sino.view(-1).copy_(y)
        astra.algorithm.run(bp)
        return vol

    return dict(
        fwd=fwd,
        adj=adj,
        x_size=N**3,
        y_size=N**3,
        to_lib=lambda a: torch.from_numpy(a).cuda(),
        to_np=lambda a: a.cpu().numpy(),
        sync=torch.cuda.synchronize,
    )


if __name__ == "__main__":
    common.main(operators)
