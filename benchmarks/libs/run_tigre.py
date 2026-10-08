"""
TIGRE wrapper for the benchmark cases of common.py (3D).

Ax(..., "interpolated") and Atb(..., "matched"): TIGRE's pair closest to an exact adjoint
(with its default "Siddon" projector the mismatch is larger). TIGRE takes host (NumPy)
arrays, so its times include the host <-> device copies.
"""

import sys
from pathlib import Path

import numpy as np
import tigre

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def operators(kind: str, D: int, N: int) -> dict:
    assert D == 3
    geo = tigre.geometry(
        mode="parallel" if kind == "parallel" else "cone",
        nVoxel=np.array([N, N, N]),
        default=True,
    )
    p = common.pitch(kind)
    geo.DSO, geo.DSD = common.sod(N), common.sdd(N)
    geo.nDetector = np.array([N, N])
    geo.dDetector = np.array([p, p])
    geo.sDetector = geo.nDetector * geo.dDetector
    geo.dVoxel = np.array([1.0, 1.0, 1.0])
    geo.sVoxel = geo.nVoxel * geo.dVoxel
    geo.offOrigin = np.zeros(3)
    geo.offDetector = np.zeros(2)
    ang = common.angles(kind, N).astype(np.float32)

    return dict(
        fwd=lambda x: tigre.Ax(x.reshape(N, N, N), geo, ang, "interpolated"),
        adj=lambda y: tigre.Atb(y.reshape(N, N, N), geo, ang, "matched"),
        x_size=N**3,
        y_size=N**3,
        to_lib=lambda a: np.ascontiguousarray(a, dtype=np.float32),
        to_np=np.asarray,
        sync=lambda: None,  # TIGRE calls return host arrays: already synchronised
    )


if __name__ == "__main__":
    common.main(operators)
