"""
parallelproj wrapper for the benchmark cases of common.py (3D).

Joseph projector joseph3d_fwd / joseph3d_back on CuPy arrays, with one line of response
(start and end point) per ray.
"""

import sys
from pathlib import Path

import cupy as cp
import parallelproj

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common  # noqa: E402


def operators(kind: str, D: int, N: int) -> dict:
    assert D == 3
    ang = cp.asarray(common.angles(kind, N), dtype=cp.float32)
    p = common.pitch(kind)
    u = (cp.arange(N, dtype=cp.float32) - (N - 1) / 2) * p
    c, s = cp.cos(ang)[:, None, None], cp.sin(ang)[:, None, None]
    # (view, row, col): detector u along e2, v along z
    U, V = u[None, :, None], u[None, None, :]
    zero = cp.zeros_like(c * U * V)
    if kind == "parallel":
        r = float(N)  # half-length of each ray, beyond the volume
        mid = cp.stack([-s * U + zero, c * U + zero, V + zero], axis=-1)
        d = cp.stack([c + zero, s + zero, zero], axis=-1)
        xstart, xend = mid - r * d, mid + r * d
    else:
        sod, odd = common.sod(N), common.sdd(N) - common.sod(N)
        xstart = cp.stack([-sod * c + zero, -sod * s + zero, zero], axis=-1)
        xend = cp.stack(
            [odd * c - s * U + zero, odd * s + c * U + zero, V + zero], axis=-1
        )
    xstart, xend = xstart.reshape(-1, 3), xend.reshape(-1, 3)
    shape = (N, N, N)
    origin = cp.asarray([-(N - 1) / 2] * 3, dtype=cp.float32)
    voxsize = cp.asarray([1.0, 1.0, 1.0], dtype=cp.float32)

    return dict(
        fwd=lambda x: parallelproj.joseph3d_fwd(
            xstart, xend, x.reshape(shape), origin, voxsize
        ),
        adj=lambda y: parallelproj.joseph3d_back(
            xstart, xend, shape, origin, voxsize, y
        ),
        x_size=N**3,
        y_size=xstart.shape[0],
        to_lib=lambda a: cp.asarray(a),
        to_np=cp.asnumpy,
        sync=cp.cuda.Device().synchronize,
    )


if __name__ == "__main__":
    common.main(operators)
