import drjit as dr
import numpy as np
import pytest

import xrt_toolkit as xtk
from xrt_toolkit.interop import vol_astra_to_xtk, vol_xtk_to_astra

try:
    from drjit.cuda import Array2f, Array3f, Float

    dr.width(Float(0))
except Exception:
    from drjit.llvm import Array2f, Array3f, Float


@pytest.mark.parametrize("shape", [(5, 7), (4, 5, 6)])
def test_volume_roundtrip(shape):
    vol = np.random.default_rng(0).random(shape)
    num = shape[::-1]  # (nx, ny[, nz])
    assert np.array_equal(vol_xtk_to_astra(vol_astra_to_xtk(vol), num), vol)


@pytest.mark.parametrize("s", [1.0, 2.0])
@pytest.mark.parametrize("geometry", ["parallel", "fanflat", "parallel_vec"])
def test_from_astra_2d(geometry, s):
    # ASTRA's 2D "line" projectors compute exact chord lengths, like XTK at order 0.
    astra = pytest.importorskip("astra")
    rng = np.random.default_rng(1)
    N, n_det, angles = 24, 36, np.linspace(0, np.pi, 12, endpoint=False)
    vg = astra.create_vol_geom(N, N, -N * s / 2, N * s / 2, -N * s / 2, N * s / 2)
    if geometry == "fanflat":
        pg = astra.create_proj_geom(
            "fanflat", 1.5 * s, n_det, 2 * angles, 50 * s, 40 * s
        )
    else:
        pg = astra.create_proj_geom("parallel", s, n_det, angles)
    if geometry == "parallel_vec":  # detector far behind the volume
        pg = astra.geom_2vec(pg)
        pg["Vectors"][:, 2:4] += 100 * s * pg["Vectors"][:, 0:2]
    pid = astra.create_projector(
        "line_fanflat" if geometry == "fanflat" else "line", pg, vg
    )
    vol = rng.random((N, N)).astype(np.float32)
    _, sino = astra.create_sino(vol, pid)

    ray_spec, knot_spec = xtk.from_astra(pg, vg, Array=Array2f)
    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, 0, Float(vol_astra_to_xtk(vol))))
    p_gt = sino.ravel() / s
    assert np.abs(p - p_gt).max() < 1e-4 * np.abs(p_gt).max()


@pytest.mark.parametrize("geometry", ["parallel3d", "cone"])
def test_from_astra_3d(geometry):
    # ASTRA's 3D GPU projector interpolates the volume linearly, as XTK's order 1 does:
    # they agree to the accuracy of ASTRA's sampling on a smooth volume.
    astra = pytest.importorskip("astra")
    if not astra.use_cuda():
        pytest.skip("ASTRA without CUDA")
    N, angles = 24, np.linspace(0, 2 * np.pi, 12, endpoint=False)
    x = np.arange(N) - (N - 1) / 2
    Z, Y, X = np.meshgrid(x, x, x, indexing="ij")
    vol = np.exp(-((X - 2) ** 2 + (Y + 1) ** 2 + (Z - 1) ** 2) / (2 * 3.0**2))
    vol = vol.astype(np.float32)
    vg = astra.create_vol_geom(N, N, N)
    if geometry == "parallel3d":
        pg = astra.create_proj_geom("parallel3d", 1.0, 1.0, 28, 32, angles / 2)
    else:
        pg = astra.create_proj_geom("cone", 1.5, 1.5, 28, 32, angles, 80.0, 40.0)
    _, sino = astra.create_sino3d_gpu(vol, pg, vg)

    ray_spec, knot_spec = xtk.from_astra(pg, vg, Array=Array3f)
    p = np.asarray(xtk.xrt_apply(ray_spec, knot_spec, 1, Float(vol_astra_to_xtk(vol))))
    p_gt = sino.ravel()
    assert np.abs(p - p_gt).max() < 1e-2 * np.abs(p_gt).max()
