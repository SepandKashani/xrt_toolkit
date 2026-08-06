"""
Tests for the reconstruction algorithms in ``xrt_toolkit.optim``.

Requires a CUDA device (the drjit backend of the library).
"""

import numpy as np
import pytest

drjit = pytest.importorskip("drjit")
import drjit as dr  # noqa: E402

try:
    from drjit.cuda.ad import Float  # noqa: E402

    dr.width(Float(0.0))
    CUDA_OK = True
except Exception:
    CUDA_OK = False

pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA backend unavailable")

import xrt_toolkit as xtk  # noqa: E402

N = 128


def _phantom():
    yy, xx = np.mgrid[:N, :N]
    r2 = (xx - N / 2 + 0.5) ** 2 + (yy - N / 2 + 0.5) ** 2
    f = (r2 < 40**2).astype(np.float32)
    f += 0.6 * (((xx - 80) ** 2 + (yy - 56) ** 2) < 8**2)
    return f, r2


def _parallel_scan(n_ang=180, n_det=192):
    knot = xtk.UniformSpec(start=(-N / 2 + 0.5,) * 2, step=1, num=(N, N))
    angles = dr.linspace(Float, 0, np.pi, n_ang, endpoint=False)
    det = xtk.DetectorSpec(size=(1.5 * N,), num_cell=(n_det,))
    return xtk.parallel_beam(angles, det), knot


def _psnr(rec, ref):
    rec = np.asarray(rec).reshape(ref.shape)
    return 10 * np.log10(ref.max() ** 2 / np.mean((rec - ref) ** 2))


def test_cg():
    ph, _ = _phantom()
    rays, knot = _parallel_scan()
    y = xtk.xrt_struct_apply(rays, knot, 0, Float(ph.reshape(-1)))
    A = lambda v: xtk.xrt_struct_apply(rays, knot, 0, v)
    At = lambda v: xtk.xrt_struct_adjoint(rays, knot, 0, v)
    rec = xtk.cg(A, At, y, N * N, n_iter=30)
    assert _psnr(rec, ph) > 30


def test_gd():
    ph, _ = _phantom()
    rays, knot = _parallel_scan()
    y = xtk.xrt_struct_apply(rays, knot, 0, Float(ph.reshape(-1)))
    A = lambda v: xtk.xrt_struct_apply(rays, knot, 0, v)
    At = lambda v: xtk.xrt_struct_adjoint(rays, knot, 0, v)
    rec = xtk.gd(A, At, y, N * N, n_iter=150)
    assert _psnr(rec, ph) > 25


def test_fbp_2d():
    # PSNR plus quantitative accuracy: values must be right, not just shapes.
    ph, r2 = _phantom()
    rays, knot = _parallel_scan()
    y = xtk.xrt_struct_apply(rays, knot, 0, Float(ph.reshape(-1)))
    rec = np.asarray(xtk.fbp(rays, knot, y)).reshape(N, N)
    assert _psnr(rec, ph) > 25
    inside = rec[(r2 < 30**2) & (ph < 1.5)]
    assert abs(inside.mean() - 1.0) < 0.03


def test_fbp_3d():
    N3 = 32
    zz, yy, xx = np.mgrid[:N3, :N3, :N3]
    ph = (((xx - 16) ** 2 + (yy - 16) ** 2 + (zz - 16) ** 2) < 10**2).astype(np.float32)
    knot = xtk.UniformSpec(start=(-N3 / 2 + 0.5,) * 3, step=1, num=(N3,) * 3)
    angles = dr.linspace(Float, 0, np.pi, 90, endpoint=False)
    det = xtk.DetectorSpec(size=(1.5 * N3, 1.5 * N3), num_cell=(48, 48))
    rays = xtk.parallel_beam(angles, det)
    y = xtk.xrt_struct_apply(rays, knot, 0, Float(ph.reshape(-1)))
    rec = np.asarray(xtk.fbp(rays, knot, y)).reshape(N3, N3, N3)
    assert _psnr(rec, ph) > 22
    assert abs(rec[ph > 0.5].mean() - 1.0) < 0.08


def test_fbp_cone():
    ph, r2 = _phantom()
    sod, sdd = 1.6 * N, 2.8 * N
    knot = xtk.UniformSpec(start=(-N / 2 + 0.5,) * 2, step=1, num=(N, N))
    angles = dr.linspace(Float, 0, 2 * np.pi, 360, endpoint=False)
    det = xtk.DetectorSpec(size=(2.2 * N,), num_cell=(256,))
    rays = xtk.cone_beam(sod=sod, sdd=sdd, angles=angles, detector_spec=det)
    y = xtk.xrt_struct_apply(rays, knot, 0, Float(ph.reshape(-1)))
    rec = np.asarray(xtk.fbp_cone(rays, knot, y, sod=sod, sdd=sdd)).reshape(N, N)
    assert _psnr(rec, ph) > 22
    inside = rec[(r2 < 25**2) & (ph < 1.5)]
    assert abs(inside.mean() - 1.0) < 0.06


def test_bpf_2d():
    ph, r2 = _phantom()
    rays, knot = _parallel_scan()
    y = xtk.xrt_struct_apply(rays, knot, 0, Float(ph.reshape(-1)))
    rec = np.asarray(xtk.bpf(rays, knot, y)).reshape(N, N)
    assert _psnr(rec, ph) > 20
    inside = rec[(r2 < 30**2) & (ph < 1.5)]
    assert abs(inside.mean() - 1.0) < 0.10
