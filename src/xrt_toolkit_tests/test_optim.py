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
    re = xtk.struct_rays(rays)
    y = xtk.xrt_apply(re, knot, 0, Float(ph.reshape(-1)))
    A = lambda v: xtk.xrt_apply(re, knot, 0, v)
    At = lambda v: xtk.xrt_adjoint(re, knot, 0, v)
    rec = xtk.cg(A, At, y, N * N, n_iter=30)
    assert _psnr(rec, ph) > 30


def test_gd():
    ph, _ = _phantom()
    rays, knot = _parallel_scan()
    re = xtk.struct_rays(rays)
    y = xtk.xrt_apply(re, knot, 0, Float(ph.reshape(-1)))
    A = lambda v: xtk.xrt_apply(re, knot, 0, v)
    At = lambda v: xtk.xrt_adjoint(re, knot, 0, v)
    rec = xtk.gd(A, At, y, N * N, n_iter=150)
    assert _psnr(rec, ph) > 25


def test_struct_rays_matches_struct_apply():
    # expanding a structured scan must reproduce the structured operator
    # exactly (same rays, same projection-major ordering)
    ph, _ = _phantom()
    f = Float(ph.reshape(-1))
    rays, knot = _parallel_scan()
    y_s = np.asarray(xtk.xrt_struct_apply(rays, knot, 1, f))
    y_e = np.asarray(xtk.xrt_apply(xtk.struct_rays(rays), knot, 1, f))
    assert np.abs(y_s - y_e).max() < 1e-4 * np.abs(y_s).max()

    cone = xtk.cone_beam(sod=1.6 * N, sdd=2.8 * N,
                         angles=dr.linspace(Float, 0, 2 * np.pi, 60, endpoint=False),
                         detector_spec=xtk.DetectorSpec(size=(2.2 * N,), num_cell=(64,)))
    y_s = np.asarray(xtk.xrt_struct_apply(cone, knot, 0, f))
    y_e = np.asarray(xtk.xrt_apply(xtk.struct_rays(cone), knot, 0, f))
    assert np.abs(y_s - y_e).max() < 1e-4 * np.abs(y_s).max()


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


def test_fbp_windows():
    # every window preserves the quantitative scale; sharper windows resolve
    # the disc edge better on noiseless data
    ph, r2 = _phantom()
    rays, knot = _parallel_scan()
    y = xtk.xrt_apply(xtk.struct_rays(rays), knot, 0, Float(ph.reshape(-1)))
    for w in ("ramp", "shepp-logan", "cosine", "hamming", "hann"):
        rec = np.asarray(xtk.fbp(rays, knot, y, window=w)).reshape(N, N)
        inside = rec[(r2 < 30**2) & (ph < 1.5)]
        assert abs(inside.mean() - 1.0) < 0.03, w
    with pytest.raises(ValueError):
        xtk.fbp(rays, knot, y, window="bogus")


def test_optim_numpy_fallback():
    # without CuPy the filtering runs on the host and must agree with the
    # GPU path to fp32 round-off
    import xrt_toolkit.optim as opt
    if opt._cp is None:
        pytest.skip("already on the NumPy path")
    ph, _ = _phantom()
    rays, knot = _parallel_scan()
    y = xtk.xrt_apply(xtk.struct_rays(rays), knot, 0, Float(ph.reshape(-1)))
    ref = np.asarray(xtk.fbp(rays, knot, y))
    saved, opt._cp = opt._cp, None
    opt._filter_cache.clear()
    try:
        alt = np.asarray(xtk.fbp(rays, knot, y))
    finally:
        opt._cp = saved
        opt._filter_cache.clear()
    assert np.abs(ref - alt).max() < 1e-4


def _cone3d_scan(N=96, n_ang=360):
    sod, sdd = 1.6 * N, 2.8 * N
    knot = xtk.UniformSpec(start=(-N / 2 + 0.5,) * 3, step=1, num=(N,) * 3)
    rays = xtk.cone_beam(sod=sod, sdd=sdd,
                         angles=dr.linspace(Float, 0, 2 * np.pi, n_ang, endpoint=False),
                         detector_spec=xtk.DetectorSpec(size=(2.2 * N, 1.75 * N),
                                                        num_cell=(216, 168)))
    return rays, knot, sod, sdd


def _ball_means(rec, N):
    zz, yy, xx = np.mgrid[:N, :N, :N].astype(np.float32)
    rec = np.asarray(rec).reshape(N, N, N)
    out = []
    for dz, dx in ((0, 0), (0, 0.18 * N), (0.18 * N, 0)):  # center, in-plane, off-plane
        m = ((xx - N / 2 + .5 - dx) ** 2 + (yy - N / 2 + .5) ** 2
             + (zz - N / 2 + .5 - dz) ** 2) < (0.07 * N) ** 2
        out.append(float(rec[m].mean()))
    return out


def test_fbp_cone_3d_fdk():
    N = 96
    zz, yy, xx = np.mgrid[:N, :N, :N].astype(np.float32)
    ph = (((xx - N/2 + .5) ** 2 + (yy - N/2 + .5) ** 2 + (zz - N/2 + .5) ** 2)
          < (0.3 * N) ** 2).astype(np.float32)
    rays, knot, sod, sdd = _cone3d_scan(N)
    y = xtk.xrt_apply(xtk.struct_rays(rays), knot, 0, Float(ph.reshape(-1)))
    rec = xtk.fbp_cone(rays, knot, y, sod=sod, sdd=sdd, window="ramp")
    for m, tol in zip(_ball_means(rec, N), (0.03, 0.03, 0.04)):
        assert abs(m - 1.0) < tol


def test_fbp_cone_bpf_raises():
    # image-domain deconvolution needs a shift-invariant blur: parallel only
    N2 = 64
    knot = xtk.UniformSpec(start=(-N2 / 2 + 0.5,) * 2, step=1, num=(N2, N2))
    cone = xtk.cone_beam(sod=1.6 * N2, sdd=2.8 * N2,
                         angles=dr.linspace(Float, 0, 2 * np.pi, 8, endpoint=False),
                         detector_spec=xtk.DetectorSpec(size=(2.2 * N2,), num_cell=(32,)))
    with pytest.raises(NotImplementedError):
        xtk.fbp_cone(cone, knot, dr.zeros(Float, 8 * 32), sod=1.6 * N2,
                     sdd=2.8 * N2, method="bpf")
    knot3 = xtk.UniformSpec(start=(-N2 / 2 + 0.5,) * 3, step=1, num=(N2,) * 3)
    cone3 = xtk.cone_beam(sod=1.6 * N2, sdd=2.8 * N2,
                          angles=dr.linspace(Float, 0, 2 * np.pi, 8, endpoint=False),
                          detector_spec=xtk.DetectorSpec(size=(2.2 * N2, 1.6 * N2),
                                                         num_cell=(32, 24)))
    with pytest.raises(NotImplementedError):
        xtk.fbp_cone(cone3, knot3, dr.zeros(Float, 8 * 32 * 24), sod=1.6 * N2,
                     sdd=2.8 * N2, method="bpf")


def test_bpf_3d_anisotropic_detector():
    # the axial detector spacing enters the backprojection density (1/du2)
    N = 64
    zz, yy, xx = np.mgrid[:N, :N, :N].astype(np.float32)
    r2 = (xx - N/2 + .5) ** 2 + (yy - N/2 + .5) ** 2 + (zz - N/2 + .5) ** 2
    ph = (r2 < (0.3 * N) ** 2).astype(np.float32)
    knot = xtk.UniformSpec(start=(-N / 2 + 0.5,) * 3, step=1, num=(N,) * 3)
    rays = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, 180, endpoint=False),
                             xtk.DetectorSpec(size=(1.5 * N, 1.5 * N), num_cell=(96, 64)))
    y = xtk.xrt_apply(xtk.struct_rays(rays), knot, 0, Float(ph.reshape(-1)))
    rec = np.asarray(xtk.bpf(rays, knot, y)).reshape(N, N, N)
    inside = rec[r2 < (0.2 * N) ** 2]
    assert abs(inside.mean() - 1.0) < 0.10
