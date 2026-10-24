import drjit as dr
import numpy as np
import pytest

import xrt_toolkit as xtk

try:
    from drjit.cuda import Float, Float64

    dr.width(Float(0))
except Exception:
    from drjit.llvm import Float, Float64


def matrix_problem(rng, m: int = 40, n: int = 25, cond: float = 2.0) -> tuple:
    # A (m, n) with singular values in [1, cond], as Dr.Jit operators; y; least-squares f
    U, _ = np.linalg.qr(rng.standard_normal((m, n)))
    V, _ = np.linalg.qr(rng.standard_normal((n, n)))
    M = U @ np.diag(np.linspace(1, cond, n)) @ V.T
    y = rng.standard_normal(m)
    A = lambda f: Float64(M @ np.asarray(f))  # noqa: E731
    At = lambda r: Float64(M.T @ np.asarray(r))  # noqa: E731
    return A, At, Float64(y), np.linalg.lstsq(M, y, rcond=None)[0]


@pytest.mark.parametrize("solver", ["cg", "gd"])
def test_least_squares(solver):
    # both solvers reach the least-squares solution of a small dense problem
    A, At, y, f_gt = matrix_problem(np.random.default_rng(0))
    if solver == "cg":
        f = xtk.cg(A, At, y, len(f_gt), n_iter=len(f_gt))
    else:
        f = xtk.gd(A, At, y, len(f_gt), n_iter=300)
    assert np.abs(np.asarray(f) - f_gt).max() < 1e-8 * np.abs(f_gt).max()


@pytest.mark.parametrize("solver", ["cg", "gd"])
def test_warm_start(solver):
    # started at the solution, the solvers stay there
    A, At, y, f_gt = matrix_problem(np.random.default_rng(1))
    fn = getattr(xtk, solver)
    f = fn(A, At, y, len(f_gt), n_iter=5, x0=Float64(f_gt))
    assert np.abs(np.asarray(f) - f_gt).max() < 1e-10 * np.abs(f_gt).max()


def test_cg_reconstruction():
    # a 2D phantom, parallel beam, XTK's operator pair
    N = 64
    knot_spec = xtk.UniformSpec.centered(step=1, num=(N, N))
    angles = Float(np.linspace(0, np.pi, 90, endpoint=False, dtype=np.float32))
    ray_spec = xtk.parallel_beam(angles, xtk.util.DetectorSpec(size=96, num_cell=96))
    x = np.arange(N) - (N - 1) / 2
    X, Y = np.meshgrid(x, x, indexing="ij")
    f_gt = (X**2 + Y**2 < 25**2) + 0.5 * ((X - 8) ** 2 + (Y + 6) ** 2 < 6**2)
    f_gt = f_gt.astype(np.float32).ravel()

    A = lambda f: xtk.xrt_struct_apply(ray_spec, knot_spec, 0, f)  # noqa: E731
    At = lambda y: xtk.xrt_struct_adjoint(ray_spec, knot_spec, 0, y)  # noqa: E731
    f = np.asarray(xtk.cg(A, At, A(Float(f_gt)), N * N, n_iter=50))
    assert np.sqrt(np.mean((f - f_gt) ** 2)) < 0.02


def disc_scan(
    D: int, N: int, n_view: int, n_det: int, size: float, step: float = 1
) -> tuple:
    # parallel-beam projections of a disc (2D) or ball (3D) of value 1 and radius N/3 (voxels)
    knot_spec = xtk.UniformSpec.centered(step=step, num=(N,) * D)
    angles = Float(np.linspace(0, np.pi, n_view, endpoint=False, dtype=np.float32))
    detector = xtk.util.DetectorSpec(
        size=(size * step,) * (D - 1), num_cell=(n_det,) * (D - 1)
    )
    ray_spec = xtk.parallel_beam(angles, detector)
    x = np.arange(N) - (N - 1) / 2
    r = np.sqrt(sum(c**2 for c in np.meshgrid(*(x,) * D, indexing="ij")))
    f = (r < N / 3).astype(np.float32)
    y = xtk.xrt_struct_apply(ray_spec, knot_spec, 0, Float(f.ravel()))
    return ray_spec, knot_spec, y, r


@pytest.mark.parametrize("window", ["hann", "hamming", "cosine", "shepp-logan", "ramp"])
def test_fbp_2d(window):
    # values are right inside the disc, close to 0 outside it (sharp windows ring more)
    N = 128
    ray_spec, knot_spec, y, r = disc_scan(2, N, 180, 192, 192)
    f = np.asarray(xtk.fbp(ray_spec, knot_spec, y, window)).reshape(N, N)
    assert abs(f[r < N / 4].mean() - 1) < 0.01
    assert np.sqrt(np.mean(f[r > N / 2.5] ** 2)) < 0.02


@pytest.mark.parametrize("D", [2, 3])
def test_fbp_step(D):
    # with a lattice step other than 1, the result is coefficients, as xrt_apply's data
    N = 48
    ray_spec, knot_spec, y, r = disc_scan(D, N, 90, 72, 1.5 * N, step=2.0)
    f = np.asarray(xtk.fbp(ray_spec, knot_spec, y)).reshape((N,) * D)
    assert abs(f[r < N / 4].mean() - 1) < 0.02


@pytest.mark.parametrize("n_det", [48, 192])
def test_fbp_3d(n_det):
    # each detector row reconstructs one slice (also with 4 detector cells per voxel)
    N = 48
    ray_spec, knot_spec, y, r = disc_scan(3, N, 90, n_det, 1.5 * N)
    f = np.asarray(xtk.fbp(ray_spec, knot_spec, y)).reshape(N, N, N)
    assert abs(f[r < N / 4].mean() - 1) < 0.02


def test_fbp_fine_detector():
    # with noisy data, a detector finer than the lattice is not worse than a matched one
    N = 64
    err = []
    for n_det in (96, 384):
        ray_spec, knot_spec, y, r = disc_scan(2, N, 180, n_det, 96)
        y = np.asarray(y)
        noise = 0.05 * y.max() * np.random.default_rng(0).standard_normal(len(y))
        f = xtk.fbp(ray_spec, knot_spec, Float((y + noise).astype(np.float32)))
        f = np.asarray(f).reshape(N, N)
        err.append(np.sqrt(np.mean((f - (r < N / 3))[r < N / 2.2] ** 2)))
    assert err[1] < 1.1 * err[0]


def test_fbp_window_error():
    ray_spec, knot_spec, y, _ = disc_scan(2, 16, 8, 24, 24)
    with pytest.raises(ValueError):
        xtk.fbp(ray_spec, knot_spec, y, window="box")


def test_fbp_numpy(monkeypatch):
    # FFTs with NumPy on the host give the same result as with CuPy on the GPU
    import xrt_toolkit.optim as optim

    ray_spec, knot_spec, y, _ = disc_scan(2, 64, 90, 96, 96)
    f = np.asarray(xtk.fbp(ray_spec, knot_spec, y))
    monkeypatch.setattr(optim, "_xp", lambda Float: np)
    f_np = np.asarray(xtk.fbp(ray_spec, knot_spec, y))
    assert np.abs(f - f_np).max() < 1e-5 * np.abs(f).max()


def test_fbp_owns_memory():
    # the result stays right after CuPy reuses the memory of its temporaries
    import xrt_toolkit.optim as optim

    cupy = optim._cupy() if (dr.backend_v(Float) == dr.JitBackend.CUDA) else None
    if cupy is None:
        pytest.skip("no CuPy")
    ray_spec, knot_spec, y, _ = disc_scan(2, 64, 90, 96, 96)
    f_ref = np.asarray(xtk.fbp(ray_spec, knot_spec, y))
    f = xtk.fbp(ray_spec, knot_spec, y)  # not evaluated yet
    junk = [cupy.full(90 * 96 + k, -777.0, dtype=cupy.float32) for k in range(8)]
    assert np.array_equal(np.asarray(f), f_ref)
    del junk


def ball_cone_scan(D: int, N: int, n_det: int, step: float = 1) -> tuple:
    # cone-beam projections (fan beam in 2D) of a disc/ball of value 1 and radius 0.3 N (voxels)
    knot_spec = xtk.UniformSpec.centered(step=step, num=(N,) * D)
    angles = Float(np.linspace(0, 2 * np.pi, 360, endpoint=False, dtype=np.float32))
    size = (2.2 * N * step, 1.75 * N * step)[: D - 1]
    detector = xtk.util.DetectorSpec(size=size, num_cell=(n_det,) * (D - 1))
    ray_spec = xtk.cone_beam(1.6 * N * step, 2.8 * N * step, angles, detector)
    x = np.arange(N) - (N - 1) / 2
    X = np.meshgrid(*(x,) * D, indexing="ij")
    f = (sum(c**2 for c in X) < (0.3 * N) ** 2).astype(np.float32)
    y = xtk.xrt_struct_apply(ray_spec, knot_spec, 0, Float(f.ravel()))
    return ray_spec, knot_spec, y, X


def test_fdk_2d():
    # fan beam: values are right inside the disc, close to 0 outside it
    N = 128
    ray_spec, knot_spec, y, (X, Y) = ball_cone_scan(2, N, 256)
    f = np.asarray(xtk.fdk(ray_spec, knot_spec, y)).reshape(N, N)
    r = np.sqrt(X**2 + Y**2)
    assert abs(f[r < 0.2 * N].mean() - 1) < 0.002
    assert np.sqrt(np.mean(f[(r > 0.35 * N) & (r < 0.45 * N)] ** 2)) < 0.02


def test_fdk_3d():
    # values are right at the center, in the orbit plane, and away from it
    N = 64
    ray_spec, knot_spec, y, (X, Y, Z) = ball_cone_scan(3, N, 128)
    f = np.asarray(xtk.fdk(ray_spec, knot_spec, y)).reshape(N, N, N)
    for cx, cz, tol in ((0, 0, 0.004), (0.15 * N, 0, 0.006), (0, 0.15 * N, 0.02)):
        m = (X - cx) ** 2 + Y**2 + (Z - cz) ** 2 < (0.07 * N) ** 2
        assert abs(f[m].mean() - 1) < tol


@pytest.mark.parametrize("D", [2, 3])
def test_fdk_step(D):
    # with a lattice step other than 1, the result is coefficients, as xrt_apply's data
    N = 48
    ray_spec, knot_spec, y, X = ball_cone_scan(D, N, 96, step=2.0)
    f = np.asarray(xtk.fdk(ray_spec, knot_spec, y)).reshape((N,) * D)
    r = np.sqrt(sum(c**2 for c in X))
    assert abs(f[r < 0.2 * N].mean() - 1) < 0.02
