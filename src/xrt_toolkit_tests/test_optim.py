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
