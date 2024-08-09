import numpy as np
import pytest

import xrt_toolkit.util as xtk_util
import xrt_toolkit_tests.conftest as ct
from xrt_toolkit.array_module import CUPY_ENABLED, NDArrayInfo
from xrt_toolkit.ray_xrt import RayXRT


class TestRayXRT:
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, dtype, stack_shape):
        # output value matches ground truth.
        translate = xtk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        # Generate RayXRT input/output (ground-truth)
        rng = np.random.default_rng()
        a = rng.standard_normal((*stack_shape, *op.cfg.N))  # (..., N1,...,ND)
        a = a.astype(fdtype)
        b_gt = []
        for axis in range(op.cfg.D):
            p = np.sum(a * op.cfg.pitch[axis], axis=len(stack_shape) + axis)
            b_gt.append(p.reshape(*stack_shape, -1))
        b_gt = np.concatenate(b_gt, axis=-1)  # (..., N_ray)

        # Test RayXRT compliance
        b = op.apply(a)
        assert b.shape == b_gt.shape
        assert ct.allclose(b, b_gt, np.single)

    @pytest.mark.parametrize("direction", ["apply", "adjoint"])
    def test_prec(self, op, dtype, direction):
        # output precision is always FP32, i.e. RayXRT's internal precision.
        translate = xtk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        rng = np.random.default_rng()
        if direction == "apply":
            a = rng.standard_normal(op.cfg.N).astype(fdtype)
        else:
            a = rng.standard_normal(op.cfg.N_ray).astype(fdtype)

        f = getattr(op, direction)
        b = f(a)
        assert b.dtype == np.dtype(op.fdtype)

    def test_math_adjoint(self, op, dtype):
        # <A x, y> == <x, A^H y>
        translate = xtk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        sh = (5, 3, 4)
        rng = np.random.default_rng()
        a = rng.standard_normal((*sh, *op.cfg.N))
        a = a.astype(fdtype)
        b = rng.standard_normal((*sh, op.cfg.N_ray))
        b = b.astype(fdtype)

        lhs = ct.inner_product(op.apply(a), b, 1)
        rhs = ct.inner_product(a, op.adjoint(b), op.cfg.D)
        assert ct.allclose(lhs, rhs, op.fdtype)

    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    @pytest.mark.parametrize("direction", ["apply", "adjoint"])
    def test_cupy(
        self,
        origin,
        pitch,
        N,
        nt_spec,
        dtype,
        # -----------------------------
        stack_shape,
        direction,
    ):
        # CuPy backend produces same results as NumPy backend.
        ndi = NDArrayInfo.CUPY
        if not CUPY_ENABLED:
            pytest.skip(f"Unsupported backend {ndi}.")
        cp = ndi.module()

        op_gt = RayXRT(
            origin=origin,
            pitch=pitch,
            N=N,
            n_spec=nt_spec[0],
            t_spec=nt_spec[1],
        )
        op_cp = RayXRT(
            origin=origin,
            pitch=pitch,
            N=N,
            n_spec=cp.asarray(nt_spec[0]),
            t_spec=cp.asarray(nt_spec[1]),
        )

        rng = np.random.default_rng()
        if direction == "apply":
            a = rng.standard_normal((*stack_shape, *op_gt.cfg.N), dtype=dtype)
            b_gt = op_gt.apply(a)
            b_cp = op_cp.apply(cp.asarray(a))
        else:  # "adjoint"
            a = rng.standard_normal((*stack_shape, op_gt.cfg.N_ray), dtype)
            b_gt = op_gt.adjoint(a)
            b_cp = op_cp.adjoint(cp.asarray(a))

        assert NDArrayInfo.from_obj(b_cp) == ndi
        assert b_cp.shape == b_gt.shape
        assert b_cp.dtype == b_gt.dtype
        assert ct.allclose(b_cp.get(), b_gt, op_gt.fdtype)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture(params=[2, 3])
    def space_dim(self, request) -> int:
        # space dimension D
        return request.param

    @pytest.fixture
    def origin(self, space_dim) -> tuple[float]:
        # Volume origin
        rng = np.random.default_rng()
        orig = rng.standard_normal(space_dim)
        return tuple(orig)

    @pytest.fixture
    def pitch(self, space_dim) -> tuple[float]:
        # Voxel pitch
        rng = np.random.default_rng()
        pitch = rng.uniform(1e-3, 1, space_dim)
        return tuple(pitch)

    @pytest.fixture
    def N(self, space_dim) -> tuple[float]:
        if space_dim == 2:
            return (5, 6)
        else:
            return (5, 3, 4)

    @pytest.fixture
    def nt_spec(self, origin, pitch, N) -> tuple[np.ndarray]:
        # To analytically test XRT correctness, we cast rays only along cardinal X/Y/Z directions,
        # with one ray per voxel side.

        D = len(N)
        n_spec = []
        t_spec = []
        for axis in range(D):
            # compute axes which are not projected
            dim = list(range(D))
            dim.pop(axis)

            # number of rays per dimension
            N_ray = np.array(N)[dim]

            n = np.zeros((*N_ray, D))
            n[..., axis] = 1
            n_spec.append(n.reshape(-1, D))

            t = np.zeros((*N_ray, D))
            _t = np.meshgrid(
                *[(np.arange(N[d]) + 0.5) * pitch[d] + origin[d] for d in dim],
                indexing="ij",
            )
            _t = np.stack(_t, axis=-1)
            t[..., dim] = _t
            t_spec.append(t.reshape(-1, D))

        n_spec = np.concatenate(n_spec, axis=0)
        t_spec = np.concatenate(t_spec, axis=0)
        return n_spec, t_spec

    @pytest.fixture(
        params=[
            np.float32,
            np.float64,
        ]
    )
    def dtype(self, request) -> np.dtype:
        # FP precision of inputs.
        # Correctness tests are performed in single-precision due to drjit constraints.
        return np.dtype(request.param)

    @pytest.fixture
    def op(self, origin, pitch, N, nt_spec) -> RayXRT:
        return RayXRT(
            origin=origin,
            pitch=pitch,
            N=N,
            n_spec=nt_spec[0],
            t_spec=nt_spec[1],
        )
