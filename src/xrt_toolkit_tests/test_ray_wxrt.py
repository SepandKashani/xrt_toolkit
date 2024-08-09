import numpy as np
import pytest

import xrt_toolkit.util as xtk_util
import xrt_toolkit_tests.conftest as ct
import xrt_toolkit_tests.test_ray_xrt as ct_xrt
from xrt_toolkit.ray_wxrt import RayWXRT


class TestRayWXRT(ct_xrt.TestRayXRT):
    @pytest.mark.parametrize("stack_shape", [(), (1,), (5, 3, 4)])
    def test_value_apply(self, op, dtype, ntw_spec, stack_shape):
        # output value matches ground truth.
        translate = xtk_util.TranslateDType(dtype)
        fdtype = translate.to_float()

        # Generate RayWXRT input/output (ground-truth)
        rng = np.random.default_rng()
        a = rng.standard_normal((*stack_shape, *op.cfg.N))  # (..., N1,...,ND)
        a = a.astype(fdtype)
        w = ntw_spec[2]  # (N1,...,ND)
        b_gt = []
        for axis in range(op.cfg.D):
            # Compute accumulated attenuation
            pad_width = [(0, 0)] * op.cfg.D
            pad_width[axis] = (1, 0)
            selector = [slice(None)] * op.cfg.D
            selector[axis] = slice(0, -1)
            _w = np.pad(w, pad_width)[tuple(selector)]

            A = np.exp(-op.cfg.pitch[axis] * np.cumsum(_w, axis=axis))
            B = np.where(
                np.isclose(w, 0),
                op.cfg.pitch[axis],
                (1 - np.exp(-w * op.cfg.pitch[axis])) / w,
            )

            p = np.sum(a * A * B, axis=len(stack_shape) + axis)
            b_gt.append(p.reshape(*stack_shape, -1))
        b_gt = np.concatenate(b_gt, axis=-1)  # (..., N_ray)

        # Test RayWXRT compliance
        b = op.apply(a)
        assert b.shape == b_gt.shape
        assert ct.allclose(b, b_gt, np.single)

    # Fixtures ----------------------------------------------------------------
    @pytest.fixture
    def ntw_spec(self, origin, pitch, N) -> tuple[np.ndarray]:
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

        # To avoid numerical inaccuracies in computing the ground-truth [due to use of np.exp()],
        # we limit the range of valid `w`.
        rng = np.random.default_rng()
        w_spec = np.linspace(0.5, 1, np.prod(N), endpoint=True)
        w_spec *= rng.choice([-1, 1], size=w_spec.shape)
        w_spec = w_spec.reshape(N)

        return n_spec, t_spec, w_spec

    @pytest.fixture
    def op(self, origin, pitch, N, ntw_spec, ndi) -> RayWXRT:
        xp = ndi.module()
        n_spec = xp.asarray(ntw_spec[0])
        t_spec = xp.asarray(ntw_spec[1])
        w_spec = xp.asarray(ntw_spec[2])
        return RayWXRT(
            origin=origin,
            pitch=pitch,
            N=N,
            n_spec=n_spec,
            t_spec=t_spec,
            w_spec=w_spec,
        )
