"""
Tests for the fused multi-channel (tensor) operators in
``xrt_toolkit.drjit.tensor_xrt``.

Requires a CUDA device (the drjit backend of the library).
"""

import numpy as np
import pytest

drjit = pytest.importorskip("drjit")
import drjit as dr  # noqa: E402

try:
    from drjit.cuda.ad import Array2f, Array3f, Float  # noqa: E402

    dr.width(Float(0.0))  # touch the backend
    CUDA_OK = True
except Exception:
    CUDA_OK = False

pytestmark = pytest.mark.skipif(not CUDA_OK, reason="CUDA backend unavailable")

import xrt_toolkit as xtk  # noqa: E402
import xrt_toolkit.drjit.ray_xrt as sx  # noqa: E402
import xrt_toolkit.drjit.tensor_xrt as tx  # noqa: E402


def _rays(D, N, L, rng):
    t = rng.uniform(4, N - 4, (D, L)).astype(np.float32)
    n = rng.normal(size=(D, L)).astype(np.float32)
    n /= np.linalg.norm(n, axis=0)
    A = Array2f if D == 2 else Array3f
    return (A(t), A(n))


def _maxrel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b).max() / (np.abs(a).max() + 1e-12)


@pytest.mark.parametrize("D,N", [(2, 64), (3, 24)])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_c1_reduces_to_scalar(D, N, order):
    # With C=1 and unit weight, the fused pair must reproduce the scalar pair.
    rng = np.random.default_rng(0)
    L = 200
    knot = xtk.UniformSpec(start=0, step=1, num=(N,) * D)
    ray = _rays(D, N, L, rng)
    c = Float(rng.random(N**D, dtype=np.float32))
    r = Float(rng.random(L, dtype=np.float32))

    y_s = sx.xrt_apply(ray, knot, order, c, mode="evaluated")
    y_t = tx.xrt_tensor_apply(ray, knot, order, [Float(1.0)], c, mode="evaluated")
    assert _maxrel(y_s, y_t) < 1e-6

    b_s = sx.xrt_adjoint(ray, knot, order, r, mode="evaluated")
    b_t = tx.xrt_tensor_adjoint(ray, knot, order, [Float(1.0)], r, mode="evaluated")
    assert _maxrel(b_s, b_t) < 1e-6


@pytest.mark.parametrize("D,N", [(2, 64), (3, 24)])
@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize("mode", ["symbolic", "evaluated"])
def test_fused_matches_composition(D, N, order, mode):
    # The fused traversal must agree with the channel-by-channel composition
    # of the scalar operators, for LRT weights (C = 3 in 2D, 6 in 3D).
    rng = np.random.default_rng(1)
    L = 200
    knot = xtk.UniformSpec(start=0, step=1, num=(N,) * D)
    ray = _rays(D, N, L, rng)
    w = tx.lrt_weights(ray[1])
    C = len(w)
    packed = tx.pack_channels(
        [Float(rng.random(N**D, dtype=np.float32)) for _ in range(C)])
    r = Float(rng.random(L, dtype=np.float32))

    y_f = tx.xrt_tensor_apply(ray, knot, order, w, packed, mode=mode)
    y_r = tx.xrt_tensor_apply_ref(ray, knot, order, w, packed, mode=mode)
    assert _maxrel(y_f, y_r) < 1e-5

    b_f = tx.xrt_tensor_adjoint(ray, knot, order, w, r, mode=mode)
    b_r = tx.xrt_tensor_adjoint_ref(ray, knot, order, w, r, mode=mode)
    assert _maxrel(b_f, b_r) < 1e-5


@pytest.mark.parametrize("D,N", [(2, 64), (3, 24)])
@pytest.mark.parametrize("order", [0, 1, 2])
def test_adjoint_dot(D, N, order):
    # <A f, r> == <f, A^T r>, normalized by the cancellation-free scale
    # sum_c |<w_c P f_c, r>| (signed weights can shrink |<A f, r>| itself).
    rng = np.random.default_rng(2)
    L = 200
    knot = xtk.UniformSpec(start=0, step=1, num=(N,) * D)
    ray = _rays(D, N, L, rng)
    w = tx.lrt_weights(ray[1])
    C = len(w)
    chans = [Float(rng.random(N**D, dtype=np.float32)) for _ in range(C)]
    packed = tx.pack_channels(chans)
    r = Float(rng.random(L, dtype=np.float32))

    y = tx.xrt_tensor_apply(ray, knot, order, w, packed, mode="evaluated")
    b = tx.xrt_tensor_adjoint(ray, knot, order, w, r, mode="evaluated")
    lhs = float(np.dot(np.asarray(y), np.asarray(r)))
    rhs = float(np.dot(np.asarray(packed), np.asarray(b)))
    scale = sum(
        abs(float(np.dot(
            np.asarray(w[c]) * np.asarray(
                sx.xrt_apply(ray, knot, order, chans[c], mode="evaluated")),
            np.asarray(r))))
        for c in range(C))
    assert abs(lhs - rhs) / scale < 1e-5


def test_tof_composes():
    # The TOF option weights the fused pair identically to the composition.
    rng = np.random.default_rng(3)
    D, N, L = 3, 24, 200
    knot = xtk.UniformSpec(start=0, step=1, num=(N,) * D)
    ray = _rays(D, N, L, rng)
    w = tx.lrt_weights(ray[1])
    packed = tx.pack_channels(
        [Float(rng.random(N**D, dtype=np.float32)) for _ in range(len(w))])
    tof = xtk.TOFSpec(center=Float(np.full(L, N / 2, np.float32)), sigma=5.0)
    y_f = tx.xrt_tensor_apply(ray, knot, 0, w, packed, mode="evaluated", tof=tof)
    y_r = tx.xrt_tensor_apply_ref(ray, knot, 0, w, packed, mode="evaluated", tof=tof)
    assert _maxrel(y_f, y_r) < 1e-6


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(4)
    chans = [Float(rng.random(100, dtype=np.float32)) for _ in range(5)]
    out = tx.unpack_channels(tx.pack_channels(chans), 5)
    for a, b in zip(chans, out):
        assert _maxrel(a, b) == 0.0
