r"""
Reconstruction algorithms.

The iterative solvers (:py:func:`cg`, :py:func:`gd`) take the forward and adjoint operators as
callables, so they work with any operator pair, e.g. :py:func:`~xrt_toolkit.xrt_apply` /
:py:func:`~xrt_toolkit.xrt_adjoint` or :py:func:`~xrt_toolkit.xrt_struct_apply` /
:py:func:`~xrt_toolkit.xrt_struct_adjoint`:

.. code-block:: python

   A = lambda f: xtk.xrt_apply(ray_spec, knot_spec, order, f)
   At = lambda y: xtk.xrt_adjoint(ray_spec, knot_spec, order, y)
   f = xtk.cg(A, At, y, math.prod(knot_spec.num))

The analytic method (:py:func:`fbp`) works on the structured scans of
:py:func:`~xrt_toolkit.parallel_beam`. It filters the projections with FFTs (with CuPy on the
GPU when it is installed, else with NumPy on the host), then backprojects them voxel by voxel.
"""

import functools
import itertools
import math
import typing as typ
import warnings

import drjit as dr
import numpy as np

import xrt_toolkit.util as xrtu

FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
RaySpecT = tuple
OperatorT = typ.Callable[[FloatT], FloatT]


def cg(
    A: OperatorT,
    At: OperatorT,
    y: FloatT,
    n_unknowns: int,
    n_iter: int = 30,
    x0: FloatT = None,
) -> FloatT:
    r"""
    Least squares :math:`\min_{f} \|A f - y\|_{2}^{2}`, by conjugate gradient on the normal
    equations :math:`A^{T} A f = A^{T} y`.

    Parameters
    ----------
    A, At: callable
        Forward and adjoint operators.
    y: FloatT
        Measurements.
    n_unknowns: int
        Size of :math:`f`.
    n_iter: int
        Number of iterations. With noisy or incomplete data, stopping early regularizes.
    x0: FloatT
        Initial :math:`f` (default: 0).

    Returns
    -------
    f: FloatT
        (n_unknowns,) solution.
    """
    Float = type(y)
    if x0 is None:
        f = dr.zeros(Float, n_unknowns)
        r = Float(At(y))
    else:
        f = Float(x0)
        r = At(y - A(f))
    p = Float(r)
    rs = dr.sum(r * r)
    for _ in range(n_iter):
        Ap = At(A(p))
        alpha = rs / (dr.sum(p * Ap) + 1e-30)
        f = dr.fma(alpha, p, f)
        r = dr.fma(-alpha, Ap, r)
        rs_new = dr.sum(r * r)
        p = dr.fma(rs_new / (rs + 1e-30), p, r)
        rs = rs_new
        dr.eval(f, r, p, rs)
    return f


def gd(
    A: OperatorT,
    At: OperatorT,
    y: FloatT,
    n_unknowns: int,
    n_iter: int = 100,
    step: float = None,
    x0: FloatT = None,
) -> FloatT:
    r"""
    Least squares :math:`\min_{f} \|A f - y\|_{2}^{2}`, by gradient descent (Landweber
    iteration): :math:`f \leftarrow f - \tau A^{T} (A f - y)`.

    Parameters
    ----------
    A, At: callable
        Forward and adjoint operators.
    y: FloatT
        Measurements.
    n_unknowns: int
        Size of :math:`f`.
    n_iter: int
        Number of iterations.
    step: float
        Step :math:`\tau`. Default: :math:`1.9 / \|A^{T} A\|`, with the norm estimated by 15
        power iterations (convergence needs :math:`\tau < 2 / \|A^{T} A\|`).
    x0: FloatT
        Initial :math:`f` (default: 0).

    Returns
    -------
    f: FloatT
        (n_unknowns,) solution.
    """
    Float = type(y)
    if step is None:
        v = Float(np.random.default_rng(0).random(n_unknowns))
        for _ in range(15):
            v = At(A(v))
            v_norm = dr.norm(v)
            v = v / (v_norm + 1e-30)
            dr.eval(v, v_norm)
        step = 1.9 / v_norm[0]
    f = dr.zeros(Float, n_unknowns) if (x0 is None) else Float(x0)
    for _ in range(n_iter):
        f = dr.fma(-step, At(A(f) - y), f)
        dr.eval(f)
    return f


def fbp(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    y: FloatT,
    window: str = "hann",
) -> FloatT:
    r"""
    Filtered backprojection of a parallel-beam scan.

    Parameters
    ----------
    ray_spec: RaySpecT
        Scan from :py:func:`~xrt_toolkit.parallel_beam`, with views covering :math:`[0, \pi)`
        uniformly. 2D, or 3D (each detector row reconstructs one slice).
    knot_spec: UniformSpec
        Reconstruction lattice, centered on the rotation axis.
    y: FloatT
        (L,) projections, in the order of :py:func:`~xrt_toolkit.xrt_struct_apply`.
    window: "hann" | "hamming" | "cosine" | "shepp-logan" | "ramp"
        Apodization of the ramp filter ("ramp": none).

    Returns
    -------
    f: FloatT
        (Q1,...,QD) flattened C-ordered voxel values; 0 outside the field of view (voxels that
        some view does not see).

    Notes
    -----
    With a detector finer than the lattice, the filter stops at the lattice resolution, as if
    the detector cells were binned to the voxel size: a finer detector never makes the
    reconstruction noisier, but it does not make it sharper either.
    """
    Float = type(y)
    _, _, u_spec = ray_spec
    n_view = dr.width(ray_spec[0])
    num = u_spec.num[:-1]  # (in-plane[, axial]) detector cells
    du = u_spec.step[0]

    # ramp filter along the in-plane detector axis, cut at the lattice Nyquist frequency
    xp = _xp(Float)
    q = _to_xp(y, xp).reshape(n_view, *num)
    q = xp.moveaxis(q, 1, -1)
    q = _ramp_filter(q, du, window, min(1, du / min(knot_spec.step[:2])), xp)
    q = _from_xp(xp.moveaxis(q, -1, 1), Float)

    r_fov = abs(u_spec.start[0]) + du / 2  # detector half-width
    f = _backproject(q, ray_spec, knot_spec, r_fov)
    return f * (math.pi * du / n_view)


# Fourier filtering
# -----------------
# CuPy works on the Dr.Jit CUDA arrays in place (DLPack); NumPy copies them to the host.


@functools.cache
def _cupy():
    # CuPy, if installed and able to run FFTs (its cuFFT must match the CUDA version), else None
    try:
        import cupy

        cupy.fft.rfft(cupy.ones(4, dtype=cupy.float32))
        return cupy
    except Exception as e:
        warnings.warn(
            f"CuPy unusable ({type(e).__name__}): FFTs run with NumPy, on the host.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None


def _xp(Float):
    # array module for the FFTs of `Float` arrays
    if dr.backend_v(Float) == dr.JitBackend.CUDA:
        return _cupy() or np
    return np


def _to_xp(x: FloatT, xp):
    return xp.from_dlpack(x) if (xp is not np) else np.asarray(x)


def _from_xp(a, Float) -> FloatT:
    # flat Dr.Jit array (a view of CuPy arrays, which Dr.Jit keeps alive)
    return Float(a.reshape(-1))


def _fast_len(n: int) -> int:
    # smallest 2^a 3^b 5^c >= n: a fast FFT length
    best = 1 << (n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            p2 = 1 << max(0, -(-n // p35) - 1).bit_length()
            best = min(best, p2 * p35)
            p35 *= 3
        p5 *= 5
    return best


@functools.cache
def _ramp(n: int, du: float, window: str, cutoff: float, xp) -> tuple:
    # (L, H): FFT length (padded against wrap-around) and the ramp filter's spectrum, from
    # the discrete Ram-Lak kernel; apodized by `window` up to `cutoff` x Nyquist, 0 beyond.
    L = _fast_len(2 * n)
    h = np.zeros(L)
    h[0] = 1 / (4 * du**2)
    k = np.arange(1, L // 2 + 1, 2)
    h[k] = h[-k] = -1 / (np.pi * k * du) ** 2
    H = np.fft.rfft(h).real
    w = np.minimum(np.arange(L // 2 + 1) / (L / 2) / cutoff, 1)  # 1 = cutoff
    windows = {
        "ramp": np.ones_like(w),
        "hann": 0.5 + 0.5 * np.cos(np.pi * w),
        "hamming": 0.54 + 0.46 * np.cos(np.pi * w),
        "cosine": np.cos(np.pi * w / 2),
        "shepp-logan": np.sinc(w / 2),
    }
    if window not in windows:
        raise ValueError(f"Unknown window {window!r}: one of {list(windows)}.")
    H = H * windows[window] * (np.arange(L // 2 + 1) / (L / 2) <= cutoff)
    return L, xp.asarray(H.astype(np.float32))


def _ramp_filter(
    q, du: float, window: str, cutoff: float, xp, max_bytes: int = 1 << 30
):
    # ramp-filter `q` along its last axis, in blocks of rows (padded spectra are large)
    L, H = _ramp(q.shape[-1], float(du), window, float(cutoff), xp)
    rows = q.reshape(-1, q.shape[-1])
    out = xp.empty(rows.shape, dtype=rows.dtype)
    block = max(1, max_bytes // (12 * L))
    for a in range(0, len(rows), block):
        Q = xp.fft.rfft(rows[a : a + block], n=L, axis=-1)
        out[a : a + block] = xp.fft.irfft(Q * H, n=L, axis=-1)[:, : q.shape[-1]]
    return out.reshape(q.shape)


# Backprojection
# --------------
# Voxel-driven, as in ASTRA/TIGRE FBP: each voxel sums the filtered projections, interpolated
# linearly at the detector position it projects to. (The exact adjoint xrt_adjoint() is the
# operator for iterative solvers; as an FBP backprojector its chords, nearly tangent to lattice
# planes, print line artifacts.)


def _backproject(
    q: FloatT,
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    r_fov: float,
) -> FloatT:
    # Sum over the views of `q` ((n_view, *detector) projections, flattened) at each voxel's
    # detector position; 0 for voxels farther than r_fov from the rotation axis (z).
    t_spec, _, u_spec = ray_spec
    Float = type(q)
    UInt = dr.uint32_array_t(Float)
    Int = dr.int32_array_t(Float)
    D = knot_spec.ndim
    n_view = dr.width(t_spec)
    num_u, u0, du = (u_spec.num[:-1], u_spec.start[:-1], u_spec.step[:-1])
    stride_u = (num_u[1], 1) if (D == 3) else (1,)

    # per view: detector axes and origin (columns of t_spec: t = u_1 d_1 [+ u_2 d_2] + o)
    T = np.asarray(t_spec)  # (D, D, n_view)
    axes = [[Float(T[c, k]) for c in range(D)] for k in range(D - 1)]
    origin = [Float(T[c, D - 1]) for c in range(D)]

    # voxel positions, one lane per voxel (C order)
    idx = dr.arange(UInt, math.prod(knot_spec.num))
    q_vox = []
    for d in reversed(range(D)):
        q_vox.insert(0, idx % knot_spec.num[d])
        idx = idx // knot_spec.num[d]
    x = [knot_spec.start[d] + knot_spec.step[d] * Float(q_vox[d]) for d in range(D)]

    def body(a, acc):
        v = [x[c] - dr.gather(Float, origin[c], a) for c in range(D)]
        p, i, ok = [], [], dr.full(dr.mask_t(Float), True, dr.width(acc))
        for k in range(D - 1):
            u = sum(v[c] * dr.gather(Float, axes[k][c], a) for c in range(D))
            p_k = (u - u0[k]) / du[k]
            i_k = Int(dr.floor(p_k))
            ok &= (i_k >= 0) & (i_k < num_u[k] - 1)
            p.append(p_k - Float(i_k))
            i.append(dr.clip(i_k, 0, num_u[k] - 2))
        base = a * math.prod(num_u) + sum(i_k * s for i_k, s in zip(i, stride_u))
        for c in itertools.product((0, 1), repeat=D - 1):
            w = math.prod(p[k] if c[k] else 1 - p[k] for k in range(D - 1))
            offset = UInt(base + sum(c_k * s for c_k, s in zip(c, stride_u)))
            acc = dr.fma(w, dr.gather(Float, q, offset, ok), acc)
        return a + 1, acc

    _, f = dr.while_loop(
        state=(dr.zeros(Int, dr.width(x[0])), dr.zeros(Float, dr.width(x[0]))),
        cond=lambda a, acc: a < n_view,
        body=body,
        labels=("a", "acc"),
    )
    return dr.select(x[0] ** 2 + x[1] ** 2 <= r_fov**2, f, 0)
