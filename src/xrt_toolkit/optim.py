r"""
Basic reconstruction algorithms.

Iterative solvers (:py:func:`cg`, :py:func:`gd`) take the forward/adjoint pair
as callables, so they work with any geometry interface of the library.
Analytic methods (:py:func:`fbp`, :py:func:`fbp_cone`, :py:func:`bpf`) operate
on structured scans built with :py:func:`~xrt_toolkit.parallel_beam` /
:py:func:`~xrt_toolkit.cone_beam`.

Everything runs on the GPU: backprojections go through the fused explicit-ray
kernels (:py:func:`~xrt_toolkit.struct_rays`), and Fourier filtering uses CuPy
FFTs on the same device memory as the Dr.Jit arrays (zero-copy through DLPack).
If CuPy is not installed, filtering transparently falls back to NumPy on the
host. Filters follow the standard discrete recipes (Kak & Slaney kernel built
periodically on the padded grid, real FFTs, fast transform lengths, the usual
window family) and are cached across calls.

All scaling constants assume the lattice ``step`` and detector units are the
same (the usual voxel-unit convention), angles uniformly spaced over
:math:`[0, \pi)` for parallel scans and :math:`[0, 2\pi)` for cone scans.
"""

import numpy as np

import drjit as dr

from .drjit.geometry import struct_rays
from .drjit.ray_xrt import xrt_adjoint, xrt_apply  # noqa: F401

__all__ = ["cg", "gd", "fbp", "fbp_cone", "bpf"]

try:  # probe with a real transform: a CuPy whose cuFFT library cannot be
    # loaded (mismatched CUDA versions) must degrade to the NumPy path, not
    # crash at the first reconstruction.
    import cupy as _cp
    _cp.fft.rfft(_cp.ones(4, dtype=_cp.float32))
except Exception:
    _cp = None


# ------------------------------------------------------------- iterative ----
def cg(A, At, y, n_unknowns, n_iter=30, x0=None):
    r"""
    Conjugate gradient on the normal equations :math:`A^{\top}A f = A^{\top}y`.

    Parameters
    ----------
    A, At: callable
        Forward and adjoint operators, e.g.
        ``lambda v: xrt_apply(rays, knot, order, v)`` and its adjoint.
        For structured scans expand the geometry once with
        :py:func:`~xrt_toolkit.struct_rays` and use the explicit operators;
        the structured ones launch one kernel per projection per call.
    y: FloatT
        Measurements.
    n_unknowns: int
        Size of the reconstruction vector.
    n_iter: int
        Number of iterations. The normal equations are ill-conditioned for
        noisy or incomplete data; early stopping acts as regularization.
    x0: FloatT, None
        Warm start.

    Returns
    -------
    f: FloatT
    """
    Float = type(y)
    if x0 is None:
        f = dr.zeros(Float, n_unknowns)
        r = Float(At(y))
    else:
        f = Float(x0)
        r = At(y) - At(A(f))
    p = Float(r)
    rs = dr.sum(r * r)
    for _ in range(n_iter):
        Ap = At(A(p))
        alpha = rs / (dr.sum(p * Ap) + 1e-30)
        f = dr.fma(alpha, p, f)
        r = dr.fma(-alpha, Ap, r)
        rs_new = dr.sum(r * r)
        p = dr.fma(rs_new / rs, p, r)
        rs = rs_new
        dr.eval(f, r, p, rs)
    return f


def gd(A, At, y, n_unknowns, n_iter=100, step=None, x0=None):
    r"""
    Gradient descent (Landweber iteration) on :math:`\frac12\|A f - y\|^{2}`.

    Parameters
    ----------
    step: float, None
        Step size. If ``None``, set to :math:`1.9 / \|A^{\top}A\|` estimated
        with 15 power iterations.

    Returns
    -------
    f: FloatT
    """
    Float = type(y)
    if step is None:
        v = Float(np.random.default_rng(0).random(n_unknowns, dtype=np.float32))
        for _ in range(15):
            v = At(A(v))
            nv = dr.sqrt(dr.sum(v * v))
            v = v / (nv + 1e-30)
            dr.eval(v, nv)
        step = 1.9 / float(nv.item())
    f = dr.zeros(Float, n_unknowns) if x0 is None else Float(x0)
    for _ in range(n_iter):
        g = At(A(f) - y)
        f = dr.fma(-step, g, f)
        dr.eval(f)
    return f


# ----------------------------------------------------- GPU FFT machinery ----
def _dev(x):
    """Dr.Jit array -> device array (CuPy view, zero copy) or NumPy array."""
    if _cp is not None:
        return _cp.from_dlpack(x)
    return np.asarray(x)


def _xp():
    return _cp if _cp is not None else np


def _fast_len(n):
    """Smallest 5-smooth number (2^a 3^b 5^c) >= n: fast FFT length."""
    best = 1 << max(0, n - 1).bit_length()
    f5 = 1
    while f5 < best:
        f35 = f5
        while f35 < best:
            p = 1 << max(0, -(-n // f35) - 1).bit_length()
            if p * f35 >= n:
                best = min(best, p * f35)
            f35 *= 3
        f5 *= 5
    return best


_filter_cache = {}


def _ramp_rfft(n_det, du, window):
    r"""
    rfft of the discrete Ram-Lak kernel on the padded grid, times the window.

    The kernel is built periodically on the padded length (as in
    scikit-image's ``iradon``) rather than truncated, which keeps its DC
    exactly zero; the padded length is at least ``2 * n_det`` against
    circular-convolution wrap-around, rounded up to an FFT-friendly size.
    Windows use the normalized frequency :math:`w \in [0, 1]` (1 = Nyquist).
    """
    key = (n_det, float(du), window, _cp is None)
    H = _filter_cache.get(key)
    if H is not None:
        return H
    xp = _xp()
    L = _fast_len(2 * n_det)
    h = xp.zeros(L, dtype=xp.float64)
    h[0] = 1.0 / (4 * du * du)
    k = xp.arange(1, L // 2 + 1, 2)
    h[k] = -1.0 / (np.pi * k * du) ** 2
    h[-k] = -1.0 / (np.pi * k * du) ** 2
    H = xp.fft.rfft(h).real
    w = xp.arange(L // 2 + 1) / (L / 2)  # 0..1, 1 = Nyquist
    if window == "hann":
        H = H * (0.5 + 0.5 * xp.cos(np.pi * w))
    elif window == "hamming":
        H = H * (0.54 + 0.46 * xp.cos(np.pi * w))
    elif window == "cosine":
        H = H * xp.cos(np.pi * w / 2)
    elif window == "shepp-logan":
        H = H * xp.where(w > 0, xp.sin(np.pi * w / 2) / xp.maximum(np.pi * w / 2, 1e-12), 1.0)
    elif window not in (None, "ramp"):
        raise ValueError(f"unknown window {window!r}")
    H = H.astype(xp.float32)
    _filter_cache[key] = (L, H)
    return L, H


def _filter_last_axis(yd, n_det, du, window):
    """Ramp-filter a device array along its last axis (real FFTs, padded)."""
    xp = _xp()
    L, H = _ramp_rfft(n_det, du, window)
    q = xp.fft.irfft(xp.fft.rfft(yd, n=L, axis=-1) * H, n=L, axis=-1)
    return q[..., :n_det].astype(xp.float32)


def _struct_meta(ray_spec):
    t_spec, _, u_spec = ray_spec
    n_ang = dr.width(t_spec)
    num = tuple(int(v) for v in u_spec.num[:-1])
    du = float(u_spec.step[0])
    return n_ang, num, du


def _own(x, Float):
    """Dr.Jit copy of a device array.

    ``Float(cupy_array)`` is a zero-copy view of CuPy *pool* memory, which is
    released as soon as the array object dies and may be handed to any later
    allocation — a use-after-free once the view outlives the array. Evaluating
    an arithmetic copy while the source is still alive moves the values into
    Dr.Jit-owned memory.
    """
    xp = _xp()
    view = Float(xp.ascontiguousarray(x.reshape(-1)))
    out = view + Float(0)
    dr.eval(out)   # runs while `x` is still referenced by the caller
    return out


def _adjoint_explicit(ray_spec, knot_spec, order, q, Float):
    """Backproject a device array through the fused explicit-ray kernel."""
    rays = struct_rays(ray_spec)
    b = xrt_adjoint(rays, knot_spec, order, _own(q, Float))
    dr.eval(b)
    return b


def _bpf_core(ray_spec, knot_spec, y, order, margin, mag, const):
    """Enlarged-lattice adjoint, coverage taper, in-plane Hann-|k| filter."""
    from .util import UniformSpec

    Float = type(y)
    _, _, du = _struct_meta(ray_spec)
    shape = tuple(knot_spec.num)
    step = tuple(knot_spec.step)
    start = tuple(knot_spec.start)
    D = len(shape)
    xp = _xp()

    # enlarged lattice, concentric with the requested one (in-plane axes only)
    big, off = [], []
    for a in range(D):
        grow = margin if (D == 2 or a < 2) else 1.0
        n_big = _fast_len(int(round(shape[a] * grow)))
        big.append(n_big)
        off.append((n_big - shape[a]) // 2)
    big_spec = UniformSpec(
        start=tuple(start[a] - off[a] * step[a] for a in range(D)),
        step=step, num=tuple(big))

    b = xrt_adjoint(struct_rays(ray_spec), big_spec, order, y)
    bn = _dev(b).reshape(tuple(big))

    # taper the cliff where detector coverage ends (finite detector width);
    # for divergent beams the coverage radius is the detector half-width
    # magnified back to the isocenter
    u_spec = ray_spec[2]
    u_max = (abs(float(u_spec.start[0])) + du / 2) * mag
    ax = [xp.asarray(big_spec.start[a] + step[a] * np.arange(big[a]),
                     dtype=xp.float32) for a in range(2)]
    r_in = xp.sqrt(ax[0][:, None] ** 2 + ax[1][None, :] ** 2)
    taper = xp.clip((u_max - r_in) / (0.1 * u_max), 0.0, 1.0)
    taper = taper * taper * (3 - 2 * taper)
    bn = bn * (taper[:, :, None] if D == 3 else taper)

    # |k| deconvolution, Hann-limited at the lattice Nyquist: the plain ramp
    # amplifies the discretization noise of the voxel-basis backprojection
    k1 = [xp.fft.fftfreq(big[0], d=step[0]).astype(xp.float32),
          xp.fft.rfftfreq(big[1], d=step[1]).astype(xp.float32)]
    K = xp.sqrt(k1[0][:, None] ** 2 + k1[1][None, :] ** 2)
    K = K * xp.where(K < 0.5, 0.5 + 0.5 * xp.cos(2 * np.pi * K), 0.0)
    if D == 2:
        g = xp.fft.irfft2(xp.fft.rfft2(bn) * K, s=tuple(big))
    else:  # the blur is in-plane (axes 0 and 1)
        g = xp.fft.irfft2(xp.fft.rfft2(bn, axes=(0, 1)) * K[:, :, None],
                          s=tuple(big[:2]), axes=(0, 1))
    sl = tuple(slice(off[a], off[a] + shape[a]) for a in range(D))
    g = (g[sl] * const).astype(xp.float32)
    return _own(g, Float)


# -------------------------------------------------------------- analytic ----
def fbp(ray_spec, knot_spec, y, order=0, window="hann"):
    r"""
    Filtered backprojection for :py:func:`~xrt_toolkit.parallel_beam` scans.

    Works in 2D and 3D (cylinder beam); the ramp filter is applied along the
    in-plane detector axis. Angles must cover :math:`[0, \pi)` uniformly.

    Parameters
    ----------
    ray_spec:
        Structured scan from :py:func:`~xrt_toolkit.parallel_beam`.
    y: FloatT
        Measurements from either projection interface (same ray ordering).
    window: "hann" | "hamming" | "cosine" | "shepp-logan" | "ramp" | None
        Smoothing window on the ramp filter.

    Returns
    -------
    f: FloatT
    """
    Float = type(y)
    n_ang, num, du = _struct_meta(ray_spec)
    n_det = num[0]  # in-plane detector axis (parallel_beam convention)
    xp = _xp()
    yd = _dev(y).reshape(n_ang, *num)
    if len(num) == 2:  # 3D: filter along the in-plane axis, keep the axial one
        yd = xp.ascontiguousarray(xp.moveaxis(yd, 2, 1))
    q = _filter_last_axis(yd, n_det, du, window)
    if len(num) == 2:
        q = xp.moveaxis(q, 1, 2)
    b = _adjoint_explicit(ray_spec, knot_spec, order, q, Float)
    return b * (np.pi * du * du / n_ang)


def fbp_cone(ray_spec, knot_spec, y, sod, sdd, order=0, window="hann",
             method="fdk"):
    r"""
    Filtered backprojection for :py:func:`~xrt_toolkit.cone_beam` scans over a
    full :math:`[0, 2\pi)` circle, flat detector.

    With a 2D detector (3D reconstruction) two methods are available:

    * ``"fdk"`` — Feldkamp-Davis-Kress: cosine pre-weighting, ramp filtering
      along the in-plane detector axis, backprojection. The matched adjoint of
      a divergent 3D beam carries the FDK distance weighting
      :math:`1/U^{2}` naturally (ray density from a point source), so no
      custom backprojector is needed; the residual per-ray obliquity factor
      is applied to the filtered projections. Exact in the midplane, the
      usual FDK cone artifacts away from it.
    * ``"bpf"`` — not available for divergent scans: image-domain
      deconvolution requires the backprojection blur to be shift-invariant,
      which holds for parallel scans (see :py:func:`bpf`) but measurably
      fails for cone geometries with a voxel-basis adjoint. The
      backprojection-filtration algorithms that do exist for cone beams
      (Hilbert filtering along PI-lines) are out of scope; use ``"fdk"`` or
      :py:func:`cg`.

    With a 1D detector (2D fan beam) the classic fan FBP is used (``method``
    is ignored except that ``"bpf"`` raises). Values are exact at the
    isocenter and drift by :math:`O((r/\text{sod})^{2})` away from it; use
    :py:func:`cg` when quantitative accuracy matters.

    Parameters
    ----------
    sod, sdd: float
        Source-object and source-detector distances used to build the scan.
    window: "hann" | "hamming" | "cosine" | "shepp-logan" | "ramp" | None
        Smoothing window on the ramp filter (``"fdk"`` path).
    method: "fdk" | "bpf"

    Returns
    -------
    f: FloatT
    """
    Float = type(y)
    n_ang, num, du = _struct_meta(ray_spec)
    n_det = num[0]
    xp = _xp()
    mag = sod / sdd

    if len(num) == 1:  # 2D fan beam
        if method == "bpf":
            raise NotImplementedError(
                "backprojection-then-filtering assumes a shift-invariant "
                "backprojection blur; that holds for parallel scans (use "
                "bpf()) but not for divergent ones. Use method='fdk' or "
                "optim.cg.")
        u = (xp.arange(n_det, dtype=xp.float32) - (n_det - 1) / 2) * du
        w = sdd / xp.sqrt(sdd**2 + u**2)
        yd = _dev(y).reshape(n_ang, n_det) * w
        q = _filter_last_axis(yd, n_det, du, window)
        b = _adjoint_explicit(ray_spec, knot_spec, order, q, Float)
        return b * (np.pi * du * du / n_ang)

    n1, n2 = num
    u_spec = ray_spec[2]
    du1, du2 = float(u_spec.step[0]), float(u_spec.step[1])
    if method == "bpf":
        raise NotImplementedError(
            "backprojection-then-filtering assumes a shift-invariant "
            "backprojection blur; that holds for parallel scans (use bpf()) "
            "but not for divergent ones. Use method='fdk' or optim.cg.")
    if method != "fdk":
        raise ValueError(f"unknown method {method!r}")

    u1 = (xp.arange(n1, dtype=xp.float32) - (n1 - 1) / 2) * du1
    u2 = (xp.arange(n2, dtype=xp.float32) - (n2 - 1) / 2) * du2
    cos = sdd / xp.sqrt(sdd**2 + u1[:, None] ** 2 + u2[None, :] ** 2)
    yd = _dev(y).reshape(n_ang, n1, n2) * cos[None]
    q = xp.ascontiguousarray(xp.moveaxis(yd, 2, 1))  # ramp along u1
    q = _filter_last_axis(q, n1, du1, window)
    q = xp.moveaxis(q, 1, 2) / (cos[None] ** 2)
    b = _adjoint_explicit(ray_spec, knot_spec, order, q, Float)
    return b * (np.pi * du1 * du1 * du2 * sod / (n_ang * sdd))


def bpf(ray_spec, knot_spec, y, order=0, margin=2.0):
    r"""
    Backprojection-then-filtering for :py:func:`~xrt_toolkit.parallel_beam`
    scans (2D and 3D cylinder beam). For cone scans use
    :py:func:`fbp_cone` with ``method="bpf"``.

    The unfiltered backprojection :math:`b = A^{\top} y` blurs the image with
    :math:`1/r` in the scan plane; deconvolution multiplies its spectrum by
    the in-plane frequency magnitude :math:`|k|` (Hann-limited at the lattice
    Nyquist). Since the filtering happens after backprojection, the data-side
    step is a plain adjoint — no detector-domain filtering — which makes the
    method easy to adapt to non-standard acquisition geometries. Values are
    accurate to a few percent; the :math:`1/r` tails are handled on a lattice
    enlarged by ``margin`` and the end of detector coverage is tapered.

    Returns
    -------
    f: FloatT
    """
    n_ang, num, du = _struct_meta(ray_spec)
    const = np.pi * du / n_ang
    if len(num) == 2:  # 3D: axial ray density adds 1/du2
        const = const * float(ray_spec[2].step[1])
    return _bpf_core(ray_spec, knot_spec, y, order, margin, 1.0, const)
