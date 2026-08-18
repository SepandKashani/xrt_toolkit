r"""
Basic reconstruction algorithms.

Iterative solvers (:py:func:`cg`, :py:func:`gd`) take the forward/adjoint pair
as callables, so they work with any geometry interface of the library.
Analytic methods (:py:func:`fbp`, :py:func:`fbp_cone`, :py:func:`bpf`) operate
on structured scans built with :py:func:`~xrt_toolkit.parallel_beam` /
:py:func:`~xrt_toolkit.cone_beam`.

Everything runs on the GPU: the analytic methods backproject with a fused
voxel-driven interpolating kernel (the textbook FBP/FDK backprojector — the
matched exact-chord adjoint belongs inside the iterative solvers, where the
pair must be adjoint, but as a backprojector its chords degenerate for rays
nearly tangent to lattice planes and print grid-aligned artifacts), and
Fourier filtering uses CuPy FFTs on the same device memory as the Dr.Jit
arrays (zero-copy through DLPack). Reconstructions are masked to the
fully-covered field of view, as usual for analytic methods.
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
except Exception as _e:  # pragma: no cover - environment dependent
    import warnings
    _cp = None
    warnings.warn(
        "xrt_toolkit.optim: CuPy is unusable "
        f"({type(_e).__name__}: {_e}); Fourier filtering falls back to NumPy "
        "on the host, which costs a device round-trip per reconstruction. "
        "Install a CuPy matching your CUDA version for the GPU path.",
        RuntimeWarning, stacklevel=2)


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


def _ramp_rfft(n_det, du, window, w_cut=1.0):
    r"""
    rfft of the discrete Ram-Lak kernel on the padded grid, times the window.

    The kernel is built periodically on the padded length (as in
    scikit-image's ``iradon``) rather than truncated, which keeps its DC
    exactly zero; the padded length is at least ``2 * n_det`` against
    circular-convolution wrap-around, rounded up to an FFT-friendly size.

    ``w_cut`` is the highest frequency the RECONSTRUCTION lattice can
    represent, in units of the detector Nyquist. When the detector samples
    finer than the lattice (``du < step``), the ramp must stop at the lattice
    Nyquist: everything above it cannot be represented by the volume and is
    aliased into broadband noise by the backprojection — finer detectors then
    make the reconstruction WORSE, not better. Windows are applied relative
    to this cutoff.
    """
    key = (n_det, float(du), window, float(w_cut), _cp is None)
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
    w = xp.arange(L // 2 + 1) / (L / 2) / w_cut  # 0..1, 1 = usable Nyquist
    if window == "hann":
        H = H * (0.5 + 0.5 * xp.cos(np.pi * xp.minimum(w, 1.0)))
    elif window == "hamming":
        H = H * (0.54 + 0.46 * xp.cos(np.pi * xp.minimum(w, 1.0)))
    elif window == "cosine":
        H = H * xp.cos(np.pi * xp.minimum(w, 1.0) / 2)
    elif window == "shepp-logan":
        H = H * xp.where(w > 0, xp.sin(np.pi * xp.minimum(w, 1.0) / 2)
                         / xp.maximum(np.pi * w / 2, 1e-12), 1.0)
    elif window not in (None, "ramp"):
        raise ValueError(f"unknown window {window!r}")
    H = xp.where(w <= 1.0, H, 0.0).astype(xp.float32)
    _filter_cache[key] = (L, H)
    return L, H


def _filter_last_axis(yd, n_det, du, window, w_cut=1.0):
    """Ramp-filter a device array along its last axis (real FFTs, padded)."""
    xp = _xp()
    L, H = _ramp_rfft(n_det, du, window, w_cut)
    q = xp.fft.irfft(xp.fft.rfft(yd, n=L, axis=-1) * H, n=L, axis=-1)
    return q[..., :n_det].astype(xp.float32)


def _fov_mask(b, ray_spec, knot_spec, r_fov):
    """Zero voxels outside the fully-covered field of view (standard FBP
    practice: outside it, projections cover the voxel only for part of the
    angles and the unbalanced filter lobes leave large artifacts)."""
    xp = _xp()
    num = tuple(knot_spec.num)
    step = tuple(float(v) for v in knot_spec.step)
    start = tuple(float(v) for v in knot_spec.start)
    ax = [xp.asarray(start[a] + step[a] * np.arange(num[a]), dtype=xp.float32)
          for a in range(2)]
    r_in = xp.sqrt(ax[0][:, None] ** 2 + ax[1][None, :] ** 2)
    m = (r_in <= r_fov).astype(xp.float32)
    bd = _dev(b).reshape(num)
    bd = bd * (m[:, :, None] if len(num) == 3 else m)
    return bd


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


def _bp_voxel(q, ray_spec, knot_spec, divergent=False, sod=None, sdd=None):
    r"""
    Voxel-driven interpolating backprojection (the textbook FBP/FDK
    backprojector, as in ASTRA/TIGRE): every voxel accumulates a (bi)linearly
    interpolated sample of each filtered projection at its projected detector
    coordinate. For divergent scans the FDK distance weight
    :math:`(\mathrm{sod}/U)^{2}` is applied per (voxel, projection).

    This is deliberately NOT the matched adjoint: the exact-chord adjoint is
    the right operator inside iterative solvers, but as a backprojector its
    chords degenerate for rays nearly tangent to lattice planes, printing
    horizontal/vertical line artifacts. Interpolating backprojection has no
    such failure mode.

    ``q``: device array shaped ``(n_ang, n_u1[, n_u2])``. Returns the plain
    sum over projections (unscaled), as a Dr.Jit array over voxels.
    """
    from drjit.cuda import Float as F, Int32, UInt32

    t_spec, n_spec, u_spec = ray_spec
    T = np.array(t_spec)                       # (D, D, n_ang) homogeneous
    Nn = np.array(n_spec)
    n_ang = T.shape[-1]
    D = len(knot_spec.num)
    num = tuple(knot_spec.num)
    step = tuple(float(v) for v in knot_spec.step)
    start = tuple(float(v) for v in knot_spec.start)
    (u1_0, du1, n_u1) = (float(u_spec.start[0]), float(u_spec.step[0]),
                         int(u_spec.num[0]))
    three_d = D == 3
    if three_d:
        (u2_0, du2, n_u2) = (float(u_spec.start[1]), float(u_spec.step[1]),
                             int(u_spec.num[1]))
    else:
        n_u2 = 1

    # per-angle tables.  Parallel scans park the detector axes in the
    # t-matrix (t = u1 d1 [+ u2 d2] + offset); divergent scans park them in
    # the n-matrix (n = sdd c + u1 d1 [+ u2 d2]) with t = source.
    A = Nn if divergent else T
    d1 = [F(np.ascontiguousarray(A[c, 0, :], np.float32)) for c in range(D)]
    off = [F(np.ascontiguousarray(T[c, -1, :], np.float32)) for c in range(D)]
    if divergent:
        cd = Nn[:, -1, :] / sdd                # central unit direction
        cdir = [F(np.ascontiguousarray(cd[c], np.float32)) for c in range(D)]
    if three_d:
        d2 = [F(np.ascontiguousarray(A[c, 1, :], np.float32)) for c in range(D)]

    qf = _own(q, F)
    dr.eval(*d1, *off, qf)

    # voxel world coordinates (lane = voxel, C-order like the volume layout)
    lane = dr.arange(UInt32, int(np.prod(num)))
    if three_d:
        ix = lane // (num[1] * num[2])
        iy = (lane // num[2]) % num[1]
        iz = lane % num[2]
        xw = [start[0] + step[0] * F(ix), start[1] + step[1] * F(iy),
              start[2] + step[2] * F(iz)]
    else:
        xw = [start[0] + step[0] * F(lane // num[1]),
              start[1] + step[1] * F(lane % num[1])]

    def body(a, acc):
        e1 = [dr.gather(F, d1[c], a) for c in range(D)]
        o = [dr.gather(F, off[c], a) for c in range(D)]
        v = [xw[c] - o[c] for c in range(D)]
        if divergent:
            c = [dr.gather(F, cdir[k], a) for k in range(D)]
            U = sum(v[k] * c[k] for k in range(D))
            mag = sdd / U
            w = (sod / U) ** 2
        else:
            mag = 1.0
            w = 1.0
        u1 = sum(v[k] * e1[k] for k in range(D)) * mag
        p1 = (u1 - u1_0) / du1
        i1 = dr.floor(p1)
        f1 = p1 - i1
        i1 = Int32(i1)
        ok = (i1 >= 0) & (i1 < n_u1 - 1)
        i1 = dr.clip(i1, 0, n_u1 - 2)
        base = a * (n_u1 * n_u2)
        if three_d:
            e2 = [dr.gather(F, d2[c], a) for c in range(D)]
            u2 = sum(v[k] * e2[k] for k in range(D)) * mag
            p2 = (u2 - u2_0) / du2
            i2 = dr.floor(p2)
            f2 = p2 - i2
            i2 = Int32(i2)
            ok = ok & (i2 >= 0) & (i2 < n_u2 - 1)
            i2 = dr.clip(i2, 0, n_u2 - 2)
            idx = UInt32(base + i1 * n_u2 + i2)
            s00 = dr.gather(F, qf, idx, ok)
            s01 = dr.gather(F, qf, idx + 1, ok)
            s10 = dr.gather(F, qf, idx + n_u2, ok)
            s11 = dr.gather(F, qf, idx + n_u2 + 1, ok)
            smp = dr.lerp(dr.lerp(s00, s01, f2), dr.lerp(s10, s11, f2), f1)
        else:
            idx = UInt32(base + i1)
            smp = dr.lerp(dr.gather(F, qf, idx, ok),
                          dr.gather(F, qf, idx + 1, ok), f1)
        return a + 1, dr.fma(dr.select(ok, w, 0.0), smp, acc)

    a0 = dr.zeros(UInt32, dr.width(lane))
    acc0 = dr.zeros(F, dr.width(lane))
    _, out = dr.while_loop(state=(a0, acc0),
                           cond=lambda a, acc: a < n_ang,
                           body=body, labels=("a", "acc"),
                           max_iterations=-1)
    dr.eval(out)
    return out


def _bpf_core(ray_spec, knot_spec, y, margin, mag, const, window="hann"):
    """Voxel-driven backprojection of the raw data on an enlarged lattice,
    coverage taper, in-plane Hann-limited |k| filter, crop, scale."""
    from .util import UniformSpec

    Float = type(y)
    n_ang, num_det, du = _struct_meta(ray_spec)
    shape = tuple(knot_spec.num)
    step = tuple(float(v) for v in knot_spec.step)
    start = tuple(float(v) for v in knot_spec.start)
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

    yd = _dev(y).reshape((n_ang,) + num_det)
    b = _bp_voxel(yd, ray_spec, big_spec)
    bn = _dev(b).reshape(tuple(big))

    # taper the cliff where detector coverage ends (finite detector width);
    # filtering that cliff rings back into the field of view
    u_max = (abs(float(ray_spec[2].start[0])) + du / 2) * mag
    ax = [xp.asarray(big_spec.start[a] + step[a] * np.arange(big[a]),
                     dtype=xp.float32) for a in range(2)]
    r_in = xp.sqrt(ax[0][:, None] ** 2 + ax[1][None, :] ** 2)
    taper = xp.clip((u_max - r_in) / (0.1 * u_max), 0.0, 1.0)
    taper = taper * taper * (3 - 2 * taper)
    bn = bn * (taper[:, :, None] if D == 3 else taper)

    k1 = [xp.fft.fftfreq(big[0], d=step[0]).astype(xp.float32),
          xp.fft.rfftfreq(big[1], d=step[1]).astype(xp.float32)]
    K = xp.sqrt(k1[0][:, None] ** 2 + k1[1][None, :] ** 2)
    if window == "hann":
        K = K * xp.where(K < 0.5, 0.5 + 0.5 * xp.cos(2 * np.pi * K), 0.0)
    elif window in (None, "ramp"):
        K = xp.where(K <= 0.5, K, 0.0)
    else:
        raise ValueError(f"unknown window {window!r}")
    if D == 2:
        g = xp.fft.irfft2(xp.fft.rfft2(bn) * K, s=tuple(big))
    else:  # the blur is in-plane (axes 0 and 1)
        g = xp.fft.irfft2(xp.fft.rfft2(bn, axes=(0, 1)) * K[:, :, None],
                          s=tuple(big[:2]), axes=(0, 1))
    sl = tuple(slice(off[a], off[a] + shape[a]) for a in range(D))
    g = (g[sl] * const).astype(xp.float32)
    return _own(g, Float)


# -------------------------------------------------------------- analytic ----
def fbp(ray_spec, knot_spec, y, window="hann"):
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
    step_ip = min(float(v) for v in knot_spec.step[:2])  # in-plane lattice step
    w_cut = min(1.0, du / step_ip)
    yd = _dev(y).reshape(n_ang, *num)
    if len(num) == 2:  # 3D: filter along the in-plane axis, keep the axial one
        yd = xp.ascontiguousarray(xp.moveaxis(yd, 2, 1))
    q = _filter_last_axis(yd, n_det, du, window, w_cut)
    if len(num) == 2:
        q = xp.ascontiguousarray(xp.moveaxis(q, 1, 2))
    b = _bp_voxel(q, ray_spec, knot_spec)
    u_half = abs(float(ray_spec[2].start[0])) + du / 2
    b = _fov_mask(b, ray_spec, knot_spec, u_half)
    return _own(b, Float) * (np.pi * du / n_ang)


def fbp_cone(ray_spec, knot_spec, y, sod, sdd, window="hann",
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
        step_ip = min(float(v) for v in knot_spec.step)
        w_cut = min(1.0, du * mag / step_ip)  # detector Nyquist at the isocenter
        yd = _dev(y).reshape(n_ang, n_det) * w
        q = _filter_last_axis(yd, n_det, du, window, w_cut)
        # the ramp kernel lives in detector units; the reconstruction needs it
        # in isocenter units, hence the sdd/sod rescale
        b = _bp_voxel(q, ray_spec, knot_spec, divergent=True, sod=sod, sdd=sdd)
        u_half = abs(float(ray_spec[2].start[0])) + du / 2
        b = _fov_mask(b, ray_spec, knot_spec, u_half * sod / (sdd + u_half))
        return _own(b, Float) * (np.pi * du * sdd / (n_ang * sod))

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
    step_ip = min(float(v) for v in knot_spec.step[:2])
    w_cut = min(1.0, du1 * mag / step_ip)  # detector Nyquist at the isocenter
    yd = _dev(y).reshape(n_ang, n1, n2) * cos[None]
    q = xp.ascontiguousarray(xp.moveaxis(yd, 2, 1))  # ramp along u1
    q = _filter_last_axis(q, n1, du1, window, w_cut)
    q = xp.ascontiguousarray(xp.moveaxis(q, 1, 2))
    # detector-unit ramp -> isocenter units: sdd/sod
    b = _bp_voxel(q, ray_spec, knot_spec, divergent=True, sod=sod, sdd=sdd)
    u_half = abs(float(ray_spec[2].start[0])) + du1 / 2
    b = _fov_mask(b, ray_spec, knot_spec, u_half * sod / (sdd + u_half))
    return _own(b, Float) * (np.pi * du1 * sdd / (n_ang * sod))


def bpf(ray_spec, knot_spec, y, margin=2.0, window="hann"):
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
    return _bpf_core(ray_spec, knot_spec, y, margin, 1.0,
                     np.pi / n_ang, window)
