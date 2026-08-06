r"""
Basic reconstruction algorithms.

Iterative solvers (:py:func:`cg`, :py:func:`gd`) take the forward/adjoint pair
as callables, so they work with any geometry interface of the library.
Analytic methods (:py:func:`fbp`, :py:func:`fbp_cone`, :py:func:`bpf`) operate
on structured scans built with :py:func:`~xrt_toolkit.parallel_beam` /
:py:func:`~xrt_toolkit.cone_beam`; filtering runs on the CPU with NumPy FFTs,
projection and backprojection on the GPU.

All scaling constants assume the lattice ``step`` and detector units are the
same (the usual voxel-unit convention), angles uniformly spaced over
:math:`[0, \pi)` for parallel scans and :math:`[0, 2\pi)` for cone scans.
"""

import numpy as np

import drjit as dr

from .drjit.ray_xrt import xrt_adjoint, xrt_apply  # noqa: F401
from .drjit.struct_xrt import xrt_struct_adjoint, xrt_struct_apply

__all__ = ["cg", "gd", "fbp", "fbp_cone", "bpf"]


# ------------------------------------------------------------- iterative ----
def cg(A, At, y, n_unknowns, n_iter=30, x0=None):
    r"""
    Conjugate gradient on the normal equations :math:`A^{\top}A f = A^{\top}y`.

    Parameters
    ----------
    A, At: callable
        Forward and adjoint operators, e.g.
        ``lambda v: xrt_apply(rays, knot, order, v)`` and its adjoint.
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


# -------------------------------------------------------------- analytic ----
def _ramlak(n_det, du, window):
    # Discrete Ram-Lak kernel (Kak & Slaney), returned as its DFT on a
    # 4*n_det grid.  Using the DFT of the space-domain kernel (rather than
    # |f| directly) avoids the DC bias of the naive ramp.
    n = np.arange(-n_det, n_det + 1)
    h = np.zeros(n.shape, np.float64)
    h[n == 0] = 1.0 / (4 * du * du)
    odd = (n % 2) != 0
    h[odd] = -1.0 / (np.pi**2 * n[odd] ** 2 * du * du)
    L = 4 * n_det
    H = np.fft.fft(np.roll(np.pad(h, (0, L - len(h))), -n_det))
    if window == "hann":
        f = np.fft.fftfreq(L)
        H = H * (0.5 + 0.5 * np.cos(2 * np.pi * f))
    return H


def _struct_meta(ray_spec):
    t_spec, _, u_spec = ray_spec
    n_ang = dr.width(t_spec)
    num = tuple(int(v) for v in u_spec.num[:-1])
    du = float(u_spec.step[0])
    return n_ang, num, du


def _filter_rows(y2, H, n_det):
    L = H.shape[0]
    q = np.real(np.fft.ifft(np.fft.fft(y2, L, axis=-1) * H, axis=-1))
    return np.ascontiguousarray(q[..., :n_det], np.float32)


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
        Measurements from :py:func:`~xrt_toolkit.xrt_struct_apply`.
    window: "hann" | None
        Smoothing window on the ramp filter.

    Returns
    -------
    f: FloatT
    """
    Float = type(y)
    n_ang, num, du = _struct_meta(ray_spec)
    n_det = num[0]  # in-plane detector axis (parallel_beam convention)
    yn = np.asarray(y, np.float64).reshape(n_ang, *num)
    if len(num) == 2:  # 3D: filter along axis 1 (in-plane), keep axis 2
        yn = np.moveaxis(yn, 2, 1)
    q = _filter_rows(yn, _ramlak(n_det, du, window), n_det)
    if len(num) == 2:
        q = np.moveaxis(q, 1, 2)
    b = xrt_struct_adjoint(ray_spec, knot_spec, order,
                           Float(np.ascontiguousarray(q.reshape(-1))))
    return b * (np.pi * du * du / n_ang)


def fbp_cone(ray_spec, knot_spec, y, sod, sdd, order=0, window="hann"):
    r"""
    Approximate filtered backprojection for :py:func:`~xrt_toolkit.cone_beam`
    scans (2D fan beam, flat detector, angles covering :math:`[0, 2\pi)`).

    The data is cosine-weighted and ramp-filtered as in flat-detector FDK,
    then backprojected with the matched adjoint. The adjoint does not apply
    the FDK distance weighting :math:`1/U^{2}`, so values are exact at the
    isocenter and drift by :math:`O(r/\text{sod})` away from it; use
    :py:func:`cg` when quantitative accuracy matters.

    Parameters
    ----------
    sod, sdd: float
        Source-object and source-detector distances used to build the scan.

    Returns
    -------
    f: FloatT
    """
    Float = type(y)
    n_ang, num, du = _struct_meta(ray_spec)
    n_det = num[0]
    u = (np.arange(n_det) - (n_det - 1) / 2) * du
    w = sdd / np.sqrt(sdd**2 + u**2)
    yn = np.asarray(y, np.float64).reshape(n_ang, n_det) * w
    q = _filter_rows(yn, _ramlak(n_det, du, window), n_det)
    b = xrt_struct_adjoint(ray_spec, knot_spec, order,
                           Float(np.ascontiguousarray(q.reshape(-1))))
    return b * (np.pi * du * du / n_ang)


def bpf(ray_spec, knot_spec, y, order=0, margin=2.0):
    r"""
    Backprojection-then-filtering for :py:func:`~xrt_toolkit.parallel_beam`
    scans (2D and 3D cylinder beam).

    The unfiltered backprojection :math:`b = A^{\top} y` blurs the image with
    :math:`1/r` in the scan plane; deconvolution multiplies its spectrum by
    the in-plane frequency magnitude :math:`|k|`. Since the filtering happens
    after backprojection, the data-side step is a plain adjoint — no
    detector-domain filtering — which makes the method easy to adapt to
    non-standard acquisition geometries.

    The :math:`1/r` tails extend far beyond the object, so the backprojection
    is computed on a lattice enlarged by ``margin`` in the scan plane and
    cropped after filtering; too small a margin shows up as a low-frequency
    bias and edge ringing.

    Returns
    -------
    f: FloatT
    """
    from .util import UniformSpec

    Float = type(y)
    n_ang, _, du = _struct_meta(ray_spec)
    shape = tuple(knot_spec.num)
    step = tuple(knot_spec.step)
    start = tuple(knot_spec.start)
    D = len(shape)

    # enlarged lattice, concentric with the requested one (in-plane axes only)
    big, off = [], []
    for a in range(D):
        grow = margin if (D == 2 or a < 2) else 1.0
        n_big = int(round(shape[a] * grow))
        pad = (n_big - shape[a]) // 2
        big.append(n_big)
        off.append(pad)
    big_spec = UniformSpec(
        start=tuple(start[a] - off[a] * step[a] for a in range(D)),
        step=step, num=tuple(big))

    b = xrt_struct_adjoint(ray_spec, big_spec, order, y)
    bn = np.asarray(b, np.float64).reshape(tuple(big))

    # The backprojection ends abruptly where detector coverage ends (finite
    # detector width); filtering that cliff rings back into the field of
    # view.  Taper it smoothly over ~10% of the coverage radius.
    u_spec = ray_spec[2]
    n_det = int(u_spec.num[0])
    u_max = abs(float(u_spec.start[0])) + du / 2  # detector half-width
    ax = [big_spec.start[a] + step[a] * np.arange(big[a]) for a in range(2)]
    r_in = np.sqrt(ax[0][:, None] ** 2 + ax[1][None, :] ** 2)
    taper = np.clip((u_max - r_in) / (0.1 * u_max), 0.0, 1.0)
    taper = taper * taper * (3 - 2 * taper)
    bn = bn * (taper[:, :, None] if D == 3 else taper)

    k1 = [np.fft.fftfreq(n, d=step[a]) for a, n in enumerate(big)]
    K = np.sqrt(k1[0][:, None] ** 2 + k1[1][None, :] ** 2)
    # Hann rolloff to the lattice Nyquist: |k| amplifies the high-frequency
    # discretization noise of the voxel-basis backprojection otherwise.
    K = K * np.where(K < 0.5, 0.5 + 0.5 * np.cos(2 * np.pi * K), 0.0)
    if D == 2:
        g = np.real(np.fft.ifft2(np.fft.fft2(bn) * K))
    else:  # 3D cylinder beam: the blur is in-plane (axes 0 and 1)
        g = np.real(np.fft.ifft2(np.fft.fft2(bn, axes=(0, 1))
                                 * K[:, :, None], axes=(0, 1)))
    sl = tuple(slice(off[a], off[a] + shape[a]) for a in range(D))
    g = g[sl] * (np.pi * du / n_ang)
    return Float(np.ascontiguousarray(g.reshape(-1), np.float32))
