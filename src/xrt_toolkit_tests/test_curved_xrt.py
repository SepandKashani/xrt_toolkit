"""
Tests for the refractive (curved-ray) operators in
``xrt_toolkit.drjit.curved_xrt``.

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
import xrt_toolkit.drjit.curved_xrt as cx  # noqa: E402


def _smooth_pos(num, base, amp, seed):
    f = np.random.default_rng(seed).standard_normal(num)
    for ax in range(f.ndim):
        for _ in range(3):
            f = 0.25 * np.roll(f, 1, ax) + 0.5 * f + 0.25 * np.roll(f, -1, ax)
    return (base + amp * f / (np.abs(f).max() + 1e-9)).ravel().astype(np.float32)


def _ring_rays(D, N, L, seed):
    rng = np.random.default_rng(seed)
    R = N * 0.9
    ang = rng.uniform(0, 2 * np.pi, L)
    if D == 2:
        t = np.stack([R * np.cos(ang), R * np.sin(ang)], 0)
    else:
        th = np.arccos(rng.uniform(-1, 1, L))
        t = np.stack([R * np.sin(th) * np.cos(ang),
                      R * np.sin(th) * np.sin(ang), R * np.cos(th)], 0)
    n = -t / np.linalg.norm(t, axis=0) + 0.25 * rng.standard_normal((D, L))
    A = Array2f if D == 2 else Array3f
    return (A(*t.astype(np.float32)), A(*n.astype(np.float32)))


@pytest.mark.parametrize("D,N", [(2, 32), (3, 20)])
@pytest.mark.parametrize("bend", [False, True])
def test_adjoint_dot(D, N, bend):
    # <A du, r> == <du, A^T r> for the frozen-path linear operator.
    spec = xtk.UniformSpec.centered(step=1.0, num=(N,) * D)
    u0 = Float(_smooth_pos((N,) * D, 1.5, 0.3, 1))
    du = Float(_smooth_pos((N,) * D, 0.0, 1.0, 2))
    ray = _ring_rays(D, N, 4096, 3)
    r = Float(np.random.default_rng(4).standard_normal(4096).astype(np.float32))

    Adu = cx.refract_apply(ray, spec, du, coeff_geom=u0, bend=bend)
    Atr = cx.refract_adjoint(ray, spec, r, u0, bend=bend)
    lhs = float(dr.sum(Adu * r)[0])
    rhs = float(dr.sum(du * Atr)[0])
    assert abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30) < 1e-4


@pytest.mark.parametrize("D,N", [(2, 32), (3, 20)])
def test_constant_field_is_slowness_times_chord(D, N):
    # In a constant slowness field the ray is straight and T = u * chord.
    spec = xtk.UniformSpec.centered(step=1.0, num=(N,) * D)
    u_const = 1.7
    coeff = Float(np.full(N ** D, u_const, np.float32))
    # axial ray through the centre; chord = full width N
    t = np.zeros((D, 1), np.float32); t[0] = -100.0
    n = np.zeros((D, 1), np.float32); n[0] = 1.0
    A = Array2f if D == 2 else Array3f
    ray = (A(*t), A(*n))
    T = cx.refract_apply(ray, spec, coeff)
    dr.eval(T)
    assert abs(float(T[0]) - u_const * N) / (u_const * N) < 1e-3


def _interp2(U, num, start, step, x):
    g = (x - start) / step
    i0 = np.floor(g).astype(int); f = g - i0
    val = 0.0; grad = np.zeros(2)
    C = U.reshape(num)
    for k in range(4):
        b = [(k >> d) & 1 for d in range(2)]
        idx = np.clip(i0 + np.array(b), 0, np.array(num) - 1)
        cq = C[tuple(idx)]
        comp = [f[d] if b[d] else 1 - f[d] for d in range(2)]
        val += comp[0] * comp[1] * cq
        grad[0] += (1.0 if b[0] else -1.0) * comp[1] / step[0] * cq
        grad[1] += comp[0] * (1.0 if b[1] else -1.0) / step[1] * cq
    return val, grad


def _trace2(U, num, start, step, x0, tau0, ds, nmax, ll, ur):
    x = np.array(x0, float); tau = np.array(tau0, float); tau /= np.linalg.norm(tau)
    path = [x.copy()]
    for _ in range(nmax):
        if np.any(x < ll) or np.any(x > ur):
            break
        u0, g0 = _interp2(U, num, start, step, x)
        a0 = (g0 - np.dot(g0, tau) * tau) / max(u0, 1e-8)
        xm = x + 0.5 * ds * tau; tm = tau + 0.5 * ds * a0; tm /= np.linalg.norm(tm)
        um, gm = _interp2(U, num, start, step, xm)
        am = (gm - np.dot(gm, tm) * tm) / max(um, 1e-8)
        x = x + ds * tm; tau = tau + ds * am; tau /= np.linalg.norm(tau)
        path.append(x.copy())
    return np.clip(x, ll, ur), np.array(path)


def test_circular_arc_radius_and_kernel():
    # A linear velocity gradient bends rays into circular arcs of radius
    # R = 1/(g p), p = cos(phi)/c at the launch point.
    N = 256
    h = 1.0 / N
    spec = xtk.UniformSpec.centered(step=h, num=(N, N))
    start = np.array(spec.start); step = np.array(spec.step); numg = np.array(spec.num)
    ll = start - step / 2; ur = ll + numg * step
    c0, g = 1.5, 0.8
    ys = start[1] + np.arange(N) * step[1]
    C = np.broadcast_to((c0 + g * ys)[None, :], (N, N)).copy()
    U = (1.0 / C).astype(np.float32)
    coeff = Float(U.ravel())

    phi = np.deg2rad(28.0)
    x0 = np.array([ll[0] + 1e-3, -0.2])
    ds = h
    nmax = int(2.0 / ds)

    # independent numpy integrator: exit point + circle-fit radius
    xn, path = _trace2(U.ravel(), numg, start, step, x0, [np.cos(phi), np.sin(phi)],
                       ds, nmax, ll, ur)
    A = np.c_[2 * path[:, 0], 2 * path[:, 1], np.ones(len(path))]
    sol, *_ = np.linalg.lstsq(A, path[:, 0] ** 2 + path[:, 1] ** 2, rcond=None)
    R_fit = np.sqrt(sol[2] + sol[0] ** 2 + sol[1] ** 2)
    R_ana = 1.0 / (g * (np.cos(phi) / (c0 + g * x0[1])))
    assert abs(R_fit - R_ana) / R_ana < 1e-2       # physics: circular arc

    ray = (Array2f(np.array([x0[0]], np.float32), np.array([x0[1]], np.float32)),
           Array2f(np.array([np.cos(phi)], np.float32), np.array([np.sin(phi)], np.float32)))
    _, xe, _ = cx.refract_apply(ray, spec, coeff, ds=ds, max_steps=nmax, return_path=True)
    dr.eval(xe)
    assert np.hypot(float(xe.x[0]) - xn[0], float(xe.y[0]) - xn[1]) < 1e-3  # kernel


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
