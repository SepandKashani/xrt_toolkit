"""Generate the documentation figures. Needs a CUDA device; the resulting PNGs
are committed so the docs build on a GPU-less machine (Read the Docs).

    python doc/make_figures.py
"""
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
import drjit as dr
from drjit.cuda.ad import Array2f, Float

import xrt_toolkit as xtk

OUT = pathlib.Path(__file__).resolve().parent / "_static"
OUT.mkdir(exist_ok=True)
N = 128
knot = xtk.UniformSpec(start=(-N/2 + 0.5,) * 2, step=1, num=(N, N))


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def _lattice(ax, B, n=8):
    ax.add_patch(plt.Rectangle((-B, -B), 2 * B, 2 * B, facecolor="#eceff3",
                               edgecolor="#8b949e", lw=1.0, zorder=1))
    for k in np.linspace(-B, B, n + 1)[1:-1]:
        ax.plot([-B, B], [k, k], color="white", lw=0.6, zorder=1.5)
        ax.plot([k, k], [-B, B], color="white", lw=0.6, zorder=1.5)


def _ray(ax, p0, p1, color, lw=1.1, dot=3.4, head=7.0):
    """One ray: dot at the start, line, arrow tip exactly at the end."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    d = d / max(np.hypot(*d), 1e-12)
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color, lw=lw,
            solid_capstyle="round", zorder=2)
    ax.plot(*p0, "o", color=color, ms=dot, zorder=4)
    ax.annotate("", xy=tuple(p1), xytext=tuple(p1 - 0.001 * d),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=head, shrinkA=0, shrinkB=0),
                zorder=4)


def _rotation(ax, r, a0, a1, label_at, color="#57606a"):
    """Dashed arc with an arrowhead and the word 'rotation' beside the head."""
    th = np.linspace(a0, a1, 80)
    ax.plot(r * np.cos(th), r * np.sin(th), color=color, lw=1.0, ls=(0, (4, 3)),
            zorder=2)
    tip = np.array([r * np.cos(th[-1]), r * np.sin(th[-1])])
    prev = np.array([r * np.cos(th[-4]), r * np.sin(th[-4])])
    ax.annotate("", xy=tuple(tip), xytext=tuple(prev),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                mutation_scale=9, shrinkA=0, shrinkB=0))
    ax.text(*(tip + label_at), "rotation", fontsize=9, color=color,
            ha="center", va="center")


def geometry_schematics():
    """One drawing per geometry, from the rays the library builds. Vector output."""
    B = 1.0
    BLUE, PURPLE = "#1f6feb", "#8250df"

    # ---------------- parallel ----------------
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    _lattice(ax, B)
    det = xtk.DetectorSpec(size=(2.2 * B,), num_cell=(9,))
    rays = xtk.struct_rays(xtk.parallel_beam(Float([0.0]), det))
    t = np.asarray(rays[0]).T.reshape(-1, 2)
    n = np.asarray(rays[1]).T.reshape(-1, 2)
    L = 3.0 * B                                   # every ray the same length
    for p, d in zip(t, n):
        _ray(ax, p - 0.5 * L * d, p + 0.5 * L * d, BLUE)
    ax.plot([0.5 * L, 0.5 * L], [-1.2 * B, 1.2 * B], color=BLUE,
            lw=4.0, solid_capstyle="butt", zorder=3)
    ax.text(0.5 * L + 0.20, 0, "detector", rotation=90, va="center", fontsize=9,
            color=BLUE)
    _rotation(ax, 2.05 * B, 0.42 * np.pi, 0.92 * np.pi, label_at=(-0.22, 0.30))
    ax.set_xlim(-2.35, 2.35); ax.set_ylim(-2.0, 2.75)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "schem_parallel.svg")

    # ---------------- cone ----------------
    fig, ax = plt.subplots(figsize=(4.7, 3.6))
    _lattice(ax, B)
    sod, sdd = 2.5 * B, 4.5 * B
    rays = xtk.struct_rays(xtk.cone_beam(
        sod=sod, sdd=sdd, angles=Float([0.0]),
        detector_spec=xtk.DetectorSpec(size=(2.7 * B,), num_cell=(9,))))
    t = np.asarray(rays[0]).T.reshape(-1, 2)
    n = np.asarray(rays[1]).T.reshape(-1, 2)
    src = t[0]
    xd = src[0] + sdd
    ends = []
    for d in n:
        y = src[1] + (xd - src[0]) / d[0] * d[1]
        ends.append(y)
        _ray(ax, src, (xd, y), PURPLE, lw=0.95, dot=0.0)
    ax.plot(*src, "o", color=PURPLE, ms=7, zorder=5)
    ax.text(src[0] + 0.10, src[1] + 0.26, "source", fontsize=9, color=PURPLE)
    h = max(abs(min(ends)), abs(max(ends)))
    ax.plot([xd, xd], [-h, h], color=PURPLE, lw=4.0, solid_capstyle="butt", zorder=3)
    ax.text(xd + 0.18, 0, "detector", rotation=90, va="center", fontsize=9,
            color=PURPLE)
    for y, ab, lab in ((-h - 0.34, (src[0], 0.0), "sod"),
                       (-h - 0.80, (src[0], xd), "sdd")):
        ax.annotate("", xy=(ab[1], y), xytext=(ab[0], y),
                    arrowprops=dict(arrowstyle="<->", color="#57606a", lw=1.0))
        ax.text(sum(ab) / 2, y - 0.26, lab, ha="center", fontsize=9, color="#57606a")
    _rotation(ax, 1.62 * B, 0.28 * np.pi, 0.86 * np.pi, label_at=(-0.62, 0.06))
    ax.set_xlim(src[0] - 0.5, xd + 0.85)
    ax.set_ylim(-h - 1.25, max(h, 1.55 * B) + 0.75)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "schem_cone.svg")

    # ---------------- arbitrary ----------------
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    _lattice(ax, B)
    spec = [((-1.55, 0.62), 0.10, "#1f6feb"),
            ((0.05, -1.62), 1.42, "#d1242f"),
            ((-1.30, -1.05), 0.62, "#2da44e"),
            ((1.62, 0.10), 2.55, "#8250df"),
            ((-0.35, 1.58), 5.05, "#bf8700")]
    for (p0, a, col) in spec:
        d = np.array([np.cos(a), np.sin(a)])
        _ray(ax, p0, np.asarray(p0) + 3.1 * B * d, col, lw=1.6, dot=5.0, head=9.0)
    ax.set_xlim(-2.25, 2.35); ax.set_ylim(-2.25, 2.35)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "schem_explicit.svg")


def bases():
    """Box-spline basis of each order and its projection."""
    ang = np.linspace(0, np.pi, 5, endpoint=False)
    ray_n = Array2f(np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32))
    for order in (0, 1, 2):
        fig = xtk.plot_2d_basis(knot, order, ray_n)
        fig.suptitle(f"order = {order}")
        save(fig, f"basis_{order}.png")


def forward_adjoint():
    """Image -> sinogram -> backprojection, the operator pair."""
    yy, xx = np.mgrid[:N, :N]
    img = ((((xx - 64) / 40) ** 2 + ((yy - 64) / 52) ** 2) < 1).astype(np.float32)
    img -= 0.5 * ((((xx - 52) / 12) ** 2 + ((yy - 56) / 18) ** 2) < 1)
    img += 0.7 * ((((xx - 82) / 9) ** 2 + ((yy - 74) / 9) ** 2) < 1)
    f = Float(img.reshape(-1))
    n_ang, n_det = 180, 192
    rays = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, n_ang, endpoint=False),
                             xtk.DetectorSpec(size=(1.5 * N,), num_cell=(n_det,)))
    re = xtk.struct_rays(rays)
    sino = xtk.xrt_apply(re, knot, 0, f)
    bp = xtk.xrt_adjoint(re, knot, 0, sino)

    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    axs[0].imshow(img, cmap="gray"); axs[0].set_title("image $f$")
    axs[1].imshow(np.asarray(sino).reshape(n_ang, n_det), cmap="gray", aspect="auto")
    axs[1].set_title(r"forward $\mathbf{A}f$ (sinogram)")
    axs[2].imshow(np.asarray(bp).reshape(N, N), cmap="gray")
    axs[2].set_title(r"adjoint $\mathbf{A}^{\top}\mathbf{A}f$")
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save(fig, "forward_adjoint.png")

    rec_fbp = xtk.fbp(rays, knot, sino, window="ramp")
    rec_cg = xtk.cg(lambda v: xtk.xrt_apply(re, knot, 0, v),
                    lambda v: xtk.xrt_adjoint(re, knot, 0, v), sino, N * N, n_iter=25)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (im, ti) in zip(axs, ((img, "phantom"),
                                  (np.asarray(rec_fbp).reshape(N, N), "fbp"),
                                  (np.asarray(rec_cg).reshape(N, N), "cg, 25 iterations"))):
        ax.imshow(im, cmap="gray", vmin=-0.1, vmax=1.7); ax.set_title(ti)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save(fig, "reconstruction.png")


def _ellipsoids(N):
    """A 3D Shepp-Logan-like phantom."""
    z, y, x = (np.mgrid[:N, :N, :N] / (N / 2) - 1).astype(np.float32)
    v = np.zeros_like(x)
    for cx, cy, cz, ax, ay, az, val in (
            (0, 0, 0, .69, .90, .88, 1.0), (0, -.0184, 0, .66, .87, .84, -.8),
            (.22, 0, 0, .11, .31, .22, -.2), (-.22, 0, 0, .16, .41, .28, -.2),
            (0, .35, -.15, .21, .25, .41, .1), (0, .1, .25, .046, .046, .046, .1),
            (-.08, -.605, 0, .046, .023, .02, .1), (.06, -.605, 0, .056, .04, .1, .1),
            (0, -.606, .35, .1, .056, .04, .15), (.15, .30, -.30, .07, .07, .07, .2)):
        v += val * (((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2
                    + ((z - cz) / az) ** 2 <= 1)
    return np.clip(v, 0, None)


def gallery_cone():
    """A cone-beam scan of a 3D phantom, reconstructed with FDK."""
    N = 192
    ph = _ellipsoids(N)
    ks = xtk.UniformSpec(start=(-N / 2 + 0.5,) * 3, step=1, num=(N,) * 3)
    sod, sdd = 8.0 * N, 10.0 * N          # a gentle cone: 4 degree half-angle
    rays = xtk.cone_beam(sod=sod, sdd=sdd,
                         angles=dr.linspace(Float, 0, 2 * np.pi, 720, endpoint=False),
                         detector_spec=xtk.DetectorSpec(size=(1.6 * N, 1.6 * N),
                                                        num_cell=(384, 384)))
    y = xtk.xrt_apply(xtk.struct_rays(rays), ks, 0, Float(ph.reshape(-1)))
    rec = np.asarray(xtk.fbp_cone(rays, ks, y, sod=sod, sdd=sdd,
                                  window="shepp-logan")).reshape(N, N, N)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (sl, ti) in zip(axs, ((rec[:, :, N // 2], "axial"),
                                  (rec[:, N // 2, :], "coronal"),
                                  (rec[N // 2, :, :], "sagittal"))):
        ax.imshow(sl.T, cmap="gray", vmin=0, vmax=1.05)
        ax.set_title(ti)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save(fig, "gallery_cone.png")


def gallery_sparse():
    """Twenty views: analytic against iterative."""
    yy, xx = np.mgrid[:N, :N]
    ph = _ellipsoids(N)[N // 2]
    f = Float(ph.reshape(-1))
    rays = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, 20, endpoint=False),
                             xtk.DetectorSpec(size=(1.5 * N,), num_cell=(192,)))
    re = xtk.struct_rays(rays)
    y = xtk.xrt_apply(re, knot, 0, f)
    a = np.asarray(xtk.fbp(rays, knot, y, window="hann")).reshape(N, N)
    b = np.asarray(xtk.cg(lambda v: xtk.xrt_apply(re, knot, 0, v),
                          lambda v: xtk.xrt_adjoint(re, knot, 0, v),
                          y, N * N, n_iter=40)).reshape(N, N)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, (im, ti) in zip(axs, ((ph, "phantom"), (a, "fbp, 20 views"),
                                  (b, "cg, 20 views"))):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1.05)
        ax.set_title(ti); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save(fig, "gallery_sparse.png")


def gallery_calibration():
    """Recover an unknown detector shift by gradient descent."""
    from drjit.cuda.ad import Array2f
    ph = _ellipsoids(N)[N // 2]
    f = Float(ph.reshape(-1))
    n_ang, n_det = 180, 192
    a = np.linspace(0, np.pi, n_ang, endpoint=False)
    u = np.linspace(-0.75 * N, 0.75 * N, n_det)
    ca, sa = np.cos(a), np.sin(a)
    t0 = np.stack([-u[None] * sa[:, None], u[None] * ca[:, None]], 0).reshape(2, -1)
    n0 = np.stack([np.broadcast_to(ca[:, None], (n_ang, n_det)),
                   np.broadcast_to(sa[:, None], (n_ang, n_det))], 0).reshape(2, -1)
    ux, uy = -n0[1], n0[0]

    def rays_at(s):
        return (Array2f(t0[0] + s * ux, t0[1] + s * uy), Array2f(n0))

    true_shift = 1.5
    y_meas = xtk.xrt_apply(rays_at(true_shift), knot, 1, f)
    s, lr, hist = 0.0, 2e-6, []
    for _ in range(40):
        rays = rays_at(s)
        resid = xtk.xrt_apply(rays, knot, 1, f) - y_meas
        gx = xtk.xrt_ad_t_x(rays, knot, 1, f)
        gy = xtk.xrt_ad_t_y(rays, knot, 1, f)
        s -= lr * float(dr.sum(resid * (Float(ux) * gx + Float(uy) * gy)).item())
        hist.append(s)

    def rec(shift):
        r = rays_at(shift)
        return np.asarray(xtk.cg(lambda v: xtk.xrt_apply(r, knot, 1, v),
                                 lambda v: xtk.xrt_adjoint(r, knot, 1, v),
                                 y_meas, N * N, n_iter=25)).reshape(N, N)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
    axs[0].imshow(rec(0.0), cmap="gray", vmin=0, vmax=1.05)
    axs[0].set_title("assumed shift 0")
    axs[1].imshow(rec(s), cmap="gray", vmin=0, vmax=1.05)
    axs[1].set_title(f"fitted shift {s:.3f}")
    for ax in axs[:2]:
        ax.set_xticks([]); ax.set_yticks([])
    axs[2].plot(hist, color="crimson")
    axs[2].axhline(true_shift, ls=":", color="0.4")
    axs[2].set_xlabel("iteration"); axs[2].set_ylabel("detector shift [voxels]")
    axs[2].set_title("gradient descent on the geometry")
    plt.tight_layout()
    save(fig, "gallery_calibration.png")
    print(f"   fitted shift {s:.4f} (true {true_shift})")


def gallery_walnut():
    """Real data: the public Walnut1 cone-beam scan, read through from_astra.

    The dataset is not part of this repository. Download Walnut1 from
    https://doi.org/10.5281/zenodo.2686726 and point WALNUT at it.
    """
    import glob
    import tifffile as tf

    WALNUT = pathlib.Path(__file__).resolve().parent.parent / \
        "src/calibration_data/Walnut1/Projections"
    if not WALNUT.is_dir():
        print("skipping gallery_walnut: Walnut1 not found")
        return

    STEP, BIN = 6, 2
    vox_mm = 0.1496 * BIN / 3.015181           # detector pitch / magnification
    # the scanner writes each frame transposed and flipped, and pairs the
    # projections with the geometry rows in reverse order
    trafo = lambda im: np.transpose(np.flipud(im))

    Ls, Gs = [], []
    for tube in ("tubeV1", "tubeV2", "tubeV3"):
        W = WALNUT / tube
        dark = trafo(tf.imread(str(W / "di000000.tif"))).astype(np.float32)
        flat = np.mean([trafo(tf.imread(str(W / f"io00000{i}.tif"))).astype(np.float32)
                        for i in (0, 1)], 0)
        files = sorted(glob.glob(str(W / "scan_*.tif")))[::-1][::STEP]
        g = np.loadtxt(str(W / "scan_geom_corrected.geom"))[::STEP]
        n = min(len(files), len(g))
        P = np.stack([trafo(tf.imread(f)).astype(np.float32) for f in files[:n]])
        L = -np.log(np.clip((P - dark) / np.maximum(flat - dark, 1.0), 1e-4, 1.0))
        Ls.append(L.reshape(len(L), 972 // BIN, BIN, 768 // BIN, BIN).mean((2, 4)))
        g = g[:n]
        g[:, 0:6] /= vox_mm                    # positions  -> voxel units
        g[:, 6:12] *= BIN / vox_mm             # detector axes -> voxel units
        Gs.append(g)

    L = np.concatenate(Ls)
    g = np.concatenate(Gs)
    n_view, n_v, n_u = L.shape
    Nw = 500
    rays, ks = xtk.from_astra(
        {"type": "cone_vec", "DetectorRowCount": n_v,
         "DetectorColCount": n_u, "Vectors": g},
        {"GridColCount": Nw, "GridRowCount": Nw, "GridSliceCount": Nw,
         "option": {"WindowMinX": -Nw/2, "WindowMaxX": Nw/2,
                    "WindowMinY": -Nw/2, "WindowMaxY": Nw/2,
                    "WindowMinZ": -Nw/2, "WindowMaxZ": Nw/2}})
    # from_astra orders 3-D rays as (det_v, angles, det_u)
    y = Float(np.ascontiguousarray(np.transpose(L, (1, 0, 2)).reshape(-1)))
    del L
    rec = xtk.cg(lambda v: xtk.xrt_apply(rays, ks, 0, v),
                 lambda v: xtk.xrt_adjoint(rays, ks, 0, v), y, Nw ** 3, n_iter=30)
    r = np.asarray(rec).reshape(Nw, Nw, Nw)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    hi = np.percentile(r, 99.8)
    for ax, (sl, ti) in zip(axs, ((r[:, :, Nw//2], "axial"),
                                  (r[:, Nw//2, :], "coronal"),
                                  (r[Nw//2, :, :], "sagittal"))):
        ax.imshow(sl.T, cmap="gray", vmin=0, vmax=hi)
        ax.set_title(ti)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save(fig, "gallery_walnut.png")
    print(f"   {n_view} views, {dr.width(rays[0])/1e6:.0f}M rays")


if __name__ == "__main__":
    geometry_schematics()
    bases()
    forward_adjoint()
    gallery_cone()
    gallery_sparse()
    gallery_calibration()
    gallery_walnut()
