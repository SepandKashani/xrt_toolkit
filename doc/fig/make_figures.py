"""
Figures of the Geometry page. The rays are XTK's own (struct_rays), on the CPU.

    python doc/fig/make_figures.py
"""

import pathlib
import sys

import drjit as dr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from drjit.llvm import Float  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import xrt_toolkit as xtk  # noqa: E402

OUT = pathlib.Path(__file__).parent
EXT = sys.argv[1] if len(sys.argv) > 1 else "svg"

INK = "#222222"
GREY = "#9a9a9a"
LIGHT = "#ececec"
RAY = "#336699"

plt.rcParams.update(
    {
        "font.size": 10,
        "mathtext.fontset": "dejavusans",
        "lines.solid_capstyle": "butt",
        "svg.fonttype": "path",
        "svg.hashsalt": "xtk",
    }
)


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name, tight=True):
    # 3D axes keep their whole box under bbox_inches="tight": crop them with the figure size instead
    meta = {"Date": None} if (EXT == "svg") else {}
    fig.savefig(
        OUT / f"{name}.{EXT}",
        bbox_inches="tight" if tight else None,
        pad_inches=0.02,
        dpi=150,
        metadata=meta,
    )
    plt.close(fig)


def arrow(ax, p, d, label=None, off=(0, 0), color=INK, lw=1.0, ha="center"):
    p, d = np.asarray(p, float), np.asarray(d, float)
    ax.annotate(
        "",
        xy=p + d,
        xytext=p,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
    )
    if label:
        q = p + d + np.asarray(off, float)
        ax.text(*q, label, color=color, ha=ha, va="center")


def span(ax, a, b, label, off, side=1):
    # dimension line between a and b, shifted by `off`, label on `side`
    a, b, off = (np.asarray(v, float) for v in (a, b, off))
    ax.annotate(
        "",
        xy=b + off,
        xytext=a + off,
        arrowprops=dict(arrowstyle="<|-|>", color=GREY, lw=0.8, shrinkA=0, shrinkB=0),
    )
    u = off / np.linalg.norm(off)
    for p in (a, b):
        ax.plot(*np.c_[p + off - 0.3 * u, p + off + 0.3 * u], color=GREY, lw=0.6)
    ax.text(
        *((a + b) / 2 + off + side * 0.45 * u),
        label,
        ha="center",
        va="center",
        color=INK,
    )


def lattice(ax, num, step=(1.0, 1.0), lines=True):
    # cells of a centered 2D lattice, light fill, cell borders
    lo = -np.asarray(step) * np.asarray(num) / 2
    hi = -lo
    ax.add_patch(plt.Rectangle(lo, *(hi - lo), fc=LIGHT, ec=GREY, lw=0.8, zorder=0))
    if lines:
        for k in range(1, num[0]):
            x = lo[0] + k * step[0]
            ax.plot([x, x], [lo[1], hi[1]], color="white", lw=0.8, zorder=0)
        for k in range(1, num[1]):
            y = lo[1] + k * step[1]
            ax.plot([lo[0], hi[0]], [y, y], color="white", lw=0.8, zorder=0)
    return lo, hi


def rays(ray_spec):
    t, n = (np.asarray(v).T for v in xtk.struct_rays(ray_spec))
    return t, n


# Volume --------------------------------------------------------------------------------------
def fig_volume():
    Q = (4, 3)
    fig, ax = canvas(4.2, 3.4, (-1.4, 4.0), (-1.2, 3.3))
    for q1 in range(Q[0]):
        for q2 in range(Q[1]):
            ax.add_patch(
                plt.Rectangle((q1 - 0.5, q2 - 0.5), 1, 1, fc=LIGHT, ec="white", lw=1.5)
            )
            ax.plot(q1, q2, "o", ms=3.5, color=INK)
            ax.text(q1 + 0.12, q2 + 0.14, f"{q1 * Q[1] + q2}", color=GREY, fontsize=8)
    ax.text(-0.1, -0.22, r"$\mathbf{x}_0$", ha="right", va="center")
    span(ax, (0, 2), (1, 2), r"$\Delta_1$", (0, 0.8))
    span(ax, (3, 1), (3, 2), r"$\Delta_2$", (0.85, 0))
    arrow(ax, (-1.0, -0.9), (0.9, 0), r"$x$", off=(0.2, 0))
    arrow(ax, (-1.0, -0.9), (0, 0.9), r"$y$", off=(0, 0.2))
    save(fig, "volume")


# Basis functions -----------------------------------------------------------------------------
def fig_basis():
    # 2D box splines (step 1): the voxel, then convolved with segments along (1, 1), (1, -1)
    h, M = 0.01, 250
    g = np.arange(-M, M + 1) * h
    X, Y = np.meshgrid(g, g, indexing="ij")
    psi = [((np.abs(X) <= 0.5) & (np.abs(Y) <= 0.5)).astype(float)]
    for d in ((1, 1), (1, -1)):
        K = int(round(1 / h))
        acc = sum(
            np.roll(psi[-1], (k * d[0], k * d[1]), axis=(0, 1))
            for k in range(-K // 2, K // 2 + 1)
        )
        psi.append(acc / (K + 1))

    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.4))
    for order, (ax, p) in enumerate(zip(axs, psi)):
        ax.imshow(
            p.T,
            origin="lower",
            extent=(g[0], g[-1]) * 2,
            cmap="Greys",
            vmin=0,
            vmax=1.15 * p.max(),
        )
        for v in np.arange(-1.5, 2.0, 1.0):
            ax.axvline(v, color=GREY, lw=0.5)
            ax.axhline(v, color=GREY, lw=0.5)
        k = np.arange(-2, 3)
        ax.plot(*np.meshgrid(k, k), "o", ms=2, color=GREY)
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-2.4, 2.4)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"order {order}", fontsize=10)
    fig.tight_layout()
    save(fig, "basis")


# Rays ----------------------------------------------------------------------------------------
def fig_ray():
    fig, ax = canvas(4.0, 3.0, (-5.2, 5.2), (-3.8, 3.8))
    lattice(ax, (6, 6))
    n = np.array([np.cos(0.45), np.sin(0.45)])
    t = np.array([-4.2, -2.6])
    a = np.r_[-1.5, 12.0]
    ax.plot(*(t[:, None] + n[:, None] * a), color=RAY, lw=1.4)
    ax.plot(*t, "o", ms=4.5, color=INK, zorder=3)
    ax.text(*(t + (0.15, -0.45)), r"$\mathbf{t}$", ha="center")
    arrow(ax, t, 1.6 * n, r"$\mathbf{n}$", off=(-0.35, 0.4))
    save(fig, "ray")


# Detector ------------------------------------------------------------------------------------
def fig_detector():
    fig, axs = plt.subplots(
        1, 2, figsize=(6.6, 2.0), gridspec_kw=dict(width_ratios=(1.2, 1))
    )

    # 1D: num_cell = 4, size = 4
    ax = axs[0]
    ax.set_xlim(-2.6, 2.9)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for i in range(4):
        c = i - 1.5
        ax.add_patch(plt.Rectangle((c - 0.5, -0.2), 1, 0.4, fc=LIGHT, ec=INK, lw=0.8))
        ax.text(c, 0, f"{i}", ha="center", va="center", fontsize=9)
    ax.plot(0, -0.2, "^", ms=5, color=RAY, clip_on=False)
    ax.text(0, -0.55, r"$u_1 = 0$", ha="center", va="top", color=RAY)
    span(ax, (-2, 0.2), (2, 0.2), "size", (0, 0.55))
    span(ax, (1, -0.2), (2, -0.2), "size / num_cell", (0, -0.55), side=1)
    arrow(ax, (2.25, 0), (0.6, 0), r"$u_1$", off=(0.2, 0), ha="left")

    # 2D: num_cell = (4, 2), u2 fastest
    ax = axs[1]
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    for i1 in range(4):
        for i2 in range(2):
            c = (i1 - 1.5, (i2 - 0.5) * 0.9)
            ax.add_patch(
                plt.Rectangle(
                    (c[0] - 0.5, c[1] - 0.45), 1, 0.9, fc=LIGHT, ec=INK, lw=0.8
                )
            )
            ax.text(*c, f"{i1 * 2 + i2}", ha="center", va="center", fontsize=9)
    arrow(ax, (-2.0, -1.15), (1.0, 0), r"$u_1$", off=(0.25, 0))
    arrow(ax, (-2.25, -0.9), (0, 0.9), r"$u_2$", off=(0, 0.22))
    fig.tight_layout()
    save(fig, "detector")


# Parallel beam -------------------------------------------------------------------------------
def fig_parallel():
    th = np.radians(30)
    det = xtk.util.DetectorSpec(size=(10.0,), num_cell=(8,))
    t, n = rays(xtk.parallel_beam(Float([th]), det))
    fig, ax = canvas(4.4, 4.0, (-7.2, 7.2), (-6.4, 6.4))
    lattice(ax, (8, 8), lines=False)
    for tl, nl in zip(t, n):
        ax.plot(*np.c_[tl - 9 * nl, tl + 9 * nl], color=RAY, lw=0.8, zorder=1)
    u1 = np.array([-np.sin(th), np.cos(th)])
    ends = np.c_[t[0] - 0.625 * u1, t[-1] + 0.625 * u1]
    ax.plot(*ends, color=INK, lw=3, zorder=2)
    arrow(ax, (-6.6, 0), (13.2, 0), r"$x$", off=(0.3, 0), color=GREY, lw=0.6)
    arrow(ax, (0, -6.2), (0, 12.4), r"$y$", off=(0, 0.3), color=GREY, lw=0.6)
    arrow(ax, t[-1] + 0.625 * u1, 1.4 * u1, r"$u_1$", off=(-0.35, 0.15))
    arrow(ax, (0, 0), 3.0 * n[0], r"$\mathbf{n}$", off=(0.3, 0.15))
    a = np.linspace(0, th, 30)
    ax.plot(2.2 * np.cos(a), 2.2 * np.sin(a), color=INK, lw=0.8)
    ax.text(
        2.5 * np.cos(th / 2), 2.5 * np.sin(th / 2), r"$\theta$", ha="left", va="center"
    )
    save(fig, "parallel")


# Cone beam -----------------------------------------------------------------------------------
def fig_cone():
    sod, sdd = 11.0, 19.0
    det = xtk.util.DetectorSpec(size=(16.0,), num_cell=(8,))
    t, n = rays(xtk.cone_beam(sod, sdd, Float([0.0]), det))
    fig, ax = canvas(5.6, 3.6, (-12.6, 10.6), (-11.6, 9.0))
    lattice(ax, (6, 6), lines=False)
    src = t[0]
    for nl in n:
        ax.plot(*np.c_[src, src + nl], color=RAY, lw=0.8, zorder=1)
    x_det = sdd - sod
    ax.plot([x_det, x_det], [-8, 8], color=INK, lw=3, zorder=2)
    ax.plot(*src, "o", ms=5, color=INK, zorder=3)
    ax.text(*(src + (0, -1.0)), "source", ha="center", va="top")
    arrow(ax, (-12.0, 0), (21.6, 0), r"$x$", off=(0.5, 0), color=GREY, lw=0.6)
    arrow(ax, (0, -8.4), (0, 16.8), r"$y$", off=(0, 0.5), color=GREY, lw=0.6)
    arrow(ax, (x_det, 8.0), (0, 1.6), r"$u_1$", off=(0.7, -0.2))
    span(ax, src, (0, 0), "sod", (0, 6.2))
    span(ax, src, (x_det, 0), "sdd", (0, -10.2), side=-1)
    save(fig, "cone")


# 3D ------------------------------------------------------------------------------------------
def fig_cone3d():
    sod, sdd = 11.0, 19.0
    det = xtk.util.DetectorSpec(size=(19.0, 15.0), num_cell=(4, 3))
    t, n = rays(xtk.cone_beam(sod, sdd, Float([0.0]), det))
    fig = plt.figure(figsize=(4.6, 3.1))
    ax = fig.add_axes((-0.12, -0.2, 1.24, 1.45), projection="3d", proj_type="ortho")
    ax.view_init(elev=16, azim=-62)
    ax.set_box_aspect((30, 22, 18))
    ax.axis("off")

    # volume
    r = 4.0
    for a in (-r, r):
        for b in (-r, r):
            ax.plot([-r, r], [a, a], [b, b], color=GREY, lw=0.7)
            ax.plot([a, a], [-r, r], [b, b], color=GREY, lw=0.7)
            ax.plot([a, a], [b, b], [-r, r], color=GREY, lw=0.7)
    # rays to the corner cells
    src = t[0]
    for nl in n[[0, 2, 9, 11]]:
        ax.plot(*np.c_[src, src + nl], color=RAY, lw=0.7)
    ax.scatter(*src, color=INK, s=12, depthshade=False)
    ax.text(*(src + (0, 0, -2.6)), "source", ha="center")
    # detector plane, u1 and u2 along its edges from the corner
    x = sdd - sod
    hw, hh = 19.0 / 2, 15.0 / 2
    corners = [(x, -hw, -hh), (x, hw, -hh), (x, hw, hh), (x, -hw, hh)]
    ax.add_collection3d(
        Poly3DCollection([corners], fc=LIGHT, ec=INK, lw=1.6, alpha=0.7)
    )
    kw = dict(arrow_length_ratio=0.2, color=INK, lw=1.0)
    ax.quiver(x, -hw, -hh - 2.5, 0, 7, 0, **kw)
    ax.text(x, -hw + 8.0, -hh - 4.5, r"$u_1$")
    ax.quiver(x, hw + 2.5, -hh, 0, 0, 7, **kw)
    ax.text(x, hw + 3.3, -hh + 8.0, r"$u_2$", ha="left", zorder=10)
    # rotation axis
    ax.plot([0, 0], [0, 0], [-9, 10], color=GREY, lw=0.6)
    ax.quiver(0, 0, 9, 0, 0, 2.5, arrow_length_ratio=0.4, color=GREY, lw=0.6)
    ax.text(0.4, 0, 12.0, r"$z$", color=GREY)
    ax.set_xlim(-12, 9)
    ax.set_ylim(-11, 15)
    ax.set_zlim(-9, 9)
    save(fig, "cone3d", tight=False)


if __name__ == "__main__":
    for f in (
        fig_volume,
        fig_basis,
        fig_ray,
        fig_detector,
        fig_parallel,
        fig_cone,
        fig_cone3d,
    ):
        f()
