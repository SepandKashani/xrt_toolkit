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


def save(fig, name, dpi=110):
    if name.endswith(".png"):
        dpi = max(dpi, 200)          # displayed near 1:1, so give it real pixels
    fig.savefig(OUT / name, dpi=dpi, bbox_inches="tight", facecolor="white")
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


def _cube(ax, B, color="#8b949e", lw=0.9):
    """Wireframe of the reconstruction lattice."""
    c = np.array([-B, B])
    for i in (0, 1):
        for j in (0, 1):
            ax.plot(c, [c[i]] * 2, [c[j]] * 2, color=color, lw=lw, zorder=1)
            ax.plot([c[i]] * 2, c, [c[j]] * 2, color=color, lw=lw, zorder=1)
            ax.plot([c[i]] * 2, [c[j]] * 2, c, color=color, lw=lw, zorder=1)


def _ray3(ax, p0, p1, color, lw=1.1, dot=8, head=0.26):
    """3-D ray: dot at the start, arrowhead of fixed size ending exactly at p1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    n = np.linalg.norm(d)
    d = d / max(n, 1e-12)
    q = p1 - min(head, 0.4 * n) * d
    ax.plot(*zip(p0, q), color=color, lw=lw, zorder=3)
    ax.quiver(*q, *(p1 - q), color=color, lw=lw, arrow_length_ratio=1.0,
              zorder=3)
    if dot:
        ax.scatter(*p0, color=color, s=dot, depthshade=False, zorder=4)


def _plane(ax, centre, u, v, hu, hv, color, alpha=0.20):
    """Filled detector rectangle spanned by u, v."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    c, u, v = (np.asarray(x, float) for x in (centre, u, v))
    corners = [c - hu * u - hv * v, c + hu * u - hv * v,
               c + hu * u + hv * v, c - hu * u + hv * v]
    ax.add_collection3d(Poly3DCollection([corners], facecolor=color, alpha=alpha,
                                        edgecolor=color, lw=1.2))


def _tidy3(ax, lim, elev=20, azim=-58):
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


GREEN, BLUE, ORANGE, GREY, INK = "#1a7f37", "#1f6feb", "#bc4c00", "#57606a", "#24292f"


def _span(ax, p0, p1, label, color, off, fs=9.5, rot=0):
    """Double-headed dimension arrow with its label."""
    ax.annotate("", xy=tuple(p1), xytext=tuple(p0),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.0))
    mid = 0.5 * (np.asarray(p0, float) + np.asarray(p1, float)) + np.asarray(off, float)
    ax.text(*mid, label, color=color, fontsize=fs, ha="center", va="center",
            rotation=rot)


def _axis_arrow(ax, p0, p1, label, color, off, fs=10):
    ax.annotate("", xy=tuple(p1), xytext=tuple(p0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4,
                                mutation_scale=11))
    ax.text(*(np.asarray(p1, float) + np.asarray(off, float)), label, color=color,
            fontsize=fs, ha="center", va="center")


def _lattice_panel(ax, nh, nv, h_span, v_span, h_axis, v_axis, ch, cv,
                   title, out_of_page=None):
    """One volume-lattice drawing: grid, start, step, and the two axes shown."""
    W, H = float(nh), float(nv)
    ax.add_patch(plt.Rectangle((0, 0), W, H, facecolor="#e6f4ea", edgecolor=GREEN,
                               lw=1.4))
    for k in range(1, nh):
        ax.plot([k, k], [0, H], color=GREEN, lw=0.7, alpha=0.55)
    for k in range(1, nv):
        ax.plot([0, W], [k, k], color=GREEN, lw=0.7, alpha=0.55)
    cx, cy = np.arange(nh) + 0.5, np.arange(nv) + 0.5
    X, Y = np.meshgrid(cx, cy)
    ax.plot(X.ravel(), Y.ravel(), "o", color=GREEN, ms=3.0, alpha=0.55)

    ax.plot(cx[0], cy[0], "o", color=GREEN, ms=7)
    ax.annotate("start", xy=(cx[0], cy[0]), xytext=(cx[0] - 1.45, cy[0] - 1.0),
                color=GREEN, fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=GREEN, lw=0.9))
    _span(ax, (cx[0], cy[-1]), (cx[1], cy[-1]), "step", GREEN, (0, 0.34), fs=9)

    _span(ax, (0, H + 0.55), (W, H + 0.55), h_span, ch, (0, 0.44), fs=9)
    _span(ax, (-0.55, 0), (-0.55, H), v_span, cv, (-0.44, 0), fs=9, rot=90)
    _axis_arrow(ax, (0, -0.85), (0.85 * W, -0.85), h_axis, ch, (0, -0.5), fs=9.5)
    _axis_arrow(ax, (W + 0.85, 0), (W + 0.85, 0.85 * H), v_axis, cv, (1.15, 0.3),
                fs=9.5)
    if out_of_page:
        y = H + 1.75
        ax.plot(0.45, y, "o", mfc="white", mec=GREY, ms=11, mew=1.2)
        ax.plot(0.45, y, ".", color=GREY, ms=4)
        ax.text(0.95, y, out_of_page, color=GREY, fontsize=9, va="center")
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_xlim(-2.7, W + 3.5)
    ax.set_ylim(-2.3, H + (2.6 if out_of_page else 1.6))
    ax.set_aspect("equal"); ax.set_anchor("N"); ax.axis("off")


def lattice_figure():
    """The voxel lattice: 2-D on the left, 3-D on the right."""
    fig = plt.figure(figsize=(9.8, 4.3))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.02)

    _lattice_panel(fig.add_subplot(gs[0]), 3, 5,
                   "num[0] cells", "num[1] cells",
                   "axis 1", "axis 2  (u1)", GREY, BLUE, "2-D volume")
    _lattice_panel(fig.add_subplot(gs[1]), 5, 4,
                   "num[1] cells", "num[2] cells",
                   "axis 2  (u1)", "axis 3  (u2)", BLUE, ORANGE, "3-D volume",
                   out_of_page="axis 1  (num[0]), out of the page")
    save(fig, "schem_lattice.svg")


def detector_spec_figure():
    """What size and num_cell mean, in 1-D and 2-D detectors."""
    fig = plt.figure(figsize=(10.4, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.04)
    aspect = 0.63          # same box height for both panels

    # ---- 1-D detector (2-D scan) ----
    ax = fig.add_subplot(gs[0])
    W, NU = 12.0, 6
    cu = W / NU
    c = (np.arange(NU) - (NU - 1) / 2) * cu
    ax.add_patch(plt.Rectangle((-W / 2, -0.5), W, 1.0, facecolor="#f6f8fa",
                               edgecolor=GREY, lw=1.3))
    for e in np.linspace(-W / 2, W / 2, NU + 1)[1:-1]:
        ax.plot([e, e], [-0.5, 0.5], color=GREY, lw=0.8)
    ax.plot(c, np.zeros_like(c), "o", color=INK, ms=4)

    _span(ax, (-W / 2, 1.95), (W / 2, 1.95), "size[0]", BLUE, (0, 0.55))
    _span(ax, (c[3] - cu / 2, -1.15), (c[3] + cu / 2, -1.15), "cell_size", BLUE,
          (0, -0.55), fs=9)
    _axis_arrow(ax, (W / 2 + 0.7, 0), (W / 2 + 2.1, 0), "u1", BLUE, (0.0, 0.55))
    ax.plot(0, 0, "+", color=INK, ms=12, mew=1.6)
    ax.text(0, 0.85, "beam axis", ha="center", fontsize=9, color=GREY)
    ax.set_title("1-D detector  —  2-D scan", fontsize=10.5, color=INK)
    x0, x1 = -W / 2 - 1.4, W / 2 + 3.0
    ax.set_xlim(x0, x1); ax.set_ylim(-aspect * (x1 - x0) / 2, aspect * (x1 - x0) / 2)
    ax.set_aspect("equal"); ax.axis("off")

    # ---- 2-D detector (3-D scan) ----
    ax = fig.add_subplot(gs[1])
    W, H, NU, NV = 12.0, 8.0, 6, 4
    cu, cv = W / NU, H / NV
    ax.add_patch(plt.Rectangle((-W / 2, -H / 2), W, H, facecolor="#f6f8fa",
                               edgecolor=GREY, lw=1.3))
    for e in np.linspace(-W / 2, W / 2, NU + 1)[1:-1]:
        ax.plot([e, e], [-H / 2, H / 2], color=GREY, lw=0.8)
    for e in np.linspace(-H / 2, H / 2, NV + 1)[1:-1]:
        ax.plot([-W / 2, W / 2], [e, e], color=GREY, lw=0.8)
    gu = (np.arange(NU) - (NU - 1) / 2) * cu
    gv = (np.arange(NV) - (NV - 1) / 2) * cv
    U, V = np.meshgrid(gu, gv, indexing="ij")
    ax.plot(U.ravel(), V.ravel(), "o", color=INK, ms=3.0)
    for i2 in range(NV):                       # u2 is the fast axis
        ax.text(gu[0], gv[i2] + 0.52, str(i2), ha="center", fontsize=8, color=GREY)
        ax.text(gu[1], gv[i2] + 0.52, str(NV + i2), ha="center", fontsize=8,
                color=GREY)
    ax.text(gu[-1], gv[-1] + 0.52, str(NU * NV - 1), ha="center", fontsize=8,
            color=GREY)

    _span(ax, (-W / 2, H / 2 + 1.0), (W / 2, H / 2 + 1.0),
          "size[0], num_cell[0] cells", BLUE, (0, 0.55))
    _span(ax, (-W / 2 - 1.0, -H / 2), (-W / 2 - 1.0, H / 2),
          "size[1], num_cell[1] cells", ORANGE, (-0.55, 0), rot=90)
    _axis_arrow(ax, (W / 2 + 0.8, -H / 2), (W / 2 + 2.3, -H / 2), "u1", BLUE,
                (-0.75, -0.8))
    _axis_arrow(ax, (W / 2 + 0.8, -H / 2 + 0.5), (W / 2 + 0.8, -H / 2 + 2.4), "u2",
                ORANGE, (0.75, -0.1))
    ax.plot([W / 2 + 3.5] * 2, [-H / 2, H / 2], color=ORANGE, lw=1.1, ls=(0, (4, 3)))
    ax.text(W / 2 + 3.95, 0, "rotation axis", color=ORANGE, fontsize=9, rotation=90,
            va="center", ha="center")
    ax.set_title("2-D detector  —  3-D scan", fontsize=10.5, color=INK)
    x0, x1 = -W / 2 - 3.0, W / 2 + 5.0
    ax.set_xlim(x0, x1); ax.set_ylim(-aspect * (x1 - x0) / 2, aspect * (x1 - x0) / 2)
    ax.set_aspect("equal"); ax.axis("off")

    save(fig, "schem_detector.svg")


def geometry_schematics_3d():
    """3-D counterparts of the three geometry drawings."""
    B = 1.0
    BLUE, PURPLE = "#1f6feb", "#8250df"

    # ---- parallel ----
    fig = plt.figure(figsize=(3.9, 3.9))
    ax = fig.add_subplot(projection="3d")
    _cube(ax, B)
    L = 5.6 * B
    for y in (-0.62, 0.0, 0.62):
        for z in (-0.62, 0.0, 0.62):
            _ray3(ax, (-0.5 * L, y, z), (0.5 * L, y, z), BLUE, lw=1.0, dot=7)
    _plane(ax, (0.5 * L, 0, 0), (0, 1, 0), (0, 0, 1), 1.0, 1.0, BLUE)
    ax.text(0.5 * L, -1.45, -1.35, "detector", color=BLUE, fontsize=9)
    ax.plot([0, 0], [0, 0], [-1.8 * B, 1.8 * B], color="#57606a", lw=1.0,
            ls=(0, (4, 3)))
    ax.text(0.05, 0.05, 1.9 * B, "rotation axis", color="#57606a", fontsize=8.5)
    _tidy3(ax, 2.2)
    save(fig, "schem_parallel_3d.svg")

    # ---- cone ----
    fig = plt.figure(figsize=(3.9, 3.9))
    ax = fig.add_subplot(projection="3d")
    _cube(ax, B)
    src = np.array([-2.4 * B, 0.0, 0.0])
    xd = 2.4 * B
    for y in (-0.8, 0.0, 0.8):
        for z in (-0.8, 0.0, 0.8):
            d = np.array([xd, y, z]) - src
            _ray3(ax, src, src + d, PURPLE, lw=0.9, dot=0)
    ax.scatter(*src, color=PURPLE, s=42, depthshade=False, zorder=5)
    ax.text(src[0] - 0.95, 0.10, 0.30, "source", color=PURPLE,
            fontsize=9)
    _plane(ax, (xd, 0, 0), (0, 1, 0), (0, 0, 1), 1.0, 1.0, PURPLE)
    ax.text(xd, -1.35, -1.30, "detector", color=PURPLE, fontsize=9)
    ax.plot([0, 0], [0, 0], [-1.75 * B, 1.75 * B], color="#57606a", lw=1.0,
            ls=(0, (4, 3)))
    ax.text(0.05, 0.05, 1.85 * B, "rotation axis", color="#57606a", fontsize=8.5)
    _tidy3(ax, 1.85)
    save(fig, "schem_cone_3d.svg")

    # ---- arbitrary ----
    C0, CM, FAINT = "#1f6feb", "#bf8700", "#8b949e"
    fig = plt.figure(figsize=(5.9, 5.2))
    ax = fig.add_subplot(projection="3d")
    _cube(ax, B)
    for p0, p1 in (((-1.70, 1.45, 0.95), (1.60, -1.15, -0.60)),
                   ((0.30, -1.80, 1.50), (-0.90, 1.70, -1.30))):
        _ray3(ax, p0, p1, FAINT, lw=1.0, dot=9, head=0.22)

    t0, e0 = (-1.90, -1.40, -1.20), (1.50, 1.30, 1.40)
    tM, eM = (-1.30, 0.60, 1.55), (1.20, 0.45, -1.50)
    _ray3(ax, t0, e0, C0, lw=1.7, dot=26)
    _ray3(ax, tM, eM, CM, lw=1.7, dot=26)
    _tidy3(ax, 1.7)
    ax.set_position([0.0, 0.05, 1.0, 0.90])

    # Labels live in the corners; colour ties each one to its ray.
    ax.text2D(0.005, 1.005, r"$\mathbf{t}^{(0)} = (t^{(0)}_x,\, t^{(0)}_y,\, "
              r"t^{(0)}_z)$" "\n"
              r"$\mathbf{n}^{(0)} = (n^{(0)}_x,\, n^{(0)}_y,\, n^{(0)}_z)$",
              transform=ax.transAxes, ha="left", va="top", fontsize=9.5, color=C0)
    ax.text2D(0.995, 1.005,
              r"$\mathbf{t}^{(M-1)} = (t^{(M-1)}_x,\, t^{(M-1)}_y,\, "
              r"t^{(M-1)}_z)$" "\n"
              r"$\mathbf{n}^{(M-1)} = (n^{(M-1)}_x,\, n^{(M-1)}_y,\, "
              r"n^{(M-1)}_z)$",
              transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color=CM)
    ax.text2D(0.5, 0.12,
              r"$\mathbf{t} = (\mathbf{t}^{(0)}, \ldots, \mathbf{t}^{(M-1)}),"
              r"\quad \mathbf{n} = (\mathbf{n}^{(0)}, \ldots, "
              r"\mathbf{n}^{(M-1)}) \ \in \mathbb{R}^{3 \times M}$",
              transform=ax.transAxes, ha="center", va="top", fontsize=10.5,
              color=INK)
    save(fig, "schem_explicit_3d.svg")


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
    _rotation(ax, 2.15 * B, 0.34 * np.pi, 0.78 * np.pi, label_at=(-0.30, 0.34))
    ax.set_xlim(-2.75, 2.55); ax.set_ylim(-1.5, 2.5)
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
    ax.text(src[0], src[1] - 0.42, "source", fontsize=9, color=PURPLE,
            ha="center", va="top")
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
    C0, CM, FAINT = "#1f6feb", "#bf8700", "#8b949e"
    fig, ax = plt.subplots(figsize=(6.0, 4.7))
    _lattice(ax, B)
    for p0, a in (((-1.95, 1.15), -0.60), ((1.90, 1.30), 3.52),
                  ((-0.30, -1.95), 1.10)):
        d = np.array([np.cos(a), np.sin(a)])
        _ray(ax, p0, np.asarray(p0) + 3.4 * B * d, FAINT, lw=1.1, dot=4.0, head=8.0)

    t0, e0 = np.array([-2.30, -1.30]), np.array([1.35, 1.25])
    tM, eM = np.array([-2.10, 1.45]), np.array([1.60, -1.20])
    _ray(ax, t0, e0, C0, lw=1.8, dot=6.5, head=10.0)
    _ray(ax, tM, eM, CM, lw=1.8, dot=6.5, head=10.0)

    def corner(text, xy, frac, ha, va, color):
        ax.annotate(text, xy=tuple(xy), xytext=frac, textcoords="axes fraction",
                    ha=ha, va=va, fontsize=10, color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8,
                                    shrinkA=3, shrinkB=5))

    corner(r"$\mathbf{t}^{(0)} = (t^{(0)}_x,\, t^{(0)}_y)$", t0,
           (0.015, 0.04), "left", "bottom", C0)
    corner(r"$\mathbf{n}^{(0)} = (n^{(0)}_x,\, n^{(0)}_y)$", e0,
           (0.985, 0.96), "right", "top", C0)
    corner(r"$\mathbf{t}^{(M-1)} = (t^{(M-1)}_x,\, t^{(M-1)}_y)$", tM,
           (0.015, 0.96), "left", "top", CM)
    corner(r"$\mathbf{n}^{(M-1)} = (n^{(M-1)}_x,\, n^{(M-1)}_y)$", eM,
           (0.985, 0.04), "right", "bottom", CM)

    ax.text(0.5, -0.03,
            r"$\mathbf{t} = (\mathbf{t}^{(0)}, \ldots, \mathbf{t}^{(M-1)}),\quad "
            r"\mathbf{n} = (\mathbf{n}^{(0)}, \ldots, \mathbf{n}^{(M-1)})"
            r"\ \in \mathbb{R}^{2 \times M}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=10.5, color=INK)

    ax.set_xlim(-3.5, 4.0); ax.set_ylim(-2.95, 2.55)
    ax.set_aspect("equal"); ax.axis("off")
    save(fig, "schem_explicit.svg")


def first_steps():
    """Exactly what the front-page First steps snippet draws."""
    N = 128
    yy, xx = np.mgrid[:N, :N] - (N - 1) / 2
    image = ((xx**2 + yy**2) < (0.38 * N) ** 2).astype(np.float32)
    image[(xx + 18) ** 2 + (yy - 12) ** 2 < 11**2] = 0.3

    k = xtk.UniformSpec.centered(step=1.0, num=(N, N))
    det = xtk.DetectorSpec(size=(1.5 * N,), num_cell=(192,))
    rays = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, 180, endpoint=False), det)

    sino = xtk.xrt_struct_apply(rays, k, 0, Float(image.ravel()))
    fbp = np.asarray(xtk.fbp(rays, k, sino)).reshape(N, N)
    re = xtk.struct_rays(rays)
    cg = np.asarray(xtk.cg(lambda v: xtk.xrt_apply(re, k, 0, v),
                           lambda v: xtk.xrt_adjoint(re, k, 0, v),
                           sino, N * N, n_iter=30)).reshape(N, N)
    print(f"  fbp {abs(fbp - image).mean():.4f}  cg {abs(cg - image).mean():.4f}")

    fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
    ax[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    ax[1].imshow(np.asarray(sino).reshape(180, 192), cmap="gray")
    ax[2].imshow(fbp, cmap="gray", vmin=0, vmax=1)
    ax[3].imshow(cg, cmap="gray", vmin=0, vmax=1)
    for a, t in zip(ax, ("phantom", "sinogram", "fbp", "cg, 30 iterations")):
        a.set_title(t, fontsize=10.5)
        a.set_axis_off()
    plt.tight_layout()
    save(fig, "first_steps.png")


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
    lattice_figure()
    detector_spec_figure()
    first_steps()
    geometry_schematics()
    geometry_schematics_3d()
    bases()
    forward_adjoint()
    gallery_cone()
    gallery_sparse()
    gallery_calibration()
    gallery_walnut()
