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


def geometries():
    """The three ways to specify rays."""
    det = xtk.DetectorSpec(size=(1.5 * N,), num_cell=(12,))
    par = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, 4, endpoint=False), det)
    cone = xtk.cone_beam(sod=1.6 * N, sdd=2.8 * N,
                         angles=dr.linspace(Float, 0, 2 * np.pi, 4, endpoint=False),
                         detector_spec=xtk.DetectorSpec(size=(2.2 * N,), num_cell=(12,)))
    t = Array2f([0.0, -20.0, 30.0], [0.0, 10.0, -50.0])
    n = Array2f([1.0, 0.6, 0.0], [0.0, 0.8, 1.0])
    for spec, name, title in ((par, "geom_parallel.png", "parallel_beam"),
                              (cone, "geom_cone.png", "cone_beam"),
                              ((t, n), "geom_explicit.png", "explicit (t, n)")):
        fig = xtk.plot_rays(spec, knot)
        fig.suptitle(title)
        save(fig, name)


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


if __name__ == "__main__":
    geometries()
    bases()
    forward_adjoint()
