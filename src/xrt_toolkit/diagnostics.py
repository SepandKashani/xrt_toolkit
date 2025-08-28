import importlib
import math
import typing as typ

import drjit as dr
import numpy as np

from .drjit.bbox import ray_bbox_intersect
from .drjit.box_spline import box_spline_1d_E, box_spline_1d_np
from .util import UniformSpec

ArrayNNfT = typ.TypeVar("ArrayNNfT", bound=dr.AnyArray)
ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNfT, ArrayNfT]
StructRaySpecT = tuple[ArrayNNfT, ArrayNNfT, UniformSpec]


def plot_rays(
    ray_spec: RaySpecT | StructRaySpecT,
    knot_spec: UniformSpec,
    show_grid: bool = False,
):
    r"""
    Plot ray trajectories inside a volume.

    Parameters
    ----------
    ray_spec: tuple[ArrayNfT, ArrayNfT] | tuple[ArrayNNfT, ArrayNNfT, UniformSpec]
        (L,) ray anchors :math:`\bbt \in \bR^{D}` and directions :math:`\bbn \in \bR^{D}`.

        There are two methods to specify ray parameters (\bbt, \bbn):

        * Explicit, as in ``xrt_apply()``,
        * Implicit, as in ``xrt_struct_apply()``.

        Refer to their docstrings for details.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    show_grid: bool
        If true, overlay the neighborhood grid.

    Returns
    -------
    fig: :py:class:`~matplotlib.figure.Figure`
        Diagnostic plot.

    Notes
    -----
    Rays which do not intersect the volume are **not** shown.
    """
    # type checking ---------------------------------------
    if len(ray_spec) == 2:
        ray_t, ray_n = ray_spec

        ArrayNf = type(ray_t)
        ArrayNu = dr.uint32_array_t(ArrayNf)
        Float = dr.value_t(ArrayNf)

        D = dr.size_v(ArrayNf)
        assert (ray_t.ndim == 2) and (D in (2, 3))
        assert type(ray_n) is ArrayNf
    elif len(ray_spec) == 3:
        ray_t_spec, ray_n_spec, ray_u_spec = ray_spec

        ArrayNNf = type(ray_t_spec)
        ArrayNf = dr.value_t(ArrayNNf)
        ArrayNu = dr.uint32_array_t(ArrayNf)
        Float = dr.value_t(ArrayNf)
        UInt = dr.value_t(dr.uint32_array_t(ArrayNf))

        D = dr.size_v(ArrayNf)
        assert (ray_t_spec.ndim == 3) and (D in (2, 3))
        assert type(ray_n_spec) is ArrayNNf
        assert ray_u_spec.ndim == D

        assert (N_proj := ray_t_spec.shape[-1]) == ray_n_spec.shape[-1]

        # implicit-ray -> explicit-ray conversion
        L_proj = math.prod(ray_u_spec.num)
        L = N_proj * L_proj
        u = [None] * D
        for d, (start, step, num) in enumerate(ray_u_spec):
            u[d] = start + step * dr.arange(Float, num)
        uu = ArrayNf(*dr.meshgrid(*u, indexing="ij"))

        ray_t = dr.zeros(ArrayNf, L)
        ray_n = dr.zeros(ArrayNf, L)
        index = dr.arange(UInt, 0, L_proj)
        for i in range(N_proj):
            # for-loop not ideal for tracing time, but doesn't matter for plotting
            H_t = dr.gather(ArrayNNf, ray_t_spec, i)
            dr.scatter(ray_t, H_t @ uu, index)

            H_n = dr.gather(ArrayNNf, ray_n_spec, i)
            dr.scatter(ray_n, H_n @ uu, index)

            index += L_proj
    else:
        raise ValueError("Unknown `ray_spec`")
    assert knot_spec.ndim == D
    # -----------------------------------------------------

    # setup figure ----------------------------------------
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        collections = importlib.import_module("matplotlib.collections")
        patches = importlib.import_module("matplotlib.patches")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("`matplotlib` missing: `pip install matplotlib`")

    if D == 2:
        fig, ax = plt.subplots()
        data = [(ax, [0, 1], ["x", "y"])]
    elif D == 3:
        fig, ax = plt.subplots(ncols=3)
        data = [
            (ax[0], [0, 1], ["x", "y"]),
            (ax[1], [0, 2], ["x", "z"]),
            (ax[2], [1, 2], ["y", "z"]),
        ]
    # -----------------------------------------------------

    # restrict rays to those that intersect with BBox -----
    knot_start = ArrayNf(*knot_spec.start)
    knot_step = ArrayNf(*knot_spec.step)
    knot_num = ArrayNu(*knot_spec.num)
    bbox_ll = knot_start - (knot_step / 2)
    bbox_ur = knot_start - (knot_step / 2) + (knot_num * knot_step)

    active, a1, a2 = ray_bbox_intersect(bbox_ll, bbox_ur, ray_t, ray_n)
    idx = dr.compress(active)
    ray_t = dr.gather(ArrayNf, ray_t, idx)
    ray_n = dr.gather(ArrayNf, ray_n, idx)
    a1 = dr.gather(Float, a1, idx).numpy()  # (N_ray,)
    a2 = dr.gather(Float, a2, idx).numpy()  # (N_ray,)
    # -----------------------------------------------------

    for _ax, dim_idx, dim_label in data:
        # subsample right dimensions ----------------------
        select = lambda _: _.numpy().flatten()[dim_idx]  # ArrayN -> (2,)
        _bbox_ll = select(bbox_ll)  # (2,)
        _knot_start = select(knot_start)  # (2,)
        _knot_step = select(knot_step)  # (2,)
        _knot_num = select(knot_num)  # (2,)
        _ray_t = ray_t.numpy()[dim_idx]  # (2, N_ray)
        _ray_n = ray_n.numpy()[dim_idx]  # (2, N_ray)

        # helper variables --------------------------------
        bbox_dim = _knot_num * _knot_step  # (2,)

        # draw bbox ---------------------------------------
        rect = patches.Rectangle(
            xy=_bbox_ll,
            width=bbox_dim[0],
            height=bbox_dim[1],
            facecolor="none",
            edgecolor="k",
            label="volume BBox",
        )
        _ax.add_patch(rect)

        # draw knot_ll ------------------------------------
        _ax.scatter(
            _knot_start[0],
            _knot_start[1],
            color="k",
            label="knot_start",
            marker="+",
        )

        # draw rays/anchors -------------------------------
        # Each (2,2) sub-array in `coords` represents line start/end coordinates.
        coords_x = _ray_t + a1 * _ray_n  # (2, N_ray)
        coords_y = _ray_t + a2 * _ray_n  # (2, N_ray)
        coords = [(x, y) for (x, y) in zip(coords_x.T, coords_y.T)]
        lines = collections.LineCollection(
            coords,
            label=r"$t + \alpha n$",
            color="k",
            alpha=0.5,
            linewidth=1,
        )
        _ax.add_collection(lines)
        _ax.scatter(
            _ray_t[0],
            _ray_t[1],
            label=r"t",
            color="g",
            marker=".",
        )

        # misc details ------------------------------------
        x_ticks = _bbox_ll[0] + _knot_step[0] * np.arange(_knot_num[0] + 1)
        y_ticks = _bbox_ll[1] + _knot_step[1] * np.arange(_knot_num[1] + 1)
        x_labels = (
            [f"{_bbox_ll[0]:0.3f}"]
            + [""] * (_knot_num[0] - 1)
            + [f"{_bbox_ll[0] + bbox_dim[0]:0.3f}"]
        )
        y_labels = (
            [f"{_bbox_ll[1]:0.3f}"]
            + [""] * (_knot_num[1] - 1)
            + [f"{_bbox_ll[1] + bbox_dim[1]:0.3f}"]
        )
        _ax.set_xticks(x_ticks, x_labels)
        _ax.set_yticks(y_ticks, y_labels)
        _ax.set_xlabel(dim_label[0])
        _ax.set_ylabel(dim_label[1])

        pad_width = 0.1 * bbox_dim  # 10% axial pad
        _ax.set_xlim(
            _bbox_ll[0] - pad_width[0],
            _bbox_ll[0] + bbox_dim[0] + pad_width[0],
        )
        _ax.set_ylim(
            _bbox_ll[1] - pad_width[1],
            _bbox_ll[1] + bbox_dim[1] + pad_width[1],
        )
        _ax.legend(loc="lower right", bbox_to_anchor=(1, 1))
        _ax.set_aspect(1)

        # draw overlay grid -------------------------------
        if show_grid:
            _ax.grid(
                linestyle="--",
                color="gray",
            )

    fig.tight_layout()
    return fig


def plot_2d_basis(
    knot_spec: UniformSpec,
    order: int,
    ray_n: ArrayNfT,
):
    r"""
    Visualize basis function :math:`\psi: \bR^{2} \to \bR`.

    Shows side-by-side:
    - a portion of the neighborhood grid with the support of :math:`\psi` overlayed.
    - 1d projections of :math:`\psi` at projection directions `ray_n`.

    Parameters
    ----------
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Data interpolation order.

        This parameter sets which :math:`\psi` is used to interpolate data values.
    ray_n: ArrayNfT
        (L,) projection directions :math:`\bbn \in \bR^{2}`.

    Returns
    -------
    fig: :py:class:`~matplotlib.figure.Figure`
        Diagnostic plot.
    """
    # type checking ---------------------------------------
    D = knot_spec.ndim
    assert D == 2

    assert order in (0, 1, 2)

    ArrayNf = type(ray_n)
    assert (ray_n.ndim == 2) and (dr.size_v(ArrayNf) == D)
    # -----------------------------------------------------

    # setup figure ----------------------------------------
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        patches = importlib.import_module("matplotlib.patches")
        sps = importlib.import_module("scipy.spatial")
    except ModuleNotFoundError:
        raise ModuleNotFoundError("`matplotlib` missing: `pip install matplotlib`")

    fig, ax = plt.subplots(ncols=2)
    # -----------------------------------------------------

    # helper variables ------------------------------------
    knot_start = np.array(knot_spec.start)
    knot_step = np.array(knot_spec.step)
    ray_n = dr.normalize(ray_n).numpy().T  # (L, 2)

    # draw knot_ll ----------------------------------------
    ax[0].scatter(
        knot_start[0],
        knot_start[1],
        color="k",
        label="knot_start",
        marker="+",
    )

    # draw neighborhood bbox ------------------------------
    if order == 0:
        N_neighbor = 1
    elif order in (1, 2):
        N_neighbor = 2
    bbox_ll = knot_start - (knot_step / 2) - N_neighbor * knot_step
    bbox_ur = knot_start + (knot_step / 2) + N_neighbor * knot_step
    bbox_dim = bbox_ur - bbox_ll
    rect = patches.Rectangle(
        xy=bbox_ll,
        width=bbox_dim[0],
        height=bbox_dim[1],
        facecolor="none",
        edgecolor="k",
    )
    ax[0].add_patch(rect)

    # draw grid -------------------------------------------
    x_ticks = bbox_ll[0] + knot_step[0] * np.arange(N_neighbor + 3)
    y_ticks = bbox_ll[1] + knot_step[1] * np.arange(N_neighbor + 3)
    x_labels = ("",) * len(x_ticks)
    y_labels = ("",) * len(y_ticks)
    ax[0].set_xticks(x_ticks, x_labels)
    ax[0].set_yticks(y_ticks, y_labels)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    pad_width = 0.1 * bbox_dim  # 10% axial pad
    ax[0].set_xlim(
        bbox_ll[0] - pad_width[0],
        bbox_ll[0] + bbox_dim[0] + pad_width[0],
    )
    ax[0].set_ylim(
        bbox_ll[1] - pad_width[1],
        bbox_ll[1] + bbox_dim[1] + pad_width[1],
    )
    ax[0].set_aspect(1)
    ax[0].grid(
        linestyle="--",
        color="gray",
    )

    # draw psi support ------------------------------------
    E_2D = box_spline_1d_E(  # (2, order+2) 2D box-spline directions
        order,
        scale=ArrayNf(knot_spec.step),
    ).numpy()

    mesh = np.stack(  # (order+2, Nx, Ny)
        # we do [-0.5, 0.5] instead of [0, 1] to be symmetric around central point
        np.meshgrid(*(np.linspace(-0.5, 0.5, 5),) * (order + 2)),
        axis=0,
    )
    vertices = np.tensordot(mesh, E_2D, axes=[[0], [1]])  # (Nx, Ny, 2)
    vertices = knot_start + vertices.reshape(-1, 2)  # (Nx*Ny, 2)

    hull = sps.ConvexHull(vertices)
    ax[0].fill(
        vertices[hull.vertices, 0],
        vertices[hull.vertices, 1],
        alpha=0.4,
        color="r",
        label=r"$\psi$ support",
    )

    # draw projections ------------------------------------
    Un_perp = ray_n[:, ::-1] * np.r_[-1, 1]  # (L, 2)
    E_1D = abs(Un_perp @ E_2D)  # (L, order+2) 1D box-spline directions

    proj_dim = np.linalg.norm((2 * N_neighbor + 1) * knot_step)
    N_offset = 1_001
    x = np.linspace(-proj_dim / 2, proj_dim / 2, N_offset)
    for _e, _n in zip(E_1D, ray_n):
        y = box_spline_1d_np(_e, x)  # (N_offset,)
        ax[1].plot(x, y, label=r"$\hat{\mathbf{n}} = $" + f"{np.around(_n, 1)}")

    # misc ------------------------------------------------
    ax[0].legend(loc="lower left", bbox_to_anchor=(0, 1))
    ax[1].legend(loc="lower left", bbox_to_anchor=(0, 1))

    fig.tight_layout()
    return fig
