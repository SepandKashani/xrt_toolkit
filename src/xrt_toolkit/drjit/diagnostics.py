import importlib
import math
import typing as typ

import drjit as dr
import numpy as np

import xrt_toolkit.util as xrtu

from .bbox import ray_bbox_intersect

ArrayNNfT = typ.TypeVar("ArrayNNfT", bound=dr.AnyArray)
ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNfT, ArrayNfT]
StructRaySpecT = tuple[ArrayNNfT, ArrayNNfT, xrtu.UniformSpec]


def diagnostic_plot(
    ray_spec: RaySpecT | StructRaySpecT,
    knot_spec: xrtu.UniformSpec,
    show_grid: bool = False,
):
    r"""
    Plot ray trajectories.

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
        If true, overlay the pixel grid.

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
            _bbox_ll[0] - pad_width[0], _bbox_ll[0] + bbox_dim[0] + pad_width[0]
        )
        _ax.set_ylim(
            _bbox_ll[1] - pad_width[1], _bbox_ll[1] + bbox_dim[1] + pad_width[1]
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
