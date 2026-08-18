r"""
Interoperability helpers to migrate from other tomography packages.

The main entry point is :py:func:`from_astra`, which converts an ASTRA
`(proj_geom, vol_geom)` pair into the explicit ray parameterization used by
:py:func:`~xrt_toolkit.drjit.xrt_apply` and
:py:func:`~xrt_toolkit.drjit.xrt_adjoint`.
"""

import math

import numpy as np

from .util import UniformSpec


def _geom_to_vec(proj_geom: dict) -> dict:
    """Normalize an ASTRA projection geometry to its `*_vec` form."""
    gtype = proj_geom["type"]
    if gtype.endswith("_vec"):
        return proj_geom
    try:
        import astra
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Converting a non-vectorized ASTRA geometry requires the "
            "`astra-toolbox` package: `pip install astra-toolbox`."
        )
    return astra.geom_2vec(proj_geom)


def _vol_frame(vol_geom: dict):
    """
    Extract (ll_center, pixel_size, num) of the volume, in ASTRA world
    coordinates and XTK axis order (x, y[, z]).
    """
    opt = vol_geom.get("option", vol_geom.get("options", {}))
    nx = vol_geom["GridColCount"]
    ny = vol_geom["GridRowCount"]
    is_3d = "GridSliceCount" in vol_geom

    x_min = opt.get("WindowMinX", -nx / 2)
    x_max = opt.get("WindowMaxX", +nx / 2)
    y_min = opt.get("WindowMinY", -ny / 2)
    y_max = opt.get("WindowMaxY", +ny / 2)
    sx = (x_max - x_min) / nx
    sy = (y_max - y_min) / ny

    if is_3d:
        nz = vol_geom["GridSliceCount"]
        z_min = opt.get("WindowMinZ", -nz / 2)
        z_max = opt.get("WindowMaxZ", +nz / 2)
        sz = (z_max - z_min) / nz
        size = (sx, sy, sz)
        ll = (x_min + sx / 2, y_min + sy / 2, z_min + sz / 2)
        num = (nx, ny, nz)
    else:
        size = (sx, sy)
        ll = (x_min + sx / 2, y_min + sy / 2)
        num = (nx, ny)

    if not all(math.isclose(s, size[0], rel_tol=1e-6) for s in size):
        raise ValueError("`from_astra` requires isotropic voxels.")
    return np.asarray(ll), size[0], num


def vol_astra_to_xtk(vol: np.ndarray) -> np.ndarray:
    """
    Reorder an ASTRA volume array into XTK's flattened C-order layout.

    ASTRA stores 2D volumes as (rows, cols) = (-y, x), the row index
    increasing towards -y, and 3D volumes as (slices, rows, cols) = (z, y, x)
    with all axes increasing. XTK stores the coefficients as a flat C-ordered
    array indexed by (x, y[, z]) with all axes increasing.

    Parameters
    ----------
    vol: np.ndarray
        (ny, nx) or (nz, ny, nx) ASTRA volume.

    Returns
    -------
    flat: np.ndarray
        ``(nx*ny[*nz],)`` flattened XTK coefficient array.
    """
    if vol.ndim == 2:
        return np.ascontiguousarray(vol[::-1, :].T).ravel()
    elif vol.ndim == 3:
        return np.ascontiguousarray(vol.transpose(2, 1, 0)).ravel()
    raise ValueError("Expected a 2D or 3D ASTRA volume.")


def vol_xtk_to_astra(flat: np.ndarray, num: tuple[int]) -> np.ndarray:
    """Inverse of :py:func:`vol_astra_to_xtk`. `num` is `knot_spec.num`."""
    vol = np.asarray(flat).reshape(num)
    if len(num) == 2:
        return np.ascontiguousarray(vol.T[::-1, :])
    elif len(num) == 3:
        return np.ascontiguousarray(vol.transpose(2, 1, 0))
    raise ValueError("Expected a 2D or 3D volume.")


def from_astra(proj_geom: dict, vol_geom: dict, Array=None):
    r"""
    Convert an ASTRA geometry pair into an XTK ray specification.

    This helper eases the migration of ASTRA-based code: the returned pair
    ``(ray_spec, knot_spec)`` can be passed directly to
    :py:func:`~xrt_toolkit.drjit.xrt_apply` and
    :py:func:`~xrt_toolkit.drjit.xrt_adjoint`. Volume arrays are converted
    with :py:func:`vol_astra_to_xtk` / :py:func:`vol_xtk_to_astra`; the
    projections are ordered exactly as the corresponding flattened ASTRA
    sinogram, that is, ``(angles, det)`` in 2D and ``(det_v, angles, det_u)``
    in 3D.

    Supported geometry types: ``parallel``, ``fanflat``, ``parallel3d``,
    ``cone``, and their ``*_vec`` variants.

    Parameters
    ----------
    proj_geom: dict
        ASTRA projection geometry, e.g. from ``astra.create_proj_geom()``.
    vol_geom: dict
        ASTRA volume geometry from ``astra.create_vol_geom()``.
        Voxels must be isotropic.
    Array: drjit array type
        Ray coordinate type, e.g. ``drjit.cuda.ad.Array2f``.
        Defaults to ``drjit.cuda.ad.Array{2,3}f`` based on the geometry.

    Returns
    -------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) explicit ray anchors and directions.
    knot_spec: UniformSpec
        Matching volume lattice.

    Notes
    -----
    Lengths are expressed in units of the voxel size: with a voxel size
    :math:`s \neq 1`, projections computed by XTK equal the ASTRA ones
    divided by :math:`s`.
    """
    vec = _geom_to_vec(proj_geom)
    gtype = vec["type"]
    V = vec["Vectors"]
    ll, s, num = _vol_frame(vol_geom)

    if gtype in ("parallel_vec", "fanflat_vec"):
        n_det = vec["DetectorCount"]
        i = np.arange(n_det) - n_det / 2 + 0.5  # detector-pixel centers
        if gtype == "parallel_vec":
            ray, d, u = V[:, 0:2], V[:, 2:4], V[:, 4:6]
            # rays: anchor on the detector, direction shared per view
            t = d[:, None, :] + i[None, :, None] * u[:, None, :]
            n = np.broadcast_to(ray[:, None, :], t.shape)
        else:  # fanflat_vec
            src, d, u = V[:, 0:2], V[:, 2:4], V[:, 4:6]
            pix = d[:, None, :] + i[None, :, None] * u[:, None, :]
            t = np.broadcast_to(src[:, None, :], pix.shape)
            n = pix - t
        t = t.reshape(-1, 2)
        n = n.reshape(-1, 2)
    elif gtype in ("parallel3d_vec", "cone_vec"):
        n_u = vec["DetectorColCount"]
        n_v = vec["DetectorRowCount"]
        iu = np.arange(n_u) - n_u / 2 + 0.5
        iv = np.arange(n_v) - n_v / 2 + 0.5
        # ASTRA 3D sinograms are laid out as (det_v, angles, det_u)
        if gtype == "parallel3d_vec":
            ray, d, u, v = V[:, 0:3], V[:, 3:6], V[:, 6:9], V[:, 9:12]
            pix = (
                d[None, :, None, :]
                + iv[:, None, None, None] * v[None, :, None, :]
                + iu[None, None, :, None] * u[None, :, None, :]
            )  # (n_v, n_angles, n_u, 3)
            t = pix
            n = np.broadcast_to(ray[None, :, None, :], pix.shape)
        else:  # cone_vec
            src, d, u, v = V[:, 0:3], V[:, 3:6], V[:, 6:9], V[:, 9:12]
            pix = (
                d[None, :, None, :]
                + iv[:, None, None, None] * v[None, :, None, :]
                + iu[None, None, :, None] * u[None, :, None, :]
            )
            t = np.broadcast_to(src[None, :, None, :], pix.shape)
            n = pix - t
        t = t.reshape(-1, 3)
        n = n.reshape(-1, 3)
    else:
        raise ValueError(f"Unsupported ASTRA geometry '{gtype}'.")

    # ASTRA world coordinates -> XTK lattice coordinates (voxel units).
    D = t.shape[1]
    t = t / s
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    knot_spec = UniformSpec(start=tuple(ll / s), step=1, num=num)

    if Array is None:
        import drjit.cuda.ad as drc

        Array = drc.Array2f if (D == 2) else drc.Array3f
    Float = __import__("drjit").value_t(Array)
    ray_t = Array(*[Float(np.ascontiguousarray(t[:, d], np.float32)) for d in range(D)])
    ray_n = Array(*[Float(np.ascontiguousarray(n[:, d], np.float32)) for d in range(D)])
    return (ray_t, ray_n), knot_spec
