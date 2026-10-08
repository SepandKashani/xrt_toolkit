r"""
Interoperability helpers to migrate from other tomography packages.

:py:func:`from_astra` converts an ASTRA ``(proj_geom, vol_geom)`` pair into the rays used by
:py:func:`~xrt_toolkit.xrt_apply` and :py:func:`~xrt_toolkit.xrt_adjoint`.
"""

import math

import numpy as np

from .util import UniformSpec


def _geom_to_vec(proj_geom: dict) -> dict:
    # ASTRA projection geometry in its `*_vec` form.
    if proj_geom["type"].endswith("_vec"):
        return proj_geom
    try:
        import astra
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Converting a non-vectorized ASTRA geometry requires `astra-toolbox`."
        ) from e
    return astra.geom_2vec(proj_geom)


def _vol_frame(vol_geom: dict) -> tuple:
    # (center of the first voxel, voxel size, voxel count) in ASTRA world coordinates,
    # XTK axis order (x, y[, z]).
    opt = vol_geom.get("option", vol_geom.get("options", {}))
    num = [vol_geom["GridColCount"], vol_geom["GridRowCount"]]
    axes = ["X", "Y"]
    if "GridSliceCount" in vol_geom:
        num.append(vol_geom["GridSliceCount"])
        axes.append("Z")

    ll, size = [], []
    for n, a in zip(num, axes):
        lo = opt.get(f"WindowMin{a}", -n / 2)
        hi = opt.get(f"WindowMax{a}", +n / 2)
        size.append((hi - lo) / n)
        ll.append(lo + size[-1] / 2)
    if not all(math.isclose(s, size[0], rel_tol=1e-6) for s in size):
        raise ValueError("from_astra() requires isotropic voxels.")
    return np.array(ll), size[0], tuple(num)


def vol_astra_to_xtk(vol: np.ndarray) -> np.ndarray:
    r"""
    Reorder an ASTRA volume array into XTK's flattened C-ordered layout.

    ASTRA stores 2D volumes as (rows, cols) = (-y, x), the row index increasing towards -y, and 3D
    volumes as (slices, rows, cols) = (z, y, x). XTK stores a flat C-ordered array indexed by
    (x, y[, z]), all axes increasing.

    Parameters
    ----------
    vol: np.ndarray
        (ny, nx) or (nz, ny, nx) ASTRA volume.

    Returns
    -------
    flat: np.ndarray
        (nx * ny [* nz],) XTK coefficients.
    """
    if vol.ndim == 2:
        return np.ascontiguousarray(vol[::-1, :].T).ravel()
    elif vol.ndim == 3:
        return np.ascontiguousarray(vol.transpose(2, 1, 0)).ravel()
    raise ValueError("Expected a 2D or 3D ASTRA volume.")


def vol_xtk_to_astra(flat: np.ndarray, num: tuple[int]) -> np.ndarray:
    r"""
    Inverse of :py:func:`vol_astra_to_xtk`.

    Parameters
    ----------
    flat: np.ndarray
        (nx * ny [* nz],) XTK coefficients.
    num: tuple[int]
        (nx, ny[, nz]) lattice size, i.e. ``knot_spec.num``.

    Returns
    -------
    vol: np.ndarray
        (ny, nx) or (nz, ny, nx) ASTRA volume.
    """
    vol = np.asarray(flat).reshape(num)
    if len(num) == 2:
        return np.ascontiguousarray(vol.T[::-1, :])
    elif len(num) == 3:
        return np.ascontiguousarray(vol.transpose(2, 1, 0))
    raise ValueError("Expected a 2D or 3D volume.")


def from_astra(proj_geom: dict, vol_geom: dict, Array=None) -> tuple:
    r"""
    Convert an ASTRA geometry pair into XTK rays and volume lattice.

    The returned ``(ray_spec, knot_spec)`` can be passed directly to
    :py:func:`~xrt_toolkit.xrt_apply` and :py:func:`~xrt_toolkit.xrt_adjoint`. The rays are ordered
    as the flattened ASTRA sinogram: (angles, det) in 2D, (det_v, angles, det_u) in 3D. Convert
    volume arrays with :py:func:`vol_astra_to_xtk` and :py:func:`vol_xtk_to_astra`.

    Supported geometries: ``parallel``, ``fanflat``, ``parallel3d``, ``cone`` and their ``*_vec``
    forms. Only the non-``*_vec`` forms need ``astra-toolbox`` to be installed.

    Parameters
    ----------
    proj_geom: dict
        ASTRA projection geometry, e.g. from ``astra.create_proj_geom()``.
    vol_geom: dict
        ASTRA volume geometry, e.g. from ``astra.create_vol_geom()``. Voxels must be isotropic.
    Array: ArrayNfT type
        Ray coordinate type. (Default: ``drjit.cuda.Array2f`` or ``drjit.cuda.Array3f``.)

    Returns
    -------
    ray_spec: tuple[ArrayNfT, ArrayNfT]
        (L,) ray anchors :math:`\bbt` and directions :math:`\bbn`.
    knot_spec: UniformSpec
        Volume lattice.

    Notes
    -----
    Lengths are in units of the voxel size :math:`s`: XTK's projections are ASTRA's divided by
    :math:`s`.
    """
    vec = _geom_to_vec(proj_geom)
    gtype, V = vec["type"], vec["Vectors"]
    ll, s, num = _vol_frame(vol_geom)

    if gtype in ("parallel_vec", "fanflat_vec"):
        n_det = vec["DetectorCount"]
        i = np.arange(n_det) - n_det / 2 + 0.5  # detector-cell centers
        a, d, u = V[:, 0:2], V[:, 2:4], V[:, 4:6]  # a: ray direction or source
        pix = d[:, None, :] + i[None, :, None] * u[:, None, :]  # (angles, det, 2)
        a = a[:, None, :]
    elif gtype in ("parallel3d_vec", "cone_vec"):
        n_u, n_v = vec["DetectorColCount"], vec["DetectorRowCount"]
        iu = np.arange(n_u) - n_u / 2 + 0.5
        iv = np.arange(n_v) - n_v / 2 + 0.5
        a, d, u, v = V[:, 0:3], V[:, 3:6], V[:, 6:9], V[:, 9:12]
        pix = (  # (det_v, angles, det_u, 3)
            d[None, :, None, :]
            + iv[:, None, None, None] * v[None, :, None, :]
            + iu[None, None, :, None] * u[None, :, None, :]
        )
        a = a[None, :, None, :]
    else:
        raise ValueError(f"Unsupported ASTRA geometry '{gtype}'.")

    if gtype.startswith("parallel"):  # anchors on the detector, one direction per view
        t, n = pix, np.broadcast_to(a, pix.shape)
    else:  # anchors at the source
        t, n = np.broadcast_to(a, pix.shape), pix - a

    # ASTRA world coordinates -> voxel units
    D = pix.shape[-1]
    t = t.reshape(-1, D) / s
    n = n.reshape(-1, D)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    knot_spec = UniformSpec(start=tuple(ll / s), step=1, num=num)

    if Array is None:
        import drjit.cuda

        Array = drjit.cuda.Array2f if (D == 2) else drjit.cuda.Array3f
    ray_t = Array(np.ascontiguousarray(t.T, dtype=np.float32))
    ray_n = Array(np.ascontiguousarray(n.T, dtype=np.float32))
    return (ray_t, ray_n), knot_spec
