import collections.abc as cabc
import typing as typ
from dataclasses import dataclass

import array_api_compat
import drjit as dr
import numpy.typing as npt

from .compat import asarray
from .mesh import UniformSpec
from .misc import broadcast_seq

ArrayNNfT = typ.TypeVar("ArrayNNfT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNNfT, ArrayNNfT, UniformSpec]


@dataclass
class DetectorSpec:
    r"""
    Physical dimensions of a 1D/2D pixelized detector.
    """

    size: tuple[float]
    num_cell: tuple[int]

    def __init__(self, size, num_cell):
        r"""
        Parameters
        ----------
        size: tuple[float]
            Detector span (unitless) \in \bR_{+}^{D}
        num_cell: tuple[int]
            (M1,...,MD) cell count per dimension

        Scalars are broadcast to all dimensions.
        """
        size = broadcast_seq(size, None, float)
        assert all(s > 0 for s in size)

        num_cell = broadcast_seq(num_cell, None, int)
        assert all(n > 0 for n in num_cell)

        D = max(map(len, [size, num_cell]))
        assert D in (1, 2)

        self.size = broadcast_seq(size, D)
        self.num_cell = broadcast_seq(num_cell, D)

    @property
    def cell_size(self) -> tuple[float]:
        c_size = tuple(self.size[d] / self.num_cell[d] for d in range(self.ndim))
        return c_size

    @property
    def ndim(self) -> int:
        D = len(self.size)
        return D

    def __iter__(self) -> cabc.Iterator:
        for d in range(self.ndim):
            yield (self.size[d], self.num_cell[d])


def parallel_beam(
    angles: npt.NDArray,
    detector_spec: DetectorSpec,
) -> RaySpecT:
    r"""
    Create parallel/cylinder-beam structured ray specification.

    This is a helper function to create pre-defined projection geometries for :py:func:`~xrt_toolkit.drjit.xrt_struct_apply` or :py:`~xrt_toolkit.drjit.xrt_struct_adjoint`.

    Parameters
    ----------
    angles: NDArray
        (N_proj,) angles[rad] in the XY plane.

        TODO: needs more explaining.
    detector_spec: DetectorSpec
        1D (width) or 2D (width, height) detector.

    Returns
    -------
    ray_spec: RaySpecT
        Implicit ray descriptor encoding `N_proj` parallel/cylinder-beam projections.

        `ray_spec` encodes ``L = N_proj * prod(detector_spec.num_cell)`` projections.
    """
    xp = array_api_compat.array_namespace(angles)
    fdtype = angles.dtype
    assert angles.ndim == 1
    assert xp.isdtype(fdtype, kind="real floating")

    # For simplicity, we build (t_spec, n_spec) for the 3D case, then reduce the matrix size if in 2D.
    rot_z = lambda a: xp.array(
        [
            [xp.cos(a), -xp.sin(a), 0],
            [xp.sin(a), xp.cos(a), 0],
            [0, 0, 1],
        ],
        dtype=fdtype,
    )
    R = xp.stack([rot_z(a) for a in angles])  # (N_proj, 3, 3)

    # \bbH_{t}(\alpha) = R(a) [[0 0 | 0]
    #                          [1 0 | 0]
    #                          [0 1 | 0]]
    ray_t_spec = R @ xp.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=fdtype,
    )

    # \bbH_{n}(\alpha) = R(a) [[0 0 | 1]
    #                          [0 0 | 0]
    #                          [0 0 | 0]]
    ray_n_spec = R @ xp.array(
        [
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=fdtype,
    )

    ll_center = tuple(
        (cs - s) / 2
        for (cs, s) in zip(
            detector_spec.cell_size,
            detector_spec.size,
        )
    )
    ray_u_spec = UniformSpec(
        start=(*ll_center, 1),
        step=(*detector_spec.cell_size, 1e-16),
        num=(*detector_spec.num_cell, 1),
    )

    D = detector_spec.ndim + 1
    if D == 2:
        # \bbH_{t}(\alpha) = [[-sin(a) | 0]
        #                     [ cos(a) | 0]]
        ray_t_spec = xp.take(ray_t_spec, [0, 1], axis=1)  # (N_proj, 2, 3)
        ray_t_spec = xp.take(ray_t_spec, [0, 2], axis=2)  # (N_proj, 2, 2)

        # \bbH_{n}(\alpha) = [[0 | cos(a)]
        #                     [0 | sin(a)]]
        ray_n_spec = xp.take(ray_n_spec, [0, 1], axis=1)  # (N_proj, 2, 3)
        ray_n_spec = xp.take(ray_n_spec, [0, 2], axis=2)  # (N_proj, 2, 2)

    ray_spec = (asarray(ray_t_spec), asarray(ray_n_spec), ray_u_spec)
    return ray_spec


def cone_beam(
    sod: float,
    sdd: float,
    angles: npt.NDArray,
    detector_spec: DetectorSpec,
) -> RaySpecT:
    r"""
    Create fan/cone-beam structured ray specification.

    This is a helper function to create pre-defined projection geometries for :py:func:`~xrt_toolkit.drjit.xrt_struct_apply` or :py:`~xrt_toolkit.drjit.xrt_struct_adjoint`.

    Parameters
    ----------
    sod: float
        Source-Object-Distance (XY plane).
    sdd: float
        Source-Detector-Distance (XY plane).
    angles: NDArray
        (N_proj,) angles[rad] in the XY plane.

        TODO: needs more explaining.
    detector_spec: DetectorSpec
        1D (width) or 2D (width, height) detector.

    Returns
    -------
    ray_spec: RaySpecT
        Implicit ray descriptor encoding `N_proj` fan/cone-beam projections.

        `ray_spec` encodes ``L = N_proj * prod(detector_spec.num_cell)`` projections.
    """
    xp = array_api_compat.array_namespace(angles)
    fdtype = angles.dtype
    assert 0 < sod < sdd
    assert angles.ndim == 1
    assert xp.isdtype(fdtype, kind="real floating")

    # For simplicity, we build (t_spec, n_spec) for the 3D case, then reduce the matrix size if in 2D.
    rot_z = lambda a: xp.array(
        [
            [xp.cos(a), -xp.sin(a), 0],
            [xp.sin(a), xp.cos(a), 0],
            [0, 0, 1],
        ],
        dtype=fdtype,
    )
    R = xp.stack([rot_z(a) for a in angles])  # (N_proj, 3, 3)

    # \bbH_{t}(\alpha) = R(a) [[0 0 | -sod]
    #                          [0 0 |    0]
    #                          [0 0 |    0]]
    ray_t_spec = R @ xp.array(
        [
            [0, 0, -sod],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=fdtype,
    )

    # \bbH_{n}(\alpha) = R(a) [[0 0 | sdd]
    #                          [1 0 |   0]
    #                          [0 1 |   0]]
    ray_n_spec = R @ xp.array(
        [
            [0, 0, sdd],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=fdtype,
    )

    ll_center = tuple(
        (cs - s) / 2
        for (cs, s) in zip(
            detector_spec.cell_size,
            detector_spec.size,
        )
    )
    ray_u_spec = UniformSpec(
        start=(*ll_center, 1),
        step=(*detector_spec.cell_size, 1e-16),
        num=(*detector_spec.num_cell, 1),
    )

    D = detector_spec.ndim + 1
    if D == 2:
        # \bbH_{t}(\alpha) = R(a) [[0 | -sod]
        #                          [0 |    0]]
        ray_t_spec = xp.take(ray_t_spec, [0, 1], axis=1)  # (N_proj, 2, 3)
        ray_t_spec = xp.take(ray_t_spec, [0, 2], axis=2)  # (N_proj, 2, 2)

        # \bbH_{n}(\alpha) = R(a) [[0 | sdd]
        #                          [1 |   0]]
        ray_n_spec = xp.take(ray_n_spec, [0, 1], axis=1)  # (N_proj, 2, 3)
        ray_n_spec = xp.take(ray_n_spec, [0, 2], axis=2)  # (N_proj, 2, 2)

    ray_spec = (asarray(ray_t_spec), asarray(ray_n_spec), ray_u_spec)
    return ray_spec
