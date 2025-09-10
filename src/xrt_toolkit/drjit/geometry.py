import drjit as dr

import xrt_toolkit.util as xrtu

from .struct_xrt import FloatT, RaySpecT


def parallel_beam(
    angles: FloatT,
    detector_spec: xrtu.DetectorSpec,
) -> RaySpecT:
    r"""
    Create parallel/cylinder-beam structured ray specification.

    This is a helper function to create pre-defined projection geometries for :py:func:`~xrt_toolkit.drjit.xrt_struct_apply` or :py:`~xrt_toolkit.drjit.xrt_struct_adjoint`.

    Parameters
    ----------
    angles: FloatT
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
    Float = type(angles)
    Matrix2 = xrtu.float_matrix_t(Float, 2)
    Matrix3 = xrtu.float_matrix_t(Float, 3)

    # For simplicity, we build (t_spec, n_spec) for the 3D case, then reduce matrix sizes if in 2D.
    # \bbH_{t}(\alpha) = R(a) [[0 0 | 0]
    #                          [1 0 | 0]
    #                          [0 1 | 0]]
    # \bbH_{n}(\alpha) = R(a) [[0 0 | 1]
    #                          [0 0 | 0]
    #                          [0 0 | 0]]
    R = Matrix3(
        *(dr.cos(angles), -dr.sin(angles), 0),
        *(dr.sin(angles), dr.cos(angles), 0),
        *(0, 0, 1),
    )
    ray_t_spec = R @ Matrix3(
        *(0, 0, 0),
        *(1, 0, 0),
        *(0, 1, 0),
    )
    ray_n_spec = R @ Matrix3(
        *(0, 0, 1),
        *(0, 0, 0),
        *(0, 0, 0),
    )

    D = detector_spec.ndim + 1
    if D == 2:
        # Drop (row 2, col 1)
        select = lambda mat3: Matrix2(
            *(mat3[0, 0], mat3[0, 2]),
            *(mat3[1, 0], mat3[1, 2]),
        )
        ray_t_spec = select(ray_t_spec)
        ray_n_spec = select(ray_n_spec)

    ll_center = tuple(
        (cs - s) / 2
        for (cs, s) in zip(
            detector_spec.cell_size,
            detector_spec.size,
        )
    )
    ray_u_spec = xrtu.UniformSpec(
        start=(*ll_center, 1),
        step=(*detector_spec.cell_size, 1e-16),
        num=(*detector_spec.num_cell, 1),
    )

    Array = dr.array_t(ray_t_spec)
    ray_spec = (Array(ray_t_spec), Array(ray_n_spec), ray_u_spec)
    return ray_spec


def cone_beam(
    sod: float,
    sdd: float,
    angles: FloatT,
    detector_spec: xrtu.DetectorSpec,
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
    angles: FloatT
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
    assert 0 < sod < sdd

    Float = type(angles)
    Matrix2 = xrtu.float_matrix_t(Float, 2)
    Matrix3 = xrtu.float_matrix_t(Float, 3)

    # For simplicity, we build (t_spec, n_spec) for the 3D case, then reduce matrix sizes if in 2D.
    # \bbH_{t}(\alpha) = R(a) [[0 0 | -sod]
    #                          [0 0 |    0]
    #                          [0 0 |    0]]
    # \bbH_{n}(\alpha) = R(a) [[0 0 | sdd]
    #                          [1 0 |   0]
    #                          [0 1 |   0]]
    R = Matrix3(
        *(dr.cos(angles), -dr.sin(angles), 0),
        *(dr.sin(angles), dr.cos(angles), 0),
        *(0, 0, 1),
    )
    ray_t_spec = R @ Matrix3(
        *(0, 0, -sod),
        *(0, 0, 0),
        *(0, 0, 0),
    )
    ray_n_spec = R @ Matrix3(
        *(0, 0, sdd),
        *(1, 0, 0),
        *(0, 1, 0),
    )

    D = detector_spec.ndim + 1
    if D == 2:
        # Drop (row 2, col 1)
        select = lambda mat3: Matrix2(
            *(mat3[0, 0], mat3[0, 2]),
            *(mat3[1, 0], mat3[1, 2]),
        )
        ray_t_spec = select(ray_t_spec)
        ray_n_spec = select(ray_n_spec)

    ll_center = tuple(
        (cs - s) / 2
        for (cs, s) in zip(
            detector_spec.cell_size,
            detector_spec.size,
        )
    )
    ray_u_spec = xrtu.UniformSpec(
        start=(*ll_center, 1),
        step=(*detector_spec.cell_size, 1e-16),
        num=(*detector_spec.num_cell, 1),
    )

    Array = dr.array_t(ray_t_spec)
    ray_spec = (Array(ray_t_spec), Array(ray_n_spec), ray_u_spec)
    return ray_spec
