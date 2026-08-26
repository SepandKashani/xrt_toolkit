import drjit as dr

import xrt_toolkit.util as xrtu

from .struct_xrt import FloatT, RaySpecT


def parallel_beam(
    angles: FloatT,
    detector_spec: xrtu.DetectorSpec,
) -> RaySpecT:
    r"""
    Create parallel/cylinder-beam structured ray specification.

    This is a helper function to create pre-defined projection geometries for :py:func:`~xrt_toolkit.drjit.xrt_struct_apply` or :py:func:`~xrt_toolkit.drjit.xrt_struct_adjoint`.

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

    This is a helper function to create pre-defined projection geometries for :py:func:`~xrt_toolkit.drjit.xrt_struct_apply` or :py:func:`~xrt_toolkit.drjit.xrt_struct_adjoint`.

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


def struct_rays(ray_spec):
    r"""
    Expand a structured scan into explicit per-ray ``(t, n)`` arrays.

    The structured operators (:py:func:`~xrt_toolkit.xrt_struct_apply` /
    :py:func:`~xrt_toolkit.xrt_struct_adjoint`) process one projection per
    kernel launch, which keeps memory use flat but costs hundreds of launches
    per call. The explicit operators (:py:func:`~xrt_toolkit.xrt_apply` /
    :py:func:`~xrt_toolkit.xrt_adjoint`) trace all rays in a single fused
    kernel and are the right choice inside iterative solvers. This helper
    converts the former representation into the latter; both produce rays in
    the same order, so measurement vectors are interchangeable.

    Parameters
    ----------
    ray_spec: RaySpecT
        Structured scan from :py:func:`parallel_beam` or :py:func:`cone_beam`.

    Returns
    -------
    rays: tuple[ArrayNfT, ArrayNfT]
        Explicit ``(t, n)``, one entry per (projection, detector cell) in
        projection-major order.
    """
    ray_t_spec, ray_n_spec, ray_u_spec = ray_spec
    ArrayNNf = type(ray_t_spec)
    ArrayNf = dr.value_t(ArrayNNf)
    Float = dr.value_t(ArrayNf)
    UInt = dr.value_t(dr.uint32_array_t(ArrayNf))

    N_proj = dr.width(ray_t_spec)
    u = [start + step * dr.arange(Float, num)
         for (start, step, num) in ray_u_spec]
    uu = ArrayNf(*dr.meshgrid(*u, indexing="ij"))
    L_proj = dr.width(uu)

    # replicate: projection index varies slowest, detector cells fastest
    idx = dr.arange(UInt, N_proj * L_proj)
    H_t = dr.gather(ArrayNNf, ray_t_spec, idx // L_proj)
    H_n = dr.gather(ArrayNNf, ray_n_spec, idx // L_proj)
    uu_rep = dr.gather(ArrayNf, uu, idx % L_proj)
    return (H_t @ uu_rep, H_n @ uu_rep)
