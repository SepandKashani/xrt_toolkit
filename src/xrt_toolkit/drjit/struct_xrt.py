import math
import typing as typ

import drjit as dr

import xrt_toolkit.util as xrtu

from .ray_xrt import xrt_adjoint, xrt_apply

ArrayNNfT = typ.TypeVar("ArrayNNfT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
RaySpecT = tuple[ArrayNNfT, ArrayNNfT, xrtu.UniformSpec]


@dr.syntax
def xrt_struct_apply(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
) -> FloatT:
    r"""
    Compute 2D/3D structured projections.

    Let :math:`f: \bR^{D} \to \bR` such that

    .. math::

       f(\bbx)
       =
       \sum_{\bbq \in \discreteRange{\bbZero}{\bbQ-1}}
       f_{\bbq} \psi_(\bbx - \bbx_{\bbq}),

    with

    .. math::

       f_{\bbq} \in \bR,
       \bbx_{q} = \bbx_{0} + \bbq \odot \bbDelta,
           \bbx_{0} \in \bR^{D},
           \bbDelta \in \bR_{+}^{D},
       \psi(\bbx; \bbE \in \bR^{D \times N}) =
           box-spline with direction vectors
           \{ \bbe_{l} \in \bR^{D} \}_{l=1..N}

    Then ``xrt_struct_apply()`` computes samples of

    .. math::

       \xrt[f](\bbn, \bbt) = \int_{\bR} f(\bbt + \alpha \bbn) d\alpha

    Parameters
    ----------
    ray_spec: tuple[ArrayNNfT, ArrayNNfT, UniformSpec]
        (L,) implicit ray descriptors :math:`\bbt \in \bR^{D}` and directions :math:`\bbn \in \bR^{D}`.

        Expected arguments are:

        * ray_t_spec: ArrayNNfT
          (N_proj, D, D) homogeneous transforms :math:`\bbH_{t} = [\bbA_{t} \in \bR^{D \times (D-1)}, \bbb_{t} \in \bR^{D}]`.
        * ray_n_spec: ArrayNNfT
          (N_proj, D, D) homogeneous transforms :math:`\bbH_{n} = [\bbA_{n} \in \bR^{D \times (D-1)}, \bbb_{n} \in \bR^{D}]`.
        * ray_u_spec: UniformSpec
          D-dim uniform mesh.

        Let :math:`\bbu \in \bR^{D}` be a point on the mesh specified by `ray_u_spec`.
        Then `ray_spec` encodes rays of the form

        .. math::

           \bbt(\bbu) = \bbH_{t} \bbu,
           \bbn(\bbu) = \bbH_{n} \bbu.

        `ray_spec` encodes ``L = N_proj * prod(ray_u_spec.num)`` projections.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Data interpolation order.

        This parameter sets which :math:`\psi` is used to interpolate data values:

        * order = 0 (2D, 3D):

          .. math::

             \bbE = \diag(\bbDelta)

        * order = 1 (2D):

          .. math::

             \bbE = [\bbDelta_{1}           0  \bbDelta_{1}
                               0  \bbDelta_{2} \bbDelta_{2}]

        * order = 2 (2D):

          .. math::

             \bbE = [\bbDelta_{1}           0  \bbDelta_{1}  \bbDelta_{1}
                               0  \bbDelta_{2} \bbDelta_{2} -\bbDelta_{2}]

        The support of :math:`\psi` and its projections can be viewed using :func:`~xrt_toolkit.drjit.diagnostics.plot_2d_basis`.

    data: FloatT
        (Q1,...,QD) flattened C-ordered volume weights :math:`f_{\bbq} \in \bR`.
    buffer: FloatT
        (L,) buffer in which to accumulate projections.
        The ``(N_proj, *ray_u_spec.num)`` mesh-points are serialized in C-order.

    Returns
    -------
    proj: FloatT
        (L,) projections :math:`\xrt[f] \in \bR`.
    """
    ray_t_spec, ray_n_spec, ray_u_spec = ray_spec

    ArrayNNf = type(ray_t_spec)
    ArrayNf = dr.value_t(ArrayNNf)
    Float = dr.value_t(ArrayNf)
    UInt = dr.value_t(dr.uint32_array_t(ArrayNf))

    # type checking ---------------------------------------
    D = dr.size_v(ArrayNf)
    assert (ray_t_spec.ndim == 3) and (D in (2, 3))
    assert type(ray_n_spec) is ArrayNNf
    assert ray_u_spec.ndim == D

    assert knot_spec.ndim == D
    assert order in (0, 1, 2)

    assert type(data) is Float
    assert len(data) == math.prod(knot_spec.num)

    assert (N_proj := ray_t_spec.shape[-1]) == ray_n_spec.shape[-1]
    L_proj = math.prod(ray_u_spec.num)
    L = N_proj * L_proj
    if buffer is None:
        buffer = dr.zeros(Float, L)
    else:
        assert type(buffer) is Float
        assert len(buffer) == L
    # -----------------------------------------------------

    u = [None] * D
    for d, (start, step, num) in enumerate(ray_u_spec):
        u[d] = start + step * dr.arange(Float, num)
    uu = ArrayNf(*dr.meshgrid(*u, indexing="ij"))

    i = UInt(0)
    index = dr.arange(UInt, 0, L_proj)
    while dr.hint(i < N_proj, mode="evaluated"):
        H_t = dr.gather(ArrayNNf, ray_t_spec, i)
        ray_t = H_t @ uu

        H_n = dr.gather(ArrayNNf, ray_n_spec, i)
        ray_n = H_n @ uu
        
        # add a tilt in last dimension for testing
        # ray_n = ray_n + 0.1 * dr.linspace(Float, -1.0, 1.0, ray_n.shape[-1])
        proj = xrt_apply(
            ray_spec=(ray_t, ray_n),
            knot_spec=knot_spec,
            order=order,
            data=data,
        )
        dr.scatter(buffer, proj, index)

        i += 1
        index += L_proj

    return buffer


@dr.syntax
def xrt_struct_adjoint(
    ray_spec: RaySpecT,
    knot_spec: xrtu.UniformSpec,
    order: int,
    data: FloatT,
    buffer: FloatT = None,
) -> FloatT:
    r"""
    Compute 2D/3D structured back-projections.

    Adjoint of ``xrt_struct_apply()``: maps projection weights to volume expansion coefficients.

    Parameters
    ----------
    ray_spec: tuple[ArrayNNfT, ArrayNNfT, UniformSpec]
        (L,) implicit ray descriptors :math:`\bbt \in \bR^{D}` and directions :math:`\bbn \in \bR^{D}`.

        Expected arguments are:

        * ray_t_spec: ArrayNNfT
          (N_proj, D, D) homogeneous transforms :math:`\bbH_{t} = [\bbA_{t} \in \bR^{D \times (D-1)}, \bbb_{t} \in \bR^{D}]`.
        * ray_n_spec: ArrayNNfT
          (N_proj, D, D) homogeneous transforms :math:`\bbH_{n} = [\bbA_{d} \in \bR^{D \times (D-1)}, \bbb_{d} \in \bR^{D}]`.
        * ray_u_spec: UniformSpec
          D-dim uniform mesh.

        Let :math:`\bbu \in \bR^{D}` be a point on the mesh specified by `ray_u_spec`.
        Then `ray_spec` encodes rays of the form

        .. math::

           \bbt(\bbu) = \bbH_{t} \bbu,
           \bbn(\bbu) = \bbH_{n} \bbu.

        `ray_spec` encodes ``L = N_proj * prod(ray_u_spec.num)`` projections.
    knot_spec: UniformSpec
        Volume properties :math:`(\bbx_{0}, \bbDelta, \bbQ)`.
    order: 0 | 1 | 2
        Data interpolation order.

        This parameter sets which :math:`\psi` is used to interpolate data values.
        The support of :math:`\psi` and its projections can be viewed using :func:`~xrt_toolkit.drjit.diagnostics.plot_2d_basis`.
    data: FloatT
        (L,) projections :math:`g_{l} \in \bR`.
        The ``(N_proj, *ray_u_spec.num)`` mesh-points are serialized in C-order.
    buffer: FloatT
        (Q1,...,QD) flattened buffer in which to accumulate back-projected weights :math:`f_{\bbq} \in \bR`.

    Returns
    -------
    b_proj: FloatT
        (Q1,...,QD) flattened C-ordered back-projected weights :math:`f_{\bbq} \in \bR`.
    """
    ray_t_spec, ray_n_spec, ray_u_spec = ray_spec

    ArrayNNf = type(ray_t_spec)
    ArrayNf = dr.value_t(ArrayNNf)
    Float = dr.value_t(ArrayNf)
    UInt = dr.value_t(dr.uint32_array_t(ArrayNf))

    # type checking ---------------------------------------
    D = dr.size_v(ArrayNf)
    assert (ray_t_spec.ndim == 3) and (D in (2, 3))
    assert type(ray_n_spec) is ArrayNNf
    assert ray_u_spec.ndim == D

    assert knot_spec.ndim == D
    assert order in (0, 1, 2)

    assert (N_proj := ray_t_spec.shape[-1]) == ray_n_spec.shape[-1]
    L_proj = math.prod(ray_u_spec.num)
    L = N_proj * L_proj
    assert type(data) is Float
    assert len(data) == L

    if buffer is None:
        buffer = dr.zeros(Float, math.prod(knot_spec.num))
    else:
        assert type(buffer) is Float
        assert len(buffer) == math.prod(knot_spec.num)
    # -----------------------------------------------------

    u = [None] * D
    for d, (start, step, num) in enumerate(ray_u_spec):
        u[d] = start + step * dr.arange(Float, num)
    uu = ArrayNf(*dr.meshgrid(*u, indexing="ij"))

    i = UInt(0)
    index = dr.arange(UInt, 0, L_proj)
    while dr.hint(i < N_proj, mode='evaluated'):
        H_t = dr.gather(ArrayNNf, ray_t_spec, i)
        ray_t = H_t @ uu

        H_n = dr.gather(ArrayNNf, ray_n_spec, i)
        ray_n = H_n @ uu

        proj = dr.gather(Float, data, index)
        xrt_adjoint(
            ray_spec=(ray_t, ray_n),
            knot_spec=knot_spec,
            order=order,
            data=proj,
            buffer=buffer,
        )

        i += 1
        index += L_proj

    return buffer
