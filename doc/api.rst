API reference
=============

Projection operators
--------------------

.. autofunction:: xrt_toolkit.xrt_apply

.. code-block:: python

   knot = xtk.UniformSpec(start=(-N/2 + 0.5,) * 2, step=1, num=(N, N))
   rays = (Array2f(t_x, t_y), Array2f(n_x, n_y))     # (2, M) anchors, directions
   y = xtk.xrt_apply(rays, knot, 0, Float(image.reshape(-1)))

.. autofunction:: xrt_toolkit.xrt_adjoint

.. code-block:: python

   b = xtk.xrt_adjoint(rays, knot, 0, y)             # exact transpose of xrt_apply

   # the dot test the suite runs
   lhs = float(dr.sum(xtk.xrt_apply(rays, knot, 0, f) * y).item())
   rhs = float(dr.sum(f * xtk.xrt_adjoint(rays, knot, 0, y)).item())

.. autofunction:: xrt_toolkit.xrt_struct_apply

.. code-block:: python

   # one kernel per projection: lean on memory, slow inside a solver
   y = xtk.xrt_struct_apply(rays_par, knot, order, f)

.. autofunction:: xrt_toolkit.xrt_struct_adjoint

Geometry
--------

.. autofunction:: xrt_toolkit.parallel_beam

.. code-block:: python

   det = xtk.DetectorSpec(size=(1.5 * N,), num_cell=(192,))
   rays = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, 180, endpoint=False), det)

.. autofunction:: xrt_toolkit.cone_beam

.. code-block:: python

   det = xtk.DetectorSpec(size=(2.2 * N, 1.75 * N), num_cell=(216, 168))
   rays = xtk.cone_beam(sod=1.6 * N, sdd=2.8 * N,
                        angles=dr.linspace(Float, 0, 2 * np.pi, 360, endpoint=False),
                        detector_spec=det)

.. autofunction:: xrt_toolkit.struct_rays

.. code-block:: python

   re = xtk.struct_rays(rays)            # expand once, then use the fused kernels
   y = xtk.xrt_apply(re, knot, order, f)

.. autoclass:: xrt_toolkit.UniformSpec
   :members: centered

.. code-block:: python

   # centred on the rotation axis, unit voxels
   knot = xtk.UniformSpec(start=(-N/2 + 0.5,) * 3, step=1, num=(N, N, N))
   knot = xtk.UniformSpec.centered(step=1, num=(N, N, N))      # the same lattice

.. autoclass:: xrt_toolkit.DetectorSpec

Reconstruction
--------------

.. autofunction:: xrt_toolkit.optim.cg

.. code-block:: python

   re = xtk.struct_rays(rays)
   rec = xtk.cg(lambda v: xtk.xrt_apply(re, knot, order, v),
                lambda v: xtk.xrt_adjoint(re, knot, order, v),
                y, N * N, n_iter=20)

.. autofunction:: xrt_toolkit.optim.gd

.. autofunction:: xrt_toolkit.optim.fbp

.. code-block:: python

   rec = xtk.fbp(rays_par, knot, y, window="ramp")     # parallel, 2-D or 3-D

.. autofunction:: xrt_toolkit.optim.fbp_cone

.. code-block:: python

   rec = xtk.fbp_cone(rays_cone, knot, y, sod=sod, sdd=sdd, window="hann")

.. autofunction:: xrt_toolkit.optim.bpf

Geometry derivatives
--------------------

Six functions, one per ray coordinate: the derivative of the projection with
respect to each component of the anchor :math:`\mathbf{t}` and of the
direction :math:`\mathbf{n}`. The ``_z`` pair applies in 3-D only.

.. autofunction:: xrt_toolkit.xrt_ad_t_x
.. autofunction:: xrt_toolkit.xrt_ad_t_y
.. autofunction:: xrt_toolkit.xrt_ad_t_z
.. autofunction:: xrt_toolkit.xrt_ad_n_x
.. autofunction:: xrt_toolkit.xrt_ad_n_y
.. autofunction:: xrt_toolkit.xrt_ad_n_z

.. code-block:: python

   # chain them to differentiate w.r.t. any parameter the rays depend on.
   # here: one scalar detector shift along the in-plane axis (ux, uy)
   resid = xtk.xrt_apply(rays_at(s), knot, order, f) - y_meas
   gx = xtk.xrt_ad_t_x(rays_at(s), knot, order, f)
   gy = xtk.xrt_ad_t_y(rays_at(s), knot, order, f)
   grad = float(dr.sum(resid * (Float(ux) * gx + Float(uy) * gy)).item())

.. code-block:: python

   # in 3-D all six are available, so the full Jacobian of one ray
   # w.r.t. its own six numbers is
   dt = [xtk.xrt_ad_t_x(rays, knot, order, f),
         xtk.xrt_ad_t_y(rays, knot, order, f),
         xtk.xrt_ad_t_z(rays, knot, order, f)]
   dn = [xtk.xrt_ad_n_x(rays, knot, order, f),
         xtk.xrt_ad_n_y(rays, knot, order, f),
         xtk.xrt_ad_n_z(rays, knot, order, f)]

Tensor tomography
-----------------

.. autofunction:: xrt_toolkit.xrt_tensor_apply

.. code-block:: python

   # C channels per voxel, L rays, one traversal for all of them
   w = xtk.lrt_weights(ray_n)                   # (L, C) contraction weights
   y = xtk.xrt_tensor_apply(rays, knot, order, f, w)

.. autofunction:: xrt_toolkit.xrt_tensor_adjoint
.. autofunction:: xrt_toolkit.lrt_weights

Interoperability
----------------

.. autofunction:: xrt_toolkit.from_astra

.. code-block:: python

   rays, knot = xtk.from_astra(proj_geom, vol_geom)
   y = xtk.xrt_apply(rays, knot, order, Float(image.reshape(-1)))

.. autofunction:: xrt_toolkit.interop.vol_astra_to_xtk
.. autofunction:: xrt_toolkit.interop.vol_xtk_to_astra

.. automodule:: xrt_toolkit.torch
   :members: XRTProjector, xrt_torch

.. code-block:: python

   proj = XRTProjector(knot, order=1)           # symbolic mode by default
   y = proj(x, t, n)
   loss = torch.mean((y - y_meas) ** 2)
   loss.backward()                              # x.grad, t.grad, n.grad

Diagnostics
-----------

.. autofunction:: xrt_toolkit.plot_rays

.. code-block:: python

   fig = xtk.plot_rays(rays, knot)              # draws the rays over the lattice

.. autofunction:: xrt_toolkit.plot_2d_basis
