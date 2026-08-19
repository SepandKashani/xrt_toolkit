Interoperability
================

Migrating from ASTRA
--------------------

:py:func:`~xrt_toolkit.from_astra` converts an ASTRA ``(proj_geom, vol_geom)``
pair into the ray form this library uses. Keep the geometry code you have and
swap the projector.

.. code-block:: python

   import astra
   import xrt_toolkit as xtk

   vol_geom  = astra.create_vol_geom(N, N)
   proj_geom = astra.create_proj_geom("parallel", 1.0, n_det,
                                      np.linspace(0, np.pi, n_ang, endpoint=False))

   rays, knot = xtk.from_astra(proj_geom, vol_geom)
   sino = xtk.xrt_apply(rays, knot, order, Float(image.reshape(-1)))

It supports ``parallel``, ``fanflat``, ``parallel3d``, ``cone`` and their
``*_vec`` forms. The ``*_vec`` forms are plain dictionaries, so a converted
geometry runs on a machine without ASTRA.

Three details matter when you port code.

**Ray order.** The rays come back in the order of the flattened ASTRA
sinogram: ``(angles, det)`` in 2-D, ``(det_v, angles, det_u)`` in 3-D. Reshape
as before. No permutation.

**Array layout.** The two packages order volume axes differently. Use
:py:func:`~xrt_toolkit.interop.vol_astra_to_xtk` and
:py:func:`~xrt_toolkit.interop.vol_xtk_to_astra` instead of transposing by
hand.

**Units.** Lengths are in voxels. If the voxel size is :math:`s`, projections
here equal the ASTRA ones divided by :math:`s`. Multiply by :math:`s` to
compare.

.. note::

   The two projectors interpolate differently, so values agree to the accuracy
   of the discretisation, not bit for bit. On a 128\ :sup:`2` disc with
   180 x 192 rays, ``from_astra`` matches the equivalent
   :py:func:`~xrt_toolkit.parallel_beam` scan to 7e-4 on a peak line integral
   of 104.

.. tip::

   Real archives carry conventions the geometry file does not state. In the
   Walnut collection of Der Sarkissian and co-workers, each frame is stored
   transposed and flipped, so ``np.transpose(np.flipud(image))`` gives the
   ``(v, u)`` order the geometry expects, and the projections pair with the
   geometry rows in reverse order. Get either wrong and the reconstruction
   comes out as concentric rings, with no other clue.

PyTorch
-------

:py:mod:`xrt_toolkit.torch` wraps the projector as a
:class:`torch.autograd.Function`. It drops into a graph and backpropagates.
Tensors stay on the GPU, since the conversions to and from Dr.Jit copy
nothing.

The kernels run in symbolic mode, so each call compiles to one fused kernel.
That is the default and it matters. On 200k rays through a 256\ :sup:`2`
lattice, a forward plus backward takes 0.36 s. In evaluated mode the same work
takes 9.2 s and gives the same numbers. ``mode="evaluated"`` remains available.

.. code-block:: python

   import torch
   import xrt_toolkit as xtk
   from xrt_toolkit.torch import XRTProjector

   knot = xtk.UniformSpec(start=(-N/2 + 0.5,) * 2, step=1, num=(N, N))
   proj = XRTProjector(knot, order=1)          # symbolic by default

   x = torch.rand(N * N, device="cuda", requires_grad=True)   # image
   t = torch.nn.Parameter(t0)                                 # (2, M) ray anchors
   n = torch.nn.Parameter(n0)                                 # (2, M) ray directions

   y = proj(x, t, n)
   loss = torch.mean((y - y_meas) ** 2)
   loss.backward()                      # fills x.grad, t.grad and n.grad

Three inputs receive gradients, not one. The image gradient goes through the
exact adjoint. The ray anchors and directions go through
:py:func:`~xrt_toolkit.xrt_ad_t_x` and
:py:func:`~xrt_toolkit.xrt_ad_n_x`. An optimiser can therefore treat detector
positions as learnable parameters.

.. warning::

   Geometry gradients are implemented and tested for 2-D. Image gradients work
   in 2-D and 3-D. In 3-D, ``order`` above 0 uses the learned spline
   footprint, which is much slower than ``order = 0``.
