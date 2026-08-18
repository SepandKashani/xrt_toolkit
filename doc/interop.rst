Interoperability
================

Migrating from ASTRA
--------------------

:py:func:`~xrt_toolkit.from_astra` converts an ASTRA ``(proj_geom, vol_geom)``
pair into the explicit ray parameterisation this library uses, so existing
ASTRA code can be ported a step at a time: keep the geometry construction you
already have, and swap the projector.

.. code-block:: python

   import astra
   import xrt_toolkit as xtk

   vol_geom  = astra.create_vol_geom(N, N)
   proj_geom = astra.create_proj_geom("parallel", 1.0, n_det,
                                      np.linspace(0, np.pi, n_ang, endpoint=False))

   rays, knot = xtk.from_astra(proj_geom, vol_geom)
   sino = xtk.xrt_apply(rays, knot, order, Float(image.reshape(-1)))

``parallel``, ``fanflat``, ``parallel3d``, ``cone`` and their ``*_vec``
variants are supported. The ``*_vec`` forms need no ASTRA import at all — they
are plain dictionaries — so a converted geometry can be stored and replayed on
a machine where ASTRA is not installed.

Three details matter when porting:

**Ray ordering.** The returned rays are ordered exactly like the flattened
ASTRA sinogram: ``(angles, det)`` in 2-D and ``(det_v, angles, det_u)`` in 3-D.
A sinogram can therefore be reshaped as before, with no permutation.

**Array layout.** ASTRA and this library disagree on volume axis order; use
:py:func:`~xrt_toolkit.interop.vol_astra_to_xtk` and
:py:func:`~xrt_toolkit.interop.vol_xtk_to_astra` to move volumes across rather
than transposing by hand.

**Units.** Lengths are in units of the voxel size. With a voxel size
:math:`s \neq 1`, projections computed here equal the ASTRA ones divided by
:math:`s`; scale by :math:`s` to compare numbers directly.

.. note::

   Interpolation differs between the two projectors, so values agree to the
   accuracy of the discretisation rather than bit-exactly. On a 128\ :sup:`2`
   disc with 180 x 192 rays, ``from_astra`` reproduces the equivalent
   :py:func:`~xrt_toolkit.parallel_beam` scan to a maximum absolute deviation
   of 7e-4 on a peak line integral of 104.

PyTorch
-------

:py:mod:`xrt_toolkit.torch` wraps the projector as a
:class:`torch.autograd.Function`, so it drops into a PyTorch graph and
backpropagates. Tensors stay on the GPU — the conversions to and from Dr.Jit
are zero-copy.

The kernels run in Dr.Jit's **symbolic** mode, so every call compiles to a
single fused kernel instead of launching one per traversal step. That is the
default and it matters: on 200k rays through a 256\ :sup:`2` lattice a full
forward-plus-backward takes 0.36 s symbolically against 9.2 s in evaluated
mode, for identical results. ``mode="evaluated"`` remains available as an
escape hatch.

What makes this more than a convenience wrapper is *which* inputs receive
gradients: not only the image, but the acquisition geometry itself.

.. code-block:: python

   import torch
   import xrt_toolkit as xtk
   from xrt_toolkit.torch import XRTProjector

   knot = xtk.UniformSpec(start=(-N/2 + 0.5,) * 2, step=1, num=(N, N))
   proj = XRTProjector(knot, order=1)          # mode="symbolic" by default

   x = torch.rand(N * N, device="cuda", requires_grad=True)   # image
   t = torch.nn.Parameter(t0)                                 # (2, M) ray anchors
   n = torch.nn.Parameter(n0)                                 # (2, M) ray directions

   y = proj(x, t, n)                    # (M,) projections
   loss = torch.mean((y - y_meas) ** 2)
   loss.backward()                      # fills x.grad, t.grad and n.grad

The image gradient goes through the exact adjoint; the geometry gradients go
through the analytic derivatives
:py:func:`~xrt_toolkit.xrt_ad_t_x` and :py:func:`~xrt_toolkit.xrt_ad_n_x`.
That means an optimiser can treat detector positions and ray directions as
learnable parameters — self-calibration, or an end-to-end network whose
acquisition geometry is trained together with the reconstruction. The
differentiation section of the 2-D tutorial does exactly this without PyTorch;
this module is the same capability inside an autograd graph.

.. warning::

   Geometry gradients are implemented and validated for 2-D geometries. Image
   gradients work in 2-D and 3-D. In 3-D, ``order`` > 0 routes through the
   learned spline footprint, which is markedly slower than ``order = 0``.
