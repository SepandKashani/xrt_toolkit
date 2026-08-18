XRT Toolkit
===========

GPU X-ray transforms and reconstruction, compiled with `Dr.Jit
<https://drjit.readthedocs.io>`_.

A forward projector and its **exact matched adjoint** for arbitrary sets of
rays in 2-D and 3-D, on box-spline bases, **differentiable with respect to the
acquisition geometry** — so the scan itself can be calibrated or optimised by
gradient descent, not just the image.

.. figure:: _static/forward_adjoint.png

   The operator pair: an image, its sinogram, and the backprojection of that
   sinogram.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Any geometry
      :link: geometry
      :link-type: doc

      Parallel, fan and cone scans from one call — or hand the projector an
      arbitrary list of rays. Listmode PET and half-calibrated benchtop scanners
      are first-class, not special cases.

   .. grid-item-card:: Differentiable acquisition
      :link: transform
      :link-type: doc

      Analytic derivatives with respect to ray positions *and* directions, so a
      detector offset or a centre of rotation can be a learnable parameter.

   .. grid-item-card:: Reconstruction included
      :link: api
      :link-type: doc

      ``fbp``, ``fbp_cone`` (FDK), ``bpf``, ``cg`` and ``gd`` — filtering on the
      GPU, quantitatively calibrated, a 640\ :sup:`3` cone-beam volume in
      seconds.

   .. grid-item-card:: Beyond attenuation
      :link: api
      :link-type: doc

      Fused multi-channel operators for tensor tomography, curved-ray
      travel-time tomography, and time-of-flight weighting for PET.

.. toctree::
   :maxdepth: 2
   :hidden:

   transform
   geometry
   interop
   pitfalls
   performance
   tutorials
   api

Installation
------------

.. code-block:: bash

   pip install "xrt_toolkit@git+https://github.com/SepandKashani/xrt_toolkit.git@v2"

A CUDA GPU is required. Dr.Jit 1.4+ needs compute capability 7.5 or newer; on
older cards (V100) install ``drjit<1.4``.

Sixty seconds in
----------------

.. code-block:: python

   import numpy as np, drjit as dr
   from drjit.cuda.ad import Float
   import xrt_toolkit as xtk

   N = 128
   knot = xtk.UniformSpec(start=(-N/2 + 0.5,) * 2, step=1, num=(N, N))
   rays = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, 180, endpoint=False),
                            xtk.DetectorSpec(size=(1.5 * N,), num_cell=(192,)))

   sino = xtk.xrt_struct_apply(rays, knot, 0, Float(image.reshape(-1)))
   rec = xtk.fbp(rays, knot, sino)          # filtered backprojection
   rec = xtk.cg(A, At, sino, N * N, 20)     # or solve it iteratively

Then read :doc:`transform` for what the operators actually compute, or jump
straight to the :doc:`tutorials`.
