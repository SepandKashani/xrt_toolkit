XRT Toolkit
===========

GPU X-ray transforms and basic reconstruction, compiled with `Dr.Jit
<https://drjit.readthedocs.io>`_.

The library provides a forward projector and its **exact matched adjoint** for
arbitrary sets of rays in 2-D and 3-D, on box-spline bases of order 0, 1 or 2,
differentiable with respect to the acquisition geometry. On top of that sit
fused multi-channel operators for tensor tomography, curved-ray travel-time
tomography, time-of-flight weighting, and a small set of reconstruction
algorithms.

.. figure:: _static/forward_adjoint.png

   The operator pair: an image, its sinogram, and the backprojection of that
   sinogram.

.. toctree::
   :maxdepth: 2

   theory
   geometry
   interop
   tutorials
   api

Installation
------------

.. code-block:: bash

   pip install "xrt_toolkit@git+https://github.com/SepandKashani/xrt_toolkit.git@v2"

A CUDA GPU is required. Dr.Jit 1.4+ needs compute capability 7.5 or newer; on
older cards (V100) install ``drjit<1.4``.

Quick start
-----------

.. code-block:: python

   import numpy as np, drjit as dr
   from drjit.cuda.ad import Float
   import xrt_toolkit as xtk

   N = 128
   knot = xtk.UniformSpec(start=(-N/2 + 0.5,) * 2, step=1, num=(N, N))
   rays = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, 180, endpoint=False),
                            xtk.DetectorSpec(size=(1.5 * N,), num_cell=(192,)))

   sino = xtk.xrt_struct_apply(rays, knot, 0, Float(image.reshape(-1)))
   rec = xtk.fbp(rays, knot, sino)
