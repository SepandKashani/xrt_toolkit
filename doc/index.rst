XRT Toolkit
===========

.. rst-class:: xtk-lead

   GPU X-ray transforms and reconstruction, built on `Dr.Jit
   <https://drjit.readthedocs.io>`_. Project a volume along any set of rays,
   apply the exact transpose, and differentiate with respect to the rays
   themselves.

.. grid:: 1 1 2 2
   :gutter: 3
   :class-container: xtk-hero

   .. grid-item::
      :columns: 12 12 8 8

      .. code-block:: python

         import xrt_toolkit as xtk

         knot = xtk.UniformSpec.centered(step=1.0, num=(512, 512))
         det = xtk.DetectorSpec(size=(768.0,), num_cell=(768,))
         rays = xtk.parallel_beam(angles, det)

         sino = xtk.xrt_struct_apply(rays, knot, 1, volume)
         rec = xtk.fbp(rays, knot, sino, window="hann")

   .. grid-item::
      :columns: 12 12 4 4

      .. button-ref:: gallery
         :ref-type: doc
         :color: primary
         :expand:

         Real-data gallery

      .. button-ref:: geometry
         :ref-type: doc
         :color: primary
         :outline:
         :expand:

         Conventions

      .. button-link:: https://github.com/SepandKashani/xrt_toolkit
         :color: primary
         :outline:
         :expand:

         Source

Both directions work in 2-D and 3-D, on box-spline bases, so you can fit the
scan geometry as well as the image.

.. figure:: _static/forward_adjoint.png

   An image, its sinogram, and the backprojection of that sinogram.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Any geometry
      :link: geometry
      :link-type: doc

      Parallel, fan and cone scans come from one call. You can also pass a
      plain list of rays. Listmode PET and an uncalibrated benchtop scanner
      need no special case.

   .. grid-item-card:: Differentiable acquisition
      :link: transform
      :link-type: doc

      Derivatives with respect to ray positions and directions. A detector
      offset or a centre of rotation can be a learnable parameter.

   .. grid-item-card:: Reconstruction included
      :link: api
      :link-type: doc

      ``fbp``, ``fbp_cone``, ``bpf``, ``cg`` and ``gd``. Filtering runs on the
      GPU. A 640\ :sup:`3` cone-beam volume takes 3.4 s.

   .. grid-item-card:: Beyond attenuation
      :link: api
      :link-type: doc

      Fused multi-channel operators for tensor tomography, curved-ray
      travel-time tomography, and time-of-flight weighting for PET.

.. toctree::
   :maxdepth: 2
   :hidden:

   gallery
   transform
   geometry
   interop
   performance
   tutorials
   api
   pitfalls
   citing
   bibliography

Installation
------------

.. code-block:: bash

   pip install "xrt_toolkit@git+https://github.com/SepandKashani/xrt_toolkit.git@v2"

You need a CUDA GPU. Dr.Jit 1.4 requires compute capability 7.5 or newer. On
older cards such as the V100, install ``drjit<1.4``.

First steps
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

Read :doc:`transform` for what the operators compute. Or open the
:doc:`tutorials`.
