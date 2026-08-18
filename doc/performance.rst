Performance
===========

All figures come from an NVIDIA A100 with 80 GB, running Dr.Jit 1.2. Each is
the best of several runs after warm-up, timed with a device synchronisation
inside the measured region.

Scale
-----

.. list-table::
   :header-rows: 1
   :widths: 44 30 26

   * - Problem
     - Operation
     - Time
   * - 256\ :sup:`3` volume, 360 x 384\ :sup:`2` detector, 53 M rays
     - forward projection
     - 0.07 s
   * -
     - ``fbp``
     - 0.47 s
   * -
     - ``cg``, 10 iterations
     - 1.6 s
   * - 512\ :sup:`3` volume, 720 x 768\ :sup:`2` detector, 425 M rays
     - forward projection
     - 0.7 s
   * -
     - ``fbp``
     - 3.5 s
   * - 640 x 640 x 836 volume, real 800-view cone-beam scan
     - ``fbp_cone``
     - 3.4 s

The forward projector sustains about 200 G cell visits per second. The
analytic backprojector reaches about 52 G voxel updates per second.

Choice of interface
-------------------

Structured operators launch one kernel per projection.
:py:func:`~xrt_toolkit.struct_rays` expands the same scan into explicit rays,
so the fused operators apply.

.. list-table::
   :header-rows: 1
   :widths: 55 22 23

   * - 60 angles, 96 detector cells, 64\ :sup:`2` volume
     - Kernel launches
     - Relative cost
   * - ``xrt_struct_apply``
     - 123
     - 1.0
   * - ``struct_rays`` and ``xrt_apply``
     - 1
     - about 0.01

Backprojection
--------------

The analytic methods use a voxel-driven interpolating backprojector. The
matched chord adjoint stays inside the iterative solvers, where the pair must
be adjoint. The interpolating kernel only gathers, so it avoids the atomic
contention a scattering adjoint pays.

.. list-table::
   :header-rows: 1
   :widths: 55 22 23

   * - Problem
     - Voxel-driven
     - Chord adjoint
   * - 2-D, 512\ :sup:`2`, 720 x 768
     - 1.2 ms
     - 5.8 ms
   * - 3-D, 256\ :sup:`3`, 360 x 384\ :sup:`2`
     - 33.9 ms
     - 128.4 ms

Symbolic against evaluated
--------------------------

Dr.Jit can trace a traversal into one kernel, which it calls symbolic mode. It
can also run it one launch per step, which it calls evaluated. Everything here
is symbolic by default. On 200k rays through a 256\ :sup:`2` lattice:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Operator
     - Evaluated
     - Symbolic
   * - ``xrt_apply``
     - 814 ms
     - 6.1 ms
   * - ``xrt_adjoint``
     - 762 ms
     - 7.6 ms
   * - ``xrt_ad_t_x``
     - 1358 ms
     - 7.8 ms
   * - ``xrt_ad_n_x``
     - 1654 ms
     - 12.5 ms

Against mumott
--------------

The comparison uses `mumott <https://mumott.org>`_'s ``SAXSProjectorCUDA`` on
the same tensor-tomography problem: 6 channels, 8 detector segments, and the
IRTT geometries of Gao and co-workers [gao2019]_.

mumott splits the model into a projector and a basis-set contraction. This
library fuses both into one traversal, so the fair comparison is against the
sum of its two stages.

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Full forward model
     - mumott
     - XTK
     - Ratio
   * - 53 k voxels, 399 k rays
     - 6.2 ms
     - 0.75 ms
     - 8.3
   * - 232 k voxels, 883 k rays
     - 11.9 ms
     - 1.13 ms
     - 10.6

The adjoint is closer, at 2.0 and 1.6. On the larger bone volume mumott wins.
Its backprojector is voxel-driven and needs no atomics, which is the better
design for pure backprojection at that size.

Reproducing
-----------

These timings come from scripts in the experiment tree, not from the
documentation build. Treat them as orders of magnitude. They move with the
GPU, the driver and the Dr.Jit version.
