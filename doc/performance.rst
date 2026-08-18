Performance
===========

All figures below were measured on an NVIDIA A100 (80 GB) with Dr.Jit 1.2,
best of several runs after warm-up, with an explicit device synchronisation
inside the timed region.

Scale
-----

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Problem
     - Operation
     - Time
   * - 256\ :sup:`3` volume, 360 x 384\ :sup:`2` detector (53 M rays)
     - forward projection
     - 0.07 s
   * -
     - ``fbp``
     - 0.47 s
   * -
     - ``cg``, 10 iterations
     - 1.6 s
   * - 512\ :sup:`3` volume, 720 x 768\ :sup:`2` detector (425 M rays)
     - forward projection
     - 0.7 s
   * -
     - ``fbp``
     - 3.5 s
   * - 640 x 640 x 836 volume, 800 x 920 x 728 real cone-beam scan
     - ``fbp_cone`` (FDK)
     - 3.4 s

The forward projector sustains roughly 200 G cell-visits per second, and about
52 G voxel-updates per second in the analytic backprojector.

Choosing an interface
---------------------

The structured operators loop over projections and launch one kernel each:
memory-lean for a single pass, but the launch overhead dominates inside a
solver. :py:func:`~xrt_toolkit.struct_rays` expands the same scan into explicit
rays so the fused single-kernel operators can be used instead.

.. list-table::
   :header-rows: 1
   :widths: 55 22 23

   * - 60 angles, 96 detector cells, 64\ :sup:`2` volume
     - Kernel launches
     - Relative cost
   * - ``xrt_struct_apply``
     - 123
     - 1.0
   * - ``struct_rays`` + ``xrt_apply``
     - 1
     - ~0.01

Backprojection
--------------

The analytic methods use a voxel-driven interpolating backprojector; the
matched exact-chord adjoint is reserved for the iterative solvers, where the
pair must be adjoint. The interpolating kernel only gathers, so it avoids the
atomic contention a scattering adjoint pays:

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

Symbolic versus evaluated
-------------------------

Dr.Jit can trace a traversal into a single fused kernel (*symbolic*) or run it
one launch per step (*evaluated*). Everything in the library is symbolic by
default, and the difference is not subtle — 200k rays through a 256\ :sup:`2`
lattice:

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

Against other packages
----------------------

Compared with `mumott <https://mumott.org>`_'s ``SAXSProjectorCUDA`` on the
same tensor-tomography problem (6 coefficient channels, 8 detector segments,
real IRTT geometries). mumott splits the model into a projector and a
basis-set contraction; this library fuses both into one lattice traversal, so
the fair comparison is against the sum of its two stages:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Full forward model
     - mumott
     - XTK
     - Speed-up
   * - 53 k voxels, 399 k rays
     - 6.2 ms
     - 0.75 ms
     - 8.3x
   * - 232 k voxels, 883 k rays
     - 11.9 ms
     - 1.13 ms
     - 10.6x

The adjoint is closer: 2.0x and 1.6x respectively. mumott's backprojector is
voxel-driven and needs no atomics, which is the better design for pure
backprojection at large volumes — worth saying plainly.

Reproducing
-----------

The timings come from scripts in the repository's experiment tree rather than
from this documentation, so they are not regenerated on a docs build. Treat
them as an order of magnitude: they move with GPU, driver and Dr.Jit version.
