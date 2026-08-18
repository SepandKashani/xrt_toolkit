Compared with other packages
============================

ASTRA, TIGRE and mumott are mature and well tested. Use them when they fit.
This page says where this library differs, and where the others are better.

.. list-table::
   :header-rows: 1
   :widths: 26 18 18 18 20

   * -
     - XTK
     - ASTRA
     - TIGRE
     - mumott
   * - Arbitrary ray lists
     - yes
     - yes (``*_vec``)
     - limited
     - no
   * - Exact matched adjoint
     - yes
     - approximate
     - approximate
     - approximate
   * - Derivatives w.r.t. ray geometry
     - yes
     - no
     - no
     - no
   * - Bases beyond voxels
     - orders 0, 1, 2
     - voxels
     - voxels
     - voxels
   * - Tensor tomography
     - fused, any channel count
     - no
     - no
     - yes
   * - Reconstruction algorithms
     - five
     - many
     - many
     - several
   * - Ecosystem and age
     - new
     - large
     - large
     - focused

What this library does better
-----------------------------

**The adjoint is the transpose.** The same traversal visits the same cells
with the same weights, and scatters instead of gathering. Iterative solvers
need that identity, and the test suite checks it to machine precision.

**The geometry is differentiable.** You get analytic derivatives with respect
to ray positions and directions. That turns calibration into an optimisation
problem. No other package on this list offers it.

**Rays are a first-class input.** Nothing has to lie on a grid.

**Multi-channel operators are fused.** Tensor tomography traverses the lattice
once for all channels. On the IRTT problem the full forward model runs 8 to 11
times faster than mumott. See :doc:`performance`.

What the others do better
-------------------------

**ASTRA** [vanaarle2016]_ has a decade of use, a large algorithm library, MATLAB bindings and
a wide user base. If you want a proven pipeline, start there.

**TIGRE** [biguri2016]_ ships far more reconstruction algorithms than the five here, along
with motion correction and scatter tools for cone-beam CT.

**mumott** has the better backprojector for large volumes. It is voxel-driven
and needs no atomics, and it beats this library on the bone dataset adjoint.
It is also a complete tensor-tomography pipeline, not only an operator.

**All three** are older, have been read by more people, and have been used on
more real data.
