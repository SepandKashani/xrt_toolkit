Conventions and pitfalls
========================

Each entry below cost someone a debugging session.

Work in voxels
--------------

Set ``step=1`` and scale ``sod``, ``sdd`` and the detector pitch by
``1/voxel_size``. Do not express the geometry in millimetres.

The kernels are single precision. A ray parameterised in millimetres over a
lattice with millimetre steps loses most of its digits inside the traversal.
On one calibration problem the millimetre form gave geometry derivatives two
orders of magnitude too small. In voxel units the same code matched finite
differences.

Angle ranges
------------

Parallel scans span :math:`[0, \pi)`. Views at :math:`\theta` and
:math:`\theta + \pi` trace the same lines, so a sweep over :math:`2\pi`
doubles the work and adds nothing.

Cone scans span :math:`[0, 2\pi)`. A divergent beam sees different paths at
:math:`\theta` and :math:`\theta + \pi`.

Array layout
------------

Volumes are flat and C-ordered, indexed ``(x, y[, z])``, every axis
increasing. Sinograms follow the ray order of the geometry that produced them.

Centre the lattice on the rotation axis with ``start=(-N/2 + 0.5,) * D``. The
structured geometries rotate about the origin. A lattice that starts at zero
reconstructs a rotating object.

Pick the right interface
------------------------

Structured operators launch one kernel per projection. That is fine for a
single pass and crippling inside a solver. Expand once with
:py:func:`~xrt_toolkit.struct_rays` and use the explicit operators there.

Expanding costs memory: about 24 bytes per ray, plus the sinogram. A
720 x 768\ :sup:`2` scan holds 425 M rays, so 10 GB of coordinates. On a
16 GB card, stream with the structured operators instead.

Symbolic, not evaluated
-----------------------

Every operator takes ``mode``. It defaults to ``"symbolic"``, which traces the
traversal into one kernel. ``"evaluated"`` launches a kernel per step and runs
one to two orders of magnitude slower for the same numbers. If something is
slow for no reason, check that nothing passed ``mode="evaluated"``.

Basis order in 3-D
------------------

In 2-D, orders 0, 1 and 2 all cost about the same. In 3-D, any order above 0
evaluates a learned spline footprint. That is far slower, and its fast path
needs cooperative vectors, so a Turing card and driver R570. Start 3-D work at
``order = 0``.

Detector finer than the lattice
-------------------------------

The ramp filter stops at the lattice Nyquist frequency. The volume cannot hold
anything above it, and those frequencies would only alias. Refining the
detector without refining the lattice therefore stops helping.

Iterative reconstruction on real data
-------------------------------------

Three things make :py:func:`~xrt_toolkit.cg` look worse than
:py:func:`~xrt_toolkit.fbp_cone` on a real scan. None of them is the solver.

**Unconstrained corners.** The volume is a box. The beam covers a cylinder.
The corners are seen by only some views, so the residual collects there and
leaks inward. Solve on the support instead. Use ``A(v*m)`` and ``m*At(r)``
with a cylinder mask ``m``, which keeps the pair adjoint.

**Axial truncation.** If the object runs past the reconstructed volume, the
measured rays include material the model has no voxels for. The solver absorbs
it as an offset that varies with height, which is why slices degrade away from
the mid-plane. Pad the volume to the axial reach of the detector, then crop.
Analytic methods are immune, because the ramp filter kills DC and they never
try to explain those rays.

**Iteration count.** Conjugate gradients converge low frequencies first, so an
early result is blurry. Padding adds unknowns and needs more iterations. Push
too far and sharpness and noise climb together. Choose the stopping point by
measuring, on a held-out subset of projections.

Environment
-----------

**Dr.Jit and old GPUs.** Dr.Jit 1.4 requires compute capability 7.5, so
Turing. On a V100 it hangs at import instead of reporting an error. Pin
``drjit<1.4`` there.

**CuPy and cuFFT.** :py:mod:`xrt_toolkit.optim` filters through CuPy. A CuPy
built against a different CUDA version than the driver cannot load cuFFT. The
library then falls back to NumPy on the host. Results stay correct and every
reconstruction pays a device round trip. It warns. Do not ignore the warning.

**Zero-copy boundaries.** ``Float(cupy_array)`` is a view of CuPy pool memory.
If the CuPy array dies while the Dr.Jit array lives, that memory can go to the
next allocation, and your values change under you. Copy across such a boundary
before the source goes out of scope.
