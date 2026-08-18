Conventions and pitfalls
========================

Most of the entries below cost somebody a debugging session. They are
collected here so they cost you a paragraph instead.

Units: work in voxels
---------------------

Express geometry in units of the voxel size — set ``step=1`` and scale
``sod``, ``sdd`` and the detector pitch by ``1/voxel_size`` — rather than in
millimetres.

This is not cosmetic. The kernels are single precision, and a ray parameterised
in millimetres over a lattice with millimetre-scale steps loses most of its
significant digits inside the traversal. On a calibration problem where the
geometry derivatives were compared against finite differences, the millimetre
parameterisation produced gradients roughly **two orders of magnitude too
small**; the same code in voxel units matched to the expected accuracy.

Angle ranges
------------

Parallel scans span :math:`[0, \pi)`. Views at :math:`\theta` and
:math:`\theta + \pi` traverse the same lines and carry no extra information, so
sweeping :math:`2\pi` doubles the traced rays for nothing.

Cone scans span :math:`[0, 2\pi)`: a divergent beam sees genuinely different
paths at :math:`\theta` and :math:`\theta + \pi`.

Array layout
------------

Volumes are flat, C-ordered, indexed ``(x, y[, z])`` with every axis
increasing. Sinograms follow the ray order of the geometry that produced them:
``(angles, detector)`` in 2-D, ``(det_v, angles, det_u)`` in 3-D for
:py:func:`~xrt_toolkit.from_astra`.

Centre the lattice on the rotation axis with
``start=(-N/2 + 0.5,) * D``. The structured geometries rotate about the origin,
so a lattice starting at zero reconstructs a rotating object.

Pick the right interface
------------------------

Structured operators launch one kernel per projection: fine for a single pass,
crippling inside a solver. Expand once with
:py:func:`~xrt_toolkit.struct_rays` and use the explicit operators there — the
same rays in the same order, about a hundred times less launch overhead.

The cost of expanding is memory: roughly 24 bytes per ray for coordinates plus
the sinogram. A 720 x 768\ :sup:`2` scan is 425 M rays, so about 10 GB of ray
coordinates alone. On a 16 GB card, stream with the structured operators
instead.

Symbolic, not evaluated
-----------------------

Every operator takes ``mode``, defaulting to ``"symbolic"``, which traces the
traversal into one fused kernel. ``"evaluated"`` launches a kernel per step and
is one to two orders of magnitude slower for identical results. If something is
mysteriously slow, check that nothing passed ``mode="evaluated"``.

Basis order in 3-D
------------------

In 2-D, orders 0, 1 and 2 are all cheap. In 3-D, ``order > 0`` evaluates a
learned spline footprint that is far slower than ``order = 0`` and needs
cooperative vectors (Turing or newer, driver R570+) for its fast path. For 3-D
work start at ``order = 0``.

Detector finer than the lattice
-------------------------------

If the detector samples more finely than the reconstruction lattice, the ramp
filter is cut at the *lattice* Nyquist: frequencies above it cannot be
represented by the volume and would only alias. Refining the detector without
refining the lattice therefore stops helping — and if the scaling is wrong, it
actively hurts.

Iterative reconstruction on real data
-------------------------------------

Three things routinely make :py:func:`~xrt_toolkit.cg` look worse than
:py:func:`~xrt_toolkit.fbp_cone` on a real scan, and none of them is the
solver:

**Unconstrained corners.** The volume is a box, the beam covers a cylinder. The
corners are seen by only some views, so the residual collects there and leaks
inward. Solve on the support instead — ``A(v*m)`` and ``m*At(r)`` with a
cylinder mask ``m`` keeps the pair adjoint.

**Axial truncation.** If the object extends past the reconstructed volume, the
measured rays include material the model has no voxels for, and the solver
absorbs it as a *z*-dependent offset — which is exactly why slices degrade with
distance from the mid-plane. Pad the volume to the detector's axial reach and
crop afterwards. Analytic methods are immune because the ramp filter kills DC:
they never try to *explain* those rays.

**Iteration count.** Conjugate gradients converge low frequencies first, so an
under-converged result is genuinely blurry, and padding adds unknowns that need
more iterations. Push too far and both sharpness and noise climb together
(semi-convergence). Choose the stopping point by measuring — a held-out subset
of projections is the honest way — rather than by eye.

Environment
-----------

**Dr.Jit and old GPUs.** Dr.Jit 1.4+ requires compute capability 7.5 (Turing).
On a V100 it does not report a clean error — it hangs at import. Pin
``drjit<1.4`` there.

**CuPy and cuFFT.** :py:mod:`xrt_toolkit.optim` filters through CuPy. A CuPy
built against a different CUDA version than the driver fails to load cuFFT, and
the library falls back to NumPy on the host — correct results, a device
round-trip per reconstruction. It emits a ``RuntimeWarning`` saying so; do not
ignore it.

**Zero-copy boundaries.** ``Float(cupy_array)`` is a *view* of CuPy pool memory.
If the CuPy array dies while the Dr.Jit array is still live, that memory can be
handed to the next allocation — the values change under you, intermittently and
depending on timing. Materialise across such a boundary before letting the
source go.
