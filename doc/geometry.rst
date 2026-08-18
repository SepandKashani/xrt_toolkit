Geometries
==========

A geometry is a set of rays. The library takes them two ways. Both feed the
same kernels.

Explicit rays
-------------

Any pair of arrays is a geometry: a point :math:`\mathbf{t}` and a direction
:math:`\mathbf{n}` for each ray. Nothing has to be regular. Listmode PET,
sparse plasma diagnostics and a scanner you are still calibrating all fit.

.. figure:: _static/geom_explicit.png
   :width: 60%

   Three arbitrary rays, drawn by :py:func:`~xrt_toolkit.plot_rays`.

Structured scans
----------------

:py:func:`~xrt_toolkit.parallel_beam` and :py:func:`~xrt_toolkit.cone_beam`
build standard acquisitions from a list of angles and a
:py:class:`~xrt_toolkit.DetectorSpec`. They store one matrix per projection
rather than every ray, so memory stays flat as the scan grows.

.. list-table::
   :widths: 50 50

   * - .. figure:: _static/geom_parallel.png

          ``parallel_beam``, angles over :math:`[0, \pi)`
     - .. figure:: _static/geom_cone.png

          ``cone_beam``, a full circle, ``sod`` and ``sdd``

Which one to use
----------------

The structured operators loop over projections and launch one kernel each.
That is fine for a single pass. Inside a solver the launch overhead dominates.

:py:func:`~xrt_toolkit.struct_rays` expands a structured scan into explicit
rays in the same order. You then call the fused operators, which run about a
hundred times faster per iteration. The cost is memory: every ray is stored.

.. code-block:: python

   rays = xtk.parallel_beam(angles, det)
   re = xtk.struct_rays(rays)              # expand once
   rec = xtk.cg(lambda v: xtk.xrt_apply(re, knot, order, v),
                lambda v: xtk.xrt_adjoint(re, knot, order, v),
                y, n_voxels, n_iter=20)
