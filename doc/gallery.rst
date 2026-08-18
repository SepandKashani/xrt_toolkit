Gallery
=======

Every figure on this page comes from ``doc/make_figures.py``, which runs the
library on a synthetic phantom. Rebuild them with one command.

Cone-beam CT
------------

A 192\ :sup:`3` phantom, 360 views, reconstructed by FDK in one call to
:py:func:`~xrt_toolkit.fbp_cone`.

.. figure:: _static/gallery_cone.png

   Three orthogonal slices of the reconstructed volume.

Fitting the geometry
--------------------

The scan below has an unknown detector shift. Gradient descent on the ray
positions recovers it to four decimals, using
:py:func:`~xrt_toolkit.xrt_ad_t_x` and its siblings. The reconstruction on the
left assumes no shift. The middle one uses the fitted value.

.. figure:: _static/gallery_calibration.png

   True shift 1.5 voxels, fitted 1.5000.

Few views
---------

Twenty projections. Filtered backprojection streaks. Conjugate gradients on
the same rays does better, because it fits the data instead of inverting an
incomplete integral.

.. figure:: _static/gallery_sparse.png

   Phantom, ``fbp`` with a Hann window, and ``cg`` after 40 iterations.

Basis functions
---------------

The support of the box spline and its projections, drawn by
:py:func:`~xrt_toolkit.plot_2d_basis`.

.. list-table::
   :widths: 33 33 33

   * - .. figure:: _static/basis_0.png

          order 0
     - .. figure:: _static/basis_1.png

          order 1
     - .. figure:: _static/basis_2.png

          order 2
