Gallery
=======

Every figure comes from ``doc/make_figures.py``, which runs the library on a
synthetic phantom. Rebuild them with one command.

Cone-beam CT
------------

A 192\ :sup:`3` phantom, 720 views, reconstructed by FDK in one call.

.. figure:: _static/gallery_cone.png

   Three orthogonal slices of the reconstructed volume.

.. code-block:: python

   sod, sdd = 8.0 * N, 10.0 * N
   rays = xtk.cone_beam(sod=sod, sdd=sdd,
                        angles=dr.linspace(Float, 0, 2*np.pi, 720, endpoint=False),
                        detector_spec=xtk.DetectorSpec(size=(1.6*N, 1.6*N),
                                                       num_cell=(384, 384)))
   y = xtk.xrt_apply(xtk.struct_rays(rays), knot, 0, Float(phantom.reshape(-1)))
   rec = xtk.fbp_cone(rays, knot, y, sod=sod, sdd=sdd, window="shepp-logan")

Fitting the geometry
--------------------

The scan below has an unknown detector shift. Gradient descent on the ray
positions recovers it to four decimals. The left panel assumes no shift. The
middle one uses the fitted value.

.. figure:: _static/gallery_calibration.png

   True shift 1.5 voxels, fitted 1.5000.

.. code-block:: python

   def rays_at(s):                      # slide the detector along its own axis
       return (Array2f(t0[0] + s*ux, t0[1] + s*uy), Array2f(n0))

   s, lr = 0.0, 2e-6
   for _ in range(40):
       rays = rays_at(s)
       resid = xtk.xrt_apply(rays, knot, 1, f) - y_meas
       gx = xtk.xrt_ad_t_x(rays, knot, 1, f)     # d(Af) / d t_x
       gy = xtk.xrt_ad_t_y(rays, knot, 1, f)     # d(Af) / d t_y
       s -= lr * float(dr.sum(resid * (Float(ux)*gx + Float(uy)*gy)).item())

Few views
---------

Twenty projections. Filtered backprojection streaks. Conjugate gradients on
the same rays does better, because it fits the data instead of inverting an
incomplete integral.

.. figure:: _static/gallery_sparse.png

   Phantom, ``fbp`` with a Hann window, and ``cg`` after 40 iterations.

.. code-block:: python

   rays = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, 20, endpoint=False), det)
   re = xtk.struct_rays(rays)
   y = xtk.xrt_apply(re, knot, 0, f)

   a = xtk.fbp(rays, knot, y, window="hann")
   b = xtk.cg(lambda v: xtk.xrt_apply(re, knot, 0, v),
              lambda v: xtk.xrt_adjoint(re, knot, 0, v), y, N*N, n_iter=40)

Basis functions
---------------

The support of the box spline and its projections.

.. list-table::
   :widths: 33 33 33

   * - .. figure:: _static/basis_0.png

          order 0
     - .. figure:: _static/basis_1.png

          order 1
     - .. figure:: _static/basis_2.png

          order 2

.. code-block:: python

   ang = np.linspace(0, np.pi, 5, endpoint=False)
   ray_n = Array2f(np.cos(ang, dtype=np.float32), np.sin(ang, dtype=np.float32))
   fig = xtk.plot_2d_basis(knot, order, ray_n)
