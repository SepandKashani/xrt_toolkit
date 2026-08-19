Gallery
=======

Real data
---------

Every reconstruction here comes from measured data in a public archive. The
scripts are in ``xtk_experiments/realdata``.

Cryo-electron tomography
~~~~~~~~~~~~~~~~~~~~~~~~

A whole *Vibrio cholerae* cell, CZ CryoET Data Portal dataset 10489
[czi10489]_. 41 tilts from -53 to +67 degrees, 1023 x 1440, 13.3 A per pixel.

.. figure:: _static/real_cryoet_vibrio.png
   :width: 78%

   The cell envelope, both polyphosphate granules and the appendage.

Plain CG on :math:`\mathbf{A}^{\top}\mathbf{A}` returns a blur here: 41 tilts
over 120 degrees leave the problem badly underdetermined, and
:math:`\mathbf{A}^{\top}\mathbf{A}` behaves like :math:`1/|k|`. Weighting the
normal equations by the ramp filter :math:`\mathbf{W}` fixes that, because
:math:`\mathbf{A}^{\top}\mathbf{W}\mathbf{A}` is close to the identity. Eight
iterations take 8 seconds.

.. code-block:: python

   # single-axis tilt, parallel beam, tilt axis along image y
   for th in np.deg2rad(angles):
       c, s = np.cos(th), np.sin(th)
       tx.append(U * c); ty.append(V); tz.append(-U * s)
       nx.append(np.full(U.size, s)); ny.append(0 * U); nz.append(np.full(U.size, c))
   ray = (Array3f(cat(tx), cat(ty), cat(tz)), Array3f(cat(nx), cat(ny), cat(nz)))

   def ramp(d):                       # W: ramp filter across the tilt axis
       a = np.asarray(d).reshape(n_tilt, NU, NV)
       return Float(np.real(np.fft.ifft(np.fft.fft(a, axis=1) * W[None, :, None],
                                        axis=1)).astype(np.float32).ravel())

   A  = lambda f: xrt_apply(ray, knot, 0, f)
   At = lambda d: S * xrt_adjoint(ray, knot, 0, d)
   rec = cg(lambda f: At(ramp(A(f))), At(ramp(y)), n_iter=8)

Cone-beam CT
~~~~~~~~~~~~

Walnut 1 of the collection of Der Sarkissian and co-workers
[dersarkissian2019]_. The dataset ships an ASTRA ``cone_vec`` geometry file, so
:py:func:`~xrt_toolkit.from_astra` reads it as it stands.

.. figure:: _static/real_ct_walnut_conebeam.png

   603 projections, 113 M rays, a 500\ :sup:`3` reconstruction by 30
   conjugate-gradient iterations. Shell, kernel and septum resolve.

.. code-block:: python

   g = np.loadtxt("scan_geom_corrected.geom")     # ASTRA cone_vec, 12 columns
   g[:, 0:6] /= voxel_mm                          # positions    -> voxel units
   g[:, 6:12] *= bin / voxel_mm                   # detector axes -> voxel units

   rays, knot = xtk.from_astra(
       {"type": "cone_vec", "DetectorRowCount": n_v,
        "DetectorColCount": n_u, "Vectors": g}, vol_geom)

   y = Float(np.transpose(L, (1, 0, 2)).reshape(-1))   # (det_v, angles, det_u)
   rec = xtk.cg(lambda v: xtk.xrt_apply(rays, knot, 0, v),
                lambda v: xtk.xrt_adjoint(rays, knot, 0, v), y, N**3, n_iter=30)

Tensor tomography
~~~~~~~~~~~~~~~~~

Small-angle scattering tensor tomography of trabecular bone, Zenodo 10074598
[saxstt_bone]_. The fused operator carries every spherical-harmonic channel
through one lattice traversal.

.. figure:: _static/real_tensor_orientation.png
   :width: 62%

   Degree of orientation: 0 where the mineral is isotropic, 1 where it is
   fully aligned. The tensor model predicts held-out projections with
   R\ :sup:`2` = 0.964, against 0.846 for an isotropic model.

.. code-block:: python

   w = xtk.lrt_weights(ray_n)                 # (L, C) contraction weights
   y = xtk.xrt_tensor_apply(rays, knot, order, f, w)      # all C channels, one pass
   b = xtk.xrt_tensor_adjoint(rays, knot, order, r, w)

TOF-PET
~~~~~~~

Real time-of-flight lines of response from the PETRIC ``GE_DMI4_NEMA_IQ``
dataset [petric]_, a GE Discovery MI 4-ring scanner. 33.7 M lines of response,
each an explicit ray with its own time-of-flight offset.

.. figure:: _static/real_pet_tof_nema.png

   The NEMA phantom outline, the hot spheres and the cold insert.

.. code-block:: python

   # every LOR is a ray; TOF localises the emission along it
   rays = (Array3f(*end1.T), Array3f(*(end2 - end1).T))
   tof = xtk.TOFSpec(center=offset_mm, sigma=sigma_mm)
   y = xtk.xrt_apply(rays, knot, 0, f, tof=tof)

Synthetic examples
------------------

These come from ``doc/make_figures.py`` and rebuild in one command.

Cone-beam FDK
~~~~~~~~~~~~~

.. figure:: _static/gallery_cone.png

   A 192\ :sup:`3` phantom, 720 views, three orthogonal slices.

.. code-block:: python

   sod, sdd = 8.0 * N, 10.0 * N
   rays = xtk.cone_beam(sod=sod, sdd=sdd,
                        angles=dr.linspace(Float, 0, 2*np.pi, 720, endpoint=False),
                        detector_spec=xtk.DetectorSpec(size=(1.6*N, 1.6*N),
                                                       num_cell=(384, 384)))
   y = xtk.xrt_apply(xtk.struct_rays(rays), knot, 0, Float(phantom.reshape(-1)))
   rec = xtk.fbp_cone(rays, knot, y, sod=sod, sdd=sdd, window="shepp-logan")

Fitting the geometry
~~~~~~~~~~~~~~~~~~~~

.. figure:: _static/gallery_calibration.png

   An unknown detector shift, recovered by gradient descent. True 1.5 voxels,
   fitted 1.5000.

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
~~~~~~~~~

.. figure:: _static/gallery_sparse.png

   Twenty projections. Phantom, ``fbp`` with a Hann window, and ``cg`` after
   40 iterations.

.. code-block:: python

   rays = xtk.parallel_beam(dr.linspace(Float, 0, np.pi, 20, endpoint=False), det)
   re = xtk.struct_rays(rays)
   y = xtk.xrt_apply(re, knot, 0, f)

   a = xtk.fbp(rays, knot, y, window="hann")
   b = xtk.cg(lambda v: xtk.xrt_apply(re, knot, 0, v),
              lambda v: xtk.xrt_adjoint(re, knot, 0, v), y, N*N, n_iter=40)

Basis functions
~~~~~~~~~~~~~~~

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
