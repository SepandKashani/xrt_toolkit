Bibliography
============

The claims in :doc:`transform` rest on these works.

Discretisation
--------------

.. [siddon1985] R. L. Siddon. *Fast calculation of the exact radiological path
   for a three-dimensional CT array*. Medical Physics, 12(2), 1985. The
   chord-length traversal that ``order = 0`` computes.

Filtered backprojection
-----------------------

.. [kak1988] A. C. Kak and M. Slaney. *Principles of Computerized Tomographic
   Imaging*. IEEE Press, 1988. The discrete ramp kernel used by
   :py:func:`~xrt_toolkit.fbp`.

.. [feldkamp1984] L. A. Feldkamp, L. C. Davis and J. W. Kress. *Practical
   cone-beam algorithm*. Journal of the Optical Society of America A, 1(6),
   1984. The FDK method behind :py:func:`~xrt_toolkit.fbp_cone`.

.. [tuy1983] H. K. Tuy. *An inversion formula for cone-beam reconstruction*.
   SIAM Journal on Applied Mathematics, 43(3), 1983. Why a circular source
   orbit cannot be exact away from the mid-plane.

Compilation
-----------

.. [jakob2022drjit] W. Jakob, S. Speierer, N. Roussel and D. Vicini. *Dr.Jit: a
   just-in-time compiler for differentiable rendering*. ACM Transactions on
   Graphics, 41(4), 2022.

Other packages
--------------

.. [vanaarle2016] W. van Aarle et al. *Fast and flexible X-ray tomography
   using the ASTRA toolbox*. Optics Express, 24(22), 2016.

.. [biguri2016] A. Biguri, M. Dosanjh, S. Hancock and M. Soleimani. *TIGRE: a
   MATLAB-GPU toolbox for CBCT image reconstruction*. Biomedical Physics and
   Engineering Express, 2(5), 2016.

Tensor tomography
-----------------

.. [gao2019] Z. Gao et al. *Nanostructure tensor tomography*. Acta
   Crystallographica Section A, 75, 2019. The IRTT model and the datasets used
   in the benchmark on :doc:`performance`.
