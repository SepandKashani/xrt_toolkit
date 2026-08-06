XRT Toolkit
===========

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT


XRT Toolkit (XTK) is a collection of utilities to compute X-Ray Transforms,
GPU-compiled via `Dr.Jit <https://drjit.readthedocs.io>`_.

Features
--------

- Forward projection and exact matched adjoint for arbitrary sets of rays (2D/3D): ``xrt_apply``, ``xrt_adjoint``.
- Basis functions beyond voxels (``order`` = 0, 1, 2 box splines).
- Analytic derivatives with respect to the ray geometry (``xrt_ad_t_*``, ``xrt_ad_n_*``) for calibration and acquisition optimization.
- Fused multi-channel tensor tomography (``xrt_tensor_apply``, ``xrt_tensor_adjoint``): all channels in a single lattice traversal, with contraction-weight builders for the longitudinal/transverse ray transforms and Doppler tomography.
- Refractive (curved-ray) travel-time tomography with a Fermat-matched adjoint: ``refract_apply``, ``refract_adjoint``, ``refract_time``.
- Time-of-flight weighting for PET (``tof=TOFSpec(center, sigma)``).
- Structured geometries (``parallel_beam``, ``cone_beam``) and ASTRA migration (``from_astra``).
- Diagnostics: ``plot_rays``, ``plot_2d_basis``.

A CUDA GPU is required.

Compatibility
-------------

- Dr.Jit 1.4+ requires an NVIDIA GPU of compute capability 7.5 or newer
  (Turing and later) and driver R535+. On older GPUs (e.g. V100/Volta),
  install ``drjit<1.4``; the toolkit supports both.
- The fast path of the 3D spline basis (``order`` > 0 in 3D) evaluates a small
  network with cooperative vectors, which need a Turing-or-newer GPU and
  driver R570+. Where unavailable, the toolkit automatically falls back to an
  equivalent plain fp32 evaluation that runs on any supported GPU (identical
  results to fp16 accuracy, slower). Set ``XRT_TOOLKIT_NO_COOPVEC=1`` to force
  the fallback.

Installation
------------

.. code-block:: bash

   pip install "xrt_toolkit@git+https://github.com/SepandKashani/xrt_toolkit.git@v2"

   # developer install
   git clone --branch v2 https://github.com/SepandKashani/xrt_toolkit.git
   cd xrt_toolkit/
   pip install -e ".[viz,dev]"

Getting started
---------------

``tutorial.ipynb`` runs small 2D and 3D reconstructions with the different
basis orders and demonstrates differentiation with respect to the acquisition
geometry.

Tests
-----

.. code-block:: bash

   pytest src/xrt_toolkit_tests
