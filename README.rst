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
