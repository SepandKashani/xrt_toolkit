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
- ``xrt_toolkit.optim`` filters on the GPU through CuPy (cuFFT). If CuPy is
  missing or built against a different CUDA version than the driver, filtering
  falls back to NumPy on the host and warns; install a CuPy matching your CUDA
  (``cupy-cuda11x`` / ``cupy-cuda12x``) for the GPU path.
- The fast path of the 3D spline basis (``order`` > 0 in 3D) evaluates a small
  network with cooperative vectors, which need a Turing-or-newer GPU and
  driver R570+. Where unavailable, the toolkit automatically falls back to an
  equivalent plain fp32 evaluation that runs on any supported GPU (identical
  results to fp16 accuracy, slower). Set ``XRT_TOOLKIT_NO_COOPVEC=1`` to force
  the fallback.

Installation
------------

Requires Python >= 3.10, < 3.14.

.. code-block:: bash

   pip install "xrt_toolkit@git+https://github.com/SepandKashani/xrt_toolkit.git@v2"

   # developer install
   git clone --branch v2 https://github.com/SepandKashani/xrt_toolkit.git
   cd xrt_toolkit/
   pip install -e ".[viz,dev]"

   # pre-Turing GPUs (compute capability < 7.5, e.g. V100) only:
   # pip resolves Dr.Jit 1.4+, which hangs at import on such GPUs instead of
   # reporting an error. Downgrade after either install above.
   pip install "drjit<1.4"

Getting started
---------------

``tutorial_2D.ipynb`` covers the forward and adjoint operators, arbitrary and
structured geometries, reconstruction with ``xrt_toolkit.optim`` and
differentiation with respect to the acquisition geometry;
``tutorial_3D.ipynb`` covers parallel (cylinder) and cone-beam scans in 3D
(FDK, BPF, CG). Problem sizes and basis order are variables at the top of
each notebook.

Tests
-----

.. code-block:: bash

   pytest src/xrt_toolkit_tests
