XRT Toolkit
===========

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT


XRT Toolkit (XTK) computes X-Ray Transforms on the GPU, compiled through
`Dr.Jit <https://drjit.readthedocs.io>`_.

Features
--------

- Forward projection and its exact matched adjoint, for arbitrary sets of rays
  in 2D and 3D: ``xrt_apply``, ``xrt_adjoint``.
- Basis functions beyond voxels: box splines of order 0, 1 and 2.
- Analytic derivatives with respect to the ray geometry (``xrt_ad_t_*``,
  ``xrt_ad_n_*``), for calibration and acquisition design.
- Fused multi-channel tensor tomography (``xrt_tensor_apply``,
  ``xrt_tensor_adjoint``). All channels share one lattice traversal. Weight
  builders cover the longitudinal and transverse ray transforms, and Doppler
  tomography.
- Refractive travel-time tomography with a Fermat-matched adjoint:
  ``refract_apply``, ``refract_adjoint``, ``refract_time``.
- Time-of-flight weighting for PET: ``tof=TOFSpec(center, sigma)``.
- Structured geometries (``parallel_beam``, ``cone_beam``) and ASTRA migration
  (``from_astra``).
- Reconstruction: ``fbp``, ``fbp_cone``, ``bpf``, ``cg``, ``gd``.
- Diagnostics: ``plot_rays``, ``plot_2d_basis``.

A CUDA GPU is required.

Compatibility
-------------

- Dr.Jit 1.4 needs compute capability 7.5 or newer, so Turing or later, with
  driver R535. On older cards such as the V100, install ``drjit<1.4``. The
  toolkit supports both.
- ``xrt_toolkit.optim`` filters on the GPU through CuPy. If CuPy is missing,
  or built against a different CUDA version than the driver, filtering falls
  back to NumPy on the host and warns. Install a CuPy matching your CUDA
  (``cupy-cuda11x`` or ``cupy-cuda12x``) for the GPU path.
- In 3D, ``order`` above 0 evaluates a small network with cooperative vectors.
  Those need a Turing card and driver R570. Without them the toolkit falls
  back to plain fp32, which gives the same results to fp16 accuracy and runs
  slower. Set ``XRT_TOOLKIT_NO_COOPVEC=1`` to force the fallback.

Installation
------------

Requires Python 3.10 to 3.13.

.. code-block:: bash

   pip install "xrt_toolkit@git+https://github.com/SepandKashani/xrt_toolkit.git@v2"

   # developer install
   git clone --branch v2 https://github.com/SepandKashani/xrt_toolkit.git
   cd xrt_toolkit/
   pip install -e ".[viz,dev]"

   # pre-Turing GPUs only, such as the V100.
   # pip resolves Dr.Jit 1.4, which hangs at import on those cards instead of
   # reporting an error. Downgrade after either install above.
   pip install "drjit<1.4"

Getting started
---------------

``tutorial_2D.ipynb`` covers the operators, both geometry interfaces,
reconstruction and geometry fitting. ``tutorial_3D.ipynb`` covers parallel and
cone-beam scans in 3D. Problem sizes and the basis order sit at the top of
each notebook.

Tests
-----

.. code-block:: bash

   pytest src/xrt_toolkit_tests

Documentation
-------------

.. code-block:: bash

   pip install -e ".[doc]"
   python -m sphinx doc doc/_build/html    # open doc/_build/html/index.html

The build needs the dependencies but no GPU. Dr.Jit imports without a CUDA
device, so Read the Docs builds the site from ``.readthedocs.yaml`` unchanged.

Figures live in ``doc/_static``. Regenerate them with
``python doc/make_figures.py``, which does need a GPU.

Citing
------

See ``CITATION.cff``, or the *Cite this repository* button on GitHub.
