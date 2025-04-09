XRT Toolkit
===========

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT


XRT Toolkit (XTK) is a collection of utilities to compute X-Ray Transforms.


Installation
------------

.. code-block:: bash

   # user install (CPU-only)
   pip install xrt_toolkit@git+https://github.com/SepandKashani/xrt_toolkit.git

   # user install (CPU/GPU)
   pip install xrt_toolkit[gpu]@git+https://github.com/SepandKashani/xrt_toolkit.git

   # user install (CPU/GPU, with visualization tools)
   pip install xrt_toolkit[all]@git+https://github.com/SepandKashani/xrt_toolkit.git

   # developer install
   git clone https://github.com/SepandKashani/xrt_toolkit.git
   cd xrt_toolkit/
   pip install -e ".[all,dev]"
   pre-commit install
