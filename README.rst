XRT Toolkit
===========

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT


XRT Toolkit (XTK) is a collection of utilities to compute X-Ray Transforms.


Installation
------------

.. code-block:: bash

   # user install
   pip install xrt_toolkit@git+https://github.com/SepandKashani/xrt_toolkit.git          # CPU-only
   pip install xrt_toolkit[cuda11]@git+https://github.com/SepandKashani/xrt_toolkit.git  # CPU + CUDA 11
   pip install xrt_toolkit[cuda12]@git+https://github.com/SepandKashani/xrt_toolkit.git  # CPU + CUDA 12

   # developer install
   git clone https://github.com/SepandKashani/xrt_toolkit.git
   cd xrt_toolkit/
   pip install -e ".[dev]"  # add cuda[11,12] targets too if needed
