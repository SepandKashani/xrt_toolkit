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
   pip install xrt_toolkit@git+https://github.com/SepandKashani/xrt_toolkit.git

   # with GUI diagnostic tools
   pip install xrt_toolkit[viz]@git+https://github.com/SepandKashani/xrt_toolkit.git

   # developer install
   git clone https://github.com/SepandKashani/xrt_toolkit.git
   cd xrt_toolkit/
   pip install -e ".[all]"
   pre-commit install
