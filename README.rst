XRT Toolkit
===========

.. |gh-badge| image:: https://img.shields.io/badge/GitHub-Repo-black?logo=github
   :alt: GitHub repository
   :target: https://github.com/SepandKashani/xrt_toolkit

.. |docs-badge| image:: https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github
   :alt: Documentation
   :target: https://SepandKashani.github.io/xrt_toolkit/

.. |license-badge| image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :alt: License: MIT
   :target: https://opensource.org/licenses/MIT

|gh-badge| |docs-badge| |license-badge|


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
