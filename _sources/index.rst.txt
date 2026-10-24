XRT Toolkit
===========

.. toctree::
   :maxdepth: 4
   :hidden:

   api
   references


.. |gh-badge| image:: https://img.shields.io/badge/GitHub-Repo-black?logo=github
   :alt: GitHub repository
   :target: https://github.com/SepandKashani/xrt_toolkit

.. |docs-badge| image:: https://img.shields.io/badge/Docs-GitHub%20Pages-blue?logo=github
   :alt: Documentation
   :target: https://SepandKashani.github.io/xrt_toolkit/

|gh-badge| |docs-badge|


XRT Toolkit (XTK) is a collection of utilities to compute X-Ray Transforms.


Rationale
---------

Tomography measures line integrals of a function :math:`f: \bR^{D} \to \bR`, :math:`D \in \{2, 3\}`:

.. math::
   :label: xrt

   \xrt[f](\bbn, \bbt) = \int_{\bR} f(\bbt + \alpha \bbn) \, d\alpha,

along rays with anchors :math:`\bbt \in \bR^{D}` and unit directions :math:`\bbn \in \bR^{D}`.
Expanding :math:`f` in shifts of a basis function :math:`\psi` on a uniform lattice

.. math::

   f(\bbx) = \sum_{\bbq} f_{\bbq} \, \psi(\bbx - \bbx_{\bbq})

makes :eq:`xrt` linear in the coefficients :math:`f_{\bbq}`:

.. math::

   \xrt[f](\bbn, \bbt) = \sum_{\bbq} f_{\bbq} \, \xrt[\psi](\bbn, \bbt - \bbx_{\bbq}).

XTK evaluates these sums along arbitrary rays, exactly, for voxels :cite:`siddon1985,amanatides1987` and for higher-order box splines and B-splines :cite:`entezari2012,haouchat2025tci,haouchat2026splinesplat`.
It computes, on CPU and GPU via Dr.Jit :cite:`jakob2022drjit`:

- the projections and their adjoint;
- their gradients with respect to the rays, for geometry calibration :cite:`haouchat2026calibration`.
