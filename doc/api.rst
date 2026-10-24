API Reference
=============

The XTK API operates on `Dr.Jit <https://drjit.readthedocs.io/>`_ arrays, on CPU (``drjit.llvm``) or GPU (``drjit.cuda``).
Volumes are flattened C-ordered coefficient arrays on a :py:class:`~xrt_toolkit.UniformSpec` lattice.
Rays are (anchor, direction) array pairs.
:py:func:`~xrt_toolkit.asarray` converts NumPy, CuPy, PyTorch and JAX arrays.


X-Ray Transforms
----------------

.. autofunction:: xrt_toolkit.xrt_apply

.. autofunction:: xrt_toolkit.xrt_adjoint

.. autofunction:: xrt_toolkit.xrt_grad


Structured Rays
---------------

.. autofunction:: xrt_toolkit.xrt_struct_apply

.. autofunction:: xrt_toolkit.xrt_struct_adjoint

.. autofunction:: xrt_toolkit.struct_rays

.. autofunction:: xrt_toolkit.parallel_beam

.. autofunction:: xrt_toolkit.cone_beam


Weights Along Rays
------------------

.. autoclass:: xrt_toolkit.TOFSpec

.. autoclass:: xrt_toolkit.PolynomialWeightSpec

.. autoclass:: xrt_toolkit.TabulatedWeightSpec


Reconstruction
--------------

.. autofunction:: xrt_toolkit.cg

.. autofunction:: xrt_toolkit.gd

.. autofunction:: xrt_toolkit.fbp

.. autofunction:: xrt_toolkit.fdk


PyTorch
-------

.. note::

   Requires `PyTorch <https://pytorch.org/docs/stable>`_, not installed by default.


.. autofunction:: xrt_toolkit.torch.xrt_torch

.. autoclass:: xrt_toolkit.torch.XRTProjector


Interoperability
----------------

.. autofunction:: xrt_toolkit.from_astra

.. autofunction:: xrt_toolkit.interop.vol_astra_to_xtk

.. autofunction:: xrt_toolkit.interop.vol_xtk_to_astra

.. autofunction:: xrt_toolkit.asarray


Helpers
-------

.. autoclass:: xrt_toolkit.UniformSpec
   :members: __init__, centered

.. autoclass:: xrt_toolkit.util.DetectorSpec
   :members: __init__


Diagnostics
-----------

.. note::

   Requires the ``viz`` extra: ``pip install xrt_toolkit[viz]``.


.. autofunction:: xrt_toolkit.plot_rays

.. autofunction:: xrt_toolkit.plot_2d_basis
