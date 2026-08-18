API reference
=============

Projection operators
--------------------

.. autofunction:: xrt_toolkit.xrt_apply
.. autofunction:: xrt_toolkit.xrt_adjoint
.. autofunction:: xrt_toolkit.xrt_struct_apply
.. autofunction:: xrt_toolkit.xrt_struct_adjoint

Geometry
--------

.. autofunction:: xrt_toolkit.parallel_beam
.. autofunction:: xrt_toolkit.cone_beam
.. autofunction:: xrt_toolkit.struct_rays
.. autoclass:: xrt_toolkit.UniformSpec
.. autoclass:: xrt_toolkit.DetectorSpec

Reconstruction
--------------

.. automodule:: xrt_toolkit.optim
   :members: cg, gd, fbp, fbp_cone, bpf

Geometry derivatives
--------------------

.. autofunction:: xrt_toolkit.xrt_ad_t_x
.. autofunction:: xrt_toolkit.xrt_ad_n_x

Tensor tomography
-----------------

.. autofunction:: xrt_toolkit.xrt_tensor_apply
.. autofunction:: xrt_toolkit.xrt_tensor_adjoint
.. autofunction:: xrt_toolkit.lrt_weights

Interoperability
----------------

.. autofunction:: xrt_toolkit.from_astra
.. autofunction:: xrt_toolkit.interop.vol_astra_to_xtk
.. autofunction:: xrt_toolkit.interop.vol_xtk_to_astra

.. automodule:: xrt_toolkit.torch
   :members: XRTProjector, xrt_torch

Diagnostics
-----------

.. autofunction:: xrt_toolkit.plot_rays
