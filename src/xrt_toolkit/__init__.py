import importlib.metadata

# __version__ = importlib.metadata.version("xrt_toolkit")

from .diagnostics import (
    plot_2d_basis as plot_2d_basis,
    plot_rays as plot_rays,
)
from .drjit.geometry import (
    cone_beam as cone_beam,
    parallel_beam as parallel_beam,
    struct_rays as struct_rays,
)
from .drjit.ray_xrt import (
    xrt_adjoint as xrt_adjoint,
    xrt_apply as xrt_apply,
)

# Geometry derivatives (2D: orders 0-2; 3D: order 0).
from .drjit.ray_xrt_new import (
    xrt_ad_t_x as xrt_ad_t_x,
    xrt_ad_t_y as xrt_ad_t_y,
    xrt_ad_t_z as xrt_ad_t_z,
    xrt_ad_n_x as xrt_ad_n_x,
    xrt_ad_n_y as xrt_ad_n_y,
    xrt_ad_n_z as xrt_ad_n_z,
)
from .drjit.struct_xrt import (
    xrt_struct_adjoint as xrt_struct_adjoint,
    xrt_struct_apply as xrt_struct_apply,
)

# Tensor tomography: fused multi-channel operators (one traversal for all
# channels) plus contraction-weight builders and layout helpers.
from .drjit.tensor_xrt import (
    xrt_tensor_apply as xrt_tensor_apply,
    xrt_tensor_adjoint as xrt_tensor_adjoint,
    doppler_weights as doppler_weights,
    lrt_weights as lrt_weights,
    trt_weights as trt_weights,
    pack_channels as pack_channels,
    unpack_channels as unpack_channels,
)

# Refractive (curved-ray) transforms: bent-ray travel-time tomography with a
# Fermat-matched adjoint (ultrasound / seismic geometrical-acoustics limit).
from .drjit.curved_xrt import (
    refract_apply as refract_apply,
    refract_adjoint as refract_adjoint,
    refract_time as refract_time,
)
from .util import (
    UniformSpec as UniformSpec,
    DetectorSpec as DetectorSpec,
    TOFSpec as TOFSpec,
    asarray as asarray,
)
from .interop import (
    from_astra as from_astra,
)
from .optim import (
    cg as cg,
    gd as gd,
    fbp as fbp,
    fbp_cone as fbp_cone,
    bpf as bpf,
)
