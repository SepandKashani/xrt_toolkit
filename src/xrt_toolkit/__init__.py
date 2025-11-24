import importlib.metadata

# __version__ = importlib.metadata.version("xrt_toolkit")

from .diagnostics import (
    plot_2d_basis as plot_2d_basis,
    plot_rays as plot_rays,
)
from .drjit.geometry import (
    cone_beam as cone_beam,
    parallel_beam as parallel_beam,
)
from .drjit.ray_xrt import (
    xrt_adjoint as xrt_adjoint,
    xrt_apply as xrt_apply,
    xrt_ad_t_x as xrt_ad_t_x,
    xrt_ad_t_y as xrt_ad_t_y,
    xrt_ad_n_x as xrt_ad_n_x,
    xrt_ad_n_y as xrt_ad_n_y,
)
from .drjit.struct_xrt import (
    xrt_struct_adjoint as xrt_struct_adjoint,
    xrt_struct_apply as xrt_struct_apply,
)
from .util import (
    UniformSpec as UniformSpec,
    DetectorSpec as DetectorSpec,
    asarray as asarray,
)
