import importlib.metadata

__version__ = importlib.metadata.version("xrt_toolkit")

from .drjit.diagnostics import (
    diagnostic_plot as diagnostic_plot,
)
from .drjit.ray_xrt import (
    xrt_adjoint as xrt_adjoint,
    xrt_apply as xrt_apply,
)
from .drjit.struct_xrt import (
    xrt_struct_adjoint as xrt_struct_adjoint,
    xrt_struct_apply as xrt_struct_apply,
)
from .util import (
    UniformSpec as UniformSpec,
    xp2dr as xp2dr,
)
