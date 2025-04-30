import enum
import importlib.util
import types

import drjit as dr
import numpy as np
import numpy.typing as npt

#: Show if CuPy-based backends are available.
CUPY_ENABLED: bool = importlib.util.find_spec("cupy") is not None
if CUPY_ENABLED:
    try:
        import cupy

        cupy.is_available()  # will fail if hardware/drivers/runtime missing
    except Exception:
        CUPY_ENABLED = False


@enum.unique
class NDArrayInfo(enum.Enum):
    """
    Supported dense array backends.
    """

    NUMPY = enum.auto()
    CUPY = enum.auto()

    def type(self) -> type:
        """Array type associated to a backend."""
        if self.name == "NUMPY":
            return np.ndarray
        elif self.name == "CUPY":
            return cupy.ndarray if CUPY_ENABLED else type(None)
        else:
            raise ValueError(f"No known array type for {self.name}.")

    @classmethod
    def from_obj(cls, obj) -> "NDArrayInfo":
        """Find array backend associated to `obj`."""
        if obj is not None:
            for ndi in cls:
                if isinstance(obj, ndi.type()):
                    return ndi
        raise ValueError(f"No known array type to match {obj}.")

    @classmethod
    def from_flag(cls, gpu: bool) -> "NDArrayInfo":
        """Find array backend suitable for in-memory CPU/GPU computing."""
        if gpu:
            return cls.CUPY
        else:
            return cls.NUMPY

    def module(self) -> types.ModuleType:
        """
        Python module associated to an array backend.
        """
        if self.name == "NUMPY":
            xp = np
        elif self.name == "CUPY":
            xp = cupy if CUPY_ENABLED else None
        else:
            raise ValueError(f"No known module(s) for {self.name}.")
        return xp


def xp2dr(x: npt.NDArray, dr_type: str) -> dr.AnyArray:
    """
    Convert a NumPy/CuPy array to a DrJit array.

    Parameters
    ----------
    x: NDArray
        NumPy/CuPy array of shape (N,), (N, D) or (N, D, D).

        In DrJit terminology, the leading dimension is assumed to be dynamic-length.
    dr_type: str
        Basename of the DrJit type to convert to. (Ex: Float, Array3f, Array22f)

    Returns
    -------
    y: AnyArray
        DrJit array of type `dr_type`.
    """
    ndi = NDArrayInfo.from_obj(x)
    if ndi == NDArrayInfo.NUMPY:
        drb = importlib.import_module("drjit.llvm")
    elif ndi == NDArrayInfo.CUPY:
        drb = importlib.import_module("drjit.cuda")
    else:
        raise ValueError

    dr_klass = getattr(drb, dr_type)
    if x.ndim == 1:
        y = dr_klass(x)
    elif x.ndim == 2:
        y = dr_klass(*x.T)
    elif x.ndim == 3:
        assert x.shape[1] == x.shape[2]
        y = dr_klass(*x.transpose(1, 2, 0))

    return y
