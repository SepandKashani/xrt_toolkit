import enum
import importlib.util
import types

import numpy as np
import numpy.typing as npt

__all__ = [
    "CUPY_ENABLED",
    "NDArrayInfo",
    "to_NUMPY",
]


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


def to_NUMPY(x: npt.ArrayLike) -> np.ndarray:
    """
    Convert an array from a specific backend to NUMPY.

    Parameters
    ----------
    x: NDArray
        Array to be converted.

    Returns
    -------
    y: NDArray
        Array with NumPy backend.

    Notes
    -----
    This function is a no-op if the array is already a NumPy array.
    """
    ndi = NDArrayInfo.from_obj(x)
    if ndi == NDArrayInfo.NUMPY:
        y = x
    elif ndi == NDArrayInfo.CUPY:
        y = x.get()
    else:
        msg = f"Dev-action required: define behaviour for {ndi}."
        raise ValueError(msg)
    return y


def _cp_vectorize(
    func: callable,
    dim_shape: tuple[int],
    codim_shape: tuple[int],
    out_dtype: np.dtype,
) -> callable:
    # Vectorize a cupy function func(*args) where args[0] contains batch dimensions.
    #
    # This is a small hack to enable batch processing until cupy.vectorize() supports the `signature` parameter.
    #
    # Parameters
    #   dim_shape: tuple[int]
    #     (N1,...,ND) shape of core input dimensions.
    #   codim_shape: tuple[int]
    #     (M1,...,MK) shape of core output dimensions.
    #   out_dtype: dtype
    #     dtype of the output array.
    # Returns
    #   vfunc: callable
    #     the vectorized function

    def wrapper(*args):
        x = args[0]

        ndi = NDArrayInfo.from_obj(x)
        assert ndi == NDArrayInfo.CUPY
        xp = ndi.module()

        dim_rank = len(dim_shape)
        sh_stack = x.shape[:-dim_rank]
        y = xp.zeros((*sh_stack, *codim_shape), dtype=out_dtype)
        for idx in xp.ndindex(sh_stack):
            y[idx] = func(x[idx], *args[1:])

        return y

    return wrapper
