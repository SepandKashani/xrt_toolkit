"""
This module allows interfacing with array-API-compatible libraries such as (NumPy, CuPy, PyTorch, JAX).
"""

import importlib

import array_api_compat
import drjit as dr


def asarray(x, dr_type: str) -> dr.AnyArray:
    """
    Convert Array-API-compatible array to a DrJit array.

    Parameters
    ----------
    x: NDArray
        Array of shape (N,), (N, D) or (N, D, D).

        In DrJit terminology, the leading dimension `N` is assumed to be dynamic-length.
    dr_type: str
        Basename of the DrJit type to convert to. (Ex: Float, Array3f, Array22f)

    Returns
    -------
    y: AnyArray
        DrJit array of type `dr_type`.
    """
    # Load correct DRJIT backend: LLVM or CUDA
    #
    # device type codes come from the array API standard:
    # https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack_device__.html#array_api.array.__dlpack_device__
    dev_type, _ = x.__dlpack_device__()
    if int(dev_type) == 1:  # CPU
        drb = importlib.import_module("drjit.llvm")
    elif int(dev_type) == 2:  # CUDA
        drb = importlib.import_module("drjit.cuda")
    dr_klass = getattr(drb, dr_type)

    # Zero-copy instantiation of DRJIT array
    xp = array_api_compat.array_namespace(x)
    if x.ndim == 1:
        _x = x
    elif x.ndim == 2:
        _x = xp.permute_dims(x, (1, 0))
    elif x.ndim == 3:
        assert x.shape[1] == x.shape[2]
        _x = xp.permute_dims(x, (1, 2, 0))
    else:
        raise ValueError("Unsupported input.")
    y = dr_klass(_x)

    return y
