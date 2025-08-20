"""
This module allows interfacing with array-API-compatible libraries such as (NumPy, CuPy, PyTorch, JAX).
"""

import importlib

import array_api_compat
import drjit as dr


def asarray(x) -> dr.AnyArray:
    """
    Convert Array-API-compatible array to a DrJit array.

    Parameters
    ----------
    x: NDArray
        float[16,32,64] array of shape (N,), (N, D) or (N, D, D).

        In DrJit terminology, the leading dimension `N` is assumed to be dynamic-length.

        `D` must be in (2, 3, 4).

    Returns
    -------
    y: AnyArray
        DrJit array of type:
        - (N,) -> Float{precision}
        - (N, D) -> ArrayDf{precision}
        - (N, D, D) -> ArrayDDf{precision}

    Notes
    -----
    The conversion is typically zero-copy via DLpack when possible.
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

    # Determine correct DRJIT type
    xp = array_api_compat.array_namespace(x)
    finfo = xp.finfo(x.dtype)
    assert finfo.bits in (16, 32, 64)
    suffix = {16: "16", 32: "", 64: "64"}[finfo.bits]

    assert x.ndim in (1, 2, 3)
    if x.ndim == 1:
        type_t = f"Float{suffix}"
    else:
        D = x.shape[1]
        assert D in (2, 3, 4)

        if x.ndim == 2:
            type_t = f"Array{D}f{suffix}"
        elif x.ndim == 3:
            assert x.shape[2] == D
            type_t = f"Array{D}{D}f{suffix}"
    type_t = getattr(drb, type_t)

    # Zero-copy instantiation of DRJIT array
    if x.ndim == 1:
        _x = x
    elif x.ndim == 2:
        _x = xp.permute_dims(x, (1, 0))
    elif x.ndim == 3:
        _x = xp.permute_dims(x, (1, 2, 0))
    y = type_t(_x)

    return y
