import importlib
import inspect
import typing as typ

import drjit as dr

ArrayNfT = typ.TypeVar("ArrayNfT", bound=dr.AnyArray)
FloatT = typ.TypeVar("FloatT", bound=dr.AnyArray)
ArrayNuT = typ.TypeVar("ArrayNuT", bound=dr.AnyArray)
IntT = typ.TypeVar("IntT", bound=dr.AnyArray)
ArrayNiT = typ.TypeVar("ArrayNiT", bound=dr.AnyArray)
UIntT = typ.TypeVar("UIntT", bound=dr.AnyArray)


def float_array_t(arg: FloatT, size_v: int) -> ArrayNfT:
    """
    Converts a Dr.Jit floating-point base type into a vector type with the same precision.

    Parameters
    ----------
    arg: FloatT
        (N,) FP array, or an FP type.

    Returns
    -------
    type_t: ArrayNfT
    """
    assert dr.is_float_v(arg)
    assert size_v in (2, 3, 4)

    Float = arg if inspect.isclass(arg) else type(arg)
    nbytes = dr.itemsize_v(Float)
    assert nbytes in (2, 4, 8)
    if nbytes == 4:
        suffix = ""
    else:
        suffix = str(nbytes * 8)

    drb = importlib.import_module(arg.__module__)
    type_t = getattr(drb, f"Array{size_v}f{suffix}")
    return type_t


def int_array_t(arg: IntT, size_v: int) -> ArrayNiT:
    """
    Converts a Dr.Jit signed-integer base type into a vector type with the same precision.

    Parameters
    ----------
    arg: IntT
        (N,) signed-int array, or a signed-int type.

    Returns
    -------
    type_t: ArrayNiT
    """
    assert dr.is_integral_v(arg) and dr.is_signed_v(arg)
    assert size_v in (2, 3, 4)

    Int = arg if inspect.isclass(arg) else type(arg)
    nbytes = dr.itemsize_v(Int)
    assert nbytes in (4, 8)
    if nbytes == 4:
        suffix = ""
    else:
        suffix = str(nbytes * 8)

    drb = importlib.import_module(arg.__module__)
    type_t = getattr(drb, f"Array{size_v}i{suffix}")
    return type_t


def uint_array_t(arg: UIntT, size_v: int) -> ArrayNuT:
    """
    Converts a Dr.Jit unsigned-integer base type into a vector type with the same precision.

    Parameters
    ----------
    arg: UIntT
        (N,) unsigned-int array, or an unsigned-int type.

    Returns
    -------
    type_t: ArrayNuT
    """
    assert dr.is_integral_v(arg) and dr.is_unsigned_v(arg)
    assert size_v in (2, 3, 4)

    UInt = arg if inspect.isclass(arg) else type(arg)
    nbytes = dr.itemsize_v(UInt)
    assert nbytes in (4, 8)
    if nbytes == 4:
        suffix = ""
    else:
        suffix = str(nbytes * 8)

    drb = importlib.import_module(arg.__module__)
    type_t = getattr(drb, f"Array{size_v}u{suffix}")
    return type_t
