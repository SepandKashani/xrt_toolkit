import collections
import collections.abc as cabc
import warnings

import numpy as np
import numpy.typing as npt

__all__ = [
    "broadcast_seq",
    "TranslateDType",
    "as_namedtuple",
    "cast_warn",
]


def broadcast_seq(
    x,
    N: int = None,
    dtype: npt.DTypeLike = None,
) -> np.ndarray:
    """
    Broadcast `x` to size `N`, optionally casting entries.

    If `N` is omitted, then no broadcasting takes place.
    """
    if isinstance(x, cabc.Iterable):
        _x = tuple(x)
    else:
        _x = (x,)

    if N is not None:
        if len(_x) == 1:
            _x *= N  # broadcast
        assert len(_x) == N

    y = np.array(_x, dtype=dtype)
    return y


class TranslateDType:
    """
    int/float/complex dtype translator.
    """

    map_to_float: dict = {
        np.dtype(np.int32): np.dtype(np.float32),
        np.dtype(np.int64): np.dtype(np.float64),
        np.dtype(np.float32): np.dtype(np.float32),
        np.dtype(np.float64): np.dtype(np.float64),
        np.dtype(np.complex64): np.dtype(np.float32),
        np.dtype(np.complex128): np.dtype(np.float64),
    }
    map_from_float: dict = {
        (np.dtype(np.float32), "i"): np.dtype(np.int32),
        (np.dtype(np.float64), "i"): np.dtype(np.int64),
        (np.dtype(np.float32), "f"): np.dtype(np.float32),
        (np.dtype(np.float64), "f"): np.dtype(np.float64),
        (np.dtype(np.float32), "c"): np.dtype(np.complex64),
        (np.dtype(np.float64), "c"): np.dtype(np.complex128),
    }

    def __init__(self, dtype: npt.DTypeLike):
        dtype = np.dtype(dtype)
        assert dtype in self.map_to_float
        self._fdtype = self.map_to_float[dtype]

    def to_int(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "i")]

    def to_float(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "f")]

    def to_complex(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "c")]


def as_namedtuple(**kwargs) -> collections.namedtuple:
    """
    Store mapping as named-tuple.
    """
    nt_t = collections.namedtuple("nt_t", kwargs.keys())
    y = nt_t(**kwargs)
    return y


def cast_warn(x: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """
    Cast `x` to `dtype` if type mis-match.

    Emit warning when cast occurs.
    """
    y = x.astype(dtype, copy=False)
    if x.dtype != y.dtype:
        msg = f"{x.shape}: {x.dtype} -> {y.dtype} cast performed."
        warnings.warn(msg)
    return y
