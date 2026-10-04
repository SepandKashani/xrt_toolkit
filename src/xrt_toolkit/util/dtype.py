import numpy as np


class TranslateDType:
    """
    float/complex dtype translator.
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

    def __init__(self, dtype: np.dtype):
        dtype = np.dtype(dtype)
        assert dtype in self.map_to_float
        self._fdtype = self.map_to_float[dtype]  # canonical internal representation

    def to_int(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "i")]

    def to_float(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "f")]

    def to_complex(self) -> np.dtype:
        return self.map_from_float[(self._fdtype, "c")]
