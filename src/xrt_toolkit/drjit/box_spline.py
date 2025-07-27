# These are for debugging only

import numpy as np


def factorial(n: int) -> int:
    assert n >= 0
    if n == 0:
        return 1
    else:
        return np.arange(1, n + 1).prod()


def A(n: int) -> np.ndarray:
    assert n >= 1

    A_prev = np.c_[1.0, -1].T  # (2, 1)
    if n == 1:
        A = A_prev
    else:
        for i in range(2, n + 1):
            A = np.zeros((2**i, i))
            A[:, 0] = np.kron([1, -1], np.ones(2 ** (i - 1)))
            A[:, 1:] = np.kron(np.c_[1, 1].T, A_prev)
            A_prev = A

    return A  # (2**n, n)


def B(n: int) -> np.ndarray:
    assert n >= 1

    B = np.r_[1.0, -1]  # (2,)
    for _ in range(2, n + 1):
        B = np.kron(B, np.r_[1, -1])

    return B  # (2**n,)


def B2(E_mask: np.ndarray) -> np.ndarray:
    n = E_mask.size
    assert E_mask.shape == (n,)

    B = np.r_[1.0]  # (1,)
    for i in range(n):
        if E_mask[i]:
            b = np.r_[1, -1]
        else:
            b = np.r_[1, 0]
        B = np.kron(B, b)

    return B  # (2**n,)


# variant 1: NumPy-style
def box_spline_1d_np(
    E: np.ndarray,  # (NE,)
    x: np.ndarray,  # (Nx,)
) -> np.ndarray:  # (Nx,)
    assert E.shape == (E.size,)  # (NE,)
    assert x.shape == (x.size,)  # (Nx,)

    E_mask = abs(E) >= 1e-3
    E = E[E_mask]  # non-zero entries only
    N_nz = E_mask.sum().item()

    if N_nz == 1:
        y = np.where(abs(x) <= E[0] / 2, 1.0, 0)
    else:
        lhs = (x.reshape(-1, 1) + (0.5 * A(N_nz) @ E)).clip(0, None) ** (N_nz - 1)
        rhs = B(N_nz)
        y = lhs @ rhs
    y /= factorial(N_nz - 1) * E.prod()

    return y


# variant 2: DrJit-style (uses masking)
def box_spline_1d_style2(
    E: np.ndarray,  # (NE,)
    x: np.ndarray,  # (Nx,)
) -> np.ndarray:  # (Nx,)
    assert E.shape == (E.size,)  # (NE,)
    assert x.shape == (x.size,)  # (Nx,)

    E_mask = abs(E) >= 1e-3
    N_tot = len(E_mask)
    N_nz = E_mask.sum().item()

    if N_nz == 1:
        y = np.where(abs(x) <= E.sum() / 2, 1.0, 0)
    else:
        lhs = (x.reshape(-1, 1) + (0.5 * A(N_tot) @ E)).clip(0, None) ** (N_nz - 1)
        rhs = B2(E_mask)
        y = lhs @ rhs
    y /= factorial(N_nz - 1) * (E + (1 - E_mask)).prod()

    return y


# DrJit-style (uses masking)
def box_spline_1d_dr():
    pass  # todo
