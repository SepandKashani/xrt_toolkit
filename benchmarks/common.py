"""
Benchmark cases shared by every library wrapper in libs/.

Cases: N views of N^(D-1) rays around an N^D volume of unit voxels centred at the origin.

- parallel: views over 180 degrees, detector pitch 1.
- cone (fan in 2D): views over 360 degrees, source at 2N from the centre, detector at 2N
  on the other side, detector pitch 2 (magnification 2).

A wrapper defines ``operators(geometry, D, N)`` and calls ``main(operators)``:

    python libs/<library>.py speed   <geometry> <D> <N> <reps>   -> {"fwd": ms, "adj": ms}
    python libs/<library>.py adjoint <geometry> <D> <N>          -> {"mismatch": eps, ...}

``operators`` returns a dict with
    fwd(x) -> y, adj(y) -> x   the library's projection and back-projection,
    x_size, y_size             number of voxels / rays,
    to_lib(np.ndarray), to_np(lib array)
    sync()                     wait for the GPU.
"""

import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

LIBRARIES = {  # name: (wrapper in libs/, interpreter command)
    "XTK": ("run_xtk.py", sys.executable),
    "ASTRA": ("run_astra.py", os.environ.get("ASTRA_PYTHON", sys.executable)),
    "TIGRE": ("run_tigre.py", os.environ.get("TIGRE_PYTHON", sys.executable)),
    "parallelproj": (
        "run_parallelproj.py",
        os.environ.get("PARALLELPROJ_PYTHON", sys.executable),
    ),
}


def run_case(library: str, *args) -> dict:
    # one case in a fresh process of the library's interpreter; None if it does not run there
    wrapper, python = LIBRARIES[library]
    cmd = [
        *shlex.split(python),
        str(Path(__file__).parent / "libs" / wrapper),
        *map(str, args),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(out.stdout.splitlines()[-1]) if out.returncode == 0 else None


def angles(geometry: str, N: int) -> np.ndarray:
    span = math.pi if geometry == "parallel" else 2 * math.pi
    return np.linspace(0, span, N, endpoint=False)


def sod(N: int) -> float:
    return 2.0 * N


def sdd(N: int) -> float:
    return 4.0 * N


def pitch(geometry: str) -> float:
    return 1.0 if geometry == "parallel" else 2.0


def speed(ops: dict, reps: int) -> dict:
    # median wall time [ms], GPU synchronised, after one warm-up call
    rng = np.random.default_rng(0)
    x = ops["to_lib"](rng.random(ops["x_size"], dtype=np.float32))
    y = ops["to_lib"](rng.random(ops["y_size"], dtype=np.float32))
    out = {}
    for name, fn, arg in (("fwd", ops["fwd"], x), ("adj", ops["adj"], y)):
        fn(arg)
        ops["sync"]()
        ts = []
        for _ in range(reps):
            ops["sync"]()
            t0 = time.perf_counter()
            fn(arg)
            ops["sync"]()
            ts.append(time.perf_counter() - t0)
        out[name] = 1e3 * float(np.median(ts))
    return out


def adjoint_mismatch(ops: dict, pairs: int = 20) -> dict:
    # ||B - A^T||_F / ||A||_F, estimated with random Gaussian pairs (x, y):
    #     E <x, (A^T - B) y>^2 = ||A^T - B||_F^2,    E <A x, y>^2 = ||A||_F^2.
    # "rescaled": the same after the best global factor c (B -> c B).
    rng = np.random.default_rng(0)
    lhs, rhs = np.zeros(pairs), np.zeros(pairs)
    for k in range(pairs):
        x = rng.standard_normal(ops["x_size"]).astype(np.float32)
        y = rng.standard_normal(ops["y_size"]).astype(np.float32)
        Ax = ops["to_np"](ops["fwd"](ops["to_lib"](x))).astype(np.float64).ravel()
        lhs[k] = Ax @ y.astype(np.float64)
        By = ops["to_np"](ops["adj"](ops["to_lib"](y))).astype(np.float64).ravel()
        rhs[k] = x.astype(np.float64) @ By
    c = (lhs @ rhs) / (rhs @ rhs)
    return {
        "mismatch": float(np.linalg.norm(lhs - rhs) / np.linalg.norm(lhs)),
        "rescaled": float(np.linalg.norm(lhs - c * rhs) / np.linalg.norm(lhs)),
        "scale": float(c),
    }


def main(operators):
    task, geometry, D, N = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    ops = operators(geometry, D, N)
    if task == "speed":
        print(json.dumps(speed(ops, int(sys.argv[5]))))
    elif task == "adjoint":
        print(json.dumps(adjoint_mismatch(ops)))
