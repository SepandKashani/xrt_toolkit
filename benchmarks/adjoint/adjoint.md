# Adjoint mismatch

Master `79b1e2e` · NVIDIA A100-SXM4-80GB · Dr.Jit 1.2.0 · float32
· ASTRA 2.5.0 · TIGRE 3.1.2 · parallelproj 1.10.2

How far a library's back-projection **B** is from the exact adjoint **Aᵀ** of its own
projection **A**:

    mismatch = ‖B − Aᵀ‖ / ‖A‖    (Frobenius norms)

estimated from 20 random dot-product tests ⟨Ax, y⟩ vs ⟨x, By⟩. **0** means B is the exact
adjoint; float32 rounding alone gives about 1e-7 to 1e-6. Lower is better.
Regenerate with `python benchmarks/adjoint/adjoint.py`.

- **2D**: 512² pixels, 512 views. **3D**: 128³ voxels, 128 views of 128² rays.
- Geometries as in [speed.md](../speed/speed.md): parallel, fan / cone, random.

## XTK

| geometry | order | 2D | 3D |
|---|---|---:|---:|
| parallel | 0 | 7e-07 | 3e-07 |
|  | 1 | 6e-07 | 6e-07 |
|  | 2 | 1e-06 | 1e-06 |
| fan / cone | 0 | 8e-07 | 4e-07 |
|  | 1 | 8e-07 | 9e-07 |
|  | 2 | 1e-06 | 1e-06 |
| random | 0 | 6e-07 | 3e-07 |
|  | 1 | 8e-07 | 8e-07 |
|  | 2 | 1e-06 | 1e-06 |

## Libraries (voxels)

Projector pairs as in [speed.md](../speed/speed.md). In brackets: the mismatch left after
the best global rescaling of B, i.e. what is not just a scale factor.

| library | 2D parallel | 2D fan | 3D parallel | 3D cone |
|---|---:|---:|---:|---:|
| XTK | 6e-07 (5e-07) | 6e-07 (6e-07) | 3e-07 (3e-07) | 3e-07 (3e-07) |
| ASTRA | 0.14 (0.11) | 0.23 (0.22) | 0.13 (0.13) | 0.16 (0.16) |
| TIGRE | - | - | 0.06 (0.06) | 0.25 (0.21) |
| parallelproj | - | - | 3e-07 (3e-07) | 5e-07 (5e-07) |

TIGRE and parallelproj are run in 3D only.

## In short

- XTK's back-projection is the exact adjoint for every order, dimension and geometry, up to float32 rounding.
- parallelproj's is exact too.
- ASTRA's and TIGRE's back-projections differ from the exact adjoint by 6–25%, and rescaling does not remove it.
