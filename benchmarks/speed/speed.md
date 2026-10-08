# Speed

Master `79b1e2e` · NVIDIA A100-SXM4-80GB · Dr.Jit 1.2.0 · float32
· ASTRA 2.5.0 · TIGRE 3.1.2 · parallelproj 1.10.2

Times in **milliseconds**, median of 3–10 runs on an idle GPU. Lower is better.
Regenerate with `python benchmarks/speed/speed.py`.

- **fwd** = projection (`xrt_apply`), **adj** = back-projection (`xrt_adjoint`, exact adjoint).
- **order** 0 = voxels, 1 = linear splines, 2 = quadratic splines.
- **2D**: 1024² pixels, 1M rays. **3D**: 256³ voxels, 16.8M rays.

| geometry | rays |
|---|---|
| parallel | N views over 180°, one ray per voxel width |
| fan (2D) / cone (3D) | N views over 360°, source and detector at 2N from the centre |
| random | the parallel rays, each shifted at random by up to ±0.5 voxel and tilted by ~0.7°: arbitrary rays |

## XTK

| geometry | order | 2D fwd | 2D adj | 3D fwd | 3D adj |
|---|---|---:|---:|---:|---:|
| parallel | 0 | 5.6 | 12.3 | 19.7 | 20.8 |
|  | 1 | 10.8 | 8.7 | 64.9 | 99.6 |
|  | 2 | 14.6 | 12.3 | 192 | 230 |
| fan / cone | 0 | 4.6 | 14.4 | 24.6 | 47.5 |
|  | 1 | 11.6 | 9.7 | 76.6 | 175 |
|  | 2 | 15.0 | 13.6 | 227 | 406 |
| random | 0 | 4.1 | 19.5 | 22.6 | 94.1 |
|  | 1 | 10.1 | 16.6 | 74.2 | 296 |
|  | 2 | 13.5 | 23.5 | 210 | 693 |

`xrt_struct_apply` / `xrt_struct_adjoint` compute the rays inside the kernel and store none.
They run as fast as the same rays stored and passed to `xrt_apply` / `xrt_adjoint` (parallel beam).

| | order | struct fwd | stored fwd | struct adj | stored adj |
|---|---|---:|---:|---:|---:|
| 2D | 0 | 5.7 | 5.6 | 12.3 | 12.4 |
|  | 1 | 10.8 | 10.8 | 8.7 | 8.6 |
|  | 2 | 14.6 | 14.5 | 11.9 | 12.0 |
| 3D | 0 | 19.7 | 19.7 | 20.8 | 21.1 |
|  | 1 | 65.1 | 65.2 | 99.7 | 96.8 |
|  | 2 | 191 | 192 | 230 | 230 |

## Libraries (3D, voxels)

N views of N² rays on N³ voxels. Each library runs with its own projector pair:

- **XTK**: exact ray–voxel intersection (Siddon) and its exact adjoint.
- **ASTRA**: `FP3D_CUDA` / `BP3D_CUDA` on GPU-resident arrays; the back-projection is voxel-driven, not the exact adjoint.
- **TIGRE**: `Ax(..., "interpolated")` / `Atb(..., "matched")`. Its Python API takes host arrays, so its times include host ↔ GPU copies.
- **parallelproj**: Joseph projector and its exact adjoint, one line of response per ray.

Projection

| volume | geometry | XTK | ASTRA | TIGRE | parallelproj |
|---|---|---:|---:|---:|---:|
| 256³ | parallel | 19.7 | 24.0 | 102 | 63.3 |
| 256³ | cone | 24.4 | 27.1 | 120 | 50.3 |
| 1024³ | parallel | 5,096 | 4,920 | 22,339 | 15,882 |
| 1024³ | cone | 6,367 | 6,372 | 25,200 | 14,164 |

Back-projection

| volume | geometry | XTK | ASTRA | TIGRE | parallelproj |
|---|---|---:|---:|---:|---:|
| 256³ | parallel | 20.8 | 18.7 | 72.7 | 54.9 |
| 256³ | cone | 47.3 | 21.0 | 81.2 | 72.0 |
| 1024³ | parallel | 7,002 | 4,079 | 8,356 | 13,755 |
| 1024³ | cone | 15,767 | 4,523 | 9,817 | 25,664 |

## In short

- 2D: splines cost 2–3× voxels on the forward; their adjoint costs about the same as the voxel adjoint.
- 3D: order 1 costs about 3× voxels, order 2 about 10×.
- Projection: XTK is the fastest or on par with ASTRA, and 2.1–3.2× faster than parallelproj.
- Back-projection: XTK's exact adjoint is 1.1–3.5× slower than ASTRA's voxel-driven back-projection, most in cone beam, where rays converge on the same voxels; it is faster than parallelproj's exact adjoint.
- Random rays project as fast as parallel ones; their adjoint is 1.6–4.5× slower, because atomic writes benefit from rays moving in lockstep.
- Ray order matters: store rays neighbour by neighbour, as a detector produces them. The same 3D parallel rays in a shuffled order project 9× and back-project 27× slower (memory access).
