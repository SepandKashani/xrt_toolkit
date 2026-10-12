# Gradients

Master `7747da6` · NVIDIA A100-SXM4-80GB · Dr.Jit 1.2.0 · float32

`xrt_grad` returns, for every ray, the gradient of its projection with respect to the ray's
anchor **t** and direction **n** (all D components of each), in one walk along the ray.
Regenerate with `python benchmarks/gradient/gradient.py`.

- **order** 0 = voxels, 1 = linear splines, 2 = quadratic splines.
- Geometries as in [speed.md](../speed/speed.md): parallel, fan / cone, random.
- **finite diff.**: what one does without `xrt_grad`: central differences of the float32
  projection `xrt_apply` in every component of t and n (step 1e-3, about the best).

## Exactness

Max error, relative to the largest gradient entry, against central differences of an exact
float64 forward (independent NumPy code). Lower is better; float32 rounding alone gives about
1e-6. Volumes of odd size, so that rays pass through basis centers; random rays of each
geometry.

**Rays at kinks**: rays where the projection is not differentiable: through a lattice edge or
corner, or lying in a knot plane. Central differences give the average of both sides there.
`xrt_grad` returns that average when the kink is exact in floating point (3D parallel,
order 1: the rays lie in the knot planes), and a one-sided derivative when only rounding
separates the faces (3D cone, order 0: rays through lattice edges in exact arithmetic).

2D, 65² pixels, 200 rays

| geometry | order | xrt_grad | xrt_grad, no kinks | finite diff., no kinks | rays at kinks |
|---|---|---:|---:|---:|---:|
| parallel | 0 | 2e-06 | 2e-06 | 5e-01 | 0 |
|  | 1 | 3e-05 | 3e-05 | 1e-02 | 4 |
|  | 2 | 5e-05 | 5e-05 | 3e-03 | 0 |
| fan / cone | 0 | 9e-07 | 9e-07 | 1e-01 | 0 |
|  | 1 | 1e-04 | 1e-04 | 1e-02 | 0 |
|  | 2 | 2e-04 | 2e-04 | 2e-02 | 0 |
| random | 0 | 3e-05 | 3e-05 | 8e-01 | 0 |
|  | 1 | 4e-05 | 4e-05 | 5e-02 | 0 |
|  | 2 | 9e-05 | 9e-05 | 1e-02 | 0 |

3D, 33³ voxels, 100 rays

| geometry | order | xrt_grad | xrt_grad, no kinks | finite diff., no kinks | rays at kinks |
|---|---|---:|---:|---:|---:|
| parallel | 0 | 9e-07 | 9e-07 | 5e-01 | 0 |
|  | 1 | 5e-06 | 3e-06 | 6e-04 | 99 |
|  | 2 | 2e-06 | 2e-06 | 5e-04 | 0 |
| fan / cone | 0 | 1e-01 | 3e-07 | 4e-01 | 3 |
|  | 1 | 1e-05 | 1e-05 | 9e-03 | 4 |
|  | 2 | 1e-05 | 1e-05 | 9e-03 | 0 |
| random | 0 | 2e-05 | 2e-05 | 9e-01 | 0 |
|  | 1 | 3e-05 | 3e-05 | 1e-01 | 0 |
|  | 2 | 4e-06 | 4e-06 | 2e-03 | 0 |

## Time

Milliseconds, median of 3–10 runs. **2D**: 1024² pixels, 1M rays. **3D**: 256³ voxels,
16.8M rays.

| geometry | order | 2D fwd | 2D xrt_grad | 2D finite diff. | 3D fwd | 3D xrt_grad | 3D finite diff. |
|---|---|---:|---:|---:|---:|---:|---:|
| parallel | 0 | 5.6 | 7.3 | 42.2 | 20.6 | 50.6 | 244 |
|  | 1 | 10.8 | 16.0 | 81.1 | 70.1 | 220 | 826 |
|  | 2 | 14.5 | 24.8 | 111 | 206 | 310 | 2,405 |
| fan / cone | 0 | 4.6 | 7.1 | 34.5 | 24.5 | 62.4 | 290 |
|  | 1 | 11.4 | 17.5 | 86.6 | 80.5 | 115 | 944 |
|  | 2 | 15.0 | 27.3 | 115 | 233 | 353 | 2,724 |
| random | 0 | 3.8 | 7.0 | 28.4 | 22.9 | 58.2 | 273 |
|  | 1 | 10.1 | 17.7 | 75.5 | 73.9 | 106 | 868 |
|  | 2 | 13.7 | 28.3 | 104 | 212 | 322 | 2,476 |

## In short

- `xrt_grad` matches the exact derivatives to 1e-6–1e-4 in float32. Finite differences of
  the float32 projection are off by 1e-3 to 1e-2 for splines, and by up to 100% for voxels,
  whose projection has a kink at every face crossing.
- 2D splines, float32: the direction derivative loses accuracy within ~0.01 rad of a lattice
  axis or diagonal, where the closed form cancels (fan beam: 1e-4–2e-4).
- `xrt_grad` costs 1.3–3× a projection, for all 2D components; finite differences cost 4D
  projections: `xrt_grad` is 4–8× faster.
- 3D parallel, order 1, costs more (220 ms): its rays lie in knot planes, where `xrt_grad`
  also evaluates the cells on the other side.
