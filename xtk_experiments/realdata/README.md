# Real-data reconstructions across modalities

Every reconstruction in this folder is computed by XTK from **real measured
data** downloaded from a public archive. No simulated data is used anywhere
here. Figures are in [`figs/`](figs/), reconstructed arrays in `results/`.

| Modality | Figure | Data source | Result |
|---|---|---|---|
| Cone-beam CT | `fig_ct_walnut_conebeam.png` | FIPS walnut CBCT, [Zenodo 6986012](https://zenodo.org/records/6986012) (CC-BY-4.0) | shell, kernel lobes and septum resolved; data RMSE **0.0152**; walnut 33×30×42 mm |
| Tensor tomography (SAXS-TT) | `fig_tensor_saxstt_bone.png` | trabecular bone SAXS-TT, [Zenodo 10074598](https://zenodo.org/records/10074598) | held-out R² = **0.964** (tensor) vs 0.846 (isotropic) |
| Bent-ray crosshole GPR | `fig_gpr_crosshole_bentray.png`, `fig_gpr_vs_pygimli_reference.png` | Arrenæs, Looms et al., *Geophysics* 75(6) J29–J41 (2010), via [SIPPI](https://github.com/cultpenguin/sippi) | traveltime RMSE **2.53 → 0.61 ns** (pick σ = 0.8 ns); **structure independently confirmed by pyGIMLi and by the published Bayesian posterior** |
| Cryo-electron tomography | `fig_cryoet_vibrio.png` | [CZ CryoET Portal 10489](https://cryoetdataportal.czscience.com/datasets/10489), whole *Vibrio cholerae* | cell envelope + both polyphosphate granules + appendage; **matches the depositors' own tomogram** |
| TOF-PET | `fig_pet_tof_nema.png` | [PETRIC](https://github.com/SyneRBI/PETRIC) `GE_DMI4_NEMA_IQ`, GE Discovery MI 4-ring, CC-BY-4.0 | NEMA phantom outline, hot spheres and cold insert recovered from 33.7 M real TOF LORs; noisier than the BSREM reference (subset, unregularised) |
| Seismic refraction (bent ray) | `fig_seismic_koenigsee_refraction.png` | Koenigsee, [pyGIMLi example-data](https://github.com/gimli-org/example-data) | RMSE 2.16 → 2.06 ms — **weak result, see caveat** |

## Scripts

| Script | Does |
|---|---|
| `ct_prep.py` → `ct_recon.py` | flat-field/log the walnut projections, then CGLS with explicit per-pixel cone-beam rays (14.9 M rays) |
| `gpr_invert.py` (+ `ray_link.py`) | crosshole GPR bent-ray tomography with two-point ray linking |
| `cryo_recon.py` | single-axis tilt-series CGLS reconstruction |
| `seis_common.py` → `seis_invert.py` | surface-refraction fan-shooting tomography |

`ray_link.py` is the two-point ray linker: it shoots all candidate rays in
parallel through XTK's refractive marcher and runs a secant iteration on the
signed miss distance until each ray lands on its receiver. It returns the
launch direction and arc length, which is what `refract_adjoint(..., smax=...)`
needs to backproject a residual along the very same path.

## Data

Small real inputs are committed in `data/` so the two geophysics cases run with
no download:

* `AM13_data.eas` — 702 picked crosshole GPR traveltimes.
  **Two traps:** the column header in the file is wrong (the six columns are
  really `Sx, Sz, Rx, Rz, traveltime, sigma`), and the times are in
  **nanoseconds** despite a "ms" label in the docs.
* `koenigsee.sgt` — 63 sensors, 714 picked refraction first arrivals.
* `IS002_291013_011.tlt` — 31 refined cryo-ET tilt angles.
* `walnut_meta.txt` — the CBCT scan geometry (SOD 210.66 mm, SDD 553.74 mm,
  721 views over 360°, 0.05 mm pixels).

The bulk downloads are **not** committed (see `.gitignore`):

* walnut projections — 4.0 GB, 8 zips from Zenodo 6986012 (`ct_prep.py` reads
  them from a `walnut/` subfolder).
* cryo-ET stack — `data/IS002_291013_011.mrcs`, 883 MB:
  `https://ftp.ebi.ac.uk/empiar/world_availability/10045/data/ribosomes/Tomograms/11/IS002_291013_011.mrcs`
  (EBI is slow, ~1 MB/s — use a resumable HTTP Range download).

## Independent validation of the bent-ray result

The crosshole GPR reconstruction is the one bent-ray case with an external check,
and it passes. `reference/` holds an inversion of the **same 702 picks** by
**pyGIMLi 1.6.0** (Dijkstra bent-ray forward, Gauss-Newton, λ chosen by the
discrepancy principle at χ² = 1.00) — an established, independent package, run in
its own conda env (`run_am13_pygimli.py` is the exact script).

| | XTK (this work) | pyGIMLi reference | published SIPPI posterior |
|---|---|---|---|
| layering | slow over fast | slow over fast | slow over fast |
| boundary depth | ~8 m | **7.9 m** | **≈8 m** |
| v above boundary | ~0.135 m/ns | ~0.136 m/ns | 0.125–0.135 m/ns |
| v below boundary | ~0.155 m/ns | ~0.155 m/ns | 0.15–0.16 m/ns |
| full range | 0.124–0.169 | 0.132–0.159 (χ²=1) / 0.126–0.163 (fit-matched) | 0.11–0.18 colour scale |
| traveltime RMSE | 0.61 ns | 0.80 ns (χ²=1) / 0.601 ns (fit-matched) | — |

All three agree on layering, boundary depth and velocity range. The third column
is the Bayesian posterior mean published for this exact dataset — viewable at
[SIPPI's AM13 example](https://sippi.readthedocs.io/en/latest/chapExamples/exGPR/am13-simple.html).

### Is the XTK forward model correct? Yes — checked directly

The two inversions *look* different (XTK streaky, pyGIMLi smooth), which raises
the fair question of whether the XTK bent-ray forward model is simply wrong.
It is not, and the test does not involve comparing inversions at all: run the
XTK forward on **pyGIMLi's own velocity model** and compare predicted travel
times against the same 702 picks.

| forward operator, evaluated on pyGIMLi's model | RMS misfit |
|---|---|
| pyGIMLi's own reported misfit | 0.801 ns |
| **XTK bent-ray** | **0.811 ns** |
| XTK straight-ray | 0.808 ns |

The two independent bent-ray forwards agree to **1.2 %**. Note also that bent and
straight predictions differ by a median of only **0.079 ns** on that model —
one tenth of the 0.8 ns pick noise — which is the quantitative reason bending is
irrelevant for this dataset.

The image difference was therefore **regularisation, not physics**: the original
XTK run stops at 0.63 ns, *below* the 0.8 ns pick noise, i.e. it overfits, and
overfitted noise projects into ray-path streaks. Re-running with explicit model
smoothing and a damped step so that it stops at χ² ≈ 1 reproduces the reference:

| | velocity range (m/ns) | small-scale roughness | RMSE |
|---|---|---|---|
| XTK under-regularised | 0.1242–0.1694 | 0.00057 | 0.626 ns |
| **XTK regularisation-matched** | **0.1316–0.1593** | **0.00016** | 0.768 ns |
| pyGIMLi reference | 0.1318–0.1593 | 0.00016 | 0.801 ns |

Velocity range agrees to ~0.0002 m/ns and the roughness statistic is identical.
See `figs/fig_gpr_regularisation_matched.png` (`results/gpr_regularised.npz`).

One residual difference, stated plainly: even matched, the XTK model retains
somewhat more *lateral* variation than pyGIMLi's, whose Laplacian smoothness
operator drives its solution close to purely 1-D layering. The roughness metric
used above is sensitive to fine-scale texture, not to that mid-scale lateral
structure, so it does not capture this. The depth profiles nonetheless track each
other closely.

Two further notes: (i) the lateral position of the *fastest* patch differs
between all three models, which is expected — bottom-corner lateral detail is
poorly constrained by crosshole ray coverage (published posterior std
~0.01–0.014 m/ns there), so it should not be over-interpreted.

## Methods, experiment by experiment

Common notation: the image is $f(\mathbf x)=\sum_{\mathbf k} c_{\mathbf k}\,
\varphi\!\big((\mathbf x-\mathbf x_0)/\boldsymbol\Delta-\mathbf k\big)$ on a
uniform lattice (`UniformSpec`), and $\mathbf A$ is the discretised forward
operator, so $(\mathbf A\mathbf c)_i$ is measurement $i$. All operators run
`order=0` (voxel/pixel basis) here. Two conventions matter everywhere:

* **Basis normalisation.** XTK's generator has *unit integral*, so
  `xrt_apply` returns $\sum_k c_k\,\ell_{ik}/\prod_d\Delta_d$ where
  $\ell_{ik}$ is the chord of ray $i$ in cell $k$ **in physical units**.
  Multiply by $\prod_d \Delta_d$ to obtain true line integrals. (Checked: a
  10 mm-radius ball gives 19.98 mm against an expected 20 mm chord.)
* **Adjoint.** `xrt_adjoint` is the exact transpose $\mathbf A^{\!\top}$ of
  `xrt_apply`, not a voxel-driven interpolator, so it may be used inside any
  Krylov or fixed-point iteration without a mismatch penalty.

### 1. Cone-beam CT — `ct_prep.py`, `ct_recon.py`

*Model.* $\mathbf A$ is the X-ray transform along cone-beam rays,
one ray per detector pixel per view (14.9 M rays built **explicitly** as
`(anchor, direction)` pairs — the arbitrary-ray path, not a structured
helper). Source at $R(\alpha)(-\mathrm{SOD},0,0)$, ray direction
$R(\alpha)(\mathrm{SDD},u_1,u_2)$ normalised to unit length, with
$\alpha\mapsto-\alpha$ because the *sample* rotates (see caveats).
Data are $g_i=-\log(I_i/I_0)$, i.e. Beer–Lambert; the unknown is $\mu$ [1/mm].

*Discretisation.* $192^3$ voxels at 0.2219 mm; isocentre FOV
$=N_{\rm col}\,\delta_{\rm det}/M$ with $M=\mathrm{SDD}/\mathrm{SOD}$.

*Solver.* **CGLS** on $\min_c\|\mathbf A\mathbf c-\mathbf g\|_2^2+\lambda\|\mathbf c\|_2^2$,
30 iterations, restricted to a cylindrical support $\mathbf M$ (the circular
trajectory only constrains the inscribed cylinder; without the mask the system
is underdetermined — $16.8$ M voxels vs $14.9$ M rays — and plain CG diverges
into the null space). CGLS rather than normal-equations CG because it applies
$\mathbf A$ and $\mathbf A^{\!\top}$ separately and is far better conditioned
in fp32.

### 2. Tensor tomography (SAXS-TT) — `../saxstt_realdata.ipynb`

*Model.* A $C$-channel field whose measurement contracts through the scalar
transform, $\;y_i=\sum_{c=1}^{C} w_c(\hat{\mathbf n}_i)\,(\mathbf A f_c)_i$,
with $w_c$ the real even spherical harmonics $Y_{\ell m}$, $\ell\le 6$
($C=28$), evaluated at the per-ray detector $\mathbf q$. Because the weights
are constant along each ray, `xrt_tensor_apply` evaluates all $C$ channels in
one traversal (basis weight computed once per cell, contracted against the
channels).

*Solver.* CG on the fused normal equations $\mathbf A^{\!\top}\mathbf A$.

*Validation.* **Held-out cross-validation**: fit on 4/5 of projections, predict
the unseen 1/5. $R^2=0.964$ (tensor) vs $0.846$ (isotropic); train 0.991 vs
held-out 0.964 rules out overfitting.

### 3. Bent-ray crosshole GPR — `gpr_invert.py`, `gpr_straight_vs_bent.py`, `ray_link.py`

*Model.* Eikonal/geometrical-optics limit. The unknown is slowness
$u=1/v$; the observable is first-arrival traveltime
$T=\int_{\rm ray} u\,\mathrm ds$ along the ray that solves

$$\frac{\mathrm d\mathbf x}{\mathrm ds}=\boldsymbol\tau,\qquad
\frac{\mathrm d\boldsymbol\tau}{\mathrm ds}=\frac1u\big(\nabla u-(\nabla u\!\cdot\!\boldsymbol\tau)\boldsymbol\tau\big),$$

integrated by RK2 (midpoint) with $\mathrm ds=\Delta/2$ and continuous
multilinear sampling of $u$ (piecewise-constant $u$ has no gradient and cannot
bend a ray).

*Two-point problem.* Source and receiver are both fixed, so the ray is
unknown: `ray_link.link_time` shoots all 702 candidate rays in parallel and runs
a **secant iteration on the signed lateral miss** at closest approach until each
ray lands on its receiver (9 iterations, final miss $<0.25$ m). It returns the
launch direction and arc length, which is what `refract_adjoint(..., smax=...)`
needs to backproject along the *same* path.

*Jacobian.* $T(u)$ is **nonlinear** (the path depends on $u$), but by
**Fermat's principle** the traveltime is stationary with respect to path
perturbations, so the path-variation term vanishes to first order and
$\partial T/\partial u_{\mathbf k}=\int_{\rm ray}\varphi_{\mathbf k}\,\mathrm ds$
— i.e. $\mathbf J(u_n)^{\!\top}$ is backprojection **along the current bent
rays**, re-traced each iterate. That is exactly
`refract_adjoint(coeff_geom=u_n)`.

*Solver.* Gauss–Newton-flavoured steepest descent in **log-slowness**
$m=\log u$ (slowness spans a decade, so $\partial T/\partial m=u\,\partial T/\partial u$
equalises sensitivity between the slow near-surface and the fast deep part),
with a scale-free **Cauchy step** $\alpha=\langle r,\mathbf Jg\rangle/\|\mathbf Jg\|^2$
computed from the frozen-path linearised forward, backtracking line search,
Gaussian smoothing of the gradient, and explicit smoothing of the model
(the regularisation). Stopping by the **discrepancy principle**: stop when
RMSE $\le\sigma=0.8$ ns, the quoted pick uncertainty.

### 4. Cryo-electron tomography — `cryo_vibrio.py`

*Model.* Parallel-beam projection, single-axis tilt about the image $y$ axis:
for tilt $\theta$ the beam is $\hat{\mathbf n}=(\sin\theta,0,\cos\theta)$ and the
detector axes are $(\cos\theta,0,-\sin\theta)$ and $(0,1,0)$. Tilt range
$-53^\circ\ldots+67^\circ$, so $\approx93^\circ$ of Fourier space is
**unmeasured** (the missing wedge) — the problem is intrinsically
rank-deficient, no solver can fill it.

*Solver.* **Weighted backprojection** = one application of
$\mathbf A^{\!\top}$ to ramp-filtered data. Each tilt image is convolved along
$U$ (perpendicular to the tilt axis) with the $|k|$ ramp windowed by
$\exp(-\tfrac12((r-0.35)/0.035)^2)$ for $r>0.35$ of Nyquist (IMOD `tilt`'s
`RADIAL 0.35 0.035`). The ramp is the $|k|^{d-1}$ Jacobian factor of the
Fourier-slice theorem; without it $\mathbf A^{\!\top}$ alone is a $1/|k|$
low-pass and returns a blur in which no organelle is visible (verified — see
the superseded panel).

### 5. TOF-PET — `pet_tof_recon.py`

*Model.* One ray per (view, axial, tangential, timing) bin — 33.7 M explicit
LORs with endpoints from STIR, i.e. a genuinely unstructured geometry. The TOF
kernel restricts each line integral to a Gaussian window,

$$(\mathbf A_{\rm tof}f)_i=\int f(\mathbf t_i+\alpha\hat{\mathbf n}_i)\,
g_{\sigma}(\alpha-\mu_i)\,\mathrm d\alpha,$$

with $\sigma=23.89$ mm (375.4 ps FWHM) and $\mu_i=\tfrac12|\mathbf p_2-\mathbf p_1|+\delta_i$
the arc length from the anchor $\mathbf p_1$ ($\delta_i$ = the bin's TOF offset
from the LOR midpoint). Passed as `TOFSpec(center=mu, sigma=sigma)`.
Measurement model (PETRIC eq. 5)

$$\text{prompts}\sim\mathrm{Poisson}\big(m_i\,[(\mathbf A_{\rm tof}\mathbf c)_i+a_i]\big),$$

$m$ = normalisation/attenuation, $a$ = scatter+randoms **inside** the bracket.

*Solver.* **MLEM**, the Poisson maximum-likelihood fixed point

$$c^{(n+1)}=c^{(n)}\odot\frac{\mathbf A^{\!\top}\big[\text{prompts}/(\mathbf A c^{(n)}+a)\big]}{\mathbf A^{\!\top} m},$$

15 iterations. Least squares is **not** applicable: 96.5 % of bins are zero
(mean 0.039 counts/bin), and CGLS returns noise (correlation 0.15 with the
reference) whereas MLEM is monotone in the likelihood and exceeds the
reference's likelihood.

*Operator scale.* STIR's $\mathbf A$ measures path length in units of the
transaxial voxel size and integrates the TOF Gaussian over a 25.3 mm bin,
whereas XTK applies a point-normalised Gaussian per mm. Since $a$ is in STIR's
units it must be added to a correctly scaled $\mathbf A c$; the scale is
calibrated by requiring the model to reproduce the measured total counts at the
reference image (156.6, against an analytic estimate of 154.3 — a 1.5 %
agreement that independently corroborates both conventions).

### 6. Seismic refraction — `seis_common.py`, `seis_invert.py`

Same eikonal model as §3, but sources and receivers are all at the free
surface, so the first arrival is a *diving* wave. Rather than two-point
linking, a **fan** of launch angles is shot per shot and the first-arrival
curve is the **lower envelope** of (emergence position, traveltime) over the
fan — the Lagrangian analogue of the causal sweep an eikonal solver performs.
See the caveats for why this dataset is the weak case.

## Honest caveats

* **TOF-PET: the count level, not the operator, sets the image quality.** The
  extracted LOR set is segment 0 only (71 of 205 axial positions) and every 4th
  view (68 of 272, literally OSEM subset 0-of-4), which leaves **1.30e6 prompt
  counts for 2.46e6 support voxels — 0.53 counts per voxel**, i.e. fewer counts
  than unknowns. Unregularised MLEM on that is dominated by Poisson noise, and
  measurably so: correlation with the reference *decreases* with iteration
  (0.678 at it 2, 0.445 at it 8, 0.257 at it 24). A post-reconstruction Gaussian
  filter — standard in every clinical PET pipeline — restores it, with a clear
  optimum at **8 iterations + 8 mm FWHM → correlation 0.847**, relative
  RMSE 36 %. The line profile in the figure shows the hot spheres at the correct
  positions and matching background, with reduced peak amplitude (1.0 vs 2.2 for
  the largest sphere): that is the expected resolution/noise trade-off of an
  8 mm filter, not a geometry error. The PETRIC reference is **BSREM on the full
  data**, so it is both better-regularised and ~10x better fed; this panel
  demonstrates that real TOF LORs reconstruct correctly, not image-quality parity.
  Evidence that the *model* is right, independent of the picture:
  (i) forward-projecting the reference correlates best with the data for the TOF
  sign as stored (+0.231) versus flipped (+0.199) versus no TOF (+0.111);
  (ii) the Poisson log-likelihood at the reference image (-998 371) beats the
  zero image (-1 139 610); (iii) the empirically calibrated operator scale
  (156.6) matches the analytic value binwidth*prod(step)/2.206 = 154.3 to 1.5 %;
  (iv) MLEM is monotone in the likelihood.
  *Bug worth remembering:* `dr.select(cond, P/ybar, 0)` evaluates **both**
  branches, so the division poisoned the update with `inf` and MLEM became
  non-monotone. Use `P / dr.maximum(ybar, eps)`.
* **Cryo-ET (now verified against a reference).** The panel uses CZ CryoET Data
  Portal dataset 10489, run `Vibrio_pilT_pilU_131` — whole *Vibrio cholerae*
  cells (MotorBench, doi:10.1101/2025.04.23.650258). 1023×1440×41, 13.328 Å/px,
  tilts −53.24°…+66.62°. Only 241 MB, and the stack is **already aligned**, so no
  alignment or CTF work is done here.
  *Trap:* the shipped `.xf` is **not** identity but its transform is already
  baked into the pixels — applying it again corrupts the reconstruction. Only the
  `.tlt` is used.
  *Solver:* **weighted backprojection** — ramp filter (IMOD `tilt` default
  `RADIAL 0.35 0.035`) along the direction perpendicular to the tilt axis, then a
  single adjoint pass. The ramp is essential; an unfiltered backprojection, and
  equally a dozen CGLS iterations on this heavily underdetermined problem,
  returns only a low-frequency blur in which no internal structure is visible.
  *Validation:* the reconstruction is compared directly with the depositors' own
  tomogram (`reference/cryoet_vibrio_portal_reference.png`). The cell outline and
  aspect ratio, **both polyphosphate granules at the same positions**, the
  envelope, the appendage at upper-left and even the ice contamination at
  lower-left all agree — which confirms the single-axis tilt geometry
  (tilt axis vertical, in image *y*) that could not be confirmed from the
  earlier dataset. Displayed dense-as-dark to match the reference convention.
  *Superseded:* the earlier EMPIAR-10045 yeast-ribosome attempt is kept as
  `figs/superseded_fig_cryoet_empiar10045_provisional.png`. It was dose-limited
  and noise-dominated: held-out tilts gave RMSE ≈ 0.93 against unit-variance data
  for **all four** tilt-axis/sign conventions, and a slab-confinement test gave
  42–43 % versus 40 % for a uniform smear — i.e. the geometry was simply not
  identifiable from that data. Recorded because the negative result is the reason
  a higher-contrast specimen was needed.
* **Bent vs straight on the GPR data: essentially no difference.** With identical
  solver, regularisation, stopping rule and bounds, the straight-ray baseline
  reaches RMSE **0.629 ns** and the bent-ray **0.634 ns**, and the two velocity
  models differ by only **0.0007 m/ns** on average (max 0.004). Over 5 m paths
  with ±12 % velocity contrast, refraction is a second-order correction, so this
  dataset does **not** demonstrate a benefit from bending — it demonstrates that
  the bent-ray operator agrees with the straight-ray limit and with pyGIMLi.
  See `fig_gpr_straight_vs_bent.png`. (`gpr_straight_vs_bent.py`.)
* **Seismic refraction is the weak case.** Surface sources and receivers make
  the problem strongly 1D-dominated, and a finite grid cannot resolve turning
  depths below one cell, so the fan has a minimum reachable offset (~1.7 m at
  0.1 m cells). A 1D analytic gradient fit already explains the data to 2.10 ms
  and the 2D inversion only reaches 2.06 ms — i.e. this dataset does not
  support much lateral structure at its ~2 ms picking noise. The crosshole GPR
  case is the one to look at for bent-ray performance: rays genuinely cross the
  domain and the residual drops by 4×.
* **GPR is electromagnetic, not seismic** — identical bent-ray first-arrival
  physics, different wave. For real crosshole *seismic* picks, Mont Terri
  ([Zenodo 11097797](https://zenodo.org/records/11097797), CC-BY) has them in
  the same geometry plus 10 time-lapse epochs.
* **CT rotation handedness (a real bug, found and fixed).** The sample rotates
  on the stage while source and detector stay fixed, so *in the sample frame*
  the source travels the opposite way: `sense = -1` in `build_rays`. This is
  not a harmless mirror — negating α is equivalent to mirroring the object
  **and** flipping the detector column axis, so the two are distinguishable
  from the data. Getting it wrong doubled the shell edges and left streaks:
  data RMSE **0.0344 (wrong) → 0.0152 (correct)**, held-out view RMSE 0.070 →
  0.046. See `figs/fig_ct_walnut_sign_comparison.png`.
* **CT magnification: nominal values kept, deliberately.** A held-out-view
  sweep prefers SOD ≈ 180 mm (M ≈ 3.08) over the nominal 210.66 mm (M ≈ 2.63).
  That is a 17 % discrepancy, implausible for a calibrated instrument, and the
  nominal geometry reconstructs the walnut at 33 × 30 × 42 mm — physically
  right for a shelled walnut, whereas M ≈ 3.08 would shrink it by 15 %. The
  residual trend is almost certainly absorbing uncorrected **beam hardening and
  scatter** (a smooth radial error) rather than revealing a geometry error, so
  the metadata values are used. Centre of rotation was checked independently by
  opposing-view mirror correlation and is only +0.9 unbinned pixels, i.e.
  sub-voxel, so no COR shift is applied.
* CT uses I₀ estimated from the free-beam border columns, since the dataset
  ships no flat/dark fields. Beam hardening and scatter are not corrected, so
  μ values are indicative rather than calibrated.
* Two library conventions worth remembering: XTK's basis has unit integral, so
  `xrt_apply` returns `chord / prod(step)` — multiply by `prod(step)` for true
  line integrals; and a circular cone-beam trajectory only constrains the
  inscribed cylinder, so the volume needs a support mask or CG diverges into
  the unconstrained corners.
