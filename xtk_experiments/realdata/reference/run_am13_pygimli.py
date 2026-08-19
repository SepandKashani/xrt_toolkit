#!/usr/bin/env python
"""Independent reference inversion of the AM13 (Arrenaes, DK) crosshole GPR
traveltime dataset with pyGIMLi's TravelTimeManager (shortest-path/Dijkstra,
i.e. bent-ray, tomography).

UNITS: SI internally -- metres and SECONDS (t_ns * 1e-9). The inverted
velocity therefore comes out in m/s and is converted to m/ns (v * 1e-9)
for all reported numbers and saved outputs.

Data columns in AM13_data.eas (names in file header are WRONG):
    Sx, Sz, Rx, Rz, traveltime [ns], sigma [ns]
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pygimli as pg
from pygimli.physics import traveltime as tt

OUT = "/tmp/claude-275574/-scratch-haouchat/9cfc19ba-5b2c-4521-91ff-f93b41e97041/scratchpad/agent_gimli"
EAS = f"{OUT}/AM13_data.eas"

# ----------------------------------------------------------------- load data
raw = np.genfromtxt(EAS, skip_header=8, encoding="latin-1")
assert raw.shape == (702, 6), raw.shape
Sx, Sz, Rx, Rz, t_ns, sig_ns = raw.T
t_s = t_ns * 1e-9          # seconds (SI)
err_s = sig_ns * 1e-9      # absolute traveltime error, seconds (0.8 ns)

# y is elevation: y = -depth. Boreholes at x=0 (Tx) and x=5 (Rx).
pos = np.vstack([np.column_stack([Sx, -Sz]), np.column_stack([Rx, -Rz])])
upos, inv_idx = np.unique(pos, axis=0, return_inverse=True)
N = len(t_s)
s_idx = inv_idx[:N]
g_idx = inv_idx[N:]

data = pg.DataContainer()
data.registerSensorIndex("s")
data.registerSensorIndex("g")
for p in upos:
    data.createSensor(pg.Pos(float(p[0]), float(p[1])))
data.resize(N)
data["s"] = s_idx.astype(float)
data["g"] = g_idx.astype(float)
data["t"] = t_s
data["err"] = err_s
data["valid"] = np.ones(N)
print(f"DataContainer: {data.size()} picks, {data.sensorCount()} sensors")

# --------------------------------------------------------------------- mesh
# Regular quad grid covering x in [0,5], depth in [0.5,12.5] (y in [-12.5,-0.5])
dx = 0.25
xg = np.arange(0.0, 5.0 + 1e-9, dx)
yg = np.arange(-12.5, -0.5 + 1e-9, dx)
mesh = pg.createGrid(x=xg, y=yg)
print(f"Mesh: {mesh.cellCount()} cells, {mesh.nodeCount()} nodes "
      f"({len(xg)-1} x {len(yg)-1} quads of {dx} m)")

# ---------------------------------------------------------------- inversion
# Lambda scan; pick run with converged chi^2 closest to 1 (discrepancy
# principle: strongest regularisation that still fits to the 0.8 ns error).
# Note: with sigma=0.8 ns even lam=50 immediately over-fits (chi^2 ~ 0.35
# after one step), so the scan goes towards much stronger smoothing.
results = []
for lam in (2000.0, 1000.0, 500.0, 200.0, 50.0):
    mgr = tt.TravelTimeManager()
    vel = mgr.invert(data=data, mesh=mesh, secNodes=3,
                     useGradient=False,          # homogeneous start (crosshole)
                     lam=lam, zWeight=1.0,        # isotropic smoothing
                     limits=[0.8e8, 2.5e8],       # generous velocity bounds m/s
                     maxIter=20, verbose=True)
    resp = np.array(mgr.inv.response)
    chi2 = mgr.inv.chi2()
    rms_ns = np.sqrt(np.mean((resp - t_s) ** 2)) * 1e9
    niter = mgr.inv.iter
    print(f"### lam={lam}: chi2={chi2:.3f} rms={rms_ns:.3f} ns iters={niter}")
    results.append(dict(lam=lam, chi2=chi2, rms_ns=rms_ns, niter=niter,
                        vel=np.array(vel), resp=resp, mgr=mgr))

# choose: chi2 closest to 1 in log space; ties -> larger lam (already ordered)
best = min(results, key=lambda r: abs(np.log(r["chi2"])))
mgr = best["mgr"]
vel_ms = best["vel"]                    # m/s on paraDomain cells
vel_mns = vel_ms * 1e-9                 # m/ns
resp = best["resp"]
print(f"\nCHOSEN lam={best['lam']}  chi2={best['chi2']:.3f}  "
      f"rms={best['rms_ns']:.3f} ns  iters={best['niter']}")

pd = mgr.paraDomain
assert pd.cellCount() == len(vel_ms)

# starting misfit: forward response of the homogeneous start model,
# using the chosen manager's already-initialised forward operator
smodel = np.array(mgr.fop.createStartModel(pg.Vector(t_s)))
resp0 = np.array(mgr.fop.response(pg.Vector(smodel)))
rms0_ns = np.sqrt(np.mean((resp0 - t_s) ** 2)) * 1e9
v0_mns = 1.0 / smodel[0] * 1e-9
print(f"Homogeneous start model: v0={v0_mns:.4f} m/ns, start RMS={rms0_ns:.3f} ns")

# ------------------------------------------------- export to regular grid
# 0.1 m sampling, x in [0,5], z (depth, positive down) in [0.5,12.5]
xr = np.round(np.arange(0.0, 5.0 + 1e-9, 0.1), 10)
zr = np.round(np.arange(0.5, 12.5 + 1e-9, 0.1), 10)
V = np.empty((len(zr), len(xr)))
xq = np.clip(xr, 1e-6, 5 - 1e-6)
for i, z in enumerate(zr):
    yq = float(np.clip(-z, -12.5 + 1e-6, -0.5 - 1e-6))
    for j, x in enumerate(xq):
        c = pd.findCell(pg.Pos(float(x), yq))
        V[i, j] = vel_mns[c.id()]
np.savez(f"{OUT}/am13_pygimli_velocity.npz", vel=V, x=xr, z=zr)
print(f"Saved grid: vel {V.shape} (z,x), v in [{V.min():.4f},{V.max():.4f}] m/ns")

# ----------------------------------------------------- structure diagnostics
prof = V.mean(axis=1)                             # horizontal-mean v(z)
dz = np.gradient(prof, zr)
zjump = zr[np.argmax(dz)]
print("\nHorizontal-mean velocity profile (m/ns):")
for z in np.arange(1.0, 12.1, 1.0):
    print(f"  z={z:5.1f} m   v={prof[np.argmin(abs(zr - z))]:.4f}")
print(f"Max positive dv/dz at z={zjump:.2f} m")
upper = V[zr < zjump - 0.5].mean() if (zr < zjump - 0.5).any() else np.nan
lower = V[zr > zjump + 0.5].mean() if (zr > zjump + 0.5).any() else np.nan
print(f"Mean v above {zjump:.1f} m: {upper:.4f} m/ns; below: {lower:.4f} m/ns")

stats = dict(
    pygimli_version=pg.__version__,
    units="SI internally (m, s); velocities reported in m/ns",
    n_data=int(N), n_sensors=int(data.sensorCount()),
    mesh_cells=int(mesh.cellCount()), mesh_nodes=int(mesh.nodeCount()),
    cell_size_m=dx, sec_nodes=3,
    lambda_scan={f"{r['lam']:g}": dict(chi2=float(r["chi2"]),
                                       rms_ns=float(r["rms_ns"]),
                                       iters=int(r["niter"]))
                 for r in results},
    chosen_lambda=float(best["lam"]),
    chi2=float(best["chi2"]), rms_ns=float(best["rms_ns"]),
    iterations=int(best["niter"]),
    start_velocity_mns=float(v0_mns), start_rms_ns=float(rms0_ns),
    vel_min_mns=float(V.min()), vel_max_mns=float(V.max()),
    vel_mean_mns=float(V.mean()),
    layer_boundary_depth_m=float(zjump),
    mean_v_above_mns=float(upper), mean_v_below_mns=float(lower),
)
with open(f"{OUT}/am13_pygimli_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

# -------------------------------------------------------------------- plots
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
INK, MUT, BLUE = "#1a1f27", "#6b7280", "#3b5bdb"

# 1) velocity tomogram
fig, ax = plt.subplots(figsize=(5.4, 8.2), constrained_layout=True)
pc = ax.pcolormesh(xr, zr, V, cmap="viridis", shading="nearest",
                   vmin=np.floor(V.min() * 100) / 100,
                   vmax=np.ceil(V.max() * 100) / 100)
sens = np.array([[p.x(), -p.y()] for p in data.sensors()])
ax.plot(sens[:, 0], sens[:, 1], ".", color="white", ms=2.5, alpha=0.85)
ax.set_xlabel("x (m)"); ax.set_ylabel("depth (m)")
ax.set_xlim(0, 5); ax.set_ylim(12.5, 0.5); ax.set_aspect("equal")
ax.set_title("AM13 crosshole GPR — pyGIMLi velocity model\n"
             f"$\\lambda$={best['lam']:g}, $\\chi^2$={best['chi2']:.2f}, "
             f"RMS={best['rms_ns']:.2f} ns", color=INK)
cb = fig.colorbar(pc, ax=ax, shrink=0.75, pad=0.03)
cb.set_label("velocity (m/ns)")
fig.savefig(f"{OUT}/am13_pygimli_velocity.png", dpi=200)
plt.close(fig)

# 2) fit: measured vs predicted + residuals
res_ns = (resp - t_s) * 1e9
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4), constrained_layout=True)
lim = [t_ns.min() - 1, t_ns.max() + 1]
a1.plot(lim, lim, "--", color=MUT, lw=1, zorder=1)
a1.scatter(t_ns, resp * 1e9, s=9, color=BLUE, alpha=0.45, lw=0, zorder=2)
a1.set_xlabel("measured traveltime (ns)")
a1.set_ylabel("predicted traveltime (ns)")
a1.set_xlim(lim); a1.set_ylim(lim); a1.set_aspect("equal")
a1.set_title("Predicted vs measured", color=INK)
a1.text(0.03, 0.94, f"N = {N}\nRMS = {best['rms_ns']:.2f} ns\n"
        f"$\\chi^2$ = {best['chi2']:.2f}", transform=a1.transAxes,
        va="top", color=INK,
        bbox=dict(fc="white", ec=MUT, lw=0.5, alpha=0.85))
a2.hist(res_ns, bins=41, color=BLUE, alpha=0.75, edgecolor="white", lw=0.4)
a2.axvline(0, color=MUT, lw=1)
for s in (-0.8, 0.8):
    a2.axvline(s, color=MUT, lw=1, ls=":")
a2.set_xlabel("residual: predicted − measured (ns)")
a2.set_ylabel("count")
a2.set_title("Traveltime residuals (dotted: ±0.8 ns = $\\sigma$)", color=INK)
fig.savefig(f"{OUT}/am13_pygimli_fit.png", dpi=200)
plt.close(fig)

print("\nAll outputs written to", OUT)
