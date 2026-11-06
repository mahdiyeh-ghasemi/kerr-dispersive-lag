"""Generate all figures and a short numerical summary for the README."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kerr_lag as k

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, RED, GREY = "#2b5aa8", "#c0392b", "#777777"

# GaAs parameters (paper: Lorenc & Alpichshev 2024)
EG = 1.42                       # eV, room-temperature band gap
U_FS = k.time_unit_fs(EG)       # hbar/Eg in fs
X_PROBE = k.wavelength_to_x(1028, EG)
TAU = 290 / U_FS                # pulse duration (paper: 290 fs) in hbar/Eg

xs = np.linspace(0.505, 0.995, 2000)

# ---------------------------------------------------------------- Fig. 1
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(xs, k.G2(xs) / np.abs(k.G2(xs)).max(), color=BLUE, label=r"Re $\bar n_2$  ($\propto G_2$)")
ax.plot(xs, (k.F2(xs) / xs) / (k.F2(xs) / xs).max(), color=RED, label=r"Im $\bar n_2$  ($\propto F_2/x$)")
ax.axhline(0, color=GREY, lw=0.6)
ax.axvline(X_PROBE, color=GREY, ls=":", lw=1)
ax.text(X_PROBE + 0.005, -0.9, "GaAs, 1028 nm", color=GREY, fontsize=8)
ax.set_xlabel(r"photon energy  $x=\hbar\omega/E_g$")
ax.set_ylabel("normalised")
ax.set_title("Complex Kerr index, two-band model")
ax.legend(frameon=False, fontsize=8)
fig.savefig("figures/fig1_n2_dispersion.png")

# ---------------------------------------------------------------- Fig. 2
lag = k.lag_analytic(xs)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
pos = lag > 0
ax.semilogy(np.where(pos, xs, np.nan), np.abs(lag) / (2 * np.pi), ".", ms=1.5, color=RED, label="phase lags amplitude")
ax.semilogy(np.where(~pos, xs, np.nan), np.abs(lag) / (2 * np.pi), ".", ms=1.5, color=BLUE, label="phase leads amplitude")
ax.axvline(X_PROBE, color=GREY, ls=":", lw=1)
ax.set_xlabel(r"photon energy  $x=\hbar\omega/E_g$")
ax.set_ylabel(r"$|\delta T|$  in units of $h/E_g$")
ax.set_title("Phase-amplitude lag, Eq. (6)")
ax.legend(frameon=False, fontsize=8, markerscale=6)
fig.savefig("figures/fig2_lag_vs_photon_energy.png")

# ---------------------------------------------------------------- Fig. 3
tt = np.sqrt(TAU**2)                     # equal pulses -> tau_tilde = tau
delays = np.linspace(-2.5 * tt, 2.5 * tt, 201)
x_demo = 0.75                            # near the Re n2 zero crossing, lag is large
dA, dphi = k.simulate_pump_probe(x_demo, TAU, TAU, delays)
tA, tphi = k.peak_position(delays, dA), k.peak_position(delays, dphi)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(delays * U_FS, dA / np.abs(dA).max(), color=BLUE, label="amplitude  (Re C)")
ax.plot(delays * U_FS, dphi / np.abs(dphi).max(), color=RED, label="phase  (Im C)")
for tp, c in [(tA, BLUE), (tphi, RED)]:
    ax.axvline(tp * U_FS, color=c, ls="--", lw=0.8)
ax.set_xlabel("pump-probe delay  (fs)")
ax.set_ylabel("normalised response")
ax.set_title(f"Time-domain simulation, x = {x_demo}  (lag = {(tA - tphi) * U_FS:.1f} fs)")
ax.legend(frameon=False, fontsize=8)
fig.savefig("figures/fig3_time_domain_traces.png")

# ---------------------------------------------------------------- Fig. 4
x_chk = np.linspace(0.56, 0.98, 22)
sim = np.array([k.lag_simulated(x, TAU, TAU) for x in x_chk])
sim_w = np.array([k.lag_simulated(x, TAU, TAU, include_omega_prefactor=True) for x in x_chk])
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(xs, lag, color=GREY, lw=1, label="analytic Eq. (6)")
ax.plot(x_chk, sim, "o", ms=4, mfc="none", color=BLUE, label=r"simulation, $f\propto i\bar n_2$")
ax.plot(x_chk, sim_w, "s", ms=3.5, color=RED, label=r"simulation, $f\propto i\omega\bar n_2$")
ax.set_ylim(-15, 35)
ax.axhline(0, color=GREY, lw=0.5)
ax.set_xlabel(r"photon energy  $x=\hbar\omega/E_g$")
ax.set_ylabel(r"$\delta T$  ($\hbar/E_g$)")
ax.set_title("Check: simulation vs analytic formula")
ax.legend(frameon=False, fontsize=8)
fig.savefig("figures/fig4_validation.png")

# ---------------------------------------------------------------- Fig. 5 (pulse-duration independence)
taus_fs = np.array([60, 100, 150, 200, 290, 400, 600])
lags_tau = np.array([k.lag_simulated(X_PROBE, t / U_FS, t / U_FS) for t in taus_fs])
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(taus_fs, lags_tau * U_FS, "o-", color=BLUE)
ax.axhline(k.lag_analytic(X_PROBE) * U_FS, color=GREY, ls="--", lw=1, label="analytic")
ref = k.lag_analytic(X_PROBE) * U_FS
ax.set_ylim(ref - 0.5, ref + 0.5)
ax.text(330, ref - 0.35, f"spread: {np.ptp(lags_tau) * U_FS * 1000:.1f} as", fontsize=8, color=GREY)
ax.set_xlabel("pulse duration  (fs)")
ax.set_ylabel("lag  (fs)")
ax.set_title("Lag does not depend on pulse duration (GaAs, 1028 nm)")
ax.legend(frameon=False, fontsize=8)
fig.savefig("figures/fig5_pulse_duration.png")

# ---------------------------------------------------------------- summary
xz = xs[np.where(np.diff(np.sign(k.G2(xs))))[0][0]]
away = np.abs(x_chk - xz) > 0.01          # exclude the point sitting on the Re n2 zero crossing
abs_err = np.abs(sim - k.lag_analytic(x_chk))
max_err = np.max(abs_err[away])
with open("results_summary.txt", "w") as fh:
    fh.write(f"hbar/Eg (GaAs, Eg = {EG} eV)          : {U_FS:.4f} fs\n")
    fh.write(f"probe photon energy x (1028 nm)       : {X_PROBE:.3f}\n")
    fh.write(f"zero crossing of Re n2 (model)        : x = {xz:.3f}  ({1239.84 / (xz * EG):.0f} nm)\n")
    fh.write(f"analytic lag at 1028 nm (model)       : {k.lag_analytic(X_PROBE) * U_FS:.2f} fs\n")
    fh.write(f"analytic lag at x = 0.75              : {k.lag_analytic(0.75) * U_FS:.2f} fs\n")
    fh.write(f"max abs. error, sim vs Eq. (6)        : {max_err:.3f} hbar/Eg  (all points except x on the zero crossing)\n")
    fh.write("at the zero crossing itself Eq. (6) diverges (first-order expansion breaks down);\n"
             "the simulation stays finite there.\n")
    fh.write(f"lag vs pulse duration (fs)            : " +
             ", ".join(f"{t}fs->{l * U_FS:.2f}" for t, l in zip(taus_fs, lags_tau)) + "\n")
    fh.write("measured in the paper (GaAs, 1028 nm) : ~100 fs (xx), ~120 fs (xy)\n")
print(open("results_summary.txt").read())
