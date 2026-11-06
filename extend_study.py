"""Extensions beyond the first reproduction (figures 6-9)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import kerr_lag as k

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
BLUE, RED, GREEN, GREY = "#2b5aa8", "#c0392b", "#2e8b57", "#777777"

EG_GAAS = 1.42
U = k.time_unit_fs(EG_GAAS)
LAM = 1028.0
out = []

# ---------------------------------------------------------------- Fig. 6: omega prefactor, analytic + simulation
xs = np.linspace(0.505, 0.995, 2000)
x_chk = np.linspace(0.56, 0.98, 22)
T = 290 / U
sim_w = np.array([k.lag_simulated(x, T, T, include_omega_prefactor=True) for x in x_chk])
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(xs, k.lag_analytic(xs), color=GREY, lw=1, label=r"Eq. (6), $f\propto i\bar n_2$")
ax.plot(xs, k.lag_analytic(xs, include_omega_prefactor=True), color=RED, lw=1.2,
        label=r"Eq. (6) + $\frac{1}{2x}\left(\frac{\mathrm{Re}}{\mathrm{Im}}+\frac{\mathrm{Im}}{\mathrm{Re}}\right)$")
ax.plot(x_chk, sim_w, "s", ms=3.5, color=RED, mfc="none", label=r"simulation, $f\propto i\omega\bar n_2$")
ax.set_ylim(-15, 35)
ax.axhline(0, color=GREY, lw=0.5)
ax.set_xlabel(r"photon energy  $x=\hbar\omega/E_g$")
ax.set_ylabel(r"$\delta T$  ($\hbar/E_g$)")
ax.set_title(r"Effect of the $\omega$ prefactor")
ax.legend(frameon=False, fontsize=7.5)
fig.savefig("figures/fig6_omega_prefactor.png")
err = np.max(np.abs(sim_w - k.lag_analytic(x_chk, True))[np.abs(x_chk - 0.74) > 0.01])
out.append(f"omega-prefactor formula vs simulation, max abs error : {err:.3f} hbar/Eg")
out.append(f"omega correction at 1028 nm in GaAs                   : "
           f"{k.omega_correction(k.wavelength_to_x(LAM, EG_GAAS)) * U:.2f} fs")

# ---------------------------------------------------------------- Fig. 7: renormalisation factors (paper Fig. 5)
tau70 = 70 / U
xr = np.linspace(0.52, 0.98, 1500)
Fph, Famp = k.renormalisation_factors(xr, tau70)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.semilogy(xr, Fph, color=RED, label=r"phase  $F_{ph}$")
ax.semilogy(xr, Famp, color=BLUE, label=r"amplitude  $F_{amp}$")
ax.set_ylim(1e-6, 10)
ax.set_xlim(0.52, 0.98)
ax.set_xlabel(r"photon energy  $x=\hbar\omega/E_g$")
ax.set_ylabel(r"$F=|S_{max}/S(0)-1|$")
ax.set_title(r"Underestimate of the Kerr response at zero delay ($\tau$ = 70 fs)")
ax.legend(frameon=False, fontsize=8)
fig.savefig("figures/fig7_renormalisation_factors.png")
# simulation check at a few points
d = np.linspace(-3 * tau70, 3 * tau70, 2001)
for x in [0.6, 0.73, 0.76, 0.85]:
    dA, dp = k.simulate_pump_probe(x, tau70, tau70, d)
    sim_F = abs(np.max(np.abs(dp)) / np.abs(dp[len(d) // 2]) - 1)
    out.append(f"F_ph at x={x:.2f}: formula {k.renormalisation_factors(x, tau70)[0]:.2e}, "
               f"simulation {sim_F:.2e}")

# ---------------------------------------------------------------- Fig. 8: sensitivity to the zero crossing
egs = np.linspace(1.30, 2.30, 20001)
lag_fs = k.lag_fs_for_gap(egs, LAM)
x_of = k.wavelength_to_x(LAM, egs)
valid = (x_of > 0.5) & (x_of < 1.0)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(egs[valid], lag_fs[valid], color=BLUE)
ax.axhspan(100, 120, color=RED, alpha=0.15, label="measured (paper): 100-120 fs")
ax.axvline(EG_GAAS, color=GREY, ls=":", lw=1)
ax.text(EG_GAAS + 0.01, -140, "GaAs\n1.42 eV", color=GREY, fontsize=8)
ax.set_ylim(-200, 200)
ax.set_xlim(1.3, 1.75)
ax.set_xlabel(r"assumed band gap $E_g$ (eV)")
ax.set_ylabel("model lag at 1028 nm (fs)")
ax.set_title("How close must the probe be to the Re n2 zero crossing?")
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.savefig("figures/fig8_gap_sensitivity.png")
inside = valid & (lag_fs >= 100) & (lag_fs <= 120)
if inside.any():
    e_in = egs[inside]
    out.append(f"band gap giving 100-120 fs at 1028 nm : {e_in.min():.4f} - {e_in.max():.4f} eV "
               f"(window {1000 * (e_in.max() - e_in.min()):.1f} meV)")
    xz = k.wavelength_to_x(LAM, e_in.mean())
    out.append(f"corresponding probe energy            : x = {xz:.4f}")
big = valid & (np.abs(lag_fs) > 50)
out.append(f"total gap range with |lag| > 50 fs    : {1000 * (egs[big].max() - egs[big].min()):.1f} meV "
           f"(around {egs[big].mean():.3f} eV)")

# ---------------------------------------------------------------- Fig. 9: GaAs vs a lead-halide perovskite
lam = np.linspace(600, 2400, 4000)
fig, ax = plt.subplots(figsize=(5.2, 3.4))
for eg, name, c in [(EG_GAAS, "GaAs (1.42 eV)", BLUE), (2.30, r"MAPbBr$_3$ (2.3 eV)", GREEN)]:
    x = k.wavelength_to_x(lam, eg)
    ok = (x > 0.505) & (x < 0.995)
    ax.plot(lam[ok], k.lag_analytic(x[ok]) * k.time_unit_fs(eg), color=c, label=name)
    xz = 0.7402
    out.append(f"{name.split()[0].replace('$_3$','3'):<10s} zero crossing at {1239.84 / (xz * eg):.0f} nm; "
               f"lag scale hbar/Eg = {k.time_unit_fs(eg):.3f} fs")
ax.axhline(0, color=GREY, lw=0.5)
ax.set_ylim(-20, 40)
ax.set_xlabel("probe wavelength (nm)")
ax.set_ylabel("model lag (fs)")
ax.set_title("Where to look for a large lag")
ax.legend(frameon=False, fontsize=8)
fig.savefig("figures/fig9_gaas_vs_perovskite.png")

with open("results_extensions.txt", "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))
