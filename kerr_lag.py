"""
Dispersive phase-amplitude lag in the time-resolved optical Kerr effect.

Reproduction study of:
    D. Lorenc and Z. Alpichshev, "Dispersive effects in ultrafast nonlinear
    phenomena: The case of optical Kerr effect", Phys. Rev. Research 6, 013042 (2024).

Units
-----
All energies are in units of the band gap Eg, so the photon energy is the
dimensionless x = hbar*omega / Eg.  Times are in units of hbar / Eg
(0.4636 fs for GaAs with Eg = 1.42 eV).

Model for the complex Kerr index
--------------------------------
We use the two-band dispersion model of Sheik-Bahae et al. (PRL 65, 96 (1990);
IEEE JQE 27, 1296 (1991)) in the *degenerate* form:

    Re n2(x) ∝ G2(x)
    beta(x)  ∝ F2(x) = (2x - 1)^{3/2} / (2x)^5              (x > 1/2)

Because Im n2 = beta * c / (2 omega), the same prefactor gives

    n2bar(x) ∝ G2(x) + i * F2(x) / x .

Simplification (stated honestly): the paper's Eq. (6) involves the
non-degenerate index n2bar(omega, Omega).  Here we use the degenerate curve
and differentiate it with respect to x.  This keeps the code short and
reproduces the qualitative structure of the paper's Fig. 1, but the numbers
should not be read as an exact reproduction.
"""
import numpy as np
from scipy.optimize import curve_fit

HBAR_EV_FS = 0.6582119569  # hbar in eV*fs


# ---------------------------------------------------------------------------
# 1. Dispersion functions
# ---------------------------------------------------------------------------
def G2(x):
    """Sheik-Bahae dispersion function for the real (refractive) Kerr index."""
    x = np.asarray(x, dtype=float)
    theta = np.where(1 - 2 * x > 0, (np.clip(1 - 2 * x, 0, None)) ** 1.5, 0.0)
    num = (-2 + 6 * x - 3 * x**2 - x**3 - 0.75 * x**4 - 0.75 * x**5 + 2 * theta)
    return num / (64 * x**6)


def F2(x):
    """Two-photon absorption spectral function (zero below half the gap)."""
    x = np.asarray(x, dtype=float)
    return np.where(x > 0.5, np.clip(2 * x - 1, 0, None) ** 1.5 / (2 * x) ** 5, 0.0)


def n2bar(x):
    """Complex Kerr index (arbitrary common prefactor)."""
    x = np.asarray(x, dtype=float)
    return G2(x) + 1j * F2(x) / x


def d_dx(fun, x, h=1e-5):
    """Central finite difference."""
    return (fun(x + h) - fun(x - h)) / (2 * h)


# ---------------------------------------------------------------------------
# 2. Analytic lag, Eq. (6) of the paper
# ---------------------------------------------------------------------------
def lag_analytic(x, include_omega_prefactor=False):
    """
    delta_T = 1/2 * [ (dRe/dx) / Im  +  (dIm/dx) / Re ]   in units of hbar/Eg.

    Only defined for 1/2 < x < 1 (Im n2 = 0 below the two-photon edge, and the
    two-band model is not meant for x > 1).

    If include_omega_prefactor is True, the response is f ∝ i*x*n2bar(x)
    (paper Eq. (4) before the omega factor is dropped).  Applying Eq. (6) to
    g = x*n2bar gives an extra term

        delta_T_omega = delta_T + (1 / 2x) * (Re/Im + Im/Re),

    which is of the order of one optical period (|extra| >= 1/x).
    """
    re = G2(x)
    im = F2(x) / x
    dre = d_dx(G2, x)
    dim = d_dx(lambda y: F2(y) / y, x)
    lag = 0.5 * (dre / im + dim / re)
    if include_omega_prefactor:
        lag = lag + omega_correction(x)
    return lag


def omega_correction(x):
    """Extra lag caused by keeping the omega prefactor in f(omega)."""
    re, im = G2(x), F2(x) / x
    return (re / im + im / re) / (2 * x)


def renormalisation_factors(x, tau, include_omega_prefactor=False):
    """
    Paper Fig. 5: how much larger the peak response is than the response at
    zero delay, F = |S_max / S(0) - 1|.  For a Gaussian trace whose centre is
    shifted by t_c, S(0) = S_max * exp(-t_c^2 / tau_tilde^2), so
        F = exp(t_c^2 / tau_tilde^2) - 1.
    Here both pulses have duration tau, hence tau_tilde = tau.
    Returns (F_phase, F_amplitude).
    """
    tA, tphi = peak_shifts_analytic(x)
    if include_omega_prefactor:
        re, im = G2(x), F2(x) / x
        tA = tA + re / (2 * x * im)     # from g = x*n2bar, see report Sec. 5
        tphi = tphi - im / (2 * x * re)
    return np.expm1(tphi**2 / tau**2), np.expm1(tA**2 / tau**2)


def lag_fs_for_gap(Eg_eV, lam_nm, include_omega_prefactor=False):
    """Model lag in femtoseconds for a probe wavelength and an assumed band gap."""
    x = wavelength_to_x(lam_nm, Eg_eV)
    return lag_analytic(x, include_omega_prefactor) * time_unit_fs(Eg_eV)


def peak_shifts_analytic(x):
    """
    Predicted delay at which each trace peaks (Eq. (5)):
        amplitude peak:  t_A   = +(dRe/dx) / (2 Im)
        phase peak:      t_phi = -(dIm/dx) / (2 Re)
    so that delta_T = t_A - t_phi.
    """
    re, im = G2(x), F2(x) / x
    dre, dim = d_dx(G2, x), d_dx(lambda y: F2(y) / y, x)
    return dre / (2 * im), -dim / (2 * re)


# ---------------------------------------------------------------------------
# 3. Time-domain pump-probe simulation (independent check of Eq. (6))
# ---------------------------------------------------------------------------
def simulate_pump_probe(x0, tau_probe, tau_pump, delays, dt=0.5, n_pts=2**15,
                        include_omega_prefactor=False, band=0.05):
    """
    Simulate the nonlinear probe field for a series of pump-probe delays.

    delta_E_NL(t) = ∫ f(t - t') E_pr(t') I_p(t') dt'          (Eqs. (2)-(3))
    f(omega)      = i * n2bar(omega)          (default, as in the paper's Eq. (5))
                  = i * omega * n2bar(omega)  (if include_omega_prefactor, Eq. (4))

    Fields use the convention E(t) = A(t) exp(-i x0 t).  The probe change is
    read out by projecting delta_E_NL onto the unperturbed probe, which is what
    the interferometer measures:
        C(t0) = ∫ delta_E_NL E_pr^* dt / ∫ |E_pr|^2 dt
        dA/A  = Re C   (amplitude / absorption channel)
        dphi  = Im C   (phase / refraction channel)

    Returns (dA, dphi) arrays, one value per delay.
    """
    t = (np.arange(n_pts) - n_pts // 2) * dt
    omega = 2 * np.pi * np.fft.fftfreq(n_pts, d=dt)   # in units of Eg/hbar

    # response function in the frequency domain, restricted to a band around x0
    mask = np.abs(omega - x0) < band
    f_w = np.zeros(n_pts, dtype=complex)
    f_w[mask] = 1j * n2bar(omega[mask])
    if include_omega_prefactor:
        f_w[mask] *= omega[mask] / x0      # normalised so magnitudes stay comparable

    E_pr = np.exp(-t**2 / tau_probe**2) * np.exp(-1j * x0 * t)
    norm = np.sum(np.abs(E_pr) ** 2)

    dA, dphi = [], []
    for t0 in delays:
        I_p = np.exp(-2 * (t - t0) ** 2 / tau_pump**2)
        P = E_pr * I_p
        # forward transform with exp(+i omega t)  ->  numpy ifft; inverse -> fft
        P_w = np.fft.ifft(np.fft.ifftshift(P))
        dE = np.fft.fftshift(np.fft.fft(f_w * P_w))
        C = np.sum(dE * np.conj(E_pr)) / norm
        dA.append(C.real)
        dphi.append(C.imag)
    return np.array(dA), np.array(dphi)


def _gauss(t, a, t0, w, c):
    return a * np.exp(-((t - t0) / w) ** 2) + c


def peak_position(delays, trace):
    """Fit a Gaussian and return its centre (robust against sign and offset)."""
    i = np.argmax(np.abs(trace))
    p0 = [trace[i], delays[i], (delays[-1] - delays[0]) / 6, 0.0]
    popt, _ = curve_fit(_gauss, delays, trace, p0=p0, maxfev=20000)
    return popt[1]


def lag_simulated(x0, tau_probe, tau_pump, n_delays=161, span=2.5, **kw):
    """Lag t_A - t_phi extracted from the time-domain simulation."""
    tt = np.sqrt((tau_probe**2 + tau_pump**2) / 2)
    delays = np.linspace(-span * tt, span * tt, n_delays)
    dA, dphi = simulate_pump_probe(x0, tau_probe, tau_pump, delays, **kw)
    return peak_position(delays, dA) - peak_position(delays, dphi)


# ---------------------------------------------------------------------------
# 4. Helpers for physical units
# ---------------------------------------------------------------------------
def time_unit_fs(Eg_eV):
    """hbar / Eg in femtoseconds."""
    return HBAR_EV_FS / Eg_eV


def wavelength_to_x(lam_nm, Eg_eV):
    return 1239.84198 / lam_nm / Eg_eV
