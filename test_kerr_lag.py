"""Basic checks.  Run with:  python -m pytest -q"""
import numpy as np
import kerr_lag as k

U = k.time_unit_fs(1.42)
TAU = 290 / U


def test_two_photon_edge():
    assert k.F2(0.49) == 0.0
    assert k.F2(0.6) > 0.0


def test_zero_crossing_of_G2():
    xs = np.linspace(0.6, 0.9, 30001)
    xz = xs[np.where(np.diff(np.sign(k.G2(xs))))[0][0]]
    assert abs(xz - 0.740) < 2e-3


def test_simulation_matches_eq6():
    for x in [0.6, 0.68, 0.8, 0.9]:
        assert abs(k.lag_simulated(x, TAU, TAU) - k.lag_analytic(x)) < 0.02


def test_simulation_matches_omega_formula():
    for x in [0.6, 0.8, 0.9]:
        sim = k.lag_simulated(x, TAU, TAU, include_omega_prefactor=True)
        assert abs(sim - k.lag_analytic(x, include_omega_prefactor=True)) < 0.02


def test_lag_independent_of_pulse_duration():
    x = 0.85
    lags = [k.lag_simulated(x, t / U, t / U) for t in (80, 200, 500)]
    assert np.ptp(lags) < 1e-2
