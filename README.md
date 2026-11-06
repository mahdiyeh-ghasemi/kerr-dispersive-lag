# Dispersive phase–amplitude lag in the time-resolved optical Kerr effect

A small reproduction and extension study of

> D. Lorenc and Z. Alpichshev, *Dispersive effects in ultrafast nonlinear phenomena: The case of optical Kerr effect*, **Phys. Rev. Research 6, 013042 (2024)**.

The paper shows that causality (Kramers–Kronig) produces a measurable delay between the
**phase** and **amplitude** responses of a probe pulse in a pump–probe Kerr experiment,
even when the electronic Kerr response is quasi-instantaneous. The lag depends only on the
dispersion of the complex Kerr index and not on pump intensity.

This repository:

1. Evaluates the paper's analytic lag formula (Eq. 6) with the two-band dispersion model of
   Sheik-Bahae *et al.* for the complex Kerr index.
2. Checks that formula independently with a **time-domain pump–probe simulation**: the
   nonlinear probe field is computed by convolution with the causal Kerr response, then
   projected onto the unperturbed probe, the way an interferometer reads it out.
3. Extends the analysis: the effect of the ω prefactor in the response function, the
   zero-delay underestimate of the Kerr response (paper Fig. 5), the sensitivity of the lag to
   the position of the Re n̄₂ zero crossing, and a comparison of GaAs with a lead-halide perovskite.

## Files

| file | content |
|---|---|
| `kerr_lag.py` | model functions, analytic formulas, time-domain simulation |
| `make_figures.py` | figures 1–5 and `results_summary.txt` |
| `extend_study.py` | figures 6–9 and `results_extensions.txt` |
| `test_kerr_lag.py` | checks (simulation vs formulas, pulse-duration independence) |
| `figures/` | output plots |

```bash
pip install -r requirements.txt
python make_figures.py
python extend_study.py
python -m pytest -q
```

## Model

Energies are in units of the band gap, x = ħω/E<sub>g</sub>; times in units of ħ/E<sub>g</sub>
(0.46 fs for GaAs). With the degenerate Sheik-Bahae functions

- Re n̄₂ ∝ G₂(x)
- β ∝ F₂(x) = (2x−1)<sup>3/2</sup>/(2x)<sup>5</sup>, and Im n̄₂ = βc/2ω, so n̄₂ ∝ G₂(x) + i F₂(x)/x

the lag is δT = ½ [ (∂Re n̄₂/∂x)/Im n̄₂ + (∂Im n̄₂/∂x)/Re n̄₂ ].

## Results

**Fig. 1 – complex Kerr index.** Re n̄₂ changes sign at x ≈ 0.74 in this model.

**Fig. 2 – lag vs photon energy.** As in the paper, the lag is largest near the two-photon
edge (x → 0.5) and diverges at the zero crossing of Re n̄₂. It changes sign across the crossing.

**Fig. 3 – time-domain traces** at x = 0.75: a shift of about 12 fs between the two channels,
with 290 fs pulses.

**Fig. 4 – validation.** The simulation agrees with Eq. 6 to within 0.014 ħ/E<sub>g</sub> at every
tested photon energy except the one sitting on the zero crossing. There the first-order
expansion behind Eq. 6 breaks down and the formula diverges, while the simulated lag stays finite.

**Fig. 5 – pulse duration.** The simulated lag is the same for 60–600 fs pulses.

**Fig. 6 – the ω prefactor.** The response function is f(ω) ≈ iωL n̄₂/c (paper Eq. 4); the
paper drops the ω factor when deriving Eq. 5. Applying Eq. 6 to g(x) = x·n̄₂(x) gives

δT<sub>ω</sub> = δT + (1/2x) (Re n̄₂/Im n̄₂ + Im n̄₂/Re n̄₂),

which the simulation confirms to within 0.004 ħ/E<sub>g</sub>. The extra term is of the order of one
optical period (|extra| ≥ 1/x in units of ħ/E<sub>g</sub>), about −0.9 fs for GaAs at 1028 nm. It is
negligible next to a ~100 fs lag but not next to the ~1.4 fs this simple model predicts.

**Fig. 7 – zero-delay underestimate (paper Fig. 5).** For Gaussian traces whose centre is shifted
by t<sub>c</sub>, the peak exceeds the zero-delay value by F = exp(t<sub>c</sub>²/τ̃²) − 1. With 70 fs pulses this
is below 10⁻³ in most of the spectrum and becomes large only next to the zero crossing.

**Fig. 8 – sensitivity to the zero crossing.** Within this model, a lag of 100–120 fs at
1028 nm requires the probe to sit almost exactly at the zero crossing: the assumed gap must lie in
a window only about 0.3 meV wide (around 1.627 eV, compared with 1.42 eV for GaAs). Such
fine-tuning suggests that this two-band, degenerate model alone cannot explain the measured
size of the lag, and that additional physics is involved.

**Fig. 9 – GaAs vs MAPbBr₃.** Scaled with the band gap, the model places the Re n̄₂ zero crossing
near 1180 nm for GaAs and near 730 nm for MAPbBr₃ (E<sub>g</sub> = 2.3 eV); these are the wavelengths
where a large lag would be expected.

## Limitations and open questions

- **Degenerate approximation.** Eq. 6 in the paper uses the non-degenerate index n̄₂(ω, Ω).
  Here the degenerate curve is differentiated instead.
- **Two-band model.** No excitons, no real band structure, no Raman or free-carrier terms beyond
  what G₂ contains. The 100× gap between model and measurement at 1028 nm is the main open question.
- **Perovskite comparison** only rescales the same model with a different gap; real perovskites
  have strong excitonic and spin–orbit effects not included here.

## Acknowledgement

This project was developed with the help of AI-assisted coding tools.
