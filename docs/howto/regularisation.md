# Regularisation

On noisy or under-constrained data the retrieved spectrum can overfit, picking
up ripples in amplitude or oscillations in phase that fit the noise rather than
the pulse. Every solver except COPRA (the L-BFGS family, the Levenberg–Marquardt
family and the global {class}`~croak.cmaes.CMAES`) accepts mild penalties that
suppress this, at the cost of a bias towards the chosen spectrum, smoothness or
time window. The objective is defined in
[retrieval theory](../explanation/retrieval_theory.md); the [summary table](#summary)
lists which option each solver takes.

Two penalties are on **by default**: amplitude smoothness (`reg_amp=0.03`) and,
when an independent spectrum is supplied, the spectral-match penalty
(`reg_spectrum=0.01`). The LM family uses `3e-5` and `1e-5` instead, because it
minimises squared residuals. Pass `0` to disable either. COPRA and `copra-jax`
take none of these penalties; `warm-lbfgs` applies them in its L-BFGS stage.

## Second-difference smoothness

Two penalties damp curvature via a discrete **second difference**
($A_{k+1} - 2A_k + A_{k-1}$, the discrete analogue of $\mathrm{d}^2/\mathrm{d}\omega^2$):

- **`reg_amp`** — penalises curvature of the spectral **amplitude**;
- **`reg_phase`** — penalises curvature of the spectral **phase**.

```python
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="lbfgs",
                    reg_amp=0.01, reg_phase=0.01, maxiters=300)
```

A second difference vanishes for a constant or linear amplitude and for a
linear spectral phase (a time shift). Quadratic phase (pure GDD) has a constant,
nonzero second difference and so pays a small, uniform penalty, while ripple and
bin-to-bin oscillation have large second differences and are suppressed
strongly. `reg_phase` therefore prefers smooth phase rather than zero chirp.

The penalties have closed-form gradients, accumulated into the solver gradient
(L-BFGS) or appended as extra residual rows (LM).

## Choosing the weights

croak's defaults are `reg_spectrum=0.01` and `reg_amp=0.03`. The validation
retrievals in the [papers](../reference/bibliography.md) use `reg_spectrum=0.03`
and `reg_amp=0.03`; use those to reproduce them. Three points matter when
retuning:

- **The natural diagnostics fail.** The retrieved duration is nearly flat in the
  weight, and the agreement with the target spectrum improves monotonically with
  it by construction: a regulariser cannot be tuned on the quantity it pulls
  towards. On simulated data, tune against the complex-field error relative to
  the known truth; on measured data, prefer the defaults and check the spread
  over random starts.
- **The two penalties work together.** Too weak a spectral weight lets the
  amplitude collapse on chirped traces (a spiky spectrum a fraction of the true
  bandwidth); the amplitude-smoothness penalty suppresses exactly that collapse,
  which is why the pair works better than either alone.
- **The measured spectrum has its own error.** A spectral weight strong enough
  to impose it exactly imports that error, or a difference in how it was
  collected, into the retrieved pulse. Excessive smoothing likewise removes real
  structure.

```{admonition} L-BFGS vs LM weight scaling
:class: note
{class}`~croak.lbfgs.LBFGS` adds the penalty to the trace error $R$;
{class}`~croak.lm.LM` minimises the *sum of squares*, so its weight scales the
penalty relative to $R^2$. The penalty *functions* are identical, but a given
numeric weight is not directly comparable between the two: at stationarity the
correspondence is roughly $\lambda_{\rm LM} \approx 2R\,\lambda$, which for the
default `reg_spectrum=0.01` at typical converged $R$ puts the LM-side weight
near `1e-5`–`3e-5`. Retune when switching solver families.
```

`reg_phase` is most valuable in **phase-only** retrieval, where the amplitude is
fixed and only the phase can absorb noise.

## Spectral-match penalty

When you have an independently measured fundamental spectrum, you can pull the
retrieved amplitude toward it (full amplitude-and-phase mode only) with
**`reg_spectrum`** and **`spectrum_target`** — the measured spectral *intensity*
on the retrieval grid, e.g. `td.Iomega` from
[preprocessing](preprocessing.md):

```python
res = croak.retrieve(td.trace, td.omega, td.delays, td.interaction,
                    algorithm="lbfgs", reg_spectrum=0.05,
                    spectrum_target=td.Iomega, omega0=td.omega0_pulse)
```

This keeps the retrieval consistent with a trusted spectrometer measurement while
still letting the trace determine the phase. {func}`~croak.pipeline.retrieve_from_tracedata`
wires `td.Iomega` in for you (and warns if `reg_spectrum > 0` but no spectrum is
available).

## Temporal regularisation

Some artefacts live in the *time* domain rather than the spectrum: a noisy or
under-constrained retrieval can grow spurious satellite pulses or a low pedestal
far from the main feature. The temporal penalty **`reg_time`** suppresses the
fraction of pulse energy $|E(t)|^2$ that falls outside a time window
**`time_window`** (a `(t_lo, t_hi)` tuple in seconds; default the measured delay
range), with Planck-tapered edges so it does not introduce a hard cut:

```python
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="lbfgs-ad",
                    reg_time=0.05, time_window=(-40e-15, 40e-15), maxiters=300)
```

Unlike the cosmetic post-retrieval [temporal filter](postprocessing.md), this acts
*during* retrieval and keeps the spectrum and field an exact Fourier pair, so it
introduces no spectral artefacts; a window narrower than the pulse will, of
course, suppress real satellites. It is available on
{class}`~croak.lbfgs_ad.LBFGSAD` and {class}`~croak.cmaes.CMAES` (it needs the JAX
field transform), and works in every parameterisation including the B-spline basis.

## The B-spline phase basis

The penalties above are *soft* constraints added to the objective. The
**B-spline phase basis** is a harder, structural one: instead of optimising the
spectral phase per frequency bin, `phase_basis="bspline"` represents it as a cubic
B-spline with a handful of control points (`n_nodes`, default 20) over the
non-zero-spectrum support, holding the amplitude at the guess.

```python
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="lbfgs-ad",
                    phase_basis="bspline", n_nodes=24, maxiters=300)
```

Because the phase then has only `n_nodes` degrees of freedom and is $C^2$-smooth by
construction, it cannot ripple from bin to bin: a strong regulariser that also makes the
problem low-dimensional enough for the derivative-free
[CMA-ES global search](../explanation/algorithms.md#global-retrieval-cma-es). It
implies phase-only retrieval. A P-spline roughness term on the control points is
folded into `reg_phase`, so you can still smooth the spline itself. Supported by
`lbfgs-ad`, `lbfgs-optx`, `lm`, `lm-optx` and `cma-es`; see
{func}`croak._jax_pulse.make_spline_phase_basis`.

## Per-frequency scaling

Not a penalty, but related: `R_omega=True` replaces the single intensity
[scale factor](../explanation/retrieval_theory.md) with a per-frequency vector,
absorbing an uncalibrated frequency-dependent detection efficiency. It *relaxes*
a constraint, so it pairs well with a spectral-match penalty that re-adds physical
structure.

:::{admonition} Rω lowers the reported error while degrading the spectrum
:class: warning
Do not use `R_omega=True` on its own. The per-frequency scale factors are exactly
the information that pins the retrieved **spectral amplitude**: one free factor
per row *is* the frequency marginal. With those extra parameters the amplitudes
are constrained only through the delay structure of each row, which is weakest
where the trace is weakest, at the edges of the band.

Because $|E(\omega)|^2 \ge 0$ and the true value sits on that bound, the freed
bins can only drift upward. The signature is the retrieved spectrum rising at
both ends of the grid, and because $I_\lambda = I_\omega\,2\pi c/\lambda^2$ the
short-wavelength end looks far worse than its long-wavelength twin, by a factor
$(\lambda_\mathrm{max}/\lambda_\mathrm{min})^2$, easily 5–10×.
{func}`~croak.processing.edge_energy_fraction` puts a number on it, and the
{func}`~croak.plotting.plot_retrieval` spectrum panel reports it, in red above
{data}`~croak.plotting.EDGE_ENERGY_WARN` (1 %).

On a perfect, noiseless, exactly modelled PG trace whose true pulse has
$10^{-6}$ % of its energy in the outer 10 bins of each end:

| | edge energy | trace error `R` |
|---|---|---|
| `R_omega=False` | 0.02 % | 7.9e-5 |
| `R_omega=True` | **5.4 %** | 4.0e-3 |

Rω usually reports the *lower* `R`, since it has one extra free parameter per
frequency row, so the error alone cannot show that the spectrum has gone wrong.
On our measured traces it was routine for Rω to halve `R` while tripling the
edge energy.

The fix is to put back what Rω removed, from an independent measurement:
`phase_only=True` (hold the amplitude at the measured spectrum) or a light
`reg_spectrum`. Widening `lam_min`/`lam_max` helps too, because it moves the
[edge taper](grids_and_filtering.md#the-edge-taper-and-why-the-band-must-be-generous)
out onto empty trace. On one of our deep-UV TG-FROG datasets, widening the band
from 180–500 nm to 140–750 nm, with a reference spectrum carrying 0.01 % edge
energy:

| | edge energy |
|---|---|
| Rω, tight band | 15.6 % |
| Rω off, tight band | 6.0 % |
| Rω off + `reg_spectrum=1e-2`, wide band | **0.01 %** |

COPRA supports phase-only retrieval but not the spectral penalty.
:::

## Summary

| Option | Solvers | Effect |
|--------|---------|--------|
| `reg_amp` | LBFGS, LBFGSAD, LBFGSHand, OptxLBFGS, LM, OptxLM, CMAES | smooth spectral amplitude |
| `reg_phase` | LBFGS, LBFGSAD, LBFGSHand, OptxLBFGS, LM, OptxLM, CMAES | smooth spectral phase (great for phase-only) |
| `reg_spectrum` + `spectrum_target` | LBFGS, LBFGSAD, LBFGSHand, OptxLBFGS, LM, OptxLM, CMAES (full mode) | pull amplitude to a measured spectrum |
| `reg_time` + `time_window` | LBFGSAD, CMAES | suppress energy outside a time window |
| `phase_basis="bspline"` | LBFGSAD, OptxLBFGS, LM, OptxLM, CMAES | low-dimensional, intrinsically smooth phase |
| `R_omega` | all solvers | per-frequency intensity scaling — **never alone**, see above |

(COPRA and its JAX twin take none of the *penalties* — their random-order local
sweep already smooths implicitly — but they do support `R_omega`; see
[per-frequency scaling](../explanation/algorithms.md#copra).)
