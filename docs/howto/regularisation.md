# Regularisation

Regularisation adds information or preferences to a retrieval when the trace
alone is weakly constraining. It can suppress noise-driven oscillations, but it
also biases the solution towards the chosen spectrum, smoothness or time window.
Assess that bias by varying justified constraints and inspecting the whole field.
Start with [retrieval theory](../explanation/retrieval_theory.md); the
[solver guide](solver_selection.md) identifies supported options.

The gradient solvers default to amplitude smoothness `reg_amp=0.03` and spectral
matching `reg_spectrum=0.01` when a target is supplied. LM-family defaults are
`3e-5` and `1e-5`, respectively, because they use squared residuals. Set a weight
to zero to disable that penalty. COPRA and `copra-jax` do not implement these
explicit penalties. Warm L-BFGS applies them during its L-BFGS stage.

## Second-difference smoothness

Two penalties damp curvature via a discrete **second difference**
($A_{k+1} - 2A_k + A_{k-1}$, the discrete analogue of $\mathrm{d}^2/\mathrm{d}\omega^2$):

- **`reg_amp`** — penalises curvature of the spectral **amplitude**;
- **`reg_phase`** — penalises curvature of the spectral **phase**.

```python
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="lbfgs",
                    reg_amp=0.01, reg_phase=0.01, maxiters=300)
```

A second difference vanishes for constant or linear values, including a linear
spectral phase (a time shift). It does **not** vanish for quadratic spectral
phase: pure GDD has nonzero curvature and is penalised. Smoothness is therefore
a modelling preference, not a distinction between physical and unphysical phase.
Rapid oscillations generally incur a larger penalty; its scale also depends on
sampling and the package's normalisation.

The penalties have closed-form gradients, accumulated into the solver gradient
(L-BFGS) or appended as extra residual rows (LM).

## Choosing the weights

The appropriate weights depend on noise, sampling, model fidelity and confidence
in the auxiliary measurement. The revised draft uses a spectral weight of 0.03,
not the package default of 0.01. Neither value is universal. A tighter spectral
constraint necessarily improves agreement with its target, so that agreement
alone cannot select the weight. Use simulations with known truth, repeated
measurements, controlled changes or held-out information where available.

Inspect sensitivity to both spectral and amplitude-smoothness weights. Excessive
smoothing can remove real structure; excessive spectral matching imports errors
from an incorrectly calibrated or differently collected auxiliary spectrum.
Solver-start spread is a separate optimisation diagnostic.

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
*during* retrieval and keeps the spectrum and field an exact Fourier pair, but it can suppress real satellites if the chosen time window is too narrow. It is available on
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
construction, it cannot represent arbitrary bin-to-bin phase variation — a strong regulariser that also makes the
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

Row gains weaken the trace's constraints on spectral amplitude. Delay-dependent
shape still contains information, but weak edge bins can become poorly determined
and acquire spurious energy. A lower residual with extra gains is not evidence of
a more accurate spectrum. The wavelength-density conversion can make a
short-wavelength artefact appear larger than the same frequency-density error.

Inspect {func}`croak.processing.edge_energy_fraction`, the auxiliary spectrum
and the complete residual. Widening the retained band can move the edge taper
onto negligible signal, but does not restore a lost spectral constraint by itself.
Use `phase_only=True` when fixing a trusted amplitude is justified, or a soft
`reg_spectrum` constraint when the trace should retain freedom to adjust it.
COPRA supports phase-only retrieval but not a spectral penalty.

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
