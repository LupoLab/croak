# Fitting the medium thickness and delay-zero

Two quantities are usually *fixed* before a retrieval but only ever *estimated*:

- the **dispersive-slab thickness** $L$ of the forward model
  ({class}`~croak.forward.ForwardModel`'s `material`/`thickness`), and
- the **delay-zero** $\tau_0$ — the point on the delay axis where the two pulse
  replicas overlap, set during preprocessing by the delay-marginal peak.

Both can instead be **retrieved** as extra scalar parameters alongside the
complex pulse, with the autodiff least-squares/gradient solvers `lbfgs-ad`, `lm`
and `lm-optx`, and with the global `cma-es` search. Because the JAX forward model
is differentiable in $L$ and $\tau_0$, the gradient/Jacobian comes for free — no
hand-derived adjoints (and `cma-es` is gradient-free anyway).

`lm-optx` is the on-device Optimistix twin of `lm`; it runs an **unconstrained**
Levenberg–Marquardt, so it cannot hold the fitted thickness non-negative and
relies on starting from a good prior — use it as a polish (or warm start), as
recommended below for thickness in any case.

`cma-es` appends the extras to its global search vector as O(1)-scaled offsets.
The search is unconstrained (no thickness lower bound) and has **no two-phase
polish** — it frees the extras from the start. Because thickness wants a good
prior, the usual recipe is to fit $\tau_0$ globally here, and fit thickness by
refining a pulse-only `cma-es` result with `lbfgs-ad` (`polish=True`).

```python
res = croak.retrieve(
    trace, g.omega, delays, "pg",
    algorithm="lm",
    material="SiO2", thickness=250e-6, npoints=20, omega0=omega0,
    fit_thickness=True,   # fit the slab thickness
    fit_tau0=True,        # fit the delay-zero offset
    polish=True,          # two-phase polish (see below)
)
print(res.thickness, res.tau0)   # fitted values (m, s)
```

## When each one helps

### Delay-zero $\tau_0$ — useful for every interaction

`fit_tau0` adds a single scalar that slides the simulated trace along the delay
axis: the gate delay phase becomes $e^{i\omega(\tau-\tau_0)}$. It is **not** a
trivial gauge of the pulse — translating the pulse in time leaves the trace
invariant, whereas a delay-axis offset genuinely moves it (for SHG, it moves the
trace's symmetry centre). So a mis-set time zero raises the trace error, and
fitting $\tau_0$ removes it.

A $\tau_0$ error produces a characteristic **red/blue tilt** in the residual:
because different colours peak at different delays in a chirped pulse, a delay
offset leaves an antisymmetric, frequency-correlated residual. If you see that
signature, fit $\tau_0$. The preprocessing marginal-peak centering is only
accurate to about one delay pixel; `fit_tau0` refines it to sub-pixel precision.

Sign convention: a positive `res.tau0` means the true delay-zero of the measured
trace lies at $+\tau_0$ (the simulated gate uses $e^{i\omega(\tau-\tau_0)}$).

### Thickness $L$ — only for the dispersive PG/SD slab

`fit_thickness` is only identifiable — and only allowed — for the **dispersive
depth-integral slab**, which croak supports for the **PG and SD** interactions.
There the nonlinear interaction happens throughout the slab (a depth quadrature),
so $L$ changes the *shape* of the trace, not merely a spectral phase.

For **SHG** a slab reduces to a pure spectral phase on the input pulse, which a
freely retrieved phase absorbs completely — fitting $L$ would be a flat direction
that does nothing, so it is **rejected with an error**. (This is the same reason
croak forbids dispersive SHG propagation in the first place.)

Even within PG/SD, the *bulk-chirp* part of $L$ is partly degenerate with the
retrieved phase; only the depth-distributed part constrains it. Thickness is
therefore best fitted as a **polish from a good prior**, and is well determined
only when the slab is dispersive enough that it visibly shapes the trace. With a
weakly dispersive slab the direction is nearly flat and $L$ will barely move —
that is the model honestly telling you the data does not constrain it.

## Triggering the fit: warm-start vs two-phase polish

The joint pulse + extra-parameter landscape is harder than the pulse alone, so
fit the extras from a **good pulse estimate**, not a cold start. Two ways:

1. **Warm start (default).** Run a normal retrieval first, then re-run with the
   extra flags enabled, seeding from the previous result. In the API pass the
   first result as `guess=`; in the GUI pick *Reuse previous result* from the
   **Initial guess** selector together with the fit options.

   A warm start seeds the **extras** as well as the pulse. The extras are
   optimised as offsets from a *centre* — `thickness` for $L$, `tau0` for
   $\tau_0$, `smear_scale` for the smearing width — so passing the previous
   fit's values makes the re-run continue from where the last one stopped rather
   than restarting from the priors (a spectrum that was retrieved jointly with a
   non-zero $\tau_0$ is otherwise re-fitted against a differently-referenced
   delay axis). The GUI does this for you: *Reuse previous result* seeds every
   extra that is **still being fitted**, and says which in the status line. An
   extra whose fit box you have unticked keeps the value you set — an unticked
   box means "hold this where I put it", and the model is evaluated there.

   ```python
   first = croak.retrieve_from_tracedata(td, algorithm="lbfgs-ad", ...)
   refined = croak.retrieve_from_tracedata(
       td,
       algorithm="lbfgs-ad",
       guess=first.spectrum,
       thickness=first.thickness,     # centres for the second fit
       tau0=first.tau0,
       fit_thickness=True,
       fit_tau0=True,
       material="SiO2",
   )
   ```

2. **Two-phase polish (`polish=True`).** A single call that internally retrieves
   the pulse with the extras held fixed, then frees them for a joint final
   phase. Convenient when the fixed-extras pulse is already a reasonable fit (it
   is for a thickness prior; less so for a large $\tau_0$ error, where a warm
   start is better).

## Uncertainty on the fitted parameters

When you fit $L$ or $\tau_0$, the analytic covariance can report their standard
errors directly — the rigorous alternative to the substrate-thickness
{func}`~croak.uncertainty.thickness_bootstrap` systematic (use one or the other,
not both):

```python
from croak.covariance import parameter_covariance

cov = parameter_covariance(
    res, trace, noise=0.01,
    material="SiO2", thickness=res.thickness, npoints=20, omega0=omega0,
    fit_thickness=True, fit_tau0=True,
)
print(cov.sigma_thickness, cov.sigma_tau0)   # σ in m, s
```

The thickness/$\tau_0$ columns are appended to the Jacobian and re-linearised
about the fitted values, so the pulse FWHM covariance (and
{func}`~croak.covariance.covariance_uncertainty`) automatically inflates to
include the extra parameters' uncertainty and their correlation with the pulse.

## In the GUI

The retrieve stage has an **Extra fit parameters** group with *Fit medium
thickness*, *Fit τ₀ delay offset*, *Fit smearing width* and *Two-phase polish*.
*Fit smearing width* is enabled only when *Enable geometrical smearing* is on
(see [Geometric smearing](geometric_smearing.md)). *Fit medium thickness*
is enabled only when *Enable dispersive propagation* is on and the chosen solver
supports it (`lbfgs-ad` / `lm` / `lm-optx`); the fitted values are shown in the result
summary. The thickness fit starts from the *Thickness (µm)* value in the
*Dispersive propagation* group.

## Summary

| Option | Solvers | Requirements | Notes |
|---|---|---|---|
| `fit_tau0` | `lbfgs-ad`, `lm`, `lm-optx`, `cma-es` | any interaction | sub-pixel time-zero; removes red/blue residual tilt |
| `fit_thickness` | `lbfgs-ad`, `lm`, `lm-optx`, `cma-es` | dispersive slab, PG/SD | identifiable only for a genuinely dispersive slab (`lm-optx`/`cma-es` are unconstrained — use a good prior) |
| `polish` | `lbfgs-ad`, `lm`, `lm-optx` | an extra enabled | fit pulse first, then free the extras (not `cma-es`) |
| `fit_smearing` | `lbfgs-ad`, `lm`, `lm-optx`, `cma-es` | a smearing kernel, PG/SD | one multiplier on the kernel widths; never fit alongside `fit_thickness` — see [Geometric smearing](geometric_smearing.md) |

## The delay origin without a fitted τ₀: `delay_origin="marginal_peak"`

The preprocessing centres a measured trace on its delay-marginal peak, but the
forward model puts delay zero at gate–probe coincidence. For a delay-symmetric
trace those coincide; for a coherently collected single-cycle trace they do not
(tens of attoseconds, growing with the chirp accumulated in the medium), and a
retrieval with the origin held at coincidence can only absorb the misalignment
by broadening the pulse. Rather than fitting `tau0`, which on a chirped pulse
wanders by hundreds of attoseconds and trades against the chirp,

```python
res = croak.retrieve(trace, omega, delays, "pg", algorithm="lbfgs-ad",
                     delay_origin="marginal_peak", ...)
```

re-centres every model trace on its own marginal peak by the same sub-sample
rule the preprocessing used — one convention on both sides, no free parameter.
It needs a uniform delay axis (the retrieval grid is). If you do fit `tau0`,
`tau0_bound` (seconds; `RetrieveParams.tau0_bound_fs`) boxes it.
