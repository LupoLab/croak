# Grids, wavelength ranges and the SHG carrier

Getting the grid and wavelength ranges right is the most common source of
confusion in FROG retrieval — especially for SHG, whose trace lives at a
*different* carrier than the pulse. This guide collects the rules.

## Three domains

Keep three things distinct:

1. **The raw measurement** — a wavelength axis and a delay/position axis,
   non-uniform, possibly noisy. The delay filtering is done with a non-uniform
   FFT, so an unevenly sampled delay axis is handled correctly without first
   resampling it.
2. **The retrieval grid** — a uniform, centred angular-frequency / time grid
   ({class}`croak.Grid`) that the solvers run on. {func}`~croak.preprocess.load_and_clean`
   builds it from `lam_min`/`lam_max`.
3. **Post-retrieval filtering** — optional windows applied to the *result*
   ({func}`croak.processing.post_filter`), which never change the retrieval itself.

## Building a grid directly

A {class}`croak.Grid` is fixed by its size `n` and one step (`dt` or `domega`); the
matched axes follow from $\Delta\omega = 2\pi/(\Delta t\,N)$:

```python
g = croak.Grid(128, dt=0.3e-15)     # 128 points, 0.3 fs step
g.omega, g.t                        # centred axes (zero in the middle)
```

To size a grid from physical ranges instead, use
{func}`~croak.grid.gridparams_omega` (or {func}`~croak.grid.gridparams_lambda`),
which picks `dt` from the Nyquist limit of the largest frequency offset and `n`
from the required time window, rounded to an FFT-friendly length:

```python
n, dt, omega0 = croak.gridparams_lambda(
    trange=200e-15, lambda_min=720e-9, lambda_max=900e-9)
g = croak.Grid(n, dt=dt)
```

## The baseband convention

croak works with the **baseband** spectrum on a grid centred at zero — the carrier
$\omega_0$ is *not* baked into the pulse shape. It enters only:

- the **dispersive model**, through `omega0` (which builds $\beta(\omega)$);
- the **wavelength axis** of a result, via
  {attr}`RetrievalResult.wavelength <croak.result.RetrievalResult.wavelength>`
  $= 2\pi c/(\omega+\omega_0)$.

So a 2.5 fs pulse at 800 nm and the same envelope at 250 nm share the same
baseband `omega`; they differ only in `omega0`. See
[Conventions](../reference/conventions.md).

## The SHG carrier doubling

Each [interaction](../explanation/interactions.md) records where its signal sits
via `omega0_scale`:

| Interaction | `omega0_scale` | Trace carrier |
|-------------|----------------|---------------|
| SHG | 2.0 | $2\omega_0$ (≈ half the fundamental wavelength) |
| SD, PG | 1.0 | $\omega_0$ (same band as the fundamental) |

`load_and_clean` uses this automatically: it builds the pulse grid from
`lam_min`/`lam_max` (the **fundamental** band you want to retrieve) and places the
*trace* at `omega0_pulse * omega0_scale`. The cleaned trace, the carrier
frequencies and the regridded delays all end up on the
{class}`~croak.preprocess.TraceData`.

:::{admonition} Worked SHG example
:class: tip
A pulse spanning **730–900 nm** produces an SHG-FROG trace near **365–450 nm**.
Pass the *fundamental* range to the retrieval grid:

```python
td = croak.load_and_clean(Ifrog, lam_frog, delays, "shg",
                         lam_min=730e-9, lam_max=900e-9, input_unit="delay")
```

`td.omega0_pulse` is the fundamental carrier; `td.omega0_trace = 2 * omega0_pulse`
is where the measured trace lives. For PG/SD the two coincide.
:::

:::{admonition} Choose the band on a log scale, not a linear one
:class: important
`lam_min`/`lam_max` apply a Planck taper, so anything outside them is gone. A
FROG trace carries real information in its wings — that is where the chirp shows
itself — and they are easy to underestimate: on a linear plot the signal looks
finished long before it is.

Pick the edges from the wavelength marginal at roughly **1e-4 of its peak**
(−40 dB), or read them off a dB image of the trace. On the synthetic measurement
in the [experimental workflow tutorial](../tutorials/04_experimental_workflow.md),
cutting at a few percent of the marginal instead costs a factor of four in the
achievable trace error (0.5 % → 2.1 %) for the sake of ~20 fewer grid points:

```python
marg = trace.sum(axis=1)
keep = np.where(marg > 1e-4 * marg.max())[0]
lam_min, lam_max = lam[keep[0]], lam[keep[-1]]
```

Widening the band also *shrinks* the grid's time step (`dt ≈ 1/Δν`), so it buys
temporal resolution as well as fidelity. The cost is only the extra frequency
points it adds.
:::

## The edge taper, and why the band must be generous

`regrid` does not simply crop at `lam_min`/`lam_max`. It applies a Planck taper
that rolls the trace to **exactly zero** over the outermost
{data}`croak.preprocess.TAPER_COLLAR_BINS` (10) frequency bins at each end. This
is not optional: the forward model FFTs the trace on a periodic grid, so a trace
that does not reach zero at the boundary rings back into the fit.

The consequence is that the outermost bins carry **no measurement**, whatever you
choose for the band. That matters more than losing the data itself, because the
nonlinear mixing still couples the field at those frequencies to the live rows —
just weakly. A probe spike at the grid edge moves the trace error roughly 100×
less than the same spike mid-band, so the optimiser can park a great deal of
energy there for almost no penalty, and since intensity is bounded below by zero
it can only drift upward. See
{func}`croak.processing.edge_energy_fraction` for how to detect the result.

So the band must be wide enough that the taper lands on trace that has genuinely
died away. {func}`~croak.preprocess.load_and_clean` measures this for you and
records it as {attr}`TraceData.taper_loss <croak.preprocess.TraceData.taper_loss>`
— the fraction of the measured trace the taper removed — and warns above 1 %:

```text
UserWarning: the retrieval grid's edge taper removed 38.7% of the measured
trace: lam_min=700.0 nm / lam_max=900.0 nm clip signal that is still present.
```

A band picked from the marginal at 1e-4 typically loses well under 0.1 %. Pass
`taper_warn_level=0.0` to silence the check when a tight band is deliberate.

## Post-retrieval windowing

To clean ringing or out-of-band energy *after* retrieval, apply
{func}`croak.processing.post_filter` — a temporal and/or spectral Planck taper that
returns a new {class}`~croak.result.RetrievalResult` (the original is untouched, so
the effect is reversible):

```python
clean = croak.processing.post_filter(result,
                                    tau_lims=td.tau_lims,    # measurement τm
                                    lam_lims=td.lamm_lims)   # measurement λm
```

Pass the **measurement** windows λm/τm — the ranges where the trace actually
carries signal, stored on the `TraceData` as `td.lamm_lims` / `td.tau_lims`. (The
full retrieval-grid extent, `td.lam_min`/`td.lam_max`, would taper only the
outermost few bins and do nothing useful.)

This is distinct from the [spectral-match penalty](regularisation.md), which
constrains the retrieval itself rather than filtering its output.

## Checklist

- Pass the **fundamental** wavelength band as `lam_min`/`lam_max`, even for SHG.
- Make the band a little wider than the visible signal so the taper has room.
- Set `omega0` on dispersive solvers (or let `retrieve_from_tracedata` take it
  from `td.omega0_pulse`).
- Use `post_filter` for cosmetic cleanup; use [regularisation](regularisation.md)
  to constrain the fit.
