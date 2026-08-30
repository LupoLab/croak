# Conventions

croak is consistent about units, axes and signs. Knowing these removes most of the
friction in using the library.

## SI units everywhere

Every quantity is SI: time in **seconds**, angular frequency in **rad/s**,
wavelength in **metres**, thickness in **metres**. There are no femtoseconds or
nanometres inside the library — convert at the boundary
({func}`croak.io.unit_to_si` does this when loading files). Dispersion
coefficients are SI too (GDD in s², TOD in s³, FOD in s⁴), though
{class}`~croak.processing.ProcessedResult` *reports* them in the customary fs²/fs³/fs⁴.

Pulse **energy** is in **joules** and the derived **peak power** in **watts**.
The temporal intensity is dimensionless (normalised to a unit peak) unless a pulse
energy is supplied, in which case it becomes absolute power in watts — see
[Post-processing](../howto/postprocessing.md#absolute-power-and-peak-power).

## The baseband, centred grid

A {class}`croak.Grid` holds matched time and angular-frequency axes that are
**centred** — zero frequency (the carrier) sits in the middle of the array, and
$t=0$ in the middle of the time axis. The pulse is represented in **baseband**:
the spectrum $\tilde E(\omega)$ is the envelope relative to the carrier, and the
carrier $\omega_0$ is *not* baked into it.

`omega0` enters only two places:

- the [dispersive forward model](../explanation/forward_model.md), where it builds
  the propagation constant $\beta(\omega)$;
- the **wavelength axis** of a result,
  $\lambda = 2\pi c/(\omega+\omega_0)$.

So the same baseband envelope describes an 800 nm pulse or a 250 nm pulse; only
`omega0` differs.

## Centred vs DFT-bin order

Public arrays — spectra, fields, cotangents — are always in **centred** order.
Internally the [forward model](../explanation/forward_model.md) works in raw
DFT-bin order (zero frequency at index 0) for efficiency, shifting at its
boundary. You only meet DFT-bin order if you call the low-level
{func}`croak.grid.raw_fft` / {func}`croak.grid.raw_ifft`.

## Fourier convention

croak uses the **physics convention** with the $1/2\pi$ on the inverse transform
and a $+i\omega t$ forward kernel:

```{math}
\tilde E(\omega) = \int E(t)\, e^{+i\omega t}\,\mathrm{d}t,
\qquad
E(t) = \frac{1}{2\pi}\int \tilde E(\omega)\, e^{-i\omega t}\,\mathrm{d}\omega.
```

{meth}`Grid.fft <croak.grid.Grid.fft>` and {meth}`Grid.ifft <croak.grid.Grid.ifft>`
implement this on centred arrays, scaled by $\Delta t$ and $\Delta\omega/2\pi$ so
the discrete transform approximates the continuous integral. See the
[FFT convention](fft_convention.md) page for the full detail.

## Trace orientation

Traces are `(Nomega, Ndelay)` — frequency on axis 0, delay on axis 1. The
preprocessing arrays use the same orientation with wavelength on axis 0,
`(Nlambda, Ndelay)`. Solvers, plotters and metrics all assume this layout.

## The trace error R

The fit quality is the normalised, dimensionless **trace error $R$**
({func}`croak.metrics.compute_R`), reported as `result.error`. It is the standard
FROG error — written $G$ in much of the literature; croak uses the symbol $R$
throughout. See [Retrieval as least squares](../explanation/retrieval_theory.md)
and the [glossary](glossary.md).

## Ambiguities

Retrieved pulses are defined only up to the geometry's
[trivial ambiguities](../explanation/pnps_framework.md#ambiguities) — an absolute
phase, an absolute time origin, and (for SHG) the direction of time. Compare
results modulo these.

The direction-of-time ambiguity is the one that bites in practice, because it
also flips the sign of the spectral phase: the reported GDD and TOD of an SHG
retrieval are only meaningful once the direction is settled. Which interactions
carry it is declared by
{attr}`Interaction.time_reversal_ambiguous <croak.interactions.Interaction.time_reversal_ambiguous>`
(SHG only), and when a known pulse is available
{func}`~croak.processing.resolve_time_direction` settles it — see
[Post-processing](../howto/postprocessing.md#shg-settle-the-direction-of-time-first).
