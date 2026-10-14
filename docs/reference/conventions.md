# Conventions

croak is consistent about units, axes and signs. Knowing these removes most of the
friction in using the library.

## SI units everywhere

The core numerical interfaces use SI: time in **seconds**, angular frequency
in **rad/s**, wavelength in **metres**, and thickness in **metres**. Convert
measurement units at the boundary; {func}`croak.io.unit_to_si` supports file
loading. Session controls with explicit suffixes such as `_fs`, `_nm`, `_um`
or `_mm` use the named display unit and are converted before calculation. Dispersion
coefficients are SI too (GDD in s², TOD in s³, FOD in s⁴), though
{class}`~croak.processing.ProcessedResult` *reports* them in the customary fs²/fs³/fs⁴.

Pulse **energy** is in **joules** and the derived **peak power** in **watts**.
The temporal intensity is dimensionless (normalised to a unit peak) unless a pulse
energy is supplied, in which case it becomes absolute power in watts — see
[Post-processing](../howto/postprocessing.md#absolute-power-and-peak-power).

## The baseband, centred grid

A {class}`croak.grid.Grid` holds matched time and angular-frequency axes that are
**centred** — zero frequency (the carrier) sits in the middle of the array, and
$t=0$ in the middle of the time axis. The pulse is represented in **baseband**:
the spectrum $\tilde E(\omega)$ is the envelope relative to the carrier, and the
carrier $\omega_0$ is *not* baked into it.

In API names such as `grid.omega`, `omega` denotes the **baseband** frequency.
In equations that need both, this manual writes $\Omega$ for baseband frequency
and $\omega_{\rm abs}=\omega_0+\Omega$ for absolute optical frequency.
The carrier determines wavelength conversion, material dispersion, chromatic
focal profiles and aperture mapping. It is therefore essential whenever those
effects are included; the SHG signal carrier is $2\omega_0$.

(field-reference-planes)=
## Field reference planes

Distinguish the source field before the instrument, the transverse-integrated
beamlet representation, the on-axis beamlet field at the slab entrance, and an
effective one-dimensional retrieval field. Their spectra and durations can
differ even for the same pulse travelling through the instrument.

For the draft's Airy-beam convention, conversion from the integrated beamlet
spectrum to the model field uses an amplitude factor
$(\omega_{\rm abs}/\omega_0)^{p_f}$. The full on-axis convention uses $p_f=1$;
the reduced effective-field convention uses $p_f=1/2$. The latter is a declared
empirical convention, not a universal broadband power law. The timing split
$p$ is a different quantity.

When comparing models, convert their **complex spectra** to a common field
convention before transforming to time. Converting only a plotted intensity or
comparing stored FWHMs from different conventions does not perform this operation.
State temporal interpolation, normalisation and alignment alongside duration.
See [simulated data](../howto/loading_simulated.md) and [validation](../explanation/validation.md).

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
({func}`croak.metrics.compute_R`), reported as `result.error`. croak divides the weighted residual norm by the weighted data norm.
Some FROG references use a peak-normalised error $G$ instead; compare definitions
before comparing numerical values. See [Retrieval as least squares](../explanation/retrieval_theory.md)
and the [glossary](glossary.md).

## Ambiguities

Retrieved pulses are defined only up to the geometry's
[trivial ambiguities](../explanation/pnps_framework.md#ambiguities) — an absolute
phase, an absolute time origin, and (for SHG) the direction of time. Compare
results modulo these.

The direction-of-time ambiguity matters for phase interpretation because it
also flips the sign of the spectral phase: the reported GDD and TOD of an SHG
retrieval are only meaningful once the direction is settled. Which interactions
carry it is declared by
{attr}`Interaction.time_reversal_ambiguous <croak.interactions.Interaction.time_reversal_ambiguous>`
(SHG only), and when a known pulse is available
{func}`~croak.processing.resolve_time_direction` can compare the temporal
intensity branches. A symmetric intensity cannot settle the sign; see
[Post-processing](../howto/postprocessing.md#shg-settle-the-direction-of-time-first).
