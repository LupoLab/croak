# The PNPS framework

This page introduces the formalism that croak is built on: **parameterised
nonlinear-process spectra** (PNPS). FROG is the most familiar member of this
family, but the same mathematics describes time-domain ptychography (TDP),
dispersion scan (d-scan) and related self-referenced techniques.

## Self-referenced pulse measurement

A photodetector or spectrometer only records *intensity*; the spectral **phase**
of an ultrashort pulse is lost. Self-referenced techniques recover it by gating
the pulse with a (delayed, dispersed, …) replica of *itself* through a nonlinear
interaction, then spectrally resolving the result. Scanning a parameter — the
delay $\tau$ in FROG — builds a two-dimensional **trace** that encodes both the
amplitude and the phase.

## The trace model

Let $\tilde E(\omega)$ be the complex pulse spectrum we wish to retrieve. A
nonlinear process produces a **signal operator** $S_\delta[\tilde E](t)$ — the
time-domain signal field for parameter value $\delta$. The measured trace is the
squared magnitude of its spectrum:

```{math}
:label: pnps
T(\delta, \omega;\, \tilde E) \;=\; \bigl|\,\mathcal{F}\{S_\delta[\tilde E](t)\}(\omega)\,\bigr|^2 .
```

In FROG, $\delta$ is the delay $\tau$ and {eq}`pnps` is a 2-D spectrogram in
$(\omega, \tau)$. This is the quantity croak's {class}`~croak.forward.ForwardModel`
computes; see [The forward model](forward_model.md) for the discrete,
dispersive form.

```{note}
croak works with the **baseband** spectrum $\tilde E(\omega)$ on a centred
angular-frequency grid (zero at the carrier). The carrier $\omega_0$ enters only
through the dispersive model. See [Conventions](../reference/conventions.md).
```

## Signal operators for FROG geometries

The signal operator is built from a *test* field $E$ (undelayed) and a *gate*
field $G(t) = E(t-\tau)$ (delayed). croak implements three standard geometries,
each a different instantaneous nonlinearity in the time domain:

| Geometry | Signal $s(t)$ | croak name | Trace carrier |
|----------|---------------|-----------|---------------|
| Second-harmonic generation | $E\,G$ | `"shg"` | $2\omega_0$ |
| Self-diffraction | $E^2\,G^*$ | `"sd"` | $\omega_0$ |
| Polarisation gating | $E\,\lvert G\rvert^2$ | `"pg"` | $\omega_0$ |

The **transient-grating** (TG-FROG) geometry that headlines the
[companion paper](validation.md) shares the **PG** signal kernel
$E\lvert G\rvert^2$, so retrieving TG-FROG data uses `interaction="pg"`. See
[Interactions](interactions.md) for the signal fields and their adjoints.

## The standard idealisations, and where they fail

Classic FROG retrieval idealises the measurement in three ways, and each fails
for few-femtosecond pulses in the deep ultraviolet:

1. **The medium is treated as infinitely thin and non-dispersive** — the
   signal {eq}`pnps` is evaluated at a single plane. Accurate in the
   near-infrared; in the DUV even a few micrometres of substrate impose
   group-velocity and higher-order dispersion that reshape the pulse *within*
   the interaction region, plus wavelength-dependent phase-matching filtering
   over the finite interaction length. A thin-medium retrieval of a ~1 fs DUV
   pulse returns a pulse up to 2.6 times too long — at a low trace error.
2. **The beam geometry is treated as ideal.** In any focused non-collinear
   arrangement the relative delay between the beams varies across the focal
   spot, smearing the trace along the delay axis. At 260 nm and practical
   BOXCARS mask ratios the blur is about a femtosecond — comparable to the
   pulse.
3. **The collection is treated as perfect.** The signal reaches the
   spectrometer through an aperture, and for an octave-spanning pulse that
   aperture is a coherent chromatic filter that reshapes the trace in a way
   no spectral calibration absorbs.

croak's [forward model](forward_model.md) contains all three effects — the
dispersive slab, the geometric smearing kernel, and the chromatic
focal/collection model — each recoverable as an explicit limit, so the
standard model is the same code with the extra physics switched off. The
[validation study](validation.md) measures where each ingredient matters.

## Ambiguities

The map from pulse to trace is not perfectly one-to-one. Every FROG geometry
carries *trivial ambiguities* that retrieval cannot resolve from the trace alone:

- an absolute **phase offset** $\tilde E \to \tilde E\,e^{i\phi_0}$;
- an absolute **time/delay origin** (a linear spectral phase);
- for SHG-FROG, the **direction of time** ($E(t) \to E^*(-t)$).

croak's results inherit these: compare retrieved pulses up to a constant phase and
a time shift. Some geometries (SD, PG, TG) break the time-reversal ambiguity.
*Non-trivial* ambiguities (genuinely distinct pulses with the same trace) are
rare for well-sampled traces and are the reason retrieval is run from several
[random restarts](../howto/solver_selection.md).

## Beyond FROG

Because the only geometry-specific ingredient is the signal operator
$S_\delta$, the framework extends to any parameterised nonlinear process — d-scan
(where $\delta$ is inserted glass), TDP (where $\delta$ is a ptychographic
position), and others — by swapping the operator while reusing the same
least-squares machinery. croak currently ships the three FROG kernels above.

## Further reading

- [Interactions](interactions.md) — the signal fields and Wirtinger adjoints.
- [The forward model](forward_model.md) — thin and dispersive trace synthesis.
- [Retrieval as least squares](retrieval_theory.md) — the objective and error metric.
- N. C. Geib *et al.*, "Common pulse retrieval algorithm," *Optica* **6**, 495 (2019).
- R. Trebino, *Frequency-Resolved Optical Gating: The Measurement of Ultrashort Laser Pulses* (Springer, 2000).
