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
angular-frequency grid (zero at the carrier). The carrier $\omega_0$ fixes absolute frequency for dispersion, wavelength
conversion and spatial instrument models. See [Conventions](../reference/conventions.md).
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

The **transient-grating** (TG-FROG) geometry used in
[validation](validation.md), with coincident gate replicas, shares the **PG** signal kernel
$E\lvert G\rvert^2$, so retrieving TG-FROG data uses `interaction="pg"`. See
[Interactions](interactions.md) for the signal fields and their adjoints.

## The standard idealisations, and where they fail

The simplest model treats the medium as thin, the replicas as spatially
uniform, and the detector response as a frequency-dependent scale. Check these
assumptions against the pulse bandwidth and instrument:

1. **Finite dispersive medium.** The pulse evolves while the nonlinear signal
   is generated, and contributions from different depths interfere. In the DUV,
   even micrometre-scale substrates can matter for few-femtosecond pulses.
2. **Crossed focused beams.** Relative arrival times vary across the focal
   spot. The resulting trace depends on the mask geometry, pulse and focusing;
   a reduced smearing model approximates this dependence under stated assumptions.
3. **Finite collection.** An aperture selects a coherent part of the signal.
   Its effect can depend on delay as well as frequency, so a row-wise spectral
   calibration need not absorb it.

The [forward-model chapter](forward_model.md) develops these models and their
limits. The [validation study](validation.md) shows bounded examples in which a
small trace residual coexists with a substantially wrong pulse. It also includes
cases that remain poorly retrieved with a more detailed model.

## Ambiguities

The map from pulse to trace is not perfectly one-to-one. Every FROG geometry
carries *trivial ambiguities* that retrieval cannot resolve from the trace alone:

- an absolute **phase offset** $\tilde E \to \tilde E\,e^{i\phi_0}$;
- an absolute **time/delay origin** (a linear spectral phase);
- for SHG-FROG, the **direction of time** ($E(t) \to E^*(-t)$).

croak's results inherit these: compare retrieved pulses up to a constant phase and
a time shift. SD, PG and TG do not have the general SHG time-reversal symmetry, although
particular fields or incomplete data can remain difficult to distinguish.
Several [random starts](../howto/solver_selection.md) test sensitivity to local
minima; they do not prove uniqueness or remove an information ambiguity.

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
