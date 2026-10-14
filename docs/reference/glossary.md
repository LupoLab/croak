# Glossary and notation

A quick map between the symbols used in the [theory pages](../explanation/pnps_framework.md)
and the names used in the code.

## Notation

| Symbol | Meaning | In the code |
|--------|---------|-------------|
| $\tilde E(\omega)$ | complex pulse spectrum (the unknown) | `spectrum`, `ew` |
| $E(t)$ | complex temporal field | `field`, `et` |
| $\Omega$ (often $\omega$ in the API) | centred frequency offset (rad/s) | `omega`, `g.omega` |
| $\omega_{\rm abs}=\omega_0+\Omega$ | absolute optical angular frequency (rad/s) | `omega + omega0` |
| $\omega_0$ | carrier angular frequency | `omega0` |
| $t$ | time (s) | `t`, `g.t` |
| $\tau$ | delay (s) | `tau`, `delays` |
| $s(E,G)$ | instantaneous nonlinear signal | `Interaction.signal` |
| $\psi(\omega,\tau)$ | frequency-domain signal for one delay | `signal_single` |
| $T(\omega,\tau)$ | trace intensity $\lvert\psi\rvert^2$ | `trace`, `t_meas`/`t_sim` |
| $\mu$ | optimal intensity scale factor | `mu`, `compute_mu` |
| $w_\omega$ | per-frequency weights | `weights` |
| $r$ | weighted residual sum of squares | `compute_r` |
| $R$ | normalised trace error ("FROG error") | `error`, `compute_R` |
| $\beta(\omega)$ | slab propagation constant | `materials.beta` |
| $\bar v = \partial L/\partial v^*$ | Wirtinger cotangent | `*_bar`, `psi_bar`, `ew_bar` |
| $z_q, w_q$ | depth quadrature nodes/weights | `quadrature_nodes_weights` |

```{admonition} R vs G
:class: note
The peak-normalised rms trace error
$R=\sqrt{r/(MN\,\max(T_\text{meas}w)^2)}$ is written **$G$** (the "G-error")
in much of the FROG literature and **$R$** throughout croak. They are the same
quantity ({func}`croak.metrics.compute_R`, `result.error`); with weights, croak
uses the weighted residual and the peak of the weighted measured trace.
```

## Terms

croak
: **Complete Retrieval by Optimisation and Adjoint Kernels** — the package
  for retrieving a complex pulse from a model of its FROG measurement.

PNPS
: **Parameterised Nonlinear-Process Spectra** — the family of self-referenced
  measurements (FROG, TDP, d-scan, …) unified by the
  [trace model](../explanation/pnps_framework.md). The scanned parameter is the
  delay $\tau$ in FROG.

FROG
: **Frequency-Resolved Optical Gating** — a delay-scanned nonlinear spectrogram;
  the canonical PNPS technique.

Interaction / geometry
: the nonlinear process defining the signal $s(E,G)$ — SHG, SD or PG in croak. See
  [Interactions](../explanation/interactions.md).

TG-FROG
: **Transient-Grating FROG**; for identical coincident gate replicas in the local model, its signal
  reduces to the **PG** kernel, so croak retrieves it with `interaction="pg"`.

Test / gate field
: the undelayed field $E$ and the delayed field $G(t)=E(t-\tau)$ entering an
  interaction.

Trace error $R$
: the normalised, dimensionless residual between measured and fitted traces.
  Interpret it relative to noise, weights, preprocessing and the model. A small
  value does not by itself establish field accuracy.

Thin medium
: the classic FROG approximation of an infinitely thin, dispersionless nonlinear
  medium — the special case of the [dispersive model](../explanation/forward_model.md)
  with $\beta=0$ and zero thickness.

Dispersive slab / D-COPRA
: retrieval that models propagation inside a finite, dispersive medium; "D-COPRA"
  is {class}`~croak.copra.COPRA` with a `material` set.

Wirtinger cotangent
: $\bar v \equiv \partial L/\partial v^*$, the conjugate derivative that
  gradient descent on a real loss of a complex variable requires. See
  [Gradients](../explanation/gradients.md).

Phase-only retrieval
: fixing the spectral amplitude (to a measured spectrum) and retrieving only the
  phase (`phase_only=True`).

`R_omega`
: per-frequency intensity scaling — the scale factor $\mu$ becomes a vector
  $\mu_\omega$. See [Regularisation](../howto/regularisation.md).

B-spline phase basis
: representing the spectral phase by the control points of a cubic B-spline over
  the spectrum support (`phase_basis="bspline"`, `n_nodes` nodes) instead of one
  value per frequency bin — a low-dimensional, intrinsically smooth
  parameterisation. See [Regularisation](../howto/regularisation.md#the-b-spline-phase-basis)
  and {func}`croak._jax_pulse.make_spline_phase_basis`.

CMA-ES / evolution strategy
: **Covariance-Matrix-Adaptation Evolution Strategy** — a derivative-free,
  population-based *global* optimiser ({class}`~croak.cmaes.CMAES`,
  `algorithm="cma-es"`, via the optional evosax backend) used to escape poor local
  minima. See [global retrieval](../explanation/algorithms.md#global-retrieval-cma-es).

Temporal regularisation
: a penalty (`reg_time`, with a `time_window` in seconds) on the fraction of pulse
  energy falling outside a time window, suppressing satellites and pedestals during
  retrieval. See [Regularisation](../howto/regularisation.md#temporal-regularisation).

Chirped mirror
: a multilayer mirror that imparts a designed, wavelength-dependent group delay per
  reflection. Registered into {data}`croak.dispersion.MIRRORS` and applied by number
  of bounces; see [Materials and chirped mirrors](../howto/materials_and_mirrors.md).

Field reference
: the plane and spatial definition of the reported complex field. Source,
  transverse-integrated beamlet, effective and on-axis entrance fields can have
  different amplitudes and temporal profiles. See [conventions](conventions.md).

$p$, $\delta$
: the two timing coordinates of the spatial (smearing) model: the gate-pair
  separation and the probe-to-gate-centroid offset ({mod}`croak.smearing`). For
  PG the two gate replicas arrive at $\tau-\delta\mp p/2$ in the probe frame;
  for SD, $p$ separates the two unconjugated replicas and $\delta$ offsets the
  conjugated arm (see [the forward model](../explanation/forward_model.md#geometric-smearing)).

$p_f$
: the exponent of a declared spectral field conversion
  $\widetilde E_{p_f}=(\omega_{\rm abs}/\omega_0)^{p_f}\widetilde E_{\rm int}$,
  referenced to the carrier $\omega_0$.
  It labels a field convention, not a detector-response correction.

Detector response
: wavelength-dependent conversion from optical signal to recorded counts.
  Distinguish it from the nonlinear source factor and the coordinate Jacobian
  used to convert a density per wavelength to one per frequency.
