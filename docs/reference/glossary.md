# Glossary and notation

A quick map between the symbols used in the [theory pages](../explanation/pnps_framework.md)
and the names used in the code.

## Notation

| Symbol | Meaning | In the code |
|--------|---------|-------------|
| $\tilde E(\omega)$ | complex pulse spectrum (the unknown) | `spectrum`, `ew` |
| $E(t)$ | complex temporal field | `field`, `et` |
| $\omega$ | centred angular frequency (rad/s) | `omega`, `g.omega` |
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
The normalised RMS trace error is written **$G$** (the "G-error") in much of the
FROG literature and **$R$** throughout croak. They are the same quantity
({func}`croak.metrics.compute_R`, `result.error`).
```

## Terms

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
: **Transient-Grating FROG**; for a single unknown pulse its signal reduces to the
  **PG** kernel, so croak retrieves it with `interaction="pg"`.

Test / gate field
: the undelayed field $E$ and the delayed field $G(t)=E(t-\tau)$ entering an
  interaction.

Trace error $R$
: the normalised, dimensionless fit metric; $\lesssim 10^{-3}$ is excellent,
  $10^{-3}$–$10^{-2}$ is a good experimental retrieval.

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
