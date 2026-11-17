# FROG trace marginals as consistency checks

The **frequency marginal** of a FROG-type trace is its integral over delay,

```{math}
M(\omega) \;=\; \int T(\omega,\tau)\,\mathrm{d}\tau ,
```

a one-dimensional function of optical frequency. It is computed from a measured
trace as `M_meas = trace.sum(axis=1)` (delay is axis 1 in croak's `(Nω, Nτ)`
convention). From an **independently measured fundamental spectrum**
$S(\omega)=|\tilde E(\omega)|^2$ we can predict $M(\omega)$ — or rigorously bound
it — and compare, *before* running a retrieval. A mismatch is a cheap
quality-control (QC) signal: it flags calibration errors, spectral-response
problems, scatter and detector nonlinearity that a retrieval would otherwise
silently absorb.

These checks are implemented in {mod}`croak.marginal_checks` and surfaced in the
GUI's **Marginal check** stage (see the [GUI guide](../howto/gui.md)); the
how-to recipe is [Marginal consistency checks](../howto/marginal_checks.md).

```{important}
The identities below assume a thin, local interaction with coincident replicas,
complete delay support, a matching fundamental-spectrum reference plane and a
calibrated spectral response. Dispersion, split gates, chromatic spatial fields,
finite collection and cropping can invalidate these ideal-model predictions.
A mismatch is then evidence to investigate, not proof of a detector fault.
The transform-limited PG width is a **soft reference**, not a universal upper bound.
```

## The general result

For the ideal local kernels considered here, the signal factorises into an *undelayed* part $P$ and a *delayed
gate* $g$,

```{math}
E_\mathrm{sig}(t,\tau) = P(t)\,g(t-\tau),
\qquad T(\omega,\tau)=\bigl|\widehat{E_\mathrm{sig}}(\omega,\tau)\bigr|^2 .
```

Integrating over delay collapses the delay-dependent phase to a delta function,
leaving a **convolution of spectral intensities**:

```{math}
M(\omega) \;\propto\; \bigl(|\tilde P|^2 * |\tilde g|^2\bigr)(\omega),
```

where $\tilde P,\tilde g$ are the Fourier transforms of $P,g$. Everything below
follows from this one identity.

Two consequences are **exact for any nonnegative factors** (no assumptions on the
pulse), because intensity-weighted moments add under convolution:

- **Centroid:** $\bar\omega[M] = \bar\omega[\,|\tilde P|^2\,] + \bar\omega[\,|\tilde g|^2\,]$
- **Variance:** $\sigma^2[M] = \sigma^2[\,|\tilde P|^2\,] + \sigma^2[\,|\tilde g|^2\,]$

with $\bar\omega[f]=\frac{\int\omega f}{\int f}$ and
$\sigma^2[f]=\frac{\int(\omega-\bar\omega)^2 f}{\int f}$
({func}`croak.marginal_checks.centroid` and
{func}`croak.marginal_checks.rms_width`).

**Phase-independence criterion.** A factor is phase-independent iff it is built
from $E$ or $E^\ast$ alone:

| factor | expression | phase-dependent? | centroid | even? |
|---|---|---|---|---|
| $\lvert\tilde E\rvert^2$ | $S(\omega)$ | no | $\bar\omega_S$ | no |
| $\lvert\widehat{E^\ast}\rvert^2$ | $S(-\omega)$ | no | $-\bar\omega_S$ | no |
| $\lvert\widehat{\lvert E\rvert^2}\rvert^2$ | intensity spectrum | **yes** | $0$ (exact) | **yes** |
| $\lvert\widehat{E^2}\rvert^2$ | SHG spectrum | **yes** | phase-dependent | no |

The marginal is fully phase-independent only when *both* factors come from the
top two rows. Among the common geometries that is **SHG only**.

| process | $P$ | $g$ | marginal $M\propto$ | phase-indep.? | centred at |
|---|---|---|---|---|---|
| SHG | $E$ | $E$ | $S * S$ | **yes** | $2\bar\omega_S$ |
| PG  | $E$ | $\lvert E\rvert^2$ | $S * \lvert\widehat{\lvert E\rvert^2}\rvert^2$ | no | $\bar\omega_S$ |
| SD  | $E^2$ | $E^\ast$ | $\lvert\widehat{E^2}\rvert^2 * S(-\omega)$ | no | $\approx\bar\omega_S$ |
| TG  | $E$ | $\lvert E\rvert^2$ | same as PG | no | $\bar\omega_S$ |

```{admonition} TG-FROG ≡ PG
:class: tip
The transient-grating geometry has the $|E|^2$ gate, so its marginal is
identical to PG regardless of which beam carries the delay (convolution
commutes). {func}`croak.marginal_checks.predicted_marginal` accepts
`interaction="tg"` and treats it as `"pg"`.
```

## SHG — the clean case

```{math}
M_\mathrm{SHG}(\omega) = (S * S)(\omega).
```

Both factors are the measured spectrum, so the marginal is the **autoconvolution
of the fundamental spectrum** ({func}`croak.marginal_checks.spectral_autoconvolution`)
— completely phase-independent and computable from $S$ alone. This is the
phase-independent check under the thin-local-model and complete-support
assumptions above. Disagreement can indicate calibration, spectral response,
noise, incomplete support or a model that is inappropriate for the instrument.
The three identities are:

- **Full predicted marginal:** $S*S$, compared shape-for-shape.
- **Centroid:** $\bar\omega[M] = 2\,\bar\omega_S$.
- **RMS width:** $\sigma[M] = \sqrt{2}\,\sigma_S$.

Place the prediction on the second-harmonic frequency axis and compare to the
measured marginal after consistent wavelength calibration.

### Through a nonlinear crystal

In croak's crystal model ({doc}`forward_model`) the second harmonic is generated
throughout the crystal and each component pair carries its own phase matching. The
marginal is still phase-independent, and it follows from the model's signal in
four steps. Frequencies here are offsets from the carriers, $\Omega$ from
$\omega_0$ for a fundamental and from $2\omega_0$ for the signal, and $\tilde E$ is
the fundamental's spectrum, with $S = \lvert\tilde E\rvert^2$.

**1. The signal of one pair.** The model propagates both arms to depth $z$
($e^{i\beta_\mathrm{in}z}$), multiplies them, and carries what is generated there to
the exit face ($a(\Omega)\,e^{i\beta_\mathrm{sig}(P-z)}$), integrating over the path
$P$. The gate arm carries the delay as $e^{i\Omega\tau}$. Crossing in the principal
plane, the arms also drift apart in delay with depth, at the rate $\kappa$
({meth}`~croak.crystals.CrystalSetup.arm_shift`): the test arm carries
$e^{i\kappa\Omega(P/2-z)}$ and the gate its inverse ($\kappa = 0$ otherwise).
A product in time is a convolution in frequency, so with $\Omega_1$ the test
component and $\Omega_2 = \Omega-\Omega_1$ the gate component, up to the
transforms' constant factors,

```{math}
\psi(\Omega,\tau) = a(\Omega)\,e^{i\beta_\mathrm{sig}(\Omega)P}\int
\tilde E(\Omega_1)\,\tilde E(\Omega_2)\,e^{i\Omega_2\tau}\,
e^{i\kappa(\Omega_1-\Omega_2)P/2}
\int_0^P e^{i q z}\,\mathrm dz\;\mathrm d\Omega_1 ,
```

with every depth-dependent phase collected in

```{math}
q(\Omega_1,\Omega_2) = \beta_\mathrm{in}(\Omega_1) + \beta_\mathrm{in}(\Omega_2)
- \beta_\mathrm{sig}(\Omega_1+\Omega_2) - \kappa(\Omega_1-\Omega_2).
```

**2. The frame drops out.** Both propagation constants are written in the frame
moving at the fundamental's group velocity, $\beta_\mathrm{in}(\Omega) =
k_z(\omega_0+\Omega) - \beta_1\Omega$ and $\beta_\mathrm{sig}(\Omega) =
k_s(2\omega_0+\Omega) - \beta_1\Omega$. In $q$ the $\beta_1$ terms are
$-\beta_1(\Omega_1+\Omega_2-\Omega)=0$, so $q = -\Delta k - \kappa(\Omega_1-\Omega_2)$
with the pair's true wavevector mismatch

```{math}
\Delta k = k_s(\omega_1+\omega_2) - k_z(\omega_1) - k_z(\omega_2), \qquad
\omega_{1,2} = \omega_0+\Omega_{1,2}.
```

**3. The depth integral.** $\int_0^P e^{iqz}\,\mathrm dz = P\,e^{iqP/2}\operatorname{sinc}(qP/2)$,
with $\operatorname{sinc}x = \sin x/x$. Its phase, like $e^{i\beta_\mathrm{sig}P}$ and
the drift's reference phase, does not depend on $\tau$.

**4. Parseval over delay.** For fixed $\Omega$, step 1 is a Fourier integral over
$\Omega_2$ with conjugate variable $\tau$, so integrating $\lvert\psi\rvert^2$ over
every delay gives $2\pi$ times the integral of the modulus squared of its integrand.
Every phase factor has unit modulus and drops out:

```{math}
M(\Omega) = \int\lvert\psi(\Omega,\tau)\rvert^2\,\mathrm d\tau
= 2\pi\,a(\Omega)^2\int S(\Omega_1)\,S(\Omega-\Omega_1)\,
P^2\operatorname{sinc}^2\!\bigl(q\,P/2\bigr)\,\mathrm d\Omega_1 .
```

On croak's grid the same steps hold for a sum over the $N$ delays of one period of
the time axis, which gives $N$ times the sum over pairs; the model does the depth
integral by quadrature, accurate to the tolerance of
{func}`~croak.crystals.depth_points`. {func}`croak.crystals.shg_marginal` evaluates
the pair sum with the analytic depth integral, and it matches the model's
delay-summed trace to about $10^{-6}$ of the peak (`tests/test_marginal_crystal.py`).

What the result says:

- It is still **phase-independent** and computable from $S$, so it remains an
  exact check, but of the model: the centroid and width are those of this $M$, no
  longer $2\bar\omega_S$ and $\sqrt2\,\sigma_S$.
- **The drift enters at second order.** The pair $(\Omega_1,\Omega_2)$ and the
  swapped pair $(\Omega_2,\Omega_1)$ have the same weight $S(\Omega_1)S(\Omega_2)$
  and the same $\Delta k$, but opposite drift terms. Their sum,
  $\operatorname{sinc}^2[(\Delta k + \kappa\delta)P/2] +
  \operatorname{sinc}^2[(\Delta k - \kappa\delta)P/2]$ with $\delta = \Omega_1-\Omega_2$,
  is even in $\kappa$, so the sign of $\kappa$ does not matter.
- **Smearing leaves it unchanged:** it convolves the trace along delay with a kernel
  of unit area, which does not change the delay integral.
- **Zero thickness:** the model then has no depth integral (one node of unit
  weight), and $M = a^2\,(S*S)$, the thin-medium marginal times the prefactor.
- **The assumptions are the thin case's:** the delay scan holds the whole trace,
  which in a crystal includes the walk-off's tails, and the whole beam is
  collected, as the model assumes.

The same $M$ is the target of the marginal correction for a crystal trace
({func}`croak.marginal_checks.apply_marginal_correction` with a crystal): the
correction then takes out the spectral response the model does not describe and
keeps the phase matching, which the retrieval's crystal model supplies.

## PG (and TG) — three checks

```{math}
M_\mathrm{PG}(\omega) = \bigl(S * \lvert\widehat{\lvert E\rvert^2}\rvert^2\bigr)(\omega).
```

The gate factor $\lvert\widehat{\lvert E\rvert^2}\rvert^2$ is the modulus-squared
FT of the **temporal intensity** $|E(t)|^2$. It carries the spectral phase
through $e^{i[\phi(\omega'+\omega)-\phi(\omega')]}$, so you cannot get it from $S$
alone. Three things are nonetheless usable.

### Transform-limited (TL) predicted marginal

Assume flat phase: $\tilde E_\mathrm{TL}=\sqrt{S}$. Then $|E_\mathrm{TL}(t)|^2$
and the gate factor are computable, so the whole marginal is computable from $S$.
Compare its **shape** to the measured marginal. A mismatch conflates "bad trace"
with "pulse is not TL", so this is a *soft* check, most useful alongside the
centroid and width anchors below.

### Centroid — exact, phase-independent

The gate factor is the modulus-squared FT of a **real** function, hence **even**
in $\omega$, hence has centroid exactly $0$ for any pulse. Therefore

```{math}
\bar\omega[M_\mathrm{PG}] = \bar\omega_S \quad\text{(exact, any phase).}
```

PG/TG sit at the fundamental (not doubled). This is a genuine **hard** check on
the frequency-axis calibration of the trace, independent of the TL assumption.

### Width — a two-sided bracket

Variances add: $\sigma^2[M_\mathrm{PG}] = \sigma_S^2 + \sigma^2[\,|\widehat{|E|^2}|^2\,]$.

- **Lower bound (rigorous):** $\sigma[M_\mathrm{PG}] > \sigma_S$. The gate term is
  strictly positive (it vanishes only for a CW field). A measured marginal
  *narrower* than the bare spectrum is unphysical — typically spectral clipping
  or detector roll-off.
- **Upper reference (soft):** the TL pulse has the most peaked $|E(t)|^2$, hence
  the broadest gate factor and the broadest marginal — *for smooth phase*.
  Pathological phase can beat $|E(t)|^2$ faster and exceed it, so this is
  typical-case, not a theorem. A measured marginal *broader* than the TL
  prediction flags additive broadband artefacts (scatter, background, stray
  light, detector nonlinearity).

So the expected width sits in the bracket
$\bigl[\sigma_S,\ \sigma_\mathrm{TL}\bigr]$
({attr}`~croak.marginal_checks.MarginalPrediction.width_bracket`), the interior
consistent with some unknown amount of chirp.

## SD — partly similar, with weaker anchors

```{math}
M_\mathrm{SD}(\omega) = \bigl(\lvert\widehat{E^2}\rvert^2 * S(-\omega)\bigr)(\omega).
```

The signal is $E_\mathrm{sig}=E(t)^2\,E^\ast(t-\tau)$, so the undelayed factor is
$E^2$ and its spectral intensity $\lvert\widehat{E^2}\rvert^2$ is the **SHG
spectrum** — phase-dependent. The delayed factor contributes only the reflected
fundamental spectrum $S(-\omega)$, which is phase-independent. **SD does not give
a phase-independent marginal**: the phase enters through the SHG-spectrum factor,
exactly as it would in an SHG measurement. Only SHG-FROG has the fully clean
marginal.

How the three PG checks carry over:

- **TL predicted marginal — fully analogous.** Set $\tilde E_\mathrm{TL}=\sqrt S$,
  form $E_\mathrm{TL}^2$, take $|\widehat{E_\mathrm{TL}^2}|^2$, convolve with
  $S(-\omega)$. Same soft-check status as PG.
- **Centroid — weaker.**
  $\bar\omega[M_\mathrm{SD}] = \bar\omega[\,|\widehat{E^2}|^2\,] - \bar\omega_S$.
  The SHG-spectrum centroid is *not* generally $2\bar\omega_S$; it coincides at TL
  and, more generally, for a **temporally symmetric intensity**. For asymmetric
  chirped pulses the SD marginal centroid drifts off $\bar\omega_S$, so treat
  "centred at the fundamental" as a TL/symmetric anchor, not a hard check
  ({attr}`~croak.marginal_checks.MarginalPrediction.centroid_is_hard` is `False`).
- **Width — messier.** Only the lower bound $\sigma[M_\mathrm{SD}] > \sigma_S$ is
  rigorous; the SHG-spectrum width depends on phase non-monotonically (for a
  Gaussian spectrum with pure GDD it is essentially invariant), so the clean
  PG-style upper bracket is lost — the prediction reports
  `width_bracket = (σ_S, None)`.
- **Bonus check (no TL assumption):** if an **independently measured SHG
  spectrum** is available, plug it in for $|\widehat{E^2}|^2$:
  $M_\mathrm{SD} = (\text{measured SHG spec}) * S(-\omega)$ is then a fully
  phase-independent prediction, recovering an SHG-grade check for the SD trace.

## What to compute, by process

| check | SHG | PG / TG | SD |
|---|---|---|---|
| predicted marginal from $S$ alone | exact | TL only | TL only (or via measured SHG spec) |
| centroid | $2\bar\omega_S$ (hard) | $\bar\omega_S$ (hard) | $\approx\bar\omega_S$ at TL/symmetric (soft) |
| width | $\sqrt2\,\sigma_S$ (hard) | bracket $[\sigma_S,\ \sigma_\mathrm{TL}]$ | $>\sigma_S$; TL estimate (weak) |
| nature of a mismatch | trace error only | trace error or non-TL phase | trace error or phase |

## Practical points

- **Normalise** the predicted and measured marginals (unit peak or unit area)
  before comparing shape; the absolute scale is set by the conversion efficiency
  and detector gain. {func}`~croak.marginal_checks.predicted_marginal` returns a
  peak-normalised marginal.
- Compare on a **common, calibrated frequency axis**: the SH axis for SHG, the
  fundamental axis for PG/TG/SD. A centroid mismatch on that axis *is* the
  calibration check.
- Width and centroid estimates can be sensitive to noise and cropping; the full-shape
  overlay is the most sensitive to scatter and detector nonlinearity but also the
  most affected by the TL assumption for PG/SD. Restrict the moments and the
  normalisation to a **spectral trust region** (in the GUI, 190–1000 nm for a
  measured trace, or the full loaded range for the noise-free simulated/synthetic
  traces) so out-of-band detector noise cannot dominate.
- For the rigorous SHG and PG-centroid checks, report the numeric discrepancy
  (measured vs predicted) as a QC metric alongside the retrieval error.

## Centroid-matching efficiency correction

A TG/PG detector's third-order generation/collection efficiency tilts the
marginal by a power law $(\lambda/\mu\mathrm{m})^n$
({func}`croak.preprocess.third_order_scale`). Because the marginal centroid is a
**hard** anchor for PG/TG (and the fundamental anchor for SD/SHG),
{func}`croak.marginal_checks.auto_third_order_exponent` picks the exponent $n$
whose corrected-marginal centroid lands on the predicted centroid. The tilt
shifts the centroid monotonically in $n$, so the match is at most one root over a non-degenerate positive trace; it is
restricted to the trust region so amplified out-of-band noise cannot bias it.
This is the physically grounded replacement for fitting the marginal *shape* to
the spectrum.
