# Marginal consistency checks

Before retrieving, it is worth asking whether the **trace** is even consistent
with the independently measured spectrum. The frequency marginal
$M(\omega)=\int T(\omega,\tau)\,\mathrm d\tau$ can be predicted (or rigorously
bounded) from the fundamental spectrum $S(\omega)$, so comparing the two flags
calibration, spectral-response, scatter and detector-nonlinearity problems that a
retrieval would otherwise absorb. The physics is in
[FROG trace marginals as consistency checks](../explanation/marginals.md); this
page is the recipe.

The identities assume a thin, local interaction and complete delay support; a
dispersive slab, smearing or a finite collection aperture also reshape the marginal.
For SHG through a crystal, the prediction can include the crystal's phase matching
(see [Through an SHG crystal](#through-an-shg-crystal) below).

## Predict a marginal from a spectrum

{func}`croak.marginal_checks.predicted_marginal` takes a uniform **absolute
angular-frequency** axis and a nonnegative spectral intensity, and returns a
{class}`~croak.marginal_checks.MarginalPrediction` carrying the predicted marginal
and its centroid/width anchors:

```python
import numpy as np
import croak
from croak import marginal_checks as mc

omega0 = float(croak.maths.wlfreq(800e-9))      # ~800 nm carrier (rad/s)
g = croak.Grid(1024, dt=0.4e-15)
omega = g.omega + omega0                        # absolute frequency axis
S = np.exp(-0.5 * ((omega - omega0) / (0.03 * omega0)) ** 2)   # fundamental spectrum

pred = mc.predicted_marginal(omega, S, "shg")
pred.centroid        # 2·ω̄_S  (rad/s) — hard for SHG
pred.rms_width       # √2·σ_S        — hard for SHG
pred.centroid_is_hard, pred.width_is_hard   # (True, True)
```

The geometry sets which anchors are rigorous:

| `interaction` | centroid | width | `width_bracket` |
|---|---|---|---|
| `"shg"` | `2·ω̄_S` (hard) | `√2·σ_S` (hard) | `None` |
| `"pg"` / `"tg"` | `ω̄_S` (hard) | TL `σ_TL` | `(σ_S, σ_TL)` |
| `"sd"` | `≈ ω̄_S` (TL/soft) | `> σ_S` | `(σ_S, None)` |

Use {func}`~croak.marginal_checks.predicted_centroid` if you only need the centroid
anchor, and {func}`~croak.marginal_checks.spectral_autoconvolution` for the raw
$S*S$ shape.

## Compare with a measured marginal

Compute the measured marginal and its moments on a **common, calibrated** axis
(work in angular frequency — convert a wavelength-density marginal to an
$\omega$-density with the $\times\lambda^2$ Jacobian first):

```python
marg = trace.sum(axis=1)               # measured marginal (delay is axis 1)
omega_t = croak.maths.wlfreq(lam)       # the trace's wavelength axis -> rad/s
weights = marg * lam**2                # λ-density -> ω-density
c_meas = mc.centroid(omega_t, weights)
sigma_meas = mc.rms_width(omega_t, weights)

print("centroid error:", c_meas - pred.centroid)      # the calibration QC metric
print("width OK:", sigma_meas > pred.sigma_spectrum)   # rigorous lower bound
```

Interpreting a mismatch:

- **SHG** — a centroid, width or shape mismatch means the trace and spectrum are
  inconsistent: calibration, spectral response, scatter or detector nonlinearity.
- **PG / TG** — the centroid is still a hard calibration check; a width below
  `σ_S` means spectral clipping or detector roll-off. The TL width `σ_TL` is a soft
  reference: phase structure can exceed it.
- **SD** — only the width lower bound is rigorous; the centroid holds at TL or for
  a temporally symmetric pulse.

Restrict the moments and the shape comparison to a spectral **trust region** so
out-of-band detector noise — amplified by any efficiency tilt — cannot dominate.

## Fit the third-order efficiency exponent

A TG/PG detector's third-order generation/collection efficiency tilts the
marginal by $(\lambda/\mu\mathrm{m})^n$ ({func}`croak.preprocess.third_order_scale`).
{func}`croak.marginal_checks.auto_third_order_exponent` picks the exponent whose
corrected-marginal **centroid** lands on the predicted centroid — a unique root,
because the tilt shifts the centroid monotonically in $n$:

```python
n = mc.auto_third_order_exponent(
    lam, marg, lam_spec, Ilam_spec, "pg", window=(210e-9, 1000e-9)
)
corrected = trace * croak.preprocess.third_order_scale(lam, n)[:, None]
```

This fits one empirical response exponent; it does not correct delay-dependent
collection effects.

## Through an SHG crystal

In a crystal of tens of µm the phase matching filters the SHG marginal strongly,
so the thin-medium $S*S$ would flag a correct trace. Give the prediction the
crystal ({class}`croak.crystals.PhaseMatching`: a
{class}`~croak.crystals.CrystalSetup` and its thickness) and the carrier the
retrieval uses, and it becomes the crystal model's own marginal
({func}`croak.crystals.shg_marginal`):

```python
import math
from croak.crystals import CrystalSetup, PhaseMatching, builtin_crystal
from croak.geometry import TwoBeamGeometry

crystal = CrystalSetup(
    builtin_crystal("BBO"),
    math.radians(29.2),
    phase_match_wavelength=800e-9,
    geometry=TwoBeamGeometry(beam_separation=5e-3, focal_length=0.2),
    polarization="parallel",
)
pm = PhaseMatching(crystal, thickness=50e-6)
pred = mc.predicted_marginal(omega, S, "shg", crystal=pm, omega0=omega0)
pred.centroid, pred.rms_width   # of the crystal's marginal, still hard checks
```

The centroid and width are no longer $2\bar\omega_S$ and $\sqrt2\,\sigma_S$ but
those of the computed marginal; for the crystal model they are as exact as the
thin-medium ones. The `omega0` is the retrieval grid's carrier
(`td.omega0_pulse`; the middle of the retrieval band), at which an explicit tilt is
refracted and the path and delay drift are taken.

The **marginal correction** can target the same prediction. Load the trace with
the crystal it was measured through:

```python
td = croak.preprocess.load_and_clean(
    trace, lam, delays, "shg", lam_min=600e-9, lam_max=1100e-9,
    lam_spec=lam_spec, Ilam_spec=Ilam_spec,
    marginal_correct=True, marginal_crystal=pm,
)
```

Each frequency row is then rescaled onto the crystal's marginal, which removes
what the model leaves out (the spectrometer's and detector's response, the
dispersion of the nonlinear coefficient, the beams' overlap) and keeps the phase
matching for the retrieval's crystal model to describe. `td.marginal_crystal`
records the crystal, and {func}`~croak.pipeline.retrieve_from_tracedata` refuses to
retrieve a corrected trace through any other medium: a thin-medium correction
divides out the phase matching, which a crystal model would apply again, and a
crystal correction leaves in a filter a thin model cannot reproduce. A session
uses the retrieval's crystal for the correction
({func}`croak.session.pipeline.correction_crystal`), so the two cannot disagree.

The correction relies on the same assumption as the check: the delay scan holds
the whole trace. A crystal's walk-off lengthens the trace in delay, so check that
the scan covers its tails.

## In the GUI

All of the above is interactive on the GUI's **Check the marginal** step, the
second step of the Data stage, which sits between the data loaders and Prepare
for measured, simulated and synthetic sessions. It overlays the predicted marginal on the measured
one, marks the predicted vs measured centroid, reports the width-bracket check,
and exposes the trust region, a log/linear toggle and the `(λ/µm)^exp` correction
with an `Auto` (centroid-matching) button. The **SHG marginal correction** also
lives here: it rescales the trace's frequency rows onto the predicted SHG
marginal ({func}`~croak.marginal_checks.apply_marginal_correction`): the
autoconvolution, or the crystal's marginal when the crystal is modelled (its
settings are on the Data stage, before this step). It is enabled only for an SHG
trace with an independent spectrum, and previews the snap
onto the prediction live (the correction itself is applied during preprocessing,
via {func}`~croak.preprocess.load_and_clean`'s `marginal_correct`). See the
[GUI guide](gui.md).
