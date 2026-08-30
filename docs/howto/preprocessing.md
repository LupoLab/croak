# Cleaning and regridding measured data

A raw FROG measurement arrives on a **wavelength** axis and a **delay or stage
position** axis, often with interferometric fringes, a background, detector
calibration and a non-uniform sampling. Retrieval, by contrast, needs a
normalised trace on a **uniform angular-frequency / delay grid**. {mod}`croak.preprocess`
bridges the two. The one call you usually need is
{func}`~croak.preprocess.load_and_clean`, which returns a
{class}`~croak.preprocess.TraceData` ready for retrieval. This pipeline —
generation-response correction, spectrometer Jacobian, Takeda de-fringing,
arPLS baseline removal, edge taper and sub-sample delay centring, with no
intensity thresholding anywhere — is the one specified and derived in the
companion paper; for the few-femtosecond DUV traces validated there, skipping
any one of these steps changed the retrieved pulse measurably.

```python
import croak

td = croak.load_and_clean(
    Ifrog, lam_frog, scanaxis, "shg",
    lam_min=380e-9, lam_max=440e-9,     # output retrieval band
    input_unit="delay",                  # scanaxis is already a delay (s)
)
# td.trace        -> normalised (Nomega, Ndelay) on td.grid
# td.delays       -> regridded delay axis (s)
# td.omega0_pulse -> carrier frequency for the retrieved pulse
```

## The pipeline

`load_and_clean` runs this sequence:

```
filter (fringe / DC / low-pass / de-fringe -> baseline) -> spectral window
   -> regrid (lambda->omega resample + anti-alias)
   -> marginal-correct -> threshold (off by default) -> delay crop
```

Background subtraction, detector calibration and third-order scaling are **not**
done by `load_and_clean`: apply them to the trace beforehand, or fold a
per-wavelength response into the `scalecurve` argument. The first three rows
below are the standalone helpers for that.

| Stage | Function | What it does |
|-------|----------|--------------|
| Background | {func}`~croak.preprocess.background_subtract` | subtract a dark frame (optionally clamp negatives) |
| Calibration | {func}`~croak.preprocess.calibration_curve_scale` | per-wavelength response from a `(λ_nm, scale)` curve |
| Third-order | {func}`~croak.preprocess.third_order_scale` | TG/χ⁽³⁾ wavelength efficiency $(\lambda/\mu m)^{p}$ |
| Trace filter | {func}`~croak.preprocess.filter_frog_trace` | remove fringes, DC and fast delay structure (NUFFT); window; centre $\tau=0$ |
| De-fringe | {func}`~croak.preprocess.defringe_carrier` | remove fringes the Takeda way: a smooth carrier-relative low-pass that keeps the slow magnitude trace |
| Baseline | {func}`~croak.preprocess.arpls_baseline` | estimate & remove the broad, *asymmetric* leakage background $\|E_b\|^2$ per wavelength row (arPLS), leaving the τ≈0 signal |
| Spectrum window | {func}`~croak.preprocess.filter_spectrum` | Planck-taper an independent spectrum |
| Regrid | {func}`~croak.preprocess.regrid` | resample λ→ω with the Jacobian and anti-aliasing onto a uniform grid |

The delay filter runs on a **non-uniform FFT** (FINUFFT), so the fringe band-stop
and `dtau_min` low-pass stay correctly calibrated even when the measured delay
axis is jittered or unevenly stepped (the filtering happens *before* regridding).
Its stop bands roll off with a steep Planck taper rather than a brick-wall cut,
and DC removal is an exact per-wavelength mean subtraction.

Input/output arrays follow croak's `(Nlambda, Ndelay)` / `(Nomega, Ndelay)`
convention — wavelength (or frequency) on axis 0, delay on axis 1.

## Key arguments

A few `load_and_clean` keywords do most of the work:

- **`lam_min`, `lam_max`** — the output retrieval wavelength band (m). These set
  the grid; choose them to bracket the signal with a little margin. For SHG the
  trace sits near $2\omega_0$ — see [Grids and filtering](grids_and_filtering.md).
- **`input_unit`** — `"position"` (stage position in m, converted via $\tau = 2z/c$)
  or `"delay"` (already seconds).
- **`filter_fringes`** — remove interferometric fringes by a band-stop in the
  (NUFFT) delay-frequency domain; `True` uses a sensible default band, or pass an
  explicit `(lo, hi)`. **Off by default** — switch it on only if your trace
  actually carries fringes.
- **`filter_dc`** — remove the delay-independent component of each wavelength
  row. **Off by default.** Enable it only for a measurement with a genuine
  delay-independent background: on a clean trace it subtracts real signal, which
  shows up as large negative excursions in the filtered trace and puts a
  percent-level floor under the achievable trace error.
- **`defringe`** — remove fringes the *Takeda* way instead: a smooth per-wavelength
  low-pass that keeps the DC band below the optical carrier $f_c(\lambda)=c/\lambda$
  and discards the $\pm f_c$ sidebands, recovering a fringe-free magnitude trace
  $a(\lambda,\tau)$. `defringe_fraction` (default `0.5`) sets the cutoff as a
  fraction of $f_c$; the carrier is *computed from the wavelength axis, not fitted*.
  **Mutually exclusive** with `filter_fringes` — these are two solutions to the same
  problem. `on_alias` guards the Nyquist condition (see below).
- **`baseline`** — remove the broad, slowly varying leakage background $|E_b|^2$ along
  delay per wavelength row by asymmetric least squares (see *Baseline removal: arPLS*
  below), applied right after de-fringing. `baseline_smoothness` (default `1e2`) sets
  the stiffness; `baseline_clip` (default off) optionally enforces non-negativity.
- **`dtau_min`** — low-pass: discard delay structure faster than this (s).
  **Off by default**; `None` or `0` applies no low-pass.
- **`roll_bins`** — steepness of the fringe/low-pass roll-off, in delay-frequency
  modes (default `3`; smaller ⇒ steeper edges).
- **`tau_lims`** (soft Planck taper, the τm window) and **`tau_crop`** (hard crop,
  applied last on the regridded delay axis) — window the delay axis. `tau_lims` is
  stored on the `TraceData` for post-retrieval windowing.
- **`lamm_lims`** — the *useful* measurement wavelength window λm (applied before
  regridding and stored on the `TraceData` for post-retrieval windowing).
- **`lam_spec`, `Ilam_spec`** — an independently measured fundamental spectrum;
  when supplied it is resampled onto the grid as `td.Iomega` and can seed the
  guess, drive the [spectral-match penalty](regularisation.md), or enable SHG
  marginal correction (`marginal_correct=True`).

## De-fringing: notch vs. Takeda low-pass

A measured TG/PG-FROG trace where the fundamental leaks into the signal arm carries
interferometric **fringes**. Per wavelength row it decomposes as

$$
I(\lambda,\tau) = a(\lambda,\tau)
  + c(\lambda,\tau)\,e^{+i\omega(\lambda)\tau}
  + c^{*}(\lambda,\tau)\,e^{-i\omega(\lambda)\tau},
$$

where $a = |E_s|^2 + |E_b|^2$ is slow in $\tau$ and the cross term (the fringes)
oscillates at the **optical carrier** $\omega(\lambda)=2\pi c/\lambda$, i.e. ordinary
fringe frequency $f_c(\lambda)=c/\lambda$. The carrier is **known a priori from the
wavelength calibration — not fitted** — which is why the fringes run diagonally
across the trace ($f_c$ tilts with $\lambda$).

There are two ways to remove them:

- **`filter_fringes`** — a band-stop **notch** straddling $f_c$. A brick-wall stop
  band has long sinc tails in delay, so it tends to add Gibbs ringing that
  masquerades as trace structure.
- **`defringe`** ({func}`~croak.preprocess.defringe_carrier`) — *Takeda FTSI*: a
  smooth **low-pass** that keeps only the slow DC band $|f| < \texttt{defringe\_fraction}\times f_c(\lambda)$
  and discards the $\pm f_c$ sidebands, leaving a smooth, fringe-free
  $a(\lambda,\tau)$. Because it uses smooth (Planck-taper) windows it does **not**
  ring, and because the carrier is computed it needs **no per-row fitting**. Unlike
  DC removal it *keeps* the slow signal content, including the zero-frequency mean
  and low-frequency wings.

```{admonition} Check the delay sampling (Nyquist)
:class: warning
The fringe sits at $f_c$ only if the delay step satisfies $\delta\tau < \lambda/(2c)$
(e.g. $\delta\tau < 0.42$ fs at 250 nm). On coarser sampling the fringe **aliases**
to a lower apparent frequency. `defringe_carrier` checks this at the signal-bearing
wavelengths and, by default (`on_alias="raise"`), raises with the $\delta\tau$ it
needs; pass `on_alias="clamp"` to low-pass below the apparent (folded) carrier
instead, or `"warn"` to proceed.
```

De-fringing removes only the cross term; the broad leakage background $|E_b|^2$
survives in $a$ and is removed by a separate baseline step
({func}`~croak.preprocess.arpls_baseline`, below). The intended order is
**de-fringe → baseline-subtract → clip negatives** — de-fringing first makes the
baseline fit robust, since fringe troughs would otherwise dip below the background.

## Baseline removal: arPLS

After de-fringing, the smooth $a(\lambda,\tau) = |E_s|^2 + |E_b|^2$ still carries the
broad, slowly varying leakage background $|E_b|^2$. {func}`~croak.preprocess.arpls_baseline`
removes it per wavelength row, **without touching the compact $\tau\approx0$ signal** —
replacing a crude DC subtraction (which subtracts a constant and so guts the signal's
low-frequency content).

It uses *asymmetrically reweighted penalised least squares* (arPLS, Baek et al. 2015):
for each row $y$ it fits a smooth baseline $z$ minimising
$\sum_i w_i (y_i-z_i)^2 + \lambda\sum_i(\Delta^2 z_i)^2$, where $\Delta^2$ is a
curvature penalty and $\lambda$ (`baseline_smoothness`) the stiffness. The weights are
reweighted each iteration from the **negative** residuals (the noise/background), so
points above the baseline (signal) lose weight and $z$ settles onto the lower envelope.

Because $z$ is a *free-form* smooth lower-envelope fit — no symmetry or shape prior —
it rides under a background that is **asymmetric about $\tau=0$** just as readily as a
flat one. This matters for real DUV TG-FROG data, where the leakage is strong at
negative delay and weak at positive delay; a constant background is simply the
zero-curvature limit. The carrier-known de-fringe and this baseline are enabled
together via `load_and_clean(..., defringe=True, baseline=True)`.

:::{tip}
Two runnable examples build a synthetic DUV TG-FROG trace with known fringes and
a known asymmetric background, then report the *actual* recovery error:
`examples/example_defringe.py` for the de-fringe step alone, and
`examples/example_arpls_baseline.py` for the full chain. Both share the trace
generator in `examples/synthetic_duv_trace.py`, which is a compact worked model
of where the fringes and the background come from — a useful starting point for
sanity-checking the settings before applying them to a measurement.
:::

- **`baseline_smoothness`** — the stiffness $\lambda$ (default `1e2`, useful
  `~1–1e6`). Too low and the baseline bends into — and removes — the signal itself
  (the dominant failure, visible as gouged-out rows); too high and it cannot follow
  the shelf in the wings. It is normalised by delay-sample count and per-row
  amplitude, but the *right* value still depends on the trace: `1e2` suits a broad,
  noisy measured background like the DUV TG-FROG case. A **clean or simulated trace, a
  narrow delay window, or a small `Ndelay`** makes the effective penalty much softer
  and needs a substantially higher value (`~1e4`+). A clean simulation has **no
  leakage background** to remove at all — leave `baseline` off and let de-fringing do
  the cleaning. Tune it live on the preprocess page.
- **`baseline_tau_exclude`** (s) — hold out $|\tau| < \texttt{baseline\_tau\_exclude}$
  from the fit, so the tall $\tau\approx0$ peak cannot pull the baseline up into it.
  See [Peak suck-up](#peak-suck-up-and-the-hold-out) below: on measured traces this is
  the difference between keeping and losing a third of the signal, and it is off
  (`0`) by default.
- **`baseline_downsample`** / **`baseline_skip_below`** — speed controls. Because the
  baseline is smooth, `baseline_downsample=q` solves it on every $q$-th delay and
  interpolates back (≈$q\times$ faster at a percent-level difference), and
  `baseline_skip_below` skips near-zero rows. Both default to exact; the GUI uses them
  for its live preview while keeping the committed/headless trace exact.

Convergence is measured on the **baseline** itself (`baseline_ratio`, the relative
change of $z$ between iterations), not on the weights — the weights keep
micro-adjusting near the signal edges long after the baseline has settled, so a weight
tolerance wastes many iterations for no change in the output.

```{admonition} Clipping is off by default
:class: warning
Subtracting the baseline can leave small negatives where it sits just above the noise.
`baseline_clip` is **off by default**, keeping them — the same maximum-likelihood
reasoning as thresholding (below): COPRA's amplitude projection handles negative
targets, and clipping censors the negative half of the noise and biases the estimate.
Set `baseline_clip=True` only if a strictly non-negative trace is required.
```

### Peak suck-up and the hold-out

The reweighting usually finds the peak on its own, but not always: a free-form smooth
fit will bow *up* into a strong $\tau\approx0$ peak, and how far it bows depends on how
well each wavelength row constrains it. That varies sharply from row to row, so the
loss is **not** a uniform dimming — it concentrates on the rows whose peak sits low
against their background and shows up as a **thin stripe at one wavelength that appears
only when baseline removal is on**.

On a measured DUV TG-FROG trace the two worst rows kept 36–39 % of their peak while
their immediate neighbours kept ~70 %, and thirteen rows in the signal band fell below
80 %. Setting `baseline_tau_exclude` to **twice the delay-marginal FWHM** took the worst
row from 23 % to 87 % and left no row below 87 % — and it converged *faster*, because
the fit settles sooner. Once the marginal FWHM is too narrow to help; three times
leaves too little wing to fit and is worse again.

{func}`~croak.preprocess.suggested_tau_exclude` computes that width for you:

```python
from croak.preprocess import suggested_tau_exclude

tau_exclude = suggested_tau_exclude(trace, delays)      # 2 x the marginal FWHM
td = croak.load_and_clean(..., baseline=True, baseline_tau_exclude=tau_exclude)
```

It measures the delay marginal above its own pedestal and caps the result at 40 % of
the delay half-span, so the hold-out can never approach the whole axis (which
`arpls_baseline` rejects). The value is insensitive to what you hand it — raw or
de-fringed, full range or windowed all agree to ~10 %. In the GUI, the **Auto** button
beside *τ exclude (fs)* does exactly this for the loaded trace.

## Thresholding and the noise model

`threshold` clamps trace values below a fraction of the peak to zero. It is
**off by default** (`threshold=None`) and, when set, is applied **once, at the
very end** — after filtering and regridding — so it never feeds back into the
centring or marginal steps. This is a *pragmatic* clean-up, not a statistically
neutral one.

```{admonition} Thresholding biases the least-squares estimate
:class: warning
Under Gaussian noise the [least-squares objective](../explanation/retrieval_theory.md)
is the maximum-likelihood estimator only if the noise is left intact. Hard
thresholding censors the negative half of the noise distribution and introduces a
small positive bias. The default `threshold=None` therefore keeps the filter's
small negative values (the strictly MLE-correct choice); COPRA's amplitude
projection handles negative targets safely. Set a small `threshold` (e.g.
`1e-5`) only to suppress obvious junk in the dark corners of the trace.
```

The default Gaussian noise model is valid above ~20 photons/pixel; very
photon-starved data is Poisson and would want signal-dependent weights (currently
uniform).

## Inspecting the result

`load_and_clean` keeps the measured *and* filtered trace in the wavelength domain
on the {class}`~croak.preprocess.TraceData` (`tau_meas`/`Ifrog_meas`,
`tau_filt`/`Ifrog_filt`, `lam_frog`, `scalecurve`, …) so you can see exactly what
the cleaning did. The six-panel {func}`croak.plotting.plot_frog_filter` visualises
it:

```python
fig = croak.plot_frog_filter(td)
```

## Settings across a re-load

In the [wizard](gui.md) the preprocess controls are bound one way — a widget
writes its parameter and triggers a recompute — so what the parameters say is
what gets computed. To keep the two in step, **entering the stage re-seeds every
widget from the current parameters**. That matters because loading a trace
re-seeds the parameters themselves from the data (grid extents, window bounds,
filter defaults), which would otherwise leave the boxes ticked while a different
set of options was applied.

Whether that reset happens is decided by the *trace signature* — the file,
dataset/window selection, units, loaded band, orientation and interaction, i.e.
everything that changes the data-seeded defaults:

* **Same trace** (reloading after tweaking a background, a calibration, the
  third-order correction, or just walking Back and forward again): your windowing
  and filter choices are kept; only the slider *bounds* are refreshed and the
  values clamped into them.
* **Different trace**: the data-seeded defaults are restored, and — since the
  widgets are re-seeded — you *see* them reset.

The **Reset defaults** button restores the data-seeded defaults for the current
trace at any time. Generating a new synthetic trace always counts as a new trace,
since its delay and wavelength grids may have moved.

## Loading the file

`load_and_clean` takes *arrays*, so load them first. {mod}`croak.io` is a generic
HDF5/NPZ dataset picker with unit handling:

```python
fd = croak.io.list_datasets("frog.h5")
lam = fd.load("wavelength") * croak.io.unit_to_si("nm")
delays = fd.load("delay") * croak.io.unit_to_si("fs")
trace = fd.load("trace")                       # (Nlambda, Ndelay)
```

See the [experimental-workflow tutorial](../tutorials/04_experimental_workflow.md)
for the full load → clean → retrieve → save run.
