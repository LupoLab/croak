---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3
  name: python3
---

# The experimental workflow

Real data does not arrive on a uniform retrieval grid. This tutorial runs the full
library pipeline — **load → clean → retrieve → post-process → plot → save** — on a
synthetic "experimental" file, exactly as the [GUI](../howto/gui.md) does but in a
script. It mirrors `examples/example_workflow.py`.

```{code-cell} ipython3
%matplotlib inline
import tempfile
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

import croak
from croak.maths import wlfreq
```

## A synthetic measurement file

We first fabricate a PG-FROG file on a **wavelength** axis (nm) and a **delay**
axis (fs), the way a spectrometer + delay stage would record it.

```{code-cell} ipython3
def make_experiment_file(path):
    # The time window (n*dt = 102 fs) must comfortably exceed the delay range plus
    # the pulse extent, or the gated signal wraps around the periodic FFT window
    # and the "measurement" contains a spurious second copy of the pulse.
    g = croak.Grid(256, dt=0.4e-15)
    omega0 = wlfreq(800e-9)
    ew = croak.gaussian_pulse(g, 6e-15, phases=[25e-30])    # chirped, ~13 fs
    delays = np.linspace(-40e-15, 40e-15, 90)
    trace = croak.maketrace(g.omega, delays, ew, "pg")

    omega_abs = g.omega + omega0
    band = (omega_abs > 0.45 * omega0) & (omega_abs < 2.0 * omega0)
    lam = wlfreq(omega_abs[band])
    order = np.argsort(lam)
    # maketrace returns a *frequency* density I_ω. A spectrometer records a
    # *wavelength* density I_λ = I_ω·|dω/dλ|, and load_and_clean multiplies by
    # λ² to undo exactly that — so convert here, or the Jacobian is applied twice.
    trace_lam = croak.preprocess.omega_to_lambda_density(
        trace[band][order, :], lam[order]
    )
    trace_lam = trace_lam / trace_lam.max()     # detector counts: arbitrary units
    with h5py.File(path, "w") as f:
        f["wavelength"] = lam[order] / 1e-9     # nm
        f["delay"] = delays / 1e-15             # fs
        f["trace"] = trace_lam

    # A real setup also records the fundamental spectrum on its own spectrometer.
    # That is an independent measurement — the retrieval never sees it — so it is
    # the best consistency check available. It is a *wavelength* density, hence
    # the |dw/dl| = 2*pi*c/lambda^2 Jacobian.
    lam_spec = lam[order]
    Ilam_spec = np.abs(ew[band][order]) ** 2 * 2 * np.pi * croak.constants.C / lam_spec**2
    Ilam_spec = Ilam_spec / Ilam_spec.max()

    with h5py.File(path, "a") as f:
        f["spectrum_wavelength"] = lam_spec / 1e-9
        f["spectrum_intensity"] = Ilam_spec

    # Pick the band from the wavelength marginal at 1e-4 of its peak — i.e. the
    # -40 dB point, which is where the signal actually ends on a log plot. A
    # few-percent cut looks generous on a linear scale and badly clips the wings.
    marg = trace_lam.sum(axis=1)
    sig = np.where(marg > 1e-4 * marg.max())[0]
    lam_nm = lam[order] / 1e-9
    # The generating pulse, kept so the tutorial can mark its own homework.
    truth = croak.TruthPulse.from_spectrum(g, ew, omega0)
    return lam_nm[sig[0]] * 1e-9, lam_nm[sig[-1]] * 1e-9, truth

h5_path = str(Path(tempfile.gettempdir()) / "croak_tutorial.h5")
lam_min, lam_max, truth = make_experiment_file(h5_path)
true_fwhm = truth.fwhm
print(f"signal band: {lam_min*1e9:.0f}-{lam_max*1e9:.0f} nm")
print(f"true FWHM  : {true_fwhm/1e-15:.2f} fs   (kept only to check ourselves)")
```

## Load

{mod}`croak.io` is a generic dataset picker with unit handling, so we read the
arrays and convert them to SI.

```{code-cell} ipython3
fd = croak.io.list_datasets(h5_path)
lam = fd.load("wavelength") * croak.io.unit_to_si("nm")
delays = fd.load("delay") * croak.io.unit_to_si("fs")
trace = fd.load("trace")

# the separately measured fundamental spectrum
lam_spec = fd.load("spectrum_wavelength") * croak.io.unit_to_si("nm")
Ilam_spec = fd.load("spectrum_intensity")
trace.shape, lam_spec.shape
```

This is what came off the instrument: counts on a **wavelength** axis, which is
not uniform in frequency, against delay.

```{code-cell} ipython3
fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, data, title in (
    (ax_lin, trace, "as loaded (linear)"),
    (ax_log, 10 * np.log10(np.maximum(trace / trace.max(), 1e-5)), "same, in dB"),
):
    m = ax.pcolormesh(delays * 1e15, lam * 1e9, data, shading="auto",
                      cmap="viridis", vmin=-50 if "dB" in title else None)
    fig.colorbar(m, ax=ax)
    ax.set_xlabel("delay (fs)")
    ax.set_title(title)
    # the chosen band, on *both* panels — the dB one is where you judge it
    ax.axhline(lam_min * 1e9, color="w", ls="--", lw=1)
    ax.axhline(lam_max * 1e9, color="w", ls="--", lw=1)
ax_lin.set_ylabel("wavelength (nm)")
fig.tight_layout()
```

The dashed lines mark the signal band we will keep, and **the dB panel is the one
to judge them against**. On the linear scale the band looks generous; on the log
scale you can see how far the trace really extends, and any signal outside the
lines is about to be tapered away for good.

This is the single easiest way to lose accuracy in the whole workflow. Choosing
the band at a few percent of the marginal peak — which looks perfectly reasonable
on a linear plot — clips the wings and costs a **factor of four** in the
achievable trace error:

| band chosen at | window | trace error $R$ | FWHM error |
|---|---|---|---|
| 3 % of the marginal peak | 635–1045 nm | 2.1 % | +1.3 % |
| 1 % | 609–1082 nm | 1.2 % | +1.0 % |
| **0.01 % (−40 dB)** | **554–1313 nm** | **0.030 %** | **+0.1 %** |

The FROG trace carries information in its wings — that is where the chirp shows
itself — so cut the band from the **log** view, well below where the signal looks
like it has ended. Widening costs only a few more frequency points.

## Clean and regrid

{func}`~croak.preprocess.load_and_clean` filters and resamples the trace onto a
uniform retrieval grid, returning a {class}`~croak.preprocess.TraceData`. See
[Preprocessing](../howto/preprocessing.md).

Note what is *not* in this call. Every filter — the fringe notch, the DC removal,
the delay low-pass, the baseline, the noise threshold — is **off by default**,
because each removes real signal along with whatever it targets. Switch one on
when your data needs it, not before. The only thing running here is `prefilter`,
the anti-alias low-pass that resampling genuinely requires.

```{code-cell} ipython3
td = croak.load_and_clean(
    trace, lam, delays, "pg",
    lam_min=lam_min, lam_max=lam_max, input_unit="delay",
    lam_spec=lam_spec, Ilam_spec=Ilam_spec,      # the independent spectrum
)
print(f"regridded trace: {td.trace.shape[0]} freqs × {td.delays.size} delays")
print(f"retrieval grid : {td.grid.n} points, dt = {td.grid.dt / 1e-15:.2f} fs")
```

{func}`~croak.plotting.plot_frog_filter` shows the whole cleaning step at once:
the measured trace, the filtered version, and what came out on the retrieval grid,
in dB on the top row and linear on the bottom.

```{code-cell} ipython3
fig = croak.plot_frog_filter(td)
fig.set_size_inches(11, 5.5)
fig
```

Two things changed. The **vertical axis** is now angular frequency, uniformly
sampled — the retrieval works in $\omega$, and resampling from $\lambda$ carries
the $\lambda^2$ Jacobian with it. And the axis has been **cropped** to the band we
asked for, so the empty wavelength ranges above and below the signal no longer
occupy rows of the fit. Compare the frequency counts printed above: this is
usually a large reduction, and it is the single biggest lever on retrieval time.

Note how little the *filtered* panel differs from the *measured* one: with every
filter off, the only thing between them is the spectral window. That is what you
want on clean data. Each panel also carries a second, white→red colourbar for
**negative** values — croak keeps them rather than clamping to zero, so that a
filter which has removed too much is visible instead of silently hidden. Here
they are at the $10^{-12}$ level, which is numerical noise. Turn `filter_dc` back
on and they grow to around 20 % of the peak, which is exactly the kind of thing
that colourbar exists to show you.

Judge this figure on whether the regridded panel still contains everything the
measured one did. Signal clipped at the edges of the band is gone for good — see
[Grids and filtering](../howto/grids_and_filtering.md) for choosing the window.

## Retrieve

{func}`~croak.pipeline.retrieve_from_tracedata` builds a sensible initial guess
from the cleaned trace, picks the solver and runs it.

```{code-cell} ipython3
result = croak.retrieve_from_tracedata(td, algorithm="copra", maxiters=250,
                                      rng=np.random.default_rng(0))
print(f"FROG error R = {result.error:.4%}")
```

Always look at the convergence curve before believing a number:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 3.5))
ax.semilogy(np.arange(1, len(result.errors) + 1), result.errors)
ax.axhline(result.error, color="C1", ls="--", lw=1,
           label=f"best R = {result.error:.2e}")
ax.set_xlabel("iteration")
ax.set_ylabel("trace error R")
ax.set_title("convergence")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
```

A curve still descending at the last iteration means `maxiters` was the binding
constraint and there is more to gain; one that flattened long ago means the solver
found what it could.

This one flattens at a few parts in $10^4$, not at $10^{-16}$ as on the raw
synthetic grid in the earlier tutorials. That floor is the regridding:
`load_and_clean` interpolates the measurement onto a new, uniform grid, so the
result is no longer *exactly* a FROG trace of any pulse, and a few parts in
$10^4$ is what that leaves. It is a floor, not a convergence failure — more
iterations, random restarts, an LM polish and a four-times finer grid all leave
it where it is.

This is the single most useful thing to internalise about $R$ on real data. It
does **not** translate into an error of that size on the answer — the next section
checks the recovered duration against the truth. A small residual floor after
regridding is what a good measured retrieval looks like;
what matters is whether the residual is *structured*.

## Post-process

{func}`~croak.processing.process_result` derives display-ready profiles and fits
the residual dispersion.

```{code-cell} ipython3
pr = croak.process_result(result, measured=td.trace)
print(f"recovered FWHM = {pr.fwhm_retr / 1e-15:.2f} fs "
      f"(transform-limited {pr.fwhm_tl / 1e-15:.2f} fs)")
print(f"true FWHM      = {true_fwhm / 1e-15:.2f} fs"
      f"   → error {100 * (pr.fwhm_retr - true_fwhm) / true_fwhm:+.1f}%")
print(f"fitted GDD = {pr.gdd_fs2:.0f} fs²,  TOD = {pr.tod_fs3:.0f} fs³"
      f"   (25 fs² went in)")
```

The duration comes back within about a tenth of a percent of
the truth and the fitted GDD reproduces the 25 fs² we put in — the dispersion fit
is a polynomial through a phase sampled at 54 frequency points, so a few fs² of
scatter is expected.

On a real measurement you would not have that middle line to check against. That
is exactly why the residual panel, the marginals and an [uncertainty
estimate](../howto/uncertainty.md) matter.

## Plot

The twelve-panel {func}`~croak.plotting.plot_retrieval` overview shows the measured
and retrieved traces, the residual, the temporal and spectral profiles, the
marginals and the convergence.

```{code-cell} ipython3
fig = croak.plot_retrieval(result, measured=td.trace, Iomega_meas=td.Iomega,
                          lam_min=td.lam_min, lam_max=td.lam_max,
                          truth=truth, processed=pr)
fig.set_size_inches(11, 8)
fig                                # return the Figure so the notebook displays it
```

Two extras are worth passing here. `Iomega_meas=td.Iomega` overlays the
**independently measured spectrum** `M` on the retrieved one `R` — the most
valuable check available on real data, because it comes from a measurement the
retrieval never saw; if `R` and `M` disagree, something is wrong regardless of how
small the trace error is. `truth=` overlays the generating pulse, which only a
synthetic study has. Passing the `processed=` result we already computed saves
deriving it twice.

Now the residual panel, and it is worth dwelling on. It is **clearly structured** —
a red-and-blue dipole running along the trace, not speckle. Structure like that
means the model cannot reproduce the data, and the usual suspects are physical: an
unmodelled dispersive medium, a missing geometric-smearing term, a mis-calibrated
axis.

Here we know none of those apply — the trace was generated by exactly this
forward model. What is left is the **regridding**: the retrieval sees an
interpolation of the measurement onto 54 frequency points, and that interpolation
is not exactly a FROG trace of any pulse. It is the same floor that limited
the convergence curve, and it is why the residual has shape.

The lesson is the useful one for real data: a structured residual tells you
something the model cannot fit, but "the model" includes everything upstream of
it — preprocessing choices as much as physics. Before concluding that the physics
is wrong, check whether a finer retrieval grid or a wider spectral window makes
the structure shrink.

## Reading the residual

The residual panel is the most informative thing in that figure, and it repays a
systematic look. Structure in it is rarely random: a **dipole** — a positive lobe
beside a negative one — is the signature of a small *displacement*, because
subtracting two copies of the same feature slightly offset from each other leaves
exactly that. Correlating the residual against the derivative of the trace along
each axis identifies which displacement:

```{code-cell} ipython3
def residual_diagnostics(result, processed, tracedata):
    """Correlate a residual against the shift/scale templates it could arise from."""
    T = result.trace * result.mu
    templates = {
        "delay shift   ∂T/∂τ": np.gradient(T, tracedata.delays, axis=1),
        "freq. shift   ∂T/∂ω": np.gradient(T, tracedata.grid.omega, axis=0),
        "amplitude     T": T,
    }
    a = processed.residual - processed.residual.mean()
    for name, template in templates.items():
        b = template - template.mean()
        c = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
        print(f"  {name}: {c:+.3f}")

residual_diagnostics(result, pr, td)
```

All three are small, so this retrieval has no systematic displacement left in it —
what remains is the regridding floor. Keep the magnitudes in mind for comparison:
a real shift shows up an order of magnitude larger, as we are about to see.

That is not automatic. Time zero is not known a priori on a measured scan;
`load_and_clean` takes it to be where the delay marginal peaks. Reading that off
as the largest *sample* would quantise it to the delay step and leave up to half a
step behind, so croak interpolates the peak instead
({func}`~croak.preprocess.marginal_peak_delay`). Half a step does not sound like
much — here it would be 0.45 fs — but the retrieval sees it clearly. Inject it
deliberately and watch:

```{code-cell} ipython3
import dataclasses

step = float(np.diff(td.delays)[0])
shifted = dataclasses.replace(td, delays=td.delays + 0.5 * step)

bad = croak.retrieve_from_tracedata(shifted, algorithm="copra", maxiters=250,
                                    rng=np.random.default_rng(0))
pr_bad = croak.process_result(bad, measured=shifted.trace)
print(f"delay step            = {step / 1e-15:.3f} fs")
print(f"R, correctly centred  = {result.error:.2e}")
print(f"R, offset by half a step = {bad.error:.2e}")
residual_diagnostics(bad, pr_bad, shifted)
```

The trace error jumps by more than an order of magnitude and the delay
correlation goes from nothing to about $-0.53$ — the dipole is back. A
sub-sample calibration error, invisible in any of the preprocessing plots, is
worth a factor of seventeen here. Side by side, on a shared colour scale:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
vmax = np.abs(pr_bad.residual).max()
for ax, resid, title in (
    (axes[0], pr.residual, f"centred      (R = {result.error:.1e})"),
    (axes[1], pr_bad.residual, f"+half a step (R = {bad.error:.1e})"),
):
    m = ax.pcolormesh(td.delays * 1e15, (td.grid.omega + td.omega0_trace) / (2e15 * np.pi),
                      resid, cmap="bwr", vmin=-vmax, vmax=vmax, shading="auto")
    ax.set_xlabel("delay (fs)")
    ax.set_title(title)
fig.colorbar(m, ax=axes)
axes[0].set_ylabel("frequency (PHz)")
```

The right panel is the dipole: red on one side of the trace ridge, blue on the
other. That antisymmetry across the ridge is what "displaced in delay" looks
like, and it is why the correlation test picks out $\partial T/\partial\tau$.

The autodiff solvers can fit the offset as a free parameter
([`fit_tau0`](../howto/fitting_thickness_tau0.md)), which works for every
geometry and recovers it exactly:

```{code-cell} ipython3
fixed = croak.retrieve_from_tracedata(shifted, algorithm="lm", fit_tau0=True,
                                      maxiters=400, abstol=1e-14,
                                      rng=np.random.default_rng(0))
print(f"fitted τ₀ = {fixed.tau0 / 1e-15:+.3f} fs   (we injected {0.5 * step / 1e-15:+.3f} fs)")
print(f"R         = {fixed.error:.2e}   (back to the centred value)")
```

**Reading the residual, not only the error, is what located this.** A
structured residual points to its cause, and no amount of extra iterations would
have touched this one. On real data `fit_tau0` is worth turning on as a matter of
course: a measured delay stage has its own zero-point uncertainty, usually larger
than the sub-sample residue left by the centring.

## Save

Persist the result (and its processed form) to a portable HDF5 file. See
[Saving and loading](../howto/saving_and_loading.md).

```{code-cell} ipython3
out = str(Path(tempfile.gettempdir()) / "croak_tutorial_result.h5")
croak.save_result(result, out, processed=pr, force=True)
print("wrote", out)
```

That is the entire workflow — and every step is a plain library call, so you can
wire it into your own scripts or batch jobs.
