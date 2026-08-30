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

# Your first retrieval

This tutorial synthesises an SHG-FROG trace from a known pulse and retrieves the
pulse back from it — the round trip that underpins everything else. Every cell
below is executed when the documentation is built, so the numbers and figures are
real.

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import croak
```

## A grid and a test pulse

A {class}`croak.Grid` carries matched, centred time and angular-frequency axes. We
build a 2.5 fs Gaussian pulse with a little group-delay dispersion (GDD), so its
spectral phase is non-trivial and worth retrieving.

```{code-cell} ipython3
g = croak.Grid(128, dt=0.3e-15)                       # 128 points, 0.3 fs step
omega0 = croak.maths.wlfreq(800e-9)                   # 800 nm carrier
ew = croak.gaussian_pulse(g, 2.5e-15, phases=[4e-30]) # phases = [GDD, TOD, ...] in SI

# the temporal intensity of the true pulse
field_true = g.ifft(ew)
I_true = np.abs(field_true) ** 2
fwhm_true = croak.maths.fwhm(g.t, I_true) * 1e15
print(f"true temporal FWHM = {fwhm_true:.2f} fs")
```

The `2.5e-15` sets the *transform-limited* width; the 4 fs² of GDD chirps the pulse
and stretches it to the number printed above. Recovering both — the duration and
the chirp that caused it — is the point of the exercise.

The carrier `omega0` does not enter a thin-medium retrieval at all; it only fixes
the wavelength axis $\lambda = 2\pi c/(\omega+\omega_0)$ used for plotting later.

## Synthesise a trace

{func}`~croak.maketrace` runs the [forward model](../explanation/forward_model.md)
to produce the `(Nomega, Ndelay)` SHG-FROG trace.

```{code-cell} ipython3
delays = np.linspace(-18e-15, 18e-15, 87)
trace = croak.maketrace(g.omega, delays, ew, "shg")
trace.shape
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(trace, aspect="auto", origin="lower",
          extent=[delays[0] * 1e15, delays[-1] * 1e15,
                  g.omega[0] / 1e15, g.omega[-1] / 1e15])
ax.set_xlabel("delay (fs)")
ax.set_ylabel(r"$\omega$ (rad/fs)")
ax.set_title("SHG-FROG trace")
fig.tight_layout()
```

## Retrieve

The [`retrieve`](../getting_started/quickstart.md) call has a uniform signature
across [algorithms](../explanation/algorithms.md). We use the default COPRA,
seeded from a slightly-too-short Gaussian and a fixed random seed so the result is
reproducible. (Passing `guess=None` instead starts from a random spectrum; on real
data you would run several such [random restarts](../howto/solver_selection.md).)

```{code-cell} ipython3
res = croak.retrieve(trace, g.omega, delays, "shg", algorithm="copra",
                    guess=croak.gaussian_pulse(g, 1.8e-15), omega0=omega0,
                    maxiters=300, rng=np.random.default_rng(0))
print(f"final trace error R = {res.error:.2e}")
print(f"iterations          = {len(res.errors)}")
```

A trace error of $R \lesssim 10^{-3}$ on clean synthetic data is an excellent fit
(see [the trace error](../explanation/retrieval_theory.md)).

## Compare retrieved and true pulse

Retrieval is defined only up to a [trivial time shift and phase](../explanation/pnps_framework.md#ambiguities),
so we peak-centre and peak-normalise both pulses before overlaying them.

```{code-cell} ipython3
def center(t, intensity):
    """Shift so the peak sits at t=0 and normalise the peak to 1."""
    i = int(np.argmax(intensity))
    return t - t[i], intensity / intensity.max()

t_true, I_true_n = center(g.t, I_true)
t_retr, I_retr_n = center(res.t, res.intensity_t)

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(t_true * 1e15, I_true_n, lw=3, alpha=0.4, label="true")
ax.plot(t_retr * 1e15, I_retr_n, "--", lw=1.5, label="retrieved")
ax.set_xlim(-15, 15)
ax.set_xlabel("time (fs)")
ax.set_ylabel("intensity (norm.)")
ax.legend()
ax.set_title("temporal intensity")
fig.tight_layout()
```

The retrieved intensity overlays the input. The
{class}`~croak.result.RetrievalResult` also exposes the spectrum, spectral phase
and wavelength axis:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(g.omega / 1e15, res.intensity_omega / res.intensity_omega.max(),
        label="spectral intensity")
ax2 = ax.twinx()
mask = res.intensity_omega > 0.02 * res.intensity_omega.max()
ax2.plot(g.omega[mask] / 1e15, res.phase_omega[mask], "C1", label="phase")
ax.set_xlabel(r"$\omega$ (rad/fs)")
ax.set_ylabel("intensity (norm.)")
ax2.set_ylabel("spectral phase (rad)")
ax.set_title("retrieved spectrum and phase")
fig.tight_layout()
```

## The standard summary figure

Everything above — and rather more — comes from one call to
{func}`~croak.plotting.plot_retrieval`. This is the figure the GUI shows after a
run, and the one to reach for when judging a retrieval.

Because this is **SHG**, one thing has to be settled first. An SHG trace is
exactly unchanged by $E(t) \to E^*(-t)$, so a pulse and its mirror image fit it
equally well and the solver returns whichever it happened to find. When the true
pulse is known, {func}`~croak.resolve_time_direction` picks the matching branch —
without it a perfect retrieval can appear mirrored against the truth, and the sign
of the reported GDD is a coin toss. It is a no-op for PG and SD.

```{code-cell} ipython3
truth = croak.TruthPulse.from_spectrum(g, ew, omega0)
res, flipped = croak.resolve_time_direction(res, truth)
print(f"time direction flipped to match the truth: {flipped}")
```

```{code-cell} ipython3
f0 = omega0 / (2 * np.pi)                       # SHG signal sits at 2*f0
fig = croak.plot_retrieval(res, measured=trace, truth=truth,
                           lam_min=600e-9, lam_max=1100e-9,
                           flim=(2 * f0 - 0.35e15, 2 * f0 + 0.35e15))
fig      # croak builds Figures without pyplot, so return it to display it
```

Read it in four groups.

**The traces (left block).** *Measured* and *Retrieved* in log and linear scale.
These should be indistinguishable — that is what a converged retrieval looks like.
The linear retrieved panel is annotated with the trace error and the grid size.
The log pair is the more revealing one: it stretches the weak wings, where a
mismatch shows up long before it is visible on a linear scale.

**The residual (bottom middle).** Measured − retrieved, on a symmetric
blue–white–red scale. This is the panel that tells you whether the fit is *done*.
Here it is white everywhere and its colourbar reads $10^{-15}$ — pure numerical
noise. Structure in this panel is the thing to worry about: coherent diagonal or
lobed patterns mean the model cannot reproduce the data, which usually means the
forward model is wrong (an unmodelled dispersive medium, say) rather than the
optimiser having failed.

**The pulse and spectrum (right block).** The retrieved temporal profile with its
transform-limited counterpart `TL`, and the retrieved spectrum with its phase and
a fitted polynomial giving GDD and TOD. Two numbers to check here: the retrieved
FWHM against `TL` — their ratio is how far from compressed the pulse is — and the
fitted GDD, which should come back as the 4 fs² we put in. The `truth` overlay is
present because we passed one.

**Convergence and marginals.** The error history on a log axis, plus the frequency
and delay marginals of measured and retrieved traces. The marginals are a fast
sanity check that survives even when the retrieval has gone wrong: they are
projections of the trace, so they must agree even if the phase is badly wrong.

## Next

- Compare the solvers on the same trace: [Choosing an algorithm](02_choosing_an_algorithm.md).
- Retrieve through a dispersive medium: [Dispersive retrieval](03_dispersive_retrieval.md).
- Clean and retrieve measured data: [Experimental workflow](04_experimental_workflow.md).
