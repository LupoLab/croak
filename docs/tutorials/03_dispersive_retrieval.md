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

# Dispersive retrieval

This is croak's headline capability: retrieving a pulse measured through a
**dispersive nonlinear medium**, where classic thin-medium FROG misjudges the
pulse. We simulate an ultraviolet pulse gated in a thin fused-silica slab and
recover it with the [dispersive forward model](../explanation/forward_model.md).

```{code-cell} ipython3
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import croak
from croak.maths import wlfreq

def center(t, intensity):
    """Peak-centre and peak-normalise (retrieval is defined up to a time shift)."""
    i = int(np.argmax(intensity))
    return t - t[i], intensity / intensity.max()

def fwhm_fs(t, intensity):
    tc, ic = center(t, intensity)
    return croak.maths.fwhm(tc, ic) * 1e15

g = croak.Grid(256, dt=0.15e-15)
omega0 = wlfreq(260e-9)                        # 260 nm carrier
ew = croak.gaussian_pulse(g, 1.5e-15)          # transform-limited, 1.5 fs
delays = np.linspace(-18e-15, 18e-15, 90)

I_true = np.abs(g.ifft(ew)) ** 2
print(f"true temporal FWHM = {fwhm_fs(g.t, I_true):.2f} fs")
```

The pulse is transform-limited, so **every** femtosecond of stretching we see
later comes from the medium and nothing else. A 1.5 fs pulse at 260 nm is where
this matters: its bandwidth is enormous, and fused silica is strongly dispersive
in the deep ultraviolet.

## A trace through 50 µm of SiO₂

Passing `material`, `thickness`, `npoints` and the carrier `omega0` to
{func}`~croak.maketrace` models the propagation of both the input and the signal
fields inside the slab. We use the **PG** interaction (the transient-grating
kernel); dispersive propagation is supported for PG and SD, not SHG.

Fifty micrometres is a *thin* window by any normal standard — a fifth the
thickness of a microscope coverslip:

```{code-cell} ipython3
disp = dict(material="SiO2", thickness=50e-6, npoints=20, omega0=omega0)
print(f"GDD of the slab at 260 nm = "
      f"{croak.dispersion.material_gdd('SiO2', omega0, disp['thickness']):.1f} fs²")

trace = croak.maketrace(g.omega, delays, ew, "pg", **disp)
trace.shape
```

Ten femtoseconds squared sounds like very little. Against a 1.5 fs pulse it is
not.

## Dispersion-aware retrieval

The retrieval must use the *same* dispersive model — pass the slab parameters to
{func}`~croak.retrieve` as well. This is the paper's "D-COPRA" (COPRA with a
material set).

```{code-cell} ipython3
near = croak.gaussian_pulse(g, 2.0e-15)
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="copra",
                    guess=near, maxiters=300, rng=np.random.default_rng(0), **disp)
print(f"trace error R  = {res.error:.2e}")
print(f"recovered FWHM = {fwhm_fs(res.t, res.intensity_t):.2f} fs"
      f"   (true {fwhm_fs(g.t, I_true):.2f} fs)")
```

The dispersion-aware retrieval recovers the true duration. The full summary shows
what that means panel by panel, with the known pulse overlaid:

```{code-cell} ipython3
truth = croak.TruthPulse.from_spectrum(g, ew, omega0)
f0 = omega0 / (2 * np.pi)          # PG is degenerate: signal at the fundamental

fig = croak.plot_retrieval(res, measured=trace, truth=truth,
                           lam_min=200e-9, lam_max=380e-9,
                           flim=(f0 - 0.55e15, f0 + 0.55e15))
fig.set_size_inches(11, 8)
fig                                # return the Figure so the notebook displays it
```

Look at the two halves of this figure together, because the contrast between them
*is* the dispersive forward model.

The **trace** is strongly distorted. It is not the symmetric bowtie of a
thin-medium PG-FROG measurement: it leans, and it trails off toward negative
delay. That asymmetry is the slab — the input replicas and the generated signal
each accumulating phase as they propagate through 50 µm of glass.

The **pulse**, meanwhile, comes back clean. The retrieved, transform-limited and
true curves all read 1.50 fs and lie on top of one another, and the fitted GDD and
TOD are zero to the digits shown: a flat spectral phase. Every bit of the
distortion visible on the left has been attributed to the medium, where it
belongs, instead of to the pulse. The residual is at $10^{-14}$, so the model
accounts for the measurement completely.

## What a thin-medium model does

Retrieving the *same* trace **without** the slab model — as classic FROG would —
misattributes the medium's dispersion to the pulse.

```{code-cell} ipython3
res_thin = croak.retrieve(trace, g.omega, delays, "pg", algorithm="copra",
                         guess=near, maxiters=300, rng=np.random.default_rng(0))
print(f"thin-model trace error R = {res_thin.error:.2e}")
print(f"thin-model FWHM          = {fwhm_fs(res_thin.t, res_thin.intensity_t):.2f} fs"
      f"   (true {fwhm_fs(g.t, I_true):.2f} fs)")
```

```{code-cell} ipython3
t_true, I_true_n = center(g.t, I_true)
t_disp, I_disp_n = center(res.t, res.intensity_t)
t_thin, I_thin_n = center(res_thin.t, res_thin.intensity_t)

fig, ax = plt.subplots(figsize=(5.5, 4))
ax.plot(t_true * 1e15, I_true_n, lw=3, alpha=0.4, label="true")
ax.plot(t_disp * 1e15, I_disp_n, "--", lw=1.5, label="dispersive model")
ax.plot(t_thin * 1e15, I_thin_n, ":", lw=1.5, label="thin model")
ax.set_xlim(-8, 8)
ax.set_xlabel("time (fs)")
ax.set_ylabel("intensity (norm.)")
ax.legend()
ax.set_title("dispersive vs thin-medium retrieval")
fig.tight_layout()
```

The thin model returns a pulse **more than twice as long as the truth** — 50 µm
of glass, and a 1.5 fs measurement becomes a 3 fs one. Its dispersion has been
misattributed to the pulse, because a thin-medium model has nowhere else to put
it.

Note the failure mode here. The thin model's trace error
is around $3\times10^{-3}$ — **0.3 %**. Reported on its own, in a paper or a lab
notebook, that is the kind of number usually reported as a well-converged FROG
retrieval — here, against the wrong forward model:

```{code-cell} ipython3
print(f"dispersive model:  R = {res.error:.1e}   FWHM = {fwhm_fs(res.t, res.intensity_t):.2f} fs")
print(f"thin model:        R = {res_thin.error:.1e}   FWHM = {fwhm_fs(res_thin.t, res_thin.intensity_t):.2f} fs")
print(f"truth:                              FWHM = {fwhm_fs(g.t, I_true):.2f} fs")
```

A small trace error only means the model reproduces the *measurement*. It says
nothing about whether the model is the right one — and no amount of iterating, no
better optimiser and no random restart will fix a missing term in the physics. The
only defence is to model the medium, which costs one extra keyword.

The effect grows with shorter pulses and deeper into the ultraviolet: in the
[companion paper](../explanation/validation.md), a thin-medium model retrieves a
1 fs pulse at 260 nm as 1.4–2.6 fs depending on the substrate thickness, while
the dispersive model recovers it at every thickness from 1 to 40 µm.

```{note}
Dispersion is a *parameter*, not a separate algorithm — `material=`, `thickness=`,
`npoints=` and `omega0=` work on every solver. If the thickness is not known
precisely, the autodiff solvers can [fit it from the data
itself](../howto/fitting_thickness_tau0.md), and its residual uncertainty
[propagates to the duration](../howto/uncertainty.md).
```

## Next

- The physics: [The forward model](../explanation/forward_model.md) and
  [validation](../explanation/validation.md).
- Compress residual dispersion: [Dispersion tuning](05_dispersion_tuning.md).
