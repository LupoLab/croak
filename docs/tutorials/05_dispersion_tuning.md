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

# Dispersion tuning

Once you have a retrieved pulse you often want to ask: *how short could it be if I
removed its residual chirp?* {mod}`croak.dispersion` applies Taylor and material
dispersion to a spectrum and auto-tunes it to maximise the temporal peak power —
i.e. to compress the pulse.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import croak

def center(t, intensity):
    i = int(np.argmax(intensity))
    return t - t[i], intensity / intensity.max()

def temporal_intensity(grid, ew):
    return np.abs(grid.ifft(ew)) ** 2

g = croak.Grid(128, dt=0.4e-15)
omega0 = croak.maths.wlfreq(800e-9)
```

## A chirped pulse

We retrieve a chirped pulse (it carries GDD and TOD), so there is real dispersion
to remove.

```{code-cell} ipython3
ew_true = croak.gaussian_pulse(g, 5e-15, phases=[40e-30, 300e-45])
delays = np.linspace(-60e-15, 60e-15, 90)
trace = croak.maketrace(g.omega, delays, ew_true, "pg")

res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="copra",
                    maxiters=250, rng=np.random.default_rng(0))
I_retr = temporal_intensity(g, res.spectrum)
print(f"trace error R  = {res.error:.2e}")
print(f"retrieved FWHM = {croak.maths.fwhm(*center(g.t, I_retr)) * 1e15:.2f} fs")
```

## Auto-compress

{func}`~croak.dispersion.auto_taylor` searches for the `(GDD, TOD, FOD)` that
**maximise the temporal peak power**. It never returns a result worse than leaving
the pulse untouched. {func}`~croak.dispersion.apply_dispersion` then adds that phase
to the spectrum.

```{code-cell} ipython3
gdd, tod, fod = croak.dispersion.auto_taylor(g, res.spectrum, res.omega0)
print(f"compensating dispersion: GDD = {gdd/1e-30:.0f} fs²,  "
      f"TOD = {tod/1e-45:.0f} fs³")

compressed = croak.apply_dispersion(g.omega, res.omega0, res.spectrum,
                                   gdd=gdd, tod=tod, fod=fod)
I_comp = temporal_intensity(g, compressed)

print(f"peak-power gain  = {I_comp.max() / I_retr.max():.2f}×")
print(f"compressed FWHM  = {croak.maths.fwhm(*center(g.t, I_comp)) * 1e15:.2f} fs")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(5.5, 4))
for label, I in [("retrieved", I_retr), ("auto-compressed", I_comp)]:
    tc, ic = center(g.t, I)
    ax.plot(tc * 1e15, ic, label=label)
ax.set_xlim(-30, 30)
ax.set_xlabel("time (fs)")
ax.set_ylabel("intensity (norm.)")
ax.set_title("compression by residual-dispersion removal")
ax.legend()
fig.tight_layout()
```

Removing the fitted dispersion flattens the pulse's spectral phase, concentrating
the energy into a shorter, more intense pulse.

## From intensity to peak power

The plots above are normalised, so the "peak-power gain" is a *ratio*. If you know
the measured pulse energy, pass it to {func}`~croak.processing.process_result` as
`energy=` (joules) to read off the absolute **peak power** in watts. With the
intensity normalised to unit peak, the peak power is `energy / ∫I dt` — the energy
divided by the effective duration — so compressing the pulse raises it towards the
transform-limited maximum at fixed energy:

```python
pr = croak.process_result(res, measured=td.trace, energy=100e-6)  # 100 µJ
print(f"peak power     = {pr.peak_power/1e6:.1f} MW")
print(f"TL peak power  = {pr.peak_power_tl/1e6:.1f} MW")  # the maximum achievable
```

In the GUI's Dispersion stage this is shown live as you tune the sliders. See
[Post-processing](../howto/postprocessing.md#absolute-power-and-peak-power).

## Other knobs

{func}`~croak.dispersion.apply_dispersion` also takes explicit GDD/TOD/FOD if you
want to dial in a known amount, or `material_thicknesses` / `mirror_bounces` to
add real glass or mirror dispersion. A **negative** bounce count *removes*
(back-propagates) a mirror out of the measured pulse, and beam-path coating mirrors
loaded with {func}`~croak.mirrors.load_coating` also divide out their reflectivity.
The corresponding auto-tuner for a material is
{func}`~croak.dispersion.auto_material` (e.g. optimal wedge insertion), and you can
register mirrors with {func}`~croak.dispersion.register_mirror`. The
[Materials and mirrors](../howto/materials_and_mirrors.md) how-to covers the built-in
chirped mirrors ({data}`croak.mirrors.BUILTIN_MIRRORS`) and coatings
({data}`croak.mirrors.BUILTIN_COATINGS`), building a mirror from your own measurement,
custom materials and the refractiveindex.info database.

```{admonition} Sign convention
:class: note
A *positive* GDD adds $+\tfrac12\,\mathrm{GDD}\,\omega^2$ of spectral phase,
matching {func}`croak.pulses.gaussian_pulse`. Applying a material's dispersion and
then its negative cancels exactly.
```

## Next

- [Post-processing](../howto/postprocessing.md) — the fitted GDD/TOD/FOD report.
- {mod}`croak.dispersion` in the [API reference](../reference/api/workflow.md).
