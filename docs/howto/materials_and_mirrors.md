# Materials and mirrors

croak models dispersion from three sources, all of which feed the same machinery —
the dispersive [forward model](../explanation/forward_model.md) (retrieving
*through* a medium) and the [dispersion tuner](../tutorials/05_dispersion_tuning.md)
(compressing a retrieved pulse):

- **built-in Sellmeier materials** (fused silica, CaF₂, …);
- **custom materials** you register yourself, including pages from the
  [refractiveindex.info](https://refractiveindex.info) database;
- **gases** (helium, air) at a chosen pressure and path length;
- **chirped mirrors**, from bundled data or your own measurement;
- **beam-path coating mirrors** (Si, MgF₂/Al) to back-propagate out of the pulse.

This guide shows how to use and extend each one.

## Built-in materials

A handful of common optics materials ship with Sellmeier coefficients:

```python
import croak
print(croak.materials.MATERIALS)   # SiO2, BK7, CaF2, BaF2, MgF2, SiO2-Franta
```

Pass a name as `material=` to any solver or to {func}`~croak.forward.maketrace` to
propagate through a slab (see [Dispersive retrieval](../tutorials/03_dispersive_retrieval.md)),
or to {func}`~croak.dispersion.apply_dispersion` to add or remove a known thickness
of glass:

```python
# add 2 mm of fused silica to a retrieved spectrum, then take it back out
ew = croak.apply_dispersion(g.omega, res.omega0, res.spectrum,
                           material_thicknesses={"SiO2": 2e-3})
```

Names are resolved case-insensitively. The raw index and propagation constant are
available directly as {func}`croak.materials.refractive_index` and
{func}`croak.materials.beta`.

## Custom materials

To use a material croak doesn't ship, register a callable `n(wavelength_m)` that
returns the real refractive index (returning `nan` outside its validity range so
the propagation constant is masked there):

```python
import numpy as np

def n_sapphire(lam_m):                       # ordinary-ray Sellmeier, λ in metres
    u2 = (lam_m * 1e6) ** 2                   # µm²
    return np.sqrt(1 + 1.4313493*u2/(u2 - 0.0726631**2)
                     + 0.65054713*u2/(u2 - 0.1193242**2)
                     + 5.3414021*u2/(u2 - 18.028251**2))

croak.materials.register_material("Sapphire", n_sapphire)
```

A registered material is then usable **everywhere a built-in name is** — including
`material="Sapphire"` in {func}`croak.retrieve`. Registered names take precedence
over the built-in set and are resolved case-insensitively; remove one with
{func}`croak.materials.unregister_material`.

## The refractiveindex.info database

The optional `croak[ridb]` extra bridges to the
[refractiveindex.info](https://refractiveindex.info) database, giving you thousands
of measured and modelled materials without writing Sellmeier formulae yourself.
Install it with `pip install croak[ridb]` (or `uv sync --extra ridb`); the first
lookup downloads the full database (~hundreds of MB) to
`~/.refractiveindex.info-database`.

```python
from croak import refractive_db

refractive_db.available()                    # False if the extra isn't installed

# browse the catalogue: shelf -> book -> page
refractive_db.shelves()                      # [(shelf_id, label), ...]
refractive_db.books("glass")                 # [(book_id, label), ...]
refractive_db.pages("glass", "SCHOTT-BK7")   # [(page_id, label), ...]

# build an n(λ) callable and register it under a friendly name
n_fn, wl_range = refractive_db.make_material("glass", "SCHOTT-BK7", "SCHOTT")
croak.materials.register_material("BK7-schott", n_fn)
```

The GUI's dispersion stage uses exactly these calls to populate its material
picker. `make_material` returns `(n_of_lambda_m, wl_range_m)`; the callable returns
`nan` outside the page's validity range.

## Gases

A gas is dispersive like a solid, but its refractive index depends on the gas
*density* — and hence on pressure and temperature — as well as on wavelength.
{func}`croak.gases.gas_refractive_index` builds an `n(wavelength_m)` callable for a
gas at a chosen pressure (bar) and temperature (K, defaulting to room
temperature), using Sellmeier data ported from Luna.jl. The index follows
$n = \sqrt{1 + \chi_1}$ with the linear susceptibility scaled from its reference
value (1 bar, 0 °C) by the **ideal-gas law**, $\chi_1 \propto P/T$:

```python
import croak

# 1.5 bar of helium, at room temperature
n_he = croak.gases.gas_refractive_index("Helium", 1.5)
croak.materials.register_material("He-cell", n_he)

# add 30 cm of that helium to a retrieved spectrum (path length is the "thickness")
ew = croak.apply_dispersion(g.omega, res.omega0, res.spectrum,
                           material_thicknesses={"He-cell": 0.30})
```

Because a gas is registered as an ordinary material, it works **everywhere a
material name does** — including the dispersion auto-compressor, which sweeps the
path length at the set pressure. The available gases are in
{data}`croak.gases.GASES` (`('Helium', 'Air')`). The ideal-gas scaling is accurate
to ~0.1 % for these gases up to a few bar; it is an approximation at high pressure
(tens of bar), where a real-gas equation of state would differ.

The GUI's dispersion stage exposes this as a **Gases** group: pick a gas, set the
pressure and the path length, and the pulse updates live. The pressure control
spans 0–10 bar at **millibar** resolution — the spin box, its arrows and the slider
all step by 1 mbar, so a few-mbar residual pressure is as easy to dial as a bar of
fill gas. (10 bar is also about where the ideal-gas scaling above stops being
quantitative.) The path length runs to **±999 cm**, covering metre-scale purge
paths; negative pre-compensates, and **Auto** optimises over the full range.

## Chirped mirrors

Chirped (dispersive) mirrors apply a designed spectral phase per reflection. Their
dispersion is data-dependent, so {data}`croak.dispersion.MIRRORS` starts empty and
you populate it with {func}`~croak.dispersion.register_mirror`. Once registered, a
mirror is applied by *number of bounces* through `mirror_bounces`.

### Built-in mirror data

Several mirrors are bundled (ported from Luna.jl), keyed by name in
{data}`croak.mirrors.BUILTIN_MIRRORS`. The `mirrors` submodule is not pulled into
the top-level namespace by `import croak`, so import it explicitly:

```python
from croak import mirrors

print(list(mirrors.BUILTIN_MIRRORS))
# ['PC70', 'HD59', 'PC147', 'PC1611', 'PC1821', 'HD120', 'ThorlabsUMC']
```

{func}`croak.mirrors.load_builtin` reads the data and returns a per-bounce
`phase_fn`, an application-ready amplitude (reflectivity) `amp_fn` and the
mirror's valid wavelength range. Register both, then add bounces when tuning;
each bounce applies the phase *and* attenuates by the measured reflectivity:

```python
phase_fn, amp_fn, valid_range = mirrors.load_builtin("PC70")
croak.dispersion.register_mirror("PC70", phase_fn, amp_fn)

# compress a retrieved pulse with 6 bounces off the PC70 pair
compressed = croak.apply_dispersion(g.omega, res.omega0, res.spectrum,
                                   mirror_bounces={"PC70": 6})
```

(For good compressor mirrors `R ≳ 99 %` in band, so the loss is small; pass only
`phase_fn` to `register_mirror` for the old phase-only behaviour.)

`mirror_bounces` can be combined with `gdd`/`tod`/`fod` and `material_thicknesses`
in a single {func}`~croak.dispersion.apply_dispersion` call, and is one of the knobs
the [dispersion-tuning](../tutorials/05_dispersion_tuning.md) auto-compressor can
sweep.

### Custom mirrors

For a mirror you measured yourself, build the phase function from arrays with
{func}`croak.mirrors.mirror_from_arrays` — either a sampled spectral **phase**
(`mode="phase"`) or **GDD** vs wavelength (`mode="gdd"`, double-integrated for you).
The overall group delay is removed automatically (it only shifts the pulse in time):

```python
lam_m = ...        # wavelength samples (m)
gdd   = ...        # GDD per bounce (s²) at each wavelength

phase_fn, valid_range = mirrors.mirror_from_arrays(lam_m, gdd, mode="gdd")
croak.dispersion.register_mirror("MyMirror", phase_fn)
```

## Beam-path coating mirrors (back-propagation)

Simple beam-path optics — bare silicon, MgF₂-coated aluminium — are not used to
compress a pulse but rather sit in the measurement path; you usually want to
**remove** (back-propagate) their dispersion to recover the pulse upstream. These
are keyed in {data}`croak.mirrors.BUILTIN_COATINGS` (by coating and angle of
incidence) and loaded with {func}`croak.mirrors.load_coating`, which returns the
per-bounce phase, an **amplitude** (reflectivity) function for the chosen
polarisation, and the valid range:

```python
phase_fn, amp_fn, valid_range = mirrors.load_coating("MgF2/Al @ 0°", "s")
croak.dispersion.register_mirror("MgF2Al", phase_fn, amp_fn)   # phase + amplitude
```

A **negative** bounce count removes the mirror: `apply_dispersion` then applies
`exp(-iNφ)` *and* divides the spectrum by `r(λ)^N` (the net amplitude gain is
clamped so a small reflectivity cannot blow it up). Several mirrors back-propagate
in one call:

```python
upstream = croak.apply_dispersion(
    g.omega, res.omega0, res.spectrum,
    mirror_bounces={"Si": -1, "MgF2Al": -6, "MgF2Al_45s": -2},
)
```

At normal incidence the `s` and `p` data coincide; at oblique incidence they
differ, so pick the polarisation that matches your beam.

## See also

- [Dispersion tuning](../tutorials/05_dispersion_tuning.md) — compressing a pulse by
  removing residual dispersion (Taylor, material and mirror).
- [The dispersive forward model](../explanation/forward_model.md) — retrieving
  *through* a material slab.
- {mod}`croak.materials`, {mod}`croak.gases`, {mod}`croak.mirrors`,
  {mod}`croak.refractive_db` and {mod}`croak.dispersion` in the
  [API reference](../reference/api/workflow.md).
