# Materials and mirrors

Use material data to describe a nonlinear medium or to propagate a retrieved
pulse through a known optical path. Use mirror data to apply or remove the
spectral phase and attenuation of reflections during post-processing. These
operations have different roles: the [forward model](../explanation/forward_model.md)
fits a measurement, while [dispersion tuning](../tutorials/05_dispersion_tuning.md)
explores the pulse after retrieval.

The optical data come from five sources:

- **built-in materials** (fused silica, BK7, CaF₂, sapphire, …), from
  [optmat](https://optmat.readthedocs.io);
- **custom materials** you register yourself, from a formula or measured data;
- **gases** (air, helium, argon, …), also from optmat, at a chosen pressure and
  path length;
- **chirped mirrors**, from bundled data or your own measurement;
- **beam-path coating mirrors** (Si, MgF₂/Al) to back-propagate out of the pulse.

Each dataset has a valid wavelength range, and mirror data are given for a
polarisation and angle of incidence.

## Built-in materials

croak takes its refractive indices from
[optmat](https://optmat.readthedocs.io), a curated library of optical constants.
Each built-in name uses **optmat's recommended model** of that material. optmat
records the model's sources, its validity range and why it was chosen. Its indices
are absolute (relative to vacuum). The uniaxial crystals use their ordinary index,
as a plate cut with the optic axis along its normal shows at normal incidence.

| Name | Material | optmat model |
|---|---|---|
| `SiO2` | fused silica | Franta et al.'s universal dispersion model, refitted to absolute refractometry (24.8 nm – 125 µm) |
| `BK7` | SCHOTT N-BK7 | SCHOTT's catalogue Sellmeier formula |
| `CaF2` | calcium fluoride | Daimon & Masumura (2002), at 20 °C |
| `BaF2` | barium fluoride | Malitson (1964) |
| `MgF2` | magnesium fluoride, ordinary ray | Dodge (1984) |
| `Al2O3` | sapphire, ordinary ray | Weber (2003) |
| `Si` | crystalline silicon (opaque below 1.1 µm) | Franta et al.'s model refitted at 20 °C |

```python
import croak
print(croak.materials.MATERIALS)   # SiO2, BK7, CaF2, BaF2, MgF2, Al2O3, Si
print(croak.materials.material_model("BK7").citation)   # the model's source
```

{func}`croak.materials.material_model` returns the optmat model behind a name, with
its `name`, `citation`, `wavelength_range` and `identity`, for recording what a
result was computed with.

Pass a name as `material=` to any solver or to {func}`~croak.forward.maketrace` to
propagate through a slab (see [Dispersive retrieval](../tutorials/03_dispersive_retrieval.md)),
or to {func}`~croak.dispersion.apply_dispersion` to add or remove a known thickness
of glass:

```python
# add 2 mm of fused silica to a retrieved spectrum
ew = croak.apply_dispersion(g.omega, res.omega0, res.spectrum,
                           material_thicknesses={"SiO2": 2e-3})
```

Names are resolved case-insensitively. The raw index and propagation constant are
available directly as {func}`croak.materials.refractive_index` and
{func}`croak.materials.beta`.

The frequency grid of a retrieval spans far beyond any model's stated range, where
the pulse carries no energy. There the models are **extrapolated without a
warning**. A grid point at zero or negative frequency has no index: it gives `nan`,
and {func}`~croak.materials.beta` sets its phase to zero.

Earlier croak versions also offered `SiO2-Franta`, Franta et al.'s tabulated fused
silica. optmat's `SiO2` is Franta's model refitted to absolute measurements, so the
name now resolves to `SiO2`, and saved sessions that use it load unchanged.

## Custom materials

To use a material croak doesn't ship, register a callable `n(wavelength_m)` that
returns the real refractive index (returning `nan` outside its validity range so
the propagation constant is masked there):

```python
import numpy as np

def n_glass(lam_m):                          # a Cauchy formula, λ in metres
    lam_um = lam_m * 1e6
    return 1.5046 + 0.00420 / lam_um**2

croak.materials.register_material("MyGlass", n_glass)
```

A registered material is then usable **everywhere a built-in name is** — including
`material="MyGlass"` in {func}`croak.retrieve`. Registered names take precedence
over the built-in set and are resolved case-insensitively; remove one with
{func}`croak.materials.unregister_material`.

Measured data are noisy, and the dispersion quantities differentiate $n(\lambda)$
up to third order, which amplifies that noise. Build the callable from samples with
{func}`croak.materials.make_tabulated_material`, which fits a smoothing spline
(in log-wavelength, with its smoothing chosen by generalised cross-validation) and
returns `nan` outside the sampled range. {func}`croak.materials.read_nk_csv` reads
the stacked `wl,n` / `wl,k` CSV layout some optical-constants databases export:

```python
lam_m, n = croak.materials.read_nk_csv("my_glass.csv")
n_func, (lo, hi) = croak.materials.make_tabulated_material(lam_m, n)
croak.materials.register_material("MyMeasuredGlass", n_func)
```

## Gases

A gas is dispersive like a solid, but its refractive index depends on the gas
*density* — and hence on pressure and temperature — as well as on wavelength.
{func}`croak.gases.gas_refractive_index` builds an `n(wavelength_m)` callable for a
gas at a chosen pressure (bar) and temperature (K, defaulting to 20 °C). The
dispersion is optmat's recommended model of the gas's polarisability $\alpha$; for
air, optmat's dry air, the sum of its constituents' models. The index follows from
the Lorentz–Lorenz relation at the **real-gas** density $N = P/(Z k_B T)$, with the
compressibility factor $Z$ of the gas's reference equation of state (CoolProp,
through optmat):

```python
import croak

# 1.5 bar of helium, at room temperature
n_he = croak.gases.gas_refractive_index("He", 1.5)
croak.materials.register_material("He-cell", n_he)

# add 30 cm of that helium to a retrieved spectrum (path length is the "thickness")
ew = croak.apply_dispersion(g.omega, res.omega0, res.spectrum,
                           material_thicknesses={"He-cell": 0.30})
```

Because a gas is registered as an ordinary material, it works **everywhere a
material name does** — including the dispersion auto-compressor, which sweeps the
path length at the set pressure. The available gases are in
{data}`croak.gases.GASES`: `Air` and optmat's species (`He`, `Ne`, `Ar`, `Kr`,
`Xe`, `H2`, `D2`, `N2`, `O2`, `CH4`, `SF6`, `N2O`, `CO2`, `O3`, `CO`). The earlier
name `Helium` still resolves to `He`. {func}`croak.gases.gas_model` returns the
optmat model behind a name. The real gas matters most for the heavy gases: at 20 °C
and 10 bar its index differs from the ideal gas's by 0.5 % for helium and air, 6 %
for CO₂ and xenon and 15 % for SF₆. A state where the gas would condense raises a
`ValueError`.

The GUI's **Gases** group selects the gas, pressure and path length. Pressure
ranges from 0–10 bar in 1 mbar steps; path length ranges to ±999 cm. Negative
length removes the modelled phase, and **Auto** optimises over the allowed range.

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

For good compressor mirrors `R ≳ 99 %` in band, so the loss is small; pass only
`phase_fn` to `register_mirror` for a phase-only calculation.

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

phase_fn, amp_fn, _ = mirrors.load_coating("Si @ 45°", "s")
croak.dispersion.register_mirror("Si45", phase_fn, amp_fn)
```

A **negative** bounce count removes the mirror: `apply_dispersion` then applies
`exp(-iNφ)` *and* divides the spectrum by `r(λ)^N` (the net amplitude gain is
clamped so a small reflectivity cannot blow it up). Several registered coatings
back-propagate in one call:

```python
upstream = croak.apply_dispersion(
    g.omega, res.omega0, res.spectrum,
    mirror_bounces={"MgF2Al": -6, "Si45": -2},
)
```

Back-propagation cannot recover spectral content where the reflectivity is
close to zero; the clamp limits the noise it amplifies there. Keep the spectrum
inside the coating's valid wavelength range.

At normal incidence the `s` and `p` data coincide; at oblique incidence they
differ, so pick the polarisation that matches your beam.

## See also

- [Dispersion tuning](../tutorials/05_dispersion_tuning.md) — compressing a pulse by
  removing residual dispersion (Taylor, material and mirror).
- [The dispersive forward model](../explanation/forward_model.md) — retrieving
  *through* a material slab.
- {mod}`croak.materials`, {mod}`croak.gases`, {mod}`croak.mirrors` and
  {mod}`croak.dispersion` in the
  [API reference](../reference/api/workflow.md).
