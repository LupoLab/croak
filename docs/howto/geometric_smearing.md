# Modelling geometric time smearing

Crossing beams have different local arrival times across a focal spot. A
spectrometer that collects the isolated signal averages over those positions;
ignoring this response can bias the inferred pulse. This page shows how to
construct and use the reduced timing kernel. Start with the
[model overview](../user_guide/models.md), and use the
[worked example](../examples/geometric_smearing.md) to compare retrievals.

The kernel assumes achromatic local profiles and full signal collection. For
finite apertures or strongly chromatic focal fields, use the
[collection model](collection_aperture.md). The derivation is retained in
[the forward model](../explanation/forward_model.md#geometric-smearing).

## The quick version

```python
import croak
from croak.smearing import square_boxcars_kernel

kernel = square_boxcars_kernel(
    "pg",                    # PG (used for TG-FROG) or SD
    hole_diameter=1.0e-3,    # mask hole diameter D (m)
    hole_spacing=0.5e-3,     # edge-to-edge gap between holes (m)
    wavelength=260e-9,       # carrier
)
print(kernel.sigma_delta * 1e15)   # 0.357 fs

res = croak.retrieve(
    trace, g.omega, delays, "pg",
    algorithm="lbfgs-ad",
    smearing=kernel,
)
```

The same `smearing=` keyword works on {func}`~croak.forward.maketrace` and
{func}`~croak.forward_jax.maketrace_jax`, so you can generate a smeared trace and
look at it without running a retrieval:

```python
smeared = croak.maketrace(g.omega, delays, ew, "pg", smearing=kernel)
```

## What the kernel is

{class}`~croak.smearing.SmearingKernel` holds three numbers: the rms of the
gate-splitting parameter `sigma_p`, the rms of the delay offset `sigma_delta`, and
their correlation `rho`. `sigma_p` changes the *shape* of the nonlinear signal and
costs an explicit quadrature (`npoints`, default 5); `sigma_delta` is a pure
translation of the delay axis, evaluated by a Fourier-domain convolution on
a uniform delay grid. For correlated coordinates, the conditional mean and
variance depend on the gate-splitting node.

Two constructors build one:

- {func}`~croak.smearing.square_boxcars_kernel` — the standard folded square with
  four holes at $(\pm d,\pm d)$, $d = (\text{spacing} + D)/2$. This is the usual
  case, and the arm assignment follows from the interaction: for `"pg"` the
  scanned delay rides the **gate pair** — croak's operator convention
  $E(t)\,|E(t-\tau)|^2$ delays both gate arms together (an instrument that
  delays the probe instead is equivalent up to a reversal of the delay axis);
  `"sd"` puts it on the conjugated arm.
- {func}`~croak.smearing.kernel_from_arms` — fully general, for any three mask hole
  positions. Give them in the interaction's **role order**: `(probe, gate
  unconjugated, gate conjugated)` for PG, `(unconjugated A, unconjugated B,
  conjugated)` for SD.

The focal length never appears: the crossing angle scales as $1/f$ and the focal
spot as $f$, so it cancels. Within this ideal-hole approximation, the widths depend on $d/D$ and wavelength.

```{admonition} The three numbers are arrangement-specific
:class: warning
For a folded square BOXCARS mask the closed forms depend on **which arm carries
the scanned delay**, and they are not interchangeable:

| Interaction | Delay on | $\sigma_p\,/\,(d/D)(\lambda/c)$ | $\sigma_\delta\,/\,(d/D)(\lambda/c)$ | $\rho$ |
|---|---|---|---|---|
| `"sd"` | conjugated arm | 0.69487 | 0.34743 | 0 |
| `"pg"` | the gate pair (the usual TG-FROG case) | 0.49135 | 0.54934 | $1/\sqrt5$ |

So the convenient SD relations — $\sigma_p = 2\sigma_\delta$ with the two
uncorrelated — do **not** carry over to PG: there the pair is correlated and
$\sigma_\delta$ is 1.58× larger. {func}`~croak.smearing.square_boxcars_kernel`
selects the right pair from the `interaction` argument, which matters if you
are constructing a {class}`~croak.smearing.SmearingKernel` by hand from numbers
quoted for the other geometry.
```

```{admonition} Which way does your delay axis run?
:class: important
The kernels are built in **croak's** delay convention, where the PG signal is
$E(t)\,|E(t-\tau)|^2$. Only $\rho$ depends on it: reversing the delay axis sends
$\delta \to -\delta$ and hence $\rho \to -\rho$, for which
`square_boxcars_kernel(..., reverse_delay=True)` is provided.

Which way it runs depends on **which arm sits on the delay stage** — the same
$\chi^{(3)}$ nonlinearity either way. Scanning the gate pair gives
$E(t)\,|E(t-\tau)|^2$; scanning the probe instead (as in the `ModelPNPS` reference
simulation, and in TG setups generally) gives $|E(t)|^2 E(t-\tau_{\rm exp})$, which
is the same operator with $\tau = -\tau_{\rm exp}$.

Use the file's documented delay convention. The simulated loader can read its
`delay_convention` marker; older files may need reversal. Reverse the data axis
and set the kernel correlation consistently. Do not infer the convention from
a favourable retrieval or assume that every simulator file uses the same sign.
```

```{note}
Full collection permits an incoherent intensity average by Parseval's identity.
A finite aperture selects a coherent spatial field and can change delay-dependent
structure. The reduced kernel is not a general upper bound on blur or duration;
compare a suitable [collection model](collection_aperture.md).
```

## Fixed or fitted

**Fixed geometry** is the natural starting point: calculate the kernel from
measured mask dimensions. Changing mask spacing provides a useful controlled
test, but it can also change spatial collection and illumination. Model those
changes before assigning a difference solely to smearing.

```{warning}
The kernel is independent of the medium thickness, but the **duration offset it
induces is not** — how much the free pulse must absorb depends on the whole trace,
whose sensitivity to the input duration changes with the slab. Do not use a
"constant offset across thicknesses" test as the discriminator; use the mask
dependence. `examples/example_geometric_smearing.py` runs both sweeps and shows
the difference.
```

**Fitted width** adds a nuisance parameter. `fit_smearing=True` multiplies both
widths by one dimensionless scale while retaining their correlation. It tests
whether that reduced response can explain the trace; the estimate can also
absorb other model errors:

```python
res = croak.retrieve(
    trace, g.omega, delays, "pg",
    algorithm="lbfgs-ad",
    smearing=kernel,
    fit_smearing=True,
)
print(res.smear_scale)   # compare with the nominal scale of 1.0
```

A value near 1 is consistent with the nominal kernel, but does not uniquely
identify the mechanism. Inspect correlations with the field and other parameters.

```{warning}
Avoid freeing both smearing width and thickness without independent information.
They can be strongly correlated with each other and with pulse phase. Fix
calibrated quantities first, then assess parameter covariance and sensitivity;
identifiability depends on the particular measurement.
```

Uncertainty follows the same route as the other extra parameters:
{func}`~croak.covariance.parameter_covariance` accepts `smearing` and
`fit_smearing` and reports `sigma_smear`.

### Splitting the two channels

The kernel has two channels: the delay offset $\vartheta$ (also written
$\delta$ above and in the reduced-kernel API) (a pure blur along
the delay axis) and the gate-shape parameter $p$ (a symmetric split of the gate
pair, which for a chirped gate shears the trace along frequency). The joint
multiplier scales both together, which is the right physical prior — both
derive from the same $d/D$ — but as a *diagnostic* it conflates them. Setting
`fit_smearing_split=True` fits the two widths as independent multipliers:

```python
res = croak.retrieve(
    trace, g.omega, delays, "pg",
    algorithm="lbfgs-ad",
    smearing=kernel,
    fit_smearing=True,
    fit_smearing_split=True,
)
print(res.smear_scale)         # p (gate-shape) channel multiplier
print(res.smear_scale_delta)   # delta (delay-blur) channel multiplier
```

The split is exact, not an approximation: the bivariate-normal kernel scales
channel-wise, with the $p$ node positions carrying one factor and the
conditional delay mean/width the other, at fixed correlation $\rho$.

```{note}
For a transform-limited **Gaussian** field, splitting the gate changes only its
amplitude, which a fitted intensity scale can absorb. This is not true for every
transform-limited field: a sum of Gaussians gives a time-dependent gate ratio.
A split-width estimate must therefore be interpreted using the pulse shape,
phase and parameter correlations, not simply whether the field is chirped.
```

### Temporal phase and spectral dispersion

Write a Gaussian field as $E(u)=\exp[-u^2/(4\sigma^2)]\exp[i b u^2/2]$.
Its split gate is

```{math}
E(u+p/2)E^*(u-p/2)
=\exp[-u^2/(2\sigma^2)]\exp[-p^2/(8\sigma^2)]\exp[i bup].
```

For $b=0$ this is attenuation only. With the forward Fourier sign
$\exp(+i\Omega t)$, the gate frequency shift is $-bp$. For a general temporal
phase $\phi$, it is

```{math}
\delta\omega_g(u;p)=-[\phi'(u+p/2)-\phi'(u-p/2)]
=-p\phi''(u)-\frac{p^3}{24}\phi^{(4)}(u)+\cdots.
```

Temporal-phase derivatives are not the GDD, TOD and FOD, which are derivatives
of **spectral** phase $\Phi(\Omega)$. In particular, a symmetric spectral
amplitude with pure odd spectral phase can give a real temporal field with
phase jumps rather than continuous temporal chirp. For Gaussian intensity of
rms width $\sigma_t$, a quadratic spectral phase $\Phi_2$ gives
$b=-\Phi_2/(4\sigma_{t,0}^4+\Phi_2^2)$, where $\sigma_{t,0}$ is the
transform-limited intensity width. This relation uses the Fourier convention
above; it is not a general conversion for non-Gaussian fields.

{func}`~croak.covariance.parameter_covariance` accepts `fit_smearing_split`
and additionally reports `sigma_smear_delta`.

## Cost

About $(K+1)/2$ times the unsmeared model, where $K$ is `npoints` — roughly 3× at
the default 5 nodes. The quadrature nodes are exactly antisymmetric, so each
shifted field is built once and used twice, and the `p` axis is a `jax.vmap` batch.
The delay integral adds nothing measurable.

The delay axis must be **uniformly spaced** (the delay integral is a convolution
along it) and comfortably wider than the kernel; both are checked, and violations
raise.

## Solver support

| Option | Solvers | Requirements |
|---|---|---|
| `smearing` | `warm-lbfgs`, `lbfgs-ad`, `lm`, `lm-optx`, `lbfgs-optx`, `cma-es` | PG (TG) or SD; uniform delay axis |
| `fit_smearing` | `warm-lbfgs`, `lbfgs-ad`, `lm`, `lm-optx`, `cma-es` | a `smearing` kernel |
| `fit_smearing_split` | `warm-lbfgs`, `lbfgs-ad`, `lm`, `lm-optx`, `cma-es` | `fit_smearing` |

`copra`, `copra-jax`, `lbfgs` and `lbfgs-hand` do not implement this spatial
mixture. Their current projection or analytic derivative path is restricted to
a local signal. Asking for unsupported smearing raises an error. This does not
exclude other vector or mixed-state projection formulations in principle.

SHG raises too: geometric smearing here is defined for the three-arm PG/SD forms.

## In the GUI

The Retrieve stage has a **Geometrical smearing** group, greyed out unless the
chosen solver supports it:

- *Enable geometrical smearing*
- *Hole diameter (mm)* and *Hole spacing (mm)* (edge to edge) — the physical inputs
- *σ delay (fs)* — derived from them at the measured carrier, and **editable**: it
  back-computes the hole spacing, holding the diameter, so you can dial a width in
  directly. The same coupling as *Thickness* ↔ *Δz* ↔ *Npoints* in the group above.
  It has a floor at zero spacing (holes touching).
- *Quadrature nodes*

*Fit smearing width* sits in **Extra fit parameters** and is enabled only once the
kernel is on. When it has been fitted, the post-retrieval status line reports the
result next to the other fitted extras, as the physical delay width and the
multiplier it came from:

```
Done. R = 0.1459%   |   retrieved FWHM 1.12 fs   |   …   |   smearing σ 14.24 fs (×1.457)
```

The width shown is `smear_scale × sigma_delta` of the kernel the run used — the
two kernel widths scale together with the multiplier
({meth}`~croak.smearing.SmearingKernel.scaled`), so one number describes the fit.
Selecting *Reuse previous result* on a re-run starts the fit from that multiplier
instead of from 1.0.

The mask geometry is what is stored and what reaches the forward model; the
displayed width is derived from it, always at the carrier of the loaded
measurement. A fitted multiplier is deliberately **not** written back into the
diameter/spacing controls: the geometry is a measured property of the instrument,
not a readout to be overwritten by a fit. To adopt a fitted width as the new
nominal, type it into *σ delay (fs)*, which back-computes the spacing.

## Worked example

`examples/example_geometric_smearing.py` synthesises a trace with a known kernel
and retrieves it with and without, printing the duration inflation:

```
uv run python examples/example_geometric_smearing.py
```
