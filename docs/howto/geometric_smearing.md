# Modelling geometric time smearing

In a non-collinear BOXCARS geometry the input beams reach the interaction plane
with tilted pulse fronts, so the relative arrival time between arms varies across
the focal spot and the spectrometer averages over it. Left out of the model, that
instrument response is absorbed by the free pulse and comes back as a **retrieved
duration that is too long**. Put it into the forward model and it becomes a
modelled response instead of a resolution limit.

The physics and the closed forms are in
[The forward model](../explanation/forward_model.md#geometric-smearing); this page
is how to use it.

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
translation of the delay axis and is applied exactly as a convolution, for free.

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
spot as $f$, so it cancels. The only lever is the mask ratio $d/D$.

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
selects the right pair from the `interaction` argument, so this only bites if you
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

Retrieving a probe-scanned trace against croak's convention without flipping the
axis returns $E^*(-t)$: the **spectral amplitude and the duration are unchanged**,
but the temporal profile is mirrored and the spectral phase is negated, so the sign
of the retrieved GDD/TOD flips. Whichever convention your delay axis is in, use
`reverse_delay=True` when it is the probe-scanned one, so $\rho$ matches the trace
the retrieval actually sees.
```

```{warning}
The widths assume the detector collects the **whole** signal beam — that is what
makes the transverse average incoherent (Parseval). Under aggressive spatial
filtering the average is partly coherent, the effective blur is narrower, and the
number here is an **upper bound**. If your signal aperture is much smaller than
the signal beam, compare against a forward simulation with it opened up.
```

## Fixed or fitted

**Fixed** is the physically honest default: compute the kernel from the mask you
built and hold it there. It is also what makes the effect *identifiable*, because
geometric smearing is the only term that responds to the **mask**: its width scales
with $d/D$, while a dispersion-model deficiency is insensitive to it. Repeating a
retrieval at two mask spacings therefore separates the two.

```{warning}
The kernel is independent of the medium thickness, but the **duration offset it
induces is not** — how much the free pulse must absorb depends on the whole trace,
whose sensitivity to the input duration changes with the slab. Do not use a
"constant offset across thicknesses" test as the discriminator; use the mask
dependence. `examples/example_geometric_smearing.py` runs both sweeps and shows
the difference.
```

**Fitted** gives an empirical upper bound on the effect. `fit_smearing=True` adds a
single dimensionless multiplier on both widths — equivalently $d/D$, the only
lever — and the gradient comes free from AD:

```python
res = croak.retrieve(
    trace, g.omega, delays, "pg",
    algorithm="lbfgs-ad",
    smearing=kernel,
    fit_smearing=True,
)
print(res.smear_scale)   # 1.0 means the geometric prediction was right
```

A fitted value near 1 is strong evidence the mechanism is identified; one much
larger says something else is being absorbed.

```{warning}
**Do not fit the smearing width and the slab thickness in the same run.** The
smearing multiplier is a nuisance parameter that will happily soak up
dispersion-model error, and the two are not separately identifiable from a single
trace. Fit one, hold the other.
```

Uncertainty follows the same route as the other extra parameters:
{func}`~croak.covariance.parameter_covariance` accepts `smearing` and
`fit_smearing` and reports `sigma_smear`.

### Splitting the two channels

The kernel has two channels: the delay offset $\vartheta$ (a pure blur along
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
The $p$ multiplier is only identifiable when the pulse (and hence the gate) is
**chirped**: for a transform-limited gate, $p$ merely attenuates the signal —
absorbed by the intensity scale $\mu$ — and leaves the trace shape unchanged.
Interpret a split fit on a near-transform-limited measurement accordingly: the
$\vartheta$ value is meaningful, the $p$ value is not constrained.
```

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
| `smearing` | `lbfgs-ad`, `lm`, `lm-optx`, `lbfgs-optx`, `cma-es` | PG (TG) or SD; uniform delay axis |
| `fit_smearing` | `lbfgs-ad`, `lm`, `lm-optx`, `cma-es` | a `smearing` kernel |
| `fit_smearing_split` | `lbfgs-ad`, `lm`, `lm-optx`, `cma-es` | `fit_smearing` |

`copra`, `copra-jax`, `lbfgs` and `lbfgs-hand` **cannot** model it: the incoherent
sum over quadrature nodes means the measurement is no longer the modulus-squared of
a single signal field, which breaks COPRA's magnitude-replacement projection, and
the hand-written adjoints are not extended to it. Asking for smearing with one of
them raises — it is never dropped silently.

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
