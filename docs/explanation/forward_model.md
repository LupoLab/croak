# The forward model

{class}`~croak.forward.ForwardModel` is the single computational core shared by
every retrieval algorithm in croak. It maps a complex spectrum $\tilde E(\omega)$
to a delay-scanned nonlinear-process spectrum (the trace), and provides the exact
reverse-mode (Wirtinger) adjoint of that map. Every gradient in croak flows
through it.

A key design choice: **the thin medium is just the dispersive model with
$\beta = 0$, zero thickness and a single quadrature node.** One code path and one
adjoint kernel therefore serve both regimes — the only difference is the model an
algorithm is handed.

## Thin medium

In the thin, non-dispersive limit the signal for one delay $\tau$ is

```{math}
\psi(\omega, \tau) \;=\; \mathcal{F}\!\left\{\,
   s\bigl(E(t),\, E(t-\tau)\bigr)\right\}(\omega),
\qquad
T(\omega, \tau) = \lvert \psi(\omega, \tau)\rvert^2,
```

where $s(\cdot,\cdot)$ is the [interaction](interactions.md) signal, $E(t)$ is the
test field and $E(t-\tau)$ the gate. Synthesising a full trace is a handful of
FFTs per delay. This is the default — `material=None`.

```python
import numpy as np, croak
g = croak.Grid(128, dt=0.3e-15)
ew = croak.gaussian_pulse(g, 2.5e-15)
delays = np.linspace(-18e-15, 18e-15, 87)

trace = croak.maketrace(g.omega, delays, ew, "shg")    # (Nomega, Ndelay)
```

{func}`~croak.forward.maketrace` is a convenience wrapper; the underlying object is
{class}`~croak.forward.ForwardModel`, whose {meth}`~croak.forward.ForwardModel.trace`
and {meth}`~croak.forward.ForwardModel.signal_single` you can call directly.

## Dispersive slab

To capture propagation inside the nonlinear medium, croak integrates the signal
generation over the slab depth. A trial spectrum defines the input replicas at the
entrance face; each propagates to depth $z$ with the **linear propagator**
$e^{i\beta(\omega) z}$; the nonlinear signal is generated locally; and it
propagates the remaining distance $L-z$ to the exit face, accumulating its own
phase. The exit signal is the coherent superposition over the thickness.

Discretised with Gauss–Legendre nodes $z_q$ and weights $w_q$ on $[0, L]$, the
per-delay signal that {class}`~croak.forward.ForwardModel` computes is

```{math}
:label: dispersive
\psi(\omega,\tau) \;=\; \sum_q w_q\, e^{i\beta(\omega)(L-z_q)} \,\odot\,
   \mathcal{F}^{-1}\!\Bigl[\, s\bigl(
      \mathcal{F}[\tilde E\, e^{i\beta z_q}],\;
      \mathcal{F}[\tilde E\, e^{i\omega\tau}\, e^{i\beta z_q}]
   \bigr)\Bigr],
```

with the trace again $T = \lvert\psi\rvert^2$. Here $\mathcal{F}$ is the raw DFT;
the model works internally in DFT-bin order and exposes centred arrays at its
boundary (see [FFT convention](../reference/fft_convention.md)). Setting
$\beta=0$, $L=0$ and a single node recovers the thin-medium formula exactly.

### The propagation constant

$\beta(\omega)$ comes from {func}`croak.materials.beta`. On the centred grid
(relative to the carrier $\omega_0$):

```{math}
\beta(\omega) \;=\; \frac{(\omega+\omega_0)\,n(\omega+\omega_0)}{c}
   \;-\; \beta_1(\omega_0)\,\omega,
```

where $n$ is the material's refractive index (Sellmeier or tabulated) and the group-delay term
$\beta_1(\omega_0)\,\omega$ keeps the pulse centred in the time window. The
built-in materials are listed in {data}`croak.materials.MATERIALS` (the Sellmeier
formulae SiO₂, BK7, CaF₂, BaF₂, MgF₂ plus the tabulated broadband dataset
SiO2-Franta, represented by a smoothing spline); entries outside a material's
valid range are set to zero. You are not limited to these — register a custom `n(λ)` or pull one
from the refractiveindex.info database and pass its name here just the same; see
[Materials and chirped mirrors](../howto/materials_and_mirrors.md).

### Using it

Pass `material`, `thickness`, `npoints` and the carrier `omega0`:

```python
omega0 = croak.maths.wlfreq(260e-9)        # 260 nm
trace = croak.maketrace(g.omega, delays, ew, "pg",
                       material="SiO2", thickness=10e-6, npoints=20, omega0=omega0)
```

Remember that dispersive propagation is supported for **PG** and **SD** only
(their signal stays at the fundamental); SHG raises a `ValueError`. See
[Interactions](interactions.md).

## Quadrature convergence

{func}`~croak.forward.quadrature_nodes_weights` builds the depth integral. The
default `"gausslegendre"` rule converges **exponentially** with the number of
nodes; the `"uniform"` rule (a plain sum, useful as a cross-check) only converges
algebraically. For a typical few-µm to tens-of-µm UV slab, `npoints ≈ 15–20`
reaches machine precision, so that is the default in the solvers. Increase it if
you use a thicker or more dispersive medium and watch the trace stop changing.

```{note}
**The GUI ties `npoints` to a depth step Δz.** The Retrieve stage's *Dispersive
propagation* group exposes a **Δz step (µm)** control (default 1 µm) alongside the
thickness and `Npoints`, with `npoints = round(thickness / Δz)`. Editing the
thickness holds Δz fixed and grows `Npoints` so the node density stays constant;
editing `Npoints` back-computes Δz to the actual spacing. Only `Npoints` reaches
{func}`~croak.forward.quadrature_nodes_weights` — Δz is a convenience for picking it,
so a 40 µm slab automatically gets ~4× the nodes of a 10 µm one. The nodes are the
(non-uniform) Gauss–Legendre abscissae, so Δz is a *nominal* step, not the literal
spacing of every node.
```

## Generation envelope (thick slabs / tight focusing)

The depth integral above assumes **uniform generation density**: every depth
contributes with its quadrature weight, modified only by the physics the 1D
model contains (the replicas disperse, so deeper slices are weaker). What a 1D
model cannot see is the *transverse* geometry of the crossing beams: the focal
spot growing away from the focus (Rayleigh range), the beamlets walking through
each other, and the Gouy phase — which enters the *coherent* depth sum as a
z-dependent phase, not just an amplitude. To good approximation all of this
factorizes into one smooth complex envelope $g(z)$ multiplying the local
generation.

`depth_weight` on {func}`~croak.forward_jax.make_param_trace_fn` (and the
`lbfgs-ad` solver) accepts one complex weight per quadrature node and
multiplies the depth-quadrature weights with it. The overall scale is
irrelevant (absorbed by $\mu$); only the shape matters. Two rules of thumb:

- **Negligible when $L \lesssim z_R$** — which covers conventional 10–50 µm
  practice with ordinary focusing (for the example DUV geometry,
  $z_R \approx 87$ µm: the envelope varies by <10 % across a 40 µm slab, and
  its smooth symmetric part is nearly pure renormalization).
- **Required once $L$ approaches the beam-overlap length** $\sim w_0/\theta$
  (the crossed beamlets walk through each other): beyond it, generation stops
  regardless of how much substrate remains, and a uniform-weight model would
  attribute signal to depths that produce none. A diagnostic signature: a
  uniform-weight retrieval with `fit_thickness` returns the *effective
  generation length*, not the physical slab.

With `fit_thickness` the envelope is pinned to the normalized nodes, i.e., it
rescales with the fitted slab rather than staying fixed in absolute $z$.

## The adjoint

{meth}`~croak.forward.ForwardModel.adjoint_single` is the exact reverse-mode
adjoint of {eq}`dispersive`. Given the trace cotangent it returns
$\partial L/\partial\tilde E^*$ — the gradient the solvers descend on — at the cost
of a single extra pass that reuses the recorded forward fields. This is what lets
COPRA and L-BFGS run in pure NumPy, with no JAX on the hot path. The
[Gradients](gradients.md) page derives it, sets out what it covers, and explains
how it is cross-checked against automatic differentiation.

## Geometric smearing

The equations above evaluate the polarisation at a *single* transverse point. In a
non-collinear BOXCARS geometry that is not quite what a spectrometer sees: the
three input beams arrive at the interaction plane with tilted pulse fronts, so
their relative arrival times vary linearly across the focal spot, and the detector
integrates over that spread. {mod}`croak.smearing` puts that instrument response
inside the forward model.

A beam apertured at mask position $\mathbf r_j$ and focused by a lens of focal
length $f$ crosses the focus at angle $\mathbf r_j/f$, so at focal-plane position
$\mathbf r$ it arrives at $t - \boldsymbol\alpha_j\cdot\mathbf r$ with
$\boldsymbol\alpha_j = -\mathbf r_j/(fc)$. Because $|\mathcal F\{\cdot\}|^2$ is
blind to a global time origin — and that origin may be chosen *per* $\mathbf r$ —
only **two** scalar combinations of the three arrival times survive:

```{math}
:label: smeared
\mathrm{PG:}\quad S &= E(t)\,E(t-\tau+\delta+p/2)\,E^*(t-\tau+\delta-p/2) \\
\mathrm{SD:}\quad S &= E(t+p/2)\,E(t-p/2)\,E^*(t-\tau+\delta)
```

Setting $p=\delta=0$ recovers `test * |gate|**2` and `test**2 * conj(gate)`
*exactly*, so switching the kernel off changes nothing.

$p$ separates the two replicas that coincide in the 1-D model, so it changes the
**shape** of the signal and needs an explicit quadrature. $\delta$ is a **pure
translation of the delay axis**, so integrating over it is a convolution along
$\tau$, applied once to the finished trace. Both are linear in $\mathbf r$, so with
an isotropic Gaussian focal weight the pair $(p,\delta)$ is jointly Gaussian —
which is all {class}`~croak.smearing.SmearingKernel` stores.

### Kernel width from the mask

The local signal amplitude is the product of the three focal-plane amplitudes, so
position $\mathbf r$ contributes with weight $|A(\mathbf r)|^6$. For a uniformly
illuminated circular hole of diameter $D$ that is the Airy amplitude
$2J_1(u)/u$, $u=\pi Dr/(\lambda f)$, whose rms per Cartesian component is
$\sigma_r = C\lambda f/D$ with $C = 0.245673$
({data}`~croak.smearing.AIRY6_RMS_COEFF`). Since $\boldsymbol\alpha_j \propto 1/f$
**the focal length cancels** — the crossing angle and the spot size scale
oppositely. For the standard folded square BOXCARS with holes at $(\pm d,\pm d)$:

| interaction | $\sigma_p$ | $\sigma_\delta$ | $\rho$ |
|---|---|---|---|
| `"pg"` (delay on the gate pair) | $0.49135\,(d/D)\,\lambda/c$ | $0.54933\,(d/D)\,\lambda/c$ | $1/\sqrt5$ |
| `"sd"` (delay on the conjugated arm) | $0.69487\,(d/D)\,\lambda/c$ | $0.34743\,(d/D)\,\lambda/c$ | $0$ |

so the **only** physical lever is the mask ratio $d/D$ (and the wavelength). The
two rows differ because the arm that carries the scanned delay decides which pair
of replicas $p$ separates; only in the SD arrangement do $p$ and $\delta$ come out
uncorrelated. Do not assume one geometry's ratios carry over to another —
{func}`~croak.smearing.kernel_from_arms` derives them from the actual hole
coordinates.

For an example DUV geometry (1 mm holes, 0.5 mm edge to edge so $d/D=0.75$,
260 nm) the PG kernel is $\sigma_p = 0.320$ fs, $\sigma_\delta = 0.357$ fs — a
0.84 fs FWHM blur along the delay axis, which matters for a 1 fs pulse. (The
companion paper's production instrument sits at $d/D = 1.0$, with proportionally
wider kernels.)

### Cost and accuracy

The kernel multiplies the depth quadrature rather than coupling to it: the
pulse-front tilts are fixed before the medium, and the longitudinal walk-off inside
a thin slab is far below a femtosecond, so the same $(p,\delta)$ applies at every
depth node. (The depth-independence of the *transverse phase* is a separate
approximation, and exactly the one the collected model's `depth_transverse` flag
relaxes — see the subsection below.) Because the quadrature nodes are exactly antisymmetric, each shifted
field is built once and used twice, and the cost is about $(K+1)/2$ times the
unsmeared model — roughly 3× at the default $K=5$ nodes. The $\delta$ integral is
free: it is a multiplication of the delay-axis DFT by the analytic Gaussian
transform, which is exact for any trace that is Nyquist-sampled in $\tau$ (so the
kernel need not span several delay steps). It does require a **uniformly spaced**
delay axis.

The one approximation is replacing the exact $|A|^6$ focal weight by a Gaussian of
the same second moments. The leading trace correction is quadratic in
$(p,\delta)$, so only those moments matter to first order; `tests/test_smearing.py`
measures the residual at about one part in $10^5$ of the trace peak for the
reference geometry.

```{note}
Geometric smearing is available on the **autodiff solvers only** (`lbfgs-ad`,
`lm`, `lm-optx`, `lbfgs-optx`, `cma-es`). The incoherent sum over quadrature nodes
means the measurement is no longer $|\tilde E_{\rm sig}(\omega,\tau)|^2$ at a
single node, which breaks COPRA's magnitude-replacement projection, and the
hand-derived adjoint of {eq}`dispersive` is not extended to it. Asking for it with
any other algorithm raises rather than silently dropping the kernel. This is a
case the projection-based algorithms cannot express and the AD formulation can.
```

See [Geometric smearing](../howto/geometric_smearing.md) for how to use it.

### Beyond the reduction: the mixture and the collection aperture

Two assumptions in the paragraphs above are separable, and croak lets each be dropped.

{mod}`croak.focal` drops the **achromatic** one. The reduction freezes the $|A|^6$
weight at the carrier, but the mask maps to transverse wavevector as
$x = k_\perp z_\text{mask} c/\omega$, so $A$ depends on the product $r\omega$ and
different focal positions see different *spectra*. The mixture keeps the sum over
$\mathbf r$ explicit, each node applying its own $A(r,\omega)$ and carrying its own
*definite* $(p,\vartheta)$ instead of a distribution over them. Held at the carrier it
reproduces the closed forms above exactly, which
{func}`~croak.focal.reduced_moments` checks.

{mod}`croak.collection` drops the **full-beam collection** one. "$|\mathcal F\{\cdot\}|^2$
is blind to a global time origin, and that origin may be chosen per $\mathbf r$" is the
step that produced $(p,\delta)$, and it is legitimate only because Parseval turns
$\int|\tilde S(\mathbf k)|^2\,\mathrm d^2k$ over *all* $\mathbf k$ into
$\int|S(\mathbf r)|^2\,\mathrm d^2 r$. Through a finite hole the signal must be
transformed *before* it is squared, and the discarded per-position phase comes back.

The reduction itself survives that intact. The exact focal-plane signal and croak's
per-node build differ by a rigid time shift,
$S(t,\mathbf r) = S_\text{croak}(t-\boldsymbol\alpha_1\cdot\mathbf r,\mathbf r)$,
so the only thing lost is one phase — and after the signal's own phase-matched carrier
$\boldsymbol\alpha_s = \boldsymbol\alpha_1+\boldsymbol\alpha_2-\boldsymbol\alpha_3$
is removed, what remains is $e^{+i\omega p}$: **exactly $\omega$ times one of the two
reduced parameters** (and $-\omega\vartheta$ for SD). The $(p,\vartheta)$
parameterisation is therefore sufficient for an apertured instrument as well as a
fully collected one.

Whether it *matters* is a question about one ratio: the aperture's k half-width
$(\omega/c)(D_\text{hole}/2)/z_\text{mask}$ against the signal's own k content
$3(\omega/c)(D/2)/f$. For the example DUV geometry that is 6.04e4 against 3.63e5 —
six times narrower — and the incoherent sum is then an upper bound on the blur rather
than an estimate of it. See
[Modelling the collection aperture](../howto/collection_aperture.md).

### Depth-resolved transverse dephasing (`depth_transverse`)

One assumption survives even the collected model: every depth node of
{eq}`dispersive` is added with the *same* transverse phase — fully coherently in
transverse wavevector. Real propagation dephases the angular spectrum. Signal
generated at depth $z$ accumulates, over the remaining slab, the extra phase

$$
\phi(\omega, k_\perp, z)
  = \bigl[k_z(\omega, k_\perp) - k_z(\omega, 0)\bigr](L - z)
  \;\approx\; -\frac{c\,k_\perp^2\,(L - z)}{2\,n(\omega)\,\omega},
\qquad k_z = \sqrt{(n\omega/c)^2 - k_\perp^2},
$$

relative to the on-axis propagation the model already applies — depth-accumulating,
quadratic in $k_\perp$, and chromatic ($\sim 1/\omega$, red-weighted). The
`depth_transverse` flag of {func}`~croak.forward_jax.make_param_trace_fn` applies it
inside the collection transform: the depth sum moves *inside* the aperture
integral, each depth node's transformed field carrying $e^{i\phi_j(\omega,z_q)}$ at
the aperture node's absolute wavevector before the coherent sum over depth. It is
the k-*dependent* sibling of the generation envelope above: `depth_weight`
carries the k-independent per-depth amplitude factor $g(z)$, `depth_transverse`
the k-dependent per-depth phase, and the two compose. Parameter-free, opt-in, and
validated (sign included) against a per-depth dense-FFT reference with the exact
$k_z$ — see [the collection how-to](../howto/collection_aperture.md) for usage,
cost and caveats.

## The model hierarchy, and when each member applies

The pieces above assemble into three usable forward models, and the companion
paper measures the domain of each against full 3-D instrument simulations
(see [Validation](validation.md)):

- The **standard** model (single plane, no geometry) is right for thin media
  and multi-cycle pulses — conventional FROG practice.
- The **extended** model (the dispersive slab plus the reduced smearing
  kernel) costs about the same and is the workhorse: it retrieves a ~1 fs DUV
  pulse correctly at every substrate thickness from 1 to 40 µm, where the
  standard model errs by up to a factor of 2.6.
- The **full** model (the chromatic focal mixture through the modelled
  collection aperture) is needed when octave-bandwidth light is measured
  through a tight collection aperture; it costs minutes rather than seconds
  per retrieval on a CPU.

Two dimensionless switches decide which member the data need: the fractional
bandwidth turns on the chromatic effects, and the collection-to-generation
aperture ratio turns on the coherent-collection effects. For multi-cycle
pulses both are off. For near-single-cycle pulses there are two working
routes: open the collection aperture (restoring the extended model's premise
in hardware) or keep any aperture and retrieve with the full model.

## API

{class}`croak.forward.ForwardModel`, {func}`croak.forward.maketrace`,
{func}`croak.forward.quadrature_nodes_weights`, {mod}`croak.smearing`,
{mod}`croak.focal`, {mod}`croak.collection` and {mod}`croak.materials` in the
[API reference](../reference/api/index.md).
