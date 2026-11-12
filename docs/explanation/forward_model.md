# The forward model

The forward model predicts the trace that a trial pulse would produce in a
specified instrument. Choosing it is a physical decision: include propagation,
beam geometry and collection when they materially affect the measurement. The
[dispersive tutorial](../tutorials/03_dispersive_retrieval.md) illustrates the
consequence of omitting propagation; [validation](validation.md) examines more
complete instrument simulations.

Start with the thin-medium model below, then add the effects justified by your
experiment. {class}`~croak.forward.ForwardModel` is the NumPy core: it maps a
complex spectrum $\tilde E(\omega)$ to the trace for the thin and dispersive-slab
models and provides the exact reverse-mode (Wirtinger) adjoint of that map, which
COPRA and the analytic-gradient solvers descend on. It also evaluates the reduced
smearing trace (a `smearing=` kernel), but without an adjoint. **The thin medium
is just the slab with $\beta=0$, $L=0$ and a single quadrature node**
({func}`~croak.forward.quadrature_nodes_weights` returns one node for zero
thickness), so one code path and one adjoint serve both regimes. The JAX model
({func}`~croak.forward_jax.make_param_trace_fn`) implements the same models plus
the spatial extensions below, differentiated by automatic differentiation; the
two paths are cross-checked where they overlap.

The unknown field must have a declared reference plane. In the slab model it is
the entrance field; the chromatic model further distinguishes an on-axis field
from a transverse-integrated or effective field. See
[field reference conventions](../reference/conventions.md#field-reference-planes).

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

where $n$ is the material's refractive index and the group-delay term
$\beta_1(\omega_0)\,\omega$ keeps the pulse centred in the time window. The
built-in materials are listed in {data}`croak.materials.MATERIALS` (SiO₂, BK7,
CaF₂, BaF₂, MgF₂, sapphire and silicon, each optmat's recommended model of the
material); a model is extrapolated beyond its stated range, and the entries at
non-positive frequency, which have no index, are set to zero. You are not limited to these — register a custom `n(λ)`
and pass its name here just the same; see
[Materials and chirped mirrors](../howto/materials_and_mirrors.md).

### Using it

Pass `material`, `thickness`, `npoints` and the carrier `omega0`:

```python
omega0 = croak.maths.wlfreq(260e-9)        # 260 nm
trace = croak.maketrace(g.omega, delays, ew, "pg",
                       material="SiO2", thickness=10e-6, npoints=20, omega0=omega0)
```

A material slab like this serves **PG** and **SD**, whose signal stays at the
fundamental. SHG needs a crystal, described next; an isotropic slab with SHG
raises a `ValueError`. See [Interactions](interactions.md).

## SHG in a Type I crystal

The SHG signal is generated near $2\omega_0$, and in a Type I (oo→e) crystal it is
the extraordinary wave while the two fundamentals are ordinary. It therefore needs
its own propagation constant. {mod}`croak.crystals` supplies two, written on the
model's centred grid and in **one** frame moving with the fundamental's group
velocity $\beta_1$:

```{math}
\beta_\mathrm{in}(\Omega)  = k_z(\omega_0+\Omega) - \beta_1\Omega, \qquad
\beta_\mathrm{sig}(\Omega) = k_s(2\omega_0+\Omega) - \beta_1\Omega,
```

with $k_z(\omega) = \sqrt{k_o(\omega)^2 - (\omega\sin\alpha/c)^2}$, the ordinary
wavevector along the bisector of two beams crossing at the external half-angle
$\alpha$, and $k_s(\omega) = \omega\, n_e(\theta,\omega)/c$, the extraordinary
wavevector at the internal angle $\theta$ to the optic axis. In
{eq}`dispersive`, $e^{i\beta z_q}$ becomes $e^{i\beta_\mathrm{in} z_q}$ and
$e^{i\beta(L-z_q)}$ becomes $a(\Omega)\,e^{i\beta_\mathrm{sig}(L-z_q)}$.

**Why this gives the phase matching.** Both constants keep their carrier
wavevectors. A component pair $(\omega_1,\omega_2)$ generated at depth $z$ then
carries $e^{-i\Delta k\,z}$ with the exact mismatch
$\Delta k = k_s(\omega_1+\omega_2) - k_z(\omega_1) - k_z(\omega_2)$: the frame terms
add up to $-\beta_1\Omega_s L$, independent of $z$, a linear spectral phase the
trace cannot see. The depth sum turns this into the pair's phase-matching
response, $P\,\mathrm{sinc}(\Delta k P/2)$ in amplitude over the path $P$ (the
plate thickness $L$ without a tilt; see below). Over a full period of delays the
frequency marginal is therefore the sum over pairs of
$a^2\lvert E(\omega_1)E(\omega_2)\rvert^2 P^2\mathrm{sinc}^2(\Delta k P/2)$, with
$a$ the generation prefactor below, which
`tests/test_forward_crystal.py` checks to the quadrature's accuracy.

**Generation prefactor.** The generated second-harmonic field grows as
$\omega_s/n_s$, so the model multiplies the signal by
$a(\Omega) = (\omega_s/n_s)/(2\omega_0/n_s(2\omega_0))$. Other spectral factors (the
beams' mode overlap, the dispersion of $\chi^{(2)}$, the detector's response) are
not modelled; the solvers' per-frequency response `R_omega` absorbs any that
matter.

**The crystal's angle.** A {class}`~croak.crystals.CrystalSetup` takes the
manufacturer's cut angle $\theta_c$ and either the external tilt $\gamma$ or the
wavelength the crystal was tuned to phase-match, the usual case when it was rotated
for the strongest signal. The beams refract to $\gamma_\mathrm{int} =
\arcsin(\sin\gamma/n_o)$, so $\theta = \theta_c + \gamma_\mathrm{int}$. In the tilted
plate the mismatch builds up along the face normal. The fundamentals' normal
wavevectors are $k\cos\gamma_\mathrm{int}$; the signal keeps their transverse
wavevector, so its direction $\psi$ solves $K(\psi)\sin\psi = k_t$ with
$K = k_s(\theta_c+\psi)$. Expanding about $\psi = \gamma_\mathrm{int}$ to first
order in the mismatch $\Delta k$ gives a normal mismatch
$\Delta k/(\cos\gamma_\mathrm{int} + (n_e'/n_s)\sin\gamma_\mathrm{int})$, so the model
integrates over the effective path
$P = L/(\cos\gamma_\mathrm{int} + (n_e'/n_s)\sin\gamma_\mathrm{int})$, with $n_s = n_e(\theta)$ and
$n_e' = \mathrm dn_e/\mathrm d\theta$ at the signal carrier. A test checks it against
the exact solve (agreement within 0.3 % of $\Delta k L$ at a 10° tilt, where
$L/\cos\gamma_\mathrm{int}$ is about 1 % off). Nothing is derived from the
retrieval grid. The
crossing angle comes from a {class}`~croak.geometry.TwoBeamGeometry` (beam
separation and focal length), the one place it is set.

**Depth resolution.** Each pair's depth integrand oscillates through
$\Delta k L$ rad. Gauss–Legendre integrates it to $10^{-6}$ of $L$ with
$\lceil\lvert\Delta k\rvert L/2\rceil + 3$ nodes; {func}`croak.crystals.depth_points`
takes the maximum over every pair of grid frequencies within the crystal's
transparency range, and the models refuse fewer nodes. It grows with the grid's
width: for 100 µm of BBO tuned to 800 nm, about 18 nodes on a 560–1440 nm grid and
about 46 on a grid twice as wide.

**The signal's tilt, and asymmetric SHG traces.** The signal of a pair
$(\omega_1,\omega_2)$ (test, gate) inherits the transverse wavevector
$k_x = (\omega_1-\omega_2)\sin\alpha/c$, so it leaves at an angle $\psi$ to the
bisector. As an extraordinary wave its index depends on that direction, and $\psi$
solves $(\omega_s/c)\, n_e(\theta+\psi)\sin\psi = k_x$; to first order
$\psi = c k_x/(\omega_s n_s)$, the self-consistency entering only at second order.
When the beams cross in the crystal's principal plane (the fundamental polarised
perpendicular to the crossing plane), the tilt changes the signal's angle to the
optic axis, so its index changes to first order, by $n_e'\psi$. The extra mismatch
is $\kappa(\omega_1-\omega_2)$ with $\kappa = (n_e'/n_s)\sin\alpha/c$, and over the
remaining path it factorises into opposite delays of the two arms that drift
apart with depth: in croak's convention (a factor $e^{i\Omega\tau}$ delays a field
by $\tau$) the test arm is delayed by $+\kappa(P-z)$ and the gate by
$-\kappa(P-z)$, so the gate-minus-test delay at depth $z$ is
$\tau - 2\kappa(P-z)$. The model includes this
({meth}`~croak.crystals.CrystalSetup.arm_shift`): about 0.57 fs per 100 µm per arm
for BBO at 800 nm crossing at 1.4°.

The same term has a picture in space. The crossing beams' wavefronts make the
delay between the arms vary across the focal spot, $\tau + 2x\sin\alpha/c$ (the
gradient behind the geometric delay smearing). The SH, an extraordinary wave,
carries its energy at the walk-off angle $\rho = -n_e'/n_s$ (68 mrad for BBO), so
SH made at depth $z$ leaves the crystal displaced sideways by $\rho(P-z)$, at a
point where the local delay differs by $2\rho\sin\alpha\,(P-z)/c = 2\lvert\kappa\rvert
(P-z)$. Light from every depth adds coherently at each exit point, each with its own
effective delay. The spectrometer sees it, though it records only the SH spectrum,
because the phase depends on $\omega_1-\omega_2$ rather than on $\omega_s$: it
changes how the pairs that make one signal frequency interfere. Crossing out of the
principal plane, the SH walks off along the direction in which the delay does not
vary, and the term vanishes.

Two consequences follow:

- **The trace is not symmetric in delay.** The drift distinguishes the two arms, so
  $I(\omega,\tau) \ne I(\omega,-\tau)$, unlike the ideal SHG FROG trace. For BBO
  at 800 nm crossing at 1.4° the term changes the signal by about 0.5 % for a
  12 fs pulse in 10 µm, 5 % in 100 µm, and about 1 % for a 5 fs pulse in 10 µm (on delays sampled finely enough to
  catch its peak, about 1 fs apart). At zero
  delay its first-order effect cancels between the pairs $(\omega_1,\omega_2)$ and
  $(\omega_2,\omega_1)$, so it shows away from zero delay.
- **Its sign depends on which beam is nearer the optic axis**
  (`delayed_beam="toward_axis"` or `"away_from_axis"`), which a real crystal's
  markings do not reveal: they give the plane of the optic axis and the cut angle,
  not which way it leans. Flipping the sign gives exactly the trace mirrored in
  delay, so the data decide it: retrieve with both and keep the better fit. A
  doubtful delay-axis convention is the same mirror.

What remains is second order in the tilt. {func}`croak.crystals.tilt_field_error`
computes it for a given spectrum and set of delays, against an exact pair sum with
each pair's signal in its own self-consistent direction, and
{func}`croak.crystals.check_tilt` warns above 1 %. It is under 0.3 % for pulses of
5-12 fs in up to 100 µm at 1.4°. Polarised in the crossing plane, the principal
plane is perpendicular to the crossing plane and the first-order term vanishes.

**What is neglected.** The SH's walk-off is kept only through its effect on the
delay (above); its other consequence, the SH beam moving sideways out of the
fundamentals' overlap, is neglected (the crystal is much thinner than the focal
spot over the walk-off angle: 6.8 µm of walk-off in 100 µm against a spot of tens
of µm). The fundamentals are ordinary waves and do not walk off. Also neglected:
Type II, the remainder of the signal tilt above, and the dispersion of the
refraction at the entrance face.

```python
import numpy as np
import croak
from croak.crystals import CrystalSetup, UniaxialCrystal
from croak.geometry import TwoBeamGeometry

crystal = UniaxialCrystal("BBO", n_o, n_e)       # n(λ in m) callables
setup = CrystalSetup(
    crystal,
    cut_angle=np.radians(29.2),
    phase_match_wavelength=800e-9,               # tuned for the strongest signal
    geometry=TwoBeamGeometry(beam_separation=5e-3, focal_length=0.1),
    polarization="parallel",                     # to the crossing plane
)
omega0 = croak.maths.wlfreq(800e-9)
npoints = croak.crystals.depth_points(setup, g.omega, omega0, 50e-6)
trace = croak.maketrace(g.omega, delays, ew, "shg", material=setup,
                        thickness=50e-6, npoints=npoints, omega0=omega0)
```

The retrieved field is the field at the crystal's entrance face.

## Quadrature convergence

{func}`~croak.forward.quadrature_nodes_weights` builds the depth integral.
Gauss–Legendre quadrature is efficient for a smooth depth integrand; the
`"uniform"` rule provides a useful independent discretisation check. Required
node counts depend on bandwidth, thickness, dispersion and field structure.
The solvers default to `npoints=20`, chosen for few-µm to tens-of-µm UV slabs. For a
thicker or more dispersive medium, increase `npoints` until the trace and the
retrieved field stop changing.

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
factorises into one smooth complex envelope $g(z)$ multiplying the local
generation.

`depth_weight` on {func}`~croak.forward_jax.make_param_trace_fn` (and the
`lbfgs-ad` solver) accepts one complex weight per quadrature node and
multiplies the depth-quadrature weights with it. The overall scale is
irrelevant (absorbed by $\mu$); only the shape matters. Two rules of thumb:

- **Negligible when $L \lesssim z_R$** — which covers conventional 10–50 µm
  practice with ordinary focusing (for the example DUV geometry,
  $z_R \approx 87$ µm: the envelope varies by <10 % across a 40 µm slab, and
  its smooth symmetric part is nearly pure renormalisation).
- **Required once $L$ approaches the beam-overlap length** $\sim w_0/\theta$,
  with $\theta$ the beam crossing angle (the crossed beamlets walk through each other): beyond it, generation stops
  regardless of how much substrate remains, and a uniform-weight model would
  attribute signal to depths that produce none. A diagnostic signature: a
  uniform-weight retrieval with `fit_thickness` returns the *effective
  generation length*, not the physical slab.

With `fit_thickness` the envelope is pinned to the normalised nodes, i.e., it
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

Within this achromatic, full-collection approximation, the widths depend on the
mask ratio $d/D$ and wavelength; more complete models also depend on collection,
focusing and depth evolution. The
two rows differ because the arm that carries the scanned delay decides which pair
of replicas $p$ separates; only in the SD arrangement do $p$ and $\delta$ come out
uncorrelated. Do not assume one geometry's ratios carry over to another —
{func}`~croak.smearing.kernel_from_arms` derives them from the actual hole
coordinates.

For an example DUV geometry (1 mm holes, 0.5 mm edge to edge so $d/D=0.75$,
260 nm) the PG kernel is $\sigma_p = 0.320$ fs, $\sigma_\delta = 0.357$ fs — a
0.84 fs FWHM blur along the delay axis, which matters for a 1 fs pulse. The
simulated instrument used in [validation](validation.md) has $d/D=1.0$, with
proportionally wider kernels.

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

Within the achromatic, fully collected reduction, the remaining approximation is
replacing the exact $|A|^6$ focal weight by a Gaussian with the same second
moments. The leading trace correction is quadratic in $(p,\delta)$, so only those
moments matter to first order. For the chirped test pulse and reference geometry
in `tests/test_smearing.py`, the Gaussian and Airy-weighted focal-plane traces
differ by about one part in $10^5$ of the trace peak.

```{note}
Geometric smearing is supported through the JAX model by `warm-lbfgs`,
`lbfgs-ad`, `lm`, `lm-optx`, `lbfgs-optx` and `cma-es`. The incoherent sum over
quadrature nodes means the measured trace is no longer $|\psi(\omega,\tau)|^2$ for
a single signal, which breaks COPRA's magnitude-replacement projection, and the
hand-derived adjoint of {eq}`dispersive` is not extended to it. Asking for
smearing with any other algorithm raises rather than silently dropping the kernel.
```

See [Geometric smearing](../howto/geometric_smearing.md) for how to use it.

### Beyond the reduction: the mixture and the collection aperture

Two assumptions in the paragraphs above are separable, and croak lets each be dropped.

{mod}`croak.focal` drops the **achromatic** one. The reduction freezes the $|A|^6$
weight at the carrier, but the mask maps to transverse wavevector as
$x = k_\perp z_\text{mask} c/\omega_\text{abs}$, so $A$ depends on the product $r\omega_\text{abs}$ and
different focal positions see different *spectra*. The mixture keeps the sum over
$\mathbf r$ explicit, each node applying its own $A(r,\omega)$ and carrying its own
*definite* $(p,\delta)$ instead of a distribution over them. Held at the carrier it
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
is removed, what remains is $e^{+i\Omega p}$, with $\Omega=\omega_\text{abs}-\omega_0$
the centred frequency: **exactly $\Omega$ times one of the two
reduced parameters** (and $-\Omega\delta$ for SD). The $(p,\delta)$
parameterisation is therefore sufficient for an apertured instrument as well as a
fully collected one.

Whether it *matters* is a question about one ratio: the aperture's k half-width
$(\omega_\text{abs}/c)(D_\text{hole}/2)/z_\text{mask}$ against the signal's own k
content $3(\omega_\text{abs}/c)(D/2)/f$, both at absolute optical frequency. For the
example DUV geometry that is 6.04e4 against 3.63e5 m⁻¹ — the aperture is six times
narrower, so the fully collected (incoherent) reduction does not describe that
instrument and the collection must be modelled. See
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

### The depth-resolved mixture (fc-z, `evolve_profiles`)

`depth_transverse` repairs the *exit* side (how the generated signal reaches the
aperture); the depth-resolved mixture repairs the *entrance* side, by allowing depth evolution: the chromatic beamlet profiles are
not frozen at the entrance face but evolve linearly through the slab, and each
of the three arms evolves about its own walked axis. With
`focal_mixture(..., evolve_profiles=True, material=..., thickness=...,
npoints=...)` every arm $j$ at node $\mathbf r_k$ and depth $z_q$ applies its
own complex profile

$$
A_j(r_k, \omega; z_q) = N(\omega)\!\int_0^{k_R(\omega)}\!
  S(k_\perp,\omega)\, J_0\!\bigl(k_\perp\,|\mathbf r_k - \bm\delta_j(z_q)|\bigr)\,
  e^{\,i[k_z(\omega,k_\perp)-k_z(\omega,0)]\,z_q}\, k_\perp\,\mathrm dk_\perp,
\qquad
\bm\delta_j(z) = -\frac{z\,\mathbf r_j}{f\,n_0},
$$

in place of the single shared entrance filter $A(r_k,\omega)$: the aperture's
k-space (a disc of radius $k_R = (\omega_\text{abs}/c)(D/2)/f$, or the Gaussian control's
$e^{-k_\perp^2 w_0^2/4}$) propagated by the *transverse* part of $k_z$ only —
the on-axis dispersion stays in the depth propagator, so $z\to 0$ reduces
exactly to the entrance model. The $(p_k,\delta_k)$ arrival offsets stay
depth-independent (exact: transverse wavevector is conserved). Under fc-z the
"Cost and accuracy" statement that the transverse weight commutes with the depth
propagator no longer holds — deliberately: that non-commutation is what lets the
model carry a gate-vs-probe arrival asymmetry that accumulates with depth. Cost is ~fc (the
depth axis is already batched; only a seconds-level numpy filter build and a
$(3,K,Q,N)$ table are new). PG only; fitted thickness is unsupported; the
spec's slab parameters must match the trace map's (verified). Validated by an
independent plane-wave propagator at $10^{-6}$ of peak and by moment-pinning
against a three-beam overlap/centroid/timing table measured on the 3-D
instrument simulations — see `tests/test_fcz.py`.

## The model hierarchy, and when each member applies

The options form a hierarchy, with different assumptions and cost:

| Model | Included physics | Main assumptions to assess |
|---|---|---|
| Standard | Local instantaneous interaction | Propagation and spatial averaging negligible |
| Extended | Coherent slab generation and reduced timing kernel | Achromatic local profiles and full isolated-signal collection adequate |
| Full | Chromatic focal fields and coherent finite-aperture collection, with selected depth corrections | Specified beam profiles, paraxial geometry, material response and detector model adequate |

The standard model is conventional FROG practice: right for thin media and
multi-cycle pulses. Two dimensionless quantities decide whether more is needed:
the fractional bandwidth (chromatic focal effects) and the ratio of collection
aperture to generation aperture (coherent-collection effects). With the
dispersive slab and smearing kernel, the extended model costs about the same as
the standard one. For near-single-cycle pulses through a restrictive collection
aperture, use the full model; opening the aperture reduces collection effects,
but pump leakage limits how far it can be opened. In the simulated TG-FROG
instrument the full model with the recorded 0.5 mm aperture recovers a 1.05 fs
pulse to 1.047–1.079 fs at every thickness from 1 to 40 µm, while the standard
model returns 1.35–2.41 fs; the [validation chapter](validation.md) gives the
details, and the [gallery](../examples/index.md#instrument-effects-and-validation)
provides runnable illustrations.

## API

{class}`croak.forward.ForwardModel`, {func}`croak.forward.maketrace`,
{func}`croak.forward.quadrature_nodes_weights`, {mod}`croak.smearing`,
{mod}`croak.focal`, {mod}`croak.collection` and {mod}`croak.materials` in the
[API reference](../reference/api/index.md).
