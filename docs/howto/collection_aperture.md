# Modelling the collection aperture

The [chromatic focal mixture](../explanation/forward_model.md#geometric-smearing) sums
**incoherently** over focal-plane position. By Parseval that is exact only if the
detector collects the whole signal beam — and most non-collinear instruments do not:
the signal is picked out of the BOXCARS pattern by a hole, which is a filter in
transverse wavevector. The field must then be transformed into the collection plane
before taking its intensity.

When the hole is narrow the average over the focal spot is partly **coherent**, and a
coherent sum sharpens where an incoherent one blurs. A finite aperture changes both
the spectrum and the delay structure of the trace; it does not act as a narrower
version of the same smearing kernel.

**Practical rule.** For near-single-cycle pulses through a restrictive aperture,
retrieve with the full model: `focal=` with `collection=` set to the instrument's
aperture. In the companion papers' simulated TG-FROG instrument this recovers a
1.05 fs pulse to within 3 % at every substrate thickness from 1 to 40 µm through a
0.5 mm collection hole, and its fixed-field trace predictions agree to 0.03–0.2 %
rms of peak ([validation](../explanation/validation.md)). The alternative is to open
the aperture until the extended model's full-collection premise holds, as far as
pump leakage allows.

## When it matters

Compare two numbers at the carrier. On this page $\omega$ is the absolute optical
angular frequency; $\Omega=\omega-\omega_0$ is the centred frequency of croak's grids.

| quantity | expression | reference instrument at 260 nm |
|---|---|---|
| signal k footprint | $3(\omega/c)(D/2)/f$ | 3.63e5 rad/m |
| aperture k half-width | $(\omega/c)(D_\text{hole}/2)/z_\text{mask}$ | 6.04e4 rad/m |

If the aperture is comparable to or wider than the footprint, the incoherent sum is
adequate. Here it is **six times narrower** — about 3 % of the solid angle — so the
instrument sits far closer to the coherent limit than to the incoherent one, and the
collection model is not a refinement.

The other way to see it: a disc of k half-width $\kappa_a$ correlates the focal plane
over an Airy kernel whose first null is at $3.83/\kappa_a \approx 63\,\mu\text{m}$,
while the region that actually emits (where
the arrival offsets $p, \delta$ stay inside the pulse duration) is only 15–30 µm
across. The detector cannot resolve the smearing it is supposed to average over.

## The quick version

```python
from croak.collection import mask_hole_aperture
from croak.focal import focal_mixture

aperture = mask_hole_aperture(
    hole_x=-1.0e-3, hole_y=-1.0e-3,   # collection hole centre, mask plane (m)
    hole_diameter=0.5e-3,             # its diameter (m)
    z_mask=0.1,                       # mask plane distance from the focus (m)
    apod="tanh", apod_param=96.9e-6,  # the edge, in mask-plane metres
)
# A pnps window (compact raised-cosine edge used as a QUADRATURE weight over the
# nominal hard hole; the weight enters the collected energy once, not squared):
# mask_hole_aperture(..., apod="rcos", apod_param=64.6e-6, weighting="quadrature").
# aperture_from_scan() picks both up from the file's window_def_* record.
mixture = focal_mixture(
    hole_diameter=1.0e-3,             # the INPUT mask holes
    hole_spacing=1.0e-3,
    f_foc=0.1,
    wavelength=260e-9,
    collection=aperture,
)

res = croak.retrieve(
    trace, g.omega, delays, "pg",
    algorithm="lbfgs-ad", focal=mixture, omega0=omega0,
)
```

`focal=` is accepted by `lbfgs-ad`, `warm-lbfgs`, `lm`, `lm-optx`,
`lbfgs-optx` and `cma-es`, and the collection aperture is part of the mixture configuration. Leaving `collection` unset keeps the full-beam incoherent sum
of local intensities.

The same object passes through the high-level entry points: both
`croak.pipeline.retrieve_from_tracedata(td, ..., focal=mixture)` and
`croak.session.pipeline.run_retrieval(params, td, focal=mixture)` forward it to the
solver (and raise for a solver that cannot model it, exactly as `smearing=` does).
Build the params with `smearing=False` — the mixture *replaces* the reduced kernel,
and asking for both is refused.

## From the GUI and from `RetrieveParams`

`RetrieveParams` carries the mixture as scalars — `focal` (the switch),
`focal_n_radial` / `focal_n_azimuth` / `focal_rmax_units` (the node grid),
`focal_f_mm` (the focusing focal length, which the reduced kernel never needs
but the explicit mixture does; read from a simulated scan's geometry record on
load), `collection` (`"off"` / `"file"` / `"manual"`), `collection_mode`
(`"integrated"` / `"reimaged"`), `collection_diam_mm` (manual mode: a
hard-edged pinhole at the phase-matched corner) and `spectrum_frame_p` (the
spectral-frame exponent — the mixture's field is the *on-axis* one, bluer than
an integrated measured spectrum; the exponent reweights the initial guess and
the `reg_spectrum` target together into that frame). The GUI exposes all of
these in the *Chromatic focal mixture (advanced)* group of the retrieve stage,
mutually exclusive with the reduced-kernel checkbox, and
`croak.session.pipeline.focal_mixture_from_params(params, td, scan_path)`
builds the object — `run_retrieval` calls it automatically when `params.focal`
is set and no explicit `focal=` is given (`scan_path` is only needed for
`collection="file"`, which rebuilds the aperture from the scan's own
`window_def_*` record).

## Reading the aperture off a simulated scan

A `scansave` file records the window it was collected through, so it does not have to be
retyped (and mistyped):

```python
from croak.collection import aperture_from_scan

aperture = aperture_from_scan("scan.h5")                      # the Iω_win window
reimaged = aperture_from_scan("scan.h5", mode="reimaged")     # Iω_win_reimaged
```

This resolves the simulator's `"default"` apodisation width from the file's own
transverse k-grid, which matters: for a soft-edged window it can be **96.9 µm on a
500 µm hole** — an edge 19 % of the diameter, passing 0.994 at the centre and still
0.006 at $r = D$. Treating that window as a hard hole is a large error, not a rounding
one. Other scans record a hard-hole integration weight instead, so read the window
from each file.
{class}`~croak.io.MaskWindowSpec` is the raw record, and
{attr}`SimulatedScan.mask_window <croak.io.SimulatedScan>` carries it.

## The two collection models

Both are functionals of the same transformed field $\tilde S(\boldsymbol\kappa,\omega)$,
selected with `mode=`:

`"integrated"`
: $\sum_j \mathrm dk_j\,W_j^2\,|\tilde S_j|^2/(2\pi)^2$ — a spectrometer fed by
everything the hole passes. The reference simulator's `Iω_win`, and the usual case.

`"reimaged"`
: $\bigl|\sum_j \mathrm dk_j\,W_j\,\tilde S_j/(2\pi)^2\bigr|^2$ — the windowed field back
at the focal origin, i.e. a spectrometer fed by an on-axis re-imaged pixel. The fully
coherent limit (`Iω_win_reimaged`).

For a physical amplitude transmission $W$, the integrated observable uses
$|W|^2$ and the reimaged observable uses $W$ inside the coherent sum. A numerical
hard-hole **quadrature weight** is different: it approximates an area integral
and enters once. `weighting="quadrature"` records that distinction. Squaring a
quadrature taper as though it were a physical transmission changes the aperture.

## Where the aperture has to be centred

At the degenerate carrier, the paraxial geometry places the selected signal along $\mathbf k_1+\mathbf k_2-\mathbf k_3$, which
maps back to the mask position $\mathbf r_s=\mathbf r_1+\mathbf r_2-\mathbf r_3$ — the
fourth BOXCARS corner. In the carrier-removed frame the aperture therefore sits at

$$
\boldsymbol\kappa_c(\omega) = \frac{\omega}{c}\,
\frac{\mathbf x_\text{hole}-\mathbf r_s}{z_\text{mask}},
$$

which is **zero** when the instrument collects on the phase-matched direction, and grows
with $\omega$ otherwise: the aperture is chromatic, and so is any misalignment of it.
{func}`~croak.collection.aperture_offset` reports
$\mathbf x_\text{hole}-\mathbf r_s$ in metres, so a permuted arm order or a hole on the
wrong corner shows up as a number rather than as a mysteriously dark trace:

```python
import numpy as np
from croak.collection import aperture_offset
assert np.abs(aperture_offset(mixture)).max() < 1e-9   # collecting on phase matching
```

{func}`~croak.focal.focal_mixture` records its `arms` because the incoherent sum
needed only the arm *differences* that make $(p,\delta)$, but the aperture needs to
know where the signal actually goes.

For broadband mixing, the carrier's fourth-corner construction does not make
every participating frequency triplet exactly phase matched. Absolute optical
frequency determines aperture mapping and its area Jacobian. The spatial phase
restored from the local timing reduction uses baseband frequency $\Omega$.

### Geometric clearance

For holes of input diameter $D$ at $(\pm d,\pm d)$, adjacent centres are separated
by $2d$. A circular collection hole centred at the signal corner avoids overlap
with an adjacent input hole only if

```{math}
D_{\rm coll}/D\leq 4d/D-1.
```

This is geometric clearance, not a guarantee against diffracted pump leakage or
aberrations. Increasing collection must be tested against the actual background.

## Cost and quadrature

The reduction is one $J\times K$ matrix product per $(\omega,\tau)$ against the $K$ field
builds already being done. Measured at production shape (K = 384 mixture nodes, J = 144
aperture nodes, 256 frequencies, 230 delays) it costs **+5 %** of runtime. Memory is
$16JK N_\omega$ bytes of precomputed transform matrix — 56 MB there — so the two node
counts multiply and both are worth keeping modest.

One warning. The coherent integrand is **not** the incoherent one, so a mixture
quadrature converged for the $|A|^6$ weight need not be converged here: on the reference
geometry `n_azimuth=8` costs 4e-5 of peak where the incoherent sum is converged, against
3e-7 at the `focal_mixture` default of 16. If you have been running a coarse azimuthal
sampling, raise it before adding an aperture.

## Where it stops being right: the depth integral

The entrance-profile model assigns each depth the same transverse profile.
This neglects input diffraction, beam walk and changes in local overlap. The
exit signal also acquires a transverse propagation phase. These are separate
approximations; the options below address different parts of them.

With the entrance-profile model and the instrument's collection aperture, the full
model already retrieves the simulated 1–40 µm series accurately (see
[validation](../explanation/validation.md)). The options below are opt-in
refinements for thicker slabs or tighter focusing; refine the depth and transverse
quadratures for the field and geometry being modelled.

### `depth_transverse`: modelling the missing decoherence

The exit propagation phase can be included without adding a fitted parameter.
Signal generated at depth $z$ accumulates, over the remaining slab, the transverse
phase the on-axis propagator $e^{i\beta(L-z)}$ leaves out,

$$
\phi_j(\omega, z) \;=\; \bigl[k_z(\omega, k_{\perp j}) - k_z(\omega, 0)\bigr](L - z),
\qquad k_z = \sqrt{(n\omega/c)^2 - k_\perp^2},
$$

evaluated at each aperture node's **absolute** transverse wavevector
$k_{\perp j} = \omega\rho_j/(c\,z_{\text{mask}})$ (mask positions with the hole centre
folded in — the quadratic $k_z$ is *not* invariant under the carrier removal, and the
remainder is physical). At the reference instrument's signal k-extent this reaches
~0.07 rad across a 40 µm slab — small, but it sits inside a coherent sum. The flag
moves the depth sum *inside* the collection transform, each depth carrying
$e^{i\phi_j(\omega,z_q)}$ before the coherent sum over depth:

```python
fn = make_param_trace_fn(
    omega, delays, "pg",
    material="SiO2", thickness=40e-6, npoints=10, omega0=omega0,
    focal=mixture,            # must carry a collection aperture
    depth_transverse=True,
)
```

The same keyword is accepted by {class}`~croak.lbfgs_ad.LBFGSAD` (and therefore by
`retrieve(..., algorithm="lbfgs-ad", depth_transverse=True, ...)`); solvers that cannot
model it refuse loudly. Notes:

* **Requires** `focal` with a `collection` and a dispersive slab. For a
  *full-collection* variant there is deliberately no separate path: use an
  `"integrated"` window much wider than the signal footprint (a large chromatic hole,
  or a `chromatic=False` window covering the whole k grid — with the flag off that
  route reproduces `collection=None` exactly, by Parseval).
* **Assumptions.** The full $k_z$ square root is used (no paraxial expansion) with the
  full Sellmeier $n(\omega)$; the input beams still propagate on-axis — only the
  generated signal's exit propagation is corrected. The *input*-side repair (evolved,
  walked, per-arm filters) is the depth-resolved mixture below, and the two compose.
* **Cost.** One aperture contraction per depth node instead of one total: ≈2.5× the
  collected trace at production shape (K = 96, J = 144, 256 frequencies, 230 delays,
  Q = 10 depth nodes; 0.61 s against 0.24 s per evaluation on an M-series CPU).
  Memory is unchanged — the per-depth transform is accumulated, never materialised.
* **Validation.** The dense 2-D FFT reference test was extended with per-depth exact
  $k_z(\omega,k_\perp)$ propagation
  (`test_matches_a_dense_fourier_reference_with_depth`): the flag improves agreement
  with the exact reference by 25–91× (to 1.4–5.7×10⁻⁵ of peak), while a *flipped*
  sign would be worse than no correction at all — the sign is test-pinned, not
  argued. With a single depth node the phase is a per-node constant: `"integrated"`
  is provably unchanged (a unit-modulus factor dies in $|\tilde S_j|^2$) while
  `"reimaged"` shifts — asserted in `tests/test_depth_transverse.py`, pinning where
  the phase sits relative to the modulus.

These tests check the calculation; check agreement with your instrument by
comparing the predicted trace at a fixed known field.

### fc-z: the depth-resolved mixture (`evolve_profiles`)

`depth_transverse` changes exit propagation. The depth-resolved mixture also
evolves each input beamlet profile through the slab about its walked axis:

```python
mixture = focal_mixture(
    hole_diameter=1e-3, hole_spacing=1e-3, f_foc=0.1, wavelength=260e-9,
    collection=aperture,                      # optional, composes as usual
    evolve_profiles=True,                     # fc-z
    material="SiO2", thickness=40e-6, npoints=10,   # MUST match the trace map
)
fn = make_param_trace_fn(
    omega, delays, "pg",
    material="SiO2", thickness=40e-6, npoints=10, omega0=omega0,
    focal=mixture,
)
```

Each arm $j$ at node $\mathbf r_k$ and depth $z_q$ then applies its own evolved
complex profile $A_j(|\mathbf r_k - \bm\delta_j(z_q)|, \omega; z_q)$ — the aperture's
k-space disc propagated by the transverse part of $k_z$ and evaluated at the in-glass
walked radius $\bm\delta_j(z) = -z\,\mathbf r_j/(f n_0)$ — in place of the single
shared entrance filter. See the
[forward-model explanation](../explanation/forward_model.md) for the formula and
conventions. Notes:

* The slab parameters on the mixture must **match** the trace map's
  (`material`/`thickness`/`npoints`/`quadrature`) — the per-depth filters must sit on
  the same depth nodes as the propagation phases; a mismatch raises.
* PG only; `fit_thickness` is not supported; `depth_transverse` composes
  orthogonally (default off).
* **Cost ≈ fc.** The focal path already batch-FFTs over the depth axis, so the
  per-depth filters slot into the existing $(N, Q)$ multiply; new cost is a
  seconds-level numpy build of the $(3, K, Q, N)$ filter table at model
  construction (~10–50 MB at production shapes).
* **Validation.** The evolved profile matches an independent plane-wave propagator
  to 10⁻⁶ of peak, a zero-evolution table reproduces the entrance filter exactly,
  and the Gaussian control is a null (`tests/test_fcz.py`; see the
  [developer validation guide](../developer_guide/validation.md)).

(only-pg-tg-for-now)=
## Supported interactions

The phase the incoherent sum discards is $\Omega p$ for PG and $-\Omega\delta$ for SD
— in both cases the centred frequency $\Omega$ times one of the two reduced smearing
parameters, so the $(p,\delta)$ reduction is sufficient for an apertured instrument and not just a fully
collected one. Only the PG case is derived and validated, so an SD interaction with a
collection aperture raises rather than quietly using the wrong sign.

$\Omega$ is the **centred** (envelope) frequency while the aperture's own position
scales with the absolute one $\omega_\text{abs}=\omega_0+\Omega$, so the two must be
kept distinct — see the warning in
{mod}`croak.collection`.

## Checking it

Two properties are worth asserting on your own geometry, and both are in
`tests/test_collection.py`: with an unbounded aperture on a Cartesian focal grid the
model must return the incoherent sum exactly (Parseval), and a dense 2-D FFT of the
focal field windowed in k must reproduce the quadrature model with no fitted scale.
Then increase the focal, aperture and depth quadratures until the trace stops
changing. The reference tests are described in
[numerical validation](../developer_guide/validation.md#collection-reference-tests).
