# Modelling the collection aperture

The [chromatic focal mixture](../explanation/forward_model.md#geometric-smearing) sums
**incoherently** over focal-plane position. By Parseval that is exact only if the
detector collects the whole signal beam — and most non-collinear instruments do not:
the signal is picked out of the BOXCARS pattern by a hole, which is precisely a filter
in transverse wavevector.

When the hole is narrow the average over the focal spot is partly **coherent**, and a
coherent sum sharpens where an incoherent one blurs. Left out of the model, that shows
up as a trace error that prefers a *narrower* smearing kernel than the geometry says —
which is exactly what a scale scan on a tightly apertured instrument finds.

## When it matters

Compare two numbers at the carrier:

| quantity | expression | reference instrument at 260 nm |
|---|---|---|
| signal k footprint | $3(\omega/c)(D/2)/f$ | 3.63e5 rad/m |
| aperture k half-width | $(\omega/c)(D_\text{hole}/2)/z_\text{mask}$ | 6.04e4 rad/m |

If the aperture is comparable to or wider than the footprint, the incoherent sum is
fine. Here it is **six times narrower** — about 3 % of the solid angle — so the
instrument sits far closer to the coherent limit than to the incoherent one, and the
collection model is not a refinement.

The other way to see it: a disc of k half-width $\kappa_a$ correlates the focal plane
over an Airy kernel whose first null is at $3.83/\kappa_a \approx 63\,\mu\text{m}$,
while the region that actually emits (where
the arrival offsets $p, \vartheta$ stay inside the pulse duration) is only 15–30 µm
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

`focal=` is already accepted by `lbfgs-ad`, `warm-lbfgs`, `lm`, `lm-optx`,
`lbfgs-optx` and `cma-es`, and the aperture rides on the mixture — so nothing else in a
retrieval script changes. Leaving `collection` unset keeps the full-beam incoherent sum
the model has always used.

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
transverse k-grid, which matters: for the reference instrument it is **96.9 µm on a
500 µm hole** — an edge 19 % of the diameter, passing 0.994 at the centre and still
0.006 at $r = D$. Treating that window as a top hat is a large error, not a rounding one.
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

## Where the aperture has to be centred

Phase matching sends the signal out along $\mathbf k_1+\mathbf k_2-\mathbf k_3$, which
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

This is why {func}`~croak.focal.focal_mixture` now records its `arms`: the incoherent sum
needed only the arm *differences* that make $(p,\vartheta)$, but the aperture needs to
know where the signal actually goes.

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

croak's depth quadrature is **one-dimensional**. Signal generated at every depth in the
slab is added with the *same* transverse phase — that is, fully coherently in
transverse wavevector. In a real 3-D instrument it is not: diffraction, Gouy phase and
the crossing beams' walk-through decorrelate one depth from the next. A coherent
collection model built on a fully coherent depth sum is therefore **too coherent**, and
increasingly so the thicker the slab.

That is measurable. At the known truth on the reference 3-D traces, with the residual
quoted against the reduced kernel at scale 1 (no fitted parameter on either side, and
`depth_transverse` — below — **off**):

| slab thickness | 1 µm | 2 µm | 4 µm | 9.5 µm | 20 µm | 40 µm |
|---|---|---|---|---|---|---|
| `focal` (no aperture) | +3.0 % | +14.0 % | +20.1 % | +29.7 % | +38.1 % | +39.0 % |
| `focal` + `collection` | +2.3 % | −10.0 % | **−23.6 %** | −4.4 % | +20.5 % | +29.8 % |

The incoherent model is uniformly worse and gets steadily worse with depth. The
apertured one is a large improvement up to about 10 µm — on the 1 fs coherent window it
beats even the *scale-fitted* reduced kernel — then decays and crosses over. Scanning a width
scale on the mixture's own $(p,\vartheta)$ confirms the direction: before the aperture
the data prefer 0.75–0.90 (a narrower blur), after it 1.10–1.20 (a wider one).

**Practical rule.** Use `collection=` for slabs up to roughly 10 µm — which includes the
9.5 µm experimental substrate — and expect it to over-correct beyond that. The
crossover is not a property of the aperture; it is the depth integral's missing
transverse decoherence, and it will move with the beam geometry.

### `depth_transverse`: modelling the missing decoherence

The leading part of that missing physics now has an optional, **parameter-free** model.
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
* **v1 caveats.** The full $k_z$ square root is used (no paraxial expansion) with the
  full Sellmeier $n(\omega)$; the input beams still propagate on-axis (only the
  generated signal's exit propagation is corrected), and the arm filters are not
  walked with depth (an amplitude effect bounded ≲1 % over 40 µm; `TODO(depth)` in the
  source).
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

Whether this closes the measured over-correction above is a question for the 3-D
reference traces, not for internal validation; the table above is the flag-off
baseline it will be judged against.

## Only PG (TG), for now

The phase the incoherent sum discards is $\omega p$ for PG and $-\omega\vartheta$ for SD
— in both cases $\omega$ times one of the two reduced smearing parameters, so the
$(p,\vartheta)$ reduction is sufficient for an apertured instrument and not just a fully
collected one. Only the PG case is derived and validated, so an SD interaction with a
collection aperture raises rather than quietly using the wrong sign.

That $\omega$ is the **centred** (envelope) frequency while the aperture's own position
scales with the absolute one, which is the single easiest thing to get wrong here and the
one mistake every internal consistency check misses — see the warning in
{mod}`croak.collection`.

## Checking it

Two properties are worth asserting on your own geometry, and both are in
`tests/test_collection.py`:

* **Parseval.** On a Cartesian focal grid with an unbounded aperture the model must
  return the incoherent sum exactly. Getting the $1/(2\pi)^2$ or the area weights wrong
  shows up here and nowhere else.
* **A dense 2-D FFT.** Building the focal field from the arm tilts directly, taking a
  plain transverse FFT and windowing it in k reproduces the quadrature model with no
  fitted scale (asserted at 2e-3 of peak at the default test resolution; the 3e-7
  figure quoted elsewhere is the mixture-quadrature convergence at `n_azimuth=16`,
  a different check). The per-depth variant of the same reference — exact
  $k_z(\omega,k_\perp)$ propagation of every depth node's signal to the slab exit —
  validates `depth_transverse` to 1.4–5.7e-5 of peak and pins its sign.
