# Validation

croak's forward models and solvers were validated in the companion paper —

> J. C. Travers and C. Brahms, *Extreme ultrashort pulse retrieval with
> differentiable physical forward models* (to be published)

— against first-principles three-dimensional simulations of a complete
TG-FROG instrument. Every retrieval in that paper was done with croak. This
page summarises the study: what was simulated, what each forward model
recovered, and what the results say about how retrievals should be run and
reported. Two reduced datasets from the study ship with croak, so the central
results can be reproduced directly (see [Reproducing](#reproducing)).

## The virtual instrument

Testing a retrieval model needs traces whose distortions are real but whose
ground truth is known. The paper generates them with
[ModelPNPS.jl](https://github.com/LupoLab/ModelPNPS.jl), which simulates the
entire instrument from the propagation physics: a transform-limited 1 fs
pulse at 260 nm illuminates a four-hole BOXCARS mask; the three transmitted
beamlets are focused into a fused-silica substrate and propagated coherently
in 3D with full angular-spectrum dispersion, diffraction and the Kerr
nonlinearity; and the signal is collected through an apertured window in the
far field, delay by delay, exactly as a spectrometer would record it.

```{image} ../_static/paper/figure_boxcars3d.png
:alt: The folded BOXCARS TG-FROG geometry — the four-aperture mask and the optical chain.
:width: 95%
:align: center
```

Nothing about the measurement is imposed: the transient grating, the
phase-matched signal, the geometrical smearing and the chromatic vignetting of
the apertures all emerge from the propagation. The apertures act as
frequency-dependent spatial filters, so the pulse that actually gates the
interaction is the mask-vignetted beamlet — 1.03 fs against the 1.0 fs source
— and all ground-truth comparisons use it. The simulated traces are
preprocessed exactly as experimental data would be (fringe removal, baseline
subtraction, regridding; see [Preprocessing](../howto/preprocessing.md)).

## The failure being fixed

Conventional retrieval fails silently on these measurements. A
standard thin-medium algorithm fits the simulated traces to trace
errors below 1% — values normally read as good convergence — while
returning pulses up to 2.6 times too long, because the optimiser absorbs the
unmodelled substrate dispersion and geometry into the pulse.

```{image} ../_static/paper/figure_dispersion_failure.png
:alt: Thin-medium retrieval fits the traces well and returns pulses up to 2.6 times too long.
:width: 85%
:align: center
```

## The three forward models

The paper organises croak's forward-model options into a hierarchy of three
usable models (see [The forward model](forward_model.md) for the equations):

| model | physics | croak options | valid when |
|---|---|---|---|
| **standard** | single-plane PNPS | defaults | thin medium, multi-cycle pulses |
| **extended** | + dispersive slab + BOXCARS smearing kernel | `dispersive=True`, `smearing=True` | any thickness; multi-cycle, or open collection |
| **full** | + chromatic focal mixture + modelled collection aperture | `focal=True`, `collection=...` | octave bandwidth through any collection aperture |

Two dimensionless switches decide which member the data need: the fractional
bandwidth (chromatic effects) and the collection-to-generation aperture ratio
(coherent-collection effects). For multi-cycle pulses both are off and the
extended model was exact for every purpose tested. The extended model costs
about the same as the standard one; the full model costs minutes rather than
seconds per retrieval on a CPU (see the [README](https://github.com/LupoLab/croak#performance)
for GPU timings).

## Retrieval versus substrate thickness

The central result. Each trace of the thickness series — the same 1.03 fs
pulse after 1 to 40 µm of fused silica — is retrieved with all three models
under identical solver settings (warm-started L-BFGS, per-frequency scale
factors, spectrum-divergence regularisation at the settled weights).

```{image} ../_static/paper/figure_thickness1fs.png
:alt: Retrieved duration versus substrate thickness for the three models.
:width: 80%
:align: center
```

With the extended model the retrieved duration is essentially independent of
the substrate: 1.04–1.15 fs across the whole series, at trace errors of
0.07–0.37%. The full model returns 1.12–1.21 fs, flat in thickness. The
standard model broadens monotonically from 1.4 fs at 1 µm to 2.6 fs at 40 µm,
at trace errors that never exceed 0.7%. Once the measurement physics is in
the forward model, the substrate thickness stops being a design constraint —
a 40 µm substrate retrieves as accurately as a 1 µm one, relaxing the usual
pressure toward fragile ultrathin media and their weak signals.

## Structured pulses and dispersion metrology

A symmetric transform-limited test pulse is blind to a whole error class (a
time-reversed retrieval with wrong-sign phase reproduces its trace exactly),
so the study also retrieves structured pulses: chirp of both signs
(±2 fs² on the 1 fs pulse), third-order phase, and a double pulse, all
simulated through the full 3D instrument.

- The **full model** recovers the chirp of both signs to within 0.13 fs²
  through 28 µm of substrate, and the third-order coefficient to better than
  8% through 20 µm.
- The **extended model** recovers only ~0.8 of the true GDD span when the
  collection aperture is tight; opening the aperture restores it — see below.
- The **standard model** does not determine the dispersion at this bandwidth
  at all. On the negatively chirped arm it returns the pulse *recompressed by
  the substrate's own dispersion* — 1.8 fs for a 5.5 fs truth — at a trace
  error indistinguishable from the correct model's.
- A double pulse with a 30% satellite retrieves with the satellite position
  pinned to ±0.1 fs by every model; only the full model holds its amplitude
  at depth (the standard model inflates it to ~50% by 20 µm).

## Beam geometry: the smearing kernel is quantitative

In a focused non-collinear geometry the relative delay between the beams
varies across the focal spot, blurring the trace along the delay axis. The
paper derives a closed-form two-parameter kernel for the BOXCARS mask — the
one implemented in {mod}`croak.smearing` — with widths set by the mask ratio
$d/D$ and the wavelength alone (the focal length cancels).

```{image} ../_static/paper/figure_smearing_dd.png
:alt: Retrieval versus mask aperture separation, with and without the smearing kernel.
:width: 80%
:align: center
```

Across a simulated series in which the mask separation doubles the smearing
widths, a smearing-blind retrieval broadens from 1.12 to 1.46 fs at the
9.5 µm substrate (and by up to 60% at the widest separation on thinner
substrates), following the kernel's analytic prediction to within 10%; with the kernel at its
parameter-free geometric prediction the duration stays within 0.05 fs of the
truth at every separation. The kernel width was also verified at the trace
level, with no retrieval in the loop, to $1.06 \pm 0.02$ of its predicted
value.

Two findings generalise. First, the trace error can actively favour the
wrong model: at small separations the smearing-blind retrieval reaches a
*lower* trace error than the correct model while reporting a pulse 34% too
long, because an unconstrained pulse buys more misfit reduction than the
correct blur does. Second, the damage from ignoring the geometry is largest
for *thin* substrates — which is what is conventionally recommended for
few-femtosecond work.

## The collection aperture: two validated routes

At octave bandwidth the collection aperture is an optical element of its own:
it selects a coherent subset of the signal field, which reshapes the trace
along the delay axis in a way no per-frequency calibration can absorb. The
paper measures the consequence directly, recording the same scans through six
collection holes.

```{image} ../_static/paper/figure_aperture.png
:alt: Recovered GDD span and retrieved duration versus collection hole diameter.
:width: 80%
:align: center
```

Through a tight hole the extended model recovers only ~0.8 of the true GDD
span; opening the hole to 1.5–2× the generation apertures restores it —
the hardware route. The full model holds the span at 0.98–1.01 at every
hole — the software route, needing no change to the instrument. The usable
optimum for the hardware route is bounded on the other side by pump leakage
entering the window. See
[Modelling the collection aperture](../howto/collection_aperture.md).

## A real single-cycle pulse

The final validation runs the entire measurement chain on a pulse with no
analytic regularity: the simulated resonant-dispersive-wave emission of a
hollow-capillary-fibre source — 1.06 fs, with a structured spectrum and
trailing satellites — injected into the virtual instrument exactly as a
measurement would receive it.

```{image} ../_static/paper/figure_rdw.png
:alt: The single-cycle RDW pulse retrieved by the extended model, against the standard model and the truth.
:width: 80%
:align: center
```

The standard model fits this trace to R = 0.21% — a figure that would pass
review — and returns 2.58 fs for the 1.06 fs pulse. The extended model with
its kernel scale and delay offset fitted reaches R = 0.03% and returns the
pulse itself: structure, spectrum and duration to +4.5% at the experimental
substrate. This dataset ships with croak (`examples/data/tgfrog_sim_rdw_duv.h5`).

## Solvers and reproducibility

The solver study behind croak's defaults:

- **Warm-starting matters more than the solver.** Over twelve random starting
  phases, warm-started L-BFGS (croak's `warm-lbfgs` default) collapses onto a
  single endpoint; plain L-BFGS reaches a similar mean with a 13× larger
  spread. A solver whose endpoints scatter is returning the initial guess as
  much as the data.
- **Multi-start with lowest-R selection is part of the protocol**, not a
  precaution: over 219 random starts, a single start lands in the true basin
  only 45–80% of the time, but the lowest-trace-error member of each ensemble
  landed in the true basin in every configuration tested.
- **Levenberg–Marquardt is a polish, not a cold solver**: it reaches the same
  solution as L-BFGS in a twelfth of the iterations but at ~23× the wall
  clock per step, and from random starts it is the least reliable arm.
- **Dispersive COPRA is a good choice within its domain** — no
  tuning parameters, monotone, reproducible — but its projection step cannot
  represent the smeared or collection models (see
  [Algorithms](algorithms.md)), and per-frequency scaling must not be handed
  to it (see [Regularisation](../howto/regularisation.md)).

## What certifies a retrieval

The study's methodological findings concern the error metrics themselves:

- **The trace error cannot certify the model.** Pushing the *true* field
  through the forward model gives R = 0.18–0.38% against the simulated
  traces; retrievals of the same traces reach 0.10–0.13%. The retrieval fits
  the measurement better than the pulse that generated it — it can only do so
  by distorting the field to absorb residual model mismatch.
- **The retrieved duration cannot certify it either**: a deliberately
  over-broad smearing kernel produced a nearly exact duration through a pair
  of compensating errors, while a better kernel read 5% long.
- Report a retrieval the way it is diagnosed: a trace error to show the fit
  converged, the retrieved spectrum overlaid on an independently measured
  one, a pulse shape rather than a single width, and — where no truth exists
  — the spread over random starts. {mod}`croak.truth_metrics` implements the
  paper's error metrics for synthetic and simulated data.

## Limits on medium thickness

How thick is too thick? In units of the dispersion length
($L_D = T_0^2/\beta_2$, 1.7 µm for a 1 fs pulse in fused silica at 260 nm)
the 40 µm validation already operates at 23 $L_D$ — for scale, like
retrieving a 10 fs, 800 nm pulse through 23 mm of glass. A one-dimensional
scaling study extends this: traces synthesised at up to 1 mm (575 $L_D$, an
820 fs exit pulse) still re-retrieve the 1 fs pulse, and a fitted thickness
recovers the 1 mm to 0.3%. What degrades first is optimisation — beyond a few
hundred micrometres many random starts stall, and the retrieval cost grows as
$L^2$.

The eventual physical ceilings are not dispersion. The delayed (Raman) part
of the nonlinear response becomes the leading unmodelled physics beyond
~40 µm in fused silica, but a direct 3D test bounds its effect on the
retrieved duration at 0.02 fs there — so croak deliberately carries no Raman
term. The method also transfers across media: a full validation ladder on
MgF₂ (the material a production DUV instrument would likely use) reproduces
the fused-silica results, confirming that thickness matters through
accumulated dispersion, not through the medium carrying it.

(reproducing)=
## Reproducing

Two reduced datasets from the study ship in `examples/data/` (trace, axes and
exact ground truth; see the [data README](https://github.com/LupoLab/croak/blob/main/examples/data/README.md)):

```bash
uv run python examples/example_paper_thickness.py   # the 1 fs thickness series
uv run python examples/example_paper_rdw.py         # the single-cycle RDW pulse
```

Each retrieves with the extended and standard models and reports the trace
errors, durations and truth errors — reproducing the pattern above, including
the case where the standard model reaches a lower trace error on a wrong
pulse. The [simulated-trace guide](../howto/loading_simulated.md) documents
the loader; the [geometric smearing](../howto/geometric_smearing.md) and
[collection aperture](../howto/collection_aperture.md) guides cover the
instrument models.
