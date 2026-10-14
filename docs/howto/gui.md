# The wizard GUI

The graphical wizard guides you from a loaded trace to a retrieved pulse,
dispersion analysis and uncertainty estimates. Its controls use the same
calculations as the Python interface. If this is your first visit, follow
[installing the GUI](../getting_started/gui.md), then try a synthetic trace
whose pulse is known.

Every plot carries the standard matplotlib navigation toolbar (Home, Pan,
Zoom-to-rectangle, Save, and a live cursor read-out), so you can zoom and pan to
inspect a trace or marginal. A full redraw (changing a control, recomputing)
resets the view; **Home** restores it.

## Launching

For a standalone tool installation:

```console
croak-gui
```

Within a uv project or source checkout, use `uv run croak-gui`. The window
remembers its size and the control-panel width between launches. A larger window
gives the plots more space; figures exported with the stage's **Save…** button
are drawn at full size independently of the window.

### Desktop integration

The application uses the croak icon. On macOS, the Dock tooltip or Cmd-Tab entry
may still display `python3` because the process is launched through Python.
On Windows, launching `uv run python -m croak.gui` from a terminal makes startup
errors visible if the normal launcher closes before opening a window.

## Navigation

Every page except the Welcome menu carries the same controls: a header bar with the
page title and a **?  Help** button that explains that page's options, and a
footer with **Cancel** and **Main menu** at the bottom left and **← Back** /
**Next →** at the bottom **right**. That includes the two entry pages, where the
forward button does the page's own work and is labelled accordingly —
**Generate →** on the synthetic generator, **Load & continue →** (or
**Load raw → Retrieve →**) on the simulated loader — and is disabled until the
page has something to hand on. **Back** from either entry page returns to the
Welcome menu, as does **Back** on Stage 1 and the **Main menu** button. If the
session already holds a loaded trace or a retrieval, the wizard asks first: every
menu entry starts a new session, so choosing one would discard them.

### Layout

Each stage keeps its controls in a scrollable column on the left and its plots on
the right, separated by a draggable divider. Drag it to trade control space for
plot space, or drag it fully to the left to hide the controls altogether; the width
you choose applies to every stage and is remembered for the next launch. A strip
beneath a plot with a titled header (the Retrieve stage's numeric read-out) folds
away with a click on the header, and that is remembered too.

## The stages

The wizard is a stacked sequence of pages with shared navigation. The six linear
stages — **Load → Marginal check → Preprocess → Retrieve → Dispersion →
Uncertainty** — sit behind three entry pages (Welcome, Synthetic, Simulated); the
synthetic generator and the simulated loader are each their session's "stage 1"
and hand the trace straight to the marginal-check stage.

1. **Welcome** — choose to load a measured file, load a numerically simulated
   scan, load a previous retrieval, or generate a synthetic trace.
2. **Synthetic** *(optional)* — build a test trace (pulse shape, interaction,
   delays, optional dispersive slab) using {func}`~croak.forward.maketrace`. Useful
   for trying the workflow without data. The **duration** is the intensity
   (power) FWHM of the temporal envelope — both Gaussian and sech² honour this
   convention ({func}`~croak.pulses.gaussian_pulse`,
   {func}`~croak.pulses.sech_pulse`). The delay and wavelength samplings each
   expose an interlocked **count** and **step** at fixed range (editing one
   updates the other), and a **Spectrum (independent)** group generates a
   fundamental spectrum that is passed downstream (range / step / # points); by
   default it **mirrors the FROG-trace λ window**, scaled to the fundamental
   (×2 for SHG). With it present, the marginal-check overlay and the SHG marginal
   correction work on synthetic traces too. The known pulse (its temporal
   intensity envelope `It`) is carried through and shown beneath the retrieved
   pulse, and saved with the result.
3. **Simulated** *(optional)* — load a numerically simulated `scansave` scan and
   preview it, then hand it to the marginal-check stage. A **thickness** dropdown
   selects which saved propagation depth to retrieve from a multi-thickness scan
   (disabled for older single-thickness files). See
   [Loading numerically simulated traces](loading_simulated.md).
4. **Load** — pick an HDF5/NPZ file and the trace/wavelength/delay datasets, with
   unit and axis auto-detection ({mod}`croak.io`). **Transpose** fixes a trace
   that loaded as (delay × wavelength), and **Reverse delay axis** flips the
   delay-sign convention — negating the scan axis, so the trace is mirrored about
   τ = 0. Which end of a stage scan is "gate late" is a wiring convention the
   file does not record. It matters for **PG and SD**, whose kernels fix the
   direction of time: a wrong setting changes the inferred time direction in the local model.
   With propagation, also check consistency with the calibrated instrument. A
   symmetric transform-limited pulse cannot determine the sign. An **SHG** trace
   is symmetric in delay even for a chirped pulse, so reversing it changes
   nothing and cannot settle SHG's own
   [direction-of-time ambiguity](../explanation/pnps_framework.md#ambiguities) —
   that is resolved after retrieval instead
   ([SHG: settle the direction of time first](postprocessing.md#shg-settle-the-direction-of-time-first)).
   It is the measured counterpart of the simulated loader's switch of the same
   name.
   Optionally enter the measured
   **pulse energy** (J) here to rescale the temporal plots to absolute power (see
   the Dispersion stage). The trace preview uses the same colour scheme as the
   preprocessing stage (white→viridis for positive, white→red for negative), with
   the display scaled to the in-trust-band signal so an aggressive efficiency tilt
   cannot wash the image out.
5. **Marginal check** — compare the trace's frequency marginal with the marginal
   **predicted** from the independent spectrum ({mod}`croak.marginal_checks`):
   the thin-local-model prediction is exact for SHG and transform-limited
   for PG/TG/SD. The [marginal guide](marginal_checks.md) states the assumptions;
   propagation, collection and detector response can alter the comparison.
   It overlays the predicted marginal, marks the predicted vs measured **centroid**,
   reports the **width-bracket** check, and exposes the spectral **trust region**
   (defaults to 190–1000 nm for a measured trace, or the full loaded range for
   the noise-free simulated/synthetic traces), a **frequency/wavelength** axis
   toggle (default frequency; wavelength uses the corresponding density Jacobian), a **log/linear**
   marginal toggle (default linear,
   peak-normalised) and the **3rd-order** `(λ/µm)^exp` efficiency correction. Its
   **Auto** button picks the exponent whose corrected-marginal centroid matches the
   prediction ({func}`~croak.marginal_checks.auto_third_order_exponent`). The
   **SHG marginal correction** is available here: it
   scales the trace's frequency rows onto the predicted SHG autoconvolution
   ({func}`~croak.marginal_checks.apply_marginal_correction`), is enabled only for
   an SHG trace with an independent spectrum, and previews the snap onto the
   prediction live (the correction itself is applied during Preprocess). All three
   entry paths (experimental, simulated, synthetic) pass through this stage; see
   [Marginal consistency checks](marginal_checks.md). Use the draggable divider to adjust the two plot heights.
6. **Preprocess** — fringe/DC/low-pass filtering, windowing and regridding
   ({func}`~croak.preprocess.load_and_clean`), with a live before/after preview.
   The controls always show the settings that were actually applied: re-entering
   the stage re-seeds every widget from the current parameters, and those
   parameters survive a round trip back to the loader as long as the trace is
   unchanged (see [Preprocessing](preprocessing.md#settings-across-a-re-load)).
   With **Baseline (arPLS)** on, the **Auto** button beside *τ exclude (fs)* sets the
   hold-out to twice the loaded trace's delay-marginal FWHM. Treat this as an initial suggestion and inspect the baseline: a free
   baseline can remove real signal, particularly where the wings constrain it
   poorly (see
   [Peak suck-up](preprocessing.md#peak-suck-up-and-the-hold-out)).
7. **Retrieve** — choose the algorithm (any of the ten, including the global
   [`cma-es`](../explanation/algorithms.md#global-retrieval-cma-es)) and its
   parameters and run. The interface remains responsive while retrieval runs. The result is shown on three
   **pages** of panels, as tabs: **Traces** (measured and retrieved trace, log and
   linear, with their colourbars), **Pulse** (retrieved pulse over the delay marginal, spectrum over the
   frequency marginal) and **Diagnostics** (convergence, residuals, spectral
   filter, spectrogram) — the same twelve panels as
   {func}`~croak.plotting.plot_retrieval`, two by two so each has room even on a
   laptop display. **Pop out** (top right of the tabs) opens the current page in
   its own window, which follows later results; each tab's toolbar **Save** writes
   that page, and the stage's **Save…** writes the full twelve-panel overview as
   `retrieval.pdf` beside `result.h5` and `options.toml`. The **Display** group's
   **Spectrum axis** draws the spectrum panel against **frequency** (PHz, the
   default — the space the retrieval works in) or **wavelength** (nm, the
   spectrometer's axis, carrying the $1/\lambda^2$ density Jacobian); both are unit-peak densities
   in their own variable, the choice is remembered, and the exported PDF follows
   it. By default **Live full-plot preview** updates the visible page during
   retrieval. Other pages update when selected. Until a full preview is
   available, the convergence curve shows progress.
   Pressing **Stop** halts the run at that point and leaves the latest full plot
   on screen as a (non-converged) result you can inspect, save or post-filter.
   Untick **Live full-plot preview** to fall back to the cheaper
   trace-error-vs-iteration curve (the full plot is still drawn once the run
   finishes). Every solver supports progress reporting, including both stages of
   `warm-lbfgs`. `warm-lbfgs` plots **both** stages on one convergence curve
   with a dashed rule at the handover; read the two sides as different units of
   work (COPRA iterations, then L-BFGS function evaluations). The **Initial
   guess** selector makes automatic, transform-limited, random, perfect-Gaussian,
   known **truth**, and reuse-previous starts mutually exclusive. Truth uses the
   complex field stored by ModelPNPS (and reports an explicit error for an older
   intensity-only file). **Reuse previous result** warm-starts one run from the
   last — handy for the global-then-local refine recipe. It seeds the previous **spectrum** as the
   initial guess *and* the extras that run fitted (thickness, τ₀, smearing width)
   as the starting values for fitting them again, naming them in the status line;
   an extra you are no longer fitting keeps the value set in the controls. When
   the run finishes, the status line reports the fitted extras alongside the
   error and FWHMs — including the geometric-smearing width as a delay σ in fs
   with its multiplier on the nominal mask geometry (which is left untouched). When the trace came from the synthetic or
   simulated workflow, the **known-truth** temporal intensity is drawn as a thin
   black line beneath the retrieved and transform-limited pulses. When its complex
   field is available, dotted truth-phase lines appear in both time and wavelength
   panels and are named in their legends. The truth arrays are also written into
   the saved `result.h5`.

   Beneath the pages sits a **numeric read-out** that updates as the run
   converges; click its header to fold it away and give the plots its height
   (remembered between launches). Its *Pulse parameters* table has one row per
   curve in the figure,
   labelled to match the legends — `R` (retrieved), `M` (the independent measured
   spectrum), `TL` (transform-limited) and `truth` — and four columns: peak
   wavelength, spectral FWHM, temporal FWHM and peak power. Cells that are not
   defined show an en dash rather than a misleading number: the measured spectrum
   carries no phase, so it has no temporal envelope; the transform-limited pulse is
   the retrieved $|E(\omega)|$ with the phase removed, so its spectrum *is* the
   retrieved one and is reported once, on the `R` row. The peak-power column needs
   a measured **pulse energy** (set it on the Load or Dispersion stage) and shares
   one SI prefix, shown in its header, so the values compare at a glance.

   Below the table, `Iteration`, `Elapsed` and `Error R` track the run itself, and
   `τ₀`, `Thickness` and `Smearing` show the forward model's extra parameters, each
   marked **(fit)** or **(fixed)** so it is clear whether the retriever was free to
   move it. The three run counters tick every iteration; the table refreshes with
   the live full-plot preview, so with that preview turned off it stays blank until
   the run finishes.

   The last line, `ε(Iₜ)`, `ε(I𝜔)` and `ε(E𝜔)`, appears when the trace came from
   the synthetic or simulated workflow and so has a **known** pulse: the temporal
   intensity, spectral intensity and complex-field errors against it
   ({mod}`croak.truth_metrics`). They answer a different question from the
   `Error R` above them — that scores the fit to the *trace*, these score the
   distance to the true *pulse*, and a FROG trace does not determine the pulse
   uniquely, so a small `R` can sit on a wrong answer. Read them together: a small
   `ε(I𝜔)` with a large `ε(E𝜔)` means the amplitude is right and the phase is not.
   The two spectral ones need the truth's complex field, so an intensity-only file
   shows `ε(Iₜ)` alone rather than a number derived from something else.

   Forward-model groups sit alongside the solver options. *Dispersive
   propagation* models a slab (material, thickness, Δz/Npoints — see
   [The forward model](../explanation/forward_model.md)). *Geometrical smearing*
   models the tilted-pulse-front temporal blur of a non-collinear BOXCARS setup
   from the mask hole diameter and edge-to-edge spacing, with the resulting delay
   width shown — and editable, which back-computes the spacing. Both groups grey
   out for solvers that cannot use them; smearing needs one of the autodiff
   solvers. The chromatic focal and collection settings are described in
   [collection aperture modelling](collection_aperture.md). See also
   [geometric smearing](geometric_smearing.md) for the reduced model.
8. **Dispersion** — apply and auto-tune residual dispersion
   ({mod}`croak.dispersion`) to compress the retrieved pulse, including real glass
   (built-in, custom, or [refractiveindex.info](../howto/materials_and_mirrors.md)
   materials) and a **mirror stack** — add rows of `(mirror, polarisation, bounces,
   remove/add)` to back-propagate (remove) beam-path coating mirrors or add
   chirped-mirror bounces — with the temporal profile and spectrogram updating as you
   drag the sliders. Enter a **pulse energy** (J) (shared with the Load stage) to
   switch the temporal plot to absolute power and read off the current and
   transform-limited **peak power**; compressing the pulse raises the peak power
   towards the transform-limited maximum at fixed energy.
9. **Uncertainty** — put an error bar on the retrieved FWHM
   ({mod}`croak.uncertainty`). Choose the estimator (parametric or resampling
   bootstrap, the substrate-thickness systematic, or the fast analytic
   covariance), set the replicate count and confidence level, and run it on a
   worker thread with a progress bar. The stage reports the point estimate with
   its 68 % and 95 % intervals, draws the temporal confidence band over the
   retrieved profile, and saves everything alongside the result. See
   [Estimating FWHM uncertainty](uncertainty.md).

## Sessions

The GUI persists its settings — preprocessing choices, algorithm and options — to
a TOML file via {func}`croak.save.save_options` / {func}`croak.save.load_options`,
so you can reload a prior session. Retrievals are saved to HDF5 with
{func}`croak.save.save_result`. **Load previous retrieval** reloads the saved
`options.toml`; if a `result.h5` sits beside it, the retrieved result is
**restored from the file** ({func}`croak.save.result_from_saved`) after the cheap
load + preprocess steps replay — so the solver is **not** re-run (it falls back to
re-running only if the file is missing or its grid no longer matches). See
[Saving and loading](saving_and_loading.md).

## Scripting the same workflow

Everything in the wizard maps to a few library calls — load → clean → retrieve →
process → plot → save:

```python
import croak

fd = croak.io.list_datasets("frog.h5")
lam = fd.load("wavelength") * croak.io.unit_to_si("nm")
delays = fd.load("delay") * croak.io.unit_to_si("fs")
trace = fd.load("trace")

td = croak.load_and_clean(trace, lam, delays, "shg", lam_min=730e-9, lam_max=900e-9,
                         input_unit="delay")
result = croak.retrieve_from_tracedata(td, algorithm="lbfgs", maxiters=200)
pr = croak.process_result(result, measured=td.trace)
croak.plot_retrieval(result, measured=td.trace).savefig("retrieval.png")
croak.save_result(result, "retrieval.h5", processed=pr)
```

The [experimental-workflow tutorial](../tutorials/04_experimental_workflow.md)
walks through exactly this, end to end.
