# The wizard GUI

croak ships a PyQt6 Qt-Widgets **wizard** that drives the whole library — load,
preprocess, retrieve, tune dispersion — with embedded Matplotlib plots and
threaded retrieval. It is pure orchestration: every number it computes comes from
the same library functions documented here, so anything the GUI does, you can do
in a script.

```{admonition} Qt is optional at import
:class: note
Importing `croak` does **not** import Qt. Only `croak.gui` does, so the library is
fully usable headless on machines without a display.
```

Every plot carries the standard matplotlib navigation toolbar (Home, Pan,
Zoom-to-rectangle, Save, and a live cursor read-out), so you can zoom and pan to
inspect a trace or marginal. A full redraw (changing a control, recomputing)
resets the view; **Home** restores it.

## Launching

```bash
croak-gui                      # console-script entry point
# or
uv run python -m croak.gui
# or, from Python
python -c "import croak.gui; croak.gui.run_wizard()"
```

{func}`croak.gui.main` is the console-script entry point;
{func}`croak.gui.run_wizard` creates the `QApplication` and shows the window
**maximised** — the plots want the room. Un-maximising gives the window the size
you last left it (remembered between launches and clamped to the screen it opens
on), or on a first run a window filling 90 % of the screen.

The wizard is laid out for displays of **1366×768 logical pixels and up** (a
1920×1080 monitor at Windows' 125 % scaling is 1536×864 logical pixels). Plot text
— titles, axis labels, tick labels, legends and their padding — shrinks as a plot
canvas gets smaller than the size its figure was designed for, down to 70 % of the
default size, and grows back when the window does, so a laptop shows the same
panels at a smaller type size rather than losing the axes behind their labels.
Figures written by a stage's **Save…** button are drawn afresh at full size,
whatever the window size ({class}`~croak.plotting.PlotScale` and
{meth}`~croak.gui.canvas.MplCanvas.render` are the mechanism).

### Desktop integration

The wizard carries its own icon (`croak/gui/resources/croak.svg`, rendered to the
sizes each platform wants) and identifies itself as **croak** rather than as the
Python interpreter that happens to be running it. That is the icon you see in the
macOS Dock, the Windows taskbar and the Linux dock, and the name in the macOS menu
bar and its *About croak* / *Quit croak* items — `run_wizard` sets it by giving Qt an
`argv[0]` of its own, so it reads the same whichever way you launch.

```{admonition} Two macOS caveats
:class: note
The **Cmd-Tab switcher and the Dock tooltip** still say `python3`. Those come from
the application *bundle*, and a plain `croak-gui` process has none; only shipping a
real `.app` would change them, which is not worth the build step for a tool run
from a virtual environment.

On **Windows** `croak-gui` is a `gui-scripts` entry point, so it does not open a
console window behind the wizard — but Windows then has nowhere to write
`stdout`/`stderr`, so a crash before the window appears is silent. Run
`python -m croak.gui` there to see the traceback.
```

## Navigation

Every page except the Welcome menu carries the same chrome: a header bar with the
page title and a **?  Help** button that explains that page's options, and a
footer with **Cancel** and **Main menu** at the bottom left and **← Back** /
**Next →** at the bottom **right**. That includes the two entry pages, where the
forward button does the page's own work and is labelled accordingly —
**Generate →** on the synthetic generator, **Load & continue →** (or
**Load raw → Retrieve →**) on the simulated loader — and is disabled until the
page has something to hand on. **Back** from either entry page returns to the
Welcome menu.

### Layout

Each stage keeps its controls in a scrollable column on the left and its plots on
the right, separated by a draggable divider. Drag it to trade control space for
plot space, or drag it fully to the left to hide the controls altogether; the width
you choose applies to every stage and is remembered for the next launch.

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
   unit and axis auto-detection ({mod}`croak.io`). Optionally enter the measured
   **pulse energy** (J) here to rescale the temporal plots to absolute power (see
   the Dispersion stage). The trace preview uses the same colour scheme as the
   preprocessing stage (white→viridis for positive, white→red for negative), with
   the display scaled to the in-trust-band signal so an aggressive efficiency tilt
   cannot wash the image out.
5. **Marginal check** — compare the trace's frequency marginal with the marginal
   **predicted** from the independent spectrum ({mod}`croak.marginal_checks`):
   exact for SHG (the spectral autoconvolution), transform-limited for PG/TG/SD.
   It overlays the predicted marginal, marks the predicted vs measured **centroid**,
   reports the **width-bracket** check, and exposes the spectral **trust region**
   (defaults to 190–1000 nm for a measured trace, or the full loaded range for
   the noise-free simulated/synthetic traces), a **frequency/wavelength** axis
   toggle (default frequency — the physically correct space; the wavelength view
   applies the λ Jacobian, which blue-shifts a broad marginal), a **log/linear**
   marginal toggle (default linear,
   peak-normalised) and the **3rd-order** `(λ/µm)^exp` efficiency correction. Its
   **Auto** button picks the exponent whose corrected-marginal centroid matches the
   prediction ({func}`~croak.marginal_checks.auto_third_order_exponent`). The
   **SHG marginal correction** also lives here (it moved from Preprocess): it
   scales the trace's frequency rows onto the predicted SHG autoconvolution
   ({func}`~croak.marginal_checks.apply_marginal_correction`), is enabled only for
   an SHG trace with an independent spectrum, and previews the snap onto the
   prediction live (the correction itself is applied during Preprocess). All three
   entry paths (experimental, simulated, synthetic) pass through this stage; see
   [Marginal consistency checks](marginal_checks.md). The trace and the marginal
   comparison share a draggable divider, so either plot can be given the height.
6. **Preprocess** — fringe/DC/low-pass filtering, windowing and regridding
   ({func}`~croak.preprocess.load_and_clean`), with a live before/after preview.
   The controls always show the settings that were actually applied: re-entering
   the stage re-seeds every widget from the current parameters, and those
   parameters survive a round trip back to the loader as long as the trace is
   unchanged (see [Preprocessing](preprocessing.md#settings-across-a-re-load)).
   With **Baseline (arPLS)** on, the **Auto** button beside *τ exclude (fs)* sets the
   hold-out to twice the loaded trace's delay-marginal FWHM. Use it on measured
   traces: without a hold-out the baseline bows into the τ≈0 peak and eats signal,
   worst on the rows that constrain the fit least, which appears as a thin stripe at
   one wavelength (see
   [Peak suck-up](preprocessing.md#peak-suck-up-and-the-hold-out)).
7. **Retrieve** — choose the algorithm (any of the ten, including the global
   [`cma-es`](../explanation/algorithms.md#global-retrieval-cma-es)) and its
   parameters and run. Retrieval executes in a background thread so the UI stays
   responsive, driven by the solver `callback`. By default a **live full-plot
   preview** redraws the complete 12-panel view (retrieved trace, pulse, spectrum,
   marginals, convergence…) as the run converges — throttled to about once a
   second, since the full redraw is far costlier than a single iteration — so a
   long Levenberg–Marquardt run shows its progress in full rather than only an
   error curve. Pressing **Stop** halts the run at that point and leaves the
   latest full plot on screen as a (non-converged) result you can inspect, save or
   post-filter. Untick **Live full-plot preview** to fall back to the cheaper
   trace-error-vs-iteration curve (the full plot is still drawn once the run
   finishes). Every solver supports it, including `copra`/`copra-jax` and the
   COPRA warm-up stage of `warm-lbfgs` — which matters there, since that warm-up
   is 300 iterations by default and the preview would otherwise stay blank for
   most of the run. `warm-lbfgs` plots **both** stages on one convergence curve
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

   Beneath the figure sits a **numeric read-out** that updates as the run
   converges. Its *Pulse parameters* table has one row per curve in the figure,
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

   Two forward-model groups sit alongside the solver options. *Dispersive
   propagation* models a slab (material, thickness, Δz/Npoints — see
   [The forward model](../explanation/forward_model.md)). *Geometrical smearing*
   models the tilted-pulse-front temporal blur of a non-collinear BOXCARS setup
   from the mask hole diameter and edge-to-edge spacing, with the resulting delay
   width shown — and editable, which back-computes the spacing. Both groups grey
   out for solvers that cannot use them; smearing needs one of the autodiff
   solvers. See [Geometric smearing](geometric_smearing.md).
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
**rehydrated from the file** ({func}`croak.save.result_from_saved`) after the cheap
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

td = croak.load_and_clean(trace, lam, delays, "shg", lam_min=380e-9, lam_max=440e-9,
                         input_unit="delay")
result = croak.retrieve_from_tracedata(td, algorithm="lbfgs", maxiters=200)
pr = croak.process_result(result, measured=td.trace)
croak.plot_retrieval(result, measured=td.trace).savefig("retrieval.png")
croak.save_result(result, "retrieval.h5", processed=pr)
```

The [experimental-workflow tutorial](../tutorials/04_experimental_workflow.md)
walks through exactly this, end to end.
