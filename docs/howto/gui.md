# Using the GUI

The graphical interface takes you from a loaded trace to a retrieved pulse,
dispersion analysis and uncertainty estimates in one window. Its controls use the
same calculations as the Python interface. If this is your first visit, follow
[installing the GUI](../getting_started/gui.md), then try a synthetic trace
whose pulse is known.

A session runs through five stages — **Data → Check → Prepare → Retrieve →
Analyse** — which you reach from **Home**. The top bar always shows where you are, which
stages you can open and the one action that moves the session on.

## Launching

For a standalone tool installation:

```console
croak-gui
```

Within a uv project or source checkout, use `uv run croak-gui` or
`uv run python -m croak.gui`. From Python, call {func}`croak.gui.run_gui`:

```python
import croak.gui

croak.gui.run_gui()
```

{func}`croak.gui.main` is the console-script entry point;
{func}`~croak.gui.run_gui` creates the `QApplication` and shows the window
({class}`croak.gui.MainWindow`) **maximised**, because the plots need the room.
Un-maximising gives the window the size you last left it (remembered between
launches and clamped to the screen it opens on), or on a first run a window
filling 90 % of the screen.

The GUI is built on PySide6. Every plot on screen is an interactive pyqtgraph
view; figures you export or save are drawn by matplotlib ({mod}`croak.plotting`),
so they look the same as the figures a script makes.

### Small screens

The window is laid out for displays of **1366×768 logical pixels and up** (a
1920×1080 monitor at Windows' 125 % scaling is 1536×864 logical pixels). Each
stage puts its settings in a narrow column on the left (340 pixels) and gives the
rest of the window to its plots; when the window is narrower than 1000 pixels
the plots move below the settings instead. The
settings are grouped in sections that fold, and Retrieve's
advanced sections start folded, so a laptop screen shows the settings you use
most without scrolling. Beneath Retrieve's plots, the run's numbers and its
convergence share one short strip.

Plot text stays at the same size whatever the window size. Exported figures
(see [Figures for publication](#figures-for-publication)) and the files a
**Save** writes are drawn afresh at their own design size, so they do not depend
on the window either.

### Desktop integration

The GUI carries its own icon (`croak/gui/resources/croak.svg`) and identifies
itself as **croak** rather than as the Python interpreter running it: that is the
icon in the macOS Dock, the Windows taskbar and the Linux dock, and the name in
the macOS menu bar. Two platform details remain:

- On **macOS** the Cmd-Tab switcher and the Dock tooltip still say `python3`.
  Those come from an application bundle, which a plain `croak-gui` process does
  not have.
- On **Windows** `croak-gui` is a `gui-scripts` entry point, so it opens no
  console window, but a crash before the window appears is then silent. Run
  `uv run python -m croak.gui` from a terminal to see the traceback.

## Home

Home is where a session starts. Its first three tiles each start a session
from one kind of data:

- **Measured trace** — load a FROG measurement from HDF5 or NPZ, with its
  independent spectrum, backgrounds and calibration curves.
- **Simulated scan** — load a numerically simulated scan (a `scansave` file),
  with its known input pulse and instrument geometry.
- **Synthetic trace** — generate a trace from a pulse you specify, with
  dispersion and noise, and test a retrieval against the known answer.

The fourth tile, **Open a session**, reopens a saved session from its
`options.toml` (see [Sessions](#sessions)). Beneath the tiles, **About croak** opens the About
page: the version, links to this manual, the source on GitHub, the package on
PyPI and the issue tracker, the authors, and the copyright and licences. The **Recent sessions** list on the
right holds the last six sessions you saved or opened, named by their folder,
with the folder's location and the time the session was last saved; a session
whose file has gone is left out. Click one to reopen it.

Starting a new session, or opening one, replaces the session in the window. If
that session holds a trace or a retrieval, the GUI asks first, with **Discard**
or **Cancel**; **Save session…** in the ⋯ menu keeps it.

## The top bar

The top bar carries, from left to right:

- the **croak** brand and a context label: what the session is (*Measured
  trace*, *Simulated scan* or *Synthetic trace*) and, once it has been saved or
  opened, its folder's name;
- **Home**, which returns to Home (so does the brand). The session stays open:
  its stage buttons are still there, so you can go back to it;
- the five stage buttons **Data · Check · Prepare · Retrieve · Analyse**. A
  stage that cannot be opened yet is greyed out, and its tooltip says why;
- the **primary action**, the button that moves the session on, and on
  Retrieve, once a retrieval is done, **Retrieve again** beside it;
- the **⋯** menu: **Save session…**, **Open a session…**, **Export this page as
  a figure…**, **Show the settings (options.toml)**, **Show as a Python
  script**, **Help for this stage** and **About croak**.

The menu bar repeats these under **File** (Home, Open a session…, Save
session… and Export this page as a figure…, with the platform's Open and Save
shortcuts and Ctrl+E, ⌘E on macOS), **View** (the settings and the script) and
**Help** (Help for this stage, F1 on Windows and Linux, ⌘? on macOS, and About
croak, which macOS moves to the application menu). Passing notices, such as "Saved …" or why a stage cannot open,
appear at the foot of the window, which also holds the
[display settings](#display-spectral-axis-and-log-range) while a session is
open, and hides on Home when there is nothing to say.

### The route

Which stages can be opened depends only on what the session holds:

| Stage | Opens once |
|---|---|
| Data | a session has started |
| Check | a trace has been loaded |
| Prepare | a trace has been loaded |
| Retrieve | the trace has been prepared |
| Analyse | a retrieval has run |

A simulated scan loaded with **Skip filtering & regrid** is retrieved on its own
grid, so it goes from Data straight to Retrieve and Prepare stays disabled
(Check stays open, to look at the scan, and its Continue also goes to
Retrieve).
Changing a setting upstream clears what was derived from it — a new trace clears
the prepared trace and the retrieval — so the GUI never shows a pulse retrieved
from superseded settings, and the later stages close again until they are
recomputed.

The primary action follows the stage:

| Where | Primary action |
|---|---|
| Data | **Continue**: load or generate the trace and go on to Check |
| Check | **Continue** to Prepare |
| Prepare | **Continue** to Retrieve |
| Retrieve | **Retrieve**, **Stop** while it runs, then **Analyse**, with **Retrieve again** beside it |
| Analyse | **Save** (the session); on the Uncertainty page **Estimate**, or **Stop** while it runs |

The action is greyed out until it can be done: Continue on Data, for example,
until there is a trace to hand on.

## The stage pages

Every stage page has the same layout. On the left is a form of **sections**,
each headed by a title you click to fold or unfold it. Hover over a control for
its help; when a control has the keyboard focus, the same help is shown as one
line under the form. **Help for this stage** (⋯ menu or F1) opens a summary of
every setting on the page. The Retrieve form also has a **Find a setting** box,
which shows only the rows whose label or help contains what you type; typing
part of a section's title shows that whole section.

At the foot of the form a **ready line** says whether the stage can go on: green
when it can, amber with a warning, red when something has to be fixed first. On
the right are the stage's plots.

## Plots

Every plot shares the same handling:

- **Hover** to show a crosshair and a read-out: the position, and the value
  of the map or of each curve under the pointer.
- **Scroll** to zoom about the pointer, and **drag** to pan.
- **Double-click** to return to the view the data were first shown with, and
  a map to its colour scale.
- **Drag a colour bar's ends** (or the bar itself) to change that map's colour
  scale on its own, for example to look further down a logarithmic trace. A
  signed map keeps its white at zero wherever the scale is dragged.

Plots that share an axis are linked, so zooming one zooms its partners: the
trace maps on Prepare, and the measured and retrieved traces on Retrieve and
Analyse. Images are shown on evenly spaced axes at their finest sample spacing,
so no detail is lost where the samples are densest. A logarithmic (dB) trace
shows the positive part of the signal; negative samples, left by background
subtraction, show on the linear maps, which run from white to viridis for
positive values and from white to red for negative ones.

### Display: spectral axis and log range

Two settings at the foot of the window hold for every plot on every page. They
are remembered between launches and stay until you change them or click
**Reset**:

- **Spectral axis**: **As computed** keeps each plot on the axis its data were
  computed on — wavelength for a loaded trace and the generator's previews,
  frequency for the prepared trace and the retrieval. **Wavelength** or
  **Frequency** draws every spectral plot, maps and curves alike, against that
  axis, with ν = c/λ. An intensity is a density per unit of its axis, so it
  takes the Jacobian too, $I_\nu = I_\lambda\,\lambda^2/c$ and
  $I_\lambda = I_\nu\,\nu^2/c$: a broad spectrum changes shape, not only
  position. Phases and filter factors only move. Each plot keeps its peak, and a
  map is resampled onto an even grid on the new axis, for drawing only
  ({mod}`croak.plotting.display`).
- **Log range**: how far below its peak every logarithmic (dB) map reaches,
  from 5 to 120 dB (30 dB by default). A dragged colour bar overrides it for one
  map until that map is next drawn.

Exported figures follow both: a plot or page exported from the screen is drawn
as shown, and croak's own figures of a result (Analyse's Overview, Traces, Pulse
and Diagnostics pages, and the `retrieval.pdf` a saved session holds) take the
log range and draw their spectrum on the chosen axis. Their trace maps are
always against frequency, so with the display on wavelength Analyse's pages
are exported as they are shown instead.

**Right-click** a plot for its own menu: **Export this plot as a figure…**,
**Save as an image…** (a PNG of the plot as it is on screen) and **Copy as an
image**.

The pages on Analyse also have **Export ▾**: **Figure…** saves that page as a
figure (as **Export this page as a figure…** does), and **Pop out** opens it in
a window of its own, which follows later results.

### Figures for publication

Any page, or any single plot, can be saved as a figure drawn by matplotlib:
**Export this page as a figure…** (⋯ menu, File menu or Ctrl+E) for the page on
screen, or **Export this plot as a figure…** from a plot's right-click menu.
Choose **PDF** or **SVG** for a vector figure, or **PNG**. In a vector figure the
axes, curves and text stay sharp and the text stays editable text; the colour
maps are embedded as images at 300 dpi (rasterized), so a trace map stays small
and opens quickly. A PNG is written at 300 dpi.

The figure shows the same data as the screen, with the same colours, line styles
and labels, sized for print: each plot is 3.4 × 2.6 inches (a journal's single
column) in 8-point type, and a page keeps its plots' arrangement. Analyse's
Overview, Traces, Pulse and Diagnostics pages are saved as croak's own figure of
that page ({func}`~croak.plotting.plot_retrieval_page`), as a script makes it.
From a script, {mod}`croak.plotting.export` draws the same screen views.

## The stages

### Data

Data holds the loader for the session's kind of data. **Continue** loads (or
generates) the trace and moves on to [Check](#check). Check opens as soon as a
trace is loaded, so a measured trace loaded with **Load & Preview** can be
checked from the top bar straight away.

#### Load the trace (measured)

Pick an HDF5/NPZ file and the trace/wavelength/delay datasets, with unit and axis
auto-detection ({mod}`croak.io`), then press **Load & Preview**; Continue is
available once the trace has loaded. An optional independent spectrum can come
from HDF5, NPZ or CSV, with background subtraction, and wavelength calibration
curves from CSV. **Transpose** fixes a trace that loaded as (delay ×
wavelength), and **Reverse delay axis** flips the delay-sign convention —
negating the scan axis, so the trace is mirrored about τ = 0. Which end of a
stage scan is "gate late" is a wiring convention the file does not record. It
matters for **PG and SD**, whose kernels fix the direction of time: a wrong
setting time-reverses the retrieved pulse and flips the sign of every phase
order. The error is invisible on a symmetric transform-limited pulse, so check
the setting once against a pulse of known chirp (for example after adding a
known glass thickness). An **SHG** trace is symmetric in delay even for a
chirped pulse, so reversing it changes nothing and cannot settle SHG's own
[direction-of-time ambiguity](../explanation/pnps_framework.md#ambiguities) —
that is resolved after retrieval instead
([SHG: settle the direction of time first](postprocessing.md#shg-settle-the-direction-of-time-first)).
It is the measured counterpart of the simulated loader's switch of the same
name.

Optionally enter the measured **pulse energy** (J) here to rescale the temporal
plots to absolute power (see the Dispersion page under [Analyse](#analyse)).
The trace preview uses the same colour scheme as Prepare (white→viridis for
positive, white→red for negative), with the display scaled to the in-trust-band
signal so an aggressive efficiency tilt cannot wash the image out.

#### Load the scan (simulated)

Load a numerically simulated `scansave` scan and preview it live. A
**thickness** dropdown selects which saved propagation depth to retrieve from a
multi-thickness scan (disabled for older single-thickness files). Continue loads
the scan and moves to Check; with **Skip filtering & regrid (raw →
retrieval)** ticked it builds the as-simulated trace on its native grid and goes
straight to Retrieve. See
[Loading numerically simulated traces](loading_simulated.md).

#### Generate the trace (synthetic)

Build a test trace (pulse shape, interaction, delays, optional dispersive slab
and noise) using {func}`~croak.forward.maketrace`, previewed live. Useful for
trying the workflow without data. The **duration** is the intensity (power) FWHM
of the temporal envelope — both Gaussian and sech² honour this convention
({func}`~croak.pulses.gaussian_pulse`, {func}`~croak.pulses.sech_pulse`). The
delay and wavelength samplings each expose an interlocked **count** and
**step** at fixed range (editing one updates the other), and a **Spectrum
(independent)** section generates a fundamental spectrum that is passed
downstream (range / step / # points); by default it **mirrors the FROG-trace λ
window**, scaled to the fundamental (×2 for SHG). With it present, the
marginal-check overlay and the SHG marginal correction work on synthetic traces
too. Continue generates the trace and moves to Check. The known
pulse (its temporal intensity envelope `It`) is carried through, shown beneath
the retrieved pulse, and saved with the result.

### Check

Compare the trace's frequency marginal with the marginal **predicted** from the
independent spectrum ({mod}`croak.marginal_checks`): exact for SHG (the spectral
autoconvolution) and transform-limited for PG/TG/SD, within the thin local
model; the [marginal guide](marginal_checks.md) states its assumptions. The page
overlays the predicted marginal, marks the predicted vs measured **centroid**,
reports the **width-bracket** check, and exposes the spectral **trust region**
(defaults to 190–1000 nm for a measured trace, or the full loaded range for the
noise-free simulated/synthetic traces), a **log/linear** marginal toggle (default
linear, peak-normalised)
and the **3rd-order** `(λ/µm)^exp` efficiency correction. Its **Auto** button
picks the exponent whose corrected-marginal centroid matches the prediction
({func}`~croak.marginal_checks.auto_third_order_exponent`). The **SHG marginal
correction** is available here: it scales the trace's frequency rows onto the
predicted SHG autoconvolution
({func}`~croak.marginal_checks.apply_marginal_correction`), is enabled only for
an SHG trace with an independent spectrum, and previews the snap onto the
prediction live (the correction itself is applied in Prepare). All three kinds
of session pass through this stage, except a raw simulated scan; see
[Marginal consistency checks](marginal_checks.md).

The marginals are drawn against frequency, the space the retrieval works in,
unless the [display](#display-spectral-axis-and-log-range) asks for wavelength.
The wavelength view applies the $1/\lambda^2$ density Jacobian to the predicted
marginal, which shifts a broad marginal towards the blue even when its
frequency centroid matches.

### Prepare

Fringe/DC/low-pass filtering, windowing and regridding
({func}`~croak.preprocess.load_and_clean`), recomputed automatically as you
change a setting. The plots show six maps in two rows of three — the measured,
filtered and regridded trace, in dB above and linear below — with linked axes:
all six share the delay axis, and the measured and filtered maps share the
wavelength axis, so a zoom on one follows on the others.

The controls always show the settings that were actually applied: re-entering
the stage re-seeds every widget from the current parameters, and those
parameters survive a round trip back to the loader as long as the trace is
unchanged (see [Preprocessing](preprocessing.md#settings-across-a-re-load)).
**Reset defaults** restores the data-seeded defaults. With **Baseline (arPLS)**
on, the **Auto** button beside *τ exclude (fs)* sets the hold-out to twice the
loaded trace's delay-marginal FWHM. Use it on measured traces: without a
hold-out the baseline bows into the τ ≈ 0 peak and removes signal, worst on the
rows that constrain the fit least. The failure shows up as a thin stripe at one
wavelength in the cleaned trace (see
[Peak suck-up](preprocessing.md#peak-suck-up-and-the-hold-out)).

### Retrieve

Choose the algorithm (any of the ten, including the global
[`cma-es`](../explanation/algorithms.md#global-retrieval-cma-es)) and its
parameters and run it with **Retrieve** in the top bar or **Run** on the form.
Retrieval runs in a background thread, driven by the solver `callback`, so the
interface stays responsive. The plots show a live overview: the measured and
retrieved traces (log) above the retrieved pulse and its spectrum. The full pages
of panels are on [Analyse](#analyse).

The form's sections, in order, are the solver and its mode, the forward-model
sections (*Dispersive propagation*, *Geometrical smearing*), the advanced
sections and *Post-filter (applied to result)*. The advanced sections —
*Chromatic focal mixture (advanced)*, *Extra fit parameters*, *Regularisation*
and *Global search (cma-es)* — start folded, and whether each is open is
remembered between launches. Below them are **Reset** (the default retrieval
settings) and **Retarget to new dataset…** (see [Sessions](#sessions)). A run
starts and stops from the top bar's **Retrieve** and **Stop**, and **Save
session…** saves it.

By default **Live full-plot preview** redraws the overview as the run converges.
Previews arrive at most four times a second, and less often on a machine where
presenting one is slow: the GUI measures what each preview costs and stretches
the interval so that presenting stays below about a quarter of the wall-clock
time, so the preview never starves the solver. The convergence curve beside the
numbers follows the run, redrawn at up to ten frames a second however fast the
solver iterates. Later snapshots update the existing curves and
images in place, so nothing shifts between frames. Pressing **Stop** halts the
run at that point and leaves the latest result on screen as a (non-converged)
result you can inspect, save or post-filter. Untick **Live full-plot preview** to
leave the overview until the run finishes, with only the convergence curve
moving. Every solver supports the preview, including
`copra`/`copra-jax` and the COPRA warm-up of `warm-lbfgs`, which is 300
iterations by default. `warm-lbfgs` runs **both** stages on one convergence
curve; the convergence panel on Analyse's Diagnostics page marks the handover
with a dashed rule. Read the two sides as different units of work (COPRA
iterations, then L-BFGS function evaluations).

The **Initial guess** selector makes automatic, transform-limited, random,
perfect-Gaussian, known **truth**, and reuse-previous starts mutually exclusive.
Truth uses the complex field stored by ModelPNPS (and reports an explicit error
for an older intensity-only file). **Reuse previous result** warm-starts one run
from the last — handy for the global-then-local refine recipe. It seeds the
previous **spectrum** as the initial guess *and* the extras that run fitted
(thickness, τ₀, smearing width) as the starting values for fitting them again,
naming them on the ready line; an extra you are no longer fitting keeps the value
set in the controls. When the run finishes, the ready line reports the fitted
extras alongside the error and FWHMs — including the geometric-smearing width as
a delay σ in fs with its multiplier on the nominal mask geometry (which is left
untouched). When the trace came from a synthetic or simulated session, the
**known-truth** temporal intensity is drawn as a thin black line beneath the
retrieved and transform-limited pulses. When its complex field is available,
dotted truth-phase lines appear in both time and spectrum panels and are named
in their legends. The truth arrays are also written into the saved `result.h5`.

Beneath the plots, the run's **numbers** sit beside its **convergence**, the
error against iteration on a log scale (the whole history once the run ends,
with the hand-overs between a solver's stages marked). The first row tracks the
run and its pulse: `Iteration`, `Error R` and `Elapsed`, then the retrieved
spectrum's peak wavelength `λ peak` and the pulse's `FWHM`. The second row shows
the forward model's extra parameters, `τ₀`, `Thickness` and `Smearing`, each
named **fit** or **fixed** so it is clear whether the retriever was free to move
it (a smearing width also carries its factor on the nominal kernel). The run
counters tick every iteration; the peak wavelength and FWHM refresh with the live
full-plot preview, so with that preview turned off they stay blank until the run
finishes. A number that is not defined shows an en dash. The transform-limited
and known FWHMs, and the peak powers once a pulse energy is set, are in the
retrieved pulse's legend.

The last three, `ε(Iₜ)`, `ε(I𝜔)` and `ε(E𝜔)`, are filled when the trace came from a
synthetic or simulated session and so has a **known** pulse: the temporal
intensity, spectral intensity and complex-field errors against it
({mod}`croak.truth_metrics`). They answer a different question from the
`Error R` above them — that scores the fit to the *trace*, these score the
distance to the true *pulse*, and a FROG trace does not determine the pulse
uniquely, so a small `R` can sit on a wrong answer. Read them together: a small
`ε(I𝜔)` with a large `ε(E𝜔)` means the amplitude is right and the phase is not.
The two spectral ones need the truth's complex field, so an intensity-only file
shows `ε(Iₜ)` alone rather than a number derived from something else.

*Dispersive propagation* models a slab (material, thickness, Δz/Npoints — see
[The forward model](../explanation/forward_model.md)). *Geometrical smearing*
models the tilted-pulse-front temporal blur of a non-collinear BOXCARS setup
from the mask hole diameter and edge-to-edge spacing, with the resulting delay
width shown — and editable, which back-computes the spacing. Both sections grey
out for solvers that cannot use them; smearing needs one of the autodiff
solvers. The chromatic focal and collection settings are described in
[collection aperture modelling](collection_aperture.md). See also
[geometric smearing](geometric_smearing.md) for the reduced model.

### Analyse

Analyse reads the retrieved result. A rail on the left lists its pages; click
one, or step through them with the Up and Down arrow keys:

- **Overview** — the live overview from Retrieve, without the form beside it;
- **Traces** — the measured and retrieved trace, log and linear;
- **Pulse** — the retrieved pulse over the delay marginal, and the spectrum
  over the frequency marginal;
- **Diagnostics** — convergence, residuals, spectral filter and spectrogram;
- **Dispersion** and **Uncertainty**, described below.

Overview, Traces, Pulse and Diagnostics are the panels of
{func}`~croak.plotting.plot_retrieval`, two by two so that each has room even on
a laptop display. A page is redrawn when it is shown, so the spectrogram, the
costliest panel, is computed only when you open Diagnostics. Each has
**Export ▾** with **Figure…** and **Pop out** (see [Plots](#plots)). The primary
action on these pages, and on Dispersion, is **Save**: it saves the session.

#### Dispersion

Apply and auto-tune residual dispersion ({mod}`croak.dispersion`) to compress
the retrieved pulse, including real glass (built-in or
[custom](../howto/materials_and_mirrors.md) materials), gases and a **mirror
stack** — add rows of `(mirror, polarisation, bounces, remove/add)` to
back-propagate (remove) beam-path coating mirrors or add chirped-mirror bounces —
with the temporal profile and spectrogram updating as you drag the sliders.
Enter a **pulse energy** (J) (shared with the measured loader) to switch the
temporal plot to absolute power and read off the current and transform-limited
**peak power**; compressing the pulse raises the peak power towards the
transform-limited maximum at fixed energy. The page's own **Save…** writes the
dispersed result as `result_dispersion.h5`, with `options.toml` and
`dispersion.pdf`.

#### Uncertainty

Put an error bar on the retrieved FWHM ({mod}`croak.uncertainty`). Choose the
estimator (parametric or resampling bootstrap, the substrate-thickness
systematic, or the fast analytic covariance), set the replicate count and
confidence level, and run it with **Estimate** in the top bar. It runs on a
worker thread; the ready line counts the replicates, and **Stop** in the top bar
cancels the run. The page reports the
point estimate with its 68 % and 95 % intervals, draws the temporal confidence
band over the retrieved profile, and its **Save…** appends the estimates to
`result.h5` and writes `options.toml` and `uncertainty.pdf`. See
[Estimating FWHM uncertainty](uncertainty.md).

## Sessions

A session is the settings of every stage — the loader, preparation, retrieval,
dispersion and uncertainty choices — kept as a TOML file
({func}`croak.save.save_options` / {func}`croak.save.load_options`), with the
retrieval in HDF5 ({func}`croak.save.save_result`).

**Save session…** (⋯ menu, File menu, or the primary action on Analyse) asks for
a folder and writes three files into it: `result.h5` (the retrieval, with any
uncertainty estimates appended), `options.toml` (the settings) and
`retrieval.pdf` (the twelve-panel overview, drawn by matplotlib). The session is
added to Recent sessions and its folder's name appears in the top bar. Saving
needs a retrieval: before one, the GUI says so.

**Open a session…** (Home, ⋯ menu or File menu) or a click on a recent session
reopens a saved `options.toml` and replays it:

1. The trace is reloaded with the saved loader settings and prepared with the
   saved preparation settings.
2. If a `result.h5` sits beside the `options.toml`, its retrieval is
   **restored from the file** ({func}`croak.save.result_from_saved`) without
   re-running the solver — but only when it was fitted to exactly the trace the
   replay produced. The saved measured trace and frequency axis are compared
   element by element ({func}`croak.session.files.result_mismatch`).
3. Otherwise the retrieval is re-run, and the ready line on Retrieve says why
   (no saved result, an unreadable file, or a trace that differs).

A synthetic session cannot be replayed this way: the generator's settings are not
part of the saved options, so the GUI loads the settings and asks you to
regenerate the trace. If the trace cannot be reloaded (a file has moved, say),
the loader says why. See [Saving and loading](saving_and_loading.md).

**Retarget to new dataset…** on the Retrieve form, available once a retrieval
has run, points a measured session at another dataset folder with the same file names, keeps the calibration files, and
replays load → prepare → retrieve with the current settings
({func}`croak.session.retarget.retarget`). Only a measured session has such a
folder; for the others the status bar gives the reason instead.

**Show the settings (options.toml)** and **Show as a Python script** open a
drawer on the right of the window with the session's settings as the
`options.toml` it would save, or as a standalone script that replays it without
the GUI ({func}`croak.scripting.generate_script`). **Copy** puts the text on the
clipboard; **Close** hides the drawer.

## Scripting the same workflow

A saved session runs without the GUI: `croak replay options.toml` replays it
headlessly, and `croak script options.toml` writes the same Python script that
**Show as a Python script** shows (see
[saving, sessions, and reproducible workflows](../user_guide/workflows.md)).

Everything in the GUI maps to a few library calls — load → clean → retrieve →
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
