# Post-processing and plotting results

A {class}`~croak.result.RetrievalResult` already exposes the retrieved field,
spectrum and their intensities/phases. {mod}`croak.processing` goes further: it
turns a result into a display-ready {class}`~croak.processing.ProcessedResult` with
oversampled profiles, fitted dispersion coefficients, marginals and residuals —
everything the figures need.

```python
pr = croak.process_result(result, measured=td.trace, Iomega_meas=td.Iomega)
pr.fwhm_retr, pr.fwhm_tl       # retrieved and transform-limited FWHM (s)
pr.gdd_fs2, pr.tod_fs3, pr.fod_fs4   # fitted dispersion (fs², fs³, fs⁴)
```

The spectral counterparts of `fwhm_retr` are two small helpers rather than fields,
since they apply to any intensity-vs-wavelength curve — the retrieved spectrum, the
measured one, or a known truth's:

```python
from croak.processing import peak_wavelength, spectral_fwhm

peak_wavelength(pr.wavelength, pr.Ilam)   # m
spectral_fwhm(pr.wavelength, pr.Ilam)     # m
```

Pass the **wavelength density** (`Ilam`, what the spectrum panel draws) rather
than the frequency density `Iw`: the peak of one is not the peak of the other. Both
helpers accept an ascending or descending axis, drop the unphysical bins where the
grid's `2πc/(ω+ω₀)` axis crosses zero, and return `nan` when the quantity is
undefined (an empty axis, or a spectrum whose half maximum the grid never reaches).
`spectral_fwhm` takes an optional `level=` to measure a width at another fraction of
the peak.

## Absolute power and peak power

A retrieval fixes the pulse *shape* but not its absolute scale, so by default the
temporal intensity is normalised to a peak of 1 ("a.u."). If you measured the
**pulse energy** `E` (e.g. with a power meter), pass it as `energy=` (in joules)
and `process_result` turns the normalised intensity into absolute instantaneous
power. With the intensity normalised to unit peak, instantaneous power is
`P(t) = E · I(t) / ∫I(t)dt`, so the **peak power** is `E / ∫I dt` — the energy
divided by the effective duration:

```python
pr = croak.process_result(result, measured=td.trace, energy=100e-6)  # 100 µJ
pr.peak_power       # retrieved-pulse peak power (W)
pr.peak_power_tl    # transform-limited peak power (W) — same energy, shortest
                    # duration, so the maximum achievable by compression
```

The transform-limited reference carries the same energy in a shorter time, so its
peak power is higher. Energy is conserved under the dispersion-stage (phase-only)
compensation, so compressing the pulse raises its peak power towards `peak_power_tl`
at fixed `E`. With `energy` unset (the default), `peak_power`/`peak_power_tl` are
`None` and every plot stays in normalised units.

{func}`~croak.plotting.plot_temporal` (and {func}`~croak.plotting.plot_retrieval`
via its own `energy=` argument) automatically switch the temporal axis to absolute
power with an SI prefix — chosen by {func}`~croak.plotting.si_power` — when a peak
power is available.

A known ground truth scales the same way through
{meth}`~croak.processing.TruthPulse.peak_power`, which applies the same
`E / ∫(I/I_peak) dt` definition over the truth's own time axis:

```python
truth.peak_power(energy)      # W, or None when the energy is unknown
```

## What `process_result` computes

{func}`~croak.processing.process_result` fills a
{class}`~croak.processing.ProcessedResult` with:

- **Temporal** — peak-centred, oversampled intensity `It_retr` and phase `phi_t`
  (linear ramp removed), the transform-limited reference `It_tl`, and the FWHMs
  `fwhm_retr` / `fwhm_tl`. When a pulse `energy` is supplied, also the absolute
  `peak_power` / `peak_power_tl` (W) — see [below](#absolute-power-and-peak-power).
- **Spectral (wavelength domain)** — spectral intensity `Iw` (and photon-weighted
  `Ilam`), phase `phi_w` with a quartic fit `phi_w_fit`, and the dispersion
  coefficients `gdd_fs2` / `tod_fs3` / `fod_fs4` from that fit. If you pass
  `Iomega_meas`, the measured spectrum `Iw_meas` is included for comparison.
- **Trace** — the retrieved trace, the measured trace and their
  [marginals](../explanation/retrieval_theory.md) and weighted `residual`, plus
  the convergence history `errors`.
- **Quality** — `edge_energy`, the fraction of the retrieved pulse energy sitting
  in the unmeasured grid-edge bins (see [below](#is-the-spectrum-real)).

Helpers are also exposed individually: {func}`~croak.processing.marginals`,
{func}`~croak.processing.residuals`, {func}`~croak.processing.oversample`,
{func}`~croak.processing.spectrogram` and {func}`~croak.processing.shiftnorm`.

## Is the spectrum real

A low trace error does not by itself mean the retrieved *spectrum* is
trustworthy. The outermost {data}`~croak.preprocess.TAPER_COLLAR_BINS` (10)
frequency bins at each end of the grid carry no measurement — `regrid`'s
[edge taper](grids_and_filtering.md#the-edge-taper-and-why-the-band-must-be-generous)
has zeroed them — so the field there is nearly free, and because intensity cannot
go below zero it drifts upward. The result is spectral energy appearing at the
band edges that was never in the trace.

{func}`~croak.processing.edge_energy_fraction` measures it, and
{attr}`ProcessedResult.edge_energy <croak.processing.ProcessedResult.edge_energy>`
carries it. {func}`~croak.plot_retrieval` prints it in the spectrum panel, in red
above {data}`croak.plotting.EDGE_ENERGY_WARN` (1 %):

```python
pr = croak.process_result(result, measured=td.trace)
if pr.edge_energy > 0.01:
    print(f"{pr.edge_energy:.1%} of the pulse energy is off the measured band")
```

Rules of thumb: **under 0.1 %** is a sound retrieval; **a few percent** means part
of the reported spectrum — and therefore of the reported duration — is an
artefact. On one DUV dataset, 15.6 % edge energy inflated the reported FWHM by
14 % against the same retrieval with the edge bins removed.

The two causes, in order of importance:

1. `R_omega=True` used without `reg_spectrum` or `phase_only` — by far the biggest
   contributor, and it *lowers* the reported error while doing it. See
   [per-frequency scaling](regularisation.md#per-frequency-scaling).
2. A `lam_min`/`lam_max` band tight enough that the taper cuts live signal —
   `load_and_clean` warns about this at load time.

Note the metric counts bins, so it is only meaningful on a grid with room for two
non-overlapping collars (≥ 21 points); it returns `0.0` below that.

## Plotting

{mod}`croak.plotting` is built on Matplotlib without global `pyplot` state, so the
figures embed cleanly (the GUI uses the same functions). The headline composite is
the twelve-panel {func}`~croak.plotting.plot_retrieval`:

```python
fig = croak.plot_retrieval(result, measured=td.trace, Iomega_meas=td.Iomega,
                          lam_min=td.lam_min, lam_max=td.lam_max)
fig.savefig("retrieval.png", dpi=110)
```

Its annotations are the numbers you judge a retrieval by, so they are printed
finely enough to compare runs:

- the **retrieved-trace panel** carries the FROG error as a percentage to two
  decimals (`0.71%`), above the trace dimensions;
- the **spectrum panel** reports the fitted **GDD** (fs²) and **TOD** (fs³) to one
  decimal, matching the dashed polynomial phase fit drawn over the retrieved phase;
- the **temporal panel**'s legend gives the intensity FWHM of every curve —
  retrieved `R`, transform-limited `TL`, and (when a ground truth is passed) the
  `truth` overlay — so the three durations read side by side.

Pass `truth=` (a {class}`~croak.processing.TruthPulse`) to draw a known pulse
beneath the retrieved one. The synthetic and simulated GUI workflows do this
automatically, but it is an ordinary library feature — build one with
{meth}`TruthPulse.from_spectrum <croak.processing.TruthPulse.from_spectrum>`
whenever you know the answer:

```python
truth = croak.TruthPulse.from_spectrum(grid, ew, omega0)
fig = croak.plot_retrieval(result, measured=trace, truth=truth)
```

### SHG: settle the direction of time first

An SHG-FROG trace is *exactly* invariant under $E(t) \to E^*(-t)$, so a pulse and
its mirror image fit the measurement equally well and the solver returns whichever
branch it happened to land in. Overlay a truth without accounting for that and a
perfect retrieval can look like a failure — the two curves mirrored about $t=0$.

When the true pulse is known, {func}`~croak.processing.resolve_time_direction`
picks the matching branch. Apply it to the `RetrievalResult` *before* anything
derives from it, so the plot, any `ProcessedResult` and the reported dispersion
all describe the same pulse:

```python
result, flipped = croak.resolve_time_direction(result, truth)
pr = croak.process_result(result, measured=trace)
```

It compares **temporal intensity** profiles by their best normalised
cross-correlation over all lags, which keeps the decision independent of the
absolute phase and the time origin (both ambiguous too) and reliable on an
imperfect retrieval — only the *relative* ranking of the two branches has to be
right. It is a no-op for PG and SD, which fix the direction of time, and on a
near-symmetric pulse where the two branches score alike.

Reversal is applied by conjugating the spectrum, which is exact: the stored trace,
`mu` and `error` remain valid. Note that the **spectral phase changes sign**, so
the retrieved GDD and TOD only become comparable with the truth once the direction
is settled. The wizard does this automatically for synthetic and simulated
sessions, where a truth is available.

### Framing the axes

Two independent controls, because the panels use two different abscissae:

- **`lam_min` / `lam_max`** (m) bound the wavelength axes — the spectrum and
  spectrogram panels. On the measured-data path take them from the cleaned
  {class}`~croak.preprocess.TraceData`, as above.
- **`flim`** — `(f_min, f_max)` in Hz — bounds the frequency axis of the trace
  and residual panels.

They are separate because the trace sits at the *signal* frequency, which for SHG
is twice the pulse's. `flim` matters when the retrieval grid is much wider than
the signal, which is typical when retrieving directly on a synthesis grid rather
than one cropped by {func}`~croak.preprocess.load_and_clean`; without it the trace
is a thin stripe across a mostly empty axis:

```python
f0 = omega0 / (2 * np.pi)
fig = croak.plot_retrieval(result, measured=trace,
                           flim=(2 * f0 - 0.3e15, 2 * f0 + 0.3e15))  # SHG
```

### Panels and pages

The twelve panels are independent functions over one data bundle, so any subset
can be drawn. {func}`~croak.plotting.retrieval_plot_data` computes everything the
panels share once — the processed result, axis bounds, colour scales, and the
spectrogram on first use; {func}`~croak.plotting.plot_retrieval_page` lays a
{class}`~croak.plotting.PageSpec` of panel keys out as a figure; and each
`draw_<key>(ax, data)` draws one panel into an axis of your own:

```python
data = croak.plotting.retrieval_plot_data(result, measured=td.trace,
                                          Iomega_meas=td.Iomega,
                                          lam_min=td.lam_min, lam_max=td.lam_max)
fig = croak.plotting.plot_retrieval_page(data, croak.plotting.RETRIEVAL_PAGES["pulse"])

ax = Figure().add_subplot()
croak.plotting.draw_spectral(ax, data)   # one panel, in your own figure
```

{data}`~croak.plotting.RETRIEVAL_PANELS` maps the keys — `measured_log`,
`retrieved_log`, `measured_lin`, `retrieved_lin`, `temporal`, `spectral`,
`convergence`, `residual`, `spectral_filter`, `freq_marginal`, `delay_marginal`,
`spectrogram` — to their titles and draw functions;
{data}`~croak.plotting.RETRIEVAL_PAGES` holds the three 2×2 pages the wizard shows
as tabs (**Traces**, **Pulse**, **Diagnostics**), and
{data}`~croak.plotting.RETRIEVAL_OVERVIEW` the 3×4 layout `plot_retrieval` draws.
Colorbars shared by a pair of trace panels are drawn whenever both panels are on
the page. Every `draw_*` returns the artists it created (for example
{class}`~croak.plotting.TemporalArtists`), which is what a live view updates in
place instead of redrawing.

Other composites and single-axis plotters:

| Function | Shows |
|----------|-------|
| {func}`~croak.plotting.plot_retrieval` | 12-panel overview (trace, residual, time, spectrum, marginals, convergence) |
| {func}`~croak.plotting.plot_retrieval_page` | one page of retrieval panels (a `PageSpec`) |
| {func}`~croak.plotting.retrieval_plot_data` | the shared inputs of the retrieval panels |
| `draw_measured_log`, …, `draw_spectrogram` | one retrieval panel each (see `RETRIEVAL_PANELS`) |
| {func}`~croak.plotting.plot_frog_filter` | 6-panel preprocessing before/after |
| {func}`~croak.plotting.plot_simulated_trace` | retrieved vs measured trace |
| `plot_residual` | a single residual image |
| `plot_temporal`, `plot_spectral` | intensity + phase in one axis (with a truth overlay) |
| `plot_convergence` | trace error vs iteration |
| `plot_marginal`, `plot_spectrogram` | a marginal / Gabor spectrogram |

The single-axis plotters take an existing Matplotlib axis, so you can compose your
own figures.

## Cosmetic windowing

To trim ringing or out-of-band energy from the *result* before plotting (without
re-running the retrieval), apply {func}`croak.processing.post_filter` — a
reversible temporal/spectral Planck taper that returns a new result. Pass the
measurement windows `td.lamm_lims` / `td.tau_lims` (λm/τm) as `lam_lims` /
`tau_lims`, not the full grid extent. See
[Grids and filtering](grids_and_filtering.md).

## Residual dispersion

`process_result` reports the fitted GDD/TOD/FOD. To explore compressing the pulse
— applying extra dispersion and watching the peak power — see
[the dispersion-tuning tutorial](../tutorials/05_dispersion_tuning.md) and
{mod}`croak.dispersion`.

## Saving

Persist a result (and its processed form) to HDF5 with
{func}`croak.save.save_result`; see [Saving and loading](saving_and_loading.md).
