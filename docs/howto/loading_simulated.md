# Loading numerically simulated traces

Besides measured data and the built-in synthetic generator, croak can retrieve
from **numerically simulated** FROG scans — for example a TG-FROG trace produced
by a full propagation code (a Luna `scansave` HDF5 file). This lets you test the
retrieval against a known input pulse with realistic physics (phase matching,
dispersion in the nonlinear medium, a chromatic input mask, finite collection
optics) that the idealised forward model does not include.

A simulated scan flows through exactly the same preprocessing and retrieval as a
measurement: the loader produces the same `load_data` dictionary that
{func}`~croak.session.pipeline.assemble_load_data` does, so you keep all of croak's
filtering, windowing and regridding options.

Two reduced scans from the companion paper's validation study ship with the
repository in `examples/data/` — the 1 fs thickness series and the single-cycle
RDW pulse — so everything on this page can be tried immediately;
`examples/example_paper_thickness.py` and `examples/example_paper_rdw.py` are
worked retrievals of them (see [Validation](../explanation/validation.md)).

## The expected file layout

{func}`~croak.io.read_simulated_scan` reads a `scansave`-style HDF5 file:

| Path | Shape | Meaning |
|------|-------|---------|
| `grid/ω` | `(Nω,)` | absolute angular frequency (rad/s) |
| `grid/ω0` | scalar | carrier angular frequency (rad/s) |
| `grid/Iω` | `(Nω,)` | **source** spectrum `\|Eω\|²` (ideal input, *before* the mask) |
| `grid/Iω_beamlet` | `(Nω,)` | **beamlet** spectrum *after* the mask (the beam that gates) — optional |
| `grid/t`, `grid/It` | `(Nt,)` | **source** time axis and temporal intensity (`To`/`Ito` if oversampled) |
| `grid/It_beamlet` | `(Nt,)` | **beamlet** temporal intensity (post-mask gate beam; `Ito_beamlet` if oversampled) — optional |
| `grid/Iω_beamlet_reimaged` | `(Nω,)` | **on-axis beamlet** spectrum (pnps files; one power of ω bluer in amplitude than `Iω_beamlet`) — optional |
| `grid/It_beamlet_reimaged` | `(Nt,)` | **on-axis beamlet** temporal intensity (`Ito_beamlet_reimaged` if oversampled) — optional |
| `grid/τfwhm` | scalar | input-pulse intensity FWHM (s) |
| `grid/zsave` | `(nz,)` | saved propagation distances (m), entrance `0` → exit `zmax` — optional (multi-thickness format) |
| `scanvariables/τ` | `(Nτ,)` | delay axis (s) |
| `Iω_win` | `(Nω, nz, Nτ)` | signal energy integrated over the selected collection window |
| `Iω_win_reimaged` | `(Nω, nz, Nτ)` | signal at the **on-axis** pixel (fibre/slit-coupled) |
| `Iω_full` | `(Nω, nz, Nτ)` | full signal beam, **no aperture crop** (no mask vignetting) — optional (newer files) |

The trace window is stored as the simulation wrote it — Julia's column-major
`(Nω, nz, Nτ)` is read by h5py reversed as `(Nτ, nz, Nω)`. The reader selects the
propagation slice (`z_index`, default the last = substrate exit, or `z_thickness`;
see [Selecting a thickness](#selecting-a-thickness)), orients the result to croak's
`(Nω, Nτ)`, and sorts onto an ascending absolute-ω axis.

## What the datasets mean

The generator is a full spatio-spectral forward model: a beam (e.g. a hollow-fibre
HE₁₁ mode) is imaged through a focusing pair onto a multi-hole **mask**, giving
masked gate and test beamlets that cross in a thin χ³ slab and four-wave-mix; the
signal beam (a new k-direction) is extracted through a signal-hole window.

- **`grid/Iω` (source)** is the ideal input spectrum before the mask. **`grid/Iω_beamlet`
  (beamlet)** is the spectrum of one gate beam *after* the mask. The fixed mask
  hole maps to k-space as `x = kx·z·c/ω`, so it collects an ω-dependent slice of
  the beam — the beamlet is therefore **chromatically vignetted** relative to the
  source (it is blue-shifted, and is the spectrum that actually gates).
- **`Iω_win` versus `Iω_win_reimaged`** selects different observables: integrated
  transmitted energy versus an on-axis reimaged field intensity. Their ratio
  need not be a fixed power of wavelength for broadband, delay-dependent spatial
  structure. Choose the observable matching the detector and relay.
- **`Iω_full`** retains the isolated signal without the collection-aperture crop.
  Input-mask filtering and the radiated source response can still be present.
  The loader offers only windows actually stored in the selected file.

(selecting-a-thickness)=
## Selecting a thickness

A simulation can save the signal at **several propagation depths through the
nonlinear medium in one file** — the `nz` axis of each trace window — and records
those depths in `grid/zsave` (strictly increasing, from the entrance `0` to the
substrate exit `zmax`). A saved intermediate plane can represent a thinner slab when the incident field,
propagation boundary conditions and collection definition match that calculation.
Verify the simulation metadata before interpreting every saved plane as an
independent instrument with a different physical thickness.

Pick the thickness with either knob of {func}`~croak.io.read_simulated_scan` (and
the matching {class}`~croak.session.params.SimulatedLoadParams` fields):

- **`z_thickness`** (metres) selects the slice whose `zsave` position is nearest
  (`argmin|zsave − z_thickness|`). This is the user-facing choice; in
  `SimulatedLoadParams` it is `z_thickness_um` (micrometres).
- **`z_index`** is the raw slice index along `nz` (default `-1`, the substrate
  exit). `z_thickness` **takes precedence** when both are given.

```python
from croak import io

io.read_simulated_z_positions("scan_collected.h5")        # -> e.g. [0, 4, …, 40] µm, or None
scan = io.read_simulated_scan("scan_collected.h5", z_thickness=20e-6)   # the 20 µm slice
scan.z_index            # the resolved slice index that was loaded
scan.z_positions        # the zsave axis (m), or None for a legacy file
```

```{note}
**Backwards compatibility.** The older single-thickness format does not store
`grid/zsave`; there `z_positions` is `None`, only the substrate-exit slice is
meaningful, and the loader keeps its previous behaviour (`z_index = -1`). Passing
`z_thickness` to such a file raises, since there is no thickness axis to match.
```

The ratio `Iω_win / Iω_full`, where defined, measures collection for that known
simulated field and delay. Since it can depend on the field and delay, it is not
in general a transferable detector calibration. croak does not automatically
use this ratio as a correction.

## The ω → λ Jacobian the loader applies

The simulation stores an **angular-frequency density** $I_\omega = |\tilde
E_\mathrm{sig}(\omega)|^2$, which is also what croak's retrieval forward model
produces. A *spectrometer*, by contrast, records a **wavelength density**
$I_\lambda$, and croak's regrid multiplies the loaded trace by $\lambda^2$ to turn
$I_\lambda$ into $I_\omega$ (the Jacobian $|\mathrm{d}\omega/\mathrm{d}\lambda|
\propto 1/\lambda^2$). So a simulated trace must be **divided by $\lambda^2$**
before it enters the wavelength pipeline — then the regrid's $\times\lambda^2$
returns the original $I_\omega$ undistorted. This is done by
{func}`~croak.preprocess.omega_to_lambda_density`. Skipping it would leave an
irreducible $\lambda^2$-shaped residual and bias the retrieved spectrum.

{func}`~croak.session.pipeline.assemble_simulated_load_data` runs:

```
read scan -> λ = 2πc/ω -> drop ω ≤ 0 and crop to the band
   -> ×(λ/µm)^n third-order correction (optional)
   -> ÷ λ²  (omega_to_lambda_density)
   -> reverse delay axis (optional) -> normalise
```

with the chosen reference spectrum (beamlet or source) offered as the independent
spectrum.

(why-the-third-order-exponent-is-large-the-physics)=
## Source response, detector response and spectral correction

The optional correction multiplies intensity by $(\lambda/\mathrm{\mu m})^n$.
It is an empirical frequency-response approximation over a chosen band, not a
universal law of third-order generation. Three separate effects must be identified:

1. **Radiated source field.** In the paraxial reduction, the generated field has
   a factor proportional to $\omega_{\rm abs}/n(\omega_{\rm abs})$ multiplying
   the nonlinear polarisation. Its intensity factor is the square. Approximating
   this by $\omega^2$ neglects refractive-index dispersion.
2. **Detector observable.** Energy density, photon counts, integrated signal and
   on-axis intensity are different quantities. The wavelength/frequency Jacobian
   converts density coordinates; it does not supply a detector calibration.
3. **Chromatic input and collection.** The nonlinear polarisation mixes frequency
   triplets. Multiplying three single-frequency vignetting exponents does not
   give a general broadband response. Finite collection can also change delay
   structure, which no frequency-only correction removes.

### Correcting the ω-dependence — a single exponent is only approximate

Prefer a measured signal-channel response when it is available. Marginal-centroid
matching can estimate a useful empirical exponent, but that estimate depends on
the reference spectrum, interaction, field plane, band and model. It does not
separate detector response from missing physics.

`R_omega=True` fits a gain for each frequency row and can absorb an unknown
**separable** frequency response. It also weakens constraints on spectral
amplitude. Pair it with a justified measured-spectrum constraint or phase-only
retrieval, and inspect [edge energy](postprocessing.md) and residual structure.
Neither choice makes the result immune to delay-dependent collection effects.
See [regularisation](regularisation.md#per-frequency-scaling).

Use an auxiliary spectrum in the field convention of the model. A source,
integrated beamlet and on-axis entrance spectrum are not interchangeable.
Compare durations only after converting the **complex field** into a common
convention; a small centroid shift need not imply a negligible duration change.
See [field reference planes](../reference/conventions.md#field-reference-planes).

### The time-domain truth overlay

Separately from the *reference spectrum fed to the retrieval*, the loader records a
known **time-domain truth** pulse ({class}`~croak.processing.TruthPulse`) for the
retrieval-screen overlay and the saved output. `truth_source` chooses which stored
pulse that is — the post-mask **beamlet** (`grid/It_beamlet`, the beam that
actually gates, the default) or the ideal pre-mask **source** (`grid/It`) — and the
matching spectrum (`grid/Iω_beamlet` or `grid/Iω`) is used for its spectral panel
so the overlaid pulse is self-consistent. The oversampled grid (`To`/`Ito*`) is
used when present. A file that stores only one falls back to it automatically
(legacy and Gaussian-beam files store only `It`). Because the blind retrieval
reconstructs the **gating** beamlet, `"beamlet"` is the like-for-like truth to
compare against; pick `"source"` only to check against the ideal input.

pnps files add a third source, **`"beamlet_reimaged"`**: the *on-axis* beamlet
(`grid/It_beamlet_reimaged`, `grid/Iω_beamlet_reimaged`, `grid/Eω_beamlet_reimaged`).
Its amplitude is one power of ω bluer than the integrated beamlet's, and its
transform limit is slightly *longer* (1.053 vs 1.033 fs for the FROG production
pulse). It is the field the chromatic focal mixture's `ew` represents, so a
`focal=` retrieval should use it both as `spectrum_source` (then no
`spectrum_frame_p` reweighting is needed) and as the `truth_source` it is scored
and, with `truth_init`, seeded from. Files without it fall back to the beamlet.

Newer ModelPNPS files may also store the complex spectra as
`Eω_beamlet_re`/`Eω_beamlet_im` and `Eω_re`/`Eω_im`. croak converts the
selected field into its Fourier convention and retains it on the
{class}`~croak.processing.TruthPulse`. This enables dotted **truth phase** curves
(with legend entries) in both temporal and spectral panels, a **Truth (known
complex field)** initial guess in the GUI, and the same initial condition in
scripts.

A useful known-field check evaluates the forward model directly on the native
grid, with the matching reference plane and detector convention. This avoids
spectral interpolation and holds the field fixed; a retrieval with `maxiters=1`
can still update the field and is not a pure forward-model test. For example,
construct the matching options and evaluate `croak.maketrace` or the JAX trace
function on the stored complex spectrum, then apply the stated gain and residual
metric. [Validation](../explanation/validation.md#fixed-field-validation-and-numerical-convergence)
explains global-gain and row-gain comparisons.

Truth initialisation is separately available through
`retrieve_from_tracedata(td, guess="truth", truth=data["truth"])` or
`RetrieveParams(truth_init=True)`. It tests local inverse behaviour, not blind
recovery. Intensity-only truth cannot provide a complex initial field.

Even on the native grid, an independently generated three-dimensional trace
need not match a reduced model to rounding error. Residuals can include physical
model discrepancy, source/observable conventions and reference discretisation.
Only a trace generated by the same discrete map is expected to have a numerical
zero at its generating field.

## Scoring a retrieval against the truth

A trace error only says how well the model fits the **data**. When the pulse is
known, {mod}`croak.truth_metrics` scores the retrieved **pulse** directly, which
is a different question — a FROG trace does not determine the pulse uniquely, so
a small `R` can sit on a wrong answer. The GUI's numeric read-out under the plot
pane shows all three beside the FROG error whenever a known pulse is loaded:

| | Compares | Blind to |
|---|---|---|
| $\epsilon_{I_t}$ | temporal intensity $\|E(t)\|^2$ | energy scale, time origin |
| $\epsilon_{I_\omega}$ | spectral intensity $\|\tilde E(\omega)\|^2$ | scale, **all** phase |
| $\epsilon_{E_\omega}$ | the complex field $\tilde E(\omega)$ | scale, absolute phase, delay |

Each removes the declared alignment freedoms that the measurement cannot
determine, and no others. Read them together: $\epsilon_{I_\omega}$ small with
$\epsilon_{E_\omega}$ large means the amplitude is right and the phase is not —
the failure a trace error hides most easily. $\epsilon_{E_\omega}$ is Geib's
$\epsilon$, the metric the PNPS literature quotes. croak evaluates its overlap
through the equivalent least-squares projection residual, avoiding the
`sqrt(1 - x²)` cancellation near a perfect match; fields differing only by the
unobservable scale, absolute phase and delay therefore score zero to machine
precision.

```python
from croak.truth_metrics import truth_errors

errs = truth_errors(result, data["truth"])
print(errs.eps_It, errs.eps_Iw, errs.eps_Ew)
```

The two spectral errors need the truth's complex field, so they are reported as
`None` for an intensity-only file; $\epsilon_{I_t}$ needs only the envelope and
is always available.

The options live on {class}`~croak.session.params.SimulatedLoadParams`:

| Option | Default | Meaning |
|--------|---------|---------|
| `window_key` | `"Iω_win"` | `"Iω_win"` (integrated), `"Iω_win_reimaged"` (on-axis), or `"Iω_full"` (full beam, no aperture crop — newer files) |
| `z_thickness_um` | `None` | material thickness (µm) to load from a multi-thickness scan; `None` = exit slice |
| `z_index` | `-1` | raw propagation slice (substrate exit); used when `z_thickness_um` is `None` |
| `lam_min_nm`, `lam_max_nm` | `140`, `800` | wavelength band to **load** (crop); see below. The GUI seeds these to the file's full range |
| `interaction` | `"pg"` | TG-FROG ≈ PG ($E\,|E(t-\tau)|^2$); or `"sd"` |
| `reverse_trace` | `None` | delay-sign convention: `None` auto-detects from the file's `/grid/delay_convention` marker (gate-frame files load unreversed; legacy files are negated); explicit `True`/`False` overrides |
| `third_order` / `third_order_exp` | `True` / `2.0` | divide out the $\propto\omega^n$ efficiency (approximate) |
| `use_spectrum` | `True` | feed a known spectrum as the independent spectrum (off for a blind test) |
| `spectrum_source` | `"beamlet"` | which spectrum: `"beamlet"` (post-mask, recommended) or `"source"` |
| `truth_source` | `"beamlet"` | which stored pulse to overlay as the time-domain truth: `"beamlet"` (post-mask `grid/It_beamlet`, the beam that gates — recommended) or `"source"` (pre-mask `grid/It`). Falls back to the other when the file stores only one |
| `raw_direct` | `False` | skip the marginal-check + preprocess and load the as-simulated trace straight into the retrieval ([below](#raw-direct-to-retrieval)) |

```{note}
**What "load λ band" does.** `lam_min_nm`/`lam_max_nm` crop the *loaded* trace to
bins with λ in that range (and always drop the unphysical ω ≤ 0 bins). It sets the
**extent of the data** handed to the marginal-check and preprocess stages — it is
*not* the retrieval grid (that is the Preprocess stage's own λ range). The GUI
defaults it to the file's **full** positive-frequency span; narrow it only to crop
noisy or empty wings.
```

```{note}
TG-FROG keeps the carrier (`omega0_scale = 1`), so the trace and the pulse sit at
the same wavelengths and `"pg"`/`"sd"` are the only sensible interactions — never
`"shg"`. The default `third_order_exp = 2.0` approximates a radiated-field intensity factor; the marginal-check stage's **Auto** refines it against your spectrum (the
`Iω_win` + beamlet combination usually wants a little more, the `"source"`
spectrum more again). It is only an approximation — see the correction options
above (Rω / phase-only / calibration).
```

## From a script

The loader mirrors the measured-data path
({func}`~croak.session.pipeline.assemble_load_data`), so it composes with
{func}`~croak.session.pipeline.build_tracedata` and
{func}`~croak.session.pipeline.run_retrieval`:

```python
import numpy as np
from croak.session import (
    SimulatedLoadParams,
    PreprocParams,
    RetrieveParams,
    assemble_simulated_load_data,
    build_tracedata,
    run_retrieval,
)

data = assemble_simulated_load_data(
    SimulatedLoadParams(
        frog_path="scan_collected.h5",
        window_key="Iω_win",
        interaction="pg",
        third_order=True,
        third_order_exp=2.0,          # approximate; refine with the marginal-check Auto
        reverse_trace=None,           # auto: the /grid/delay_convention marker decides
        use_spectrum=True,
        spectrum_source="beamlet",    # the beam that actually gates
    )
)

# stage 2 onward is identical to a measurement: filter / window / regrid …
td = build_tracedata(
    PreprocParams(lam_min_nm=200, lam_max_nm=340, lamm_min_nm=210, lamm_max_nm=330),
    data,
)
res = run_retrieval(RetrieveParams(solver="copra", maxiters=200), td)
print(res.error)
```

To avoid choosing the exponent, drop the `third_order` correction and let the
retrieval absorb the response with the per-frequency Rω scaling (anchoring the
amplitude with a light spectral regularisation toward the gating spectrum):

```python
data = assemble_simulated_load_data(
    SimulatedLoadParams(frog_path="scan_collected.h5", third_order=False,
                        spectrum_source="beamlet")
)
td = build_tracedata(PreprocParams(lam_min_nm=200, lam_max_nm=340), data)
res = run_retrieval(
    RetrieveParams(solver="lbfgs", R_omega=True, reg_spectrum=0.3, maxiters=200), td
)
```

Need just the raw arrays? {func}`~croak.io.read_simulated_scan` returns a
{class}`~croak.io.SimulatedScan` with the oriented trace and both spectra
(`Iomega`, `Iomega_beamlet`) plus the chosen truth pulse `It`/`τfwhm` to compare
the retrieval against (`truth_source="beamlet"` by default; see
{func}`~croak.io.read_simulated_truth_keys` for what a file offers).

(raw-direct-to-retrieval)=
## Retrieving the raw trace directly (skip filtering & regrid)

To sanity-check the retrieval with **no** preprocessing in the way, load the
**as-simulated** trace on the simulation's **native FFT grid** straight into the
retrieval — bypassing the ÷λ² of {func}`~croak.session.pipeline.assemble_simulated_load_data`
*and* the regrid/filtering of {func}`~croak.session.pipeline.build_tracedata`.
{func}`~croak.session.pipeline.assemble_simulated_tracedata` returns a ready
{class}`~croak.preprocess.TraceData` built directly on `grid/ω` (the file's uniform
FFT grid): it is **fully raw** — no third-order correction, no resampling — and
only honours the window/thickness selection, the delay-sign convention and the
known-spectrum choice.

```python
from croak.session import (
    SimulatedLoadParams, RetrieveParams, assemble_simulated_tracedata, run_retrieval,
)

td = assemble_simulated_tracedata(
    SimulatedLoadParams(frog_path="scan_collected.h5", use_spectrum=True)
)
res = run_retrieval(RetrieveParams(solver="copra", maxiters=200), td)
print(res.error)
```

Because the trace is the ω-density on its own grid, there is no Jacobian, no
interpolation and no windowing between the file and the solver — which isolates the effect of preprocessing. A good fit here does not by itself
validate the physical model.
Compare it against the regular regrid path (same options, `raw_direct=False`) to
see what the filtering and regridding change. The retrieval accepts the
simulation's delays directly (the forward model gates-and-FFTs per delay), so no
delay resampling is needed.

## From the GUI

The welcome menu's **Load simulated trace** entry opens a loader page: choose the
`.h5` file, the trace window, the **thickness**, the corrections, the
**reference spectrum** and the **time-domain truth** source, then preview the trace
live and press the footer's **Load & continue →** button (bottom right, like every
other page) to hand it straight to the **Preprocess** stage. The
**time-domain truth** dropdown lists what the file stores (the beamlet first, so
`It_beamlet` is the default) and is disabled when the file offers only one. For a
multi-thickness file the **thickness**
dropdown lists every saved depth (largest/exit first, the default); changing it
re-previews the trace at that depth. For an older single-thickness file the dropdown
is disabled (only the exit slice exists). Every control has a hover tooltip, and the **? Help** button
summarises the options (including the exponent/reference guidance above). From the
filter stage on, the workflow is exactly as for a measured trace — so you can apply
croak's fringe/DC/threshold filters to the modelled data too. **Back** from
preprocess returns to the loader page to adjust the options.

The preview is shown in the **file-native representation**: the simulated trace
and its spectral marginal are an **angular-frequency density** plotted against
**frequency** (the loader's ÷λ² is undone for display), so the load page reads
identically to the marginal-check screen and the retrieval — the delay marginal
stays in fs. The **load λ band** seeds to the file's full range (see the note
above).

Tick **Skip filtering & regrid (raw → retrieval)** to load the as-simulated trace
on its native grid **straight into the Retrieve stage**, bypassing the
marginal-check and preprocess stages (see
[above](#raw-direct-to-retrieval)); the footer's forward button relabels to
**Load raw → Retrieve →** to say so. The known-truth overlay is still shown, and
**Back** from Retrieve reaches Preprocess so you can run the regrid path and
compare.

The Rω / phase-only alternatives to the third-order exponent live on the
**Retrieve** stage (the *Rω adaptive scaling* and *phase-only* toggles); turn the
loader's third-order correction off and let Rω absorb the response there instead.

## Saving and reloading a simulated session

Saving from the Retrieve stage writes the loader's options into the session file
as a `[simulated]` table, alongside `entry = "simulated"` (see
[Saving and loading](saving_and_loading.md)) — the scan file, trace window,
thickness slice, loaded band, third-order exponent, delay-sign convention and the
spectrum/truth choices. So the retrieval records *which* simulated trace it came
from and how that trace was built.

**Load previous retrieval** reads `entry` and reloads through this loader rather
than the measured one: it re-reads the scan with the saved options (including the
raw *skip filtering & regrid* path, which rebuilds the native-grid trace
directly), restores the loader page's controls, replays the saved preprocessing
and rehydrates the sibling `result.h5` without re-running the solver. The same
file replays headlessly:

```bash
uv run croak replay options.toml
```

```{note}
**Retargeting does not apply.** {func}`croak.session.retarget.retarget` re-points a session at
another *per-dataset folder*, a convention of measured acquisitions (trace,
spectrum and backgrounds sharing a folder with stable filenames). A simulated
session is one scansave file with no such siblings, so retargeting it raises — and
the Retrieve stage's **Retarget…** button reports that instead of silently
replaying the same scan.
```
