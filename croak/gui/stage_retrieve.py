"""Stage 3 — run the retrieval (threaded) and show the 12-panel result.

Post-retrieval λ/τ filtering can be toggled on the result pane and re-plots the
stored result without re-running the retrieval.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import materials, plotting, save, smearing
from ..constants import C
from ..processing import (
    ProcessedResult,
    TruthPulse,
    peak_wavelength,
    post_filter,
    process_result,
    resolve_time_direction,
    spectral_fwhm,
)
from ..retrieve import ALGORITHMS, algorithm_params
from ..session.params import reg_defaults, reg_family
from ..session.pipeline import extra_param_centres, smearing_kernel
from ..truth_metrics import truth_errors
from .base import Stage
from .canvas import MplCanvas, check, combo, dspin, group, spin, with_toolbar
from .readout import DASH, RetrievalReadout
from .widgets import SciSpinBox
from .worker import RetrievalWorker

#: How often the elapsed-time read-out ticks during a run (ms).
_ELAPSED_TICK_MS = 500

_INITIAL_GUESSES = {
    "Automatic (centroid phase)": None,
    "Transform-limited measured spectrum": "tl_init",
    "Random initial phase": "random_init",
    "Perfect guess (TL Gaussian)": "perfect_init",
    "Truth (known complex field)": "truth_init",
    "Reuse previous result": "reuse_result",
}


class StageRetrieve(Stage):
    HELP_TITLE = "Retrieve"
    HELP_INTRO = (
        "Run the phase-retrieval algorithm on the preprocessed trace. The result "
        "is shown in a 12-panel view. Post-retrieval λ/τ filtering re-plots the "
        "stored result without re-running the retrieval."
    )
    HELP = [
        (
            "Solver",
            [
                (
                    "Solver",
                    "Retrieval algorithm. COPRA is a fast, robust general-purpose "
                    "solver; LBFGS is analytic-gradient gradient descent; "
                    "LBFGS-HAND is its jitted JAX twin (same analytic gradient, "
                    "vmapped/JIT, no autodiff); LBFGS-AD is the JAX autodiff twin; "
                    "LM is Levenberg–Marquardt least-squares (strong near "
                    "convergence).",
                ),
                ("Max iters", "Maximum number of optimiser iterations."),
                (
                    "reltol 10^",
                    "Relative function tolerance as a power of ten (10^n): stop "
                    "when the FROG error improves by less than this fraction.",
                ),
                (
                    "abstol 10^",
                    "Absolute target/step tolerance as a power of ten; for COPRA "
                    "this enables early stopping.",
                ),
            ],
        ),
        (
            "Mode",
            [
                (
                    "Retrieve",
                    "Retrieve both amplitude and phase, or the phase only with the "
                    "spectral amplitude held fixed.",
                ),
                (
                    "Phase basis",
                    "Phase parameterisation (AD/global solvers): Pointwise (one "
                    "value per frequency) or a reduced cubic B-spline over the "
                    "spectrum support — far fewer parameters, ideal for the global "
                    "search. B-spline implies phase-only (amplitude held fixed).",
                ),
                (
                    "Phase nodes",
                    "Number of B-spline control points for the phase (typically "
                    "20–30); only used when the phase basis is B-spline.",
                ),
                (
                    "Rω support",
                    "Use a per-frequency error weighting (Rω) restricted to the "
                    "spectral support, rather than a single global error.",
                ),
                (
                    "Initial guess",
                    "Choose exactly one starting field. Truth uses the known "
                    "complex ModelPNPS/synthetic field and is available only "
                    "when the loaded truth contains a complex spectrum. Reuse "
                    "warm-starts from the previous compatible retrieval.",
                ),
                (
                    "Perfect FWHM (fs)",
                    "Intensity FWHM of the transform-limited Gaussian initial guess.",
                ),
                (
                    "Live full-plot preview",
                    "Redraw the full 12-panel view (retrieved trace, pulse, "
                    "spectrum, marginals…) as the retrieval converges, throttled "
                    "to about once a second, instead of only the convergence "
                    "curve. Useful for long LM runs. Pressing Stop keeps the "
                    "latest full plot on screen. Turn off for the cheaper "
                    "convergence-only curve.",
                ),
            ],
        ),
        (
            "Dispersive propagation",
            [
                (
                    "Enable dispersive propagation",
                    "Model propagation through a dispersive medium during "
                    "retrieval, for thick/DUV media where the thin-medium "
                    "approximation fails.",
                ),
                ("Material", "Dispersive material modelled in the forward map."),
                (
                    "Thickness (µm)",
                    "Medium thickness used by the dispersive forward model. Editing "
                    "it holds Δz fixed and grows Npoints to keep the node density "
                    "constant.",
                ),
                (
                    "Δz step (µm)",
                    "Nominal depth step used to pick Npoints: Npoints = round("
                    "thickness / Δz). Tied to the thickness and Npoints — editing "
                    "Δz recomputes Npoints (thickness held). The actual nodes are "
                    "non-uniform Gauss–Legendre abscissae, so Δz is nominal.",
                ),
                (
                    "Npoints",
                    "Number of quadrature slabs integrated through the medium "
                    "thickness (more = more accurate, slower). Tied to Δz: editing "
                    "Npoints back-computes the displayed Δz step.",
                ),
            ],
        ),
        (
            "Geometrical smearing",
            [
                (
                    "Enable geometrical smearing",
                    "Model the temporal smearing of a non-collinear BOXCARS "
                    "geometry: the tilted pulse fronts make the relative arrival "
                    "time between arms vary across the focal spot, and the detector "
                    "averages over it. Thickness-independent, so it is separable "
                    "from dispersion. PG (TG) / SD only.\n\n"
                    "Available on the autodiff solvers, which model it "
                    "throughout, and on COPRA-jax, which can only apply it in "
                    "its GLOBAL stage: the local projection replaces one "
                    "signal's modulus per delay, and a smeared trace is an "
                    "incoherent mixture with no single such signal. COPRA "
                    "therefore runs a hybrid — unsmeared projection, smeared "
                    "descent — which on the Gaussian control is the most "
                    "robust option tried across transform-limited and both "
                    "chirp signs, but is not the identical algorithm to "
                    "unsmeared COPRA.",
                ),
                (
                    "Hole diameter (mm)",
                    "Diameter of the boxcar mask holes. It sets the focal spot "
                    "size, so a larger hole means a smaller spot and less smearing.",
                ),
                (
                    "Hole spacing (mm)",
                    "Edge-to-edge gap between adjacent mask holes. Together with "
                    "the diameter this fixes the mask ratio d/D — the only physical "
                    "lever on the effect. Widening the mask increases the smearing "
                    "roughly linearly.",
                ),
                (
                    "σ delay (fs)",
                    "RMS spread of the delay offset the kernel applies, derived "
                    "from the geometry at the measured carrier wavelength. Tied to "
                    "the two fields above: editing it back-computes the hole "
                    "spacing (holding the diameter), so it can be dialled in "
                    "directly. It has a floor at zero spacing.",
                ),
                (
                    "Quadrature nodes",
                    "Gauss–Hermite nodes across the gate-splitting parameter (the "
                    "delay integral is exact and needs none). The cost is about "
                    "(nodes + 1) / 2 times the unsmeared model; 5 is ample.",
                ),
            ],
        ),
        (
            "Extra fit parameters",
            [
                (
                    "Fit medium thickness",
                    "Also fit the dispersive-slab thickness as a retrieval "
                    "parameter (lbfgs-ad / lm / lm-optx / cma-es). Needs "
                    "dispersive propagation enabled, and is only identifiable for "
                    "PG/SD (for SHG a slab is a pure spectral phase the retrieved "
                    "phase absorbs). Starts from the Thickness value above.",
                ),
                (
                    "Fit τ₀ delay offset",
                    "Also fit a delay-zero offset, correcting a mis-set "
                    "experimental time zero (works for every interaction). Removes "
                    "the characteristic red/blue residual tilt of a timing error.",
                ),
                (
                    "Two-phase polish",
                    "Retrieve the pulse first with the extra parameters held "
                    "fixed, then free them for a joint final phase — robust when "
                    "starting from a good prior.",
                ),
            ],
        ),
        (
            "Regularisation",
            [
                (
                    "λ amp",
                    "Weight of the spectral-amplitude smoothness penalty "
                    "(suppresses noisy amplitude).",
                ),
                (
                    "λ phase",
                    "Weight of the spectral-phase smoothness penalty (suppresses "
                    "noisy phase).",
                ),
                (
                    "λ spectrum",
                    "Pull the retrieved spectrum towards the measured independent "
                    "spectrum (full mode; needs a loaded spectrum).",
                ),
                (
                    "λ time",
                    "Penalise pulse energy falling outside the time window below "
                    "(suppresses low-lying temporal pedestal). A self-consistent "
                    "alternative to the temporal post-filter — applied during "
                    "retrieval, so it introduces no spectral artefacts. Works in "
                    "every mode (LBFGS-AD / cma-es).",
                ),
                (
                    "τ window lo (fs)",
                    "Lower edge of the time window kept by the λ time penalty; "
                    "energy outside it is penalised. Typically the measurement τm "
                    "range.",
                ),
                (
                    "τ window hi (fs)",
                    "Upper edge of the time window kept by the λ time penalty; "
                    "energy outside it is penalised. Typically the measurement τm "
                    "range.",
                ),
            ],
        ),
        (
            "Global search (cma-es)",
            [
                (
                    "Strategy",
                    "Evolution strategy for the cma-es retriever: CMA-ES (best for "
                    "the low-dimensional B-spline basis), Separable CMA-ES or "
                    "Differential Evolution (scale better to full dimension).",
                ),
                (
                    "Population (0=auto)",
                    "Population size per generation (0 uses the evosax default ≈ "
                    "4+⌊3 ln N⌋). The whole population is evaluated in one device "
                    "call.",
                ),
                (
                    "Init σ",
                    "Initial search spread (CMA mean std / DE population scatter).",
                ),
            ],
        ),
        (
            "Post-filter (applied to result)",
            [
                (
                    "Spectral (λ window)",
                    "Mask the retrieved result outside the measurement λ window — "
                    "applied to the stored result, no re-run.",
                ),
                (
                    "Temporal (τ window)",
                    "Mask the retrieved result outside the delay window — applied "
                    "to the stored result, no re-run.",
                ),
            ],
        ),
    ]

    def __init__(self, state):
        super().__init__(state)
        p = state.retrieve
        self._worker = None
        self._base_result = None  # unfiltered retrieval, for post-filter toggles
        self._elapsed = None  # wall-clock time of the last retrieval (s)
        # Whether a live full-plot preview has taken over the canvas this run; until
        # the first preview arrives we draw the cheap convergence curve.
        self._live_full = False
        # Last regularisation weights used with each solver family, so switching
        # solver to compare and back does not throw away hand-tuned values. The
        # families are kept apart because the same number does not mean the same
        # thing to them; see croak.session.params.REG_DEFAULTS.
        self._reg_by_family = {reg_family(p.solver): (p.reg_spectrum, p.reg_amp)}
        # Live elapsed clock for the read-out. The worker keeps its own t0 (and its
        # value is the authoritative one reported on completion); this one only has
        # to tick while the run is in flight, so the few-ms offset is immaterial.
        self._t0 = 0.0
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(_ELAPSED_TICK_MS)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        box, form = group("Solver")
        # Single source of truth: the registry of every retrieval algorithm.
        self.solver_combo = combo(
            sorted(ALGORITHMS),
            p.solver,
            self._on_solver_changed,
        )
        self.solver_combo.setToolTip(
            "Retrieval algorithm. warm-lbfgs is the default: L-BFGS on the full "
            "extended model, started from COPRA's local projection sweep rather "
            "than from the raw initial guess. It costs one extra COPRA run and "
            "makes the result much less sensitive to that guess -- on chirped "
            "traces the spread in retrieved duration over starting points is "
            "five to twelve times tighter. Pick lbfgs-ad for the same model "
            "without the warm-up, or copra-jax for COPRA alone."
        )
        form.addRow("Solver", self.solver_combo)
        self.maxiters_spin = spin(
            1, 10000, p.maxiters, lambda v: setattr(p, "maxiters", v)
        )
        form.addRow("Max iters", self.maxiters_spin)
        # Convergence tolerances, entered as base-10 exponents (reltol = 10**n).
        # reltol is the optimiser relative function tolerance; abstol the
        # absolute target / step tolerance (for COPRA they enable early stopping).
        self.reltol_spin = spin(
            -12, 0, round(math.log10(p.reltol)), lambda v: setattr(p, "reltol", 10.0**v)
        )
        form.addRow("reltol 10^", self.reltol_spin)
        self.abstol_spin = spin(
            -14, 0, round(math.log10(p.abstol)), lambda v: setattr(p, "abstol", 10.0**v)
        )
        form.addRow("abstol 10^", self.abstol_spin)
        self.controls.addWidget(box)

        self.mode_box, form = group("Mode")
        box = self.mode_box
        self.mode_combo = combo(
            ["Amp. + phase", "Phase only"],
            "Amp. + phase" if p.full else "Phase only",
            lambda v: setattr(p, "full", v == "Amp. + phase"),
        )
        form.addRow("Retrieve", self.mode_combo)
        self.basis_combo = combo(
            ["Pointwise", "B-spline"],
            "B-spline" if p.phase_basis == "bspline" else "Pointwise",
            self._on_basis_changed,
        )
        form.addRow("Phase basis", self.basis_combo)
        self.nodes_spin = spin(4, 200, p.n_nodes, lambda v: setattr(p, "n_nodes", v))
        form.addRow("Phase nodes", self.nodes_spin)
        self.romega_check = check(
            "Rω support", p.R_omega, lambda v: setattr(p, "R_omega", v)
        )
        form.addRow(self.romega_check)
        self.guess_combo = combo(
            list(_INITIAL_GUESSES),
            self._initial_guess_label(),
            self._on_initial_guess_changed,
        )
        form.addRow("Initial guess", self.guess_combo)
        self.perfect_fwhm_spin = dspin(
            1,
            1e5,
            p.perfect_init_fs,
            lambda v: setattr(p, "perfect_init_fs", v),
            decimals=0,
        )
        form.addRow("Perfect FWHM (fs)", self.perfect_fwhm_spin)
        self._on_initial_guess_changed(self.guess_combo.currentText())
        #: COPRA local-only. Its two stages behave very differently: the local
        #: sweep is a per-delay projection, the global stage a gradient
        #: descent. Never handing over to the global stage is better on five of
        #: six test cases, and the resulting spectrum is a good seed for a
        #: gradient solver -- run COPRA local-only, then re-run with a gradient
        #: solver and "Reuse previous result" as the initial guess.
        self.copra_local_check = check(
            "COPRA: local projection only",
            p.copra_stall_patience > 1000,
            lambda v: setattr(p, "copra_stall_patience", 10**9 if v else 5),
        )
        self.copra_local_check.setToolTip(
            "Never hand over from COPRA's local projection sweep to its global "
            "gradient descent.\n\nThe local sweep replaces each delay's signal "
            "modulus with the measured one, injecting far more measured "
            "information per iteration than a gradient step. Staying in it is "
            "better on five of six test cases (20-35% in complex error).\n\n"
            "Also the recommended way to warm start: run this, then switch to "
            "L-BFGS-AD with 'Reuse previous result' selected. On chirped pulses "
            "that cuts the start-to-start spread in retrieved duration by five "
            "to twelve times, and improves accuracy on the harder cases."
        )
        form.addRow(self.copra_local_check)
        self.live_check = check(
            "Live full-plot preview",
            p.live_preview,
            lambda v: setattr(p, "live_preview", v),
        )
        form.addRow(self.live_check)
        self.controls.addWidget(box)

        self.disp_box, form = group("Dispersive propagation")
        box = self.disp_box
        self.dispersive_check = check(
            "Enable dispersive propagation",
            p.dispersive,
            self._on_dispersive_changed,
        )
        form.addRow(self.dispersive_check)
        self.material_combo = combo(
            list(materials.MATERIALS),
            p.material,
            lambda v: setattr(p, "material", v),
        )
        form.addRow("Material", self.material_combo)
        # Thickness, depth step Δz and Npoints are tied: npoints = thickness/Δz.
        # Editing the thickness or Δz grows npoints to hold the step fixed;
        # editing npoints back-computes Δz. Only npoints reaches the forward model.
        self.thickness_spin = dspin(
            0, 1000, p.thickness_um, self._on_thickness_changed, decimals=1
        )
        form.addRow("Thickness (µm)", self.thickness_spin)
        self.dz_spin = dspin(
            0.001, 1000, p.delta_z_um, self._on_dz_changed, decimals=3, step=0.1
        )
        form.addRow("Δz step (µm)", self.dz_spin)
        self.npoints_spin = spin(1, 10000, p.npoints, self._on_npoints_changed)
        form.addRow("Npoints", self.npoints_spin)
        self.controls.addWidget(box)

        self.smear_box, form = group("Geometrical smearing")
        box = self.smear_box
        self.smearing_check = check(
            "Enable geometrical smearing",
            p.smearing,
            self._on_smearing_changed,
        )
        form.addRow(self.smearing_check)
        # Hole diameter, edge-to-edge spacing and the delay-offset width sigma are
        # tied, exactly as thickness/Δz/Npoints above: sigma is proportional to the
        # mask ratio d/D with d = (spacing + D)/2, so editing either geometry field
        # updates sigma and editing sigma back-computes the spacing. Only the
        # geometry is stored as truth and reaches the forward model.
        self.hole_diameter_spin = dspin(
            0.01,
            100,
            p.smear_hole_diameter_mm,
            self._on_hole_diameter_changed,
            decimals=3,
            step=0.1,
        )
        self.hole_diameter_spin.setToolTip("Diameter D of one mask aperture (mm).")
        form.addRow("Hole diameter (mm)", self.hole_diameter_spin)
        self.hole_spacing_spin = dspin(
            0.0,
            100,
            p.smear_hole_spacing_mm,
            self._on_hole_spacing_changed,
            decimals=3,
            step=0.1,
        )
        self.hole_spacing_spin.setToolTip(
            "EDGE-TO-EDGE gap between adjacent apertures (mm), not their "
            "centre-to-centre distance. The kernel widths scale with the beam "
            "offset d = (spacing + D)/2, so equal spacing and diameter give "
            "d/D = 1 and a centre-to-centre distance of 2d."
        )
        form.addRow("Hole spacing (mm)", self.hole_spacing_spin)
        # Four decimals: sub-femtosecond kernels are the interesting regime, and a
        # finer display keeps the sigma -> spacing round-trip tight.
        self.smear_sigma_spin = dspin(
            0.0,
            1000,
            p.smear_sigma_delay_fs,
            self._on_smear_sigma_changed,
            decimals=4,
            step=0.01,
        )
        form.addRow("σ delay (fs)", self.smear_sigma_spin)
        self.smear_layout_combo = combo(
            ["boxcars", "sd2"],
            p.smear_layout,
            lambda v: setattr(p, "smear_layout", str(v)),
        )
        self.smear_layout_combo.setToolTip(
            "Aperture layout the kernel is built from.\n\n"
            "boxcars — the four-hole BOXCARS mask (TG/PG, and a three-hole "
            "SD in which the two unconjugated arms come from separate holes).\n\n"
            "sd2 — a genuine TWO-beam self-diffraction: two collinear "
            "apertures, the probe entering the nonlinearity twice and the gate "
            "once with conjugation. Here the two unconjugated arms are the SAME "
            "beam, so the gate-shape parameter p — proportional to the "
            "difference of their tilts — is identically zero and the smearing "
            "becomes a pure convolution along the delay axis, with sigma_theta "
            "= 0.49135 (d/D) lambda/c.\n\n"
            "Choosing the wrong one is not cosmetic: on a two-beam SD "
            "measurement the boxcars form attaches a p channel of order 0.4 fs "
            "that the experiment does not have."
        )
        form.addRow("Aperture layout", self.smear_layout_combo)
        # Fixed width multiplier. Separate from "Fit kernel width" below, which
        # optimises the same quantity: this one is applied and held.
        self.smear_scale_spin = dspin(
            0.0,
            5.0,
            p.smear_scale,
            lambda v: setattr(p, "smear_scale", float(v)),
            decimals=3,
            step=0.05,
        )
        self.smear_scale_spin.setToolTip(
            "Multiplier on the kernel widths from the hole geometry. 1.0 is "
            "the parameter-free geometric value and is correct for the "
            "apertured instrument: scanning the width against a 3D simulation "
            "at the known input pulse puts the optimum at 0.85-1.00, at both "
            "9.5 and 20 um substrates and at both d/D = 1.0 and 1.5.\n\n"
            "Lower it only with reason. Gaussian-beam traces (no aperture "
            "diffraction) want about 0.5, because the three-beam overlap "
            "concentrates the generation more than the single-beam focal "
            "weight alone predicts; at 1.0 the kernel over-broadens the model "
            "and the retrieval compensates with a pulse SHORTER than the "
            "truth. This knob exists because the geometry alone cannot express "
            "it: d/D = (spacing + D)/2D is at least 0.5 for any physical gap."
        )
        form.addRow("Kernel scale", self.smear_scale_spin)
        self.smear_nodes_spin = spin(
            1, 99, p.smear_nodes, lambda v: setattr(p, "smear_nodes", int(v))
        )
        form.addRow("Quadrature nodes", self.smear_nodes_spin)
        self.controls.addWidget(box)

        self.focal_box, form = group("Chromatic focal mixture (advanced)")
        box = self.focal_box
        self.focal_check = check(
            "Enable focal mixture",
            p.focal,
            self._on_focal_changed,
        )
        self.focal_check.setToolTip(
            "Replace the reduced smearing kernel by the explicit chromatic "
            "focal-plane mixture (croak.focal): each focal node carries the "
            "chromatic Airy filter A(r, ω) and its own definite (p, θ). With "
            "a collection model this is the full instrument model needed "
            "for quantitative single-cycle GDD retrieval. Uses the mask "
            "geometry above; mutually exclusive with the reduced kernel; "
            "autodiff solvers only; ~40× the cost per iteration."
        )
        form.addRow(self.focal_check)
        self.focal_f_spin = dspin(
            1.0,
            10000,
            p.focal_f_mm,
            lambda v: setattr(p, "focal_f_mm", float(v)),
            decimals=1,
            step=10.0,
        )
        self.focal_f_spin.setToolTip(
            "Focusing focal length (mm). The reduced kernel never needs it "
            "(it cancels between spot size and tilt); the explicit mixture "
            "does. Read from the scan file's geometry for simulated data "
            "written after 2026-08-26."
        )
        form.addRow("Focal length (mm)", self.focal_f_spin)
        self.collection_combo = combo(
            ["off", "file", "manual"],
            p.collection,
            self._on_collection_changed,
        )
        self.collection_combo.setToolTip(
            "Collection model riding on the mixture.\n\n"
            "off — incoherent sum over the focal plane: exact only when the "
            "whole signal beam is collected (Parseval). A tightly apertured "
            "instrument sits far from that limit and the mixture over-blurs "
            "without a collection model.\n\n"
            "file — rebuild the collection aperture from a simulated scan's "
            "own window_def record (exact geometry, no typing).\n\n"
            "manual — a hard-edged hole of the given diameter at the "
            "phase-matched BOXCARS corner (−d, −d), d = (gap + D)/2; for "
            "experimental data, where the aperture is a physical pinhole "
            "set by hand."
        )
        form.addRow("Collection", self.collection_combo)
        self.collection_mode_combo = combo(
            ["integrated", "reimaged"],
            p.collection_mode,
            lambda v: setattr(p, "collection_mode", str(v)),
        )
        self.collection_mode_combo.setToolTip(
            "integrated — a spectrometer collecting all light through the "
            "hole (|field|² integrated over the aperture).\n"
            "reimaged — the on-axis re-imaged pixel, the fully coherent limit."
        )
        form.addRow("Collection mode", self.collection_mode_combo)
        self.collection_diam_spin = dspin(
            0.01,
            100,
            p.collection_diam_mm,
            lambda v: setattr(p, "collection_diam_mm", float(v)),
            decimals=3,
            step=0.1,
        )
        form.addRow("Hole diameter (mm)", self.collection_diam_spin)
        self.frame_p_spin = dspin(
            -2.0,
            4.0,
            p.spectrum_frame_p,
            lambda v: setattr(p, "spectrum_frame_p", float(v)),
            decimals=2,
            step=0.25,
        )
        self.frame_p_spin.setToolTip(
            "Spectral-frame exponent p: the measured spectrum (initial guess "
            "and reg-spectrum target) is reweighted by (ω/ω₀)^{2p} in "
            "intensity. The mixture's field is the ON-AXIS one, bluer than "
            "an integrated spectrum by ω² for an aperture-diffraction "
            "beamlet; the collection-effective exponent is smaller — "
            "measured ≈1.0 at a 0.5 mm hole with gap 0.5 mm, ≈0.5 at gap "
            "1.0 mm, →0 as the hole opens. Leave 0 for the "
            "reduced kernel. GDD is insensitive to it; durations move by a "
            "few %."
        )
        form.addRow("Spectrum frame p", self.frame_p_spin)
        # Auto frame: p = p_model − p_source, where the mixture's field is the
        # on-axis one (p_model = 1), a 1-D model's the collection-weighted one
        # (p_model ≈ 0.5 at the production geometry, App D), and the loaded
        # reference spectrum is either transverse-integrated (p_source = 0) or,
        # for pnps files, the on-axis beamlet (p_source = 1).
        self.frame_auto_check = check(
            "Auto frame from model + spectrum source",
            False,
            lambda v: self._apply_frame_auto(),
        )
        self.frame_auto_check.setToolTip(
            "Set p automatically: chromatic mixture → on-axis frame (p = 1 for an "
            "integrated reference spectrum, 0 for a 'beamlet_reimaged' one); "
            "1-D kernels → the collection-weighted frame (p ≈ 0.5 at the "
            "production geometry; geometry-dependent, see App. D)."
        )
        form.addRow(self.frame_auto_check)
        self.focal_check.toggled.connect(lambda _v: self._apply_frame_auto())
        self.collection_diam_spin.setEnabled(p.collection == "manual")
        self.controls.addWidget(box)

        self.extra_box, form = group("Extra fit parameters")
        box = self.extra_box
        self.fit_thickness_check = check(
            "Fit medium thickness",
            p.fit_thickness,
            lambda v: setattr(p, "fit_thickness", v),
        )
        form.addRow(self.fit_thickness_check)
        self.fit_tau0_check = check(
            "Fit τ₀ delay offset",
            p.fit_tau0,
            lambda v: setattr(p, "fit_tau0", v),
        )
        form.addRow(self.fit_tau0_check)
        self.fit_smearing_check = check(
            "Fit smearing width",
            p.fit_smearing,
            lambda v: setattr(p, "fit_smearing", v),
        )
        form.addRow(self.fit_smearing_check)
        self.polish_check = check(
            "Two-phase polish",
            p.polish_two_phase,
            lambda v: setattr(p, "polish_two_phase", v),
        )
        form.addRow(self.polish_check)
        self.tau0_bound_spin = dspin(
            0.0,
            5000.0,
            p.tau0_bound_fs * 1e3,
            lambda v: setattr(p, "tau0_bound_fs", float(v) * 1e-3),
            decimals=0,
            step=10.0,
        )
        self.tau0_bound_spin.setToolTip(
            "Box bound on the fitted τ₀, |τ₀| ≤ bound (attoseconds); 0 = free. "
            "A free τ₀ on a chirped pulse wanders by hundreds of as and trades "
            "against the chirp; the physical collection offset is tens of as."
        )
        form.addRow("τ₀ bound (as)", self.tau0_bound_spin)
        self.delay_origin_combo = combo(
            ["coincidence", "marginal_peak"],
            p.delay_origin,
            lambda v: setattr(p, "delay_origin", str(v)),
        )
        self.delay_origin_combo.setToolTip(
            "Model delay zero. 'coincidence': gate–probe coincidence (native). "
            "'marginal_peak': every model trace is re-centred on its own "
            "delay-marginal peak, exactly as the preprocessing centres the data — "
            "one convention on both sides, no free parameter. Use for coherently "
            "collected single-cycle traces (their peak sits tens of as from "
            "coincidence); it replaces 'Fit τ₀' and stays put on chirped pulses."
        )
        form.addRow("Delay origin", self.delay_origin_combo)
        self.controls.addWidget(box)

        self.reg_box, form = group("Regularisation")
        box = self.reg_box
        # Scientific-notation spin boxes (decade-aware stepping) so weights well
        # below 0.001 — common with LM — can be entered, e.g. 5e-4.
        self.reg_amp_spin = SciSpinBox(
            0.0, 1.0, p.reg_amp, lambda v: setattr(p, "reg_amp", v), sigfigs=2
        )
        form.addRow("λ amp", self.reg_amp_spin)
        self.reg_phase_spin = SciSpinBox(
            0.0, 1.0, p.reg_phase, lambda v: setattr(p, "reg_phase", v), sigfigs=2
        )
        form.addRow("λ phase", self.reg_phase_spin)
        self.reg_spectrum_spin = SciSpinBox(
            0.0, 1.0, p.reg_spectrum, lambda v: setattr(p, "reg_spectrum", v), sigfigs=2
        )
        form.addRow("λ spectrum", self.reg_spectrum_spin)
        self.reg_time_spin = SciSpinBox(
            0.0, 1.0, p.reg_time, lambda v: setattr(p, "reg_time", v), sigfigs=2
        )
        form.addRow("λ time", self.reg_time_spin)
        self.reg_time_lo_spin = dspin(
            -1e5,
            1e5,
            p.reg_time_lo_fs,
            lambda v: setattr(p, "reg_time_lo_fs", v),
            decimals=0,
            step=10,
        )
        form.addRow("τ window lo (fs)", self.reg_time_lo_spin)
        self.reg_time_hi_spin = dspin(
            -1e5,
            1e5,
            p.reg_time_hi_fs,
            lambda v: setattr(p, "reg_time_hi_fs", v),
            decimals=0,
            step=10,
        )
        form.addRow("τ window hi (fs)", self.reg_time_hi_spin)
        self.controls.addWidget(box)

        self.global_box, form = group("Global search (cma-es)")
        box = self.global_box
        self.strategy_combo = combo(
            ["cma", "sep-cma", "de"],
            p.cma_strategy,
            lambda v: setattr(p, "cma_strategy", v),
        )
        form.addRow("Strategy", self.strategy_combo)
        self.popsize_spin = spin(
            0, 100000, p.cma_popsize, lambda v: setattr(p, "cma_popsize", v)
        )
        form.addRow("Population (0=auto)", self.popsize_spin)
        self.std_init_spin = dspin(
            0.01,
            100.0,
            p.cma_std_init,
            lambda v: setattr(p, "cma_std_init", v),
            decimals=2,
            step=0.1,
        )
        form.addRow("Init σ", self.std_init_spin)
        self.controls.addWidget(box)

        # post-retrieval cleanup (applied to the stored result, no re-run)
        box, form = group("Post-filter (applied to result)")
        self.pf_lam = check(
            "Spectral (λ window)", p.post_filter_lam, self._on_post_filter
        )
        self.pf_tau = check(
            "Temporal (τ window)", p.post_filter_tau, self._on_post_filter
        )
        self.pf_lam.setEnabled(False)
        self.pf_tau.setEnabled(False)
        form.addRow(self.pf_lam)
        form.addRow(self.pf_tau)
        self.controls.addWidget(box)

        btns = QWidget()
        bl = QHBoxLayout(btns)
        bl.setContentsMargins(0, 0, 0, 0)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self._stop)
        self.stop_btn.setEnabled(False)
        self.save_btn = QPushButton("Save…")
        self.save_btn.clicked.connect(self._save)
        self.save_btn.setEnabled(False)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setToolTip("Restore the default retrieval settings.")
        self.reset_btn.clicked.connect(self._reset_defaults)
        bl.addWidget(self.run_btn)
        bl.addWidget(self.stop_btn)
        bl.addWidget(self.save_btn)
        bl.addWidget(self.reset_btn)
        self.controls.addWidget(btns)

        # Retarget: rerun this exact session on a different dataset folder (same
        # file names, calibration kept fixed). On its own row so the label fits.
        self.retarget_btn = QPushButton("Retarget to new dataset…")
        self.retarget_btn.setToolTip(
            "Point this session at a different dataset folder (same file names "
            "and calibration) and rerun load → preprocess → retrieve."
        )
        self.retarget_btn.clicked.connect(self._retarget)
        self.retarget_btn.setEnabled(False)
        self.controls.addWidget(self.retarget_btn)
        self.controls.addWidget(self.status_label)
        self.controls.addStretch(1)

        # Figure on top (it takes every spare pixel), numeric read-out beneath it
        # at its natural height — see croak.gui.readout.
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        self.canvas = MplCanvas(figsize=(12, 7))
        rv.addWidget(with_toolbar(self.canvas), stretch=1)
        self.readout = RetrievalReadout()
        rv.addWidget(self.readout, stretch=0)
        self.set_plot_area(right)
        self._update_extras_readout(None)

        self.apply_help()
        # Remember the Retrieve combo's help tooltip so the B-spline lock can
        # swap in an explanation and restore it afterwards.
        self._mode_tip = self.mode_combo.toolTip()
        self._update_solver_caps()

    def _on_solver_changed(self, value):
        p = self.state.retrieve
        was = reg_family(p.solver)
        p.solver = value
        now = reg_family(value)
        if now != was:
            # The families minimise different objectives (R + lambda*P against
            # R^2 + lambda*P), so a weight carried across is off by about 1/2R
            # — three orders of magnitude at a good trace error, which shows up
            # as an LM run that is visibly over-regularised for no stated
            # reason. Remember what was in use, restore this family's own.
            self._reg_by_family[was] = (p.reg_spectrum, p.reg_amp)
            p.reg_spectrum, p.reg_amp = self._reg_by_family.get(
                now, reg_defaults(value)
            )
            for w, v in (
                (self.reg_spectrum_spin, p.reg_spectrum),
                (self.reg_amp_spin, p.reg_amp),
            ):
                w.blockSignals(True)
                w.setValue(v)
                w.blockSignals(False)
            self.set_status(
                f"Regularisation rescaled for the {now} objective: "
                f"λ spectrum {p.reg_spectrum:.3g}, λ amp {p.reg_amp:.3g}."
            )
        self._update_solver_caps()

    def _on_basis_changed(self, text):
        self.state.retrieve.phase_basis = (
            "bspline" if text == "B-spline" else "pointwise"
        )
        self._update_solver_caps()

    def _initial_guess_label(self) -> str:
        """Return the one GUI label represented by the stored guess flags."""
        p = self.state.retrieve
        enabled = [
            label
            for label, attr in _INITIAL_GUESSES.items()
            if attr is not None and bool(getattr(p, attr))
        ]
        # Old session files could persist several independent checkboxes. Keep
        # the first legacy choice visible; selecting any combo entry immediately
        # rewrites the state into the new mutually-exclusive representation.
        return enabled[0] if enabled else next(iter(_INITIAL_GUESSES))

    def _on_initial_guess_changed(self, label: str) -> None:
        """Store one initial condition and clear every competing condition."""
        selected = _INITIAL_GUESSES[label]
        p = self.state.retrieve
        for attr in (
            "tl_init",
            "random_init",
            "perfect_init",
            "truth_init",
            "reuse_result",
        ):
            setattr(p, attr, attr == selected)
        self.perfect_fwhm_spin.setEnabled(selected == "perfect_init")

    def _on_dispersive_changed(self, value):
        # Thickness fitting is only available with dispersive propagation, so
        # refresh the capability gating when this toggles.
        self.state.retrieve.dispersive = bool(value)
        self._update_solver_caps()

    # -- coupled thickness / Δz / Npoints -----------------------------------
    # The slab integral uses ``npoints`` quadrature nodes over the thickness; a
    # constant depth step Δz means ``npoints = round(thickness / Δz)``. Holding Δz
    # while the thickness changes therefore grows npoints. Each handler updates
    # the stored param and the *derived* widget signal-safely (no feedback loop).
    def _set_npoints(self, npoints: int) -> None:
        """Store and display ``npoints`` (clamped to the spin range), signal-safe."""
        npoints = max(1, min(self.npoints_spin.maximum(), int(npoints)))
        self.state.retrieve.npoints = npoints
        self.npoints_spin.blockSignals(True)
        self.npoints_spin.setValue(npoints)
        self.npoints_spin.blockSignals(False)

    def _set_delta_z(self, delta_z_um: float) -> None:
        """Store and display Δz (clamped to the spin range), signal-safe."""
        delta_z_um = max(
            self.dz_spin.minimum(), min(self.dz_spin.maximum(), delta_z_um)
        )
        self.state.retrieve.delta_z_um = delta_z_um
        self.dz_spin.blockSignals(True)
        self.dz_spin.setValue(delta_z_um)
        self.dz_spin.blockSignals(False)

    def _on_thickness_changed(self, value: float) -> None:
        # New thickness, Δz held fixed -> grow npoints to keep the step constant.
        self.state.retrieve.thickness_um = value
        dz = self.state.retrieve.delta_z_um
        if value > 0 and dz > 0:
            self._set_npoints(round(value / dz))

    def _on_dz_changed(self, value: float) -> None:
        # New step, thickness held fixed -> recompute npoints.
        self.state.retrieve.delta_z_um = value
        thickness = self.state.retrieve.thickness_um
        if value > 0 and thickness > 0:
            self._set_npoints(round(thickness / value))

    def _on_npoints_changed(self, value: int) -> None:
        # Explicit npoints override, thickness held fixed -> back-compute Δz so the
        # displayed step matches the actual node spacing.
        self.state.retrieve.npoints = int(value)
        thickness = self.state.retrieve.thickness_um
        if value > 0 and thickness > 0:
            self._set_delta_z(thickness / value)

    def _on_smearing_changed(self, value):
        # Fitting the smearing width is only available once the kernel is modelled.
        self.state.retrieve.smearing = bool(value)
        # The reduced kernel and the explicit focal mixture are two models of
        # the same physics — never both. Enabling one disarms the other.
        if bool(value) and self.state.retrieve.focal:
            self.focal_check.setChecked(False)
        self._update_solver_caps()

    def _on_focal_changed(self, value):
        self.state.retrieve.focal = bool(value)
        if bool(value) and self.state.retrieve.smearing:
            self.smearing_check.setChecked(False)
        self._update_solver_caps()

    def _on_collection_changed(self, value):
        self.state.retrieve.collection = str(value)
        # The manual hole diameter only matters in manual mode.
        self.collection_diam_spin.setEnabled(str(value) == "manual")

    def _scan_path(self) -> str | None:
        """Loaded scan file, for a collection model rebuilt from its record.

        Simulated sessions carry their file on ``state.simulated`` (the
        experimental loader's ``state.load.frog_path`` stays empty), so fall
        through to it — otherwise ``collection="file"`` cannot resolve.
        """
        path = (
            getattr(self.state.load, "frog_path", "")
            or getattr(self.state.simulated, "frog_path", "")
            or None
        )
        return path

    def _apply_frame_auto(self) -> None:
        """Set the spectral-frame exponent from the model and the spectrum source.

        ``p = p_model - p_source``: the mixture's field is on-axis (1), a 1-D
        kernel's the collection-weighted effective one (≈0.5 at the production
        geometry); an integrated reference spectrum sits at 0, the on-axis
        ``beamlet_reimaged`` one at 1. Only acts while the auto box is checked;
        the spin is greyed out then so the rule is visible.
        """
        auto = self.frame_auto_check.isChecked()
        self.frame_p_spin.setEnabled(not auto)
        if not auto:
            return
        p = self.state.retrieve
        source = getattr(self.state.simulated, "spectrum_source", "beamlet")
        p_source = 1.0 if source == "beamlet_reimaged" else 0.0
        p_model = 1.0 if p.focal else 0.5
        p.spectrum_frame_p = p_model - p_source
        self.frame_p_spin.setValue(p.spectrum_frame_p)

    def _scan_window(self) -> str | None:
        """Which collection hole of a multi-window simulated scan is loaded.

        Forwarded with the scan path so ``collection="file"`` rebuilds the
        aperture of the window actually retrieved (``Iω_win_5`` is a 2.0 mm
        hole on the campaign files, ``Iω_win`` the 0.5 mm one); ``None`` for
        experimental sessions.
        """
        if getattr(self.state.load, "frog_path", ""):
            return None
        return getattr(self.state.simulated, "window_key", None) or None

    # -- coupled hole diameter / spacing / σ delay ---------------------------
    # The delay-offset width of a square BOXCARS mask is proportional to the mask
    # ratio d/D with d = (spacing + D)/2 (see :mod:`croak.smearing`), so the three
    # controls are one degree of freedom viewed three ways. The geometry is the
    # stored truth; σ is a derived readout that can also be edited to drive the
    # spacing, mirroring the Δz ↔ Npoints coupling above.
    def _smear_interaction(self) -> str:
        """Interaction the kernel is built for; PG unless the data says otherwise."""
        td = self.state.tracedata
        return getattr(td, "interaction", "pg") if td is not None else "pg"

    def _smear_wavelength(self) -> float:
        """Carrier wavelength (m) for the kernel: the measurement's when loaded."""
        td = self.state.tracedata
        if td is not None and getattr(td, "omega0_pulse", 0.0):
            return 2.0 * math.pi * C / td.omega0_pulse
        return 260e-9  # deep-UV default, matching the reference instrument

    def _refresh_smear_sigma(self) -> None:
        """Recompute and display σ from the current geometry, signal-safe."""
        p = self.state.retrieve
        try:
            sigma_fs = (
                smearing.square_boxcars_delay_width(
                    self._smear_interaction(),
                    hole_diameter=p.smear_hole_diameter_mm * 1e-3,
                    hole_spacing=p.smear_hole_spacing_mm * 1e-3,
                    wavelength=self._smear_wavelength(),
                )
                * 1e15
            )
        except ValueError:
            # SHG has no three-arm form; leave the displayed width alone.
            return
        p.smear_sigma_delay_fs = sigma_fs
        self.smear_sigma_spin.blockSignals(True)
        self.smear_sigma_spin.setValue(sigma_fs)
        self.smear_sigma_spin.blockSignals(False)

    def _on_hole_diameter_changed(self, value: float) -> None:
        self.state.retrieve.smear_hole_diameter_mm = value
        self._refresh_smear_sigma()

    def _on_hole_spacing_changed(self, value: float) -> None:
        self.state.retrieve.smear_hole_spacing_mm = value
        self._refresh_smear_sigma()

    def _on_smear_sigma_changed(self, value: float) -> None:
        # Explicit width override -> back-compute the edge-to-edge spacing, holding
        # the hole diameter fixed. Widths below the geometric floor (holes touching)
        # are unreachable, so the spacing clamps at zero and σ is re-displayed.
        p = self.state.retrieve
        p.smear_sigma_delay_fs = value
        try:
            spacing = smearing.square_boxcars_spacing(
                self._smear_interaction(),
                hole_diameter=p.smear_hole_diameter_mm * 1e-3,
                sigma_delta=value * 1e-15,
                wavelength=self._smear_wavelength(),
            )
        except ValueError:
            return
        spacing_mm = max(
            self.hole_spacing_spin.minimum(),
            min(self.hole_spacing_spin.maximum(), spacing * 1e3),
        )
        p.smear_hole_spacing_mm = spacing_mm
        self.hole_spacing_spin.blockSignals(True)
        self.hole_spacing_spin.setValue(spacing_mm)
        self.hole_spacing_spin.blockSignals(False)
        self._refresh_smear_sigma()

    # -- geometry from the simulated file -----------------------------------
    def apply_simulated_geometry(self, geom) -> str:
        """Adopt a simulated scan's recorded geometry; return what changed.

        Called when a simulated scan is loaded interactively (not on session
        restore, where the saved settings win). It sets the slab thickness to
        the propagation distance of the selected slice, and the smearing mask
        to the one the simulation actually used.

        These are the two settings a user cannot check by looking at the trace,
        and getting them wrong is quiet: a kernel built from the wrong hole
        spacing cost ~2% of retrieved duration in one sweep while *improving*
        the trace error, and a thickness that does not match the slice mis-fits
        the dispersion in the same undetectable way. Where the file does not
        record the mask (anything written before the simulator stored it), the
        returned message says so rather than leaving the impression that the
        displayed numbers came from the file.
        """
        if geom is None:
            return ""
        p = self.state.retrieve
        parts = []
        if geom.thickness_um is not None and geom.thickness_um > 0:
            if abs(geom.thickness_um - p.thickness_um) > 1e-6:
                self._on_thickness_changed(geom.thickness_um)  # also sets npoints
                parts.append(f"thickness {geom.thickness_um:.3g} µm")
        if geom.has_mask:
            changed = (
                abs(geom.mask_diameter_mm - p.smear_hole_diameter_mm) > 1e-9
                or abs(geom.mask_spacing_mm - p.smear_hole_spacing_mm) > 1e-9
            )
            p.smear_hole_diameter_mm = geom.mask_diameter_mm
            p.smear_hole_spacing_mm = geom.mask_spacing_mm
            if changed:
                parts.append(
                    f"mask D = {geom.mask_diameter_mm:.3g} mm, "
                    f"gap {geom.mask_spacing_mm:.3g} mm"
                )
        if geom.layout is not None and geom.layout != p.smear_layout:
            p.smear_layout = geom.layout
            parts.append(f"kernel layout {geom.layout}")
        if geom.f_foc_mm is not None and geom.f_foc_mm > 0:
            if abs(geom.f_foc_mm - p.focal_f_mm) > 1e-6:
                parts.append(f"focal length {geom.f_foc_mm:.4g} mm")
            p.focal_f_mm = geom.f_foc_mm
        self._seed_widgets()
        self._refresh_smear_sigma()
        if not geom.has_mask:
            note = (
                "the file does not record the mask geometry — check the hole "
                "diameter and spacing by hand"
            )
            return (
                f"From the scan: {', '.join(parts)}; {note}."
                if parts
                else (f"Note: {note}.")
            )
        return f"From the scan: {', '.join(parts)}." if parts else ""

    def _update_solver_caps(self):
        """Grey out option groups the selected algorithm cannot use.

        Capabilities come from the solver's constructor signature (see
        :func:`croak.retrieve.algorithm_params`) — e.g. COPRA/COPRA-jax take no
        regularisation, and only the AD/global solvers take a phase basis or the
        global-search settings.
        """
        try:
            params = algorithm_params(self.state.retrieve.solver)
        except ValueError:
            return
        self.reg_box.setEnabled(
            bool({"reg_amp", "reg_phase", "reg_spectrum", "reg_time"} & params)
        )
        # The temporal penalty is only on the build_parameterisation solvers
        # (LBFGS-AD / cma-es); grey just those controls where unsupported.
        has_time = "reg_time" in params
        self.reg_time_spin.setEnabled(has_time)
        self.reg_time_lo_spin.setEnabled(has_time)
        self.reg_time_hi_spin.setEnabled(has_time)
        self.disp_box.setEnabled("material" in params)
        # Geometrical smearing needs the incoherent quadrature sum. COPRA's LOCAL
        # projection cannot represent it — there is no single signal whose
        # modulus the measurement constrains — but its GLOBAL stage is a descent
        # and can, so copra-jax accepts the kernel and applies it there only
        # (the NumPy copra and the hand-written adjoints do not take it at all).
        # Capability comes from the constructor signature, so this follows
        # automatically.
        self.smear_box.setEnabled("smearing" in params)
        # The explicit chromatic focal mixture rides the same capability
        # mechanism: only solvers whose constructors take `focal` can model it.
        self.focal_box.setEnabled("focal" in params)
        # COPRA-only: the local/global split does not exist for the others.
        self.copra_local_check.setEnabled("stall_patience" in params)
        # Extra fitted parameters: the AD/LM solvers (lbfgs-ad, lm, lm-optx) and
        # the global cma-es accept them. Thickness
        # fitting additionally needs dispersive propagation enabled (and is PG/SD
        # only — the solver guards SHG at run time). The smearing width likewise
        # needs a kernel to scale.
        has_fit_thickness = "fit_thickness" in params
        has_fit_tau0 = "fit_tau0" in params
        has_fit_smearing = "fit_smearing" in params
        self.extra_box.setEnabled(has_fit_thickness or has_fit_tau0 or has_fit_smearing)
        self.fit_thickness_check.setEnabled(
            has_fit_thickness and self.state.retrieve.dispersive
        )
        self.fit_tau0_check.setEnabled(has_fit_tau0)
        self.fit_smearing_check.setEnabled(
            has_fit_smearing and self.state.retrieve.smearing
        )
        # Two-phase polish is a local-solver option; the global cma-es frees the
        # extras from the start, so it has no `polish` argument.
        self.polish_check.setEnabled(
            "polish" in params
            and (has_fit_thickness or has_fit_tau0 or has_fit_smearing)
        )
        self.tau0_bound_spin.setEnabled("tau0_bound" in params and has_fit_tau0)
        self.delay_origin_combo.setEnabled("delay_origin" in params)
        self.romega_check.setEnabled("R_omega" in params)
        # Reduced phase basis (AD + global solvers); nodes only matter for B-spline.
        has_basis = "phase_basis" in params
        self.basis_combo.setEnabled(has_basis)
        is_spline = has_basis and self.state.retrieve.phase_basis == "bspline"
        self.nodes_spin.setEnabled(is_spline)
        # The B-spline basis is inherently phase-only (amplitude held fixed), so
        # lock the Retrieve control to "Phase only" while it is active — without
        # touching the stored amp+phase choice, which is restored on switch-back.
        self.mode_combo.blockSignals(True)
        if is_spline:
            self.mode_combo.setCurrentText("Phase only")
            self.mode_combo.setEnabled(False)
            self.mode_combo.setToolTip(
                "Locked to phase-only: the B-spline basis parameterises the phase "
                "and holds the amplitude fixed. Switch Phase basis to Pointwise to "
                "retrieve the amplitude too."
            )
        else:
            self.mode_combo.setCurrentText(
                "Amp. + phase" if self.state.retrieve.full else "Phase only"
            )
            self.mode_combo.setEnabled("phase_only" in params)
            self.mode_combo.setToolTip(getattr(self, "_mode_tip", ""))
        self.mode_combo.blockSignals(False)
        # Global-search settings only apply to the evolution-strategy retriever.
        self.global_box.setEnabled("popsize" in params)

    def _seed_widgets(self):
        """Push every retrieval parameter into its widget (signal-safe)."""
        p = self.state.retrieve
        combos = {
            self.solver_combo: p.solver,
            self.mode_combo: "Amp. + phase" if p.full else "Phase only",
            self.basis_combo: "B-spline" if p.phase_basis == "bspline" else "Pointwise",
            self.material_combo: p.material,
            self.strategy_combo: p.cma_strategy,
            self.smear_layout_combo: p.smear_layout,
            self.collection_combo: p.collection,
            self.collection_mode_combo: p.collection_mode,
            self.delay_origin_combo: p.delay_origin,
            self.guess_combo: self._initial_guess_label(),
        }
        for w, val in combos.items():
            w.blockSignals(True)
            w.setCurrentText(str(val))
            w.blockSignals(False)
        # Also normalise legacy sessions that stored several independent guess
        # checkboxes before this control became a single exclusive selector.
        self._on_initial_guess_changed(self.guess_combo.currentText())
        self.collection_diam_spin.setEnabled(p.collection == "manual")
        checks = {
            self.romega_check: p.R_omega,
            self.copra_local_check: p.copra_stall_patience > 1000,
            self.live_check: p.live_preview,
            self.dispersive_check: p.dispersive,
            self.fit_thickness_check: p.fit_thickness,
            self.fit_tau0_check: p.fit_tau0,
            self.fit_smearing_check: p.fit_smearing,
            self.smearing_check: p.smearing,
            self.focal_check: p.focal,
            self.polish_check: p.polish_two_phase,
            self.pf_lam: p.post_filter_lam,
            self.pf_tau: p.post_filter_tau,
        }
        for w, val in checks.items():
            w.blockSignals(True)
            w.setChecked(bool(val))
            w.blockSignals(False)
        self._update_solver_caps()
        spins = {
            self.maxiters_spin: p.maxiters,
            self.reltol_spin: round(math.log10(p.reltol)),
            self.abstol_spin: round(math.log10(p.abstol)),
            self.perfect_fwhm_spin: p.perfect_init_fs,
            self.thickness_spin: p.thickness_um,
            self.dz_spin: p.delta_z_um,
            self.npoints_spin: p.npoints,
            self.hole_diameter_spin: p.smear_hole_diameter_mm,
            self.hole_spacing_spin: p.smear_hole_spacing_mm,
            self.smear_sigma_spin: p.smear_sigma_delay_fs,
            self.smear_scale_spin: p.smear_scale,
            self.smear_nodes_spin: p.smear_nodes,
            self.focal_f_spin: p.focal_f_mm,
            self.collection_diam_spin: p.collection_diam_mm,
            self.frame_p_spin: p.spectrum_frame_p,
            self.tau0_bound_spin: p.tau0_bound_fs * 1e3,
            self.reg_amp_spin: p.reg_amp,
            self.reg_phase_spin: p.reg_phase,
            self.reg_spectrum_spin: p.reg_spectrum,
            self.reg_time_spin: p.reg_time,
            self.reg_time_lo_spin: p.reg_time_lo_fs,
            self.reg_time_hi_spin: p.reg_time_hi_fs,
            self.nodes_spin: p.n_nodes,
            self.popsize_spin: p.cma_popsize,
            self.std_init_spin: p.cma_std_init,
        }
        for w, val in spins.items():
            w.blockSignals(True)
            w.setValue(val)
            w.blockSignals(False)
        self._update_solver_caps()

    def _reset_defaults(self):
        """Restore the default retrieval settings (as if re-initialised)."""
        from .state import RetrieveParams

        defaults = RetrieveParams()
        for f in RetrieveParams.__dataclass_fields__:
            setattr(self.state.retrieve, f, getattr(defaults, f))
        self._seed_widgets()
        self.set_status("Retrieval settings reset to defaults.")

    def on_enter(self):
        if self.state.tracedata is None:
            self.set_status("Preprocess a trace first.")
            return
        # The smearing width depends on the measurement's carrier, which is only
        # known once a trace is loaded — refresh the readout so it always shows the
        # width the forward model will actually use.
        self._refresh_smear_sigma()
        # If a result already exists (navigated back/forward, or a restored
        # session) but this stage hasn't drawn it, restore the view so it is
        # clear the retrieval is still valid — no re-run needed to proceed.
        if self.state.result is not None and self._base_result is None:
            self._base_result = self.state.result
            self.save_btn.setEnabled(True)
            self.retarget_btn.setEnabled(True)
            self.pf_lam.setEnabled(True)
            self.pf_tau.setEnabled(True)
            self._update_view()

    # -- retrieval ----------------------------------------------------------
    def _seeded_extras(self, seed) -> str:
        """Describe the extra parameters warm-started from ``seed`` (may be empty).

        Reports only what actually changes: an extra is warm-started when it is
        being fitted again *and* the previous run fitted it, so the rule lives
        once in :func:`~croak.session.pipeline.extra_param_centres` and this
        compares its warm centres against the cold (nominal) ones.
        """
        p = self.state.retrieve
        nominal = extra_param_centres(p, None)
        warm = extra_param_centres(p, seed)
        parts = []
        if warm[0] != nominal[0]:
            parts.append(f"thickness {warm[0] / 1e-6:.2f} µm")
        if warm[1] != nominal[1]:
            parts.append(f"τ₀ {warm[1] / 1e-15:.2f} fs")
        if warm[2] != nominal[2]:
            parts.append(f"smearing ×{warm[2]:.3f}")
        return ", ".join(parts)

    def _run(self):
        td = self.state.tracedata
        if td is None:
            self.set_status("No trace to retrieve.")
            return
        p = self.state.retrieve
        truth = self.state.truth
        if p.truth_init and (truth is None or truth.Eomega is None):
            self.set_status(
                "Truth initialisation needs a known complex field; load a "
                "ModelPNPS file containing the complex Eω datasets."
            )
            return
        # "Reuse previous result" warm-starts the run from the last result: its
        # spectrum as the initial guess, and the extras it fitted as their
        # starting centres (see session.pipeline.extra_param_centres).
        guess_override = extras_seed = None
        if p.reuse_result:
            r = self.state.result
            if r is not None and r.spectrum.shape[0] == td.grid.n:
                guess_override, extras_seed = r.spectrum, r
                seeded = self._seeded_extras(r)
                if seeded:
                    self.set_status(f"Reusing the previous result; seeded {seeded}.")
            else:
                self.set_status(
                    "No compatible previous result to reuse; using the default guess."
                )
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.save_btn.setEnabled(False)
        self.retarget_btn.setEnabled(False)
        self._live_full = False
        self._worker = RetrievalWorker(
            td,
            p,
            guess_override=guess_override,
            extras_seed=extras_seed,
            truth=truth,
            preview=p.live_preview,
            scan_path=self._scan_path(),
            scan_window=self._scan_window(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.preview.connect(self._on_preview)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.stopped.connect(self._on_stopped)
        self._worker.failed.connect(self._on_failed)
        self._errors: list[float] = []
        # Blank the read-out rather than leaving the previous run's numbers on
        # screen; the extras block shows what this run starts from.
        self.readout.clear()
        self.readout.set_scalars({"iteration": "0", "elapsed": "0.0 s"})
        self._update_extras_readout(None)
        self._elapsed = None
        self._t0 = time.perf_counter()
        self._elapsed_timer.start()
        self._worker.start()

    def _stop(self):
        if self._worker is not None:
            self._worker.request_stop()

    def _on_progress(self, iteration, R, best_R):
        self._errors.append(R)
        self.set_status(f"iter {iteration}: R = {R:.4%} (best {best_R:.4%})")
        # Cheap: this fires every iteration and has no processed result to draw on.
        self.readout.set_scalars({"iteration": str(iteration), "error": f"{R:.4%}"})
        # Until the first full-plot preview arrives (or for convergence-only
        # solvers), animate the cheap convergence curve. Once a live full plot has
        # taken over the canvas, leave it to :meth:`_on_preview`.
        if not self._live_full:
            self.canvas.render(
                lambda fig: plotting.plot_convergence(
                    fig.add_subplot(111), self._errors
                )
            )

    def _on_preview(self, result):
        """Draw the full 12-panel view for a throttled in-progress result.

        Mirrors :meth:`_update_view` but for a transient snapshot: no post-filter,
        and the result is not stored as the session result. ``process_result``
        divides by the wavelength axis, which has a zero crossing on the centred
        grid — harmless for the live panels, so the divide warning is suppressed.
        The processed result is built here and handed to ``plot_retrieval`` so the
        read-out can share it instead of deriving everything a second time.
        """
        td = self.state.tracedata
        if td is None:
            return
        self._live_full = True
        energy = self.state.load.energy_j or None
        with np.errstate(divide="ignore", invalid="ignore"):
            pr = process_result(
                result, measured=td.trace, Iomega_meas=td.Iomega, energy=energy
            )

        def plot(fig):
            # Re-run by the canvas on a resize, so it carries its own errstate.
            with np.errstate(divide="ignore", invalid="ignore"):
                plotting.plot_retrieval(
                    result,
                    measured=td.trace,
                    Iomega_meas=td.Iomega,
                    lam_min=td.lam_min,
                    lam_max=td.lam_max,
                    truth=self.state.truth,
                    energy=energy,
                    fig=fig,
                    processed=pr,
                )

        self.canvas.render(plot)
        # Snapshots carry the extras the solver has reached, so they move live too.
        self._update_curve_readout(pr, self.state.truth)
        self._update_extras_readout(result)
        self._update_truth_error_readout(result, self.state.truth)
        self.set_status(
            f"iter {len(result.errors)}: R = {result.error:.4%}  (live preview)"
        )

    def _on_finished(self, result, elapsed):
        self._finalise(result, elapsed)

    def _on_stopped(self, result, elapsed):
        """Adopt the partial result captured at the user's Stop and show it.

        The retrieval was aborted mid-run, so the snapshot is a legitimate (if
        non-converged) result: store it, enable Save/post-filters, and keep its
        full plot on screen — labelled "Stopped" rather than "Done".
        """
        self._finalise(result, elapsed, done_label="Stopped")

    def _finalise(self, result, elapsed, *, done_label="Done"):
        """Shared completion path for a finished or stopped retrieval."""
        # An SHG trace is invariant under E(t) -> E*(-t), so the solver returns
        # whichever of the two branches it landed in. When the session carries a
        # known pulse (synthetic or simulated), settle the ambiguity in its favour
        # here — before anything derives from the result — so the plot overlay,
        # the read-out and the reported GDD/TOD sign all describe the same branch.
        if self.state.truth is not None:
            result, _ = resolve_time_direction(result, self.state.truth)
        self._base_result = result
        # The worker timed the run itself; prefer its value over the ticking clock.
        self._elapsed = float(elapsed)
        self._elapsed_timer.stop()
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(True)
        self.retarget_btn.setEnabled(True)
        self.pf_lam.setEnabled(True)
        self.pf_tau.setEnabled(True)
        self._update_view(done_label=done_label)

    def _on_post_filter(self, _=None):
        self.state.retrieve.post_filter_lam = self.pf_lam.isChecked()
        self.state.retrieve.post_filter_tau = self.pf_tau.isChecked()
        self._update_view()

    def _update_view(self, *, done_label="Done"):
        """(Re)apply the post-filters to the stored result and redraw — no re-run.

        ``done_label`` heads the status summary ("Done" on completion, "Stopped"
        when the user aborted the run).
        """
        if self._base_result is None:
            return
        td = self.state.tracedata
        if td is None:
            return
        # window to the measurement λm/τm ranges (not the full grid/trace extent)
        lam_lims = td.lamm_lims if self.pf_lam.isChecked() else None
        tau_lims = td.tau_lims if self.pf_tau.isChecked() else None
        if lam_lims is None and tau_lims is None:
            result = self._base_result
        else:
            result = post_filter(
                self._base_result, lam_lims=lam_lims, tau_lims=tau_lims
            )
        energy = self.state.load.energy_j or None
        pr = process_result(
            result, measured=td.trace, Iomega_meas=td.Iomega, energy=energy
        )
        self.state.set_result(result, pr)
        self.canvas.render(
            lambda fig: plotting.plot_retrieval(
                result,
                measured=td.trace,
                Iomega_meas=td.Iomega,
                lam_min=td.lam_min,
                lam_max=td.lam_max,
                truth=self.state.truth,
                energy=energy,
                fig=fig,
                processed=pr,
            )
        )
        self._update_curve_readout(pr, self.state.truth)
        self._update_extras_readout(result)
        self._update_truth_error_readout(result, self.state.truth)
        # A result rehydrated from file carries no error history, so there is no
        # honest iteration count to show.
        self.readout.set_scalars(
            {
                "iteration": str(len(result.errors)) if result.errors else DASH,
                "error": f"{result.error:.4%}",
                "elapsed": DASH if self._elapsed is None else f"{self._elapsed:.2f} s",
            }
        )
        # Mirror the terminal summary report: error, retrieved & transform-
        # limited FWHM, iteration count and wall-clock time.
        parts = [
            f"{done_label}. R = {result.error:.4%}",
            f"retrieved FWHM {pr.fwhm_retr / 1e-15:.2f} fs",
            f"transform-limited {pr.fwhm_tl / 1e-15:.2f} fs",
            f"{len(result.errors)} iters",
        ]
        # Report any fitted extra parameters (thickness / delay-zero offset /
        # geometric-smearing width).
        if result.thickness is not None:
            parts.append(f"thickness {result.thickness / 1e-6:.2f} µm")
        if result.tau0:
            parts.append(f"τ₀ {result.tau0 / 1e-15:.2f} fs")
        if result.smear_scale is not None:
            parts.append(self._smear_summary(result.smear_scale, td))
        if self._elapsed is not None:
            parts.append(f"{self._elapsed:.2f} s")
        self.set_status("   |   ".join(parts))

    def _smear_summary(self, smear_scale: float, td) -> str:
        """Summarise a fitted smearing strength as a delay width plus its factor.

        The fit optimises one dimensionless multiplier on the kernel widths, which
        scale linearly with it (:meth:`croak.smearing.SmearingKernel.scaled`), so
        the physical delay width is ``smear_scale × sigma_delta`` of the kernel
        the run actually used. The nominal mask geometry is deliberately left
        alone — it is the stored truth, not a readout to be overwritten.
        """
        sigma_fs = self._smear_width_fs(smear_scale, td)
        if sigma_fs is None:
            # The kernel was switched off after the run (this view is re-rendered
            # by the post-filter toggles), so there is no width to convert to.
            return f"smearing ×{smear_scale:.3f}"
        return f"smearing σ {sigma_fs:.2f} fs (×{smear_scale:.3f})"

    def _smear_width_fs(self, smear_scale: float, td) -> float | None:
        """The modelled smearing delay width (fs) at ``smear_scale``, or ``None``.

        ``None`` when there is no kernel to measure — smearing switched off, or an
        SHG trace, which has no three-arm form and so no smearing model at all — so
        callers can fall back to reporting the bare multiplier or nothing. Shared by
        the status line and the read-out so the two cannot drift apart.
        """
        try:
            kernel = smearing_kernel(self.state.retrieve, td)
        except ValueError:
            # SHG: square_boxcars_kernel refuses, and rightly so.
            return None
        if kernel is None:
            return None
        return kernel.sigma_delta * smear_scale / 1e-15

    # -- numeric read-out ---------------------------------------------------
    def _tick_elapsed(self):
        """Advance the live elapsed clock while a retrieval is in flight."""
        self.readout.set_scalars({"elapsed": f"{time.perf_counter() - self._t0:.1f} s"})

    @staticmethod
    def _nm(value_m: float | None) -> str:
        """Format a wavelength/width in nm, or the en dash when undefined."""
        if value_m is None or not np.isfinite(value_m):
            return DASH
        return plotting.sig3(value_m / 1e-9)

    @staticmethod
    def _fs(value_s: float | None) -> str:
        """Format a duration in fs, or the en dash when undefined."""
        if value_s is None or not np.isfinite(value_s):
            return DASH
        return plotting.sig3(value_s / 1e-15)

    @staticmethod
    def _power(value_w: float | None, scale: float) -> str:
        """Format a power against the column's shared SI prefix."""
        if value_w is None or not np.isfinite(value_w):
            return DASH
        return plotting.sig3(value_w / scale)

    def _update_curve_readout(self, pr: ProcessedResult, truth: TruthPulse | None):
        """Fill the per-curve table from a processed result (+ optional truth).

        Row/column coverage follows what is physically defined: the measured
        spectrum carries no phase, so it has no temporal envelope; the
        transform-limited pulse is the retrieved ``|E(ω)|`` with the phase removed,
        so its spectrum *is* the retrieved one and is left blank rather than
        duplicated. Peak powers need a measured pulse energy.
        """
        truth_power = truth.peak_power(pr.energy) if truth is not None else None
        powers = [
            p
            for p in (pr.peak_power, pr.peak_power_tl, truth_power)
            if p is not None and np.isfinite(p)
        ]
        # One SI prefix for the whole column, so the numbers stay comparable.
        scale, unit = plotting.si_power(max(powers)) if powers else (1.0, None)
        self.readout.set_power_unit(unit)

        # The wavelength axis is descending and can hold a zero crossing; the
        # spectral helpers mask that off (see croak.processing.peak_wavelength).
        with np.errstate(divide="ignore", invalid="ignore"):
            self.readout.set_cells(
                "R",
                {
                    "lam_peak": self._nm(peak_wavelength(pr.wavelength, pr.Ilam)),
                    "lam_fwhm": self._nm(spectral_fwhm(pr.wavelength, pr.Ilam)),
                    "fwhm": self._fs(pr.fwhm_retr),
                    "peak_power": self._power(pr.peak_power, scale),
                },
            )
            meas = {}
            if pr.Iw_meas is not None:
                meas = {
                    "lam_peak": self._nm(peak_wavelength(pr.wavelength, pr.Iw_meas)),
                    "lam_fwhm": self._nm(spectral_fwhm(pr.wavelength, pr.Iw_meas)),
                }
            self.readout.set_cells("M", meas)
        self.readout.set_cells(
            "TL",
            {
                "fwhm": self._fs(pr.fwhm_tl),
                "peak_power": self._power(pr.peak_power_tl, scale),
            },
        )
        truth_cells = {}
        if truth is not None:
            truth_cells = {
                "fwhm": self._fs(truth.fwhm),
                "peak_power": self._power(truth_power, scale),
            }
            if truth.lam is not None and truth.Iw is not None:
                truth_cells["lam_peak"] = self._nm(peak_wavelength(truth.lam, truth.Iw))
                truth_cells["lam_fwhm"] = self._nm(spectral_fwhm(truth.lam, truth.Iw))
        self.readout.set_cells("truth", truth_cells)

    def _update_truth_error_readout(self, result, truth) -> None:
        """Show the known-truth errors, or dashes when there is no known pulse.

        These answer a different question from the ``R`` beside them: ``R`` says
        how well the forward model fits the *trace*, the ε's say how close the
        answer is to the true *pulse*. A FROG trace does not determine the pulse
        uniquely, so a small ``R`` can sit on a wrong answer, and reading the two
        together is the point of showing them on one strip.

        The spectral pair needs the truth's complex field, so an intensity-only
        load shows ε(Iₜ) alone rather than a number derived from something else.
        """
        values: dict[str, str] = dict.fromkeys(("eps_It", "eps_Iw", "eps_Ew"), DASH)
        if result is not None and truth is not None:
            # score in the measurement frame: undo any spectrum_frame_p
            # reweighting the retrieval ran with, so a phase-only retrieval
            # reads eps_Iw = 0 rather than the frame factor's footprint
            errors = truth_errors(
                result, truth, frame_p=self.state.retrieve.spectrum_frame_p
            )
            for key, value in (
                ("eps_It", errors.eps_It),
                ("eps_Iw", errors.eps_Iw),
                ("eps_Ew", errors.eps_Ew),
            ):
                if value is not None:
                    # `.3g`: these span decades — a good retrieval near 1e-3, a
                    # truth-seeded forward-model check near 1e-8.
                    values[key] = f"{value:.3g}"
        self.readout.set_scalars(values)

    def _update_extras_readout(self, result):
        """Show τ₀ / thickness / smearing, each marked ``(fit)`` or ``(fixed)``.

        ``result`` is the retrieval to report (``None`` before any run, which shows
        the nominal values the forward model would use). A fitted extra is one the
        solver was free to move: the result records it (``thickness`` and
        ``smear_scale`` are ``None`` when held fixed, ``tau0`` is ``0.0``), and the
        fixed values are the same starting centres the run would use, from
        :func:`~croak.session.pipeline.extra_param_centres`.
        """
        p = self.state.retrieve
        td = self.state.tracedata
        thickness_m, _tau0, smear_scale, _smear_delta = extra_param_centres(p, None)

        if result is not None and result.thickness is not None:
            thickness = f"{result.thickness / 1e-6:.2f} µm (fit)"
        elif p.dispersive:
            thickness = f"{thickness_m / 1e-6:.2f} µm (fixed)"
        else:
            thickness = DASH  # no slab is modelled at all

        # A fitted τ₀ that lands exactly on zero is indistinguishable from a fixed
        # one in the result; the status line makes the same trade.
        if result is not None and result.tau0:
            tau0 = f"{result.tau0 / 1e-15:.2f} fs (fit)"
        else:
            tau0 = "0.00 fs (fixed)"

        smear = DASH
        if result is not None and result.smear_scale is not None:
            smear_scale, fitted = result.smear_scale, True
            smear = f"×{smear_scale:.3f} (fit)"
        else:
            fitted = False
        if p.smearing and td is not None:
            sigma_fs = self._smear_width_fs(smear_scale, td)
            if sigma_fs is not None:
                marker = "fit" if fitted else "fixed"
                smear = f"{sigma_fs:.2f} fs (×{smear_scale:.3f}, {marker})"

        self.readout.set_scalars(
            {"tau0": tau0, "thickness": thickness, "smearing": smear}
        )

    def _on_failed(self, message):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._elapsed_timer.stop()
        # Keep the last true iteration/error/elapsed; blank the curve table, whose
        # numbers would otherwise be silently stale.
        self.readout.clear()
        self._update_extras_readout(None)
        self.set_status("Stopped." if message == "stopped" else f"Failed: {message}")

    def _retarget(self):
        """Ask for a new dataset folder and hand off to the wizard to replay it.

        The wizard rewrites the dataset-folder load paths (keeping calibration
        fixed) and reruns load → preprocess → retrieve with the current settings.
        """
        folder = QFileDialog.getExistingDirectory(self, "Select new dataset folder")
        if not folder:
            return
        self.set_status(f"Retargeting to {folder} …")
        self.state.retarget_requested.emit(folder)

    def _save(self):
        if self.state.result is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Select output folder")
        if not folder:
            return
        save.save_result(
            self.state.result,
            os.path.join(folder, "result.h5"),
            processed=self.state.processed,
            truth=self.state.truth,
            force=True,
        )
        save.save_options(self.state.to_options(), os.path.join(folder, "options.toml"))
        self.canvas.save_figure(os.path.join(folder, "retrieval.pdf"), dpi=600)
        self.set_status(f"Saved result.h5, options.toml, retrieval.pdf to {folder}")
