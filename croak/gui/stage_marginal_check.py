"""Marginal-check stage — pre-retrieval trace QC from the frequency marginal.

Sits between the data loaders (experimental, simulated, synthetic) and the
preprocess stage. It compares the measured trace's wavelength marginal with the
marginal **predicted** from the independent fundamental spectrum
(:func:`croak.marginal_checks.predicted_marginal`): exact for SHG (the spectral
autoconvolution), transform-limited for PG/TG/SD. It overlays the predicted
marginal, marks the predicted vs measured centroid, reports the width
(bracket) check, and lets the user tune the ``(λ/µm)^exp`` third-order
efficiency correction — with an ``Auto`` that matches the marginal centroid onto
the prediction.

The trust region (190–1000 nm for a measured trace; the full loaded range for
the noise-free simulated/synthetic traces) bounds the peak finding, the
centroid/width moments, the marginal normalisation and the trace display
scaling, so out-of-band detector noise — strongly amplified by the tilt — cannot
distort either the checks or the colour scaling (the cause of the load preview
"saturating" as the exponent was raised).

The marginals are plotted against frequency by default: a marginal is a
frequency density and the convolution/centroid/width identities live in
frequency, so that view is undistorted. A wavelength view is available but
applies the λ Jacobian (×1/λ²), which blue-shifts a broad marginal even when its
frequency centroid matches the measured one.
"""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .. import io
from ..marginal_checks import (
    auto_third_order_exponent,
    centroid,
    predicted_marginal,
    rms_width,
)
from ..maths import planck_taper, wlfreq
from ..plotting import cmap_white, signed_pcolormesh
from ..preprocess import (
    omega_to_lambda_density,
    resample_spectrum,
    third_order_scale,
)
from ..session.pipeline import assemble_load_data, assemble_simulated_load_data
from .base import Stage
from .canvas import MplCanvas, add_wide_row, check, combo, dspin, group, with_toolbar
from .widgets import RangeControl

# Number of uniform angular-frequency samples used to resample the independent
# spectrum before predicting the marginal (the prediction's own grid).
_PRED_GRID = 1024
# Angular frequency (rad/s) -> ordinary frequency (PHz), for the frequency axis.
_TWO_PI_PHZ = 2.0 * np.pi * 1e15


def _omega_density(lam: np.ndarray, marg_lam: np.ndarray) -> np.ndarray:
    r"""Angular-frequency density of a wavelength-density marginal (``× λ²``)."""
    return marg_lam * lam**2


class StageMarginalCheck(Stage):
    """Compare the trace marginal with the spectrum-predicted marginal."""

    HELP_TITLE = "Marginal check"
    HELP_INTRO = (
        "Quality-control the trace before retrieval using its frequency "
        "marginal (the trace integrated over delay). The marginal predicted "
        "from the independent fundamental spectrum is exact for SHG (the "
        "spectral autoconvolution) and transform-limited for PG/TG/SD; comparing "
        "shape, centroid and width flags calibration, spectral-response, scatter "
        "and detector-nonlinearity problems. Tune the (λ/µm)^exp third-order "
        "efficiency correction here; 'Auto' matches the marginal centroid to the "
        "prediction."
    )
    HELP = [
        (
            "Spectral trust region",
            [
                (
                    "λ range (nm)",
                    "Lower/upper wavelength bounds (nm) of the trust region. It "
                    "windows the centroid/width checks and the marginal "
                    "normalisation, scales the trace display, and Planck-tapers the "
                    "independent spectrum before the marginal is predicted — so "
                    "noisy measured wings (amplified by the (λ/µm)^exp tilt) cannot "
                    "contaminate the prediction. Defaults to 190–1000 nm for a "
                    "measured trace; the noise-free simulated/synthetic traces "
                    "default to their full loaded wavelength range.",
                ),
            ],
        ),
        (
            "Display",
            [
                (
                    "marginal axis",
                    "Plot the marginals against frequency or wavelength. Frequency "
                    "(the default) is the physically correct space: the marginal is "
                    "a frequency density and the centroid/width identities live "
                    "there. On a wavelength axis the predicted marginal is converted "
                    "to a wavelength density (×1/λ²), which blue-shifts a broad "
                    "(e.g. transform-limited) marginal even when its frequency "
                    "centroid matches the measured one — visually misleading.",
                ),
                (
                    "marginal scale",
                    "Plot the marginal comparison on a linear or logarithmic "
                    "intensity axis. Both are peak-normalised within the trust "
                    "region; linear is the default.",
                ),
            ],
        ),
        (
            "Third-order correction",
            [
                (
                    "3rd-order (λ/µm)^exp",
                    "Apply a (λ/µm)^exp efficiency correction to the trace (a "
                    "TG/PG χ³ detector response). Redraws live; available for a "
                    "measured or simulated trace, not a synthetic one.",
                ),
                (
                    "exp",
                    "Exponent of the (λ/µm)^exp scaling (a real number). 'Auto' "
                    "picks the exponent that brings the trace-marginal centroid "
                    "onto the centroid predicted from the independent spectrum "
                    "(needs a loaded spectrum).",
                ),
            ],
        ),
        (
            "Marginal correction",
            [
                (
                    "Marginal correct (SHG)",
                    "Scale the trace's frequency rows onto the SHG marginal "
                    "predicted from the independent spectrum (the spectral "
                    "autoconvolution). The correction is applied during preprocess; "
                    "here a dashed trace-colour curve previews it snapping onto the "
                    "prediction. Available for an SHG trace with an independent "
                    "spectrum (all workflows).",
                ),
            ],
        ),
    ]

    def __init__(self, state, wizard):
        super().__init__(state)
        self.wizard = wizard
        # Measured default; re-seeded per trace in on_enter (full range for the
        # noise-free simulated/synthetic traces, 190–1000 nm for measured).
        self._trust_nm = (190.0, 1000.0)
        self._trust_key: tuple | None = None  # the trace a default was seeded for

        # -- trust region --
        box, form = group("Spectral trust region")
        self.trust = RangeControl(100.0, 3000.0, *self._trust_nm, decimals=0, step=10.0)
        self.trust.changed.connect(self._on_trust)
        add_wide_row(form, "λ range (nm)", self.trust)
        self.controls.addWidget(box)

        # -- display --
        box, form = group("Display")
        self.axis_combo = combo(
            ["frequency", "wavelength"], "frequency", self._on_scale
        )
        form.addRow("marginal axis", self.axis_combo)
        self.scale_combo = combo(["linear", "log"], "linear", self._on_scale)
        form.addRow("marginal scale", self.scale_combo)
        self.controls.addWidget(box)

        # -- third-order correction --
        box, form = group("Third-order correction")
        self.third_order_check = check(
            "3rd-order (λ/µm)^exp", False, self._on_third_order_toggled
        )
        form.addRow(self.third_order_check)
        self.exp_spin = dspin(
            0.0, 12.0, 2.0, self._on_exp_changed, decimals=2, step=0.1
        )
        self.exp_spin.setKeyboardTracking(False)  # commit on Enter/step
        self.auto_btn = QPushButton("Auto")
        self.auto_btn.setToolTip(
            "Pick the exponent whose corrected marginal centroid matches the "
            "centroid predicted from the independent spectrum."
        )
        self.auto_btn.clicked.connect(self._auto_exponent)
        exp_row = QWidget()
        exp_layout = QHBoxLayout(exp_row)
        exp_layout.setContentsMargins(0, 0, 0, 0)
        exp_layout.addWidget(self.exp_spin)
        exp_layout.addWidget(self.auto_btn)
        form.addRow("exp", exp_row)
        self.controls.addWidget(box)

        # -- SHG marginal correction (applied at preprocess; previewed here) --
        box, form = group("Marginal correction")
        self.marginal_correct_check = check(
            "Marginal correct (SHG)",
            self.state.preproc.marginal_correct,
            self._on_marginal_correct,
        )
        self.marginal_correct_check.setToolTip(
            "Scale the trace's frequency rows onto the spectrum-predicted SHG "
            "marginal (the spectral autoconvolution). Applied during preprocess; "
            "the dashed curve here previews the snap onto the prediction. SHG only "
            "and needs an independent spectrum."
        )
        form.addRow(self.marginal_correct_check)
        self.controls.addWidget(box)

        self.controls.addWidget(self.status_label)
        self.controls.addStretch(1)

        right = QWidget()
        rv = QVBoxLayout(right)
        self.trace_canvas = MplCanvas()
        self.marg_canvas = MplCanvas()
        rv.addWidget(with_toolbar(self.trace_canvas))
        rv.addWidget(with_toolbar(self.marg_canvas))
        self.set_plot_area(right)

        self.apply_help()

    # -- loader coupling ----------------------------------------------------
    def _loader_params(self):
        """The active entry's loader params, or ``None`` for a synthetic trace.

        The third-order exponent is a *loader* parameter baked into the trace by
        the assembler, so this stage reads/writes it on the right params object —
        experimental :class:`~croak.session.params.LoadParams` or simulated
        :class:`~croak.session.params.SimulatedLoadParams` (both carry the
        ``third_order`` / ``third_order_exp`` fields). Synthetic traces carry no
        efficiency tilt, so the controls are disabled and this returns ``None``.
        """
        entry = self.state.entry
        if entry == "simulated":
            return self.state.simulated
        if entry == "experimental":
            return self.state.load
        return None  # synthetic

    # -- lifecycle ----------------------------------------------------------
    def _seed_trust(self, data) -> None:
        """Default the trust region for a freshly loaded trace.

        Measured traces default to 190–1000 nm (a UV–NIR band that excludes the
        detector's noisy out-of-band wings); the noise-free simulated and
        synthetic traces default to their full loaded wavelength range. Keyed on
        the trace so navigating away and back keeps a manual adjustment, while a
        genuinely different trace re-defaults (an exponent change does not — it
        re-enters via the draw path, not here, and keeps the same λ range).
        """
        lam_nm = data["lam"] / 1e-9
        entry = self.state.entry
        key = (entry, round(float(lam_nm.min())), round(float(lam_nm.max())))
        if key == self._trust_key:
            return
        if entry == "experimental":
            trust = (190.0, 1000.0)
        else:  # simulated / synthetic: no detector noise -> the full range
            trust = (float(np.floor(lam_nm.min())), float(np.ceil(lam_nm.max())))
        self._trust_nm = trust
        self._trust_key = key
        self.trust.set_values(*trust)  # signal-safe

    def on_enter(self) -> None:
        data = self.state.load_data
        if data is None:
            self.set_status("Load a trace first.")
            return
        self._seed_trust(data)
        params = self._loader_params()
        enabled = params is not None
        self.third_order_check.setEnabled(enabled)
        self.exp_spin.setEnabled(enabled)
        if params is not None:
            self.third_order_check.blockSignals(True)
            self.exp_spin.blockSignals(True)
            self.third_order_check.setChecked(bool(params.third_order))
            self.exp_spin.setValue(float(params.third_order_exp))
            self.third_order_check.blockSignals(False)
            self.exp_spin.blockSignals(False)
        # Auto needs an independent spectrum.
        self.auto_btn.setEnabled(enabled and data.get("Ilam_spec") is not None)
        self._seed_marginal_correct(data)
        self._draw()

    def _marginal_correct_available(self, data) -> bool:
        """Whether SHG marginal correction can run (SHG trace + a spectrum)."""
        return (
            data is not None
            and data.get("interaction") == "shg"
            and data.get("Ilam_spec") is not None
        )

    def _seed_marginal_correct(self, data) -> None:
        """Enable the correction only when valid; force it off otherwise.

        Leaving ``marginal_correct`` set for a non-SHG / spectrum-less trace would
        make the preprocess stage raise, so an unavailable correction is cleared.
        """
        available = self._marginal_correct_available(data)
        self.marginal_correct_check.setEnabled(available)
        if not available and self.state.preproc.marginal_correct:
            self.state.preproc.marginal_correct = False
        self.marginal_correct_check.blockSignals(True)
        self.marginal_correct_check.setChecked(
            available and self.state.preproc.marginal_correct
        )
        self.marginal_correct_check.blockSignals(False)

    # -- control callbacks --------------------------------------------------
    def _on_trust(self, lo: float, hi: float) -> None:
        self._trust_nm = (lo, hi)
        self._draw()  # trust only affects display/QC, no re-assembly

    def _on_scale(self, _value: str) -> None:
        self._draw()

    def _on_marginal_correct(self, checked: bool) -> None:
        # The flag drives the preprocess regrid (croak.preprocess.load_and_clean);
        # here we only redraw to preview the marginal snapping onto the prediction.
        self.state.preproc.marginal_correct = bool(checked)
        self._draw()

    def _on_third_order_toggled(self, value: bool) -> None:
        params = self._loader_params()
        if params is None:
            return
        params.third_order = bool(value)
        self._reassemble()

    def _on_exp_changed(self, value: float) -> None:
        params = self._loader_params()
        if params is None:
            return
        params.third_order_exp = float(value)
        if params.third_order:
            self._reassemble()

    def _auto_exponent(self) -> None:
        """Fit the exponent so the marginal centroid matches the prediction."""
        data = self.state.load_data
        params = self._loader_params()
        if data is None or params is None:
            return
        if data.get("Ilam_spec") is None:
            self.set_status("Load an independent spectrum to auto-fit the exponent.")
            return
        lam = data["lam"]
        marg = data["trace"].sum(axis=1)
        # Recover the marginal before any tilt (the (λ/µm)^n factor is separable
        # along delay), so the fit returns the absolute exponent to apply.
        if params.third_order:
            marg = marg / third_order_scale(lam, params.third_order_exp)
        window = (self._trust_nm[0] * 1e-9, self._trust_nm[1] * 1e-9)
        # Match the *displayed* predicted centroid (same spectrum windowing as the
        # overlay) so the two dashed centroid lines coincide after the fit.
        pred = self._predict(data["lam_spec"], data["Ilam_spec"], data["interaction"])
        target = pred["centroid_omega"] if pred is not None else None
        try:
            exp = round(
                auto_third_order_exponent(
                    lam,
                    marg,
                    data["lam_spec"],
                    data["Ilam_spec"],
                    data["interaction"],
                    window=window,
                    target_centroid=target,
                ),
                2,
            )
        except ValueError as exc:
            self.set_status(f"Auto-fit failed: {exc}")
            return
        params.third_order = True
        params.third_order_exp = exp
        self.third_order_check.blockSignals(True)
        self.exp_spin.blockSignals(True)
        self.third_order_check.setChecked(True)
        self.exp_spin.setValue(exp)
        self.third_order_check.blockSignals(False)
        self.exp_spin.blockSignals(False)
        # _reassemble redraws and writes the centroid/width QC summary; the fitted
        # exponent itself is shown in the spin box, so don't overwrite that summary.
        self._reassemble()

    def _reassemble(self) -> None:
        """Rebuild ``load_data`` from the active loader after a correction change."""
        entry = self.state.entry
        try:
            if entry == "simulated":
                data = assemble_simulated_load_data(self.state.simulated)
            elif entry == "experimental":
                data = assemble_load_data(self.state.load)
            else:
                return  # synthetic: no efficiency tilt to re-apply
        except Exception as exc:  # surfaced in the GUI
            self.set_status(f"Reload failed: {exc}")
            return
        # The exponent is not part of the trace signature, so set_load_data here
        # invalidates the downstream result without reseeding the preprocess
        # windows (those are kept; preprocess regrids on its on_enter).
        self.state.set_load_data(data)
        self._draw()

    # -- plotting -----------------------------------------------------------
    def _draw(self) -> None:
        data = self.state.load_data
        if data is None:
            return
        lam = data["lam"]
        lam_nm = lam / 1e-9
        scan = data["scanaxis"]
        tau_fs = (
            io.position_to_delay(scan) if data["input_unit"] == "position" else scan
        ) / 1e-15
        trace = data["trace"]
        lo, hi = self._trust_nm
        in_band = (lam_nm >= lo) & (lam_nm <= hi)

        self._draw_trace(tau_fs, lam_nm, trace, in_band)
        self._draw_marginal(lam, lam_nm, trace, in_band, data)

    def _draw_trace(self, tau_fs, lam_nm, trace, in_band) -> None:
        """Trace image with the preprocess-page colour scheme and trust-scaled vmax."""
        ax = self.trace_canvas.single_axes()
        # Scale to the in-trust-region peak so amplified out-of-band noise (the
        # (λ/µm)^exp tilt blows up the long-λ tail) cannot wash out the signal.
        band = trace[in_band] if np.any(in_band) else trace
        vmax = float(np.nanmax(np.abs(band))) if band.size else 1.0
        signed_pcolormesh(
            ax,
            tau_fs,
            lam_nm,
            trace,
            db=False,
            cmap_pos=cmap_white("viridis"),
            vmax=vmax if vmax > 0 else None,
        )
        # Clip the wavelength axis to the in-band signal so the view tracks it.
        sig = lam_nm[in_band & (trace.max(axis=1) > 1e-2 * (vmax or 1.0))]
        if sig.size:
            ax.set_ylim(sig.min(), sig.max())
        ax.set_xlabel("Delay (fs)")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_title("Measured trace")
        self.trace_canvas.draw_idle()

    def _draw_marginal(self, lam, lam_nm, trace, in_band, data) -> None:
        lo, hi = self._trust_nm  # nm
        log = self.scale_combo.currentText() == "log"
        freq = self.axis_combo.currentText() == "frequency"
        ax = self.marg_canvas.single_axes()

        # Marginals are frequency densities. On the frequency axis we plot the
        # ω-density vs ν=ω/2π (undistorted). On the wavelength axis we plot the
        # λ-density vs λ; the λ Jacobian (×1/λ²) then blue-shifts a broad marginal
        # even when its frequency centroid matches — see the 'marginal axis' help.
        # When the SHG marginal correction is active, dim the raw marginal and
        # overlay the corrected one (which, by construction, equals the predicted
        # SHG autoconvolution — see _draw the dashed C0 curve below).
        correcting = self.marginal_correct_check.isChecked() and (
            self._marginal_correct_available(data)
        )
        marg = trace.sum(axis=1)
        if freq:
            x_meas, y_meas = wlfreq(lam) / _TWO_PI_PHZ, marg * lam**2
        else:
            x_meas, y_meas = lam_nm, marg
        ax.plot(
            x_meas,
            _peak_norm(y_meas, in_band),
            label="trace marginal",
            c="C0",
            alpha=0.45 if correcting else 1.0,
        )

        interaction = data["interaction"]
        pred = None
        if data.get("Ilam_spec") is not None:
            lam_spec = np.asarray(data["lam_spec"])
            Ilam_spec = np.asarray(data["Ilam_spec"])
            sl_band = (lam_spec / 1e-9 >= lo) & (lam_spec / 1e-9 <= hi)
            if freq:
                x_sp, y_sp = wlfreq(lam_spec) / _TWO_PI_PHZ, Ilam_spec * lam_spec**2
            else:
                x_sp, y_sp = lam_spec / 1e-9, Ilam_spec
            ax.plot(x_sp, _peak_norm(y_sp, sl_band), label="spectrum", c="C1", lw=1)
            pred = self._predict(lam_spec, Ilam_spec, interaction)
            if pred is not None:
                if freq:
                    x_pr, y_pr = pred["omega"] / _TWO_PI_PHZ, pred["marg_omega"]
                    pb = (x_pr >= self._f_lo()) & (x_pr <= self._f_hi())
                else:
                    x_pr, y_pr = pred["lam_nm"], pred["marg_lam"]
                    pb = (x_pr >= lo) & (x_pr <= hi)
                ax.plot(x_pr, _peak_norm(y_pr, pb), label="predicted", c="C3", ls="--")
                if correcting:
                    # the correction makes the trace marginal equal the predicted
                    # SHG autoconvolution; show that snap in the trace colour.
                    ax.plot(
                        x_pr,
                        _peak_norm(y_pr, pb),
                        c="C0",
                        ls="--",
                        lw=1.5,
                        label="trace marginal (corrected)",
                    )

        # Centroid check: vertical lines. The spectrum centroid (C1) sits on the
        # predicted-marginal centroid (C3) for PG/TG/SD, but at half it for SHG
        # (whose marginal is centred at 2ω̄_S).
        c_meas_nm, sigma_meas = self._measured_moments(lam, marg, in_band)
        to_x = (lambda nm: float(wlfreq(nm * 1e-9) / _TWO_PI_PHZ)) if freq else float
        if c_meas_nm is not None:
            ax.axvline(to_x(c_meas_nm), c="C0", ls=":", lw=1, label="meas centroid")
        if pred is not None:
            ax.axvline(
                to_x(pred["spectrum_centroid_nm"]),
                c="C1",
                ls="--",
                lw=1,
                label="spec centroid",
            )
            ax.axvline(
                to_x(pred["centroid_nm"]), c="C3", ls=":", lw=1, label="pred centroid"
            )

        self._annotate_checks(ax, c_meas_nm, sigma_meas, pred, interaction)

        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.5)
        else:
            ax.set_ylim(0.0, 1.1)
        if freq:
            ax.set_xlim(self._f_lo(), self._f_hi())
            ax.set_xlabel("Frequency (PHz)")
        else:
            ax.set_xlim(lo, hi)
            ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.legend(fontsize=7)
        self.marg_canvas.draw_idle()

    def _f_lo(self) -> float:
        """Low frequency edge (PHz) of the trust region (from its long-λ bound)."""
        return float(wlfreq(self._trust_nm[1] * 1e-9) / _TWO_PI_PHZ)

    def _f_hi(self) -> float:
        """High frequency edge (PHz) of the trust region (from its short-λ bound)."""
        return float(wlfreq(self._trust_nm[0] * 1e-9) / _TWO_PI_PHZ)

    # -- physics plumbing ---------------------------------------------------
    def _trust_window(self, lam: np.ndarray) -> np.ndarray:
        """Near-hard window over the trust region (1 inside, 0 outside).

        Zero outside ``[lo, hi]`` with only a few-nm Planck roll-off just inside
        the bounds — enough to stop the sharp cut ringing through the FFTs, but
        narrow enough that it does not reshape the in-band spectrum (a wide taper
        would clip a broadband spectrum's wing and bias the predicted centroid).
        """
        lo_nm, hi_nm = self._trust_nm
        lo, hi = lo_nm * 1e-9, hi_nm * 1e-9
        edge = min(0.02 * max(hi - lo, 1e-12), 5e-9)
        return planck_taper(lam, lo, lo + edge, hi - edge, hi)

    def _predict(self, lam_spec, Ilam_spec, interaction) -> dict | None:
        """Predicted marginal on the trace's wavelength axis, plus its centroid (nm).

        Planck-tapers the spectrum to the **trust region** first — the noisy
        measured wings outside it would otherwise contaminate the transform-limited
        field and the autoconvolution — then resamples onto a uniform
        angular-frequency grid (applying the λ→ω Jacobian), predicts the marginal,
        and maps it back to a wavelength density for overlay. The taper is local to
        this prediction; preprocessing re-derives its own spectrum from the
        untouched ``load_data`` (windowed to the retrieval grid in
        :func:`croak.preprocess.load_and_clean`).
        """
        order = np.argsort(lam_spec)
        lam_s = lam_spec[order]
        Il = np.clip(Ilam_spec[order], 0.0, None) * self._trust_window(lam_s)
        if not np.any(Il > 0):
            return None
        # Predict on a grid spanning the trust ∩ spectrum band, so the spectrum
        # width σ_S is measured over the same window as the trace marginal — the
        # width bracket and the measured width are then directly comparable.
        lo_nm, hi_nm = self._trust_nm
        lam_lo = max(lo_nm * 1e-9, float(lam_s[0]))
        lam_hi = min(hi_nm * 1e-9, float(lam_s[-1]))
        if lam_hi <= lam_lo:
            return None
        omega = np.linspace(wlfreq(lam_hi), wlfreq(lam_lo), _PRED_GRID)
        try:
            s_omega = resample_spectrum(lam_s, Il, omega)
            pred = predicted_marginal(omega, s_omega, interaction)
        except ValueError, ZeroDivisionError:
            return None
        lam_pred = wlfreq(pred.omega)
        marg_lam = omega_to_lambda_density(pred.marginal, lam_pred)
        srt = np.argsort(lam_pred)
        # Centroid of the (windowed) fundamental spectrum, ω̄_S. It equals the
        # predicted marginal centroid for PG/TG/SD (the hard ω̄_S check) but is at
        # half it for SHG, whose marginal sits at 2ω̄_S.
        spec_centroid = float(centroid(omega, s_omega))
        return {
            "lam_nm": lam_pred[srt] / 1e-9,
            "marg_lam": marg_lam[srt],
            # frequency-domain form (the physical one): the ω-density on its own
            # ascending ω axis, undistorted by the λ Jacobian.
            "omega": pred.omega,
            "marg_omega": pred.marginal,
            "centroid_nm": float(wlfreq(pred.centroid) / 1e-9),
            "spectrum_centroid_nm": float(wlfreq(spec_centroid) / 1e-9),
            "rms_width": pred.rms_width,
            "sigma_spectrum": pred.sigma_spectrum,
            "width_bracket": pred.width_bracket,
            "centroid_is_hard": pred.centroid_is_hard,
            "width_is_hard": pred.width_is_hard,
            "centroid_omega": pred.centroid,
        }

    def _measured_moments(self, lam, marg, in_band):
        """Measured marginal centroid (nm) and RMS width (rad/s) in the trust band."""
        sel = in_band & (marg > 0)
        if np.count_nonzero(sel) < 3:
            return None, None
        omega = wlfreq(lam[sel])
        weights = _omega_density(lam[sel], marg[sel])
        c = centroid(omega, weights)
        return float(wlfreq(c) / 1e-9), rms_width(omega, weights)

    def _annotate_checks(self, ax, c_meas_nm, sigma_meas, pred, interaction) -> None:
        """Write the numeric centroid/width QC summary to the status line + plot."""
        if pred is None or c_meas_nm is None:
            self.set_status("No independent spectrum: showing the trace marginal only.")
            return
        d_centroid = c_meas_nm - pred["centroid_nm"]
        hardness = "hard" if pred["centroid_is_hard"] else "TL/soft"
        lines = [
            f"{interaction.upper()} marginal check",
            f"centroid: meas {c_meas_nm:.1f} nm vs pred {pred['centroid_nm']:.1f} nm "
            f"(Δ {d_centroid:+.1f} nm, {hardness})",
        ]
        # Report RMS widths as wavelengths (σ_λ ≈ λ̄·σ_ω/ω̄ at the centroid), since
        # the moments themselves are computed in ω. min = σ_S (rigorous lower
        # bound for PG/SD), max = the transform-limited marginal width σ_TL.
        sigma_s = pred["sigma_spectrum"]
        _, upper = pred["width_bracket"] if pred["width_bracket"] else (sigma_s, None)
        cw, cn = pred["centroid_omega"], pred["centroid_nm"]
        c_meas_w = float(wlfreq(c_meas_nm * 1e-9))
        w_meas = c_meas_nm * sigma_meas / c_meas_w if c_meas_w > 0 else float("nan")
        w_min = cn * sigma_s / cw if cw > 0 else float("nan")
        if pred["width_is_hard"]:  # SHG: exact predicted width
            w_pred = cn * pred["rms_width"] / cw if cw > 0 else float("nan")
            lines.append(
                f"width RMS: meas {w_meas:.1f} nm vs predicted {w_pred:.1f} nm (exact)"
            )
        elif upper is not None:  # PG/TG: a [σ_S, σ_TL] bracket
            w_max = cn * upper / cw if cw > 0 else float("nan")
            if sigma_meas < sigma_s:
                flag = "⚠ below σ_S (clipping?)"
            elif sigma_meas > upper:
                flag = "⚠ above TL (scatter/background?)"
            else:
                flag = "✓ in bracket"
            lines.append(
                f"width RMS: meas {w_meas:.1f} nm | predicted [{w_min:.1f}, "
                f"{w_max:.1f}] nm (σ_S…TL) {flag}"
            )
        else:  # SD: only the lower bound is rigorous
            ok = sigma_meas > sigma_s
            lines.append(
                f"width RMS: meas {w_meas:.1f} nm | predicted min {w_min:.1f} nm "
                f"(σ_S) {'✓' if ok else '⚠ below σ_S'}"
            )
        self.set_status("  •  ".join(lines))


def _peak_norm(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Peak-normalise ``y`` to its in-mask maximum (for shape comparison)."""
    y = np.asarray(y, dtype=float)
    sel = y[mask] if np.any(mask) else y
    peak = sel.max() if sel.size and sel.max() > 0 else (y.max() if y.size else 1.0)
    return y / peak if peak > 0 else y
