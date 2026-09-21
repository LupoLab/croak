"""Generate-synthetic-trace panel.

Builds a known pulse (Gaussian or sech², with GDD/TOD/FOD and optional material
dispersion), simulates a FROG measurement of it on a user-chosen wavelength ×
delay sampling, previews the pulse and trace live, and then hands the synthetic
"measurement" into the normal preprocess → retrieve pipeline (so the retriever
rediscovers the pulse). The internal computation grid is chosen automatically to
represent the pulse faithfully, independently of the output sampling.

The known ground-truth pulse (and an independent fundamental spectrum) travel
with the loaded data (``load_data["truth"]``) so the marginal-check and retrieval
screens can compare against them.
"""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QTimer

from .. import dispersion
from ..constants import C
from ..forward import maketrace
from ..grid import Grid
from ..interactions import INTERACTIONS, get_interaction
from ..materials import MATERIALS
from ..processing import TruthPulse
from ..pulses import gaussian_pulse, sech_pulse
from .base import Stage
from .canvas import (
    MplCanvas,
    add_wide_row,
    check,
    combo,
    dspin,
    group,
    spin,
    with_toolbar,
)
from .stage_load import seed_preproc_defaults
from .widgets import RangeControl, ValueControl

# dispersion slider ranges follow StageDispersion (fs² / fs³ / fs⁴)
_GDD_RANGE = (-10000.0, 10000.0)
_TOD_RANGE = (-100000.0, 100000.0)
_FOD_RANGE = (-1000000.0, 1000000.0)
_SHAPES = ["Gaussian", "Sech²"]


class StageSynthetic(Stage):
    """Synthetic-pulse / synthetic-FROG generator (reached from the menu)."""

    def __init__(self, state, wizard):
        super().__init__(state)
        self.wizard = wizard
        self._last = None  # cache of the last computed (grid, ew, omega0)

        # debounce rapid slider changes into one recompute
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._recompute)

        # -- Pulse ------------------------------------------------------
        box, form = group("Pulse")
        self.shape = combo(_SHAPES, "Gaussian", self._changed)
        form.addRow("shape", self.shape)
        self.dur = ValueControl(
            1.0, 100.0, 10.0, decimals=1, step=0.5, slider_scale=10.0
        )
        self.wl = ValueControl(
            200.0, 2000.0, 800.0, decimals=0, step=1.0, slider_scale=1.0
        )
        self.gdd = ValueControl(
            *_GDD_RANGE, 0.0, decimals=2, step=0.1, slider_scale=0.02
        )
        self.tod = ValueControl(
            *_TOD_RANGE, 0.0, decimals=2, step=1.0, slider_scale=0.002
        )
        self.fod = ValueControl(
            *_FOD_RANGE, 0.0, decimals=2, step=10.0, slider_scale=0.0002
        )
        for c in (self.dur, self.wl, self.gdd, self.tod, self.fod):
            c.changed.connect(self._changed)
        form.addRow("duration (fs)", self.dur)
        form.addRow("wavelength (nm)", self.wl)
        form.addRow("GDD (fs²)", self.gdd)
        form.addRow("TOD (fs³)", self.tod)
        form.addRow("FOD (fs⁴)", self.fod)
        self.controls.addWidget(box)

        # -- FROG trace -------------------------------------------------
        box, form = group("FROG trace")
        self.interaction = combo(list(INTERACTIONS), "shg", self._on_interaction)
        form.addRow("interaction", self.interaction)
        # RangeControl exposes a changed(lo, hi) signal (no public getter), so we
        # track the current values ourselves. The delay/λ sampling is set by an
        # interlocked count + step at fixed range: editing one updates the other.
        self._delay = (-60.0, 60.0)
        self.delay_range = RangeControl(-1000.0, 1000.0, *self._delay)
        self.delay_range.changed.connect(self._on_delay)
        add_wide_row(form, "delay (fs)", self.delay_range)
        self.ndelay = spin(8, 1024, 128, self._on_ndelay)
        form.addRow("# delays", self.ndelay)
        self.delay_step = dspin(
            1e-3, 1e4, 1.0, self._on_delay_step, decimals=3, step=0.1
        )
        self.delay_step.setKeyboardTracking(False)
        form.addRow("delay step (fs)", self.delay_step)
        # Wide enough that the default trace decays to nothing inside its own λ
        # window. A window cropped to the visible signal looks tidier but leaves
        # the regrid's edge taper cutting live rows, which is the very mistake
        # :attr:`croak.preprocess.TraceData.taper_loss` exists to catch — a demo
        # should not ship it.
        self._wl = (300.0, 550.0)
        self.wl_range = RangeControl(100.0, 2000.0, *self._wl)
        self.wl_range.changed.connect(self._on_wl)
        add_wide_row(form, "λ window (nm)", self.wl_range)
        self.nwl = spin(8, 2048, 256, self._on_nwl)
        form.addRow("# λ points", self.nwl)
        self.wl_step = dspin(1e-4, 1e3, 1.0, self._on_wl_step, decimals=4, step=0.1)
        self.wl_step.setKeyboardTracking(False)
        form.addRow("λ step (nm)", self.wl_step)
        self.controls.addWidget(box)

        # -- Spectrum (independent) -------------------------------------
        box, form = group("Spectrum (independent)")
        self.spec_mirror = check(
            "Mirror FROG trace λ window", True, self._on_spec_mirror
        )
        form.addRow(self.spec_mirror)
        # fundamental-spectrum λ window (nm); mirrors the trace window scaled to
        # the fundamental (×2 for SHG) until the user unmirrors it.
        self._spec_wl = (700.0, 900.0)
        self.spec_range = RangeControl(
            100.0, 4000.0, *self._spec_wl, decimals=0, step=10.0
        )
        self.spec_range.changed.connect(self._on_spec_range)
        add_wide_row(form, "λ range (nm)", self.spec_range)
        self.nspec = spin(8, 4096, 256, self._on_nspec)
        form.addRow("# points", self.nspec)
        self.spec_step = dspin(1e-4, 1e3, 1.0, self._on_spec_step, decimals=4, step=0.1)
        self.spec_step.setKeyboardTracking(False)
        form.addRow("step (nm)", self.spec_step)
        self.controls.addWidget(box)

        # -- Dispersive model -------------------------------------------
        box, form = group("Dispersive model")
        self.use_material = check("Propagate through material", False, self._changed)
        form.addRow(self.use_material)
        self.material = combo(list(MATERIALS), MATERIALS[0], self._changed)
        form.addRow("material", self.material)
        self.thickness = ValueControl(
            0.0, 50.0, 1.0, decimals=3, step=0.1, slider_scale=100.0
        )
        self.thickness.changed.connect(self._changed)
        add_wide_row(form, "thickness (mm)", self.thickness)
        self.controls.addWidget(box)

        # -- Noise ------------------------------------------------------
        box, form = group("Noise (optional)")
        self.noise_add = ValueControl(
            0.0, 0.5, 0.0, decimals=3, step=0.005, slider_scale=1000.0
        )
        self.noise_pois = ValueControl(
            0.0, 1e5, 0.0, decimals=0, step=100.0, slider_scale=0.01
        )
        self.noise_add.changed.connect(self._changed)
        self.noise_pois.changed.connect(self._changed)
        add_wide_row(form, "additive (frac)", self.noise_add)
        add_wide_row(form, "Poisson counts", self.noise_pois)
        self.controls.addWidget(box)

        # Navigation (Back / "Generate →") lives in the shared wizard footer, via
        # the entry-page hooks below.
        self.controls.addWidget(self.status_label)
        self.controls.addStretch(1)

        self._seed_sampling()

        self.canvas = MplCanvas(figsize=(9, 7))
        self.set_plot_area(with_toolbar(self.canvas))

    # -- lifecycle ----------------------------------------------------------
    def on_enter(self):
        self._recompute()

    def _changed(self, *_):
        self._timer.start()

    # -- sampling interlock (count <-> step at fixed range) -----------------
    @staticmethod
    def _count_to_step(span: float, count: int) -> float:
        """Linspace step for ``count`` points spanning ``span``."""
        return abs(span) / (count - 1) if count > 1 else abs(span)

    @staticmethod
    def _step_to_count(span: float, step: float, lo: int, hi: int) -> int:
        """Number of points giving ``step`` over ``span`` (clamped to ``[lo, hi]``)."""
        if step <= 0:
            return lo
        return int(min(max(round(abs(span) / step) + 1, lo), hi))

    @staticmethod
    def _set_spin(widget, value) -> None:
        """Set a spin box without emitting its change signal."""
        widget.blockSignals(True)
        widget.setValue(value)
        widget.blockSignals(False)

    def _seed_sampling(self) -> None:
        """Initialise the step displays and the mirrored-spectrum controls."""
        self._set_spin(
            self.delay_step,
            self._count_to_step(
                self._delay[1] - self._delay[0], int(self.ndelay.value())
            ),
        )
        self._set_spin(
            self.wl_step,
            self._count_to_step(self._wl[1] - self._wl[0], int(self.nwl.value())),
        )
        mirror = self.spec_mirror.isChecked()
        for w in (self.spec_range, self.nspec, self.spec_step):
            w.setEnabled(not mirror)
        self._sync_spec_mirror()
        if not mirror:
            self._set_spin(
                self.spec_step,
                self._count_to_step(
                    self._spec_wl[1] - self._spec_wl[0], int(self.nspec.value())
                ),
            )

    def _on_interaction(self, _value):
        # the SHG fundamental sits at 2× the signal wavelength, so a mirrored
        # spectrum window must follow the interaction's omega0 scaling.
        self._sync_spec_mirror()
        self._changed()

    def _on_delay(self, lo, hi):
        self._delay = (lo, hi)
        self._set_spin(
            self.delay_step, self._count_to_step(hi - lo, int(self.ndelay.value()))
        )
        self._changed()

    def _on_ndelay(self, count):
        span = self._delay[1] - self._delay[0]
        self._set_spin(self.delay_step, self._count_to_step(span, int(count)))
        self._changed()

    def _on_delay_step(self, step):
        span = self._delay[1] - self._delay[0]
        count = self._step_to_count(span, step, 8, 1024)
        self._set_spin(self.ndelay, count)
        self._set_spin(self.delay_step, self._count_to_step(span, count))
        self._changed()

    def _on_wl(self, lo, hi):
        self._wl = (lo, hi)
        self._set_spin(
            self.wl_step, self._count_to_step(hi - lo, int(self.nwl.value()))
        )
        self._sync_spec_mirror()
        self._changed()

    def _on_nwl(self, count):
        span = self._wl[1] - self._wl[0]
        self._set_spin(self.wl_step, self._count_to_step(span, int(count)))
        self._sync_spec_mirror()
        self._changed()

    def _on_wl_step(self, step):
        span = self._wl[1] - self._wl[0]
        count = self._step_to_count(span, step, 8, 2048)
        self._set_spin(self.nwl, count)
        self._set_spin(self.wl_step, self._count_to_step(span, count))
        self._sync_spec_mirror()
        self._changed()

    # -- independent-spectrum controls --------------------------------------
    def _on_spec_mirror(self, checked):
        for w in (self.spec_range, self.nspec, self.spec_step):
            w.setEnabled(not checked)
        if checked:
            self._sync_spec_mirror()
        self._changed()

    def _on_spec_range(self, lo, hi):
        self._spec_wl = (lo, hi)
        self._set_spin(
            self.spec_step, self._count_to_step(hi - lo, int(self.nspec.value()))
        )
        self._changed()

    def _on_nspec(self, count):
        span = self._spec_wl[1] - self._spec_wl[0]
        self._set_spin(self.spec_step, self._count_to_step(span, int(count)))
        self._changed()

    def _on_spec_step(self, step):
        span = self._spec_wl[1] - self._spec_wl[0]
        count = self._step_to_count(span, step, 8, 4096)
        self._set_spin(self.nspec, count)
        self._set_spin(self.spec_step, self._count_to_step(span, count))
        self._changed()

    def _sync_spec_mirror(self) -> None:
        """Mirror the trace λ window onto the independent spectrum (when enabled).

        The fundamental spectrum sits at the signal wavelength scaled by the
        interaction's ``omega0_scale`` (×2 for SHG, ×1 for PG/SD), and the point
        count tracks ``# λ points``.
        """
        if not self.spec_mirror.isChecked():
            return
        scale = get_interaction(self.interaction.currentText()).omega0_scale
        lo, hi = self._wl[0] * scale, self._wl[1] * scale
        self._spec_wl = (lo, hi)
        self.spec_range.set_values(lo, hi)  # signal-safe
        self._set_spin(self.nspec, int(self.nwl.value()))
        self._set_spin(
            self.spec_step, self._count_to_step(hi - lo, int(self.nspec.value()))
        )

    # -- physics ------------------------------------------------------------
    def _coeffs(self):
        """Return (fwhm_s, omega0, gdd, tod, fod, material_thicknesses)."""
        fwhm = self.dur.value() * 1e-15
        omega0 = 2 * math.pi * C / (self.wl.value() * 1e-9)
        gdd = self.gdd.value() * 1e-30
        tod = self.tod.value() * 1e-45
        fod = self.fod.value() * 1e-60
        mats = None
        if self.use_material.isChecked() and self.thickness.value() != 0:
            mats = {self.material.currentText(): self.thickness.value() * 1e-3}
        return fwhm, omega0, gdd, tod, fod, mats

    def _build_grid(self, fwhm, gdd):
        """Auto-size an internal grid that resolves the (broadened) pulse.

        ``dt`` sets the spectral bandwidth (fine enough to resolve the
        transform-limited spectrum); the window ``n·dt`` must hold the
        dispersion-broadened pulse and the full delay span.
        """
        dt = max(fwhm / 24.0, 1e-17)
        # Gaussian GDD broadening estimate (TOD/FOD add structure -> generous margin)
        denom = fwhm * fwhm if fwhm > 0 else 1e-30
        tb = fwhm * math.sqrt(1.0 + (4.0 * math.log(2.0) * gdd / denom) ** 2)
        lo, hi = self._delay[0] * 1e-15, self._delay[1] * 1e-15
        delay_span = abs(hi - lo)
        span = max(12.0 * tb, 2.5 * delay_span, 16.0 * fwhm, 60e-15)
        n = 1 << int(math.ceil(math.log2(span / dt)))
        n = int(min(max(n, 256), 8192))
        return Grid(n, dt=dt)

    def _pulse(self):
        fwhm, omega0, gdd, tod, fod, mats = self._coeffs()
        grid = self._build_grid(fwhm, gdd)
        if self.shape.currentText().startswith("Gauss"):
            ew = gaussian_pulse(grid, fwhm)
        else:
            ew = sech_pulse(grid, fwhm)
        ew = dispersion.apply_dispersion(
            grid.omega, omega0, ew, gdd=gdd, tod=tod, fod=fod, material_thicknesses=mats
        )
        return grid, ew, omega0

    def _delays(self):
        lo, hi = self._delay[0] * 1e-15, self._delay[1] * 1e-15
        return np.linspace(lo, hi, int(self.ndelay.value()))

    def _spectrum(self, grid, ew, omega0):
        """Independent fundamental spectrum sampled on the chosen λ grid.

        Returns ``(lam_spec, Ilam_spec)`` in SI (m, normalised to unit peak), or
        ``(None, None)`` when the window holds no spectral power. The intensity is
        a wavelength density ``∝ |E(ω)|²·ω²`` — the same convention as
        :func:`croak.session.pipeline.assemble_load_data`'s ``Ilam_spec`` and the
        retrieved spectral panel — so the synthetic trace exposes an independent
        spectrum the marginal check and SHG marginal correction can consume.
        """
        omega_abs = grid.omega + omega0
        pos = omega_abs > 0
        lam = 2 * math.pi * C / omega_abs[pos]
        Il = (np.abs(ew) ** 2 * omega_abs**2)[pos]
        order = np.argsort(lam)
        lam, Il = lam[order], Il[order]
        lo, hi = self._spec_wl[0] * 1e-9, self._spec_wl[1] * 1e-9
        lam_spec = np.linspace(lo, hi, int(self.nspec.value()))
        Ilam_spec = np.interp(lam_spec, lam, Il, left=0.0, right=0.0)
        m = Ilam_spec.max()
        if m <= 0:
            return None, None
        return lam_spec, Ilam_spec / m

    def _signal_trace(self, grid, ew, omega0):
        """Synthetic trace on the signal-wavelength axis (sorted ascending).

        Returns ``(lam_sig, delays, trace)`` with ``trace`` shape
        ``(len(lam_sig), len(delays))`` on the signal-domain wavelengths.
        """
        delays = self._delays()
        interaction = self.interaction.currentText()
        trace = maketrace(grid.omega, delays, ew, interaction)  # (Nomega, Ndelay)
        scale = get_interaction(interaction).omega0_scale
        omega_abs = grid.omega + omega0 * scale
        pos = omega_abs > 0
        lam = 2 * math.pi * C / omega_abs[pos]
        T = trace[pos]
        order = np.argsort(lam)
        return lam[order], delays, T[order]

    def _add_noise(self, trace):
        add = self.noise_add.value()
        pois = self.noise_pois.value()
        if add <= 0 and pois <= 0:
            return trace
        rng = np.random.default_rng()
        out = np.clip(trace, 0.0, None)
        if pois > 0:
            out = rng.poisson(out * pois).astype(float) / pois
        if add > 0:
            peak = out.max() or 1.0
            out = out + rng.normal(0.0, add * peak, out.shape)
        out = np.clip(out, 0.0, None)
        m = out.max()
        return out / m if m > 0 else out

    # -- preview ------------------------------------------------------------
    def _recompute(self):
        try:
            grid, ew, omega0 = self._pulse()
            lam_sig, delays, trace = self._signal_trace(grid, ew, omega0)
        except Exception as exc:  # pragma: no cover - surfaced in the GUI
            self.set_status(f"Generation failed: {exc}")
            return
        self._last = (grid, ew, omega0)
        # show the trace as it will be handed on (with any noise applied)
        self._draw(grid, ew, omega0, lam_sig, delays, self._add_noise(trace))
        self.set_status("Adjust parameters, then Generate to run retrieval.")

    def _draw(self, grid, ew, omega0, lam_sig, delays, trace):
        def plot(fig):
            axd = fig.subplot_mosaic("ab\ncd")
            self._plot_temporal(axd["a"], grid, ew)
            self._plot_spectral(axd["b"], grid, ew, omega0)
            self._plot_spectrogram(axd["c"], grid, ew, omega0)
            self._plot_frog(axd["d"], lam_sig, delays, trace)

        self.canvas.render(plot)

    @staticmethod
    def _masked_phase(intensity, field, frac=1e-3):
        phase = np.unwrap(np.angle(field))
        mask = intensity < intensity.max() * frac
        phase = phase - phase[int(np.argmax(intensity))]
        phase[mask] = np.nan
        return phase

    def _plot_temporal(self, ax, grid, ew):
        Et = grid.ifft(ew)
        inten = np.abs(Et) ** 2
        inten = inten / inten.max() if inten.max() > 0 else inten
        t = grid.t / 1e-15
        ax.plot(t, inten, c="C0")
        axp = ax.twinx()
        axp.plot(t, self._masked_phase(inten, Et), c="C1", lw=1)
        axp.set_ylabel("Phase (rad)", color="C1")
        # zoom to the pulse
        sig = t[inten > 1e-3]
        if sig.size:
            pad = 0.2 * (sig.max() - sig.min() + 1e-9)
            ax.set_xlim(sig.min() - pad, sig.max() + pad)
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Intensity")
        ax.set_title("Temporal")

    def _plot_spectral(self, ax, grid, ew, omega0):
        inten = np.abs(ew) ** 2
        inten = inten / inten.max() if inten.max() > 0 else inten
        omega_abs = grid.omega + omega0
        pos = omega_abs > 0
        lam = 2 * math.pi * C / omega_abs[pos] / 1e-9
        order = np.argsort(lam)
        lam = lam[order]
        Ip = inten[pos][order]
        phase = self._masked_phase(inten, ew)[pos][order]
        ax.plot(lam, Ip, c="C0")
        axp = ax.twinx()
        axp.plot(lam, phase, c="C1", lw=1)
        axp.set_ylabel("Phase (rad)", color="C1")
        sig = lam[Ip > 1e-3]
        if sig.size:
            pad = 0.2 * (sig.max() - sig.min() + 1e-9)
            ax.set_xlim(sig.min() - pad, sig.max() + pad)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.set_title("Spectral")

    def _plot_spectrogram(self, ax, grid, ew, omega0, n_t=160):
        """A Gabor spectrogram of the field (time × wavelength)."""
        Et = grid.ifft(ew)
        t = grid.t
        inten = np.abs(Et) ** 2
        sig = t[inten > inten.max() * 1e-3] if inten.max() > 0 else t
        tmin = sig.min() - 0.3 * (sig.max() - sig.min() + 1e-15)
        tmax = sig.max() + 0.3 * (sig.max() - sig.min() + 1e-15)
        tcs = np.linspace(tmin, tmax, n_t)
        fw = max((tmax - tmin) / 30.0, grid.dt * 4)
        S = np.empty((grid.n, n_t))
        for i, tc in enumerate(tcs):
            gate = np.exp(-0.5 * ((t - tc) / fw) ** 2)
            sp = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(Et * gate)))
            S[:, i] = np.abs(sp) ** 2
        omega_abs = grid.omega + omega0
        pos = omega_abs > 0
        S = S[pos] * (omega_abs[pos] ** 2)[:, None]
        lam = 2 * math.pi * C / omega_abs[pos] / 1e-9
        order = np.argsort(lam)
        S = S[order]
        lam = lam[order]
        S = S / S.max() if S.max() > 0 else S
        ax.pcolormesh(tcs / 1e-15, lam, S, shading="auto")
        lam_sig = lam[(S.max(axis=1) > 1e-2)]
        if lam_sig.size:
            ax.set_ylim(lam_sig.min(), lam_sig.max())
        ax.set_xlabel("Time (fs)")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_title("Spectrogram")

    def _plot_frog(self, ax, lam_sig, delays, trace):
        ax.pcolormesh(delays / 1e-15, lam_sig / 1e-9, trace, shading="auto")
        sig = lam_sig[(trace.max(axis=1) > 1e-2)] / 1e-9
        if sig.size:
            ax.set_ylim(sig.min(), sig.max())
        ax.set_xlabel("Delay (fs)")
        ax.set_ylabel("Wavelength (nm)")
        ax.set_title("Synthetic FROG trace")

    # -- generate (the entry-page forward action) ---------------------------
    def next_label(self) -> str:
        return "Generate →"

    def advance(self):
        """Build the synthetic trace from the current settings and move on.

        Wired to the wizard footer's forward button. On success the trace (and the
        generating field, as the known ground truth) is put on the shared state and
        the wizard advances to the marginal-check stage; on failure the reason is
        shown in the status label and the page stays put.
        """
        try:
            grid, ew, omega0 = self._pulse()
            lam_sig, delays, trace = self._signal_trace(grid, ew, omega0)
        except Exception as exc:
            self.set_status(f"Generation failed: {exc}")
            return
        interaction = self.interaction.currentText()
        # sample onto the user's measurement wavelength axis (emulate a spectrometer)
        lo, hi = self._wl[0] * 1e-9, self._wl[1] * 1e-9
        lam_user = np.linspace(lo, hi, int(self.nwl.value()))
        meas = np.empty((lam_user.size, delays.size))
        for j in range(delays.size):
            meas[:, j] = np.interp(lam_user, lam_sig, trace[:, j], left=0.0, right=0.0)
        meas = self._add_noise(meas)
        m = meas.max()
        if m <= 0:
            self.set_status("Trace is empty in the chosen λ window; widen the range.")
            return
        meas = meas / m

        lam_spec, Ilam_spec = self._spectrum(grid, ew, omega0)

        data = dict(
            lam=lam_user,
            scanaxis=delays,
            input_unit="delay",
            trace=meas,
            lam_spec=lam_spec,
            Ilam_spec=Ilam_spec,
            interaction=interaction,
            # known ground truth for the retrieval screen + saved output
            truth=TruthPulse.from_spectrum(grid, ew, omega0),
        )
        self.state.load.interaction = interaction
        self.state.set_load_data(data)
        seed_preproc_defaults(self.state)
        self.wizard.enter_marginal_from_synthetic()
