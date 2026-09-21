"""Smoke tests for :mod:`croak.plotting` (Agg backend, no display)."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from croak import plotting, preprocess, processing
from croak.forward import maketrace
from croak.grid import Grid
from croak.maths import wlfreq
from croak.pulses import gaussian_pulse
from croak.retrieve import retrieve


@pytest.fixture
def result_and_trace():
    g = Grid(96, dt=0.5e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 7e-15, phases=[20e-30])
    delays = np.linspace(-50e-15, 50e-15, 60)
    trace = maketrace(g.omega, delays, ew, "pg")
    res = retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        maxiters=60,
        guess=ew,
        omega0=omega0,
    )
    return res, trace


def test_colormaps():
    assert plotting.cmap_white("viridis").N == 512
    cm = plotting.cmap_negwhite("viridis", negfrac=0.1)
    assert cm.N == 512


def test_plot_retrieval_panel_count(result_and_trace):
    res, trace = result_and_trace
    fig = plotting.plot_retrieval(res, measured=trace, lam_min=700e-9, lam_max=950e-9)
    # 12-panel mosaic
    assert len(fig.axes) >= 12
    fig.clf()


def test_plot_retrieval_flim_bounds_the_trace_frequency_axis(result_and_trace):
    """``flim`` (Hz) crops the trace/residual panels; the default shows the grid."""
    res, trace = result_and_trace
    f0 = res.omega0 / (2 * np.pi)
    lo, hi = f0 - 0.05e15, f0 + 0.05e15

    unbounded = plotting.plot_retrieval(res, measured=trace)
    wide = [ax.get_ylim() for ax in unbounded.axes]
    unbounded.clf()

    fig = plotting.plot_retrieval(res, measured=trace, flim=(lo, hi))
    # The limits are in PHz on the axis, i.e. Hz / 1e15 — at least one panel must
    # now carry exactly the requested window, and none may be wider than before.
    limits = [ax.get_ylim() for ax in fig.axes]
    assert any(
        np.isclose(a, lo / 1e15) and np.isclose(b, hi / 1e15) for a, b in limits
    ), f"no panel bounded to {(lo / 1e15, hi / 1e15)}; got {limits}"
    assert max(b - a for a, b in limits) <= max(b - a for a, b in wide)
    fig.clf()


def test_plot_simulated_trace(result_and_trace):
    res, trace = result_and_trace
    omega0_trace = res.omega0
    fig = plotting.plot_simulated_trace(res.omega, res.delays, trace, omega0_trace)
    assert len(fig.axes) >= 4


def test_plot_frog_filter(experiment):
    exp = experiment
    td = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
    )
    fig = plotting.plot_frog_filter(td)
    assert len(fig.axes) >= 6


def test_frog_filter_view_reuses_artists(experiment):
    """FrogFilterView.update mutates the existing images, not rebuild them."""
    from matplotlib.figure import Figure

    exp = experiment
    common = dict(
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
    )
    td1 = preprocess.load_and_clean(
        exp.trace, exp.lam_nm * 1e-9, exp.delay_fs * 1e-15, exp.interaction, **common
    )
    # a different crop -> different delay count, exercising set_extent/shape change
    td2 = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        tau_crop=(-30e-15, 30e-15),
        **common,
    )

    fig = Figure()
    view = plotting.FrogFilterView()
    view.attach(fig)
    view.update(td1)
    images_before = {k: (id(p), id(n)) for k, (p, n) in view.images.items()}
    n_axes = len(fig.axes)

    view.update(td2)  # must not recreate axes/images/colorbars
    images_after = {k: (id(p), id(n)) for k, (p, n) in view.images.items()}
    assert images_before == images_after
    assert len(fig.axes) == n_axes
    # the regridded panel's data tracks the new (cropped) delay count
    assert view.images["e"][0].get_array().shape[0] == td2.delays.size
    assert td2.delays.size != td1.delays.size  # the crop really changed it


def test_single_axis_plotters(result_and_trace):
    from matplotlib.figure import Figure

    from croak.processing import process_result

    res, trace = result_and_trace
    pr = process_result(res, measured=trace)
    fig = Figure()
    ax = fig.add_subplot(111)
    plotting.plot_temporal(ax, pr)
    fig2 = Figure()
    plotting.plot_spectral(fig2.add_subplot(111), pr)
    fig3 = Figure()
    plotting.plot_convergence(fig3.add_subplot(111), res.errors)
    assert True  # rendered without error


def test_plot_temporal_absolute_power_axis(result_and_trace):
    from matplotlib.figure import Figure

    from croak.processing import process_result

    res, trace = result_and_trace
    # without energy: normalised axis
    pr = process_result(res, measured=trace)
    ax = Figure().add_subplot(111)
    plotting.plot_temporal(ax, pr)
    assert ax.get_ylabel() == "Power (a.u.)"
    # with energy: SI-prefixed absolute power axis
    pr_e = process_result(res, measured=trace, energy=100e-6)
    ax_e = Figure().add_subplot(111)
    plotting.plot_temporal(ax_e, pr_e)
    assert ax_e.get_ylabel().startswith("Power (") and "W)" in ax_e.get_ylabel()
    assert ax_e.get_ylabel() != "Power (a.u.)"


def test_si_power_prefixes():
    assert plotting.si_power(1.2e7) == (1e6, "MW")
    assert plotting.si_power(2.0e12) == (1e12, "TW")
    assert plotting.si_power(0.0) == (1.0, "W")  # unspecified falls back to W


def test_resample_uniform_x_leaves_uniform_axis_unchanged():
    x = np.linspace(200.0, 800.0, 64)
    C = np.random.default_rng(0).random((10, 64))
    xu, Cu = plotting._resample_uniform_x(x, C)
    np.testing.assert_allclose(xu, x)
    np.testing.assert_allclose(Cu, C)


def test_resample_uniform_x_places_feature_at_true_wavelength():
    """A λ axis from a uniform ω grid must not shift a feature's wavelength.

    Regression for the simulated-trace preprocess panel painting a 260 nm signal
    near 480 nm: imshow's linear extent assumes even spacing, so the strongly
    non-uniform (ω-uniform) λ axis must be resampled first.
    """
    # uniform angular-frequency grid spanning 150–800 nm; descending ω gives an
    # ascending (non-uniform) wavelength axis, as the simulated loader yields.
    omega = np.linspace(wlfreq(150e-9), wlfreq(800e-9), 400)
    lam_nm = wlfreq(omega) / 1e-9  # non-uniform, ascending (150 → 800 nm)
    peak_nm = 260.0
    row = np.exp(-0.5 * ((lam_nm - peak_nm) / 5.0) ** 2)  # narrow feature at 260 nm
    C = np.tile(row, (8, 1))  # (ndelay, nlam)

    xu, Cu = plotting._resample_uniform_x(lam_nm, C)
    # on the uniform display axis the peak column maps linearly to its wavelength
    j = int(np.argmax(Cu.sum(axis=0)))
    wl = xu[0] + j / (xu.size - 1) * (xu[-1] - xu[0])
    assert wl == pytest.approx(peak_nm, abs=3.0)
    # the raw (non-uniform) axis would have mis-placed it far to the red
    i = int(np.argmax(C.sum(axis=0)))
    wl_raw = lam_nm[0] + i / (lam_nm.size - 1) * (lam_nm[-1] - lam_nm[0])
    assert wl_raw > 400.0  # the bug being fixed


def test_retrieval_annotations_carry_expected_precision(result_and_trace):
    """The error is annotated to 2 dp (%) and GDD/TOD to 1 dp.

    These read-outs are how a retrieval is judged by eye, so the digit counts are
    part of the contract: 0.1% steps on the error hid genuine differences between
    runs, and whole-fs² GDD hid the effect of small dispersion tweaks.
    """
    import re

    res, trace = result_and_trace
    fig = plotting.plot_retrieval(res, measured=trace, lam_min=700e-9, lam_max=950e-9)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]

    err = next(
        t for t in texts if t.endswith(f"{res.trace.shape[0]}×{res.trace.shape[1]}")
    )
    assert re.fullmatch(r"-?\d+\.\d{2}%", err.splitlines()[0]), err

    disp = next(t for t in texts if t.startswith("GDD:"))
    gdd, tod = disp.splitlines()
    assert re.fullmatch(r"GDD: -?\d+\.\d fs²", gdd), gdd
    assert re.fullmatch(r"TOD: -?\d+\.\d fs³", tod), tod
    fig.clf()


def test_truth_overlay_legend_reports_its_fwhm(result_and_trace):
    """The ground-truth overlay carries its FWHM, like the R and TL entries."""
    import re

    from croak.processing import TruthPulse

    res, trace = result_and_trace
    g = Grid(96, dt=0.5e-15)
    truth = TruthPulse.from_spectrum(g, gaussian_pulse(g, 7e-15), float(wlfreq(800e-9)))
    fig = plotting.plot_retrieval(
        res, measured=trace, lam_min=700e-9, lam_max=950e-9, truth=truth
    )
    # panel "e" is the temporal one: it holds the R/TL curves and the truth overlay
    labels = [
        t.get_text()
        for ax in fig.axes
        if ax.get_legend() is not None
        for t in ax.get_legend().get_texts()
    ]
    truth_label = next(t for t in labels if t.startswith("truth"))
    # three significant digits (see plotting._FWHM_FMT)
    assert re.fullmatch(r"truth \(\d+(\.\d+)? fs\)", truth_label), truth_label
    # and it agrees with the truth's own FWHM
    assert float(truth_label.split("(")[1].split()[0]) == pytest.approx(
        truth.fwhm / 1e-15, abs=0.05
    )
    fig.clf()


def test_truth_phase_lines_are_dotted_and_listed_in_both_legends(result_and_trace):
    """Known temporal and spectral phases are visible and named in each panel."""
    res, trace = result_and_trace
    g = Grid(96, dt=0.5e-15)
    truth = processing.TruthPulse.from_spectrum(
        g,
        gaussian_pulse(g, 7e-15, phases=[8e-30]),
        float(wlfreq(800e-9)),
    )
    fig = plotting.plot_retrieval(
        res, measured=trace, lam_min=700e-9, lam_max=950e-9, truth=truth
    )
    panels = [
        ax for ax in fig.axes if ax.get_title() in {"Retrieved pulse", "Spectrum"}
    ]
    assert len(panels) == 2
    for panel in panels:
        labels = [text.get_text() for text in panel.get_legend().get_texts()]
        assert "truth phase" in labels
        phase_axis = next(
            ax
            for ax in fig.axes
            if ax is not panel
            and ax.get_position().bounds == pytest.approx(panel.get_position().bounds)
            and any(line.get_label() == "truth phase" for line in ax.get_lines())
        )
        line = next(
            line for line in phase_axis.get_lines() if line.get_label() == "truth phase"
        )
        assert line.get_linestyle() == ":"
    fig.clf()


def test_sig3_formats_three_significant_digits():
    """Trailing zeros kept, no exponent — the read-outs must line up in a column."""
    assert [plotting.sig3(v) for v in (9.512, 7.0, 0.85, 953.2, 1234.0)] == [
        "9.51",
        "7.00",
        "0.850",
        "953",
        "1234",
    ]
    # _fs is the femtosecond wrapper the legends use; it must not drift from sig3
    assert plotting._fs(9.512e-15) == plotting.sig3(9.512)
    assert plotting.sig3(float("nan")) == "nan"


def test_plot_retrieval_reuses_a_precomputed_processed_result(result_and_trace):
    """Passing ``processed=`` skips the internal process_result entirely."""
    res, trace = result_and_trace
    pr = processing.process_result(res, measured=trace)

    calls = []
    real = plotting.process_result

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    plotting.process_result = counting
    try:
        fig = plotting.plot_retrieval(
            res, measured=trace, lam_min=700e-9, lam_max=950e-9, processed=pr
        )
        assert not calls
        assert len(fig.axes) >= 12
        fig.clf()
        # ...and it is still computed when not supplied
        fig = plotting.plot_retrieval(
            res, measured=trace, lam_min=700e-9, lam_max=950e-9
        )
        assert len(calls) == 1
        fig.clf()
    finally:
        plotting.process_result = real


def test_plot_retrieval_rejects_a_processed_result_from_another_retrieval(
    result_and_trace,
):
    """Guards against drawing trace panels and pulse panels from different runs."""
    res, trace = result_and_trace
    other = processing.post_filter(res, lam_lims=(700e-9, 900e-9))
    pr_other = processing.process_result(other, measured=trace)

    with pytest.raises(ValueError, match="ProcessedResult of this result"):
        plotting.plot_retrieval(res, measured=trace, processed=pr_other)


def test_truth_overlay_uses_the_shared_peak_power(result_and_trace):
    """The truth line's absolute scale comes from TruthPulse.peak_power."""
    res, trace = result_and_trace
    g = Grid(96, dt=0.5e-15)
    truth = processing.TruthPulse.from_spectrum(
        g, gaussian_pulse(g, 7e-15), float(wlfreq(800e-9))
    )
    energy = 2e-6
    fig = plotting.plot_retrieval(
        res,
        measured=trace,
        lam_min=700e-9,
        lam_max=950e-9,
        truth=truth,
        energy=energy,
    )
    ax = next(a for a in fig.axes if a.get_title() == "Retrieved pulse")
    line = next(ln for ln in ax.get_lines() if ln.get_label().startswith("truth"))
    scale, _unit = plotting.si_power(
        max(processing.process_result(res, measured=trace, energy=energy).peak_power, 0)
    )
    expected = truth.peak_power(energy) / scale
    assert line.get_ydata().max() == pytest.approx(expected, rel=1e-9)
    fig.clf()


def test_filter_view_scale_ignores_out_of_band_negatives(experiment):
    """A big out-of-band negative must not set the linear panels' positive scale.

    Regression: the scale was ``max|·|`` over the three arrays, so an absolute
    calibration curve amplifying background-subtracted noise outside the signal
    band (measured: −5.7 at 1097 nm against a signal peak of +1) squashed every
    linear panel's signal into the bottom 18 % of the colour range.
    """
    from dataclasses import replace as dc_replace

    exp = experiment
    td = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        lamm_lims=(exp.lam_min, exp.lam_max),
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
    )
    assert plotting.filter_view_vmax(td) == pytest.approx(1.0, rel=1e-9)

    # plant a large negative outside lamm_lims, as the calibration curve does
    meas = np.array(td.Ifrog_meas, dtype=float)
    out_of_band = np.nonzero(td.lam_frog > exp.lam_max)[0]
    assert out_of_band.size, "fixture must have rows outside the measurement window"
    meas[out_of_band[-1], :] = -7.0
    spoiled = dc_replace(td, Ifrog_meas=meas)

    assert np.nanmax(np.abs(spoiled.Ifrog_meas)) == pytest.approx(7.0)
    assert plotting.filter_view_vmax(spoiled) == pytest.approx(1.0, rel=1e-9)


def test_filter_view_clims_follow_the_positive_scale(experiment):
    """The linear panels take the positive scale; the dB row stays fixed."""
    from matplotlib.figure import Figure

    exp = experiment
    td = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
        filter_fringes=False,
        dtau_min=None,
    )
    view = plotting.FrogFilterView(tracedb=30.0)
    view.attach(Figure())
    view.update(td)
    vmax = plotting.filter_view_vmax(td)
    for key in ("c", "d", "f"):  # the linear row
        for image in view.images[key]:
            assert image.get_clim() == pytest.approx((0.0, vmax))
    for key in ("a", "b", "e"):  # the dB row is on its own fixed scale
        for image in view.images[key]:
            assert image.get_clim() == pytest.approx((-30.0, 0.0))


def test_filter_view_vmax_falls_back_when_nothing_is_positive():
    """An all-negative trace still gets a usable, positive scale."""
    from dataclasses import dataclass

    @dataclass
    class _Stub:
        Ifrog_meas: np.ndarray
        Ifrog_filt: np.ndarray
        trace: np.ndarray
        lam_frog: np.ndarray
        lamm_lims: tuple[float, float] | None

    lam = np.linspace(200e-9, 400e-9, 8)
    neg = -np.ones((8, 4))
    stub = _Stub(neg, neg, neg, lam, None)
    assert plotting.filter_view_vmax(stub) == pytest.approx(1.0)
    stub_zero = _Stub(np.zeros((8, 4)), np.zeros((8, 4)), np.zeros((8, 4)), lam, None)
    assert plotting.filter_view_vmax(stub_zero) == pytest.approx(1.0)


def test_plot_convergence_marks_a_stage_handover():
    """A spliced two-stage curve gets a rule where the algorithm changed.

    ``warm-lbfgs`` counts COPRA iterations before the join and L-BFGS function
    evaluations after it; unmarked, the kink there reads as convergence
    behaviour rather than as a change of algorithm.
    """
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot(111)
    plotting.plot_convergence(ax, [0.5, 0.3, 0.2, 0.1, 0.05], boundaries=[3])
    lines = [ln.get_xdata()[0] for ln in ax.lines if ln.get_linestyle() == "--"]
    assert lines == [3.5]  # between the third and fourth points
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "stage handover" in labels
    fig.clf()


def test_plot_convergence_ignores_out_of_range_boundaries():
    """A boundary at or past the end of the log draws nothing (and no legend)."""
    from matplotlib.figure import Figure

    fig = Figure()
    ax = fig.add_subplot(111)
    plotting.plot_convergence(ax, [0.5, 0.3], boundaries=[0, 2, 9])
    assert not [ln for ln in ax.lines if ln.get_linestyle() == "--"]
    assert ax.get_legend() is None
    fig.clf()


# -- on-screen density -------------------------------------------------------
@pytest.mark.parametrize(
    ("width", "height", "design"),
    [(15.45, 7.33, (12.0, 7.0)), (13.53, 7.70, (12.0, 7.0)), (9.73, 5.73, (5.0, 4.0))],
)
def test_plot_scale_is_full_at_or_above_the_design_size(width, height, design):
    """A canvas at or above its design size keeps matplotlib's default text size."""
    assert plotting.plot_scale(width, height, design) == plotting.FULL_SCALE


@pytest.mark.parametrize(
    ("width", "height", "design", "expected"),
    [
        (9.91, 4.21, (12.0, 7.0), 0.7),  # Retrieve canvas at 1366x768 (clamped)
        (9.91, 5.91, (9.0, 7.0), 0.85),  # the 2x2 stages at 1366x768
        (9.73, 2.64, (5.0, 4.0), 0.7),  # the stacked marginal-check pair
        (9.91, 5.91, (11.0, 4.5), 0.9),  # Uncertainty at 1366x768
    ],
)
def test_plot_scale_follows_the_tighter_dimension_and_clamps(
    width, height, design, expected
):
    assert plotting.plot_scale(width, height, design).font == pytest.approx(expected)


def test_plot_scale_is_quantised_to_the_step():
    """0.883 of the design height rounds to the nearest 0.05 step, 0.9."""
    font = plotting.plot_scale(10.6, 7.0, (12.0, 7.0)).font
    assert font == pytest.approx(0.9)
    steps = font / plotting.FONT_SCALE_STEP
    assert steps == pytest.approx(round(steps))


@pytest.mark.parametrize(
    ("width", "height", "design"),
    [(0.0, 4.0, (5.0, 4.0)), (5.0, 4.0, (5.0, 0.0)), (-1.0, 4.0, (5.0, 4.0))],
)
def test_plot_scale_rejects_non_positive_sizes(width, height, design):
    with pytest.raises(ValueError, match="must be positive"):
        plotting.plot_scale(width, height, design)


def test_plot_scale_compact_flag():
    assert plotting.PlotScale(0.85).compact
    assert not plotting.PlotScale(0.9).compact
    assert not plotting.FULL_SCALE.compact


def test_scaled_rc_keys_are_valid_and_scale_uniformly():
    """Every key is a real rc key, values scale together, full scale = defaults."""
    from matplotlib.font_manager import FontProperties

    small = plotting.scaled_rc(plotting.PlotScale(0.7))
    full = plotting.scaled_rc(plotting.FULL_SCALE)
    with matplotlib.rc_context(small):  # unknown keys would raise KeyError
        pass
    for key, value in small.items():
        assert value == pytest.approx(0.7 * full[key])
    for key, value in full.items():
        default = matplotlib.rcParamsDefault[key]
        if isinstance(default, str):  # 'large'/'medium' resolve against 10 pt
            default = FontProperties(size=default).get_size_in_points()
        assert value == pytest.approx(default)


def test_scaled_rc_sizes_artists_created_inside_the_context():
    """rc sizes bind at artist creation: inside the context text is scaled, and an
    axes created afterwards is back at the defaults even on the same figure."""
    from matplotlib.figure import Figure

    with matplotlib.rc_context(plotting.scaled_rc(plotting.PlotScale(0.7))):
        fig = Figure()
        ax = fig.add_subplot()
        ax.plot([0.0, 1.0], [0.0, 1.0], label="x")
        ax.set_title("t")
        ax.set_xlabel("x")
        legend = ax.legend()
    fig.canvas.draw()  # drawing later, outside the context, does not undo it
    assert ax.title.get_fontsize() == pytest.approx(8.4)
    assert ax.xaxis.label.get_fontsize() == pytest.approx(7.0)
    assert ax.xaxis.get_majorticklabels()[0].get_fontsize() == pytest.approx(7.0)
    assert legend.get_texts()[0].get_fontsize() == pytest.approx(7.0)
    late = fig.add_subplot()
    late.set_title("late")
    assert late.title.get_fontsize() == pytest.approx(12.0)


# -- retrieval panels and pages ----------------------------------------------------
def _plot_data(res, trace, **kw):
    return plotting.retrieval_plot_data(
        res, measured=trace, lam_min=700e-9, lam_max=950e-9, **kw
    )


def test_retrieval_registry_pages_partition_the_panels():
    """Every panel is on exactly one page and once in the overview; colorbar
    pairs never straddle a page."""
    keys = set(plotting.RETRIEVAL_PANELS)
    assert len(keys) == 12
    page_keys = [k for page in plotting.RETRIEVAL_PAGES.values() for k in page.keys]
    assert sorted(page_keys) == sorted(keys)
    assert sorted(plotting.RETRIEVAL_OVERVIEW.keys) == sorted(keys)
    assert [len(row) for row in plotting.RETRIEVAL_OVERVIEW.mosaic] == [4, 4, 4]
    for page in plotting.RETRIEVAL_PAGES.values():
        assert [len(row) for row in page.mosaic] == [2, 2]
    for cbar in plotting.RETRIEVAL_COLORBARS:
        assert any(
            set(cbar.panels) <= set(page.keys)
            for page in plotting.RETRIEVAL_PAGES.values()
        ), cbar
    for key, spec in plotting.RETRIEVAL_PANELS.items():
        assert spec.key == key and spec.title


def test_plot_retrieval_titles_match_the_registry(result_and_trace):
    res, trace = result_and_trace
    fig = plotting.plot_retrieval(res, measured=trace, lam_min=700e-9, lam_max=950e-9)
    titles = sorted(ax.get_title() for ax in fig.axes if ax.get_title())
    assert titles == sorted(s.title for s in plotting.RETRIEVAL_PANELS.values())


@pytest.mark.parametrize("key", ["traces", "pulse", "diagnostics"])
def test_plot_retrieval_page_draws_the_page_in_mosaic_order(result_and_trace, key):
    res, trace = result_and_trace
    page = plotting.RETRIEVAL_PAGES[key]
    fig = plotting.plot_retrieval_page(_plot_data(res, trace), page)
    titled = [ax.get_title() for ax in fig.axes if ax.get_title()]
    assert titled == [plotting.RETRIEVAL_PANELS[k].title for k in page.keys]
    colorbars = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    if key == "traces":
        assert len(colorbars) >= 2  # one per trace pair, plus any negative bars
    elif key == "pulse":
        assert not colorbars
        assert len(fig.axes) == 6  # four panels and the two phase twins
    else:
        assert len(colorbars) == 1  # the residual's


def test_each_panel_alone_matches_its_overview_twin(result_and_trace):
    """A panel drawn on its own carries the same title and axis labels as in
    the full figure — the pages show exactly what the overview shows."""
    from matplotlib.figure import Figure

    res, trace = result_and_trace
    data = _plot_data(res, trace)
    full = plotting.plot_retrieval_page(data, plotting.RETRIEVAL_OVERVIEW)
    by_title = {ax.get_title(): ax for ax in full.axes if ax.get_title()}
    for spec in plotting.RETRIEVAL_PANELS.values():
        ax = Figure().add_subplot()
        spec.draw(ax, data)
        twin = by_title[spec.title]
        assert ax.get_title() == spec.title, spec.key
        assert (ax.get_xlabel(), ax.get_ylabel()) == (
            twin.get_xlabel(),
            twin.get_ylabel(),
        ), spec.key


def test_retrieval_plot_data_validates_like_plot_retrieval(result_and_trace):
    from dataclasses import replace

    res, trace = result_and_trace
    with pytest.raises(ValueError, match="carrying a simulated trace"):
        plotting.retrieval_plot_data(replace(res, trace=None))
    other = replace(res)  # a different object: its ProcessedResult is not ours
    pr_other = processing.process_result(other, measured=trace)
    with pytest.raises(ValueError, match="ProcessedResult of this result"):
        plotting.retrieval_plot_data(res, measured=trace, processed=pr_other)
    data = plotting.retrieval_plot_data(res, measured=trace)
    lams = wlfreq(res.omega + res.omega0)
    assert data.lam_min == pytest.approx(float(lams.min()))
    assert data.lam_max == pytest.approx(float(lams.max()))
    assert data.power_scale is None
    with_energy = plotting.retrieval_plot_data(res, measured=trace, energy=1e-6)
    assert with_energy.processed.peak_power is not None
    assert with_energy.power_scale is not None


def test_spectrogram_is_lazy_and_cached(result_and_trace, monkeypatch):
    """The Gabor transform runs only for a page that shows it, and only once."""
    res, trace = result_and_trace
    calls = []
    real = plotting.spectrogram

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(plotting, "spectrogram", counted)
    data = _plot_data(res, trace)
    assert calls == []
    plotting.plot_retrieval_page(data, plotting.RETRIEVAL_PAGES["traces"])
    plotting.plot_retrieval_page(data, plotting.RETRIEVAL_PAGES["pulse"])
    assert calls == []
    plotting.plot_retrieval_page(data, plotting.RETRIEVAL_PAGES["diagnostics"])
    assert len(calls) == 1
    plotting.plot_retrieval_page(data, plotting.RETRIEVAL_PAGES["diagnostics"])
    assert len(calls) == 1


def test_truth_overlay_survives_the_split(result_and_trace):
    """Truth intensity and phase reach the Pulse page and a lone temporal panel."""
    from matplotlib.figure import Figure

    from croak.processing import TruthPulse

    res, trace = result_and_trace
    g = Grid(96, dt=0.5e-15)
    truth = TruthPulse.from_spectrum(g, gaussian_pulse(g, 7e-15), float(wlfreq(800e-9)))
    data = _plot_data(res, trace, truth=truth)
    fig = plotting.plot_retrieval_page(data, plotting.RETRIEVAL_PAGES["pulse"])
    labels = {
        ax.get_title(): [t.get_text() for t in ax.get_legend().get_texts()]
        for ax in fig.axes
        if ax.get_legend() is not None
    }
    assert any(t.startswith("truth (") for t in labels["Retrieved pulse"])
    assert "truth phase" in labels["Retrieved pulse"]
    assert "truth" in labels["Spectrum"]
    assert "truth phase" in labels["Spectrum"]
    alone = plotting.draw_temporal(Figure().add_subplot(), data)
    assert alone.line_truth is not None
    assert alone.line_truth_phase is not None
    assert alone.legend_retr.get_text().startswith("R (")
    assert alone.legend_tl.get_text().startswith("TL (")


def test_plot_convergence_returns_its_artists_even_when_empty():
    from matplotlib.figure import Figure

    artists = plotting.plot_convergence(Figure().add_subplot(), [])
    assert artists.line.get_xydata().shape == (0, 2)
    assert artists.boundaries == ()
    assert artists.legend is None
    with_rule = plotting.plot_convergence(
        Figure().add_subplot(), [1.0, 0.5, 0.2, 0.1], boundaries=[2]
    )
    assert len(with_rule.boundaries) == 1
    assert with_rule.legend is not None


# -- in-place updates ----------------------------------------------------------------
def _retrieve_on(n_delays, maxiters):
    g = Grid(96, dt=0.5e-15)
    omega0 = wlfreq(800e-9)
    ew = gaussian_pulse(g, 7e-15, phases=[20e-30])
    delays = np.linspace(-50e-15, 50e-15, n_delays)
    trace = maketrace(g.omega, delays, ew, "pg")
    res = retrieve(
        trace,
        g.omega,
        delays,
        "pg",
        algorithm="copra",
        maxiters=maxiters,
        guess=ew,
        omega0=omega0,
    )
    return res, trace


@pytest.fixture(scope="module")
def update_results():
    """Two retrievals on one grid (same shapes, different values) and a third on
    another delay grid (different shapes)."""
    r1, trace = _retrieve_on(60, 60)
    r2, _ = _retrieve_on(60, 15)
    r3, trace3 = _retrieve_on(40, 15)
    return (r1, trace), (r2, trace), (r3, trace3)


def _artist_state(value):
    from matplotlib.axes import Axes
    from matplotlib.collections import QuadMesh
    from matplotlib.colors import to_rgba
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    if value is None:
        return None
    if isinstance(value, tuple):
        return [_artist_state(v) for v in value]
    if isinstance(value, Line2D):
        return value.get_xydata()
    if isinstance(value, QuadMesh):
        arr = np.ma.filled(np.ma.asarray(value.get_array(), dtype=float), np.nan)
        return [arr, value.get_clim()]
    if isinstance(value, Text):
        return [value.get_text(), to_rgba(value.get_color())]
    if isinstance(value, Legend):
        return [t.get_text() for t in value.get_texts()]
    if isinstance(value, Axes):
        return [value.get_xlim(), value.get_ylim()]
    raise TypeError(type(value))


def _panel_state(artists, ax):
    """Everything a viewer sees: axes limits/labels plus every artist's data."""
    from dataclasses import fields

    state = {
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
        "ylabel": ax.get_ylabel(),
        "title": ax.get_title(),
    }
    for f in fields(artists):
        state[f.name] = _artist_state(getattr(artists, f.name))
    return state


def _assert_equal(x, y, where):
    if isinstance(x, np.ndarray):
        np.testing.assert_allclose(x, np.asarray(y), rtol=1e-12, atol=0, err_msg=where)
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), where
        for i, (p, q) in enumerate(zip(x, y, strict=True)):
            _assert_equal(p, q, f"{where}[{i}]")
    else:
        assert x == y, (where, x, y)


@pytest.mark.parametrize("key", list(plotting.RETRIEVAL_PANELS))
def test_update_panel_matches_a_fresh_draw(update_results, key):
    """update(draw(d1), d2) shows exactly what draw(d2) shows, on the same artists."""
    from matplotlib.figure import Figure

    (r1, trace), (r2, _), _ = update_results
    d1 = _plot_data(r1, trace)
    d2 = _plot_data(r2, trace)
    spec = plotting.RETRIEVAL_PANELS[key]
    fig_u = Figure()
    ax_u = fig_u.add_subplot()
    artists = spec.draw(ax_u, d1)
    ids = [id(a) for a in fig_u.axes] + [id(v) for v in vars(artists).values()]
    spec.update(artists, d2)
    assert [id(a) for a in fig_u.axes] + [id(v) for v in vars(artists).values()] == ids
    fig_f = Figure()
    ax_f = fig_f.add_subplot()
    fresh = spec.draw(ax_f, d2)
    for k, v in _panel_state(fresh, ax_f).items():
        _assert_equal(_panel_state(artists, ax_u)[k], v, f"{key}.{k}")


def test_retrieval_legends_sit_in_fixed_corners(result_and_trace):
    """No retrieval-panel legend uses loc="best", which jumps between live frames."""
    res, trace = result_and_trace
    fig = plotting.plot_retrieval(res, measured=trace, lam_min=700e-9, lam_max=950e-9)
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    assert legends
    for legend in legends:
        assert legend._loc != 0, legend  # 0 is matplotlib's code for "best"


def test_update_temporal_legend_follows_the_fwhm(update_results):
    from matplotlib.figure import Figure

    (r1, trace), (r2, _), _ = update_results
    artists = plotting.draw_temporal(Figure().add_subplot(), _plot_data(r1, trace))
    plotting.update_temporal(artists, _plot_data(r2, trace))
    pr2 = processing.process_result(r2, measured=trace)
    assert plotting._fs(pr2.fwhm_retr) in artists.legend_retr.get_text()
    assert plotting._fs(pr2.fwhm_tl) in artists.legend_tl.get_text()


def test_update_convergence_curve_grows_and_rescales():
    from matplotlib.figure import Figure

    ax = Figure().add_subplot()
    artists = plotting.plot_convergence(ax, [1.0, 0.5, 0.3])
    line = artists.line
    plotting.update_convergence_curve(artists, [1.0, 0.5, 0.3, 0.1, 0.01])
    assert artists.line is line
    assert line.get_xydata().shape == (5, 2)
    assert ax.get_xlim()[1] >= 5
    assert ax.get_ylim()[0] <= 0.01
    plotting.update_convergence_curve(artists, [])  # a rehydrated result
    assert line.get_xydata().shape == (0, 2)


def test_update_residual_colorbar_follows_the_new_scale(update_results):
    from matplotlib.figure import Figure

    (r1, trace), (r2, _), _ = update_results
    fig = Figure()
    ax = fig.add_subplot()
    artists = plotting.draw_residual(ax, _plot_data(r1, trace))
    assert artists.mesh is not None
    cbar = fig.colorbar(artists.mesh, ax=ax)
    d2 = _plot_data(r2, trace)
    plotting.update_residual(artists, d2)
    assert d2.processed.residual is not None
    mr = float(np.max(np.abs(d2.processed.residual)))
    assert artists.mesh.get_clim() == pytest.approx((-mr, mr))
    assert cbar.ax.get_ylim() == pytest.approx((-mr, mr))


def test_update_signature_separates_shapes_from_values(update_results):
    from croak.processing import TruthPulse

    (r1, trace), (r2, _), (r3, trace3) = update_results
    d1 = _plot_data(r1, trace)
    d2 = _plot_data(r2, trace)
    d3 = _plot_data(r3, trace3)
    assert plotting._update_signature(d1) == plotting._update_signature(d2)
    assert plotting._update_signature(d1) != plotting._update_signature(d3)
    g = Grid(96, dt=0.5e-15)
    truth = TruthPulse.from_spectrum(g, gaussian_pulse(g, 7e-15), float(wlfreq(800e-9)))
    assert plotting._update_signature(d1) != plotting._update_signature(
        _plot_data(r1, trace, truth=truth)
    )


@pytest.mark.parametrize("key", ["traces", "pulse", "diagnostics"])
def test_page_view_updates_in_place_and_falls_back_to_draw(update_results, key):
    from matplotlib.figure import Figure

    (r1, trace), (r2, _), (r3, trace3) = update_results
    view = plotting.RetrievalPageView(plotting.RETRIEVAL_PAGES[key])
    assert not view.attached
    with pytest.raises(RuntimeError, match="draw"):
        view.update(_plot_data(r1, trace))
    fig = Figure(layout="constrained")
    view.draw(fig, _plot_data(r1, trace))
    axes_before = list(fig.axes)
    d2 = _plot_data(r2, trace)
    assert view.can_update(d2)
    view.update(d2)
    assert list(fig.axes) == axes_before
    d3 = _plot_data(r3, trace3)
    assert not view.can_update(d3)
    with pytest.raises(ValueError, match="draw"):
        view.update(d3)
    view.draw(fig, d3)  # the fallback: a fresh page
    assert list(fig.axes) != axes_before
    view.reset()
    assert not view.attached


def test_page_view_freezes_the_layout_after_the_first_draw(update_results):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.layout_engine import (
        ConstrainedLayoutEngine,
        PlaceHolderLayoutEngine,
    )

    (r1, trace), _, _ = update_results
    view = plotting.RetrievalPageView(plotting.RETRIEVAL_PAGES["pulse"])
    fig = Figure(layout="constrained")
    FigureCanvasAgg(fig)
    view.draw(fig, _plot_data(r1, trace))
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
    fig.canvas.draw()  # the first real draw solves the layout, then holds it
    assert isinstance(fig.get_layout_engine(), PlaceHolderLayoutEngine)
    view.draw(fig, _plot_data(r1, trace))  # a redraw solves it again
    assert isinstance(fig.get_layout_engine(), ConstrainedLayoutEngine)
