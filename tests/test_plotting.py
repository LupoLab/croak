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
    assert plotting.units._fs(9.512e-15) == plotting.sig3(9.512)
    assert plotting.sig3(float("nan")) == "nan"


def test_plot_retrieval_reuses_a_precomputed_processed_result(result_and_trace):
    """Passing ``processed=`` skips the internal process_result entirely."""
    res, trace = result_and_trace
    pr = processing.process_result(res, measured=trace)

    calls = []
    real = plotting.panels.process_result

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    plotting.panels.process_result = counting
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
        plotting.panels.process_result = real


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
    real = plotting.panels.spectrogram

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(plotting.panels, "spectrogram", counted)
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


def test_retrieval_legends_sit_in_fixed_corners(result_and_trace):
    """No retrieval-panel legend uses loc="best", which jumps between live frames."""
    res, trace = result_and_trace
    fig = plotting.plot_retrieval(res, measured=trace, lam_min=700e-9, lam_max=950e-9)
    legends = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    assert legends
    for legend in legends:
        assert legend._loc != 0, legend  # 0 is matplotlib's code for "best"


# -- spectrum axis --------------------------------------------------------------------
@pytest.mark.parametrize(
    ("axis", "xlabel"),
    [("frequency", "Frequency (PHz)"), ("wavelength", "Wavelength (nm)")],
)
def test_plot_spectral_draws_against_the_chosen_axis(result_and_trace, axis, xlabel):
    from matplotlib.figure import Figure

    res, trace = result_and_trace
    pr = processing.process_result(res, measured=trace)
    ax = Figure().add_subplot()
    plotting.plot_spectral(ax, pr, lam_min=700e-9, lam_max=950e-9, axis=axis)
    assert ax.get_xlabel() == xlabel
    lo, hi = ax.get_xlim()
    assert lo < hi
    if axis == "frequency":
        two_pi_phz = 2 * np.pi * 1e15
        assert lo == pytest.approx(float(wlfreq(950e-9)) / two_pi_phz)
        assert hi == pytest.approx(float(wlfreq(700e-9)) / two_pi_phz)
    else:
        assert (lo, hi) == pytest.approx((700.0, 950.0))
    assert float(np.nanmax(ax.lines[0].get_ydata())) == pytest.approx(1.0)


def test_plot_spectral_defaults_to_frequency_and_rejects_unknown_axes(
    result_and_trace,
):
    from matplotlib.figure import Figure

    res, trace = result_and_trace
    pr = processing.process_result(res, measured=trace)
    ax = Figure().add_subplot()
    plotting.plot_spectral(ax, pr)
    assert ax.get_xlabel() == "Frequency (PHz)"
    with pytest.raises(ValueError, match="axis must be one of"):
        plotting.plot_spectral(Figure().add_subplot(), pr, axis="energy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spectral_axis must be one of"):
        plotting.retrieval_plot_data(res, measured=trace, spectral_axis="energy")  # type: ignore[arg-type]


def test_frequency_spectrum_is_the_omega_density(result_and_trace):
    """On the frequency axis the retrieved curve is |E(ω)|² at unit peak, i.e. the
    wavelength density with its ω² Jacobian divided out."""
    from matplotlib.figure import Figure

    res, trace = result_and_trace
    pr = processing.process_result(res, measured=trace)
    ax = Figure().add_subplot()
    plotting.plot_spectral(ax, pr, axis="frequency")
    np.testing.assert_allclose(ax.lines[0].get_ydata(), pr.Iw / pr.Iw.max(), atol=1e-12)
    omega_abs = pr.omega + pr.omega0
    np.testing.assert_allclose(ax.lines[0].get_xdata(), omega_abs / (2 * np.pi * 1e15))


# ---------------------------------------------------------------------------
# Uncertainty figures
# ---------------------------------------------------------------------------
def _uresult(*, profiles=False, thickness=False, prior=True, seed=0):
    """A lightweight UncertaintyResult (no retrieval) for the figure tests."""
    from croak.uncertainty import UncertaintyResult

    rng = np.random.default_rng(seed)
    n = 60
    pe = 5e-15
    samples = pe + 0.2e-15 * rng.standard_normal(n)
    kw = {}
    if profiles:
        t = np.linspace(-20e-15, 20e-15, 41)
        base = np.exp(-((t / pe) ** 2))
        kw["profiles"] = np.clip(
            base + 0.02 * rng.standard_normal((n, t.size)), 0, None
        )
        kw["t_profile"] = t
    if thickness:
        kw["thicknesses"] = 100e-6 + 5e-6 * rng.standard_normal(n)
        if prior:
            kw["thickness_central"] = 100e-6
            kw["thickness_sigma"] = 5e-6
    return UncertaintyResult(
        statistic="fwhm",
        point_estimate=pe,
        samples=samples,
        interval_68=(pe - 0.2e-15, pe + 0.2e-15),
        interval_95=(pe - 0.4e-15, pe + 0.4e-15),
        interval_method="percentile",
        method="thickness" if thickness else "parametric",
        n_resamples=n,
        n_converged=n,
        frog_errors=0.01 + 0.001 * rng.random(n),
        base_error=0.01,
        **kw,
    )


def test_plot_uncertainty_histogram_only_and_with_a_band():
    fig = plotting.plot_uncertainty(_uresult(), truth=5.1e-15)
    (ax,) = fig.axes
    assert ax.get_xlabel() == "FWHM (fs)"
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["95 %", "68 %", "estimate", "truth"]
    assert ax.get_title().startswith("5.00 ±")

    fig = plotting.plot_uncertainty(_uresult(profiles=True))
    hist, band = fig.axes
    assert band.get_xlabel() == "Time (fs)"
    # zoomed to the envelope's support (~ +-8 fs above 1 %) plus padding,
    # well inside the +-20 fs profile axis
    lo, hi = band.get_xlim()
    assert -20 < lo < 0 < hi < 20


def test_plot_intensity_band_requires_profiles():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="collect_profiles=True"):
        plotting.plot_intensity_band(ax, _uresult())
    plt.close(fig)


def test_plot_intensity_band_of_an_empty_envelope_keeps_the_full_axis():
    import dataclasses

    import matplotlib.pyplot as plt

    u = dataclasses.replace(_uresult(profiles=True), profiles=np.zeros((4, 41)))
    fig, ax = plt.subplots()
    plotting.plot_intensity_band(ax, u)
    # nothing above 1 % of a zero median: no zoom is applied
    assert ax.get_xlim()[0] <= -20 and ax.get_xlim()[1] >= 20
    plt.close(fig)


@pytest.mark.parametrize("profiles", [False, True])
def test_plot_uncertainty_dispatches_thickness_results(profiles):
    fig = plotting.plot_uncertainty(
        _uresult(thickness=True, profiles=profiles), truth=5e-15
    )
    assert len(fig.axes) == (3 if profiles else 2)
    fwhm_ax, err_ax = fig.axes[:2]
    assert fwhm_ax.get_xlabel() == "Substrate thickness (µm)"
    assert err_ax.get_ylabel() == "FROG error R"
    # the prior band L0 +- sigma is shaded (in um)
    assert "L₀ ± σ_L" in [t.get_text() for t in fwhm_ax.get_legend().get_texts()]


def test_thickness_plots_without_a_prior_or_thicknesses():
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2)
    no_prior = _uresult(thickness=True, prior=False)
    plotting.plot_fwhm_vs_thickness(a, no_prior)
    assert "L₀ ± σ_L" not in [t.get_text() for t in a.get_legend().get_texts()]
    with pytest.raises(ValueError, match="no thicknesses"):
        plotting.plot_fwhm_vs_thickness(a, _uresult())
    with pytest.raises(ValueError, match="no thicknesses"):
        plotting.plot_error_vs_thickness(b, _uresult())
    plt.close(fig)


def _variant_kwargs(res, variant):
    """plot-data options exercising the optional curves of every panel."""
    if variant == "truth":
        truth = processing.TruthPulse.from_spectrum(res.grid, res.spectrum, res.omega0)
        return {
            "truth": truth,
            "Iomega_meas": np.abs(res.spectrum) ** 2,
            "energy": 1e-6,
            "spectral_axis": "wavelength",
        }
    if variant == "truth_without_phases":
        t = np.linspace(-30e-15, 30e-15, 61)
        truth = processing.TruthPulse.from_intensity(t, np.exp(-((t / 7e-15) ** 2)))
        return {"truth": truth}
    # "bare": a processed result without the measured trace (no residual, no
    # measured marginals) and no trace band limits
    return {"processed": processing.process_result(res)}


def _assert_close(x, y, where):
    """Like _assert_equal, to 1e-6 relative (1e-9 for scalars).

    The truth overlays reach the same values by a different order of operations
    (the update path rereads the static truth phase from its line), which
    differs in the last bits.
    """
    if isinstance(x, np.ndarray):
        np.testing.assert_allclose(
            x, np.asarray(y), rtol=1e-6, atol=1e-30, err_msg=where
        )
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), where
        for i, (p, q) in enumerate(zip(x, y, strict=True)):
            _assert_close(p, q, f"{where}[{i}]")
    elif isinstance(x, float):
        assert x == pytest.approx(y, rel=1e-9), where
    else:
        assert x == y, (where, x, y)


# ---------------------------------------------------------------------------
# Small helpers and defensive paths
# ---------------------------------------------------------------------------
def test_cmap_negwhite_without_a_negative_part_is_cmap_white():
    a = plotting.cmap_negwhite("viridis", negfrac=0.0)
    b = plotting.cmap_white("viridis")
    np.testing.assert_allclose(a(np.linspace(0, 1, 9)), b(np.linspace(0, 1, 9)))


def test_si_power_below_the_smallest_prefix_uses_it():
    scale, unit = plotting.si_power(1e-30)
    assert (scale, unit) == plotting.units._POWER_PREFIXES[-1]


def test_phase_ylim_of_nothing_finite_is_none():
    assert plotting.pulse._phase_ylim(None, np.array([np.nan]), np.array([])) is None


def test_phase_ylim_keeps_a_flat_phase_from_filling_the_axis():
    floor = plotting.pulse.PHASE_MIN_HALF_SPAN
    # Rounding noise of a transform-limited pulse: the axis stays ±floor wide.
    low, high = plotting.pulse._phase_ylim(np.array([0.0, 1e-15, -2e-15]))
    assert high - low == pytest.approx(2 * floor)
    # A real phase keeps its own span as margin either side.
    assert plotting.pulse._phase_ylim(np.array([0.0, 4.0])) == (-2.0, 6.0)


def test_plot_residual_and_spectrogram_defaults(result_and_trace):
    import matplotlib.pyplot as plt

    res, trace = result_and_trace
    fig, (a, b) = plt.subplots(1, 2)
    resid = trace - res.trace
    plotting.plot_residual(a, res.omega, res.delays, resid, res.omega0)
    # no halfwidth: the full delay span is shown
    assert a.get_xlim()[1] >= res.delays.max() / 1e-15
    tc, lam, S = processing.spectrogram(res, n_t=8)
    plotting.plot_spectrogram(b, tc, lam, S)  # default colormap, no limits
    assert b.get_ylim()[1] >= lam.max() / 1e-9 * 0.999
    plt.close(fig)


def test_plot_spectral_frequency_axis_with_a_measured_spectrum(result_and_trace):
    import matplotlib.pyplot as plt

    res, trace = result_and_trace
    data = plotting.retrieval_plot_data(
        res, measured=trace, Iomega_meas=np.abs(res.spectrum) ** 2
    )
    fig, ax = plt.subplots()
    artists = plotting.draw_spectral(ax, data)  # the frequency axis is the default
    assert artists.line_meas is not None
    # the measured curve is the same omega-density as the retrieved one here
    np.testing.assert_allclose(
        artists.line_meas.get_ydata(), artists.line_retr.get_ydata(), atol=1e-9
    )
    plt.close(fig)


def test_mappables_of_a_curve_panel_are_empty():
    from matplotlib.lines import Line2D

    assert plotting.retrieval._mappables(
        plotting.LineArtists(line=Line2D([0], [0]))
    ) == (
        None,
        None,
    )


def test_page_without_a_residual_or_with_negatives_lays_out_its_colorbars(
    result_and_trace,
):
    from matplotlib.figure import Figure

    res, trace = result_and_trace
    bare = plotting.retrieval_plot_data(res, processed=processing.process_result(res))
    fig = Figure()
    plotting.plot_retrieval_page(bare, plotting.RETRIEVAL_OVERVIEW, fig=fig)
    # 12 panels, 2 phase twins and 2 trace colorbars; the empty residual panel
    # gets none
    assert len(fig.axes) == 16
    signed = plotting.retrieval_plot_data(res, measured=trace - 0.05 * trace.max())
    fig = Figure()
    plotting.plot_retrieval_page(signed, plotting.RETRIEVAL_OVERVIEW, fig=fig)
    # each trace pair gains a negative colorbar beside its positive one, and the
    # residual has its own
    assert len(fig.axes) == 12 + 2 + 2 * 2 + 1


def test_filter_view_vmax_ignores_a_window_outside_the_axis():
    from dataclasses import dataclass

    @dataclass
    class _Stub:
        Ifrog_meas: np.ndarray
        Ifrog_filt: np.ndarray
        trace: np.ndarray
        lam_frog: np.ndarray
        lamm_lims: tuple[float, float] | None

    lam = np.linspace(200e-9, 400e-9, 8)
    data = np.ones((8, 4))
    data[0, 0] = 3.0
    # a window that excludes every wavelength falls back to the whole array
    stub = _Stub(data, data, np.ones((8, 4)), lam, (900e-9, 1000e-9))
    assert plotting.filter_view_vmax(stub) == pytest.approx(3.0)


def test_decimate_for_display_strides_large_images():
    C = np.arange(2500.0 * 3).reshape(2500, 3)
    out, x, y = plotting.traces._decimate_for_display(
        C, np.arange(3.0), np.arange(2500.0)
    )
    assert out.shape == (1250, 3) and y.size == 1250 and x.size == 3


def test_plot_simulated_trace_with_negative_samples(result_and_trace):
    res, trace = result_and_trace
    fig = plotting.plot_simulated_trace(
        res.omega, res.delays, trace - 0.05 * trace.max(), res.omega0
    )
    # 4 panels + a positive and a negative colorbar for each of the two images
    assert len(fig.axes) == 8


def test_plot_frog_filter_without_negative_samples(experiment):
    """Non-negative traces need only the positive colorbar on each row."""
    from dataclasses import replace

    exp = experiment
    td = preprocess.load_and_clean(
        exp.trace,
        exp.lam_nm * 1e-9,
        exp.delay_fs * 1e-15,
        exp.interaction,
        lam_min=exp.lam_min,
        lam_max=exp.lam_max,
        input_unit="delay",
    )
    clipped = replace(
        td,
        Ifrog_meas=np.clip(td.Ifrog_meas, 0, None),
        Ifrog_filt=np.clip(td.Ifrog_filt, 0, None),
        trace=np.clip(td.trace, 0, None),
    )
    fig = plotting.plot_frog_filter(clipped)
    # six panels and one colorbar per row
    assert len(fig.axes) == 6 + 2


def test_retrieval_plot_data_carries_the_spectral_axis(result_and_trace):
    from matplotlib.figure import Figure

    res, trace = result_and_trace
    default = _plot_data(res, trace)
    assert default.spectral_axis == "frequency"
    by_wavelength = _plot_data(res, trace, spectral_axis="wavelength")
    ax = Figure().add_subplot()
    plotting.draw_spectral(ax, by_wavelength)
    assert ax.get_xlabel() == "Wavelength (nm)"
    fig = plotting.plot_retrieval(res, measured=trace, spectral_axis="wavelength")
    spectrum = next(a for a in fig.axes if a.get_title() == "Spectrum")
    assert spectrum.get_xlabel() == "Wavelength (nm)"


def test_flat_phases_leave_the_phase_axis_unscaled(result_and_trace):
    """With no finite phase sample there is no phase range to set."""
    from dataclasses import replace

    from matplotlib.figure import Figure

    from croak.plotting import screen

    res, trace = result_and_trace
    pr = processing.process_result(res, measured=trace)
    pr = replace(
        pr, phi_t=np.full_like(pr.phi_t, np.nan), phi_w=np.full_like(pr.phi_w, np.nan)
    )
    data = plotting.retrieval_plot_data(res, measured=trace, processed=pr)
    fig = Figure()
    a, b = fig.subplots(1, 2)
    t_art = plotting.draw_temporal(a, data)
    s_art = plotting.draw_spectral(b, data)
    # matplotlib's own autoscale, which no phase range overrode
    assert t_art.ax_phase.get_autoscaley_on() and s_art.ax_phase.get_autoscaley_on()
    assert screen.temporal_view(data).right_range is None
    assert screen.spectral_view(data).right_range is None
