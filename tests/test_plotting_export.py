"""Publication figures of the screen views (:mod:`croak.plotting.export`)."""

from __future__ import annotations

import numpy as np
import pytest

from croak.plotting.export import (
    PANEL_SIZE,
    draw_view,
    save_figure,
    views_figure,
)
from croak.plotting.screen import (
    BandView,
    BarsView,
    CurveView,
    LineView,
    MapView,
    SpanView,
)

X = np.linspace(-10.0, 10.0, 41)


def _line(**kwargs) -> LineView:
    """A line view with every decoration a page can give one."""
    defaults = dict(
        curves=(
            CurveView(X, np.exp(-(X**2) / 20), "retrieved", "retrieved"),
            CurveView(X, np.exp(-(X**2) / 18), "truth", "truth"),
            CurveView(X[::5], np.full(9, 0.5), "estimate", "points", points=True),
            CurveView(X, 0.01 * X**2, "phase", "phase", right=True),
        ),
        title="Temporal",
        xlabel="Time (fs)",
        ylabel="Intensity",
        right_label="Phase (rad)",
        xrange=(-8.0, 8.0),
        yrange=(0.0, 1.1),
        right_range=(-1.0, 2.0),
        notes=(("FWHM 7 fs", False),),
        marks=((0.0, "boundary"),),
        levels=((0.5, "faint"),),
        bands=(BandView(X, np.zeros_like(X), 0.2 + 0 * X, "estimate"),),
        spans=(SpanView(-2.0, 2.0, "prior"), SpanView(0.1, 0.2, "fit", vertical=False)),
        bars=BarsView(np.linspace(-5, 5, 6), np.arange(5.0), "estimate"),
    )
    defaults.update(kwargs)
    return LineView(**defaults)


def _map(signed: bool = False) -> MapView:
    y = np.linspace(380.0, 420.0, 30)
    values = np.exp(-(X[None] ** 2) / 30 - ((y[:, None] - 400) / 8) ** 2)
    return MapView(
        values=values - (0.05 if signed else 0.0),
        x=X,
        y=y,
        levels=(-0.05 if signed else 0.0, 1.0),
        title="Measured trace",
        xlabel="Delay (fs)",
        ylabel="Wavelength (nm)",
        unit="a.u.",
        signed=signed,
        xrange=(-5.0, 5.0),
        yrange=(390.0, 410.0),
    )


def test_export_line_view_keeps_its_curves_axes_and_ranges():
    fig = views_figure([[_line()]])
    ax, right = fig.axes
    assert ax.get_xlabel() == "Time (fs)" and right.get_ylabel() == "Phase (rad)"
    assert ax.get_xlim() == (-8.0, 8.0) and right.get_ylim() == (-1.0, 2.0)
    assert ax.get_title(loc="left") == "Temporal"
    assert ax.get_title(loc="right") == "FWHM 7 fs"
    names = [text.get_text() for text in ax.get_legend().get_texts()]
    assert names == ["retrieved", "truth", "points", "phase"]  # both axes, one legend
    truth = next(line for line in ax.lines if line.get_label() == "truth")
    assert truth.get_linewidth() < 1.0  # a thin reference beneath the result


def test_export_line_view_with_one_curve_and_a_log_axis_has_no_legend():
    view = _line(
        curves=(CurveView(X, 1 + X**2, "error"),),
        right_label=None,
        log_y=True,
        yrange=None,
        xrange=None,
        notes=(),
        bars=None,
    )
    fig = views_figure([[view]])
    (ax,) = fig.axes
    assert ax.get_yscale() == "log" and ax.get_legend() is None


@pytest.mark.parametrize("signed", [False, True])
def test_export_map_view_draws_a_rasterized_image_with_its_scale(signed):
    fig = views_figure([[_map(signed)]])
    ax = fig.axes[0]
    (mesh,) = ax.collections
    bar = mesh.colorbar.ax
    assert mesh.get_rasterized()
    assert mesh.get_clim() == pytest.approx((-0.05 if signed else 0.0, 1.0))
    # White at zero: an empty background reads as paper on either scale.
    zero = mesh.cmap(mesh.norm(0.0))
    assert zero == pytest.approx((1.0, 1.0, 1.0, 1.0), abs=0.02)
    assert bar.get_ylabel() == "a.u."
    assert ax.get_ylim() == (390.0, 410.0)


def test_export_map_view_takes_a_panels_own_colour_map():
    fig = views_figure([[_map()]], colours=[["bwr"]], white=[[False]])
    mesh = fig.axes[0].collections[0]
    assert mesh.cmap(0.0)[2] > 0.5  # bwr starts blue, not white


def test_export_figure_lays_rows_out_with_empty_cells():
    fig = views_figure([[_line(), _map()], [_map(), None]])
    assert tuple(fig.get_size_inches()) == pytest.approx(
        (2 * PANEL_SIZE[0], 2 * PANEL_SIZE[1])
    )
    assert sum(not ax.axison for ax in fig.axes) == 1  # the empty cell


def test_export_figure_needs_a_view():
    with pytest.raises(ValueError, match="at least one view"):
        views_figure([[None]])


def test_export_draw_view_dispatches_on_the_kind():
    from matplotlib.figure import Figure

    ax = Figure().add_subplot()
    draw_view(ax, _map())
    assert ax.collections


def test_export_saves_vector_files_with_text_kept_as_text(tmp_path):
    fig = views_figure([[_line(), _map()]])
    pdf, svg, png = (tmp_path / f"page.{ext}" for ext in ("pdf", "svg", "png"))
    for path in (pdf, svg, png):
        save_figure(fig, str(path))
    assert b"/Image" in pdf.read_bytes()  # the map, rasterized
    assert b"/FontFile2" in pdf.read_bytes()  # TrueType text, not outlines
    content = svg.read_text()
    assert "<image" in content and "Measured trace" in content
    assert png.read_bytes()[:4] == b"\x89PNG"


def test_export_views_without_ranges_or_units_keep_matplotlibs_own():
    from dataclasses import replace

    bare = replace(_map(), unit="", xrange=None, yrange=None)
    fig = views_figure([[bare, _line(right_range=None)]])
    ax = fig.axes[0]
    assert ax.collections[0].colorbar.ax.get_ylabel() == ""
    assert ax.get_xlim() == pytest.approx((X[0] - 0.25, X[-1] + 0.25))  # the cells
