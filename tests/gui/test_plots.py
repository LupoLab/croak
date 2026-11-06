"""The pyqtgraph views: maps, curves, colour maps, hover readouts and views."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from numpy.testing import assert_allclose
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QMouseEvent

from croak.gui.plots import (
    NEGATIVE_RGBA,
    Curve,
    LinePlot,
    MapPlot,
    Plot,
    curve_reader,
    is_uniform,
    map_reader,
    signed_colormap,
)


def _shown(qtbot, widget):
    qtbot.addWidget(widget)
    widget.resize(500, 360)
    widget.show()
    qtbot.waitExposed(widget)
    return widget


def _scene_point(plot: Plot, x: float, y: float) -> QPointF:
    return plot.box.mapViewToScene(QPointF(x, y))


def _map(qtbot, **kwargs) -> MapPlot:
    plot = _shown(qtbot, MapPlot(xlabel="Delay (fs)", ylabel="f (PHz)", **kwargs))
    x = np.linspace(-50.0, 50.0, 11)
    y = np.linspace(0.6, 0.9, 7)
    data = np.outer(np.arange(7.0), np.ones(11))
    plot.show_map(data, x, y, (0.0, 6.0))
    return plot


# -- colour maps --------------------------------------------------------------
def test_plots_signed_colormap_is_white_at_zero_and_red_below():
    cmap = signed_colormap("viridis", -1.0, 3.0)
    # 0 lies a quarter of the way along the levels.
    # Qt stores colours in 8 bits, hence the tolerance of one step.
    assert_allclose(cmap.map([0.25], mode=cmap.FLOAT)[0], (1, 1, 1, 1), atol=1 / 255)
    assert_allclose(cmap.map([0.0], mode=cmap.FLOAT)[0], NEGATIVE_RGBA, atol=1 / 255)


def test_plots_signed_colormap_ignores_rounding_noise_below_zero():
    # a bottom level of -2e-12 (rounding noise) must not paint zero red
    cmap = signed_colormap("viridis", -2e-12, 1.0)
    assert_allclose(cmap.map([0.0], mode=cmap.FLOAT)[0], (1, 1, 1, 1), atol=1 / 255)


def test_plots_unsigned_colormap_starts_white():
    cmap = signed_colormap("viridis", 0.0, 1.0)
    assert_allclose(cmap.map([0.0], mode=cmap.FLOAT)[0], (1, 1, 1, 1), atol=1e-6)
    assert cmap.map([1.0], mode=cmap.FLOAT)[0][3] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("name", "low", "high", "match"),
    [
        ("viridis", 0.0, 0.0, "high > 0"),
        ("viridis", 2.0, 1.0, "low < high"),
        ("no-such-map", 0.0, 1.0, "unknown colour map"),
        ("bwr", 0.0, 1.0, ""),
    ],
)
def test_plots_signed_colormap_rejects_bad_levels_and_names(name, low, high, match):
    if not match:  # a matplotlib map is found too
        assert signed_colormap(name, low, high) is not None
        return
    with pytest.raises(ValueError, match=match):
        signed_colormap(name, low, high)


def test_plots_is_uniform_needs_increasing_equal_steps():
    assert is_uniform(np.linspace(0, 1, 5))
    assert not is_uniform(np.array([1.0]))
    assert not is_uniform(np.array([2.0, 1.0, 0.0]))
    assert not is_uniform(np.array([0.0, 1.0, 1.5]))


# -- maps -----------------------------------------------------------------------
def test_plots_map_shows_data_centred_on_its_samples(qtbot):
    plot = _map(qtbot)
    rect = plot.image.mapRectToParent(plot.image.boundingRect())
    # 11 samples 10 fs apart, each a pixel centred on its delay.
    assert rect.left() == pytest.approx(-55.0)
    assert rect.width() == pytest.approx(110.0)
    assert plot.bar.levels() == pytest.approx((0.0, 6.0))
    assert plot.value_at(0.0, 0.6) == 0.0
    assert plot.value_at(10.0, 0.9) == 6.0
    assert plot.value_at(500.0, 0.7) is None


@pytest.mark.parametrize(
    ("data", "x", "match"),
    [
        (np.zeros((3, 3)), np.linspace(0, 1, 4), "does not match"),
        (np.zeros((7, 3)), np.array([0.0, 1.0, 3.0]), "evenly spaced"),
    ],
)
def test_plots_map_rejects_mismatched_or_uneven_axes(qtbot, data, x, match):
    plot = _shown(qtbot, MapPlot())
    with pytest.raises(ValueError, match=match):
        plot.show_map(data, x, np.linspace(0, 1, 7), (0.0, 1.0))


def test_plots_map_updates_in_place_and_changes_levels_freely(qtbot):
    plot = _map(qtbot)
    plot.update_map(np.full((7, 11), 2.0))
    assert plot.value_at(0.0, 0.6) == 2.0
    with pytest.raises(ValueError, match="shape already shown"):
        plot.update_map(np.zeros((2, 2)))
    plot.set_levels((-40.0, 0.0))
    assert plot.bar.levels() == pytest.approx((-40.0, 0.0))


def test_plots_signed_map_recolours_for_negative_levels(qtbot):
    plot = _map(qtbot, signed=True)
    plot.set_levels((-2.0, 6.0))
    cmap = plot.bar.colorMap()
    assert_allclose(cmap.map([0.0], mode=cmap.FLOAT)[0], NEGATIVE_RGBA, atol=1 / 255)
    plot.set_levels((0.0, 6.0))  # no negatives: back to white at the floor
    cmap = plot.bar.colorMap()
    assert_allclose(cmap.map([0.0], mode=cmap.FLOAT)[0], (1, 1, 1, 1), atol=1 / 255)


def test_plots_map_colour_bar_drags_and_double_click_restores_it(qtbot):
    plot = _map(qtbot, signed=True)
    plot.set_levels((-2.0, 6.0))
    # a drag on the bar moves its levels; white stays at zero on a signed map
    plot.bar.setLevels(values=(-1.0, 3.0))
    plot.bar.sigLevelsChanged.emit(plot.bar)
    cmap = plot.bar.colorMap()
    assert_allclose(cmap.map([0.25], mode=cmap.FLOAT)[0], (1, 1, 1, 1), atol=1 / 255)
    plot.reset_view()  # a double-click
    assert plot.bar.levels() == pytest.approx((-2.0, 6.0))


def test_plots_unsigned_map_keeps_its_colours_when_dragged(qtbot):
    plot = _map(qtbot)
    before = plot.bar.colorMap()
    plot.bar.setLevels(values=(0.2, 0.8))
    plot.bar.sigLevelsChanged.emit(plot.bar)
    assert plot.bar.colorMap() is before


def test_plots_map_starts_white_unless_told_otherwise(qtbot):
    white = _map(qtbot)
    cmap = white.bar.colorMap()
    assert_allclose(cmap.map([0.0], mode=cmap.FLOAT)[0], (1, 1, 1, 1), atol=1 / 255)
    plain = _map(qtbot, white=False, colours="bwr")
    cmap = plain.bar.colorMap()
    assert cmap.map([0.0], mode=cmap.FLOAT)[0][2] > 0.9  # bwr starts blue


def test_plots_map_says_a_message_in_place_of_the_image(qtbot):
    plot = _map(qtbot)
    plot.say("No trace loaded")
    assert plot.data is None
    assert plot.message.isVisible()
    assert plot.value_at(0.0, 0.7) is None
    plot.show_map(np.ones((2, 2)), np.arange(2.0), np.arange(2.0), (0.0, 1.0))
    assert not plot.message.isVisible()


def test_plots_map_update_needs_an_image_first(qtbot):
    plot = _shown(qtbot, MapPlot())
    with pytest.raises(ValueError, match="shape already shown"):
        plot.update_map(np.zeros((2, 2)))


# -- hover and view ---------------------------------------------------------------
def test_plots_hover_reads_the_map_under_the_pointer(qtbot):
    plot = _map(qtbot)
    plot.reader = map_reader(plot, "τ", "f")
    assert plot.reader(500.0, 0.65) == ""  # off the image: nothing to say
    plot.hover(_scene_point(plot, -40.0, 0.65))
    assert plot.note.isVisible() and plot.cross_x.isVisible()
    assert plot.note.toPlainText().strip() == "τ -40   f 0.65   1"
    # Past 62 % of the width the readout flips to the pointer's left.
    plot.hover(_scene_point(plot, 40.0, 0.92))
    assert plot.note.anchor.x() == pytest.approx(1.0)
    assert plot.note.anchor.y() == pytest.approx(0.0)


def test_plots_hover_hides_outside_the_box_without_a_reader_or_text(qtbot):
    plot = _map(qtbot)
    plot.hover(_scene_point(plot, 0.0, 0.7))  # no reader yet
    assert not plot.note.isVisible()
    plot.reader = lambda _x, _y: ""
    plot.hover(_scene_point(plot, 0.0, 0.7))
    assert not plot.note.isVisible()
    plot.reader = map_reader(plot, "τ", "f")
    plot.hover(QPointF(-1000.0, -1000.0))
    assert not plot.note.isVisible()
    plot.hover(_scene_point(plot, 0.0, 0.7))
    plot.leaveEvent(QEvent(QEvent.Type.Leave))
    assert not plot.cross_y.isVisible()


def test_plots_double_click_returns_to_the_home_view(qtbot):
    plot = _shown(qtbot, Plot(xlabel="x", ylabel="y"))
    event = QMouseEvent(
        QEvent.Type.MouseButtonDblClick,
        QPointF(10, 10),
        QPointF(10, 10),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    plot.mouseDoubleClickEvent(event)  # no home yet: fits whatever is there
    plot.set_view((0.0, 10.0), (-1.0, 1.0))
    plot.box.setRange(xRange=(4.0, 5.0), padding=0)
    plot.mouseDoubleClickEvent(event)
    assert plot.box.viewRange()[0] == pytest.approx([0.0, 10.0])
    plot.set_labels("Delay (fs)", None)
    assert plot.item.getAxis("bottom").labelText == "Delay (fs)"
    plot.set_labels(None, "Power (W)")
    assert plot.item.getAxis("left").labelText == "Power (W)"


# -- lines --------------------------------------------------------------------------
def test_plots_lines_take_colours_in_turn_and_a_legend_when_several(qtbot):
    plot = _shown(qtbot, LinePlot(xlabel="t (fs)", ylabel="I"))
    x = np.linspace(-1, 1, 21)
    plot.set_curves([Curve(x, x**2)])
    assert plot.legend is None
    plot.set_curves(
        [
            Curve(x, x**2, "retrieved"),
            Curve(x, x**2 + 1, "truth", colour="#000000", dashed=True),
            Curve(x, x**3, "fit"),
        ]
    )
    assert plot.legend is not None and plot.legend.isVisible()
    assert len(plot.legend.items) == 3
    pens = [item.opts["pen"] for item in plot.items]
    assert pens[0].color().name() == "#1f6fb4"
    assert pens[1].style() == Qt.PenStyle.DashLine
    # The explicitly coloured curve does not use up a palette colour.
    assert pens[2].color().name() == "#d2670b"
    plot.set_curves([Curve(x, x)])
    assert not plot.legend.isVisible()


def test_plots_log_axis_drops_non_positive_values(qtbot):
    plot = _shown(qtbot, LinePlot())
    x = np.arange(5.0)
    plot.set_curves([Curve(x, np.array([0.0, 1.0, 10.0, -1.0, 100.0]))], log_y=True)
    assert_allclose(plot.items[0].yData, [1.0, 10.0, 100.0])
    plot.update_curve(0, x, np.array([1.0, 0.0, 1.0, 1.0, 1.0]))
    assert plot.items[0].yData.size == 4
    plot.fit()
    # A log axis spans whole decades (at least one), so it always has ticks.
    assert plot.box.viewRange()[1] == pytest.approx([0.0, 1.0])
    # ...labelled at each decade only, which a short plot has room for
    left = plot.item.getAxis("left")
    assert left.tickSpacing(0.0, 2.0, 100.0) == [(1.0, 0), (1.0, 0)]
    plot.set_curves([Curve(x, x)])  # a linear axis spaces its ticks itself
    assert left.tickSpacing(0.0, 2.0, 100.0) != [(1.0, 0), (1.0, 0)]


def test_plots_right_axis_carries_its_own_curves(qtbot):
    plot = _shown(qtbot, LinePlot(ylabel="I", right_label="Phase (rad)"))
    x = np.linspace(0, 1, 11)
    plot.set_curves([Curve(x, x, "I"), Curve(x, -x, "φ", right=True)], log_y=True)
    assert plot.right_box is not None
    assert plot.items[1] in plot.right_box.addedItems
    # Negative phase survives the log intensity axis.
    assert plot.items[1].yData.min() == pytest.approx(-1.0)
    plot.set_right_range(-3.0, 3.0)
    assert plot.right_box.viewRange()[1] == pytest.approx([-3.0, 3.0])
    plot.set_curves([Curve(x, x)])  # removes the right-hand curve again
    assert plot.items[0] not in plot.right_box.addedItems
    plot.box.resize(300, 200)  # the right view box follows the main one


def test_plots_right_range_is_ignored_without_a_right_axis(qtbot):
    plot = _shown(qtbot, LinePlot())
    plot.set_right_range(0.0, 1.0)
    assert plot.right_box is None


def test_plots_right_label_needs_a_right_axis(qtbot):
    with_right = _shown(qtbot, LinePlot(right_label="Phase"))
    with_right.set_right_label("Phase (rad)")
    assert with_right.item.getAxis("right").labelText == "Phase (rad)"
    without = _shown(qtbot, LinePlot())
    without.set_right_label("ignored")  # no right axis: nothing to label


def test_plots_curves_can_be_thin_references_drawn_beneath(qtbot):
    plot = _shown(qtbot, LinePlot())
    x = np.arange(3.0)
    plot.set_curves([Curve(x, x), Curve(x, x, width=0.9, under=True)])
    assert plot.items[1].zValue() < plot.items[0].zValue()
    assert plot.items[1].opts["pen"].widthF() == pytest.approx(0.9)


def test_plots_marks_replace_the_previous_set(qtbot):
    plot = _shown(qtbot, LinePlot())
    plot.set_marks([(1.0, ""), (2.0, "")])
    plot.set_marks([(3.0, "#ff0000")], dashed=False)
    assert [m.value() for m in plot.marks] == [3.0]


def test_plots_fit_frames_the_left_curves(qtbot):
    plot = _shown(qtbot, LinePlot())
    plot.fit()  # nothing to fit
    x = np.linspace(0.0, 2.0, 3)
    plot.set_curves([Curve(x, np.array([0.0, 10.0, 0.0])), Curve(x, x, right=True)])
    plot.fit()
    assert plot.box.viewRange()[0] == pytest.approx([0.0, 2.0])
    assert plot.box.viewRange()[1] == pytest.approx([-0.5, 10.5])
    plot.set_curves([Curve(x, np.full(3, np.nan))])
    plot.fit()  # no finite value: a default window
    assert plot.box.viewRange()[1] == pytest.approx([-0.05, 1.05])


def test_plots_curve_reader_reads_each_named_left_curve(qtbot):
    plot = _shown(qtbot, LinePlot())
    x = np.linspace(0.0, 1.0, 11)
    plot.set_curves(
        [
            Curve(x, 2 * x, "R"),
            Curve(x, x),
            Curve(x, x, "φ", right=True),
            Curve(x + 5, x, "far"),
            Curve(np.zeros(0), np.zeros(0), "empty"),
        ]
    )
    read = curve_reader(plot, "t")
    assert read(0.5, 0.0) == "t 0.5   R 1   0.5"
    assert read(-3.0, 0.0) == ""
