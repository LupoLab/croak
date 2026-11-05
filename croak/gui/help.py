"""Per-stage help: a ``?`` summary dialog and matching control tooltips.

Each stage declares :data:`Stage.HELP` as an ordered list of
``(section, [(label, text)])``. The same ``text`` is shown in two places from a
single source:

* the scrollable summary dialog opened by Help for this stage (F1, or the ⋯ menu)
  (:class:`HelpDialog`), and
* the hover tooltip on the matching control, applied by :func:`apply_tooltips`.

Tooltips are matched by the control's ``(section title, form label)`` so that
labels reused across groups (e.g. ``"λ unit"``) stay distinct. Widget labels spell
a literal ampersand ``&&`` (see :func:`croak.gui.sheet.plain`), while
:data:`Stage.HELP` keys are written the way the user reads them, so the escape is
undone before matching.
"""

from __future__ import annotations

from html import escape

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .sheet import Section, row_field, row_text

__all__ = ["HelpDialog", "HelpSections", "apply_tooltips", "build_html", "flatten"]

#: A stage's help content: ordered ``(section_title, [(label, text), ...])``.
HelpSections = list[tuple[str, list[tuple[str, str]]]]


def flatten(sections: HelpSections) -> dict[tuple[str, str], str]:
    """Map ``(section_title, label) -> text`` for tooltip lookup."""
    return {
        (section, label): text for section, rows in sections for label, text in rows
    }


def build_html(title: str, intro: str, sections: HelpSections) -> str:
    """Render the help content as a rich-text HTML document."""
    parts = [f"<h2>{escape(title)}</h2>"]
    if intro:
        parts.append(f"<p>{escape(intro)}</p>")
    for section, rows in sections:
        parts.append(f"<h3>{escape(section)}</h3>")
        parts.append("<dl>")
        for label, text in rows:
            parts.append(f"<dt><b>{escape(label)}</b></dt><dd>{escape(text)}</dd>")
        parts.append("</dl>")
    return "\n".join(parts)


class HelpDialog(QDialog):
    """A scrollable, modeless summary of one stage's options."""

    def __init__(
        self,
        title: str,
        intro: str,
        sections: HelpSections,
        parent: QWidget | None = None,
    ):
        """Build the dialog from a stage's title/intro/help sections."""
        super().__init__(parent)
        self.setWindowTitle(f"Help — {title}")
        self.resize(560, 640)
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(build_html(title, intro, sections))
        layout.addWidget(browser)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


def _section_title(form: QFormLayout) -> str:
    """The title of the section a form row sits in ("" outside any section)."""
    widget = form.parentWidget()
    while widget is not None and not isinstance(widget, Section):
        widget = widget.parentWidget()
    return widget.section_title if widget is not None else ""


def apply_tooltips(
    root: QWidget, help_map: dict[tuple[str, str], str]
) -> set[tuple[str, str]]:
    """Set tooltips on ``root``'s form controls from ``help_map``.

    Walks every :class:`QFormLayout` under ``root`` and, for each row, looks up
    ``(enclosing section title, row label)`` in ``help_map``; on a hit the
    field widget's tooltip is set. Returns the set of keys that matched, so
    callers (and tests) can detect help entries with no corresponding control.
    """
    matched: set[tuple[str, str]] = set()
    for form in root.findChildren(QFormLayout):
        section = _section_title(form)
        for row in range(form.rowCount()):
            field = row_field(form, row)
            label = row_text(form, row)
            if field is None or not label:
                continue
            key = (section, label)
            text = help_map.get(key)
            if text is not None:
                field.setToolTip(text)
                matched.add(key)
    return matched
