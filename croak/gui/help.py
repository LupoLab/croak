"""Per-stage help: a ``?`` summary dialog and matching control tooltips.

Each stage declares :data:`Stage.HELP` as an ordered list of
``(section, [(label, text)])``. The same ``text`` is shown in two places from a
single source:

* the scrollable summary dialog opened by the wizard header's ``?`` button
  (:class:`HelpDialog`), and
* the hover tooltip on the matching control, applied by :func:`apply_tooltips`.

Tooltips are matched by the control's ``(group-box title, form label)`` so that
labels reused across groups (e.g. ``"λ unit"``) stay distinct. Widget labels spell
a literal ampersand ``&&`` (see :func:`_plain`), while :data:`Stage.HELP` keys are
written the way the user reads them, so the escape is undone before matching.
"""

from __future__ import annotations

from html import escape

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

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


def _plain(text: str) -> str:
    """Undo Qt's mnemonic escaping, giving the label as the user reads it.

    Qt treats a lone ``&`` in a check box, push button, group-box title or form-row
    label as a keyboard-mnemonic marker and does not draw it, so a *literal*
    ampersand must be written ``&&``. Help keys are written unescaped, so strip the
    escape before comparing.

    Examples
    --------
    >>> _plain("Skip filtering && regrid")
    'Skip filtering & regrid'
    """
    return text.replace("&&", "&")


def _field_widget(form: QFormLayout, row: int) -> QWidget | None:
    """The widget in a form row, whether it occupies the field or spans both."""
    for role in (QFormLayout.ItemRole.FieldRole, QFormLayout.ItemRole.SpanningRole):
        item = form.itemAt(row, role)
        if item is not None and item.widget() is not None:
            return item.widget()
    return None


def _row_label(form: QFormLayout, row: int, field: QWidget) -> str | None:
    """Label text for a form row: the label column, else a checkbox's own text."""
    item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
    if item is not None:
        label = item.widget()
        if isinstance(label, QLabel):
            return _plain(label.text())
    if isinstance(field, QCheckBox):
        return _plain(field.text())
    return None


def apply_tooltips(
    root: QWidget, help_map: dict[tuple[str, str], str]
) -> set[tuple[str, str]]:
    """Set tooltips on ``root``'s form controls from ``help_map``.

    Walks every :class:`QFormLayout` under ``root`` and, for each row, looks up
    ``(enclosing group-box title, row label)`` in ``help_map``; on a hit the
    field widget's tooltip is set. Returns the set of keys that matched, so
    callers (and tests) can detect help entries with no corresponding control.
    """
    matched: set[tuple[str, str]] = set()
    for form in root.findChildren(QFormLayout):
        box = form.parentWidget()
        section = _plain(box.title()) if isinstance(box, QGroupBox) else ""
        for row in range(form.rowCount()):
            field = _field_widget(form, row)
            if field is None:
                continue
            label = _row_label(form, row, field)
            if label is None:
                continue
            key = (section, label)
            text = help_map.get(key)
            if text is not None:
                field.setToolTip(text)
                matched.add(key)
    return matched
