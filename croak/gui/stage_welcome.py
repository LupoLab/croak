"""Welcome / main-menu page: the first screen the wizard shows.

Offers the entry points into the application: load experimental traces for
retrieval (the normal Stage 1 pathway), load a numerically simulated trace
(``scansave`` HDF5 → preprocess), load a previous retrieval (the saved
``options.toml`` replay), or generate a synthetic trace. The page only drives
navigation; the actual work lives in the wizard and the stages it routes to.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

__all__ = ["StageWelcome"]

_WELCOME = (
    "Welcome to <b>croak</b> — a pulse-retrieval workbench.<br>"
    "Choose how you would like to begin."
)


class StageWelcome(QWidget):
    """A simple centred menu with the entry-point buttons.

    Takes the owning :class:`~croak.gui.wizard.Wizard` so the buttons can drive
    navigation (``start_load``/``load_simulated``/``load_previous``/
    ``show_synthetic``).
    """

    def __init__(self, wizard):
        super().__init__()
        self.wizard = wizard

        outer = QVBoxLayout(self)
        outer.addStretch(1)

        title = QLabel("croak")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 42px; font-weight: bold; color: #22324a;")
        outer.addWidget(title)

        msg = QLabel(_WELCOME)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setStyleSheet("font-size: 15px; padding: 8px 0 24px 0;")
        outer.addWidget(msg)

        for label, slot in (
            ("Load experimental traces for retrieval", wizard.start_load),
            ("Load simulated trace", wizard.load_simulated),
            ("Load previous retrieval", wizard.load_previous),
            ("Generate synthetic trace", wizard.show_synthetic),
        ):
            btn = QPushButton(label)
            btn.setMinimumHeight(56)
            btn.setMaximumWidth(420)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setStyleSheet("font-size: 15px; padding: 10px;")
            btn.clicked.connect(slot)
            wrap = QWidget()
            wl = QVBoxLayout(wrap)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            wl.addWidget(btn)
            outer.addWidget(wrap)

        outer.addStretch(2)
