"""Run the wizard with ``python -m croak.gui``."""

from __future__ import annotations

import sys

from . import run_wizard

if __name__ == "__main__":
    sys.exit(run_wizard())
