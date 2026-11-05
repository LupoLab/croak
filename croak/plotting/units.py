"""Display units and number formatting shared by the plotters.

Retrieval quantities are SI (s, rad/s, W); the axes show fs, PHz and an
SI-prefixed power, and read-outs use three significant digits (:func:`sig3`).
"""

from __future__ import annotations

import numpy as np

_TWOPI_PHZ = 2 * np.pi * 1e15


def sig3(value: float) -> str:
    """Format a number to three significant digits, for a read-out.

    Three significant digits keeps a read-out equally informative across the
    ranges croak covers (sub-fs attosecond pulses through hundreds of fs;
    nanometre-wide through hundred-nanometre-wide spectra), where a fixed number
    of decimals is too coarse at one end and spuriously precise at the other.
    Unlike the ``g`` format this keeps trailing zeros (so ``7.00``, not ``7``) and
    never switches to exponent notation, both of which read badly beside a sibling
    entry in a legend or a table column.

    Parameters
    ----------
    value : float
        The number to format, already in its display unit.

    Returns
    -------
    str
        The formatted number, with no unit attached.

    Examples
    --------
    >>> sig3(9.512), sig3(7.0), sig3(0.85), sig3(953.2)
    ('9.51', '7.00', '0.850', '953')
    """
    if not np.isfinite(value) or value == 0.0:
        return f"{value:.2f}"
    # digits before the decimal point (<= 0 for |value| < 1), so 3 - that many is
    # how many decimals are still needed to reach three significant digits.
    integer_digits = int(np.floor(np.log10(abs(value)))) + 1
    return f"{value:.{max(0, 3 - integer_digits)}f}"


def _fs(seconds: float) -> str:
    """Format a duration (s) as femtoseconds to three significant digits."""
    return sig3(seconds / 1e-15)


_POWER_PREFIXES: tuple[tuple[float, str], ...] = (
    (1e15, "PW"),
    (1e12, "TW"),
    (1e9, "GW"),
    (1e6, "MW"),
    (1e3, "kW"),
    (1.0, "W"),
    (1e-3, "mW"),
    (1e-6, "µW"),
)


def si_power(watts: float) -> tuple[float, str]:
    """Return ``(scale, unit)`` to display ``watts`` with an SI power prefix.

    Picks the largest prefix not exceeding ``|watts|`` (so e.g. ``1.2e7`` W →
    ``(1e6, "MW")``). Non-finite or non-positive inputs fall back to plain watts;
    values below the smallest prefix use that smallest prefix.
    """
    w = abs(float(watts))
    if not np.isfinite(w) or w <= 0:
        return 1.0, "W"
    for scale, unit in _POWER_PREFIXES:
        if w >= scale:
            return scale, unit
    return _POWER_PREFIXES[-1]
