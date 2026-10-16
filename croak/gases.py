r"""Gas refractive index at a pressure and temperature, from optmat.

A gas medium is dispersive like a solid (see :mod:`croak.materials`), but its
refractive index depends on the gas *density*, and hence on pressure and
temperature, as well as on wavelength. This module provides
:func:`gas_refractive_index`, which builds an ``n(wavelength_m)`` callable for a
gas at a chosen pressure and temperature. That callable has the same shape as
:func:`croak.materials.refractive_index`'s result, so a gas plugs into the whole
dispersion pipeline by registering it with
:func:`croak.materials.register_material` and treating the optical path length as
the "thickness".

Physics
-------
The dispersion of each gas is optmat's recommended model of its per-molecule
polarisability :math:`\alpha(\lambda)` (`optmat <https://optmat.readthedocs.io>`_,
``optmat.gas.models.registry.default_model``); air is optmat's dry air, the
mole-fraction sum of its constituents' models. The index at a number density
:math:`N` follows from the Lorentz–Lorenz relation,

.. math::

    \frac{n^2 - 1}{n^2 + 2} = \frac{N\alpha}{3\varepsilon_0},

and the density from the **real gas**, :math:`N = P/(Z k_\mathrm{B}T)`, with the
compressibility factor :math:`Z` of each gas's reference equation of state
(CoolProp, through optmat; air is Lemmon et al.'s pseudo-pure dry air, and ozone
has none, so it is taken as ideal). :math:`n - 1` is then nearly proportional to
:math:`P/T`, but not exactly: at 20 °C the real gas differs from the ideal by
0.05 % for helium and air at 1 bar, 0.5 % at 10 bar, and up to 15 % for SF₆ at
10 bar. A state where the gas would condense (SF₆ above about 21 bar at 20 °C, for
example) is refused.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from optmat.conditions import State
from optmat.gas.air import AirModel, air_model
from optmat.gas.gases import Gas
from optmat.gas.models.base import Model
from optmat.gas.models.registry import default_model

__all__ = [
    "GASES",
    "ROOM_TEMPERATURE_K",
    "canonical_name",
    "gas_model",
    "gas_refractive_index",
]

#: Default gas temperature (K) when none is given (ca. 20 °C).
ROOM_TEMPERATURE_K: float = 293.15

_AIR = "Air"
_PASCAL_PER_BAR = 1e5

#: Names of the gases available (case-insensitive): air, then optmat's species.
GASES: tuple[str, ...] = (_AIR, *(gas.value for gas in Gas))

# Names earlier croak versions offered, kept so saved sessions still resolve.
_ALIASES: dict[str, str] = {"Helium": "He"}


def canonical_name(gas: str) -> str:
    """Return the current name of a gas given by any case or an earlier name.

    Parameters
    ----------
    gas : str
        A gas name.

    Returns
    -------
    str
        The :data:`GASES` name ``gas`` refers to (``"He"`` for ``"Helium"``).

    Raises
    ------
    ValueError
        If ``gas`` is not a known gas.

    Examples
    --------
    >>> canonical_name("helium")
    'He'
    """
    lowered = gas.lower()
    for old, new in _ALIASES.items():
        if old.lower() == lowered:
            return new
    for name in GASES:
        if name.lower() == lowered:
            return name
    raise ValueError(f"unknown gas {gas!r}; available: {', '.join(GASES)}")


@functools.cache
def gas_model(gas: str) -> Model | AirModel:
    """Return the optmat model of a gas's polarisability.

    Parameters
    ----------
    gas : str
        A name in :data:`GASES` (or an earlier name of one), any case.

    Returns
    -------
    optmat.gas.models.base.Model or optmat.gas.air.AirModel
        optmat's recommended model of the gas, or its dry-air model, with its
        ``name``, ``wavelength_range`` (m) and ``identity`` for provenance.

    Raises
    ------
    ValueError
        If ``gas`` is not a known gas.

    Examples
    --------
    >>> gas_model("Helium").name
    'fit/He'
    """
    name = canonical_name(gas)
    return air_model() if name == _AIR else default_model(name)


def gas_refractive_index(
    gas: str,
    pressure_bar: float,
    temperature_k: float = ROOM_TEMPERATURE_K,
) -> Callable[[ArrayLike], NDArray[np.float64]]:
    r"""Build an ``n(wavelength_m)`` callable for a gas at given pressure/temperature.

    The index follows from optmat's polarisability model of the gas through the
    Lorentz–Lorenz relation at the real-gas density :math:`P/(Z k_\mathrm{B}T)` (see
    the module docstring). The returned callable has the same contract as
    :func:`croak.materials.refractive_index`'s result, so it can be registered with
    :func:`croak.materials.register_material` and used wherever a material is.

    Parameters
    ----------
    gas : str
        Gas name (case-insensitive); one of :data:`GASES`.
    pressure_bar : float
        Gas pressure in bar, ``>= 0``. ``0.0`` gives vacuum (``n = 1``).
    temperature_k : float, optional
        Gas temperature in kelvin, ``> 0`` (default :data:`ROOM_TEMPERATURE_K`,
        ca. 20 °C).

    Returns
    -------
    callable
        Function mapping vacuum wavelength in metres to the real refractive index,
        ``nan`` at a non-positive or non-finite wavelength (masked by
        :func:`croak.materials.beta`) and extrapolated beyond the model's stated
        range.

    Raises
    ------
    ValueError
        If ``gas`` is not a known gas, the pressure is negative or not finite, the
        temperature is not positive and finite, or the gas is not a gas at that
        state (condensed, or beyond its equation of state's range;
        ``optmat.exceptions.StateError``, a ``ValueError``).

    Examples
    --------
    >>> n = gas_refractive_index("Air", 1.0, 273.15)
    >>> float(n(800e-9)) - 1.0  # doctest: +ELLIPSIS
    0.000286...
    """
    model = gas_model(gas)
    if not (math.isfinite(pressure_bar) and pressure_bar >= 0.0):
        raise ValueError(f"pressure must be finite and >= 0 bar, got {pressure_bar}")
    if not (math.isfinite(temperature_k) and temperature_k > 0.0):
        raise ValueError(f"temperature must be finite and > 0 K, got {temperature_k}")
    # optmat refuses a zero-pressure state (no density to evaluate); at zero
    # density the index is exactly 1.
    state = (
        State(pressure_bar * _PASCAL_PER_BAR, temperature_k) if pressure_bar else None
    )
    if state is not None:
        # Fail here, not on the first use: the equation of state refuses a state
        # where the gas condenses, and the density depends only on the state.
        model.refractive_index(1e-6, state, warn=False)

    def n(wavelength_m: ArrayLike) -> NDArray[np.float64]:
        lam = np.asarray(wavelength_m, dtype=float)
        # As for the solids (croak.materials): a grid point at ω ≤ 0 is nan, and
        # the model is extrapolated without warning beyond its stated range.
        physical = np.isfinite(lam) & (lam > 0.0)
        out = np.full(lam.shape, np.nan)
        if state is None:
            out[physical] = 1.0
            return out
        index = model.refractive_index(lam[physical], state, warn=False)
        # The propagation model is lossless: the real part.
        out[physical] = np.real(index)
        return out

    n.__doc__ = (
        f"Refractive index of {gas} at {pressure_bar} bar, {temperature_k} K "
        "as a function of wavelength (m)."
    )
    return n
