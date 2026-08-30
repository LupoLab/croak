r"""Gas refractive index (pressure-scaled Sellmeier).

Sellmeier coefficients and the density scaling are taken from
`Luna.jl <https://github.com/LupoLab/Luna.jl>`_ (MIT), an open-source Julia
package for nonlinear optical pulse propagation.

A gas medium is dispersive like a solid (see :mod:`croak.materials`), but its
refractive index depends on the gas *density*, and hence on pressure and
temperature, as well as on wavelength. This module provides
:func:`gas_refractive_index`, which builds an ``n(wavelength_m)`` callable for a
gas at a chosen pressure and temperature. That callable has the same shape as
:func:`croak.materials.refractive_index`'s result, so a gas plugs into the whole
dispersion pipeline by registering it with
:func:`croak.materials.register_material` and treating the optical path length as
the "thickness" — exactly the route the refractiveindex.info bridge uses.

Physics
-------
Following ``Luna.jl`` (``PhysData.jl`` ``sellmeier_gas``), the linear
susceptibility of a gas is *linear in the number density* :math:`\rho`:

.. math::

    \chi_1(\lambda, P, T) = \gamma(\lambda)\,\rho(P, T),
    \qquad n = \sqrt{1 + \chi_1},

where :math:`\gamma` is the single-particle polarisability from a Sellmeier
expansion. Luna divides the literature Sellmeier coefficients by the reference
number density :math:`\rho(1\,\mathrm{bar}, 273.15\,\mathrm{K})` and multiplies
back by :math:`\rho(P, T)`; equivalently, the **raw** coefficients give
:math:`\chi_1` at the reference conditions (1 bar, 0 °C). We scale the density
with the **ideal-gas law** :math:`\rho \propto P/T` rather than a real-gas
equation of state, so

.. math::

    n(\lambda, P, T) = \sqrt{1 + \chi_{1,\mathrm{ref}}(\lambda)\,
    \frac{P}{1\,\mathrm{bar}}\,\frac{273.15\,\mathrm{K}}{T}}.

The ideal-gas approximation is accurate to ~0.1 % for helium and air up to a few
bar; it deviates from a real-gas treatment (Luna uses CoolProp) only at high
pressure (tens of bar, e.g. hollow-core-fibre compression). The Sellmeier poles
sit in the deep XUV, so :math:`n` is smooth across the optical/IR band; any
non-finite value outside the valid range is masked downstream by
:func:`croak.materials.beta`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "GASES",
    "GAS_REFERENCE_PRESSURE_BAR",
    "GAS_REFERENCE_TEMPERATURE_K",
    "ROOM_TEMPERATURE_K",
    "gas_refractive_index",
]

#: Pressure (bar) at which the Sellmeier coefficients below are normalised.
GAS_REFERENCE_PRESSURE_BAR: float = 1.0
#: Temperature (K) at which they are normalised (0 °C — Luna's ``dens_1bar_0degC``).
GAS_REFERENCE_TEMPERATURE_K: float = 273.15
#: Default gas temperature (K) when none is given (ca. 20 °C; Luna's ``roomtemp``).
ROOM_TEMPERATURE_K: float = 293.15

# Reference linear susceptibility χ₁_ref(λ_µm) of each gas at the reference
# conditions (1 bar, 273.15 K). These are the Sellmeier expansions for the
# single-particle polarisability γ from Luna.jl ``PhysData.jl`` with the raw
# literature coefficients (Luna's ``B/dens`` cancels the ``×dens`` it applies
# afterwards, leaving χ₁ at the reference density). Wavelength is in micrometres,
# the unit in which the literature coefficients are expressed.
_GAS_SUSCEPTIBILITY_UM: dict[
    str, Callable[[NDArray[np.float64]], NDArray[np.float64]]
] = {
    # Helium — γ_JCT, three-term fit to high-frequency data,
    # Phys. Rev. A 92, 033821 (2015). C1 < 0, so no pole in the optical range.
    "Helium": lambda u: (
        2.16463842e-05 * u**2 / (u**2 - -6.80769781e-04)
        + 2.10561127e-07 * u**2 / (u**2 - 5.13251289e-03)
        + 4.75092720e-05 * u**2 / (u**2 - 3.18621354e-03)
    ),
    # Air — γ_Börzsönyi, two-term Sellmeier, Appl. Opt. 47, 27, 4856 (2008).
    "Air": lambda u: (
        1.492644e-04 * u**2 / (u**2 - 1.936e-05)
        + 4.180757e-04 * u**2 / (u**2 - 7.434e-03)
    ),
}

#: Names of the gases available (case-insensitive).
GASES: tuple[str, ...] = tuple(_GAS_SUSCEPTIBILITY_UM)


def _resolve(gas: str) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """Return the reference-susceptibility callable for ``gas`` (case-insensitive)."""
    for name, chi in _GAS_SUSCEPTIBILITY_UM.items():
        if name.lower() == gas.lower():
            return chi
    raise ValueError(f"unknown gas {gas!r}; available: {', '.join(GASES)}")


def gas_refractive_index(
    gas: str,
    pressure_bar: float,
    temperature_k: float = ROOM_TEMPERATURE_K,
) -> Callable[[ArrayLike], NDArray[np.float64]]:
    r"""Build an ``n(wavelength_m)`` callable for a gas at given pressure/temperature.

    The refractive index follows the ideal-gas-scaled Sellmeier model

    .. math::

        n(\lambda) = \sqrt{1 + \chi_{1,\mathrm{ref}}(\lambda)\,
        \frac{P}{1\,\mathrm{bar}}\,\frac{273.15\,\mathrm{K}}{T}},

    with :math:`\chi_{1,\mathrm{ref}}` the susceptibility at 1 bar, 0 °C (see the
    module docstring). The returned callable has the same contract as
    :func:`croak.materials.refractive_index`'s result, so it can be registered with
    :func:`croak.materials.register_material` and used wherever a material is.

    Parameters
    ----------
    gas : str
        Gas name (case-insensitive); one of :data:`GASES` (``"Helium"``, ``"Air"``).
    pressure_bar : float
        Gas pressure in bar. The susceptibility scales linearly with it
        (ideal-gas density), so ``0.0`` gives vacuum (``n = 1``).
    temperature_k : float, optional
        Gas temperature in kelvin (default :data:`ROOM_TEMPERATURE_K`, ca. 20 °C).
        The susceptibility scales as ``1/T``.

    Returns
    -------
    callable
        Function mapping vacuum wavelength in metres to the real refractive index.
        Values where the (deep-XUV) Sellmeier poles would make the radicand
        negative come out non-finite, to be masked by
        :func:`croak.materials.beta`.

    Raises
    ------
    ValueError
        If ``gas`` is not a known gas.

    Examples
    --------
    >>> n = gas_refractive_index("Air", 1.0, 273.15)
    >>> float(n(800e-9)) - 1.0  # doctest: +ELLIPSIS
    0.000286...
    """
    chi_ref = _resolve(gas)
    # Ideal-gas density scaling relative to the reference conditions (1 bar, 0 °C).
    scale = (pressure_bar / GAS_REFERENCE_PRESSURE_BAR) * (
        GAS_REFERENCE_TEMPERATURE_K / temperature_k
    )

    def n(wavelength_m: ArrayLike) -> NDArray[np.float64]:
        um = np.asarray(wavelength_m, dtype=float) * 1e6
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(1.0 + chi_ref(um) * scale)

    n.__doc__ = (
        f"Refractive index of {gas} at {pressure_bar} bar, {temperature_k} K "
        "as a function of wavelength (m)."
    )
    return n
