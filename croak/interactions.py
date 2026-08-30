"""Nonlinear FROG interactions and their Wirtinger adjoints.

Each interaction defines the instantaneous nonlinear signal field produced by a
(undelayed) *test* field ``E`` and a (delayed) *gate* field ``G`` in the time
domain, together with the reverse-mode (Wirtinger) adjoint used to back-propagate
a cotangent through that signal. The three standard geometries are supported:

================  =====================  =================================
Interaction       signal ``s``           geometry
================  =====================  =================================
``SHG``           ``E * G``              second-harmonic generation
``SD``            ``E**2 * conj(G)``     self-diffraction
``PG``            ``E * |G|**2``         polarization gating
================  =====================  =================================

Adjoint convention: the cotangent of a complex variable ``v`` is
``v_bar = dL/dv*`` (Wirtinger). :meth:`Interaction.signal_adjoint` returns the
cotangents ``(E_bar, G_bar)`` for given output cotangent ``s_bar``.

Geometric smearing (:mod:`croak.smearing`) needs the same signals written out as a
product of *three* separately shifted replicas, because the two factors that coincide
in the thin 1-D model acquire different arrival times across the focal spot. That form
is exposed by :meth:`Interaction.smeared_shifts` and :meth:`Interaction.smeared_signal`;
at zero smearing they reproduce :meth:`signal` exactly.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

__all__ = ["Interaction", "SHG", "SD", "PG", "get_interaction", "INTERACTIONS"]

Complex = NDArray[np.complex128]


class Interaction:
    """Base class for FROG nonlinear interactions.

    Subclasses set :attr:`name` and implement :meth:`signal` and
    :meth:`signal_adjoint`.
    """

    name: ClassVar[str]
    #: Factor relating the trace carrier to the pulse carrier (SHG doubles it).
    omega0_scale: ClassVar[float] = 1.0
    #: Whether the trace is invariant under time reversal of the pulse,
    #: :math:`E(t) \to E^*(-t)` — equivalently conjugating the spectrum. When it
    #: is, the retrieval cannot distinguish a pulse from its mirror image and the
    #: solver returns whichever branch it happened to land in. True only for SHG,
    #: whose signal ``E*G`` is symmetric in the two arms; the SD and PG kernels
    #: treat their arms differently and so fix the direction of time. See
    #: :func:`croak.processing.resolve_time_direction`.
    time_reversal_ambiguous: ClassVar[bool] = False

    def signal(self, test: Complex, gate: Complex) -> Complex:
        """Return the time-domain nonlinear signal field."""
        raise NotImplementedError

    def signal_adjoint(
        self, test: Complex, gate: Complex, s_bar: Complex
    ) -> tuple[Complex, Complex]:
        """Return ``(test_bar, gate_bar)`` for output cotangent ``s_bar``."""
        raise NotImplementedError

    def smeared_shifts(self, tau: float) -> tuple[float, float]:
        r"""Time shifts of the three replicas under geometric smearing.

        The smeared signal is a product of three replicas of ``E``. One of them does not
        depend on the smearing parameter ``p``; the other two are shifted by
        ``varying_base -/+ p/2`` and therefore *swap* under ``p -> -p``. With symmetric
        quadrature nodes that lets the forward model build each shifted field once and
        use it twice.

        Parameters
        ----------
        tau : float
            Delay (s).

        Returns
        -------
        tuple of float
            ``(fixed_shift, varying_base)``, both in seconds. Replica time shifts are
            ``fixed_shift`` and ``varying_base -/+ p/2``, where a shift ``s`` means the
            replica is ``E(t - s)``.
        """
        raise NotImplementedError

    def smeared_signal(
        self, fixed: Complex, varying: Complex, varying_swapped: Complex
    ) -> Complex:
        """Combine the three smeared replicas into the nonlinear signal.

        Parameters
        ----------
        fixed : numpy.ndarray
            The replica at :meth:`smeared_shifts`' ``fixed_shift``.
        varying : numpy.ndarray
            The replica at ``varying_base - p/2``.
        varying_swapped : numpy.ndarray
            The replica at ``varying_base + p/2`` — i.e. ``varying`` evaluated at the
            mirrored node ``-p``.
        """
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        """Return ``ClassName()``."""
        return f"{type(self).__name__}()"


class SHG(Interaction):
    """Second-harmonic generation: ``s = E * G``."""

    name = "shg"
    omega0_scale = 2.0
    time_reversal_ambiguous = True

    def signal(self, test: Complex, gate: Complex) -> Complex:
        """Return the SHG signal ``E * G``."""
        return test * gate

    def signal_adjoint(self, test, gate, s_bar):
        """Return ``(test_bar, gate_bar)`` for the SHG signal ``E * G``."""
        test_bar = np.conj(gate) * s_bar
        gate_bar = np.conj(test) * s_bar
        return test_bar, gate_bar


class SD(Interaction):
    """Self-diffraction: ``s = E**2 * conj(G)``."""

    name = "sd"

    def signal(self, test, gate):
        """Return the SD signal ``E**2 * conj(G)``."""
        return test**2 * np.conj(gate)

    def signal_adjoint(self, test, gate, s_bar):
        """Return ``(test_bar, gate_bar)`` for the SD signal ``E**2 * conj(G)``."""
        test_bar = 2.0 * np.conj(test) * gate * s_bar
        gate_bar = test**2 * np.conj(s_bar)
        return test_bar, gate_bar

    def smeared_shifts(self, tau):
        """Return ``(tau, 0.0)``: the *conjugated* replica is the delayed, fixed one.

        The smeared SD signal is ``E(t + p/2) E(t - p/2) conj(E(t - tau))`` — the two
        squared replicas separate by ``p`` while the conjugated one only ever moves with
        the delay.
        """
        return tau, 0.0

    def smeared_signal(self, fixed, varying, varying_swapped):
        """Return ``E(t+p/2) E(t-p/2) conj(E(t-tau))``."""
        return varying * varying_swapped * np.conj(fixed)


class PG(Interaction):
    """Polarization gating: ``s = E * |G|**2``."""

    name = "pg"

    def signal(self, test, gate):
        """Return the PG signal ``E * |G|**2``."""
        return test * np.abs(gate) ** 2

    def signal_adjoint(self, test, gate, s_bar):
        """Return ``(test_bar, gate_bar)`` for the PG signal ``E * |G|**2``."""
        test_bar = np.abs(gate) ** 2 * s_bar
        gate_bar = 2.0 * gate * np.real(np.conj(test) * s_bar)
        return test_bar, gate_bar

    def smeared_shifts(self, tau):
        """Return ``(0.0, tau)``: the undelayed probe is the fixed replica.

        The smeared PG signal is ``E(t) E(t - tau + p/2) conj(E(t - tau - p/2))`` — the
        gate pair, which coincide in ``|G|**2`` when unsmeared, separate by ``p``.
        """
        return 0.0, tau

    def smeared_signal(self, fixed, varying, varying_swapped):
        """Return ``E(t) E(t-tau+p/2) conj(E(t-tau-p/2))``."""
        return fixed * varying * np.conj(varying_swapped)


#: Registry of interaction instances keyed by lowercase name.
INTERACTIONS: dict[str, Interaction] = {c.name: c() for c in (SHG, SD, PG)}


def get_interaction(interaction: str | Interaction) -> Interaction:
    """Resolve an interaction from a name (case-insensitive) or instance.

    Parameters
    ----------
    interaction : str or Interaction
        ``"shg"``, ``"sd"``, ``"pg"`` (any case) or an :class:`Interaction`.

    Returns
    -------
    Interaction
    """
    if isinstance(interaction, Interaction):
        return interaction
    try:
        return INTERACTIONS[interaction.lower()]
    except AttributeError, KeyError:
        raise ValueError(
            f"unknown interaction {interaction!r}; available: "
            f"{', '.join(sorted(INTERACTIONS))}"
        ) from None
