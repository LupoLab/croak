r"""The two-beam (non-collinear SHG) geometry, the one source of the crossing angle.

A non-collinear SHG FROG splits the pulse into two parallel, collimated beams a
distance ``beam_separation`` apart and focuses both with one optic of focal length
``focal_length``. They cross at the focus with the external half-angle

.. math:: \alpha = \arctan\!\left(\frac{d}{2f}\right),

each beam at :math:`\pm\alpha` to their bisector, along which the signal leaves.
The same :math:`\alpha` sets two effects: the phase mismatch of the crystal (the
fundamentals' longitudinal wavevector shrinks as the cosine of the refracted,
internal half-angle; see
:mod:`croak.crystals`) and the delay smearing across the focal spot. Both read it
from one :class:`TwoBeamGeometry`, so they can never use different angles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["TwoBeamGeometry"]


@dataclass(frozen=True)
class TwoBeamGeometry:
    """Two parallel beams focused to cross by one optic.

    Parameters
    ----------
    beam_separation : float
        Centre-to-centre distance of the two beams at the focusing optic (m),
        ``> 0``.
    focal_length : float
        Focal length of the focusing optic (m), ``> 0``.
    beam_diameter : float or None, optional
        1/e² intensity diameter of each (Gaussian) beam at the focusing optic (m),
        ``> 0``. Only the delay-smearing kernel needs it; ``None`` (default) when
        smearing is not modelled.

    Raises
    ------
    ValueError
        If a length is not finite and positive.

    Examples
    --------
    >>> g = TwoBeamGeometry(beam_separation=5e-3, focal_length=0.1)
    >>> round(math.degrees(g.crossing_half_angle), 3)
    1.432
    """

    beam_separation: float
    focal_length: float
    beam_diameter: float | None = None

    def __post_init__(self) -> None:
        """Validate the lengths."""
        lengths = {
            "beam_separation": self.beam_separation,
            "focal_length": self.focal_length,
        }
        if self.beam_diameter is not None:
            lengths["beam_diameter"] = self.beam_diameter
        for name, value in lengths.items():
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be finite and positive, got {value!r}")

    @property
    def crossing_half_angle(self) -> float:
        """External half-angle between each beam and their bisector (rad)."""
        return math.atan(0.5 * self.beam_separation / self.focal_length)
