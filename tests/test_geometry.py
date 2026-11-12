"""The two-beam geometry: validation and the crossing angle."""

import math

import pytest

from croak.geometry import TwoBeamGeometry


def test_crossing_half_angle_is_half_the_separation_over_the_focal_length():
    g = TwoBeamGeometry(beam_separation=10e-3, focal_length=0.2, beam_diameter=3e-3)
    assert g.crossing_half_angle == pytest.approx(math.atan(0.025))
    assert g.beam_diameter == 3e-3


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        (dict(beam_separation=0.0, focal_length=0.1), "beam_separation"),
        (dict(beam_separation=5e-3, focal_length=-0.1), "focal_length"),
        (
            dict(beam_separation=5e-3, focal_length=0.1, beam_diameter=math.nan),
            "beam_diameter",
        ),
    ],
)
def test_lengths_must_be_finite_and_positive(kwargs, name):
    with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
        TwoBeamGeometry(**kwargs)
