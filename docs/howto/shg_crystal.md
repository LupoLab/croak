# SHG FROG with a thick crystal

In a thin crystal, an SHG FROG trace is the ideal one: every pair of fundamental
frequencies converts equally well. In a crystal of tens of micrometres, with a
few-cycle pulse, it is not: phase matching filters the signal's spectrum, the
fundamental and the second harmonic walk off in time, and the generated field
grows as $\omega_s/n_s$. A retrieval with the ideal model then attributes all of
that to the pulse. This page shows how to retrieve with croak's model of a Type I
crystal instead. The physics is in
[the forward model](../explanation/forward_model.md#shg-in-a-type-i-crystal).

## The quick version

```python
import math

import croak
from croak.crystals import CrystalSetup, builtin_crystal, depth_points
from croak.geometry import TwoBeamGeometry

crystal = CrystalSetup(
    builtin_crystal("BBO"),
    cut_angle=math.radians(29.2),         # the manufacturer's cut
    phase_match_wavelength=800e-9,        # tilted to phase-match 800 nm
    geometry=TwoBeamGeometry(beam_separation=5e-3, focal_length=0.1),
    polarization="parallel",              # fundamental in the crossing plane
)
thickness = 100e-6
npoints = depth_points(crystal, g.omega, omega0, thickness)

res = croak.retrieve(
    trace, g.omega, delays, "shg",
    algorithm="lbfgs-ad",
    material=crystal, thickness=thickness, npoints=npoints, omega0=omega0,
)
```

`material=` takes the crystal wherever it takes a material name for PG and SD: in
{func}`~croak.forward.maketrace`, {class}`~croak.forward.ForwardModel`, every
solver, {func}`~croak.pipeline.retrieve_from_tracedata` and the uncertainty
estimates. The retrieved field is the field at the crystal's entrance face.

## Describing the crystal

- **The crystal.** {func}`~croak.crystals.builtin_crystal` gives a built-in crystal
  ({data}`~croak.crystals.CRYSTALS`) with optmat's recommended indices of both
  rays; BBO uses Tamošauskas et al. (2018), valid from 188 nm to 5.2 µm. Any
  other uniaxial crystal can be described with
  {class}`~croak.crystals.UniaxialCrystal` and two index functions.
- **Its angle.** Give the cut angle $\theta_c$ (optic axis to face normal) and
  either the external tilt you set (`tilt=`) or the fundamental wavelength the
  crystal was tuned to phase-match (`phase_match_wavelength=`), from which the
  tilt is solved. {meth}`~croak.crystals.CrystalSetup.internal_angle` and
  {meth}`~croak.crystals.CrystalSetup.external_tilt` report the result.
- **The beams.** {class}`~croak.geometry.TwoBeamGeometry` takes the beam
  separation at the focusing optic and its focal length, and is the one place
  the crossing angle comes from. `geometry=None` models collinear beams.
- **The depth nodes.** {func}`~croak.crystals.depth_points` gives the number of
  depth nodes the crystal's phase mismatch needs on your grid; the models refuse
  fewer.

## Polarisation, and which side the delayed beam is on

In Type I SHG the fundamental is ordinary, polarised perpendicular to the
crystal's principal plane. So the polarisation decides how the beams cross:

- **`polarization="parallel"`** (the fundamental in the crossing plane): the beams
  cross perpendicular to the principal plane. The trace stays symmetric in delay.
- **`polarization="perpendicular"`**: the beams cross in the principal plane. The
  signal's tilt then changes its index to first order, which makes the trace
  asymmetric in delay. The model includes this, and needs to know whether the
  delayed beam is on the optic axis's side of the bisector
  (`delayed_beam="toward_axis"` or `"away_from_axis"`). A crystal's markings do
  not show it, and the two choices give traces mirrored in delay, so retrieve
  with both and keep the better fit.

{func}`~croak.crystals.check_tilt` estimates what the first-order treatment of
the tilt leaves out for your spectrum, and warns above 1 %.

## In a session and in the GUI

The session settings ({class}`~croak.session.params.RetrieveParams`) describe the
crystal by `crystal`, `cut_angle_deg`, `angle_mode` (`"phase-match"` with
`phase_match_nm`, or `"tilt"` with `tilt_deg`), `polarization`, `delayed_beam`,
`beam_separation_mm` and `focal_f_mm`; `dispersive` switches it on and
`thickness_um` is its thickness. For an SHG trace the session builds the crystal
from them ({func}`~croak.session.pipeline.crystal_setup`) and raises the depth
nodes to what it needs. In the GUI the Retrieve stage shows these as the
*Nonlinear crystal* and *Beam geometry* groups for SHG data, with a read-out of
the internal angle, the tilt, the path and the depth nodes. The Synthetic stage
can generate an SHG trace through the same crystal.

## What is not modelled yet

- **Delay smearing.** The two crossing beams' local delay varies across the
  focus, which blurs the trace along delay. croak's smearing kernels do not yet
  cover SHG, and for few-cycle pulses this matters: a 6 fs pulse crossing at
  2.9° with a 20 µm waist is blurred by 2.4 fs.
- **Fitting the crystal.** Its thickness and angle are inputs, not fitted
  parameters ([issue #26](https://github.com/LupoLab/croak/issues/26)).
- **The marginal correction.** A trace loaded with `marginal_correct=True` has
  already had an estimate of the phase-matching response divided out, so a
  crystal model refuses it.

## How well it works

On full-field simulations of SHG FROG in BBO (Luna.jl, two Gaussian beams
crossing in two dimensions, the signal collected over the whole beam), the
crystal model reproduces the simulated traces to about 1 % of their peak. A
10 fs pulse with 30 fs² of GDD in 100 µm, crossing at 1.4°, was retrieved with a
duration of 13.80 fs against 13.81 fs and a complex-field error of 0.10, against
0.52 with the ideal model, which narrows the spectrum to mimic the phase-matching
filter.
