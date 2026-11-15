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
the tilt leaves out for your spectrum, and warns above 1 %. A session or GUI
retrieval with the beams in the principal plane runs it on the retrieved
spectrum; with the API, call it yourself.

## In a session and in the GUI

The session settings ({class}`~croak.session.params.RetrieveParams`) describe the
crystal by `crystal`, `cut_angle_deg`, `angle_mode` (`"phase-match"` with
`phase_match_nm`, or `"tilt"` with `tilt_deg`), `polarization`, `delayed_beam`,
`beam_separation_mm` and `focal_f_mm`; `dispersive` switches it on and
`thickness_um` is its thickness. For an SHG trace the session builds the crystal
from them ({func}`~croak.session.pipeline.crystal_setup`) and uses the depth
nodes it needs. `phase_match_nm` defaults to 800 nm: set it to the wavelength the
crystal was tuned for. In the GUI they are set on the Data stage, in the
*Nonlinear crystal* and *Beam geometry* groups shown for SHG data, with a read-out
of the internal angle, the tilt, the path and the depth nodes: they describe the
measurement, and the Check step's marginal prediction and correction use them
before the retrieval does. The Retrieve stage shows a read-only summary. The
Synthetic stage generates an SHG trace through the same crystal when it is
switched on.

## Delay smearing

The two crossing beams' relative delay varies across the focus, which blurs the
trace along delay. For few-cycle pulses this matters: beams of 20 µm waist crossing
at a half-angle of 2.9° blur the delay by 2.4 fs rms (5.5 fs FWHM), against a 6 fs
pulse. Give the beams their diameter and pass the two-beam kernel built from the
same geometry ([two-beam SHG smearing](geometric_smearing.md#two-beam-shg)):

```python
from croak.smearing import shg_two_beam_kernel

beams = TwoBeamGeometry(beam_separation=5e-3, focal_length=0.1, beam_diameter=3e-3)
crystal = CrystalSetup(builtin_crystal("BBO"), math.radians(29.2),
                       phase_match_wavelength=800e-9, geometry=beams,
                       polarization="parallel")
res = croak.retrieve(
    trace, g.omega, delays, "shg", algorithm="lbfgs-ad",
    material=crystal, thickness=thickness, npoints=npoints, omega0=omega0,
    smearing=shg_two_beam_kernel(beams, wavelength=800e-9),
)
```

In a session, `smearing` switches it on and `beam_diameter_mm` gives the
diameter; the GUI sets the diameter in *Beam geometry* on the Data stage and the
switch, *Model the delay smearing*, on the Retrieve stage, since only some solvers
support it.

## The marginal correction

A measured trace carries the instrument's spectral response (spectrometer,
detector, the nonlinear coefficient's dispersion), which the model does not. The
marginal correction removes it by rescaling the trace's frequency rows onto the
marginal predicted from the independent spectrum. Through a crystal that
prediction must include the phase matching, or the correction divides it out and
the crystal model applies it a second time; give the correction the crystal:

```python
from croak.crystals import PhaseMatching

td = croak.preprocess.load_and_clean(
    ..., lam_spec=lam_spec, Ilam_spec=Ilam_spec,
    marginal_correct=True, marginal_crystal=PhaseMatching(crystal, thickness),
)
```

A retrieval of the corrected trace must use the same crystal and thickness, and
refuses anything else. A session does this by itself: with `marginal_correct` on,
the correction uses the retrieval's crystal. The marginal checks predict through
the crystal in the same way; see
[Marginal consistency checks](marginal_checks.md#through-an-shg-crystal).

## What is not modelled yet

- **Fitting the crystal.** Its thickness and angle are inputs, not fitted
  parameters ([issue #26](https://github.com/LupoLab/croak/issues/26)); a
  session's `fit_thickness` is not applied to a crystal.

## How well it works

On full-field simulations of SHG FROG in BBO (Luna.jl, two Gaussian beams
crossing in two dimensions, the signal collected over the whole beam), the
crystal model reproduces the simulated traces to about 1 % of their peak
([issue #27](https://github.com/LupoLab/croak/issues/27),
[#28](https://github.com/LupoLab/croak/issues/28)). A
10 fs pulse with 30 fs² of GDD in 100 µm, crossing at 1.4°, was retrieved with a
duration of 13.80 fs against 13.81 fs and a complex-field error of 0.10, against
0.52 with the ideal model, which narrows the spectrum to mimic the phase-matching
filter.
