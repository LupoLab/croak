# Quickstart

Generate a synthetic FROG trace, retrieve a pulse from it, and inspect the result.
This small calculation checks your [installation](installation.md); the
[first retrieval tutorial](../tutorials/01_first_retrieval.md) explains the physics
and plots. All times, frequencies and lengths passed to croak use SI units.

## Retrieve a pulse

```python
import numpy as np
import croak

g = croak.Grid(128, dt=0.3e-15)
ew = croak.gaussian_pulse(g, 2.5e-15, phases=[4e-30])   # phases = [GDD, TOD, ...]
delays = np.linspace(-18e-15, 18e-15, 87)
trace = croak.maketrace(g.omega, delays, ew, "shg")
res = croak.retrieve(trace, g.omega, delays, "shg", algorithm="copra",
                     guess=croak.gaussian_pulse(g, 1.8e-15), maxiters=300,
                     rng=np.random.default_rng(0))
print(f"trace error R = {res.error:.3g}")
```

`Grid` defines matched time and angular-frequency axes. `gaussian_pulse` creates
a pulse with a transform-limited intensity FWHM of 2.5 fs; the 4 fs² of spectral
group-delay dispersion stretches it. `maketrace` predicts the intensity recorded
at each frequency and delay in an SHG-FROG measurement. `retrieve` searches for a
complex pulse that reproduces that trace. The seed makes this demonstration
repeatable.

The trace is simulated, so its generating pulse is known. On measured data a
small trace error shows that the model reproduces the trace, not that the model
is the right one; the [measurement modelling guide](../user_guide/models.md)
explains when that distinction matters.

## Choose an algorithm

This example explicitly selects COPRA. Omitting `algorithm` selects
`"warm-lbfgs"`, a COPRA start followed by L-BFGS refinement. The same entry point
supports all solvers, but their model support and options differ. Use the
[solver selection guide](../howto/solver_selection.md) for practical choices and
the executed [algorithm comparison](../tutorials/02_choosing_an_algorithm.md)
for a worked example.

`retrieve` displays progress by default. Set `progress=False` for quiet batch
runs, or supply a callback for your own monitoring. You can also instantiate a
solver, for example `croak.COPRA(maxiters=100)`, and call its `run` method; the
[solver API](../reference/api/retrieval.md) documents this interface.

## Dispersive retrieval

To include propagation through a nonlinear slab, specify the same material,
thickness, depth quadrature and carrier in trace generation and retrieval:

```python
omega0 = croak.maths.wlfreq(260e-9)
medium = dict(material="SiO2", thickness=10e-6, npoints=20, omega0=omega0)
trace = croak.maketrace(g.omega, delays, ew, "pg", **medium)
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="copra",
                     rng=np.random.default_rng(0), **medium)
```

The slab model supports PG and SD, whose signals remain near the fundamental
carrier, but not SHG. The [dispersive tutorial](../tutorials/03_dispersive_retrieval.md)
uses a suitable ultraviolet pulse and explains the recovered field's reference
plane. The [measurement modelling guide](../user_guide/models.md) explains when
spatial geometry and collection must also be included.

## Initial guesses

`guess` accepts `None` for a random Gaussian-spectrum start, a complex spectral
array on the retrieval grid, or a {class}`croak.pulses.Pulse`. Guesses are normalised
internally. The objective is non-convex, so on measured data retrieve from
several random starts and keep the lowest-error result. Agreement between starts
is a convergence check, not a measurement uncertainty.

## The result object

All solvers return a {class}`~croak.result.RetrievalResult`:

```python
res.error          # normalised trace error R
res.errors         # trace-error history
res.spectrum       # complex baseband spectrum, centred order
res.field          # complex temporal field envelope
res.intensity_t    # temporal intensity |E(t)|²
res.phase_omega    # unwrapped spectral phase
res.wavelength     # absolute wavelength axis (m), using omega0
res.trace          # predicted trace, already scaled by the fitted factor res.mu
```

`res.trace` can be compared directly with the supplied data. The absolute
wavelength axis is correct only when `omega0` is the true fundamental carrier.
See [interpreting results](../user_guide/results.md) for plotting, ambiguity
resolution, duration and spectral-phase interpretation.

## Next steps

Continue with [your first retrieval](../tutorials/01_first_retrieval.md), then
[dispersive retrieval](../tutorials/03_dispersive_retrieval.md) and the
[experimental workflow](../tutorials/04_experimental_workflow.md). The
[example gallery](../examples/index.md) covers more specialised calculations.
