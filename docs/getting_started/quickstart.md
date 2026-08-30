# Quickstart

This page is the ten-line tour. For a step-by-step, executed walk-through see the
[first-retrieval tutorial](../tutorials/01_first_retrieval.md).

## Retrieve a pulse

```python
import numpy as np
import croak

# 1. A matched time/frequency grid and a test pulse (2.5 fs Gaussian with GDD).
g = croak.Grid(128, dt=0.3e-15)
ew = croak.gaussian_pulse(g, 2.5e-15, phases=[4e-30])   # phases = [GDD, TOD, ...]
delays = np.linspace(-18e-15, 18e-15, 87)

# 2. Synthesise a measured trace (here SHG-FROG).
trace = croak.maketrace(g.omega, delays, ew, "shg")

# 3. Retrieve — the API is uniform across algorithms.
res = croak.retrieve(trace, g.omega, delays, "shg", algorithm="copra")

print(res.error)        # final trace error R (the normalised FROG error)
field = res.field       # retrieved E(t)
spectrum = res.spectrum # retrieved E(omega)
```

Equivalently, instantiate the solver class directly — useful when you want to set
algorithm options:

```python
solver = croak.COPRA(maxiters=100)
res = solver.run(trace, g.omega, delays, "shg")
```

## Choose an algorithm

`algorithm=` selects from the registry {data}`croak.retrieve.ALGORITHMS`:

| Name | Class | Gradient | Notes |
|------|-------|----------|-------|
| `"warm-lbfgs"` | {class}`~croak.warm_lbfgs.WarmLBFGS` | JAX autodiff | **the default**: `lbfgs-ad` warm-started from COPRA |
| `"copra"` | {class}`~croak.copra.COPRA` | analytic Wirtinger adjoint | fast, robust from poor guesses |
| `"lbfgs"` | {class}`~croak.lbfgs.LBFGS` | analytic Wirtinger adjoint | flexible; regularisation, phase-only |
| `"lbfgs-ad"` | {class}`~croak.lbfgs_ad.LBFGSAD` | JAX autodiff | widest model coverage: geometric smearing, fitted thickness/τ₀, B-spline basis, temporal penalty |
| `"lm"` | {class}`~croak.lm.LM` | JAX Jacobian | Levenberg–Marquardt; lowest final error |
| `"cma-es"` | {class}`~croak.cmaes.CMAES` | none (global) | population search to escape local minima (needs `croak[evo]`) |

These six are the ones you will reach for most often; croak ships ten
algorithms in total — the rest are numerically-equivalent on-device JAX twins
(`copra-jax`, `lbfgs-hand`, `lbfgs-optx`, `lm-optx`). See
[Choosing an algorithm](../tutorials/02_choosing_an_algorithm.md), the
[solver-selection guide](../howto/solver_selection.md) and
[Retrieval algorithms](../explanation/algorithms.md) for the full set.

By default `retrieve()` shows a live progress bar and a short end-of-run summary;
pass `progress=False` to silence it (or a `callback` to monitor it yourself).

## Dispersive retrieval

To model propagation through a material slab, pass `material`, `thickness`,
`npoints` and the carrier `omega0`. The same call generates the trace and
retrieves from it:

```python
omega0 = croak.maths.wlfreq(260e-9)                     # 260 nm carrier
trace = croak.maketrace(g.omega, delays, ew, "pg",
                       material="SiO2", thickness=10e-6, npoints=20, omega0=omega0)
res = croak.retrieve(trace, g.omega, delays, "pg", algorithm="copra",
                    material="SiO2", thickness=10e-6, npoints=20, omega0=omega0)
```

Dispersive propagation is available for the **PG** and **SD** interactions (whose
signal stays at the fundamental frequency), not **SHG**.

## Initial guesses

The `guess` argument accepts:

- `None` — a random Gaussian-spectrum start (the default);
- a complex spectrum array of length `g.n`;
- a {class}`croak.Pulse` (e.g. another `gaussian_pulse`).

Retrieval is scale-invariant; guesses are normalised internally. Because the
objective is non-convex, the standard practice is to retrieve from several random
guesses and keep the lowest-`error` result.

## The result object

Every algorithm returns a {class}`~croak.result.RetrievalResult` with a uniform
interface:

```python
res.error          # final trace error R
res.errors         # R at each iteration (convergence history)
res.spectrum       # complex spectrum E(omega), centred order
res.field          # complex field E(t)
res.intensity_t    # |E(t)|^2
res.phase_omega    # unwrapped spectral phase
res.wavelength     # absolute wavelength axis (m), using omega0
res.trace          # the retrieved (simulated) trace
```

## Next steps

- Clean and retrieve **measured** data: [Experimental workflow](../tutorials/04_experimental_workflow.md).
- Understand the physics: [The PNPS framework](../explanation/pnps_framework.md).
- Drive everything from a UI: [GUI](../howto/gui.md).
