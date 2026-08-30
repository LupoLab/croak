"""Propagate a measured substrate-thickness uncertainty onto the retrieved FWHM.

For a *dispersive* retrieval, croak fits the trace through a forward model that
propagates the field through a substrate of an assumed thickness. When that thickness
is an independently measured input with its own error bar — e.g. ``9.952 ± 0.539`` µm
— the retrieved pulse, and its temporal-intensity FWHM, inherit a **systematic**
uncertainty. For short, deep-UV pulses this substrate term routinely dominates the
*statistical* bootstrap (which is blind to it: every bootstrap replicate reuses the
same fixed thickness).

This example builds a small synthetic dispersive experiment, retrieves it, and then:

* runs :func:`croak.thickness_bootstrap` to get the FWHM error from the thickness prior,
* runs a statistical :func:`croak.parametric_bootstrap` for comparison,
* combines the two independent error bars in quadrature, and
* saves the thickness-sensitivity figure.

Run::

    uv run python examples/example_thickness_uncertainty.py

Writes ``thickness_uncertainty.png`` to the working directory. The bottom of the file
sketches how to run the same analysis on your own measurement via an ``options.toml``.
"""

from __future__ import annotations

from math import hypot

import matplotlib
import numpy as np

matplotlib.use("Agg")

import croak
from croak.maths import wlfreq

# --- substrate prior (the measured glass thickness and its 1σ uncertainty) ----
# The motivating measurement is a *thin* DUV substrate, ~9.952 ± 0.539 µm (≈5.4 % 1σ).
# This self-contained demo instead uses a thicker slab at a longer (near-UV) wavelength
# so the retrieval is numerically clean and fast while the thickness term still clearly
# dominates the statistical one. The physics is identical: near 230 nm the GVD of
# silica is ~10× higher, so the real DUV case reaches the same regime with only ~10 µm
# of glass.
CENTRAL_THICKNESS = 100.0e-6  # L0 (m)
THICKNESS_SIGMA = 5.0e-6  # σ_L (m) — ≈5 %, matching the DUV case's relative error
MATERIAL = "SiO2-Franta"
NPOINTS = 20  # depth-quadrature nodes through the slab


def make_dispersive_experiment() -> dict:
    """Synthesise a short near-UV pulse propagated through ``CENTRAL_THICKNESS``."""
    grid = croak.Grid(128, dt=0.3e-15)
    omega0 = float(wlfreq(300e-9))  # near-UV carrier
    ew = croak.gaussian_pulse(grid, 3e-15)  # transform-limited
    delays = np.linspace(-45e-15, 45e-15, 80)
    trace = croak.maketrace(
        grid.omega,
        delays,
        ew,
        "pg",
        material=MATERIAL,
        thickness=CENTRAL_THICKNESS,
        npoints=NPOINTS,
        omega0=omega0,
    )
    return {"grid": grid, "omega0": omega0, "ew": ew, "delays": delays, "trace": trace}


def main() -> None:
    exp = make_dispersive_experiment()
    grid, omega0, delays = exp["grid"], exp["omega0"], exp["delays"]
    # A touch of detector noise so the *statistical* bootstrap is non-trivial and the
    # comparison with the thickness systematic is meaningful.
    measured = croak.NoiseModel(sigma=0.005).draw(
        exp["trace"], np.random.default_rng(7)
    )

    # --- base dispersive retrieval at the central thickness L0 ---
    result = croak.retrieve(
        measured,
        grid.omega,
        delays,
        "pg",
        guess=exp["ew"],
        omega0=omega0,
        material=MATERIAL,
        thickness=CENTRAL_THICKNESS,
        npoints=NPOINTS,
        maxiters=150,
        progress=False,
    )
    pr = croak.process_result(result, measured=measured)
    print(
        f"Base retrieval: R = {result.error:.4%},  FWHM = {pr.fwhm_retr / 1e-15:.2f} fs"
    )

    # --- systematic: propagate the measured substrate-thickness prior ---
    thick = croak.thickness_bootstrap(
        result,
        measured,
        central_thickness=CENTRAL_THICKNESS,
        thickness_sigma=THICKNESS_SIGMA,
        material=MATERIAL,
        npoints=NPOINTS,
        n_resamples=40,
        maxiters=120,
        collect_profiles=True,
        rng=np.random.default_rng(0),
    )
    print(thick.summary())

    # --- statistical: parametric bootstrap for comparison (noise from the residual) ---
    stat = croak.parametric_bootstrap(
        result,
        noise=croak.noise_from_residual(measured, result),
        material=MATERIAL,
        thickness=CENTRAL_THICKNESS,
        npoints=NPOINTS,
        n_resamples=40,
        maxiters=120,
        rng=np.random.default_rng(1),
    )
    print(stat.summary())

    # --- total error budget: independent terms add in quadrature ---
    total_pm = hypot(thick.plus_minus, stat.plus_minus)
    print(
        f"Total: FWHM = {thick.point_estimate / 1e-15:.2f} "
        f"± {total_pm / 1e-15:.2f} fs "
        f"(stat ⊕ thickness; thickness {thick.plus_minus / 1e-15:.2f} fs, "
        f"stat {stat.plus_minus / 1e-15:.2f} fs)"
    )

    fig = croak.plot_thickness_sensitivity(thick)
    fig.savefig("thickness_uncertainty.png", dpi=110)
    print("Wrote thickness_uncertainty.png")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Running this on your own data (an options.toml saved from the GUI)
# ---------------------------------------------------------------------------
# The block below is illustrative pseudocode — uncomment and adapt the paths. It loads
# a saved options file, runs the same load → clean → retrieve pipeline the GUI uses,
# and then propagates *your* measured substrate thickness and its 1σ.
#
#     import croak
#     opts = croak.save.load_options("options.toml")  # [load]/[preproc]/[retrieve]
#     # ... build a TraceData `td` from opts["load"]/opts["preproc"] via
#     #     croak.io + croak.load_and_clean (see examples/example_workflow.py), then:
#     rp = opts["retrieve"]
#     result = croak.retrieve_from_tracedata(
#         td,
#         algorithm=rp["solver"],
#         material=rp["material"],
#         thickness=9.952e-6,          # your *measured* central thickness (m)
#         npoints=rp["npoints"],
#         maxiters=rp["maxiters"],
#     )
#     u = croak.thickness_bootstrap(
#         result, td.trace,
#         central_thickness=9.952e-6,  # L0 (m)
#         thickness_sigma=0.539e-6,    # σ_L (m)
#         material=rp["material"],
#         npoints=rp["npoints"],
#         maxiters=rp["maxiters"],
#         n_resamples=300,
#         collect_profiles=True,
#     )
#     print(u.summary())               # FWHM = X ± Y fs (1σ from substrate thickness …)
#
# In the GUI this is stage 5 → method "thickness": set "Thickness σ (µm)" and click
# Estimate.
