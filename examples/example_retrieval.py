"""End-to-end FROG retrieval examples for the croak package.

Demonstrates the uniform API across pulse shapes, interaction geometries, noise
levels and retrieval algorithms, including a dispersive (pseudo-propagation)
example. Run with::

    uv run python examples/example_retrieval.py

Prints the final FROG error and the recovered temporal FWHM for each case. No
plotting is required; if matplotlib is available a summary figure is saved.
"""

from __future__ import annotations

import numpy as np

import croak
from croak.maths import fwhm as measure_fwhm
from croak.maths import wlfreq


def recovered_fwhm(res: croak.RetrievalResult) -> float:
    """Temporal intensity FWHM (fs) of a retrieved pulse, peak-centred."""
    intensity = res.intensity_t
    return measure_fwhm(res.t, intensity) * 1e15


def banner(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def thin_examples() -> None:
    """Gaussian and sech pulses across all three interactions, clean + noisy."""
    banner("Thin-medium retrieval (Gaussian & sech, PG/SD/SHG)")
    g = croak.Grid(128, dt=0.3e-15)
    delays = np.linspace(-18e-15, 18e-15, 87)
    rng = np.random.default_rng(0)

    pulses = {
        "Gaussian 2.5 fs, GDD": croak.gaussian_pulse(g, 2.5e-15, phases=[4e-30]),
        "sech 3 fs": croak.sech_pulse(g, 3.0e-15),
    }
    print(
        f"{'pulse':22s} {'inter':5s} {'algo':6s} {'noise':>6s} "
        f"{'R':>10s} {'FWHM/fs':>9s}"
    )
    for label, ew in pulses.items():
        true_fwhm = measure_fwhm(g.t, np.abs(g.ifft(ew)) ** 2) * 1e15
        for interaction in ("pg", "sd", "shg"):
            clean = croak.maketrace(g.omega, delays, ew, interaction)
            for sigma in (0.0, 0.01):
                trace = clean
                if sigma:
                    trace = np.clip(
                        clean + sigma * clean.max() * rng.standard_normal(clean.shape),
                        0.0,
                        None,
                    )
                algo = "copra" if sigma == 0 else "lbfgs"
                near = croak.gaussian_pulse(g, 1.8e-15)
                res = croak.retrieve(
                    trace,
                    g.omega,
                    delays,
                    interaction,
                    algorithm=algo,
                    guess=near,
                    maxiters=250,
                )
                print(
                    f"{label:22s} {interaction:5s} {algo:6s} {sigma:6.0%} "
                    f"{res.error:10.2e} {recovered_fwhm(res):9.2f}"
                )
        print(f"{'  (true FWHM = ' + f'{true_fwhm:.2f} fs)':22s}")


def dispersive_example() -> None:
    """Few-fs UV pulse through a fused-silica slab: D-COPRA and dispersive L-BFGS."""
    banner("Dispersive retrieval (UV pulse, 10 um SiO2 slab)")
    g = croak.Grid(128, dt=0.5e-15)
    omega0 = wlfreq(260e-9)  # 260 nm carrier
    ew = croak.gaussian_pulse(g, 4.0e-15, phases=[6e-30])
    delays = np.linspace(-22e-15, 22e-15, 80)
    trace = croak.maketrace(
        g.omega,
        delays,
        ew,
        "pg",
        material="SiO2",
        thickness=10e-6,
        npoints=20,
        omega0=omega0,
    )
    true_fwhm = measure_fwhm(g.t, np.abs(g.ifft(ew)) ** 2) * 1e15
    near = croak.gaussian_pulse(g, 3.0e-15)

    print(f"true temporal FWHM = {true_fwhm:.2f} fs\n")
    print(f"{'algorithm':10s} {'R':>10s} {'FWHM/fs':>9s}")
    for algorithm in ("copra", "lbfgs"):
        res = croak.retrieve(
            trace,
            g.omega,
            delays,
            "pg",
            algorithm=algorithm,
            guess=near,
            maxiters=300,
            material="SiO2",
            thickness=10e-6,
            npoints=20,
            omega0=omega0,
        )
        print(f"{algorithm:10s} {res.error:10.2e} {recovered_fwhm(res):9.2f}")


def main() -> None:
    thin_examples()
    dispersive_example()
    print("\nDone.")


if __name__ == "__main__":
    main()
