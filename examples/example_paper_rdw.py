"""Retrieve a simulated single-cycle pulse: the companion paper's final test case.

The dataset (``examples/data/tgfrog_sim_rdw_duv.h5``) is the paper's
end-to-end validation: the resonant-dispersive-wave (RDW) emission of a
hollow-capillary-fibre source — a 1.06 fs single-cycle deep-UV pulse with a
structured spectrum and trailing satellites — simulated at the generation
stage, injected into the first-principles 3D instrument simulation, and
recorded exactly as a measurement would be (here through the open 2.0 mm
collection hole, after 9.5 um of fused silica).

Two retrievals of the same trace:

* the **extended model** with the smearing-kernel scale and the delay zero
  fitted from the data — the paper's route for this case;
* the **standard thin-medium model**.

The standard model fits the trace to a low trace error and returns a smooth
pulse about 2.4x too long. The extended
model recovers the pulse — main peak, satellites and structured spectrum.

Run::

    uv run python examples/example_paper_rdw.py

Takes a few minutes (two retrievals on a 400-point grid). Writes
``paper_rdw.png``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 - must follow matplotlib.use above

import croak  # noqa: E402 - imports pyplot transitively
from croak.session import (  # noqa: E402 - see note above
    PreprocParams,
    RetrieveParams,
    SimulatedLoadParams,
    assemble_simulated_load_data,
    build_tracedata,
    run_retrieval,
)

DATA = "examples/data/tgfrog_sim_rdw_duv.h5"
Z_UM = 9.5
TRUE_FWHM_FS = 1.06

# The paper's band for this scan (the RDW spectrum reaches further to the red
# than the Gaussian test pulse), with the delay crop at the scan's uniform core.
PREPROC = PreprocParams(
    lam_min_nm=148.0,
    lam_max_nm=830.0,
    lam_bound_min_nm=74.0,
    lam_bound_max_nm=1245.0,
    trange_fs=150.0,
    tau_min_fs=-38.0,
    tau_max_fs=38.0,
    lamm_min_nm=148.0,
    lamm_max_nm=830.0,
    lamm_bound_min_nm=148.0,
    lamm_bound_max_nm=830.0,
    taum_min_fs=-38.0,
    taum_max_fs=38.0,
    tau_bound_min_fs=-38.0,
    tau_bound_max_fs=38.0,
    defringe=True,
)


def main() -> None:
    """Retrieve with both models and plot the comparison against the truth."""
    sim = SimulatedLoadParams(
        frog_path=DATA,
        z_thickness_um=Z_UM,
        lam_min_nm=148.0,
        lam_max_nm=830.0,
        third_order=True,
        third_order_exp=4.0,
        use_spectrum=True,
        spectrum_source="beamlet",
    )
    data = assemble_simulated_load_data(sim)
    td = build_tracedata(PREPROC, data)
    truth = data["truth"]

    extended = RetrieveParams(
        solver="warm-lbfgs",
        maxiters=300,
        dispersive=True,
        material="SiO2",
        thickness_um=Z_UM,
        npoints=10,
        smearing=True,
        smear_hole_diameter_mm=1.0,
        smear_hole_spacing_mm=1.0,
        fit_smearing=True,  # the kernel scale is fitted from the data
        fit_tau0=True,
    )
    standard = RetrieveParams(solver="warm-lbfgs", maxiters=300)

    out = {}
    for label, rp in (("extended (fitted kernel)", extended), ("standard", standard)):
        res = run_retrieval(rp, td)
        pr = croak.process_result(res, measured=td.trace)
        out[label] = pr
        print(
            f"{label:24s}  R = {100 * res.error:.3f}%  "
            f"FWHM = {pr.fwhm_retr * 1e15:.2f} fs  (truth {TRUE_FWHM_FS} fs)"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")

    ax1.fill_between(truth.t * 1e15, truth.It, color="0.8", label="true pulse")
    for label, style in zip(out, ("-", "--"), strict=True):
        pr = out[label]
        t0 = pr.t_over[pr.It_retr.argmax()]
        ax1.plot((pr.t_over - t0) * 1e15, pr.It_retr, style, label=label)
    ax1.set_xlim(-6, 12)
    ax1.set_xlabel("time (fs)")
    ax1.set_ylabel("intensity (norm.)")
    ax1.legend()

    if truth.lam is not None:
        ax2.fill_between(truth.lam * 1e9, truth.Iw, color="0.8", label="true spectrum")
    for label, style in zip(out, ("-", "--"), strict=True):
        pr = out[label]
        ax2.plot(pr.wavelength * 1e9, pr.Ilam, style, label=label)
    ax2.set_xlim(200, 340)
    ax2.set_xlabel("wavelength (nm)")
    ax2.set_ylabel("spectral intensity (norm.)")
    ax2.legend()

    fig.savefig("paper_rdw.png", dpi=150)
    print("wrote paper_rdw.png")


if __name__ == "__main__":
    main()
