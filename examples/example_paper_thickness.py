"""Retrieve the companion paper's 1 fs DUV trace: dispersion in the medium matters.

The dataset (``examples/data/tgfrog_sim_1fs_uvfs.h5``) is a reduced cut of the
first-principles 3D TG-FROG instrument simulation behind the paper's central
result: a transform-limited 1 fs pulse at 260 nm (1.03 fs after the input
mask), measured through 4, 9.5 and 40 um of fused silica. Nothing in the trace
is synthetic to croak — the dispersion, phase matching, geometric smearing and
chromatic collection all come out of the propagation physics.

At each thickness the trace is retrieved twice:

* with the **extended model** — dispersive propagation through the substrate
  plus the geometric BOXCARS smearing kernel (the paper protocol), and
* with the **standard thin-medium model** every conventional FROG code uses.

Both fit the trace well, but only the extended model returns the correct
pulse: the standard model absorbs the unmodelled substrate dispersion into the
pulse and broadens with thickness, reaching ~2.4x the true duration at 40 um
while its trace error stays below 1% — so the trace error alone does not
certify the retrieval.

Run::

    uv run python examples/example_paper_thickness.py

Takes a couple of minutes (six retrievals). Writes ``paper_thickness.png``.
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
from croak.truth_metrics import truth_errors  # noqa: E402 - see note above

DATA = "examples/data/tgfrog_sim_1fs_uvfs.h5"
THICKNESSES_UM = [4.0, 9.5, 40.0]
TRUE_FWHM_FS = 1.03  # the mask-vignetted gate beamlet (see the data README)

# The retrieval band and delay window of the paper's validation study. The
# scan's delay axis has a uniform 0.25 fs core to +-25 fs and coarser wing
# points beyond; the crop keeps the core, which the smearing kernel's
# delay-axis convolution needs.
PREPROC = PreprocParams(
    lam_min_nm=131.0,
    lam_max_nm=793.0,
    lam_bound_min_nm=65.5,
    lam_bound_max_nm=1189.5,
    trange_fs=100.0,
    tau_min_fs=-25.0,
    tau_max_fs=25.0,
    lamm_min_nm=131.0,
    lamm_max_nm=793.0,
    lamm_bound_min_nm=131.0,
    lamm_bound_max_nm=793.0,
    taum_min_fs=-25.0,
    taum_max_fs=25.0,
    tau_bound_min_fs=-25.0,
    tau_bound_max_fs=25.0,
    defringe=True,
)


def load(z_um: float):
    """Load one thickness slice, referenced to the gating beamlet spectrum."""
    sim = SimulatedLoadParams(
        frog_path=DATA,
        z_thickness_um=z_um,
        lam_min_nm=131.0,
        third_order=True,
        third_order_exp=4.0,  # the marginal-check Auto refines this per trace
        use_spectrum=True,
        spectrum_source="beamlet",
    )
    data = assemble_simulated_load_data(sim)
    return data, build_tracedata(PREPROC, data)


def retrieve(td, z_um: float, extended: bool):
    """One retrieval with the paper protocol (or the thin-medium baseline)."""
    rp = RetrieveParams(
        solver="warm-lbfgs",
        maxiters=300,
        perfect_init=True,
        perfect_init_fs=1.0,
        dispersive=extended,
        material="SiO2",  # Malitson Sellmeier, as in the simulation
        thickness_um=z_um,
        npoints=max(1, round(z_um)),  # one depth node per micrometre
        smearing=extended,
        smear_hole_diameter_mm=1.0,  # the simulated BOXCARS mask geometry
        smear_hole_spacing_mm=1.0,
        fit_tau0=True,
    )
    return run_retrieval(rp, td)


def main() -> None:
    """Run both models at every thickness and plot the comparison."""
    results: dict[str, list] = {"extended": [], "standard": []}
    truth = None
    for z in THICKNESSES_UM:
        data, td = load(z)
        truth = data["truth"]
        for label in ("extended", "standard"):
            res = retrieve(td, z, extended=(label == "extended"))
            pr = croak.process_result(res, measured=td.trace)
            errs = truth_errors(res, truth)
            results[label].append((z, res, pr))
            print(
                f"z = {z:5.1f} um  {label:8s}  R = {100 * res.error:.3f}%  "
                f"FWHM = {pr.fwhm_retr * 1e15:.2f} fs  "
                f"eps_It = {errs.eps_It:.3f}"
            )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")

    # Retrieved duration against thickness, with the known truth.
    for label, marker in (("extended", "o"), ("standard", "s")):
        zs = [z for z, _, _ in results[label]]
        fw = [pr.fwhm_retr * 1e15 for _, _, pr in results[label]]
        ax1.plot(zs, fw, marker + "-", label=f"{label} model")
    ax1.axhline(TRUE_FWHM_FS, color="k", ls=":", label="true pulse")
    ax1.set_xlabel("substrate thickness (um)")
    ax1.set_ylabel("retrieved FWHM (fs)")
    ax1.legend()

    # Temporal intensity at the thickest substrate, over the truth.
    assert truth is not None
    ax2.fill_between(truth.t * 1e15, truth.It, color="0.8", label="true pulse")
    for label, style in (("extended", "-"), ("standard", "--")):
        _, res, pr = results[label][-1]
        # centre the retrieved peak at t = 0, the truth's convention (a pure
        # time translation, which the trace does not determine)
        t0 = pr.t_over[pr.It_retr.argmax()]
        ax2.plot((pr.t_over - t0) * 1e15, pr.It_retr, style, label=f"{label} model")
    ax2.set_xlim(-8, 8)
    ax2.set_xlabel("time (fs)")
    ax2.set_ylabel("intensity (norm.)")
    ax2.set_title(f"{THICKNESSES_UM[-1]:.0f} um substrate")
    ax2.legend()

    fig.savefig("paper_thickness.png", dpi=150)
    print("wrote paper_thickness.png")


if __name__ == "__main__":
    main()
