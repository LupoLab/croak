"""Clean a DUV TG-FROG trace: de-fringe, then arPLS baseline removal.

The full cleaning chain. Takeda de-fringing
(:func:`croak.preprocess.defringe_carrier`) removes the interferometric cross
term, then :func:`croak.preprocess.arpls_baseline` removes the broad,
*asymmetric* leakage background ``|E_b|²`` along delay — strong before the beams
separate and weak after — without touching the compact τ≈0 signal. Run::

    uv run python examples/example_arpls_baseline.py

The trace is synthetic (see :mod:`synthetic_duv_trace`), so the signal and the
background are both known exactly and the example reports the true recovery
error. Writes ``baseline_check.png``; read it as:

* **row cuts** (key panel) — the dashed baseline should ride *under* the
  de-fringed data, following the asymmetric shelf, without bowing up into the
  τ≈0 peak;
* **2-D maps** de-fringed → baseline → cleaned — the background should vanish
  while the compact signal stays put;
* **marginals** — the cleaned trace should land on the true signal.

Tuning ``SMOOTHNESS``: if the baseline bows into the peak (the cleaned peak comes
out too low) raise it, and/or set ``TAU_EXCLUDE_FS`` to hold out the peak; if it
fails to follow the shelf in the wings, leaving residual background, lower it.
The useful range for this trace is roughly 4–10. Note the best value depends on
how well separated the background and signal are in delay: a stiffer baseline
than the background's own curvature cannot follow it, and the leftover shows up
as a residual pedestal in the marginals long before it is visible in a row cut.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from synthetic_duv_trace import make_fringed_trace

from croak.preprocess import arpls_baseline, defringe_carrier

# arPLS stiffness (larger ⇒ flatter baseline) and an optional peak hold-out window.
SMOOTHNESS = 5.0
TAU_EXCLUDE_FS = 0.0  # e.g. 8.0 to hold out |τ| < 8 fs if the peak is sucked up


def main() -> None:
    """Build, de-fringe, baseline-subtract, and write the comparison figure."""
    ft = make_fringed_trace()

    defr = defringe_carrier(ft.measured, ft.lam, ft.tau, carrier_fraction=0.5)
    baseline = arpls_baseline(
        defr,
        smoothness=SMOOTHNESS,
        tau=ft.tau,
        tau_exclude=TAU_EXCLUDE_FS * 1e-15 if TAU_EXCLUDE_FS > 0 else None,
    )
    cleaned = defr - baseline

    peak = ft.signal.max()
    print(f"trace: {ft.lam.size} wavelengths × {ft.tau.size} delays")
    print(f"RMS baseline error: {np.std(baseline - ft.baseline) / peak:.2%} of peak")
    print(f"RMS signal error:   {np.std(cleaned - ft.signal) / peak:.2%} of peak")
    print(f"peak height error:  {abs(cleaned.max() - peak) / peak:.2%}")

    _plot(ft, defr, baseline, cleaned)


def _plot(ft, defr, baseline, cleaned) -> None:
    """Six-panel de-fringe → baseline diagnostic; writes ``baseline_check.png``."""
    tau_fs = ft.tau / 1e-15
    lam_nm = ft.lam / 1e-9

    marg_lam = defr.sum(axis=1)
    sig = np.where(marg_lam > 0.05 * marg_lam.max())[0]
    lo, hi = lam_nm[sig[0]], lam_nm[sig[-1]]
    pad = 0.25 * (hi - lo)
    band = (lam_nm >= lo - pad) & (lam_nm <= hi + pad)
    vmax = float(defr[band].max())

    fig, ax = plt.subplots(2, 3, figsize=(15, 8))

    ax[0, 0].pcolormesh(tau_fs, lam_nm, defr, shading="auto", vmin=0, vmax=vmax)
    ax[0, 0].set_title("de-fringed a(λ,τ)")
    ax[0, 0].set_ylabel("λ (nm)")
    ax[0, 1].pcolormesh(tau_fs, lam_nm, baseline, shading="auto", vmin=0, vmax=vmax)
    ax[0, 1].set_title("estimated baseline |E_b|²")
    ax[0, 2].pcolormesh(tau_fs, lam_nm, cleaned, shading="auto", vmin=0, vmax=vmax)
    ax[0, 2].set_title("cleaned = de-fringed − baseline")
    for a in ax[0]:
        a.set_ylim(lo - pad, hi + pad)
        a.set_xlabel("delay (fs)")

    # KEY panel: row cuts with the baseline riding under the de-fringed data.
    rows = np.argsort(defr.sum(axis=1))[-3:]
    for i in rows:
        (line,) = ax[1, 0].plot(tau_fs, defr[i], lw=0.8, alpha=0.6)
        ax[1, 0].plot(tau_fs, baseline[i], "--", lw=1.2, color=line.get_color())
        ax[1, 0].plot(
            tau_fs,
            cleaned[i],
            lw=1.8,
            color=line.get_color(),
            label=f"{lam_nm[i]:.0f} nm",
        )
    ax[1, 0].set_title(
        "row cuts: de-fringed (thin) · baseline (dashed) · cleaned (thick)"
    )
    ax[1, 0].set_xlabel("delay (fs)")
    ax[1, 0].legend(fontsize=7)

    ax[1, 1].plot(tau_fs, defr.sum(axis=0), label="de-fringed")
    ax[1, 1].plot(tau_fs, cleaned.sum(axis=0), label="cleaned")
    ax[1, 1].plot(tau_fs, ft.signal.sum(axis=0), "k--", lw=1.0, label="true signal")
    ax[1, 1].set_title("delay marginal")
    ax[1, 1].set_xlabel("delay (fs)")
    ax[1, 1].legend()

    ax[1, 2].plot(lam_nm, defr.sum(axis=1), label="de-fringed")
    ax[1, 2].plot(lam_nm, cleaned.sum(axis=1), label="cleaned")
    ax[1, 2].plot(lam_nm, ft.signal.sum(axis=1), "k--", lw=1.0, label="true signal")
    ax[1, 2].set_title("wavelength marginal")
    ax[1, 2].set_xlabel("λ (nm)")
    ax[1, 2].legend()

    fig.tight_layout()
    fig.savefig("baseline_check.png", dpi=120)
    print("wrote baseline_check.png")


if __name__ == "__main__":
    main()
