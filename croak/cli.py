"""The ``croak`` command-line interface: replay a saved retrieval session.

``croak replay options.toml`` reruns a session headlessly (load → preprocess →
retrieve → dispersion → process), saving ``result.h5`` and the ``options.toml``
used. Point it at one or more new dataset folders with ``--base-dir`` to rerun
the same settings on similarly-laid-out acquisitions (calibration files are kept
fixed; see :func:`croak.session.retarget.retarget`).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .save import save_result, save_uncertainty
from .scripting import generate_script
from .session import (
    DEFAULT_STAGES,
    SessionOptions,
    SessionResult,
    retarget,
    run_session,
)

__all__ = ["main"]


def _resolve_outdir(out: str | None, base: str | None, options_path: str) -> Path:
    """Pick the output directory for one replay target.

    Priority: an explicit ``--out`` (with a per-base subfolder when batching) →
    the retargeted dataset folder → the options file's own folder.
    """
    if out is not None:
        return Path(out) if base is None else Path(out) / Path(base).name
    if base is not None:
        return Path(base)
    return Path(options_path).resolve().parent


def _save_session(
    res: SessionResult, options: SessionOptions, outdir: Path, *, force: bool
) -> Path:
    """Write the final pulse, the options used and any uncertainty to ``outdir``."""
    outdir.mkdir(parents=True, exist_ok=True)
    # The "final" pulse is the dispersion-compensated one when stage 4 ran.
    final = res.dispersed if res.dispersed is not None else res.result
    if final is None:
        raise RuntimeError("replay produced no retrieval result to save")
    measured = res.tracedata.trace if res.tracedata is not None else None
    result_path = outdir / "result.h5"
    save_result(
        final, str(result_path), processed=res.processed, measured=measured, force=force
    )
    options.to_toml(str(outdir / "options.toml"))
    if res.uncertainty is not None:
        save_uncertainty(
            {res.uncertainty.method: res.uncertainty}, str(result_path), force=True
        )
    return result_path


def _report(label: str, res: SessionResult, path: Path) -> None:
    """Print a one-line summary of a completed replay."""
    parts = [f"[croak] {label}:"]
    if res.result is not None:
        parts.append(f"R = {res.result.error:.4%}")
    if res.processed is not None:
        parts.append(f"FWHM = {res.processed.fwhm_retr / 1e-15:.2f} fs")
    if res.uncertainty is not None:
        parts.append(f"± {res.uncertainty.plus_minus / 1e-15:.2f} fs")
    parts.append(f"→ {path}")
    print("   ".join(parts))


def _cmd_replay(args: argparse.Namespace) -> int:
    options = SessionOptions.from_toml(args.options)
    stages = DEFAULT_STAGES + (("uncertainty",) if args.uncertainty else ())
    targets: list[str | None] = list(args.base_dir) if args.base_dir else [None]

    progress: Callable[[int, float, float], None] | None = None
    if args.verbose:

        def _progress(iteration: int, error: float, best: float) -> None:
            print(f"  iter {iteration:4d}: R = {error:.4%}", file=sys.stderr)

        progress = _progress

    failures = 0
    for base in targets:
        label = Path(base).name if base is not None else Path(args.options).stem
        outdir = _resolve_outdir(args.out, base, args.options)
        rng = np.random.default_rng(args.seed) if args.seed is not None else None
        print(f"[croak] replaying {label} …")
        try:
            # Inside the try: retargeting a non-experimental session raises, and
            # one bad target must not abort the batch either.
            opts = retarget(options, base) if base is not None else options
            res = run_session(opts, stages=stages, rng=rng, retrieve_callback=progress)
            path = _save_session(res, opts, outdir, force=args.force)
        except Exception as exc:  # one bad dataset must not abort a batch
            failures += 1
            print(f"[croak] {label}: FAILED — {exc}", file=sys.stderr)
            continue
        _report(label, res, path)
    return 1 if failures else 0


def _cmd_script(args: argparse.Namespace) -> int:
    options = SessionOptions.from_toml(args.options)
    name = Path(args.out).name if args.out else "replay.py"
    try:
        source = generate_script(
            options,
            base_dir=args.base_dir,
            output=args.output,
            seed=args.seed,
            script_name=name,
        )
    except ValueError as exc:  # synthetic session / non-retargetable entry
        print(f"[croak] {exc}", file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).write_text(source)
        print(f"[croak] wrote {args.out}")
    else:
        sys.stdout.write(source)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="croak", description="Replay croak retrieval sessions headlessly."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    script = sub.add_parser(
        "script", help="generate an editable Python script from an options.toml"
    )
    script.add_argument("options", help="path to the saved options.toml")
    script.add_argument(
        "-o", "--out", metavar="FILE", help="write the script here (default: stdout)"
    )
    script.add_argument(
        "--base-dir",
        metavar="DIR",
        help="retarget the script's load paths to this dataset folder",
    )
    script.add_argument(
        "--output",
        metavar="PATH",
        default="result.h5",
        help="result.h5 path the generated script writes to (default: result.h5)",
    )
    script.add_argument(
        "--seed", type=int, default=0, help="RNG seed embedded in the script"
    )
    script.set_defaults(func=_cmd_script)

    replay = sub.add_parser(
        "replay", help="replay a saved options.toml (optionally on new datasets)"
    )
    replay.add_argument("options", help="path to the saved options.toml")
    replay.add_argument(
        "--base-dir",
        action="append",
        metavar="DIR",
        help="rerun against this dataset folder (repeatable for a batch); "
        "calibration files are kept fixed",
    )
    replay.add_argument(
        "--out",
        metavar="DIR",
        help="output directory (default: the dataset folder, or the options "
        "file's folder); batched runs get a per-dataset subfolder",
    )
    replay.add_argument(
        "--uncertainty",
        action="store_true",
        help="also run the FWHM-uncertainty bootstrap (slow)",
    )
    replay.add_argument(
        "--seed", type=int, default=None, help="RNG seed for a reproducible run"
    )
    replay.add_argument(
        "--force", action="store_true", help="overwrite an existing result.h5"
    )
    replay.add_argument(
        "--verbose", action="store_true", help="print per-iteration retrieval progress"
    )
    replay.set_defaults(func=_cmd_replay)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``croak`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
