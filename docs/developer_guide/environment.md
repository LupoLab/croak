# Development environment and checks

Use Python 3.14 and `uv` for development. Dependencies are declared in
`pyproject.toml` and resolved in `uv.lock`. From a source checkout, install the
development and documentation dependencies with:

```bash
uv sync --group docs
```

The default development group includes the test, formatting, linting, and type
checking tools, as well as Qt for the GUI tests. Run commands through `uv run`
to use the project environment.

## Check a change

One command runs the whole definition-of-done gate, in order, exactly as CI
does:

```bash
uv run poe check
```

It runs `ruff check --fix`, `ruff format`, `ruff check` (what autofix could not
fix), `pyright`, and finally `poe cov`: the test suite on every core
(pytest-xdist), doctests included, measuring statement **and** branch coverage
of the whole `croak` package, GUI included, and failing below 100%. The XML
report is written to `build/coverage/coverage.xml`.

GitHub Actions runs this gate on `ubuntu-latest`, `windows-latest`, and
`macos-latest`, using Python 3.14 and the committed lockfile on each platform.
Every platform must independently meet the 100% coverage threshold, including
GUI tests and doctests. Linux installs Qt system libraries and runs under Xvfb;
Windows and macOS run the gate directly. Qt uses its offscreen backend on all
three platforms. A failure on one platform leaves the others running so all
failures can be inspected together.

The workflow runs for pull requests and pushes to `main` or `master`, and can
also be started manually from the Actions tab. CI fails if the gate's lint
autofixes or formatting leave a tracked-file diff.

For focused work run single files serially, without coverage:

```bash
uv run pytest tests/test_preprocess.py
uv run poe test          # the whole suite on every core, no coverage
```

The suite is strict, and each rule is set in `pyproject.toml`:

- **Warnings are errors.** A test that expects a warning asserts it with
  `pytest.warns`; a warning nobody expects is a bug to fix, not to filter.
- **Random order.** pytest-randomly shuffles the tests (the seed is printed;
  rerun with `-p randomly --randomly-seed=N`), and xdist spreads modules over
  workers, so tests must not depend on each other or share writable files.
  Write under `tmp_path`.
- **Doctests run.** Every `Examples` section is executed, so examples must be
  deterministic (seed every random generator).
- **Full coverage.** New code arrives with tests that exercise its branches and
  error paths, with assertions about its behaviour. Code that cannot be
  reached is removed rather than excluded; `# pragma: no cover` is reserved for
  code that cannot run under test at all (an `if __name__ == "__main__":`
  block, a guarded optional import) and carries its reason on the same line.
  The threshold and the exclusions are not lowered to make a change pass.

GUI tests live in `tests/gui/` and run offscreen through pytest-qt; the shared
`tests/conftest.py` points `QSettings` at a per-test directory for them. The
`evo` extra (evosax) is part of the development group so the `cma-es` retriever
is tested.

Tests live in `tests/` and follow the package's domain structure. Numerical tests
should state their tolerances and check physical or mathematical invariants as
well as individual values. Document public behaviour in NumPy-style docstrings
and in the manual, and record notable changes in `CHANGELOG.md`.

See [numerical validation](validation.md) for model and gradient checks, and
[building the documentation](documentation.md) for executed tutorial caches.
