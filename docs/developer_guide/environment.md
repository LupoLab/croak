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

Run the required checks in order:

```bash
uv run ruff format .
uv run ruff check --fix .
uv run pyright
uv run pytest
```

Tests live in `tests/` and follow the package's domain structure. Numerical tests
should state their tolerances and check physical or mathematical invariants as
well as individual values. Document public behaviour in NumPy-style docstrings
and in the manual, and record notable changes in `CHANGELOG.md`.

See [numerical validation](validation.md) for model and gradient checks, and
[building the documentation](documentation.md) for executed tutorial caches.
