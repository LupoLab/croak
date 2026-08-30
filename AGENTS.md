# AGENTS.md

## Guiding principles

- There should be one obvious way to do each thing. Prefer the existing pattern over inventing a new one; if you find two ways to do the same thing, consolidate them.
- Optimise for the reader. Clear beats clever — code is read far more often than it is written.
- Small, single-purpose functions with explicit inputs and outputs. Prefer pure functions; push side effects (I/O, global state, mutation) to the edges.
- Make the implicit explicit: explicit arguments over hidden state, explicit types on interfaces, explicit errors over silent fallbacks.
- Fail loudly and early with a clear exception. Never swallow errors or return a sentinel where raising is correct.

## Environment and tooling

- Python is managed with **uv**. Dependencies are declared in `pyproject.toml`; the resolved lockfile is `uv.lock`. Never `pip install` into the environment by hand.
- Add dependencies with `uv add <pkg>` (and `uv add --dev <pkg>` for dev/test tools). Sync the environment with `uv sync`.
- Run everything through the project environment by prefixing commands with `uv run`.
- Target Python [3.14]; keep it consistent with `requires-python` in `pyproject.toml`.

## Commands

- Sync environment: `uv sync`
- Format: `uv run ruff format .`
- Lint (with autofix): `uv run ruff check --fix .`
- Type-check: `uv run pyright`
- Test: `uv run pytest`
- Build docs: `uv run sphinx-build -b html docs docs/_build/html`

## Definition of done

Before treating any change as complete, run these in order and ensure each passes with no new warnings:

1. `uv run ruff format .`
2. `uv run ruff check --fix .`
3. `uv run pyright`
4. `uv run pytest`

Do not declare work finished while any step fails. If a check genuinely cannot pass for a justified reason, say so explicitly rather than suppressing it.

Furthermore, ensure all added functionality is documented both in docstrings *and* in the manual under docs.

## Code style

- Formatting is owned by `ruff format`; do not hand-format or fight the formatter. Line length is the ruff default (88) unless `pyproject.toml` overrides it.
- Linting is owned by `ruff check`. Fix the cause rather than silencing the warning. A suppression must be rule-specific with a reason (`# noqa: E501 — URL must stay on one line`), never a blanket `# noqa`.
- Imports are absolute and sorted by ruff. No wildcard (`from x import *`) imports.
- Naming: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE` for constants. Names state what a thing is or does; avoid abbreviations beyond well-established domain ones.
- Use f-strings for interpolation and `pathlib.Path` for filesystem paths. Reach for the standard-library idiom before adding a dependency.

## Types

- Annotate every public function signature and class attribute. Add types where they clarify intent or catch mistakes; do not annotate obvious locals just to fill space.
- Code must pass `pyright` in the project's configured mode with no new errors.
- Prefer precise types: `Sequence`/`Mapping` over `list`/`dict` for read-only parameters, `X | None` over `Optional[X]`, and built-in generics (`list[int]`). Avoid `Any`; if it is unavoidable, confine it to one place and comment why.
- Use a `@dataclass` (frozen where it can be) for structured data instead of passing loose tuples or dicts around.

## Docstrings

- Every public module, class, and function has a **NumPy-style** docstring (rendered by Sphinx via the napoleon extension). Private helpers get a one-line docstring when the name is not fully self-explanatory.
- The first line is an imperative one-sentence summary. Then include, as applicable: `Parameters`, `Returns`, `Raises`, and `Examples`.
- For numerical quantities, state units and valid ranges in the parameter descriptions.
- Make `Examples` runnable (doctest style) where practical. Describe behaviour and contracts, not implementation details that will drift.

## Comments

- Code comments should be comprehensive. They should explain both the maths and physics as well as coding details.
- Comments cover: rationale, assumptions, references (a paper, an issue link), and non-obvious trade-offs.
- Keep each comment next to what it describes and update it when the code changes. Delete commented-out code — version history is the archive.
- Mark deliberate follow-ups as `# TODO(context): ...` so they are greppable.

## Functions and modules

- One function, one responsibility. If a function needs a paragraph to explain, or has many nested branches, split it.
- Keep parameter lists short. Group related parameters into a small dataclass or config object rather than passing many positional arguments.
- Prefer returning new values over mutating arguments in place. Keep the core logic pure and testable; isolate I/O at the boundaries.
- Organise modules by domain concept, not by a catch-all `utils`. Each module should have a clear, nameable purpose. Declare the public surface explicitly with `__all__` where it helps.

## Tests

- Tests use **pytest** and live in `tests/`, mirroring the package layout. Name them `test_<unit>_<behaviour>`.
- Every public function has tests for the normal case, the edge cases, and the error paths. Add a regression test with every bug fix.
- Tests are deterministic, isolated, and fast: use fixtures for setup, `pytest.mark.parametrize` instead of copy-pasted cases, and no network access or hidden global state.
- For numerical code, assert with explicit tolerances (`numpy.testing.assert_allclose`, `pytest.approx`) and test invariants and conservation laws, not only point values.
- Write the test alongside the code; a feature is not done until it is tested.

## Documentation

- Docs are built with **Sphinx**, and the API reference is generated from docstrings — so the docstring is the source of truth. Keep the `README.md` and any usage guide current with behaviour changes.
- Record notable changes in `CHANGELOG.md` (Keep a Changelog style).
- All functionality of this package must be described in the user manual under the docs/ folder. This should include background explanations and context, how the functionality works, and examples
- When adding functionality, always add documentation.
- When working on something that does not appear to be documented, check this and add appropriate documentation.

## Git

- Make clean, logical git commits with descriptive but not overly verbose commit messages.
- Prefer more frequenct clean commits over large big ones.
- *Never* push.
- You can fetch, pull, branch when instructed to do so. If you want to do this, ask.
- Do not make releases or change release versions
