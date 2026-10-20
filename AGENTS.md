# AGENTS.md

Working instructions for coding agents (and people) in this repository. The long form
is the Developer's guide under `docs/developer_guide/` (environment, architecture,
extending, GUI sessions, validation, documentation).

Rules marked ⚙ are enforced by the configuration in `pyproject.toml` (ruff, pyright,
pytest, coverage). If a ⚙ rule and the configuration disagree, the configuration wins and
this file is out of date — say so. When a judgement call concerns physics or numerics
rather than style, ask rather than guess.

## Project facts

| | |
|---|---|
| Package | `croak` (flat layout: `croak/`, not `src/`) |
| Purpose | FROG pulse retrieval: amplitude and phase, dispersive media, geometric smearing, uncertainty |
| Python | 3.14, equal to `requires-python` in `pyproject.toml` |
| Type checker | `pyright`, `basic` mode, over `croak/` |
| Numerics | NumPy/SciPy, with JAX (optimistix, lineax) for the differentiable solvers |
| GUI | the `gui` extra (PyQt6, superqt); the dev group always installs it, and its tests run offscreen with pytest-qt |
| Optional extras | `evo` (evosax, the `cma-es` retriever; in the dev group) |

## Guiding principles

- There should be one obvious way to do each thing. Prefer the existing pattern over inventing a new one; if you find two ways to do the same thing, consolidate them.
- Optimise for the reader. Clear beats clever — code is read far more often than it is written.
- Small, single-purpose functions with explicit inputs and outputs. Prefer pure functions; push side effects (I/O, global state, mutation) to the edges.
- Make the implicit explicit: explicit arguments over hidden state, explicit types on interfaces, explicit errors over silent fallbacks.
- Fail loudly and early with a clear exception. Never swallow errors or return a sentinel where raising is correct.

## Environment and tooling

- Python is managed with **uv**. Dependencies are declared in `pyproject.toml`; the resolved lockfile is `uv.lock`. Never `pip install` into the environment by hand, and never edit `uv.lock` by hand.
- **Ask before adding a dependency.** Then add it with `uv add <pkg>` (`uv add --dev <pkg>` for tooling).
- Run everything through the project environment by prefixing commands with `uv run`.
- `uv sync` is *exact*: it installs only the default `dev` group and removes the `docs` group. Use `uv sync --group docs` before building the docs.

## Commands

| Task | Command |
|---|---|
| Sync environment | `uv sync` (add `--group docs` for the docs) |
| **Full gate** (see below) | `uv run poe check` |
| Tests, all cores | `uv run poe test` |
| Tests with the 100% coverage gate | `uv run poe cov` |
| Focused tests, serial | `uv run pytest tests/test_x.py` |
| Lint with autofix / format | `uv run ruff check --fix .` / `uv run ruff format .` |
| Type-check | `uv run pyright` |
| Build docs | `uv run poe docs` |

## Definition of done

`uv run poe check` runs, in order:

1. `ruff check --fix .` — lint first: autofix rewrites code;
2. `ruff format .` — so the fixes are formatted;
3. `ruff check .` — catches what autofix could not;
4. `pyright` — no errors;
5. `poe cov` — the suite (doctests included) on every core with pytest-xdist, requiring
   **100% statement and branch coverage** of `croak/`, GUI included ⚙.

CI runs exactly this task; if you change the sequence, change `.github/workflows/ci.yml`
in the same commit. Do not declare work finished while a step fails. If a step genuinely
cannot pass, say so and why — never suppress a check to make the gate green.

**The coverage gate is not negotiable.** Do not lower the threshold, add `omit` entries or
exclusion patterns, or add `# pragma: no cover` to make it pass. Cover new code with tests
that assert intended behaviour, not tests that merely execute lines. Code that cannot be
reached is deleted or simplified, not excluded. A pragma is only for code that genuinely
cannot run under test (`if __name__ == "__main__":`, a guarded optional import's
`ImportError`) and carries its reason on the same line.

**Documentation-only changes** (the manual, docstring prose, AGENTS.md) do not need the
full suite: run `git diff --check` and `uv run poe docs`.

New functionality is not done until it is **tested**, **documented in the manual** as well
as in docstrings, and recorded in `CHANGELOG.md`.

## Working agreement

- **Stay in scope.** No refactoring, renaming or reformatting unrelated to the task.
- **Never hand-edit generated files**: `uv.lock`, `docs/_build/`, `docs/.jupyter_cache/`. Regenerate them.
- **Report failure honestly.** A check that fails, a test you could not make pass, an assumption you had to invent: say it plainly. Silent partial completion is worse than an admitted gap.

## Code style

- Formatting is owned by `ruff format` ⚙; do not hand-format or fight the formatter. Line length is the ruff default (88).
- Linting is owned by `ruff check` ⚙. Fix the cause rather than silencing the warning. A suppression must be rule-specific with a reason (`# noqa: E501 — URL must stay on one line`), never a blanket `# noqa`.
- Imports are absolute and sorted by ruff ⚙. No wildcard imports.
- Naming: `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_SNAKE` for constants. Names state what a thing is or does; avoid abbreviations beyond well-established domain ones.
- Use f-strings for interpolation and `pathlib.Path` for filesystem paths. Reach for the standard-library idiom before adding a dependency.

## Types

- Annotate every public function signature and class attribute. Do not annotate obvious locals just to fill space.
- Code must pass `pyright` ⚙ with no errors.
- Prefer precise types: `Sequence`/`Mapping` over `list`/`dict` for read-only parameters, `X | None` over `Optional[X]`, and built-in generics (`list[int]`). Avoid `Any`; if it is unavoidable, confine it to one place and comment why.
- Use a `@dataclass` (frozen where it can be) for structured data instead of passing loose tuples or dicts around.

## Errors and warnings

- Raise a specific exception (`ValueError`, `TypeError`, or a croak-specific subclass) with a message that says what was wrong and what was expected; never raise bare `Exception`.
- Validate inputs at public boundaries. Never let NaN or Inf propagate silently.
- Warn (`warnings.warn`, with a category and `stacklevel`) only when the result is still usable; otherwise raise.

## Docstrings

- Every public module, class, and function has a **NumPy-style** docstring ⚙ (rendered by Sphinx via the napoleon extension). Private helpers get a one-line docstring when the name is not fully self-explanatory.
- The first line is an imperative one-sentence summary. Then include, as applicable: `Parameters`, `Returns`, `Raises`, and `Examples`.
- For numerical quantities, state units and valid ranges in the parameter descriptions.
- `Examples` run as doctests in the suite ⚙, so they must be runnable and deterministic. Describe behaviour and contracts, not implementation details that will drift.

## Comments

Comments explain **why**, not **what**: the code already says what it does. Be generous
with the *why*:

- the physics and mathematics — derivations, approximations, the assumption that makes a step valid; refer to the manual page that carries a long derivation;
- the numerics — why this grid, cutoff, tolerance or regularisation; stability; where precision is lost;
- sign, unit and normalisation choices; references (a paper with DOI, an equation number, an issue);
- non-obvious trade-offs, and anything a reader would be tempted to "simplify" incorrectly.

Keep each comment next to what it describes and update it when the code changes. Delete commented-out code — version history is the archive. Mark deliberate follow-ups as `# TODO(context): ...` so they are greppable.

## Functions and modules

- One function, one responsibility. If a function needs a paragraph to explain, or has many nested branches, split it.
- Keep parameter lists short. Group related parameters into a small dataclass or config object rather than passing many positional arguments.
- Prefer returning new values over mutating arguments in place. Keep the core logic pure and testable; isolate I/O at the boundaries.
- Organise modules by domain concept, not by a catch-all `utils`. Declare the public surface with `__all__` where it helps.
- Keep plotting in `croak.plotting` as headless panels (`draw_*` / `update_*`) that the GUI composes; GUI modules hold Qt glue, not numerics.

## Tests

- Tests use **pytest** and live in `tests/`, mirroring the package layout. Name them `test_<unit>_<behaviour>`.
- Every public function has tests for the normal case, the edge cases, and the error paths (`pytest.raises(..., match=...)`). Add a regression test with every bug fix.
- Tests are deterministic and isolated: they run in random order ⚙ and in parallel ⚙, so use fixtures for setup, write only under `tmp_path`, and keep no hidden global state. No network access. Use `pytest.mark.parametrize` instead of copy-pasted cases.
- Randomness is explicit: `np.random.default_rng(seed)`.
- Warnings are errors ⚙. Assert an expected warning with `pytest.warns`; do not filter it away.
- For numerical code, assert with explicit tolerances (`numpy.testing.assert_allclose`, `pytest.approx`) and say in a comment what sets each tolerance. Test invariants and conservation laws, not only point values.
- Keep tests fast: small grids and few iterations unless the test is about convergence. GUI tests drive the real widgets through `qtbot`.
- Write the test alongside the code; a feature is not done until it is tested.

## Documentation

- Docs are built with **Sphinx**, and the API reference is generated from docstrings — so the docstring is the source of truth. Keep `README.md` current with behaviour changes.
- Record notable changes in `CHANGELOG.md` (Keep a Changelog style), under `Unreleased`.
- All functionality must be described in the user manual under `docs/`: background and context, how it works, and examples. When you add functionality, add documentation; when you touch something undocumented, document it.
- The tutorials under `docs/tutorials/` are executed notebooks whose results are cached in `docs/.jupyter_cache`, which is **committed** so Read the Docs never re-executes unchanged tutorials (its builders are too slow for the solver tutorials; see `docs/conf.py`). Whenever you edit a tutorial — or change croak behaviour that alters a tutorial's output — rebuild the docs locally (`uv run poe docs`) and commit the refreshed `docs/.jupyter_cache` together with the change. The build must succeed with no warnings.

## Git

- Work in an isolated Git worktree by default, reusing the current worktree when it is already dedicated to the task. Keep the primary checkout untouched.
- Work on a topic branch (`fix/…`, `feature/…`, `docs/…`, `test/…`). Create or switch to an appropriate topic branch and worktree as needed without asking for permission.
- Make clean, logical commits with descriptive but not verbose messages. Prefer several small commits over one large one.
- Stage deliberately: `git add <paths>`, not `git add -A`.
- Never tag, make releases or change the package version.
- **Coding agents:** never push; never run a destructive or history-rewriting command (`git reset --hard`, `git checkout .`, `git restore` over uncommitted changes, `git clean`, `git commit --amend`, `git rebase`, a force push). `fetch`, `pull` and `merge` only when asked.
