# Contributing to croak

Thank you for your interest in croak. Bug reports, questions about the physics or the
numerics, documentation fixes and new functionality are all welcome.

## Check what already exists

Before opening an issue or a pull request, and before starting work on either, look at
the [issues](https://github.com/LupoLab/croak/issues) and
[pull requests](https://github.com/LupoLab/croak/pulls), open and recently closed, for
what already covers it or touches it:

```sh
gh issue list --state all --search "covariance"
gh pr list --state all --search "covariance"
```

If one already covers your work, add to it rather than opening another. If one only
touches it, such as an open pull request that changes the same files, link it from what
you open, even if you need nothing from it: GitHub shows the reference on both sides, and
the next reader then knows the two are connected. For a bug report, a quick search of the
issues is enough.

## Reporting a problem

Use the [report form](https://github.com/LupoLab/croak/issues/new?template=report_a_problem.yml),
which asks for the smallest script that shows it, the measurement and the retrieval, what
you expected, what you got, and the versions.

For a physics question — a coefficient, a default, a model's validity range — say what
you compared against (a paper, an independent code, an analytic limit) and at what
parameters.

That is all a report needs. A maintainer fills in the rest.

## Writing an issue

Planned work lives in [GitHub issues](https://github.com/LupoLab/croak/issues). So an issue
for planned work, written by a person or an agent, has to let someone pick up the task
from its text alone.

[Check what already exists](#check-what-already-exists) first, and link what you find.
Write one issue per coherent piece of work, not one per finding or per file.

Use these sections, which the [planned work template](.github/ISSUE_TEMPLATE/planned_work.md)
lays out, leaving out any that do not apply:

- **What happens**, or what is missing, so that someone new to this code can follow it.
- **Why it matters:** what a user or a developer gets wrong or pays for today.
- **Reproduction.** For a bug, the smallest script that shows it, which you have run,
  with its output. For a performance issue, the numbers, the machine and the command.
- **Context:** links to what a reader needs, namely the code, the manual, pull requests
  and other issues. Link code as a permalink to a commit (press `y` on a file's page on
  GitHub), and say which commit you checked against.
- **To do:** checkboxes that are actions, not restated symptoms. Name what must not change
  or break. Check that a suggested fix would work before writing it down.
- **Done when:** what someone can check to know it is finished, with a tolerance and a
  machine for a measurement. A criterion should cover every box, and one that no pull
  request can show should say what stands in for it.

Be exact about what you know:

- Say what you ran and what you only read ("read, not run"), and label a suspicion as one.
- Give a measurement its machine, grid and date, and a count its scope.
- Mark a choice that is not the contributor's with **Needs a decision:**, saying who
  decides and what you recommend. A physics judgement, a changed default and a changed
  public signature are the maintainers' ([`AGENTS.md`](AGENTS.md)).
- Issues are public, so leave out local paths, private notes and nicknames for machines.
  Name hardware by its model.

### Labels

Issues are filtered by [label](https://github.com/LupoLab/croak/labels), so an issue
carries:

- one **type**, or two if it really is both: `bug` (wrong behaviour or a wrong result),
  `performance` (slower or larger than it should be, with no wrong result), `enhancement`
  (a new feature or a request), `documentation`, `refactor` (no change in behaviour) or
  `testing` (tests and CI coverage);
- one **priority**: `priority: high` (wrong results, wrong public statements or a user's
  first contact), `priority: medium` (real work that should not wait long) or
  `priority: low` (worth doing, no hurry);
- an **area** where one fits: `area: forward model`, `area: solvers`, `area: uncertainty`,
  `area: preprocessing`, `area: gui` or `area: packaging`;
- `good first issue` if it needs no physics judgement.

If you cannot set labels, suggest them in the text and a maintainer will.

## Setting up

croak uses [uv](https://docs.astral.sh/uv/) and Python 3.14:

```sh
git clone https://github.com/LupoLab/croak.git
cd croak
uv sync                 # the package, editable, with the development tools and Qt
uv run poe check        # the full gate: lint, format, types, spelling, notes, tests
```

`uv sync` removes the documentation tools, so use `uv sync --group docs` before building
the documentation with `uv run poe docs`. The
[Developer's guide](https://croak.readthedocs.io/en/latest/developer_guide/index.html)
describes the environment, the architecture and how to extend the package.

## Making a change

1. [Check what already exists](#check-what-already-exists), so that you neither repeat
   nor collide with work in hand. Look especially for open pull requests that change the
   same files.
2. Work on a branch named for its topic: `fix/…`, `feature/…`, `docs/…`, `test/…` or
   `chore/…`. For anything larger than a focused fix — a new model, a changed physics
   default, a new keyword, a changed public signature — open an issue first so the
   approach can be agreed.
3. Read [`AGENTS.md`](AGENTS.md): it holds the coding standards (types, docstrings,
   tests, documentation) for people and coding agents alike.
4. Keep the diff to the task. Add tests with the code, document new functionality in the
   manual as well as in docstrings, and add a release-note fragment under `changelog.d/`
   for a notable change. Use a unique change slug or PR number; no separate issue is
   needed for the fragment. Follow [the fragment format](changelog.d/README.md). Leave
   `CHANGELOG.md` to the rolling release PR, which collects fragments after merges.
5. Run `uv run poe check` before opening the pull request. CI takes far longer than the
   local gate, so it should not be the first to find a failure. A change to the
   documentation alone needs only `git diff --check` and `uv run poe docs`.
6. Open a pull request describing what changed, why, and how it was verified. Say plainly
   what you assumed, skipped or could not make pass. Search again first, since work may
   have appeared while you were busy. Link the issue it closes (`Closes #N`) and any
   issue or pull request it overlaps with or depends on, and give it a label or two: its
   type and, where one fits, its area, as for an issue. If you cannot set labels, a
   maintainer will. CI runs on every pull request: the full gate on Linux, and the same
   complete test suite on Windows and macOS.

## Review checklist

A pull request is ready to merge when:

- It links the issues and pull requests it closes, overlaps with or depends on.
- The applicable checks pass: the documentation checks for a documentation-only change;
  otherwise the gate and CI, with coverage still at 100 %.
- The physics is checked against a primary source or an independent derivation, with its
  validity range; algorithms are checked for conditioning, convergence and gradient
  correctness as relevant ([numerical validation](docs/developer_guide/validation.md)).
- Every numerical tolerance in a test is justified in a comment, and every new dependency
  has been agreed.
- The manual page, docstrings and release-note fragment are present.
- A tutorial whose source or output changed has its refreshed `docs/.jupyter_cache`
  committed with it, and someone has read the new output.

## Licence

By contributing you agree that your contributions are licensed under the project's
[MIT licence](LICENSE).
