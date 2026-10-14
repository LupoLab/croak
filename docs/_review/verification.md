# Layout checkpoint verification

This record covers phase 1 of the documentation restructuring. It does not
certify the later scientific editing, public-docstring review, or expanded
worked gallery, which remain pending layout approval.

## Documentation and navigation

- Synced the locked project environment with `uv sync --locked --group docs`.
- Built with `uv run sphinx-build -W --keep-going -b html docs docs/_build/html`:
  successful, with no warnings. Rebuilt after preserving the home-page
  `#highlights` anchor; that build also succeeded without warnings.
- Accounted for all 44 original Markdown pages. The prototype has 66 published
  pages, each with exactly one navigation parent except the root page.
- Retained all original document paths and the original home-page heading
  anchors. Existing detailed pages and tutorial source were not rewritten.
- Checked 7,328 local links across the 66 rendered pages, including fragment
  targets: no missing files or anchors.
- Verified the five header links on the home page and all five section landing
  pages at 1366 px and 1024 px. No More menu or page-width overflow.
- Inspected the 390 px mobile layout, opened the navigation drawer, and followed
  User guide → Modelling the measurement. The drawer exposes all five sections.
- Checked representative mathematical, tutorial, API, and reference-table pages
  at desktop and mobile widths: correct active section, no page-width overflow,
  and no MathJax error elements. Visually inspected screenshots of the home page,
  guide, examples, mobile introduction and drawer, and forward-model page.
- Confirmed that `docs/_review` is excluded from the published HTML.
- All five tutorial notebooks were cache hits. No calculation or notebook output
  changed; restored access-time-only changes to the cache database after
  comparing its substantive records with HEAD.

Browser screenshots and machine-readable reports are in
`docs/_build/review/` (ignored build artifacts). The local preview is served at
`http://127.0.0.1:8765/` while its preview server remains running.

## Repository checks

The checks were started in the required order:

1. `uv run ruff format .`: 139 files unchanged.
2. `uv run ruff check --fix .`: passed.
3. `uv run pyright`: 0 errors, 0 warnings, 0 informations. The wrapper also
   printed an available-version notice; the locked dependency was not changed.
4. `QT_QPA_PLATFORM=offscreen uv run pytest`: **944 passed, 2 skipped**, with
   **87.84% coverage** (85% required), in 664.64 seconds. Exit status 0.

The suite emitted **47 runtime warnings**: 27 warnings about spectral
regularisation without an independent spectrum, 18 NLopt early-stop warnings,
and two divide-by-zero warnings. It also printed Qt/Matplotlib exceptions about
a deleted `MplCanvas` during GUI tests. These diagnostics originate in unchanged
package/test code; no package source, tests, dependency declarations, or lockfile
was changed by this phase. A separate baseline run was not performed. The test
output is therefore not described as warning-free, and no warning or exception
was suppressed. Runtime diagnostic cleanup remains outside this layout change.

`git diff --check` passed. Build and pytest logs are retained with the browser
reports in `docs/_build/review/` for local inspection.
