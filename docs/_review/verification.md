# Documentation verification

The first two sections record the approved phase-1 layout checkpoint. The
final verification below covers the completed introduction, guide, gallery,
developer material and API editing pass.

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

## Final manual and scientific review

The layout checkpoint was approved before phases 2–5. The completed manual has
74 content pages under the same five top-level sections. All 44 original pages
remain at their original paths. An HTML comparison with the layout-checkpoint
build found no lost section headings or explicit IDs, including API targets and
named equation anchors. Generated cell/index IDs were excluded from that comparison.

The review covers the introduction and nine guide chapters in reading order,
the full original API surface plus the previously unrendered delay-origin module,
all example scripts, dataset descriptions, README, citations and changelog.
The page map, API inventory, scientific audit and runtime-findings records
separate editorial changes, evidence limits and possible future runtime work.
Uncertain historical numerical claims are preserved in excluded review archives.

Scientific statements and six replacement evidence figures were checked against
the two unpublished drafts and their notation/audit records. The full research
campaign was not rerun. The bundled examples remain illustrative calculations,
with their own settings and field references. No private manuscript path or new
research-data download is required to build the manual.

## Executed calculations

All five tutorials and eight new gallery notebooks have local execution results
in the committed cache. Gallery calculations load the runnable Python scripts;
the shared synthetic-data helper participates in the relevant cache keys.
Generated files are isolated in temporary directories. Original committed cache
entries are retained; unused intermediate entries created during this edit were
removed. The final clean build reuses all 13 current notebooks from cache.

README and Quickstart Python examples were executed separately. Their seeded
first retrieval gives a trace error near 1.08e-16. The dispersive introduction now
uses a 153.6 fs periodic window and a ±40 fs scan: doubling depth quadrature from
20 to 40 nodes changes the fixed-field trace by about 1.33e-5 in relative L2, and
the delay-edge signal is about 4.68e-5 of peak. It retrieves 1.51 fs with the slab
model versus 2.77 fs with the thin model for the same 1.51 fs generator.

The thickness-uncertainty example deliberately retains its biased and unstable
result. Independent seeded executions gave about 2.30–2.64 fs for a 3.00 fs
generator, with similar residuals near 0.308 %. The final cache shows 2.30 fs.
The cause of the variation remains a separate runtime investigation, recorded
in the gallery and audit; a narrow statistical interval is not evidence of field
accuracy. This is not presented as a reproducible quantitative benchmark.

All executed figures and all six added draft figures were visually inspected,
including the final dispersive and uncertainty plots. The cached notebook output,
not a separate hand-maintained calculation, supplies gallery figures and numbers.

## Final validation

- `uv run ruff format .`: 139 files unchanged on the final check.
- `uv run ruff check --fix .`: passed.
- `uv run pyright`: 0 errors, 0 warnings, 0 informations. The available-version
  notice remains; no dependency was updated.
- `QT_QPA_PLATFORM=offscreen uv run pytest`: **944 passed, 2 skipped**, with
  **87.84% coverage**, in **642.47 seconds**. The full suite was started after
  the required format/lint/type checks. Subsequent reference-role corrections
  did not change executable statements or line positions; the three fast checks
  were repeated and passed.
- The test suite emitted **46 warnings** in the same categories as the layout
  baseline: 27 missing-independent-spectrum warnings, 17 NLopt early-stop
  warnings, and two divide-by-zero warnings. The existing deleted-`MplCanvas`
  Qt/Matplotlib diagnostic also appeared. No warnings were suppressed.
- `uv run sphinx-build -E -W --keep-going -b html docs docs/_build/html`:
  successful, with no warnings.
- Checked **8,637 local links** across **77 HTML pages** (74 content pages plus
  generated indexes/search): no missing files or fragments. No unresolved
  croak API references remain in the rendered pages.
- Loaded all 77 pages in headless Chrome: **1,186 rendered mathematical
  expressions**, no MathJax errors, broken images or document-width overflow.
- Rechecked the home page and five section landings at 1366 px, 1024 px and
  390 px. Desktop headers contain exactly the five requested sections and no
  More menu. The mobile drawer exposes all five sections and chapter navigation.
- Inspected detailed mathematical, API, gallery and table pages at desktop and
  mobile widths. Long equations and tables scroll within their containers;
  API signatures wrap at narrow widths. Navigation active states are correct.
- Compared the AST of all **77 package Python files** with the layout checkpoint
  after removing docstrings: no runtime differences. Public signatures,
  defaults, saved-data formats, versions, dependency files and tests are unchanged.
- `git diff --check`: passed. No push, publication or release was made.

Logs, screenshots and machine-readable reports are retained locally under
`docs/_build/review/`. These build artifacts are intentionally not committed.
