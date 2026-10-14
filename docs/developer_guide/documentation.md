# Building and maintaining the documentation

The manual uses Sphinx, MyST Markdown, and the PyData Sphinx Theme. API pages are
generated from NumPy-style docstrings. Tutorials use executable MyST notebooks.
Build from the repository root:

```bash
uv sync --group docs
uv run sphinx-build -E -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to inspect the result. Check navigation,
cross-references, equations, tables, and figures as well as the build log.

## Executed tutorial caches

The notebook execution cache in `docs/.jupyter_cache` is committed. It lets
Read the Docs reuse tutorial results instead of repeating expensive retrievals.
After editing a tutorial or changing behaviour that affects its output, rebuild
locally and commit the refreshed cache with the source change, and read the
refreshed outputs: a notebook can execute cleanly and still show a changed
result.

## Gallery calculations

Worked gallery pages execute the repository scripts with MyST-NB's code-cell
`:load:` option. Keep calculations in `examples/`, with prose and figure display
in the notebook page. A hidden setup cell gives each run a temporary output
directory and access to the bundled datasets. Include calculation helpers in
loaded cells where their content must participate in the execution-cache key.

Use `-E` when reviewing changes to included Python sources so Sphinx rereads all
notebooks. Unchanged code reuses the execution cache. A scientific output change
requires both refreshed cache entries and inspection of the figures and numbers.
Do not add local file paths or research-data downloads to the build. The paper
figures in `docs/_static/paper/` have their provenance recorded in
`docs/_static/paper/README.md`.

## Navigation and reader guidance

The root toctree has five entries: Introduction, User guide, Examples,
Developer's guide, and API. Give each page one navigation parent; link to it
from other relevant sections. Preserve established URLs and section anchors
when reorganising content.

Use the introduction for paced learning, topic chapters for practical guidance
and scientific detail, and examples for worked calculations. Keep contributor
architecture and maintenance procedures in this section.
