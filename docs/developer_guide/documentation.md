# Building and maintaining the documentation

The manual uses Sphinx, MyST Markdown, and the PyData Sphinx Theme. API pages are
generated from NumPy-style docstrings. Tutorials use executable MyST notebooks.
Build from the repository root:

```bash
uv sync --group docs
uv run sphinx-build -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to inspect the result. Check navigation,
cross-references, equations, tables, and figures as well as the build log.

## Executed tutorial caches

The notebook execution cache in `docs/.jupyter_cache` is committed. It lets
Read the Docs reuse tutorial results instead of repeating expensive retrievals.
After editing a tutorial or changing behaviour that affects its output, rebuild
locally and commit the refreshed cache with the source change. Inspect the
outputs; successful execution alone does not establish a clear explanation or
a meaningful scientific result.

## Navigation and reader guidance

The root toctree has five entries: Introduction, User guide, Examples,
Developer's guide, and API. Give each page one navigation parent; link to it
from other relevant sections. Preserve established URLs and section anchors
when reorganising content.

Use the introduction for paced learning, topic chapters for practical guidance
and scientific detail, and examples for worked calculations. Keep contributor
architecture and maintenance procedures in this section. Editorial inventories
and audit notes in `docs/_review` are excluded from the published manual.
