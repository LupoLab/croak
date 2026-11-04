The contributor workflow follows solprop's. `CONTRIBUTING.md` and the issue forms set
the standard for issues, labels and pull requests. A notable change now adds a
`changelog.d/` fragment instead of editing this file, and a rolling draft PR collects
the fragments after each merge. `poe check` also runs `codespell` and validates the
fragments. A new test keeps the version records in step, and `.zenodo.json` no longer
names a stale version. The workflows' actions are pinned by commit, and the release
checklist they refer to now exists in the Developer's guide.
