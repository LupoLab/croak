# Releasing

A croak release is one source distribution and one wheel, published to PyPI when a
GitHub Release is published. Nothing merges, tags or publishes automatically: each of
those steps is a maintainer's.

## The rolling release PR

Implementation PRs do not edit `CHANGELOG.md`, because every PR would then change the
same lines. Each adds a Markdown fragment under `changelog.d/`, named
`<short-slug>.<category>.md`; [`changelog.d/README.md`](https://github.com/LupoLab/croak/blob/main/changelog.d/README.md)
gives the format and the categories.

```bash
uv run poe changelog-check     # part of poe check: validate the fragments
uv run poe changelog-preview   # print the assembled changelog, changing nothing
```

After each push to `main`, `.github/workflows/release-notes.yml` creates or refreshes
one draft PR, **Prepare the next release notes**, on `chore/release-after-<last-version>`.
The last released version comes from `CITATION.cff`. The draft groups every pending
fragment under `## [Unreleased]` and removes the collected fragments on its branch.
Correct a note by editing its fragment in an ordinary PR, not by editing the draft.
To refresh it by hand:

```bash
gh workflow run release-notes.yml --ref main
```

The updater checks that its branch is exactly what it generated before writing another
commit, so a manual edit stops it rather than being overwritten. It keeps the previous
head as a parent, so updates are ordinary fast-forward pushes. A closed PR, or one
marked ready, pauses updates; reopen a draft closed by mistake to resume them.

Commits made with the workflow's token do not start CI, so the draft shows no test
results while it is being refreshed. The maintainer's release commit below starts CI,
and the required `lint-type-test` check still has to pass on an up-to-date branch.

## Making a release

1. Mark the rolling PR ready **before** editing it, which freezes automatic refreshes.
2. In a worktree of its branch, bring in `main` and collect any fragments that arrived
   since the last refresh:

   ```bash
   git merge origin/main
   uv run poe changelog-collect
   ```

3. Choose the version, and read the collected notes as a whole. Then set the version in:

   - `pyproject.toml` and `croak/__init__.py`;
   - `uv.lock`, by running `uv lock`;
   - `CITATION.cff`;
   - `CHANGELOG.md`: rename the section to `## [<version>] - <date>`, add a new empty
     `## [Unreleased]` above it, and update the links at the foot of the file.

   `tests/test_release_metadata.py` fails if these disagree.
4. Run `uv run poe check` and `uv run poe docs`, push, and wait for the required checks.
5. Merge the PR. The next pending fragment starts a new draft for the next cycle.
6. Publish a GitHub Release from `main` with the tag `v<version>`.
   `.github/workflows/release.yml` checks that the tag matches `pyproject.toml`, builds
   the distributions, imports the wheel in a clean environment and publishes with PyPI
   trusted publishing. `.zenodo.json` names no version, so the archive record takes it
   from the release.

## One-time setup

A new repository, or one whose settings have been reset, needs:

- **PyPI:** a trusted publisher for the project `croak`, naming the owner `LupoLab`, the
  repository `croak`, the workflow `release.yml` and the environment `pypi`.
- **GitHub environment:** one named `pypi`, used by the publishing job.
- **Workflow permissions:** under *Settings → Actions → General*, turn on *Allow GitHub
  Actions to create and approve pull requests*, or the release-notes workflow fails when
  it opens its draft. Keep the default token permissions read-only: only that job asks
  for `contents: write` and `pull-requests: write`.
- **Codecov:** optionally the repository secret `CODECOV_TOKEN`. Tokenless upload works
  for a public repository, but is subject to rate limits.
