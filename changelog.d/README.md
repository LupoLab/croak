# Pending release notes

Add one Markdown file for each notable change instead of editing `CHANGELOG.md` in an
implementation PR. Name it `<short-slug>.<category>.md`; a PR number can be the slug.
No separate issue is required for a fragment. Categories are `added`, `changed`,
`breaking`, `deprecated`, `removed`, `fixed` and `security`. `breaking` becomes
**Changed (breaking)** in the changelog.

For example, `edge-peak-delay.fixed.md` might contain:

```text
`marginal_peak_delay_jax` returns the peak sample itself when the peak is at the edge of the grid (#4).
```

Write one entry without a heading or leading bullet. Describe the observable change,
including migration instructions for a break. Markdown links are relative to the root
`CHANGELOG.md`, where the text will appear. Use a unique slug to avoid conflicts. A later
correction can edit the original fragment while it is still pending.

```sh
uv run poe changelog-check     # validate fragments and the Unreleased structure
uv run poe changelog-preview   # print the assembled changelog without changing files
```

After merges to `main`, the **Release notes** workflow creates or updates one draft PR
on `chore/release-after-<last-version>`. It collects every fragment into `CHANGELOG.md`
and removes those fragments on that branch. Existing Unreleased notes and released
history are retained. Edit source fragments in ordinary PRs; the rolling draft is
generated. The workflow stops on manual edits, a closed PR, or a PR marked ready.

Only the rolling release PR normally changes `CHANGELOG.md`. Mark it ready when release
preparation starts, then follow the
[release checklist](../docs/developer_guide/releasing.md).
`uv run poe changelog-collect` is the manual equivalent: it updates the changelog and
deletes the collected fragments in the current checkout. Use it only in the release
worktree, after bringing in any final changes. There are no automatic merges, tags or
publications.
