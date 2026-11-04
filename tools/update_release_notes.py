"""Refresh the one draft release-notes PR from trusted main-branch contents."""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from tools.release_notes import _fragments, _render

__all__ = ["main"]

_TRAILER = "Release-Notes-Base: "
_BODY = """Collect pending release notes into `CHANGELOG.md`.

This draft is refreshed from `main` after merges. Each update collects the Markdown
fragments and removes the collected files in this branch. Released history is preserved.
No issue is required for a fragment; link the implementation PR or issue in its text
when useful. Edit fragments in implementation PRs rather than this generated draft.

To prepare the release, mark this PR ready to freeze automatic updates, then follow
[`docs/developer_guide/releasing.md`](https://github.com/LupoLab/croak/blob/main/docs/developer_guide/releasing.md).
Review the notes, choose the version and date, and update the version records here.
Tagging and publishing remain manual. Required checks still apply; approve any pending
workflow runs before merging. Nothing in this workflow approves or merges a PR.
"""


def _run(*args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        args,
        input=input_text,
        text=True,
        encoding="utf-8",
        check=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _git(*args: str) -> str:
    return _run("git", *args)


def _show(ref: str, path: str) -> str:
    # Keep trailing newlines: released changelog history is immutable.
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        text=True,
        encoding="utf-8",
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _tree(base: str) -> str:
    files = {}
    entries = _git("ls-tree", "-r", "-z", base, "--", "changelog.d")
    for entry in entries.split("\0"):
        if not entry:
            continue
        metadata, path = entry.split("\t", 1)
        if not metadata.startswith("100644 blob "):
            raise ValueError(f"Release notes must be regular files: {path}")
        files[path.removeprefix("changelog.d/")] = _show(base, path)
    fragments = _fragments(files)
    rendered = _render(_show(base, "CHANGELOG.md"), fragments)
    # A private index builds the generated tree without touching the checkout or
    # its index. All other files come from main, including concurrent code changes.
    with tempfile.TemporaryDirectory() as directory:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(directory) / "index")}

        def index(*args: str) -> str:
            return subprocess.run(
                ["git", *args],
                env=env,
                text=True,
                encoding="utf-8",
                check=True,
                stdout=subprocess.PIPE,
            ).stdout.strip()

        index("read-tree", base)
        blob = _run("git", "hash-object", "-w", "--stdin", input_text=rendered)
        index("update-index", "--add", "--cacheinfo", f"100644,{blob},CHANGELOG.md")
        for fragment in fragments:
            index("update-index", "--force-remove", f"changelog.d/{fragment.name}")
        return index("write-tree")


def _commit(base: str, previous: str | None) -> str | None:
    tree = _tree(base)
    if previous is not None:
        message = _git("show", "-s", "--format=%B", previous)
        match = re.fullmatch(
            rf"Collect release notes\n\n{_TRAILER}([0-9a-f]{{40}})", message
        )
        if match is None or _tree(match[1]) != _git(
            "rev-parse", f"{previous}^{{tree}}"
        ):
            raise ValueError("Release branch has manual edits; leave it unchanged")
        if tree == _git("rev-parse", f"{previous}^{{tree}}"):
            return None
    elif tree == _git("rev-parse", f"{base}^{{tree}}"):
        return None
    parents = ["-p", base] if previous is None else ["-p", previous, "-p", base]
    # Retaining the previous head as a parent makes every update a fast-forward.
    # The generated tree was verified above; no human edits or history are replaced.
    return _run(
        "git",
        "commit-tree",
        tree,
        *parents,
        input_text=f"Collect release notes\n\n{_TRAILER}{base}\n",
    )


def main() -> None:
    """Create or refresh the rolling draft without force pushes or auto-merging."""
    repository = os.environ["GITHUB_REPOSITORY"]
    base = _git("rev-parse", "origin/main")
    citation = _show(base, "CITATION.cff")
    version = re.search(r"^version: ['\"]?(\d+\.\d+\.\d+)['\"]?$", citation, re.M)
    if version is None:
        raise ValueError("CITATION.cff must name the latest released version")
    branch = f"chore/release-after-{version[1]}"
    pulls = json.loads(
        _run(
            "gh",
            "pr",
            "list",
            "--repo",
            repository,
            "--head",
            branch,
            "--state",
            "all",
            "--json",
            "number,state,isDraft",
            "--limit",
            "2",
        )
    )
    if pulls and (
        len(pulls) != 1 or pulls[0]["state"] != "OPEN" or not pulls[0]["isDraft"]
    ):
        sys.stdout.write(
            "Release PR is closed or ready for review; refresh is paused.\n"
        )
        return
    previous = None
    if _git("ls-remote", "--heads", "origin", f"refs/heads/{branch}"):
        _git("fetch", "origin", f"refs/heads/{branch}")
        previous = _git("rev-parse", "FETCH_HEAD")
    commit = _commit(base, previous)
    if commit is not None:
        _git("push", "origin", f"{commit}:refs/heads/{branch}")
    if not pulls and (commit is not None or previous is not None):
        sys.stdout.write(
            _run(
                "gh",
                "pr",
                "create",
                "--repo",
                repository,
                "--base",
                "main",
                "--head",
                branch,
                "--draft",
                "--title",
                "Prepare the next release notes",
                "--body-file",
                "-",
                input_text=_BODY,
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
