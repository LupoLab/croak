"""Release fragments collect once, and the rolling PR refreshes without force."""

import json
from pathlib import Path

import pytest

from tools import release_notes
from tools import update_release_notes as updater

CHANGELOG = """# Changelog

## [Unreleased]

### Fixed

- An existing correction.

## [0.1.0] - 2026-11-01

### Added

- Released prose, with deliberate trailing spaces.

[Unreleased]: https://example.org/compare
[0.1.0]: https://example.org/release
""".replace("trailing spaces.", "trailing spaces.  ")


def test_groups_notes_and_preserves_released_history():
    files = {
        "README.md": "Instructions, not a note.",
        "second.fixed.md": "Another correction.\nWith a continuation.",
        "first.fixed.md": "The first correction.",
        "interface.breaking.md": "Rename a keyword.\n\nUse its new name.",
        "feature.added.md": "A new feature.",
    }
    result = release_notes._render(CHANGELOG, release_notes._fragments(files))
    assert (
        result[result.index("## [0.1.0]") :]
        == CHANGELOG[CHANGELOG.index("## [0.1.0]") :]
    )
    assert result.count("### Fixed") == 1
    assert "- Rename a keyword.\n\n  Use its new name." in result
    assert result.index("### Added") < result.index("### Changed (breaking)")
    assert result.index("existing correction") < result.index("first correction")
    assert result.index("first correction") < result.index("Another correction")
    assert "Instructions, not a note" not in result


@pytest.mark.parametrize(
    "name,text",
    [
        ("wrong.md", "A correction."),
        ("note.unknown.md", "A correction."),
        ("../note.fixed.md", "A correction."),
        ("note.fixed.md", "  \n"),
        ("note.fixed.md", "# A heading"),
        ("note.fixed.md", "A correction.\n<<<<<<< branch"),
        ("note.fixed.md", "- A bullet"),
    ],
)
def test_rejects_invalid_fragments(name, text):
    with pytest.raises(ValueError):
        release_notes._fragments({name: text})


@pytest.mark.parametrize(
    "text",
    [
        "# No Unreleased section\n",
        "## [Unreleased]\n\n## [Unreleased]\n",
        "## [Unreleased]\n\n### Unknown\n",
        "## [Unreleased]\n\n### Fixed\n\n### Fixed\n",
    ],
)
def test_rejects_ambiguous_changelog(text):
    with pytest.raises(ValueError):
        release_notes._render(text, ())


def test_collect_is_idempotent_and_check_and_preview_do_not_write(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "CHANGELOG.md"
    path.write_text(CHANGELOG, encoding="utf-8")
    folder = tmp_path / "changelog.d"
    folder.mkdir()
    note = folder / "one.fixed.md"
    note.write_text("A new correction.\n", encoding="utf-8")
    instructions = folder / "README.md"
    instructions.write_text("Instructions\n", encoding="utf-8")
    for command in ("check", "preview"):
        monkeypatch.setattr("sys.argv", ["release_notes", command])
        release_notes.main()
        assert path.read_text(encoding="utf-8") == CHANGELOG
        assert note.exists()
    assert "A new correction." in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["release_notes", "collect"])
    release_notes.main()
    collected = path.read_bytes()
    assert not note.exists()
    assert instructions.exists()
    release_notes.main()
    assert path.read_bytes() == collected
    assert collected.count(b"A new correction.") == 1


def test_invalid_fragment_leaves_all_inputs_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (tmp_path / "changelog.d").mkdir()
    (tmp_path / "changelog.d" / "invalid.md").write_text("Text", encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.md")}
    monkeypatch.setattr("sys.argv", ["release_notes", "collect"])
    with pytest.raises(SystemExit, match="2"):
        release_notes.main()
    assert {path: path.read_bytes() for path in tmp_path.rglob("*.md")} == before


def test_current_repository_notes_validate():
    root = Path(__file__).resolve().parents[1]
    release_notes._render(
        (root / "CHANGELOG.md").read_text(encoding="utf-8"),
        release_notes._load(root),
    )


# The rolling PR, against isolated local Git repositories and without a token.


@pytest.fixture
def repository(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    updater._git("init", "--bare", str(remote))
    updater._git("init", "-b", "main")
    updater._git("config", "user.name", "Release test")
    updater._git("config", "user.email", "release@example.org")
    updater._git("config", "commit.gpgsign", "false")
    updater._git("config", "core.autocrlf", "false")
    updater._git("remote", "add", "origin", str(remote))
    (checkout / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    (checkout / "CITATION.cff").write_text("version: 0.1.0\n", encoding="utf-8")
    (checkout / "changelog.d").mkdir()
    (checkout / "changelog.d" / "README.md").write_text("Notes\n", encoding="utf-8")
    (checkout / "changelog.d" / "one.fixed.md").write_text(
        "Correction.\n", encoding="utf-8"
    )
    _save_main()
    return checkout


def _save_main():
    updater._git("add", ".")
    updater._git("commit", "-m", "Implementation change")
    updater._git("push", "origin", "main")
    return updater._git("rev-parse", "HEAD")


def test_refresh_tracks_main_and_preserves_history_and_checkout(repository):
    base = updater._git("rev-parse", "HEAD")
    first = updater._commit(base, None)
    assert first is not None
    assert updater._git("rev-list", "--parents", "-1", first).split() == [first, base]
    assert updater._commit(base, first) is None
    assert updater._show(first, "CHANGELOG.md").count("- Correction.") == 1
    assert (
        updater._git("ls-tree", "--name-only", first, "changelog.d/one.fixed.md") == ""
    )
    # A corrected old fragment, a new note and new code all come from current main.
    (repository / "changelog.d" / "one.fixed.md").write_text(
        "Corrected text.\n", encoding="utf-8"
    )
    (repository / "changelog.d" / "two.added.md").write_text(
        "New feature.\n", encoding="utf-8"
    )
    (repository / "code.py").write_text("answer = 42\n", encoding="utf-8")
    latest = _save_main()
    second = updater._commit(latest, first)
    assert second is not None
    assert updater._git("rev-list", "--parents", "-1", second).split() == [
        second,
        first,
        latest,
    ]
    notes = updater._show(second, "CHANGELOG.md")
    assert notes.count("Corrected text.") == 1
    assert "Correction." not in notes
    assert notes.count("New feature.") == 1
    assert (
        notes[notes.index("## [0.1.0]") :] == CHANGELOG[CHANGELOG.index("## [0.1.0]") :]
    )
    assert updater._show(second, "code.py") == "answer = 42\n"
    assert updater._git("status", "--porcelain") == ""
    assert updater._git("rev-parse", "HEAD") == latest
    # Both pushes succeed without force, including the update of an existing branch.
    updater._git("push", "origin", f"{first}:refs/heads/release")
    updater._git("push", "origin", f"{second}:refs/heads/release")


@pytest.mark.parametrize("preserve_message", [False, True])
def test_manual_branch_edits_are_not_overwritten(repository, preserve_message):
    base = updater._git("rev-parse", "HEAD")
    generated = updater._commit(base, None)
    assert generated is not None
    # Even a human commit that preserves the generator's trailer must match its tree.
    message = (
        updater._git("show", "-s", "--format=%B", generated)
        if preserve_message
        else "Prepare release"
    )
    manual = updater._run(
        "git",
        "commit-tree",
        updater._git("rev-parse", "HEAD^{tree}"),
        "-p",
        generated,
        input_text=message,
    )
    with pytest.raises(ValueError, match="manual edits"):
        updater._commit(base, manual)
    assert updater._git("status", "--porcelain") == ""


@pytest.mark.parametrize(
    "state,draft", [("CLOSED", True), ("MERGED", False), ("OPEN", False)]
)
def test_closed_or_ready_pr_pauses_before_pushing(
    repository, monkeypatch, state, draft
):
    original_run = updater._run

    def run(*args, input_text=None):
        if args[:3] == ("gh", "pr", "list"):
            return json.dumps([{"number": 1, "state": state, "isDraft": draft}])
        assert "push" not in args
        return original_run(*args, input_text=input_text)

    monkeypatch.setattr(updater, "_run", run)
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    updater.main()


def test_create_retry_and_refresh_keep_one_draft(repository, monkeypatch):
    original_run = updater._run
    pulls = []
    created = []

    def run(*args, input_text=None):
        if args[:3] == ("gh", "pr", "list"):
            return json.dumps(pulls)
        if args[:3] == ("gh", "pr", "create"):
            assert "--draft" in args
            assert "--body-file" in args
            created.append(input_text)
            return "https://example.org/pull/1"
        return original_run(*args, input_text=input_text)

    monkeypatch.setattr(updater, "_run", run)
    monkeypatch.setenv("GITHUB_REPOSITORY", "example/project")
    # Retry after a pushed branch but failed PR creation must still create the PR.
    base = updater._git("rev-parse", "HEAD")
    commit = updater._commit(base, None)
    assert commit is not None
    updater._git("push", "origin", f"{commit}:refs/heads/chore/release-after-0.1.0")
    updater.main()
    assert len(created) == 1
    pulls.append({"number": 1, "state": "OPEN", "isDraft": True})
    updater.main()
    assert len(created) == 1
    (repository / "changelog.d" / "two.fixed.md").write_text(
        "Another correction.\n", encoding="utf-8"
    )
    _save_main()
    updater.main()
    head = updater._git(
        "ls-remote", "--heads", "origin", "chore/release-after-0.1.0"
    ).split()[0]
    assert updater._show(head, "CHANGELOG.md").count("Another correction.") == 1
    assert len(created) == 1


def test_no_pending_notes_creates_no_commit(repository):
    (repository / "changelog.d" / "one.fixed.md").unlink()
    base = _save_main()
    assert updater._commit(base, None) is None
