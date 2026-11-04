"""Validate and collect independent Markdown release-note fragments."""

import argparse
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["main"]

_CATEGORIES = {
    "added": "Added",
    "changed": "Changed",
    "breaking": "Changed (breaking)",
    "deprecated": "Deprecated",
    "removed": "Removed",
    "fixed": "Fixed",
    "security": "Security",
}
_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\.([a-z]+)\.md")


@dataclass(frozen=True)
class _Fragment:
    name: str
    category: str
    text: str


def _fragments(files: Mapping[str, str]) -> tuple[_Fragment, ...]:
    fragments = []
    for name, content in sorted(files.items()):
        if name == "README.md":
            continue
        match = _NAME.fullmatch(name)
        if match is None or match[1] not in _CATEGORIES:
            raise ValueError(f"Invalid release-note filename: {name}")
        text = content.strip()
        if not text or re.search(r"^(?:#|<<<<<<<|=======|>>>>>>>)", text, re.M):
            raise ValueError(f"Empty note, heading or conflict marker in {name}")
        if re.match(r"(?:[-*+] |\d+\. )", text):
            raise ValueError(f"Write one entry without its leading bullet in {name}")
        fragments.append(_Fragment(name, _CATEGORIES[match[1]], text))
    return tuple(fragments)


def _render(changelog: str, fragments: Sequence[_Fragment]) -> str:
    headings = list(re.finditer(r"^## \[Unreleased\]\n", changelog, re.M))
    if len(headings) != 1:
        raise ValueError("CHANGELOG.md must have exactly one '## [Unreleased]' heading")
    start = headings[0].end()
    following = re.search(r"^## ", changelog[start:], re.M)
    end = len(changelog) if following is None else start + following.start()
    pending = changelog[start:end]
    sections: dict[str, str] = {}
    parts = re.split(r"^### (.+)\n", pending, flags=re.M)
    for title, body in zip(parts[1::2], parts[2::2], strict=True):
        if title not in _CATEGORIES.values() or title in sections:
            raise ValueError(f"Unknown or repeated Unreleased category: {title}")
        sections[title] = body.strip()
    if not fragments:
        return changelog
    for fragment in fragments:
        lines = fragment.text.splitlines()
        entry = (
            "- "
            + lines[0]
            + "".join(f"\n  {line}" if line else "\n" for line in lines[1:])
        )
        sections[fragment.category] = (
            sections.get(fragment.category, "") + "\n" + entry
        ).strip()
    blocks = [parts[0].strip()] if parts[0].strip() else []
    blocks.extend(
        f"### {title}\n\n{sections[title]}"
        for title in _CATEGORIES.values()
        if title in sections
    )
    # The released suffix is copied verbatim, including its whitespace and links.
    return changelog[:start] + "\n" + "\n\n".join(blocks) + "\n\n" + changelog[end:]


def _load(root: Path) -> tuple[_Fragment, ...]:
    folder = root / "changelog.d"
    files = {}
    for path in folder.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Release notes must be regular files: {path.name}")
        files[path.name] = path.read_text(encoding="utf-8")
    return _fragments(files)


def main() -> None:
    """Check, preview or collect the current checkout's pending release notes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "preview", "collect"))
    args = parser.parse_args()
    root = Path.cwd()
    try:
        fragments = _load(root)
        path = root / "CHANGELOG.md"
        rendered = _render(path.read_text(encoding="utf-8"), fragments)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if args.command == "preview":
        sys.stdout.write(rendered)
    elif args.command == "collect":
        path.write_text(rendered, encoding="utf-8")
        for fragment in fragments:
            (root / "changelog.d" / fragment.name).unlink()


if __name__ == "__main__":
    main()
