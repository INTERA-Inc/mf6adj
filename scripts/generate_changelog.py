"""Generate a changelog entry for a new release.

Uses merged GitHub pull requests since the last tag as the source of changes.
Falls back to git commit messages if the GitHub CLI is unavailable.

The generated entry is prepended to changelog/CHANGELOG.md after the header block.

Usage:
    python scripts/generate_changelog.py --version 1.2.3
    python scripts/generate_changelog.py --version 1.2.3 --repo INTERA-Inc/mf6adj
    python scripts/generate_changelog.py --version 1.2.3 --dry-run
"""

import argparse
import json
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

_project_root = Path(__file__).parent.parent
_changelog_path = _project_root / "changelog" / "CHANGELOG.md"
_default_repo = "INTERA-Inc/mf6adj"


def get_previous_tag():
    """Return the most recent git tag, or None if no tags exist."""
    result = subprocess.run(
        ["git", "describe", "--tags", "--abbrev=0"],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def get_prs_since_tag(tag, repo):
    """Return merged PRs since the given tag using the GitHub CLI.

    Returns a list of dicts with keys: number, title, author login.
    Returns None if the GitHub CLI is unavailable or the call fails.
    """
    if tag:
        date_result = subprocess.run(
            ["git", "log", "-1", "--format=%aI", tag],
            capture_output=True,
            text=True,
        )
        if date_result.returncode != 0:
            return None
        since = date_result.stdout.strip()
        search = f"merged:>{since}"
    else:
        search = "is:merged"

    result = subprocess.run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "merged",
            "--json",
            "number,title,author,mergedAt",
            "--search",
            search,
            "--repo",
            repo,
            "--limit",
            "200",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    prs = json.loads(result.stdout)
    return sorted(prs, key=lambda x: x["mergedAt"])


def get_commits_since_tag(tag):
    """Return non-merge commit subjects since the given tag (or all if no tag)."""
    base = f"{tag}..HEAD" if tag else "HEAD"
    result = subprocess.run(
        ["git", "log", base, "--oneline", "--no-merges"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    commits = []
    for line in result.stdout.strip().splitlines():
        # strip short hash prefix (first word)
        parts = line.split(" ", 1)
        commits.append(parts[1] if len(parts) == 2 else line)
    return commits


def build_entry(version, prs=None, commits=None):
    """Build a Keep-a-Changelog formatted entry string."""
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [f"## [{version}] - {today}", ""]

    if prs:
        lines += ["### Changes", ""]
        for pr in prs:
            author = pr.get("author", {}).get("login", "")
            author_str = f" (@{author})" if author else ""
            lines.append(f"- {pr['title']} (#{pr['number']}){author_str}")
        lines.append("")
    elif commits:
        lines += ["### Changes", ""]
        for msg in commits:
            lines.append(f"- {msg}")
        lines.append("")
    else:
        lines += ["### Changes", "", "- No recorded changes", ""]

    return "\n".join(lines)


def prepend_entry(entry, dry_run=False):
    """Insert the entry into CHANGELOG.md after the header block."""
    content = _changelog_path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)

    # Find the first blank line after the header (before any ## section)
    insert_pos = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("## "):
            insert_pos = i
            break

    # Ensure a blank line separates header from new entry
    new_block = entry + "\n\n"
    updated = "".join(lines[:insert_pos]) + new_block + "".join(lines[insert_pos:])

    if dry_run:
        print("--- DRY RUN: CHANGELOG.md would be updated as follows ---")
        print(updated)
    else:
        _changelog_path.write_text(updated, encoding="utf-8")
        print(f"Updated {_changelog_path}")


def main():
    parser = argparse.ArgumentParser(
        prog="generate_changelog",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__),
    )
    parser.add_argument(
        "-v",
        "--version",
        required=True,
        help="Version number for the new release (e.g. 1.2.3)",
    )
    parser.add_argument(
        "--repo",
        default=_default_repo,
        help=f"GitHub repository in owner/name format (default: {_default_repo})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the updated changelog without writing to disk",
    )
    args = parser.parse_args()

    tag = get_previous_tag()
    if tag:
        print(f"Generating changelog since tag: {tag}")
    else:
        print("No previous tags found — including all changes")

    prs = get_prs_since_tag(tag, args.repo)
    if prs is not None:
        print(f"Found {len(prs)} merged PR(s) via GitHub CLI")
        entry = build_entry(args.version, prs=prs)
    else:
        print("GitHub CLI unavailable or failed — falling back to git log")
        commits = get_commits_since_tag(tag)
        print(f"Found {len(commits)} commit(s) via git log")
        entry = build_entry(args.version, commits=commits)

    prepend_entry(entry, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
