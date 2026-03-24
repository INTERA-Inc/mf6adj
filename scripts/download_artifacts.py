"""Download and extract GitHub Actions artifacts for ReadTheDocs builds.

This script downloads the rendered notebooks and built docs artifacts from
the most recent successful GitHub Actions workflow run. Selection is
deterministic and follows this priority:

1. Match an explicit commit SHA (if provided).
2. Otherwise use the most recent successful push run on the selected branch.

This allows ReadTheDocs to reuse outputs built in CI rather than
re-executing notebooks and rebuilding documentation locally.

Usage:
    python scripts/download_artifacts.py
        [--branch BRANCH]
        [--commit SHA]
        [--workflow WORKFLOW]
        [--repo REPO]

Environment variables:
    GITHUB_TOKEN: GitHub API token (required). Can be set automatically in
                  ReadTheDocs via build environment.
    DOWNLOAD_ARTIFACTS_WORKFLOW: Workflow file name or workflow id to query.
    DOWNLOAD_ARTIFACTS_BRANCH: Explicit branch override.
    READTHEDOCS_GIT_COMMIT_HASH: Commit SHA for commit-level run matching.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_RECOVERABLE = 10
EXIT_REQUIRED_ARTIFACT_MISSING = 20


def get_github_api_headers() -> dict[str, str]:
    """Get headers for GitHub API requests."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning("GITHUB_TOKEN not set. API rate limit will be very restrictive.")
        return {"Accept": "application/vnd.github+json"}
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
    }


def get_repo_info() -> tuple[str, str]:
    """Get repository owner and name from GITHUB_REPOSITORY env var or git."""
    repo_env = os.environ.get("GITHUB_REPOSITORY")
    if repo_env:
        owner, repo = repo_env.split("/")
        return owner, repo

    # Try to read from git
    try:
        import subprocess

        result = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=True,
        )
        url = result.stdout.strip()
        # Parse GitHub URL
        if "github.com" in url:
            if url.endswith(".git"):
                url = url[:-4]
            parts = url.split("/")
            owner = parts[-2]
            repo = parts[-1]
            return owner, repo
    except Exception as e:
        logger.debug(f"Failed to get repo from git: {e}")

    raise RuntimeError(
        "Could not determine repository. Set GITHUB_REPOSITORY env var "
        "or ensure .git is configured."
    )


def _is_placeholder_rtd_version(value: str) -> bool:
    """Return whether a ReadTheDocs version token is an alias."""
    return value.strip().lower() in {"latest", "stable"}


def _looks_like_commit_hash(value: str) -> bool:
    """Return whether a string looks like a git commit hash."""
    return bool(re.fullmatch(r"[0-9a-fA-F]{7,40}", value.strip()))


def get_current_branch() -> str | None:
    """Get the current branch name.

    ReadTheDocs often sets ``READTHEDOCS_VERSION`` to ``latest`` or ``stable``,
    which are aliases and not real branch names for the GitHub Actions API.
    Prefer explicit git identifiers and skip alias-like values.
    """
    # Highest-priority explicit override.
    explicit_branch = os.environ.get("DOWNLOAD_ARTIFACTS_BRANCH")
    if explicit_branch:
        return explicit_branch

    # Prefer RTD's underlying git identifier when available.
    git_identifier = os.environ.get("READTHEDOCS_GIT_IDENTIFIER")
    if git_identifier and not _looks_like_commit_hash(git_identifier):
        return git_identifier

    # Fall back to commonly available branch-ish variables.
    env_candidates = [
        os.environ.get("READTHEDOCS_VERSION_NAME"),
        os.environ.get("READTHEDOCS_VERSION"),
        os.environ.get("GITHUB_HEAD_REF"),
        os.environ.get("GITHUB_REF_NAME"),
    ]
    for candidate in env_candidates:
        if not candidate:
            continue
        if _is_placeholder_rtd_version(candidate):
            continue
        if _looks_like_commit_hash(candidate):
            continue
        return candidate

    # Try git
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = result.stdout.strip()
        if branch and branch != "HEAD":
            return branch
    except Exception as e:
        logger.debug(f"Failed to get current branch from git: {e}")

    return None


def get_default_branch(owner: str, repo: str) -> str:
    """Get the repository default branch from GitHub API."""
    url = f"https://api.github.com/repos/{owner}/{repo}"
    data = make_github_api_request(url)
    default_branch = data.get("default_branch")
    if not default_branch:
        raise RuntimeError("Could not determine repository default branch")
    return str(default_branch)


def get_target_commit() -> str | None:
    """Return a target commit SHA from CI/RTD environment, if available."""
    candidates = [
        os.environ.get("DOWNLOAD_ARTIFACTS_COMMIT"),
        os.environ.get("READTHEDOCS_GIT_COMMIT_HASH"),
        os.environ.get("GITHUB_SHA"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        value = candidate.strip()
        if _looks_like_commit_hash(value):
            return value.lower()
    return None


def make_github_api_request(url: str) -> Any:
    """Make an authenticated request to GitHub API."""
    headers = get_github_api_headers()
    request = Request(url, headers=headers)
    try:
        with urlopen(request) as response:
            return json.loads(response.read().decode())
    except HTTPError as e:
        logger.error(f"GitHub API request failed: {e.code} {e.reason}")
        raise


def get_workflow_runs(
    owner: str,
    repo: str,
    workflow: str,
    branch: str | None,
    per_page: int = 30,
) -> list[dict[str, Any]]:
    """Get successful push workflow runs, optionally filtered by branch."""
    url = (
        f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow}/runs"
        f"?status=success&event=push&per_page={per_page}"
    )
    if branch:
        url += f"&branch={branch}"

    logger.info(f"Fetching workflow runs from: {url}")
    data = make_github_api_request(url)
    runs = data.get("workflow_runs") or []
    if not isinstance(runs, list):
        return []
    return runs


def select_workflow_run(
    runs: list[dict[str, Any]],
    target_commit: str | None,
) -> dict[str, Any] | None:
    """Select the best workflow run using commit-first semantics."""
    if not runs:
        return None

    if target_commit:
        target_commit = target_commit.lower()
        for run in runs:
            head_sha = str(run.get("head_sha", "")).lower()
            if head_sha == target_commit:
                return run
        logger.warning(
            "No successful workflow run found for commit "
            + f"'{target_commit}' in queried run set"
        )

    # GitHub API returns most recent runs first.
    return runs[0]


def get_artifacts_for_run(owner: str, repo: str, run_id: int) -> dict[str, Any] | None:
    """Get artifacts for a specific workflow run."""
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/runs/{run_id}/artifacts"
    logger.info(f"Fetching artifacts from: {url}")
    data = make_github_api_request(url)

    if not data.get("artifacts"):
        logger.warning(f"No artifacts found for run {run_id}")
        return None

    # Create a mapping of artifact name to download URL
    artifacts = {
        artifact["name"]: artifact["archive_download_url"]
        for artifact in data["artifacts"]
    }
    return artifacts


def download_artifact(download_url: str, artifact_name: str) -> BytesIO:
    """Download an artifact from GitHub."""
    logger.info(f"Downloading artifact: {artifact_name}")
    headers = get_github_api_headers()
    request = Request(download_url, headers=headers)
    try:
        with urlopen(request) as response:
            return BytesIO(response.read())
    except HTTPError as e:
        logger.error(f"Failed to download artifact {artifact_name}: {e.code}")
        raise


def extract_artifact(
    artifact_zip: BytesIO, extract_path: Path, artifact_name: str
) -> None:
    """Extract a downloaded artifact."""
    logger.info(f"Extracting {artifact_name} to {extract_path}")
    extract_path.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(artifact_zip) as zf:
        zf.extractall(extract_path)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download GitHub Actions artifacts for ReadTheDocs builds"
    )
    parser.add_argument("--branch", help="Branch name (defaults to current branch)")
    parser.add_argument(
        "--repo",
        help="Repository in format 'owner/repo' "
        + "(defaults to GITHUB_REPOSITORY env var)",
    )
    parser.add_argument(
        "--workflow",
        default=os.environ.get("DOWNLOAD_ARTIFACTS_WORKFLOW", "ci.yml"),
        help=(
            "Workflow file name or workflow id to query "
            + "(default: env DOWNLOAD_ARTIFACTS_WORKFLOW or 'ci.yml')"
        ),
    )
    parser.add_argument(
        "--commit",
        help="Target commit SHA (defaults to READTHEDOCS_GIT_COMMIT_HASH/GITHUB_SHA)",
    )
    args = parser.parse_args()

    try:
        # Get repository information
        if args.repo:
            owner, repo = args.repo.split("/")
        else:
            owner, repo = get_repo_info()
        logger.info(f"Repository: {owner}/{repo}")

        # Get branch
        branch = args.branch or get_current_branch()
        if not branch:
            branch = get_default_branch(owner, repo)
            logger.info(
                "Could not resolve branch from environment; "
                + f"falling back to default branch '{branch}'"
            )
        logger.info(f"Branch: {branch}")

        target_commit = args.commit or get_target_commit()
        if target_commit:
            logger.info(f"Target commit: {target_commit}")
        logger.info(f"Workflow selector: {args.workflow}")

        # Query runs for selected workflow and branch.
        runs = get_workflow_runs(owner, repo, args.workflow, branch=branch)
        run = select_workflow_run(runs, target_commit)
        if not run:
            logger.error(
                "No workflow run found for selected workflow/branch. "
                + "Will fall back to local docs preparation."
            )
            return EXIT_RECOVERABLE

        run_id = run["id"]
        logger.info(
            f"Found workflow run: {run_id} (created at {run['created_at']}, "
            + f"head_sha={run.get('head_sha', 'unknown')})"
        )

        # Get artifacts
        artifacts = get_artifacts_for_run(owner, repo, run_id)
        if not artifacts:
            logger.error(
                "No artifacts found in workflow run. "
                + "Will fall back to local docs preparation."
            )
            return EXIT_RECOVERABLE

        logger.info(f"Found artifacts: {', '.join(artifacts.keys())}")

        # Download and extract artifacts
        root_dir = Path(__file__).resolve().parent.parent
        artifacts_to_download = {
            "rendered-notebooks": {
                "path": root_dir / "docs" / "source" / "examples",
                "required": True,
            },
            "built-docs": {
                "path": root_dir / "docs" / "_build" / "html",
                "required": False,
            },
        }

        for artifact_name, artifact_spec in artifacts_to_download.items():
            target_path = artifact_spec["path"]
            required = artifact_spec["required"]
            if artifact_name not in artifacts:
                msg = f"Artifact '{artifact_name}' not found in run"
                if required:
                    logger.error(msg + " (required)")
                    return EXIT_REQUIRED_ARTIFACT_MISSING
                logger.warning(msg + " (optional)")
                continue

            try:
                download_url = artifacts[artifact_name]
                artifact_zip = download_artifact(download_url, artifact_name)
                extract_artifact(artifact_zip, target_path, artifact_name)
                logger.info(f"Successfully extracted {artifact_name}")
            except Exception as e:
                if required:
                    logger.error(
                        "Failed to download/extract required artifact "
                        + f"{artifact_name}: {e}"
                    )
                    return EXIT_REQUIRED_ARTIFACT_MISSING
                logger.warning(
                    "Failed to download/extract optional artifact "
                    + f"{artifact_name}: {e}"
                )
                continue

        logger.info("Successfully downloaded and extracted all artifacts")
        return EXIT_OK

    except Exception as e:
        logger.error(f"Fatal error (recoverable): {e}")
        return EXIT_RECOVERABLE


if __name__ == "__main__":
    sys.exit(main())
