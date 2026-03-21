"""Download and extract GitHub Actions artifacts for ReadTheDocs builds.

This script downloads the rendered notebooks and built docs artifacts from
the most recent successful GitHub Actions workflow run on the current branch.
This allows ReadTheDocs to reuse the outputs built in CI rather than
re-executing notebooks and rebuilding documentation.

Usage:
    python scripts/download_artifacts.py [--branch BRANCH] [--repo REPO]

Environment variables:
    GITHUB_TOKEN: GitHub API token (required). Can be set automatically in
                  ReadTheDocs via build environment.
    READTHEDOCS_VERSION: ReadTheDocs version name (used to determine branch
                        if --branch not provided).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
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


def get_current_branch() -> str:
    """Get the current branch name."""
    # Try ReadTheDocs environment variable
    version = os.environ.get("READTHEDOCS_VERSION")
    if version:
        return version

    # Try git
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Failed to get current branch: {e}")
        raise


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


def get_latest_workflow_run(
    owner: str, repo: str, branch: str
) -> dict[str, Any] | None:
    """Get the latest successful workflow run for the given branch."""
    url = (
        f"https://api.github.com/repos/{owner}/{repo}/actions/runs"
        f"?branch={branch}&status=success&event=push&per_page=1"
    )
    logger.info(f"Fetching workflow runs from: {url}")
    data = make_github_api_request(url)

    if not data.get("workflow_runs"):
        logger.warning(f"No successful workflow runs found for branch: {branch}")
        return None

    return data["workflow_runs"][0]


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
        logger.info(f"Branch: {branch}")

        # Get latest workflow run
        run = get_latest_workflow_run(owner, repo, branch)
        if not run:
            logger.error("No workflow run found. Cannot download artifacts.")
            return 1

        run_id = run["id"]
        logger.info(f"Found workflow run: {run_id} (created at {run['created_at']})")

        # Get artifacts
        artifacts = get_artifacts_for_run(owner, repo, run_id)
        if not artifacts:
            logger.error("No artifacts found in workflow run.")
            return 1

        logger.info(f"Found artifacts: {', '.join(artifacts.keys())}")

        # Download and extract artifacts
        root_dir = Path(__file__).resolve().parent.parent
        artifacts_to_download = {
            "rendered-notebooks": root_dir / "docs" / "source" / "examples",
            "built-docs": root_dir / "docs" / "_build" / "html",
        }

        for artifact_name, target_path in artifacts_to_download.items():
            if artifact_name not in artifacts:
                logger.warning(f"Artifact '{artifact_name}' not found in run")
                continue

            try:
                download_url = artifacts[artifact_name]
                artifact_zip = download_artifact(download_url, artifact_name)
                extract_artifact(artifact_zip, target_path, artifact_name)
                logger.info(f"Successfully extracted {artifact_name}")
            except Exception as e:
                logger.error(f"Failed to download/extract {artifact_name}: {e}")
                return 1

        logger.info("Successfully downloaded and extracted all artifacts")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
