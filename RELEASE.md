# Making a release

Releases are managed with the `release.yml` GitHub Actions workflow.
Everything after step 1 runs automatically or directly on the GitHub website —
no local git commands are required.

---

## Overview

| Step | Who does it | What happens |
|------|-------------|--------------|
| 1 | You (GitHub website) | Create a `v*.*.*` branch to start the workflow |
| 2 | Workflow (automatic) | Version bump + changelog committed to branch |
| 3 | You (GitHub website) | Review and merge the draft PR into `main` |
| 4 | Workflow (automatic) | Draft GitHub release created with changelog |
| 5 | You (GitHub website) | Publish the draft release |
| 6 | Workflow (automatic) | Package built and published to PyPI |

---

## Step 1 — Create a release branch on GitHub

1. Go to the repository on GitHub.
2. Click the **branch dropdown** (top-left of the file browser, shows the
   current branch name).
3. Type a new branch name in the format `v<major>.<minor>.<patch>` — for
   example `v1.2.0`.  For a release candidate use `v1.2.0rc1`.
4. Select **Create branch: v1.2.0 from `main`** (or whichever base branch
   you want to release from).

Pushing the new branch triggers `release.yml` automatically.

---

## Step 2 — Automated prep (no action needed)

The **Prepare release** job runs on the new branch and does the following:

1. **Extracts the version** from the branch name (strips the leading `v`).
2. **Updates `mf6adj/version.py`** with the new version number.
3. **Generates `changelog/CHANGELOG.md`** by calling
   `scripts/generate_changelog.py`.  This script collects all pull requests
   merged since the last git tag (using the GitHub CLI) and formats them as a
   Keep-a-Changelog entry.  If the GitHub CLI is unavailable it falls back to
   git commit messages.
4. **Commits and pushes** both changed files back to the release branch with
   the message `bump version to <version> [skip ci]`.

For **release candidates** (`rc` suffix) the workflow stops here — no PR is
opened.

For **full releases** a second job opens a **draft pull request** titled
`Release <version>` targeting `main`.

> **Previewing the changelog locally (optional)**
>
> If you want to see what the changelog will look like before starting the
> release, run this from the repository root:
>
> ```bash
> python scripts/generate_changelog.py --version 1.2.0 --dry-run
> ```
>
> This prints the new changelog entry without writing anything to disk.

---

## Step 3 — Review and merge the release PR

1. Go to the [Pull Requests](https://github.com/INTERA-Inc/mf6adj/pulls) page
   on GitHub.
2. Open the draft PR titled **Release \<version\>**.
3. Check the automated commit — verify `mf6adj/version.py` has the right
   version and `changelog/CHANGELOG.md` looks correct.  You can edit either
   file directly on GitHub if corrections are needed (commit to the release
   branch).
4. Click **Ready for review**, then merge the PR into `main`.

Merging into `main` triggers the **Draft release** job, which creates a
GitHub release (still in draft) tagged `v<version>` and populates the release
notes from the changelog.

---

## Step 4 — Publish the draft GitHub release

1. Go to the [Releases](https://github.com/INTERA-Inc/mf6adj/releases) page.
2. Open the draft release (labelled **Draft**).
3. Review the title and release notes; edit if needed.
4. Click **Publish release**.

Publishing the release triggers the final **Publish package** job.

---

## Step 5 — PyPI publish (automatic)

The **Publish package** job runs automatically after the release is published.
It builds the package with `uv build`, validates it with `twine check`, and
uploads it to [PyPI](https://pypi.org/p/mf6adj) using trusted publishing
(no API token required).

The job runs inside the `release` GitHub environment.  If it hasn't been
configured yet, go to **Settings → Environments** in the repository and create
an environment named `release` with the appropriate protection rules.

---

## Release checklist

- [ ] Release branch named `v<major>.<minor>.<patch>` created from `main`
- [ ] Automated prep commit visible on the release branch
- [ ] `changelog/CHANGELOG.md` reviewed and correct
- [ ] `mf6adj/version.py` shows the correct version
- [ ] Draft PR merged into `main`
- [ ] Draft GitHub release published
- [ ] **Publish package** job completed successfully on PyPI
