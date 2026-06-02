# How to Make a PyPI Release

## Prerequisites (one-time setup)

1. Ensure a `release` environment exists in the repository:
   - Go to **Settings → Environments → New environment** and name it `release`
   - Optionally add required reviewers to control who can approve releases

2. Configure trusted publishing on PyPI:
   - Go to [pypi.org](https://pypi.org) → your project → **Publishing → Add a publisher**
   - Set the following values:
     - Owner: `INTERA-Inc`
     - Repository: `mf6adj`
     - Workflow: `release.yml`
     - Environment: `release`

---

## Release Steps

### 1. Update the changelog

Edit `changelog/CHANGELOG.md` and add notes for the new release. These notes will become the GitHub release description.

### 2. Create a release branch

Create and push a branch named with the version number (must start with `v`):

```bash
git checkout -b v1.2.3
git push origin v1.2.3
```

This triggers the `prep` job in `release.yml`, which:
- Bumps the version in `mf6adj/version.py` to match the branch name
- Commits and pushes the version bump back to the branch
- Saves the version and changelog as workflow artifacts

> For release candidates, name the branch `v1.2.3rc1`. The `prep` job will strip the `rc*` suffix and set the version to `1.2.3`. No pull request is created for RC branches.

### 3. Review and merge the pull request

After the `prep` job completes, a draft pull request from `v1.2.3` into `main` is automatically created. Review the version bump and merge it.

### 4. Draft GitHub release is created automatically

When the pull request is merged into `main`, the `release` job runs and creates a **draft** GitHub release tagged `v1.2.3` with the changelog as release notes.

### 5. Publish the GitHub release

Go to **Releases** on GitHub, review the draft release, and click **Publish release**.

This triggers the `publish` job, which:
- Checks out the `v1.2.3` tag
- Builds the package with `uv build`
- Validates the package with `twine check --strict`
- Publishes to PyPI using trusted publishing (no API token required)

The release will be live at: https://pypi.org/p/mf6adj

---

## Summary

```
Create branch v1.2.3
        │
        ▼
  prep job runs  ──► bumps version.py, creates artifacts
        │
        ▼
   pr job runs   ──► opens draft PR to main
        │
        ▼
   Merge PR to main
        │
        ▼
 release job runs ──► creates draft GitHub release (v1.2.3)
        │
        ▼
 Publish the release manually on GitHub
        │
        ▼
 publish job runs ──► builds & publishes to PyPI
```
