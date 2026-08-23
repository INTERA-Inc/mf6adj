# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking changes

- A model whose packages add their own equations to the MODFLOW 6 solution
  matrix is rejected instead of returning corrupted sensitivities. `maw6`
  always adds equations, and so does an implicitly coupled lake (MODFLOW 6
  6.8.0 and later); `sfr6`, `lak6`, and `uzf6` are solved in the outer
  iteration and are unaffected.

### Added

- `lak6` performance measures, so a measure can sum the exchange between a lake
  and the aquifer, with lake stage and conductance among the parameters.
- The lake water balance is solved with the flow equations, so a sensitivity to
  a lake with a free stage is a total derivative rather than one that holds the
  stage fixed. Outlets and stage-volume-area tables are included. Inflows
  routed to a lake by `mvr6`, and the stage dependence of a horizontal
  connection's conductance, are not, and are reported as warnings.

### Fixed

- Boundary values are read from each package's own arrays rather than from
  `BOUND`, which MODFLOW 6 leaves allocated but zeroed. The stage and
  conductance sensitivities of `ghb6`, `riv6`, `drn6`, and `chd6` were reported
  as zero.
- A cell holding more than one boundary from the same package accumulates all
  of them rather than keeping only the last.
- A flux performance measure uses its entry weight, and its direct term is
  applied only to the boundaries the measure names rather than to every
  head-dependent package.

## [1.2.0] - 2026-08-03

### Breaking changes

- `Mf6Adj.solve_adjoint()` and `PerfMeas.solve_adjoint()` no longer accept
  `skip_solve`. The flag applied to every performance measure form, but a
  transient `direct` or `residual` measure carries information backward from
  one time step to the next, so skipping a time step returned incorrect
  sensitivities with no indication that anything was wrong. Time steps with no
  entries are now skipped automatically, and only for the `instantaneous` form,
  where each time step is solved on its own and skipping is correct.

### Changes

- post v1.1.0 updates (#63) (@jdhughes-dev)
- Bump actions/checkout from 6 to 7 (#65) (@app/dependabot)
- Bump prefix-dev/setup-pixi from 0.9.6 to 0.10.0 (#67) (@app/dependabot)
- Bump actions/setup-python from 6 to 7 (#68) (@app/dependabot)
- fix(adj): detect IHIGHCELLSAT instead of comparing version strings (#69) (@jdhughes-dev)
- feat(pm)!: Add instantaneous performance measure type and remove skip_solve (#64) (@jdhughes-dev)
- ci(release): start a release from dropdowns and add rehearsal modes (#70) (@jdhughes-dev)
- ci(release): only allow a release to be cut from main (#71) (@jdhughes-dev)


## [1.1.0] - 2026-06-02

### Changes

- Release 1.0.0 (#53) (@app/github-actions)
- release: resync develop with main (#54) (@jdhughes-dev)
- release: develop resync after release (#55) (@jdhughes-dev)
- Bump prefix-dev/setup-pixi from 0.9.4 to 0.9.5 (#56) (@app/dependabot)
- Bump dawidd6/action-download-artifact from 19 to 20 (#57) (@app/dependabot)
- Bump dawidd6/action-download-artifact from 20 to 21 (#58) (@app/dependabot)
- Bump prefix-dev/setup-pixi from 0.9.5 to 0.9.6 (#59) (@app/dependabot)
- Add jacobi preconditioner (#60) (@jdhughes-dev)


## [1.0.0] - 2026-03-29

### Changes

- ruff formatting (#7) (@jdhughes-dev)
- remove use of local versions of python packages and executables (#8) (@jdhughes-dev)
- Add pyproject.toml (#9) (@jdhughes-dev)
- Std line endings (#10) (@jdhughes-dev)
- Refs/heads/feat mhtests (#11) (@jtwhite79)
- add support for disu grids (#12) (@jdhughes-dev)
- Feat dewater (#13) (@jtwhite79)
- add get-modflow bit to readme (#14) (@kmarkovich)
- fix lint issues (#16) (@jdhughes-dev)
- add pixi for ci (#17) (@jdhughes-dev)
- add support for high_cell_sat functionality (#18) (@jdhughes-dev)
- merge develop into main (#20) (@jdhughes-dev)
- Main (#21) (@jdhughes-dev)
- add pre-commit hook (#22) (@jdhughes-dev)
- v1.1.0rc (#23) (@jdhughes-dev)
- optimization and solver updates (#25) (@jdhughes-dev)
- Fix logger so that it can be called multiple times in a loop (#26) (@jdhughes-dev)
- Add custom dvclose convergence criteria callback for scipy solvers (#27) (@jdhughes-dev)
- feat(solve_adjoint): add rclose custom convergence check (#28) (@jdhughes-dev)
- Add option to skip adjoint solve for time steps without performance measures (#29) (@jdhughes-dev)
- feat(util): add workspace context manager (#30) (@jdhughes-dev)
- doc: add initial readthedocs files and GHActions workflow (#31) (@jdhughes-dev)
- doc: add rendered notebooks to readthedocs (#32) (@jdhughes-dev)
- docs: allow trigger_rtd with push or workflow_dispatch (#34) (@jdhughes-dev)
- Change GitHub token environment variable to RTDS (#35) (@jdhughes-dev)
- doc: fix paths for uploaded assets (#36) (@jdhughes-dev)
- rtd: fix issue with readthedocs push branch identification (#38) (@jdhughes-dev)
- ci: add dependabot and update release.yml (#39) (@jdhughes-dev)
- Bump dawidd6/action-download-artifact from 14 to 19 (#44) (@app/dependabot)
- Bump actions/upload-artifact from 4 to 7 (#43) (@app/dependabot)
- Bump actions/setup-python from 5 to 6 (#42) (@app/dependabot)
- Add rtds-action to project dependencies (#45) (@jdhughes-dev)
- Bump prefix-dev/setup-pixi from 0.9.3 to 0.9.4 (#40) (@app/dependabot)
- Bump actions/checkout from 4 to 6 (#41) (@app/dependabot)
- refactor: major refactor (#37) (@jdhughes-dev)
- doc: add rtd usage section (#47) (@jdhughes-dev)
- ci: update release markdown and add checklist to draft release PR (#48) (@jdhughes-dev)
- doc: update README.md for pypi and add citation (#49) (@jdhughes-dev)
- fix: change master -> main in release workflow and docs (#50) (@jdhughes-dev)


