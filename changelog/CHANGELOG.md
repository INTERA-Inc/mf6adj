# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking changes

- A model whose packages add their own equations to the MODFLOW 6 solution
  matrix is now rejected instead of returning corrupted sensitivities. The
  adjoint matrix is rebuilt from the groundwater-flow grid connectivity, which
  has no rows for those equations, so the coefficients were misaligned and the
  backward recursion diverged - on a 21-period model the sensitivities grew by
  about a factor of ten per period, reaching 1e18, with no indication that
  anything was wrong. `maw6` always adds equations, and an implicitly coupled
  lake (MODFLOW 6 6.8.0 and later) does too. `sfr6`, `lak6`, and `uzf6` are
  solved in the outer iteration, add nothing to the matrix, and are unaffected.

### Added

- The lake water balance is solved with the flow equations, so a measure's
  sensitivity to a lake with a free stage is a total derivative rather than a
  partial derivative that holds the stage fixed. The adjoint system is bordered
  with one equation per lake, and the lake carries its own storage backward in
  time the way the aquifer does. A lake held at a constant stage is unaffected,
  since its stage is not a dependent variable. Lake outlets are included: a
  Manning or weir outlet discharge follows a power of the depth over its
  invert, so its derivative joins the lake's own equation, and an outlet
  feeding another lake couples the two. A lake with a stage-volume-area table
  stores against that table rather than against its connection areas, and its
  evaporating surface grows with stage. Inflows routed to a lake by the water
  mover are not differentiated, because that couples UZF, LAK, and SFR to each
  other, so a lake fed through `mvr6` still has a residual error; nor is the
  dependence of a horizontal connection's conductance and wetted area on stage.
  Both cases are reported as warnings. The lake terms live in `mf6adj/lake.py`
  so the adjoint solver stays free of package detail.
- `lak6` performance measures. A lake connection exposes the same nodelist,
  conductance, and flux terms as the other head-dependent boundaries, so a
  measure can now sum the exchange between a lake and the aquifer, and lake
  stage and conductance appear as parameters in the results.

### Fixed

- Boundary values are read from each package's own arrays rather than from
  `BOUND`. MODFLOW 6 leaves `BOUND` allocated but zeroed for these packages, so
  the stage and conductance sensitivities of `ghb6`, `riv6`, `drn6`, and `chd6`
  were reported as zero.
- A cell holding more than one boundary from the same package accumulates every
  boundary rather than keeping only the last one. A lake connected both
  vertically and horizontally to one cell, or two river reaches in one cell,
  previously contributed once.
- A flux performance-measure entry uses its weight. The forward value scales the
  package flux by the entry weight, but the adjoint right-hand side ignored it,
  so a weighted measure returned unscaled sensitivities.
- The direct term of a flux measure is applied only to the boundaries the
  measure names, and with that entry's weight. It was applied to every boundary
  of every head-dependent package as soon as a measure contained any flux entry,
  which biased the sensitivities to boundary parameters the measure never used.

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


