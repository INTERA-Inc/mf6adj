# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-09-09

### Breaking changes

- Python 3.11 is now the minimum. modflowapi 1.0.0, which this release
  requires, does not support Python 3.9 or 3.10. `pyemu` is no longer a
  dependency, and `pandas` 3 is now allowed.

- A sensitivity to recharge was short by the area of the cell, because MODFLOW 6
  treats recharge as a rate over that area and the sensitivity was reported as
  the adjoint state alone. On a 250 m grid the reported value was too small by a
  factor of 62,500, and by a varying factor on a grid with variable cell sizes.
  Any recharge sensitivity from an earlier release is wrong by that factor.

- A sensitivity to hydraulic conductivity was formed from the conductance the
  grid describes rather than the one the model used, so a model with a
  horizontal flow barrier was wrong at every cell a barrier touched. On a
  confined test model the reported value was 694 times a finite-difference
  derivative. Cells away from a barrier were unaffected, and a model with no
  barrier is unchanged.

- A package given `AUXMULTNAME` scales the values it applies by an auxiliary
  variable, and MODFLOW applies that where it forms its terms rather than
  folding it into the values it keeps. The multiplier was not carried, so
  recharge, well, drain, river and general-head sensitivities were reported for
  the unscaled values.

- Storage terms are now selected the way MODFLOW 6 selects them, on the cell
  saturation, and under `SS_CONFINED_ONLY` the specific-storage term is dropped
  for a cell that is not full rather than scaled. Specific storage was
  previously applied at full cell thickness everywhere, so a partially saturated
  cell carried a storage term the forward model does not have.

- A specific-yield sensitivity is reported beside the specific-storage one, as
  `sy` in the composite results. A convertible cell releases water both ways and
  only one was reported before, so a storage sensitivity for such a cell was
  incomplete rather than merely differently named.

- A drain sitting on its activation threshold is dropped from the
  performance-measure derivative. A drain without a drainage depth switches on
  and off at its elevation and a converged solution leaves drains sitting on
  that corner, where the derivative is the conductance from one side and zero
  from the other. The full conductance was taken regardless.

- Three cases that used to run are now refused, each because the answer they
  produced could not be right:

  - a flow model that used XT3D, whose flow between two cells is not the
    conductance the sensitivity differentiates;
  - a performance measure of a specified flow, a well rate or a recharge rate,
    which has no derivative and was reported as zero everywhere;
  - a performance measure naming a package the adjoint forms no terms for,
    which used to fail later with a `KeyError` naming an object in an HDF file.

- A model whose matrix is not the derivative of its equations is now reported.
  Under the standard formulation the derivative of the transmissivity with
  respect to the head is lagged rather than assembled, so a model with
  convertible cells holds the transmissivity fixed in the sensitivity. On a
  single-layer unconfined model the error was 5.9 percent with the pumped cell
  near the middle of its thickness and 10.2 percent near the bottom. The term is
  not recovered; the condition is reported.

### Changes

- ci(release): use only the new version's changelog entry as release notes (#73) (@jdhughes-dev)
- docs(changelog): remove the duplicate 1.1.0 entry (#74) (@jdhughes-dev)
- Bump actions/download-artifact from 7 to 8 (#76) (@app/dependabot)
- Bump prefix-dev/setup-pixi from 0.10.0 to 0.10.1 (#77) (@app/dependabot)
- chore: open the 1.3.0 development cycle (#75) (@jdhughes-dev)
- feat(pm): add lak6 performance measures and reject matrix-coupled packages (#79) (@jdhughes-dev)
- feat(pm): solve the lake water balance with the flow equations (#80) (@jdhughes-dev)
- feat(sfr): solve the reach routing with the flow equations (#81) (@jdhughes-dev)
- fix(pm): drop drain entries sitting on their activation threshold (#84) (@jdhughes-dev)
- fix(adj): select the storage terms the way MODFLOW 6 does (#87) (@jdhughes-dev)
- fix(pm): scale the recharge sensitivity by the cell area (#88) (@jdhughes-dev)
- test(theis): verify the adjoint against the Theis analytical solution (#92) (@jdhughes-dev)
- fix(adj): reject a performance measure of a specified flow (#91) (@jdhughes-dev)
- feat(adj): report specific storage and specific yield separately (#93) (@jdhughes-dev)
- feat(sfr): differentiate a reach with a cross section (#94) (@jdhughes-dev)
- feat(sfr): differentiate the flow a diversion takes (#95) (@jdhughes-dev)
- docs(suppinfo): add supplemental technical information (#89) (@jdhughes-dev)
- fix(pm): carry the auxiliary multiplier into the recharge sensitivity (#96) (@jdhughes-dev)
- refactor: move standard package terms out of the solve loop (#98) (@jdhughes-dev)
- fix(packages): carry the auxiliary multiplier into the remaining sensitivities (#100) (@jdhughes-dev)
- test(well): verify a reduced well rate, and report where it is not carried (#101) (@jdhughes-dev)
- feat(adj): report a model whose matrix is not the derivative of its equations (#103) (@jdhughes-dev)
- fix(adj): refuse an unsupported package while the input is read (#105) (@jdhughes-dev)
- chore(deps): bump prefix-dev/setup-pixi from 0.10.1 to 0.10.2 (#106) (@app/dependabot)
- refactor(adj): locate the adjoint matrix with the solution's sparsity (#107) (@jdhughes-dev)
- feat(maw): measure the exchange between a multi-aquifer well and the aquifer (#108) (@jdhughes-dev)
- chore(deps): require modflowapi 1.0.0 on python 3.11, and hold pandas below 3 (#109) (@jdhughes-dev)
- ci: test the second supported python version on linux (#110) (@jdhughes-dev)
- ci: leave the largest model out of the second python version (#111) (@jdhughes-dev)
- fix(npf): carry a horizontal flow barrier into the conductivity sensitivity (#114) (@jdhughes-dev)
- chore(dependencies): drop pyemu (#117) (@jdhughes-dev)
- feat(hfb): report the sensitivity to a barrier's hydraulic characteristic (#116) (@jdhughes-dev)
- fix(adj): refuse a flow model that used XT3D (#121) (@jdhughes-dev)
- build: drop Intel macOS, which MODFLOW 6 no longer builds for (#119) (@jdhughes-dev)
- build: let a task that runs other tasks fail when one of them does (#120) (@jdhughes-dev)
- chore(deps): update pandas requirement from <3,>=2.0.0 to >=2.0.0,<4 (#115) (@app/dependabot)
- chore(examples): remove the synthdewater build script (#118) (@jdhughes-dev)
- build: lift the pandas bound in the environment files too (#122) (@jdhughes-dev)


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


