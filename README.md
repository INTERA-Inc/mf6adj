# mf6adj: a generic adjoint solver for MODFLOW-6

![mf6adj](.images/mf6adj.png)

`mf6adj` is a python implementation of the adjoint sensitivity analysis approach.  It does not require modification to the MODFLOW-6, instead it uses the MODFLOW-6 API (Hughes and others, 2022) to access the requisite solution components.  `mf6adj` supports a wide range of performance measures and parameters.


## Installation

`mamba env create -f environment.yml`

Activate the environment and add the MODFLOW6 executables to the mamba environment bin with the following:

`mamba activate mf6adj`

`get-modflow --subset mf6,libmf6,gridgen :python`

## VS Code

When working in VS Code, use the repository-local pixi interpreter so editor
diagnostics resolve project dependencies such as `modflowapi` correctly:

`./.pixi/envs/default/bin/python`

The workspace settings are already configured for this interpreter. If VS Code
still shows unresolved imports, run `Python: Select Interpreter` and choose the
pixi environment for this repository.

## Examples

Several notebooks are provide that demonstrate how to use `mf6adj`

