# mf6adj: a generic adjoint solver for MODFLOW-6

![mf6adj](.images/mf6adj.png)

`mf6adj` is a python implementation of the adjoint sensitivity analysis approach.  It does not require modification to the MODFLOW-6, instead it uses the MODFLOW-6 API (Hughes and others, 2022) to access the requisite solution components.  `mf6adj` supports a wide range of performance measures and parameters.


## Installation

`mamba env create -f environment.yml`

Activate the environment and add the MODFLOW6 executables to the mamba environment bin with the following:

`mamba activate mf6adj`

`get-modflow --subset mf6,libmf6,gridgen :python`

## Examples

Several notebooks are provide that demonstrate how to use `mf6adj`

