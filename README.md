# mf6adj

![mf6adj](https://raw.githubusercontent.com/INTERA-Inc/mf6adj/main/.images/mf6adj.png)

**mf6adj** is a Python package for adjoint-state sensitivity analysis with
MODFLOW 6.  It uses the [MODFLOW 6 API](https://github.com/MODFLOW-USGS/modflowapi)
to access the internal solution components at run time — no modifications to
MODFLOW 6 are required.  Given one or more user-defined performance measures
(heads, boundary fluxes, or composite objectives), mf6adj computes the
sensitivity of each measure to model parameters across the full model domain.

## Installation

**pip**

```bash
pip install mf6adj
```

**conda-forge**

```bash
conda install -c conda-forge mf6adj
```

mf6adj drives MODFLOW 6 through its shared library (`libmf6`).  The easiest
way to get both is through the
[`flopy`](https://github.com/modflowpy/flopy) helper:

```bash
get-modflow --subset mf6,libmf6 :python
```

## Quick start

```python
import flopy
import mf6adj

# locate the MF6 binary and shared library in the active conda environment
mf6_bin, lib_name = mf6adj.get_conda_mf6_paths()

# run the baseline forward model
flopy.run_model(exe_name=mf6_bin, namefile=None, model_ws="path/to/model")

# write a performance-measure file
with open("path/to/model/model.adj", "w") as f:
    f.write("begin performance_measure head_obs\n")
    f.write("1 1 1 5 5 head direct 1.0 -1.0e+30\n")
    f.write("end performance_measure\n")

# solve forward and adjoint
adj = mf6adj.Mf6Adj("model.adj", str(lib_name), working_directory="path/to/model")
adj.solve_forward_model()
sensitivity_dfs = adj.solve_adjoint()
adj.finalize()

print(sensitivity_dfs["head_obs"])
```

## Documentation

Full documentation, including API reference and example notebooks, is available
at [mf6adj.readthedocs.io](https://md6adj.readthedocs.io).

## How to cite

If you use mf6adj in your work, please cite:

> Hayek, M., White, J. T., Markovich, K. H., Hughes, J. D., & Lavenue, M. (2025).
> MF6-ADJ: A Non-Intrusive Adjoint Sensitivity Capability for MODFLOW 6.
> *Groundwater*, 63(6), 874–888. https://doi.org/10.1111/gwat.70025

## License

[CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/)
