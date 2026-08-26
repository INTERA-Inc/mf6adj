# Supplemental technical information

Technical detail behind extensions made to mf6adj after the original
publication. One chapter per topic: the instantaneous performance measure,
storage, the Lake Package, and the Streamflow Routing Package.

## Building

Requires a LaTeX distribution providing `pdflatex` and `bibtex`.

```shell
make          # writes mf6adjsuppinfo.pdf
make clean    # removes the build artifacts
```

## Figures

The figures are committed under `Figures/`, so building the document never runs
a model. They are drawn from a small archive of capture fractions in `data/`,
which `scripts/make_figures.py` reads:

```shell
python scripts/make_figures.py data/synthetic-valley-capture.npz Figures
```

The archives are recomputed only when the results change:

```shell
python scripts/make_package_summary.py data        # lake and streamflow
python scripts/make_instantaneous_summary.py /tmp/inst data/instantaneous-comparison.npz
```

The lake and streamflow archives are built from the models the mf6adj tests use,
so the figures show the lake and the stream whose behavior those tests check.

## Adding a chapter

Write the chapter as its own `.tex` file containing the body only, then add it
to `body.tex` with a `\chapter` heading and a label, and list it among the
prerequisites in the `Makefile`.
