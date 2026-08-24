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

The archive itself is recomputed only when the results change, by
`scripts/make_summary.py`. That step runs the synthetic valley model and the
adjoint solution, and needs a clone of the MODFLOW 6 training repository, which
is where that model lives:

```shell
python scripts/make_summary.py <mf6-training clone> \
    data/synthetic-valley-capture.npz /tmp/sv
```

## Adding a chapter

Write the chapter as its own `.tex` file containing the body only, then add it
to `body.tex` with a `\chapter` heading and a label, and list it among the
prerequisites in the `Makefile`.
