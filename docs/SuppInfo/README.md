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

## Adding a chapter

Write the chapter as its own `.tex` file containing the body only, then add it
to `body.tex` with a `\chapter` heading and a label, and list it among the
prerequisites in the `Makefile`.
