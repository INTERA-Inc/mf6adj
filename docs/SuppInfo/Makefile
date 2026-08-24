# Build the supplemental technical information document.
#
#   make        build the pdf
#   make clean  remove the build artifacts

DOC = mf6adjsuppinfo
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: $(DOC).pdf

$(DOC).pdf: $(DOC).tex body.tex introduction.tex instantaneous.tex \
            storage.tex lake.tex sfr.tex references.bib \
            Figures/sfr-capture.pdf Figures/lake-capture.pdf
	$(LATEX) $(DOC)
	-bibtex $(DOC)
	$(LATEX) $(DOC)
	$(LATEX) $(DOC)

clean:
	rm -f $(DOC).aux $(DOC).bbl $(DOC).blg $(DOC).log $(DOC).out $(DOC).toc
