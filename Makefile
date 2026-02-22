.PHONY: all clean

all: main.pdf

main.pdf: main.tex references.bib
	pdflatex -interaction=nonstopmode main
	bibtex main
	pdflatex -interaction=nonstopmode main
	pdflatex -interaction=nonstopmode main

clean:
	rm -f main.pdf main.aux main.bbl main.blg main.log main.out main.toc
