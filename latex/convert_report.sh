#!/bin/bash

bash figures/convert_figs.sh

# Run pdflatex multiple times to resolve all references, citations, and cross-references
pdflatex report.tex
pdflatex report.tex
pdflatex report.tex

rm report.aux report.log report.out
