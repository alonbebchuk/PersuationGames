#!/bin/bash

bash figures/convert_figures.sh

# Run pdflatex multiple times to resolve all references, citations, and cross-references
pdflatex project.tex
pdflatex project.tex
pdflatex project.tex

rm project.aux project.log project.out
