#!/bin/bash

bash figures/convert_figs.sh

pdflatex report.tex

rm report.aux report.log report.out
