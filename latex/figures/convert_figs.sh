#!/bin/bash

rm -rf figures/pdf
mkdir -p figures/pdf

for tikz_file in figures/*.tex; do
    base_name=$(basename "$tikz_file" .tex)
    echo "Compiling $tikz_file to PDF..."
    cd figures && pdflatex -interaction=nonstopmode "$base_name.tex" && cd ..
    mv "figures/$base_name.pdf" figures/pdf/ 2>/dev/null || true
done

rm -f figures/*.aux figures/*.log
