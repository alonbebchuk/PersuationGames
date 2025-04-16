#!/bin/bash

rm -rf figures/temp_pdf
mkdir -p figures/temp_pdf
rm -rf figures/png
mkdir -p figures/png

for tex_file in figures/tex/*.tex; do
    base_name=$(basename "$tex_file" .tex)
    cd figures/tex && pdflatex -interaction=nonstopmode "$base_name.tex" && cd ../..
    mv "figures/tex/$base_name.pdf" figures/temp_pdf/ 2>/dev/null || true
    pdf_file="figures/temp_pdf/${base_name}.pdf"
    output_file="figures/png/${base_name}.png"
    gs -dQUIET -dBATCH -dNOPAUSE -sDEVICE=png16m -r300 -sOutputFile="$output_file" "$pdf_file"
done

rm -rf figures/temp_pdf
rm -f figures/tex/*.aux figures/tex/*.log
