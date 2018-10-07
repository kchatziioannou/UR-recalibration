#!/bin/bash
echo "Compiling.."
rm *blg *log *aux *Notes* *bbl output.text
pdflatex --shell-escape bin-love-after-gw170817.tex > output.text
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> output.text
pdflatex --shell-escape bin-love-after-gw170817.tex >> output.text
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> output.text
bibtex bin-love-after-gw170817.aux >> output.text
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> output.text
pdflatex --shell-escape bin-love-after-gw170817.tex >> output.text
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> output.text
pdflatex --shell-escape bin-love-after-gw170817.tex >> output.text
echo "Done. Enjoy your PDF!"
evince bin-love-after-gw170817.pdf &
