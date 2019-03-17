#!/bin/bash
echo "Compiling.."
pdflatex --shell-escape bin-love-after-gw170817.tex > output.text
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> output.text
pdflatex --shell-escape bin-love-after-gw170817.tex >> output.text
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> output.text
bibtex bin-love-after-gw170817.aux >> output.text
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> output.text
pdflatex --shell-escape bin-love-after-gw170817.tex >> output.text
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" >> output.text
pdflatex --shell-escape bin-love-after-gw170817.tex >> output.text
rm *blg *log *aux *Notes*  output.text *out
mv bin-love-after-gw170817.pdf ../
echo "Done. Enjoy your PDF!"
evince ../bin-love-after-gw170817.pdf &
