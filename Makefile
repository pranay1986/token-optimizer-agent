.PHONY: all clean

all: main.pdf

main.pdf: main.tex sections/*.tex
	pdflatex main.tex
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm -f *.aux *.log *.out *.toc *.synctex.gz
	rm -f sections/*.aux
