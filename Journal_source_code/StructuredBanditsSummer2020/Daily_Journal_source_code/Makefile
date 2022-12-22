SOURCE = journal
#ERRATA = errata
TEXFILES = *.tex
PICS =  # Evantually add graphics path (e.g., figures)
BIBFILES = ../bibs/*.bib

all: $(SOURCE).pdf 

$(SOURCE).pdf: $(TEXFILES) $(BIBFILES) $(PICS)
	pdflatex --shell-escape $(SOURCE).tex
	bibtex $(SOURCE)
	pdflatex $(SOURCE).tex
	pdflatex $(SOURCE).tex
	pdflatex $(SOURCE).tex
#	pdflatex $(ERRATA).tex
#	bibtex $(ERRATA)
#	pdflatex $(ERRATA).tex
#	pdflatex $(ERRATA).tex


view: $(SOURCE).pdf
	open $(SOURCE).pdf
clean:
	-rm -f  $(SOURCE).pdf $(SOURCE).aux $(SOURCE).bbl $(SOURCE).blg \
	*.out *.backup *.tex~ *.loa *.lot *.lof *.log *.toc *.brf *.dvi

