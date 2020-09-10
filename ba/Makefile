fname=Notes_about_network_propagation
baname=Bachelorthesis
Notes_about_network_propagation.pdf: Notes_about_network_propagation.tex mybib.bib
	pdflatex $(fname)
	biber $(fname)
	pdflatex $(fname)

Bachelorthesis.pdf: BachelorThesis.pdf mybib.bib
	pdflatex $(baname)
	biber $(baname)
	pdflatex $(baname)
