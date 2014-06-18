document = characters

${document}.jpg: ${document}.pdf
	convert -density 150 ${document}.pdf -quality 90 ${document}.jpg

${document}.pdf: ${document}.tex
	pdflatex ${document}.tex
