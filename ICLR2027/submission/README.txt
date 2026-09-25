ICLR 2027 submission — source

Compile from THIS directory, keeping the subdirectories as they are:

    tectonic -X compile main_iclr2027_final.tex

or, with a standard TeX Live:

    pdflatex main_iclr2027_final
    bibtex   main_iclr2027_final        # not needed: main_iclr2027_final.bbl ships here
    pdflatex main_iclr2027_final
    pdflatex main_iclr2027_final

Expected result: 29 pages, main text through page 9, appendix from page 16.

Layout (every path in the .tex is relative to this directory):

    main_iclr2027_final.tex        the paper
    main_iclr2027_final.bbl        the bibliography, already compiled
    references.bib                 its source
    tab_khrulkov_final.tex         the table of Section 5.1, \input from the appendix
    appendix_tables/final/*.tex    the 12 appendix tables, \input by name
    figures/*.pdf                  the 9 figures, \includegraphics by name
    iclr2027_conference.sty/.bst   the style files, natbib.sty and fancyhdr.sty beside them
    main_iclr2027_final.pdf        the compiled paper, for comparison

If a table or a figure is reported as missing, the subdirectories were flattened
on upload: upload submission.zip whole, or keep appendix_tables/final/ and
figures/ as directories.
