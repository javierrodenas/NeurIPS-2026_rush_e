ICLR 2027 submission - source

Upload this whole directory (or submission.zip) to Overleaf and compile
main_iclr2027_final.tex. Every path in it is relative to this directory:

    main_iclr2027_final.tex   the paper            appendix_tables/   the 12 appendix tables
    references.bib            the bibliography     figures/           the 9 figures
    main_iclr2027_final.bbl   already compiled, so no bibtex pass is needed
    iclr2027_conference.sty/.bst, natbib.sty, fancyhdr.sty, tab_khrulkov_final.tex

    tectonic -X compile main_iclr2027_final.tex      # or pdflatex three times

Expected: 29 pages, main text through page 9, appendix from page 16.
main_iclr2027_final.pdf is the reference build; delete it before uploading if you prefer.
