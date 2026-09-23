# FINAL_CHECK — `main_iclr2027_final.pdf`

## Page budget

- PDF pages: 37. Main text (through the Conclusion and Limitations section) ends on page 9; the Reproducibility, Ethics and AI Use statements start on page 9 (ICLR line 471); the references start on page 10 (line 490).
- Appendix: Proofs on page 13, tables from page 13 to page 37 (25 pages including the proofs).
- Cut order applied (brief §5): s55, s6, table, s2 = S5.5 to three sentences, S6 three paragraphs, model table in the appendix, S2 to its first six sentences. Figures were not resized.
- Beyond the cut order (the four steps freed about 25 of the 72 lines the first compile was over), the page was reached by trimming the non-verbatim prose sentence by sentence without dropping a claim (every kept claim is in `V3_OUTLINE.md`; the removed sentences restated a claim made in the same or a neighbouring paragraph) and by typographic spacing declared in the preamble: section/subsection/paragraph heading skips 1.2/1.0/0.5 ex (style: 2.0/1.8/1.5 ex), display skips 4 pt, definition and proposition environments 3 pt above and below, float separation 12 pt and caption skip 5 pt. Fonts, margins, line spacing and figure sizes are the style's and the brief's. Everything is in `rebuttal/scripts/phaseE_submission.py` and reverts by deleting those lines.

## Sentence length per section (compiled text of the final file; verbatim sections included in the statistics, the rules are enforced on the non-verbatim prose)

| section | sentences | mean words | longest | over 30 |
|---|---|---|---|---|
| Introduction | 35 | 21.8 | 50 | 0 |
| Related Work | 7 | 28.6 | 60 | 0 |
| Methodology | 37 | 18.1 | 30 | 0 |
| Experimental Setup | 16 | 16.2 | 25 | 0 |
| Results | 77 | 17.7 | 30 | 0 |
| Implications for Hyperbolic Representation Learning | 7 | 15.9 | 21 | 0 |
| Conclusion and Limitations | 21 | 18.8 | 34 | 0 |

Non-verbatim prose of S3–S7: 141 sentences, mean 17.7 words (rule: mean ≤ 22).
Rule of 2026-09-23: no sentence over 30 words in S3–S7 (pages 3–9), definitions and citation lists excepted; sentences over the cap outside those: 0; the thesis sentence, verbatim from the abstract, is the one kept exception (34 words).

## Numbers per paragraph in S5–S6 (rule: ≤ 1 per sentence, ≤ 2 per paragraph, headline numbers only)

| section | paragraph | numbers |
|---|---|---|
| Results | The premise does not survive calibration where it is read. | — |
| Results | A published reading is reproduced and calibrated. | 4, 0.03, 4 |
| Results | Most cells show structure beyond the second moments. | 44 of 72, 30 of 36, 3, 47, 18 of 24 |
| Results | The count survives resampling. | — |
| Results | The structure depends on the class set more than on the mode | 5 of 12 |
| Results | A star of clusters already passes the census. | — |
| Results | Neural collapse is the flat limit, not what the census sees. | — |
| Results | The hierarchy test certifies that clusters are oriented towa | 12 |
| Results | What it certifies is alignment. | 4, -1.61, -1.61 to -1.76, 0 of 60, 0 of 4, 4, -2.26 to -3.99 |
| Results | What the test can see. | 4 of 5, 4, 9 of 12, 0.83 to 1.00, 3, 0.01 to 0.60, 3, 1.5 to 3.0, 1.3 to 3.9, 9, 20 of 20, 2, -2.51, 6 of 20, -1.37, 0 of 5, 8 of 50, 4, -4.19 to -4.82, 0 of 50 |
| Results | No hub hierarchy is found where the test has power. | 9 |
| Results | The certified set depends on the frame. | 38 of 50, 7 of 10 |
| Results | Leaf labels can produce the alignment but do not guarantee i | — |
| Results | MERU's objective does not create hub structure. | — |
| Results | The naive map manufactures an island. | 12 |
| Results | Controlled, the trees share their topology, not their metric | 0.77, 0.90, 0.33 |
| Results | The self-supervised tree is angular. | — |
| Results | Agreement with WordNet follows supervision and recipe. | — |
| Results | The recovery is not WordNet circularity. | — |
| Results | Text depends on recipe and scale. | 7 of 15 |
| Results | Models share neighborhoods, not metrics. | 66 |
| Implications for Hyperbolic  | The raw reading cannot select a curvature. | 0.48 to 2.5 |
| Implications for Hyperbolic  | Both readings predict the gain, and the hierarchy verdict pr | — |
| Implications for Hyperbolic  | The calibration certifies structure and does not choose the  | +0.9 to +1.3 |

## Appendix material deleted from the final (kept in the frozen v1 file and its tables)

- Whole tables: tab_q11_interventions, tab_q12_xi (the parallelogram descriptor ξ; the interventions/ORC table: Ollivier–Ricci curvature by edge type and the intervention rows beyond the two the text cites).
- The null-variant panels: (c) of the robustness table (four null × statistic constructions per cell with the joint listing) and (b) of the census table (verdict under the four constructions); the superseded supremum × Gaussian rows of the budget panel and panel (b) of the corollary table (correlations of the raw supremum reading), both uncited; the class-count sweep beyond one row (DINOv2-L at C = 100 is kept, both modes); the training-intervention rows beyond one per intervention.
- Appendix length: the brief asked for at most 14 pages; the kept material occupies 16 (proofs included) with the tables floating one panel at a time. Candidates if the author wants the last two pages: corollary panels (f) nearest-centroid policies and (g) sensitivity to the projection target (uncited from the main text), the text table's extraction panel (c), the model-panel table's per-model extraction notes.
- Every appendix figure (depth on CIFAR-100, depth power, causal interventions, CIFAR-100 tree map, text nulls, best-metric scatter) and every reference to them.
- The limitation item on ξ (former (viii)) went with the descriptor.
- Kept, in first-citation order: tab_q08_robust, tab_q10_calibration, tab_q01_census, tab_q14_panel, tab_q02_text, tab_q03_sample, tab_q04_depth, tab_q05_power, tab_q06_treemap, tab_q07_wordnet, tab_q13_local, tab_q09_corollary, plus the provenance index (tab_z_provenance_final).

## Sweep (rebuttal/scripts/sweep_freeze.py)

- Total: 224/227 checks passed.
- Final-version checks:

    PASS final: every main-text figure caption opens with a bold takeaway and names its source files 
    PASS final: the appendix ARI-matrix figure states the naive and the selected DINOv2-vs-block agreement read from the figure's data, and Figure 5 carries the author's caption (consolidated pass) 
    PASS final: the appendix inputs exactly the tables the main text cites (twelve question tables, robustness in its final form) plus the provenance index, in first-citation order, from appendix_tables/final/ 
    PASS final: xi, ORC/interventions table, appendix figures, null-variant panel and the class-count sweep are gone from the final and nothing refers to them 
    PASS final: every kept appendix table keeps its provenance comments (% prov: lines and % source comments) and the provenance index lists all thirteen 
    PASS final: each final copy in appendix_tables/final/ carries the numbers and captions of its v1 table (minus the dropped panel in the census and corollary tables), floating and split by panel 
    PASS final: references.bib is the author's bib of 2026-09-23 merged with the three cited entries it lacked: keys unique, every cited key present, Groger et al. as ICML 2026 with Shuo Wen, no escaped underscores in doi/url (they print a backslash), Gromov protected 
    PASS final: every cross-reference of the final resolves 
    PASS final: Table 1 is the census on the centered Haar record with the short caption (tab_census_final from expR75, twelve rows, same layout as v1's tab_census) 
    PASS final: AI Use Statement with exactly the three declared items and the responsibility sentence; Reproducibility with the anonymized-repository placeholder; Ethics present 
    PASS final: preamble of the frozen v1 plus amsthm only (same class, same packages) 
    FAIL(all): final: the recorded edits are exactly the briefs' (shadow x2, geometric face, intent of the supremum, the bridge; 4th/5th reviews: abstract, S1 confounds and counts, S2 citations) and none of the old phrases survives 
