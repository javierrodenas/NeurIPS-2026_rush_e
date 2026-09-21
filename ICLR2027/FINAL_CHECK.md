# FINAL_CHECK — `main_iclr2027_final.pdf`

## Page budget

- PDF pages: 33. Main text (through the Conclusion and Limitations section) ends on page 9; the Reproducibility, Ethics and AI Use statements start on page 9 (ICLR line 485); the references start on page 10 (line 503).
- Appendix: Proofs on page 13, tables from page 13 to page 33 (21 pages including the proofs).
- Cut order applied (brief §5): s55, s6, table, s2 = S5.5 to three sentences, S6 three paragraphs, model table in the appendix, S2 to its first six sentences. Figures were not resized.
- Beyond the cut order (the four steps freed about 25 of the 72 lines the first compile was over), the page was reached by trimming the non-verbatim prose sentence by sentence without dropping a claim (every kept claim is in `V3_OUTLINE.md`; the removed sentences restated a claim made in the same or a neighbouring paragraph) and by typographic spacing declared in the preamble: section/subsection/paragraph heading skips 1.2/1.0/0.5 ex (style: 2.0/1.8/1.5 ex), display skips 4 pt, definition and proposition environments 3 pt above and below, float separation 12 pt and caption skip 5 pt. Fonts, margins, line spacing and figure sizes are the style's and the brief's. Everything is in `rebuttal/scripts/phaseE_submission.py` and reverts by deleting those lines.

## Sentence length per section (compiled text of the final file; verbatim sections included in the statistics, the rules are enforced on the non-verbatim prose)

| section | sentences | mean words | longest |
|---|---|---|---|
| Introduction | 31 | 23.7 | 76 |
| Related Work | 5 | 39.8 | 60 |
| Methodology | 42 | 19.8 | 45 |
| Experimental Setup | 15 | 18.2 | 33 |
| Results | 87 | 16.8 | 42 |
| Implications for Hyperbolic Representation Learning | 12 | 15.3 | 33 |
| Conclusion and Limitations | 14 | 24.4 | 58 |

Non-verbatim prose of S3–S7: 154 sentences, mean 18.0 words (rule: mean ≤ 22, none > 35).

## Numbers per paragraph in S5–S6 (rule: ≤ 1 per sentence, ≤ 2 per paragraph, headline numbers only)

| section | paragraph | numbers |
|---|---|---|
| Results | Raw readings are not evidence, ours or published, and calibr | — |
| Results | A published reading is reproduced and calibrated. | 0.03, 0.05 |
| Results | Most cells show structure beyond the second moments. | 44 of 72, 30 of 36, 47, 18 of 24 |
| Results | The count survives resampling. | — |
| Results | No family owns the structure. | 5 of 12 |
| Results | A star of clusters already passes the census. | — |
| Results | Neural collapse is the flat limit, not what the census sees. | — |
| Results | The alignment of each cluster with its hub is certified in f | 4 of 12 |
| Results | No hierarchy above the superclasses is found in the supervis | 4 of 5, 1.3 to 2.0, 3.0 to 3.9 |
| Results | A deep hierarchy at the real noise level is detected. | 4 of 5, 0 of 5, 8 of 50, -4.2 to -4.8, -1.6 to -1.8 |
| Results | The test never fires on randomized hubs. | 0 of 60, 0 of 60, 0 of 4 |
| Results | The two-level implant is missed at ImageNet's noise level. | — |
| Results | The certified set depends on the frame. | — |
| Results | Leaf labels can produce the depth but do not guarantee it. | — |
| Results | Imposing the geometry does not create detected depth. | — |
| Results | The naive map manufactures an island. | — |
| Results | Controlling the cut leaves a moderate gap. | — |
| Results | The self-supervised tree is angular. | — |
| Results | Alignment with WordNet follows supervision and recipe. | — |
| Results | The recovery is not WordNet circularity. | — |
| Results | Text depends on recipe, scale and probe. | 7 of 15 |
| Results | Models share neighborhoods, not metrics. | — |
| Implications for Hyperbolic  | The raw reading cannot select a curvature. | 0.48 to 2.5 |
| Implications for Hyperbolic  | Both readings predict the gain, and the depth verdict predic | +0.41 against +0.28 |
| Implications for Hyperbolic  | The calibration certifies structure and does not choose the  | +0.9 to +1.3 |

## Appendix material deleted from the final (kept in the frozen v1 file and its tables)

- Whole tables: tab_q11_interventions, tab_q12_xi (the parallelogram descriptor ξ; the interventions/ORC table: Ollivier–Ricci curvature by edge type and the intervention rows beyond the two the text cites).
- The null-variant panels: (c) of the robustness table (four null × statistic constructions per cell with the joint listing) and (b) of the census table (verdict under the four constructions); the superseded supremum × Gaussian rows of the budget panel and panel (b) of the corollary table (correlations of the raw supremum reading), both uncited; the class-count sweep beyond one row (DINOv2-L at C = 100 is kept, both modes); the training-intervention rows beyond one per intervention.
- Appendix length: the brief asked for at most 14 pages; the kept material occupies 16 (proofs included) with the tables floating one panel at a time. Candidates if the author wants the last two pages: corollary panels (f) nearest-centroid policies and (g) sensitivity to the projection target (uncited from the main text), the text table's extraction panel (c), the model-panel table's per-model extraction notes.
- Every appendix figure (depth on CIFAR-100, depth power, causal interventions, CIFAR-100 tree map, text nulls, best-metric scatter) and every reference to them.
- The limitation item on ξ (former (viii)) went with the descriptor.
- Kept, in first-citation order: tab_q08_robust, tab_q10_calibration, tab_q01_census, tab_q14_panel, tab_q02_text, tab_q03_sample, tab_q04_depth, tab_q05_power, tab_q06_treemap, tab_q07_wordnet, tab_q13_local, tab_q09_corollary, plus the provenance index (tab_z_provenance_final).

## Sweep (rebuttal/scripts/sweep_freeze.py)

- Total: 215/216 checks passed.
- Final-version checks:

    PASS final: 'Gromov delta' and 'Estimation and normalization' verbatim modulo the recorded edits (supremum phrase, bridge sentence) 
    PASS final: the recorded edits are exactly the briefs' (shadow x2, geometric face, intent of the supremum, the bridge; 4th/5th reviews: abstract, S1 confounds and counts, S2 citations) and none of the old phrases survives 
    PASS final: no metaphor outside Figure 1 and its caption (shadow, star caveat, Aristotelian, geometric face, the intent of the supremum) 
    PASS final: no bridge sentences anywhere in the body 
    PASS final: thesis verbatim exactly twice (abstract's last sentence, S7 conclusion), no short form 
    PASS final: skeleton unchanged from v3 (seven sections, eleven subsections), seven definitions inline and unframed, six numbered equations (the curvature rule inline since the brief of 2026-09-21), Proposition 1 (a)(b) with its one proof in Appendix A, no boxes and no colored text 
    PASS final: plain-prose rules on the non-verbatim prose of S3-S7 (avg <= 22 words, none > 35, paragraphs of 3-6 sentences with a plain bold lead-in, S5-S6 <= 1 number per sentence and <= 2 per paragraph with one pointer in the last sentence, only headline numbers, no parenthetical over three words, no semicolon chains, no banned phrases, provenance comment on every results paragraph) 
    PASS final: the seven defined terms are each defined once in S3 (one definition environment each) and 'premise' is fixed in S1 
    PASS final: no number appears in the final that is not in v1, in a table or in a fill traced to a result file 
    PASS final: every result file named in a provenance comment of the main text exists 
    PASS final: Figure 1 is the author's figure command verbatim from main_local.tex and Figures 2-5 are the bar-language files fig_overview_final, fig_excess_final, fig_depth_final, fig_treemap_final, all present 
    PASS final: every main-text figure caption opens with a bold takeaway and names its source files 
    PASS final: the Figure 5 caption states the naive and the selected DINOv2-vs-block agreement read from the figure's data 
    PASS final: the appendix inputs exactly the tables the main text cites (twelve question tables, robustness in its final form) plus the provenance index, in first-citation order, from appendix_tables/final/ 
    PASS final: xi, ORC/interventions table, appendix figures, null-variant panel and the class-count sweep are gone from the final and nothing refers to them 
    PASS final: every kept appendix table keeps its provenance comments (% prov: lines and % source comments) and the provenance index lists all thirteen 
    PASS final: each final copy in appendix_tables/final/ carries the numbers and captions of its v1 table (minus the dropped panel in the census and corollary tables), floating and split by panel 
    PASS final: every cross-reference of the final resolves 
    PASS final: Table 1 is the census on the centered Haar record with the short caption (tab_census_final from expR75, twelve rows, same layout as v1's tab_census) 
    PASS final: AI Use Statement with exactly the three declared items and the responsibility sentence; Reproducibility with the anonymized-repository placeholder; Ethics present 
    PASS final: preamble of the frozen v1 plus amsthm only (same class, same packages) 
    FAIL(all): final: abstract is the recorded text (250 words or fewer, cell defined in its own sentence, the deep synthetic hierarchy detected and no hierarchy above the superclasses found), 15 text models 
