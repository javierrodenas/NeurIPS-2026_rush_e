# FINAL_CHECK — `main_iclr2027_final.pdf`

## Page budget

- PDF pages: 31. Main text (through the Conclusion and Limitations section) ends on page 9; the Reproducibility, Ethics and AI Use statements start on page 10 (ICLR line 487); the references start on page 10 (line 506).
- Appendix: Proofs on page 16, tables from page 16 to page 31 (16 pages including the proofs).
- Cut order applied (brief §5): s55, s6, table, s2 = S5.5 to three sentences, S6 three paragraphs, model table in the appendix, S2 to its first six sentences. Figures were not resized.
- Beyond the cut order (the four steps freed about 25 of the 72 lines the first compile was over), the page was reached by trimming the non-verbatim prose sentence by sentence without dropping a claim (every kept claim is in `V3_OUTLINE.md`; the removed sentences restated a claim made in the same or a neighbouring paragraph) and by typographic spacing declared in the preamble: section/subsection/paragraph heading skips 1.2/1.0/0.5 ex (style: 2.0/1.8/1.5 ex), display skips 4 pt, definition and proposition environments 3 pt above and below, float separation 12 pt and caption skip 5 pt. Fonts, margins, line spacing and figure sizes are the style's and the brief's. Everything is in `rebuttal/scripts/phaseE_submission.py` and reverts by deleting those lines.

## Sentence length per section (compiled text of the final file; verbatim sections included in the statistics, the rules are enforced on the non-verbatim prose)

| section | sentences | mean words | longest | over 30 |
|---|---|---|---|---|
| Introduction | 35 | 21.8 | 50 | 0 |
| Related Work | 7 | 28.6 | 60 | 0 |
| Methodology | 37 | 18.1 | 30 | 0 |
| Experimental Setup | 17 | 17.3 | 30 | 0 |
| Results | 83 | 17.4 | 30 | 0 |
| Implications for Hyperbolic Representation Learning | 8 | 16.9 | 24 | 0 |
| Conclusion and Limitations | 21 | 18.8 | 34 | 0 |

Non-verbatim prose of S3–S7: 149 sentences, mean 17.6 words (rule: mean ≤ 22).
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
| Results | MERU's objective does not create hub structure. | 95, 0.29 |
| Results | The naive map manufactures an island. | 12 |
| Results | Controlled, the trees share their topology, not their metric | 0.77, 0.90, 0.33 |
| Results | The self-supervised tree is angular. | — |
| Results | Agreement with WordNet follows supervision and recipe. | — |
| Results | The recovery is not WordNet circularity. | — |
| Results | Text depends on recipe and scale. | 7 of 15 |
| Results | Models share neighborhoods, not metrics. | — |
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

- Total: 230/230 checks passed.
- Final-version checks:

    PASS final: Figure 5 carries the author's caption (consolidated pass); the ARI-matrix figure is out of the appendix (cleanup of 2026-09-23: it is cited from nowhere) 
    PASS final: the appendix inputs exactly the tables the main text cites (twelve question tables, robustness in its final form) plus the provenance index, in first-citation order, from appendix_tables/final/ 
    PASS final: xi, ORC/interventions table, appendix figures, null-variant panel and the class-count sweep are gone from the final and nothing refers to them 
    PASS final: every kept appendix table keeps its provenance comments (% prov: lines and % source comments) and the provenance index lists all thirteen 
    PASS final: each final copy in appendix_tables/final/ carries only numbers of its v1 table (the cleanup of 2026-09-23 drops panels and shortens captions; nothing new appears), floats one panel at a time and opens with the short caption 
    PASS final: references.bib is the author's bib of 2026-09-23 merged with the three cited entries it lacked: keys unique, every cited key present, Groger et al. as ICML 2026 with Shuo Wen, no escaped underscores in doi/url (they print a backslash), Gromov protected 
    PASS final: every cross-reference of the final resolves 
    PASS final: Table 1 is the census on the centered Haar record with the short caption (tab_census_final from expR75, twelve rows, same layout as v1's tab_census) 
    PASS final: AI Use Statement with exactly the three declared items and the responsibility sentence; Reproducibility with the anonymized-repository placeholder; Ethics present 
    PASS final: preamble of the frozen v1 plus amsthm and tcolorbox (the boxed statements of 2026-09-24), same class, same packages otherwise 

## Cited bibliography entries and their sources (DBLP key, else DOI, else publisher URL)

| key | source |
|---|---|
| BGE | DOI https://doi.org/10.1145/3626772.3657878 |
| CLIP | DBLP conf/icml/RadfordKHRGASAM21 |
| Gromov1987 | DOI https://doi.org/10.1007/978-1-4613-9586-7_3 |
| Khrulkov_2020_CVPR | DOI https://doi.org/10.1109/CVPR42600.2020.00645 |
| adcock2013treelike | DOI https://doi.org/10.1109/ICDM.2013.77 |
| aggarwal2001surprising | DBLP conf/icdt/AggarwalHK01 |
| alper2024hierarcaps | DOI https://doi.org/10.1007/978-3-031-72943-0_13 |
| ansuini2019intrinsic | DBLP conf/nips/AnsuiniLMZ19 |
| atigh2022hyperbolic | DOI https://doi.org/10.1109/CVPR52688.2022.00441 |
| augreg | DBLP journals/tmlr/SteinerKZWUB22 |
| bdeir2024fully | DBLP conf/iclr/BdeirSL24 |
| benjamini1995controlling | DOI https://doi.org/10.1111/j.2517-6161.1995.tb02031.x |
| beyer1999nearest | DBLP conf/icdt/BeyerGRS99 |
| chami2019hyperbolic | DBLP conf/nips/ChamiYRL19 |
| chen2022fully | DOI https://doi.org/10.18653/v1/2022.acl-long.389 |
| cifar10 | URL https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf |
| cub | URL https://authors.library.caltech.edu/records/cvm3y-5hh21 |
| dbpedia | DBLP journals/semweb/LehmannIJJKMHMK15 |
| deit | URL https://proceedings.mlr.press/v139/touvron21a.html |
| desai2023meru | URL https://proceedings.mlr.press/v202/desai23a.html |
| dinov1 | DOI https://doi.org/10.1109/ICCV48922.2021.00951 |
| dinov2 | URL https://openreview.net/forum?id=a68SUt6zFt |
| dosovitskiy2021an | URL https://openreview.net/forum?id=YicbFdNTTy |
| dtd | DBLP conf/cvpr/CimpoiMKMV14 |
| e5 | DBLP journals/corr/abs-2212-03533 |
| ermolov2022hyperbolic | DBLP conf/cvpr/ErmolovMKSO22 |
| fashion | DBLP journals/corr/abs-1708-07747 |
| fournier2015computing | DOI https://doi.org/10.1016/j.ipl.2015.02.002 |
| ganea2018hyperbolic | URL https://proceedings.neurips.cc/paper_files/paper/2018/hash/dbab2adc8f9d078009ee3fa810bea142-Abstract.html |
| gpt2 | URL https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf |
| groger2026aristotelian | DBLP journals/corr/abs-2602-14486 |
| gte | DBLP journals/corr/abs-2308-03281 |
| gu2019learning | DBLP conf/iclr/GuSGR19 |
| he2026helm | URL https://openreview.net/forum?id=RnbJPkakkm |
| huh2024platonic | DBLP conf/icml/HuhC0I24 |
| imagenet | DBLP conf/cvpr/DengDSLL009 |
| kennedy2013hyperbolicity | DBLP journals/corr/KennedyNS13 |
| koepke2026cave | URL https://arxiv.org/abs/2604.18572 |
| kornblith2019cka | DBLP conf/icml/Kornblith0LH19 |
| miniimagenet | DBLP conf/nips/VinyalsBLKW16 |
| mnist | DBLP journals/pieee/LeCunBBH98 |
| moreira2024hyperbolic | DOI https://doi.org/10.1109/WACV57701.2024.00208 |
| narayan2011curvature | DOI https://doi.org/10.1103/PhysRevE.84.066108 |
| nica2016strong | DOI https://doi.org/10.4171/GGD/372 |
| nickel2017poincare | DBLP conf/nips/NickelK17 |
| olmo | DBLP conf/acl/GroeneveldBWBKT24 |
| pal2025compositional | URL https://openreview.net/forum?id=3i13Gev2hV |
| papyan2020prevalence | DOI https://doi.org/10.1073/pnas.2015509117 |
| park2024geometry | DBLP conf/iclr/0001CJV25 |
| pope2021intrinsic | DBLP conf/iclr/PopeZAGG21 |
| pythia | DBLP conf/icml/BidermanSABOHKP23 |
| resnet | DOI https://doi.org/10.1109/CVPR.2016.90 |
| sala2018representation | URL https://proceedings.mlr.press/v80/sala18a.html |
| sarkar2011low | DBLP conf/gd/Sarkar11 |
| siglip | DOI https://doi.org/10.1109/ICCV51070.2023.01100 |
| sinha2024learning | DOI https://doi.org/10.52202/079017-2895 |
| tifrea2019poincare | DBLP conf/iclr/TifreaBG19 |
| wordnet | DOI https://doi.org/10.1145/219717.219748 |
| yang2025hyperbolic | URL https://openreview.net/forum?id=TkEdQv0bXB |

59 entries cited; without a source: 0.
