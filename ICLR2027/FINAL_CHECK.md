# FINAL_CHECK — `main_iclr2027_final.pdf`

## Page budget

- PDF pages: 29. Main text (through the Conclusion and Limitations section) ends on page 9; the Reproducibility, Ethics and AI Use statements start on page 10 (ICLR line 487); the references start on page 10 (line 508).
- Appendix: Proofs on page 16, implementation and additional results from page 17 to page 29 (14 pages including the proofs).
- Cut order applied (brief §5): s55, s6, table, s2 = S5.5 in the appendix behind a one-sentence pointer, S6 in one paragraph since 2026-09-24, model table in the appendix, S2 to its first six sentences. Figures were not resized.
- Main text without tables (brief of 2026-09-24): the census table and the published-reading table are inputted in the appendix, each beside the question it answers, and the published-reading table keeps every column including the supremum. Their place in Section 5.1 is taken by a figure, `fig_premise_final`, with the 24 sample-level cells and the four published datasets.
- Template compliance (brief of 2026-09-24): the preamble no longer overrides the style's spacing. The display-skip block, `\textfloatsep`, `\abovecaptionskip`, `\parskip` and `\floatsep` are gone and the main text no longer sets `\raggedbottom`, so the style's `\parskip .5pc`, `\parindent 0pt` and `\flushbottom` apply. Page 9 was recovered by moving content to the appendix in the author's order: the limitations (four kept in S7, the ten in Appendix B), S6 to one paragraph, the WordNet, DBpedia, HierarCaps and text readings of S5.4 to one sentence each, the frames, leaf-label and MERU paragraphs of S5.3 to one sentence each with a pointer, and the class-set and neural-collapse paragraphs of S5.2 to Appendix C. Every moved sentence is verbatim in the appendix and figures were not resized.
- What still departs from the style, all of it typographic and reversible in `rebuttal/scripts/phaseE_submission.py`: the section, subsection and paragraph heading skips (0.3/0.3/0 ex before and 0.2/0.2/-1 em after, against the style's 2.0/1.8/1.5 ex and 1.5/0.8/-1 em); `\topsep`, `\partopsep` and `\parskip` zeroed inside the definition and proposition boxes only; the author's `\vspace{2pt}` between Figure 1 and its caption; and, after `\appendix`, `\raggedbottom`, the float-placement fractions and counters and `\arraystretch 0.92`.
## Sentence length per section (compiled text of the final file; verbatim sections included in the statistics, the rules are enforced on the non-verbatim prose)

| section | sentences | mean words | longest | over 30 |
|---|---|---|---|---|
| Introduction | 36 | 21.8 | 51 | 0 |
| Related Work | 7 | 26.4 | 60 | 0 |
| Methodology | 38 | 18.2 | 30 | 0 |
| Experimental Setup | 16 | 17.7 | 33 | 0 |
| Results | 72 | 17.8 | 30 | 0 |
| Implications for Hyperbolic Representation Learning | 6 | 19.3 | 24 | 0 |
| Conclusion and Limitations | 10 | 21.7 | 34 | 0 |

Non-verbatim prose of S3–S7: 125 sentences, mean 18.3 words (rule: mean ≤ 22).
Rule of 2026-09-23: no sentence over 30 words in S3–S7 (pages 3–9), definitions and citation lists excepted; sentences over the cap outside those: 0; the thesis sentence, verbatim from the abstract, is the one kept exception (34 words).

## Numbers per paragraph in S5–S6 (rule: ≤ 1 per sentence, ≤ 2 per paragraph, headline numbers only)

| section | paragraph | numbers |
|---|---|---|
| Results | The premise does not survive calibration where it is read. | 14 of 24, 24 |
| Results | A published reading is reproduced and calibrated. | 0.03, 4 |
| Results | Most cells show structure beyond the second moments. | 44 of 72, 30 of 36, 3, 47, 18 of 24 |
| Results | The count survives resampling. | — |
| Results | A star of clusters already passes the census. | — |
| Results | The hierarchy test certifies that clusters are oriented towa | 12 |
| Results | What it certifies is alignment, how each cluster is oriented | 4, -1.61, -1.61 to -1.76, 0 of 60, 0 of 4, 4, -2.08 to -3.99 |
| Results | What the test can see. | 4 of 5, 4, 9 of 12, 0.83 to 1.00, 3, 0.01 to 0.60, 3, 1.5 to 3.0, 1.3 to 3.9, 9 |
| Results | A trained hierarchy survives, and the test leans toward firi | 20 of 20, 2, -2.51, 6 of 20, -1.37, 0 of 5, 8 of 50, 4, -4.19 to -4.82, 0 of 50 |
| Results | No hub hierarchy is found where the test has power. | 9 |
| Results | The certified set depends on the grouping. | — |
| Results | Leaf labels can produce the alignment but do not guarantee i | — |
| Results | Training in hyperbolic space leaves the clustering unchanged | 95, 0.29 |
| Results | The naive map manufactures an island. | 12, 7, 7 |
| Results | Controlled, the trees share their topology, not their metric | 0.77, 0.90, 0.33 |
| Results | The self-supervised tree is angular. | — |
| Results | Agreement with WordNet follows supervision and recipe. | +0.57 to +0.59, +0.49 to +0.53, +0.18 to +0.22 |
| Results | Text depends on recipe and scale. | 7 of 15 |
| Implications for Hyperbolic  | The raw reading cannot select a curvature. | 0.48 to 2.5, +0.9 to +1.3 |

## Appendix material deleted from the final (kept in the frozen v1 file and its tables)

- Whole tables: tab_q11_interventions, tab_q12_xi (the parallelogram descriptor ξ; the interventions/ORC table: Ollivier–Ricci curvature by edge type and the intervention rows beyond the two the text cites).
- The null-variant panels: (c) of the robustness table (four null × statistic constructions per cell with the joint listing) and (b) of the census table (verdict under the four constructions); the superseded supremum × Gaussian rows of the budget panel and panel (b) of the corollary table (correlations of the raw supremum reading), both uncited; the class-count sweep beyond one row (DINOv2-L at C = 100 is kept, both modes); the training-intervention rows beyond one per intervention.
- Appendix length: the brief asked for at most 14 pages; the kept material occupies 16 (proofs included) with the tables floating one panel at a time. Candidates if the author wants the last two pages: corollary panels (f) nearest-centroid policies and (g) sensitivity to the projection target (uncited from the main text), the text table's extraction panel (c), the model-panel table's per-model extraction notes.
- Every appendix figure (depth on CIFAR-100, depth power, causal interventions, CIFAR-100 tree map, text nulls, best-metric scatter) and every reference to them.
- The limitation item on ξ (former (viii)) went with the descriptor.
- Kept, in first-citation order: tab_q10_calibration, tab_q01_census, tab_q14_panel, tab_q03_sample, tab_q08_robust, tab_q04_depth, tab_q05_power, tab_q06_treemap, tab_q13_local, tab_q07_wordnet, tab_q02_text, tab_q09_corollary, plus the provenance index (tab_z_provenance_final).

## Sweep (rebuttal/scripts/sweep_freeze.py)

- Total: 244/245 checks passed.
- Final-version checks:

    PASS final: every cross-reference of the final resolves 
    PASS final: the corollary panel drops the McNemar column (the main text never used it, author's brief 2026-09-24): 9 columns, 60 rows of accuracies and advantages, and a caption that defines Eucl. and Poinc. and gives the advantages in percentage points 
    PASS final: Appendix A is the author's expanded proof (2026-09-24): the proposition restated by reference, an intuition paragraph, the two parts proved with 6 displayed equations, and the labels it cites resolve 
    PASS final: every row of the model panel carries its citation, inline after the model name since it fits the text width (author's brief, 2026-09-24), Barlow Twins and BYOL among them, and both new entries have a publisher source 
    PASS final: the per-cell census is the centered Haar record with the two self-supervised ResNets as rows (author's brief, 2026-09-24), and it answers the reference of the former main-text table 
    PASS final: AI Use Statement with exactly the three declared items and the responsibility sentence; Reproducibility pointing at the supplementary material, with no TODO or anonymized-repository placeholder left in the file; Ethics present 
    PASS final: preamble of the frozen v1 plus amsthm, amssymb (the \square of the proofs, 2026-09-24) and tcolorbox (the boxed statements), same class, same packages otherwise 

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
| grill2020bootstrap | URL https://proceedings.neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html |
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
| zbontar2021barlow | URL https://proceedings.mlr.press/v139/zbontar21a.html |

61 entries cited; without a source: 0.
