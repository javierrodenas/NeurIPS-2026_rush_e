# V2_DIFF — the parallel version (`main_iclr2027_v2.tex`) against the frozen submission file

Both files are generated from templates with the same fills (`rebuttal/results/phaseE_fills.json`): v1 from
`rebuttal/scripts/phaseE_paper.tex.tmpl` (`phaseE_final.py`), v2 from `phaseE_paper_v2.tex.tmpl` (`phaseE_v2.py`). v2 reuses
the v1 preamble (plus `amsthm`, `tcolorbox`, `caption`), the same figures, the same `tab_census.tex`, the same appendix tables and
bibliography, and copies the statements and the appendix verbatim from the v1 template. `main_iclr2027.tex` is untouched.

**Compile:** v2 = 41 pages, 0 warnings; main text (through §8 Limitations) ends on page 12, statements on pages 12–13, references
from page 13; v1 = 38 pages, main text on page 9. Rasterizations in `qa_pages_v2/` (pages 1–13). Sweep: v1 checks 160/160,
v2 checks 10/10 (`SWEEP_REPORT_v1_v2.md`).

## Section by section

| v2 | What it holds | Moved from v1 | Became a definition / equation / box | Deleted as redundant |
|---|---|---|---|---|
| 1 Introduction | v1 §1 verbatim, Figure 1 | — | — | — |
| 2 Related Work | v1 §2 "The premise" and "Calibrating geometry against nulls" as one paragraph | v1 §2 (two run-in paragraphs) | — | the two run-in titles |
| 3 Background and Problem Setup | Hypothesis box; 3.1 Gromov δ; 3.2 Estimation; 3.3 Objects and notation | v1 §2 "Background" → 3.1; v1 §3 "The premise lives at the sample level…" → 3.3 and 3.2; v1 §3 "Two choices changed" (supremum, budget-stable) → 3.2 | **Hypothesis under test** box (the premise, two lines); **Definition 1** (four-point defect), **Eq. 1**; **Definition 2** (reading of record), **Eq. 2**; notation table (unnumbered, so Table 1 stays the census) | the sentence "The raw reading is the 99.9th-percentile four-point statistic…" (Eq. 2 says it); "so the percentile is the raw reading" clause |
| 4 The Instrument | opening as v1 §3; Figure 2; 4.1 null; 4.2 excess and rank; 4.3 depth test; 4.4 projection; "Definitions at a glance" box | v1 §3 "A low raw reading is not evidence" → 4.1; "The reading is an excess over a matched null" → 4.1/4.2; "Two choices changed" (Gaussian vs Haar, rank-based calibration) → 4.1/4.2; v1 §4 "Depth is certified…" first two sentences (matched star, hub null keeps the clusters) → Definitions 6–7; v1 §4 "conservative and weak" verdict sentence → 4.3 (regime as measured, in words) | **Definition 3** (spectrum-matched null, X′ = QΣVᵀ); **Definition 4** (excess), **Eq. 3**; **Eq. 4** (rank and p, K = 200); **Definition 5** (genuine), **Eq. 5** (BH threshold); **Definition 6** (hub-randomizing null and excess B); **Definition 7** (matched star and depth), **Eq. 6** (depth, z, z ≤ −2); glance box with the seven terms and their equation numbers | "For every cell we synthesize null clouds with the same n, d, mean and covariance spectrum. We use the Haar construction throughout…" (Definition 3); "The excess is the raw reading minus the mean reading of 200 such replicates." (Eq. 3); "The primary evidence is the percentile rank…" (Eq. 4); "A cell is genuine when its BH-corrected p … at most five per cent" (Definition 5); "The depth test compares the real centroids with a matched star… Because the null keeps the clusters, the test certifies hierarchy above the labelled clusters only." (Definitions 6–7) |
| 5 Experimental Setup | Models and datasets; Extraction; Replicates, seeds and budgets; bridge "With the instrument in place…" | v1 §3 (cells, centroids, replicates, seeds, budget), appendix A.14 (extraction protocol), v1 §4.1 (sample-level protocol), v1 §5 (cosine census protocol); model panel by pointer to Appendix A.14 (the compact table is Table 15, shared) | — | — (new connective prose only: names of the families and controls, no numbers beyond v1's) |
| 6 Results | opening as v1 §4; 6.1 sample level; 6.2 class level (Table 1, Figure 3); 6.3 depth (Figure 4, star caveat, both stars, conservative/weak, MERU, neural collapse, scale); 6.4 whose tree (Figure 5, island, sharing, angles, WordNet+DBpedia, circularity, text); 6.5 local agreement | v1 §4.1 → 6.1; §4.2–4.3 → 6.2; §4.4–4.9 → 6.3; §5.1–5.6 → 6.4; §5.7 → 6.5 | **Finding 1–4** boxes, two to three lines each, no numerals | in "Depth is certified…": the two definitional sentences now in Definitions 6–7 |
| 7 Implications for hyperbolic representation learning | v1 §6 verbatim | — | **Eq. 7** (the Khrulkov curvature rule), evaluated on the Gaussian band of Table 2 in the sentence that follows | the inline formula sentence (Eq. 7 says it) |
| 8 Discussion and Limitations | v1 §7 verbatim | — | — | — |

## Bridges, numbers, vocabulary

- Bridges stay at three: end of §5 ("With the instrument in place, we read the premise where it is read."), end of 6.3 ("…is the
  question we turn to next."), 6.5 ("What a practitioner can collect from this local agreement is the subject of the next section.").
- Number groups in the prose of §1–§8: 17, within the limit of 25 (the sweep counts them); ≤ 2 per paragraph, none in
  parentheses; §6–§7 keep the results rules (≤ 1 per sentence, no semicolons, ≤ 9 sentences per paragraph). The finding boxes and
  the hypothesis box carry no numerals. Equations and the notation table are formulas and are excluded from the count.
- Every number of v2 appears in v1 (prose, tables or captions), and every provenance comment (`% file.csv`) of v2 exists in v1.
- Fixed vocabulary unchanged; "reading of record" is the name of Definition 2 as the brief specifies.

## Page budget: 12 pages of main text against 9 in v1 (about 2.5 pages over 9, 1.5 over the 10 the brief allowed)

Where the extra pages come from: the seven shaded definitions and equations (~0.9 page), the hypothesis, findings and glance boxes
(~0.6 page), the notation table and §5 Experimental Setup (~0.8 page), and one layout gap on page 3 (the unbreakable notation
table moved whole to page 4, leaving eight empty lines). Candidates to cut, nothing cut, the author decides:

1. §6.5 Local agreement (3 sentences, ~6 lines) → fold its one claim into Finding 4 and point to Table 12.
2. The second half of §7 ("Measure before imposing" and "A zero-cost readout collects what is there", ~12 lines) → keep the first
   two paragraphs (curvature rule, calibrated prediction) and send the readout paragraph to Appendix A.12.
3. §5 → half a page by dropping "Extraction" (already in Appendix A.14) and the cosine-protocol sentence (in Appendix A.2): ~9 lines.
4. The "Definitions at a glance" box (~12 lines) duplicates the definitions on the facing page; the notation table already carries
   the equation numbers.
5. Layout only: let the notation table break across pages (a `longtable`) or move it before the prose of 3.3 to close the page-3 gap
   (~8 lines).

Items 1–4 together recover about 1.5 pages; with 5 the file lands on 10 pages of main text.
