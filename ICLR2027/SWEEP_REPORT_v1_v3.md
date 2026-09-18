# Sweep report — v1 (`main_iclr2027.tex`) and v3 (`main_iclr2027_v3.tex`)

`rebuttal/scripts/sweep_freeze.py`, run on 2026-09-18: **183/183** checks pass (no failures).

v1 keeps its 160 checks and v2 its 10; the v3 block adds:

- v3: abstract, S1 (with Figure 1) and S2 verbatim from main_local.tex 
- v3: the 'Gromov delta' and 'Estimation and normalization' paragraphs verbatim from main_local.tex 
- v3: no number appears in v3 that is not in v1 or in a fill traced to a result file 
- v3: every result file named in a provenance comment exists 
- v3: no boxes, no colored text, definitions inline (amsthm) 
- v3: thesis verbatim exactly twice (abstract's last sentence, S7 conclusion), no short form 
- v3: skeleton of seven numbered sections and eleven subsections, seven definitions, one lemma with a corollary and a remark, seven numbered equations, two proofs in the appendix 
- v3: the prose rules hold (S5: <=2 number groups per paragraph and <=1 per sentence; S1/S7: only 49/72, 18/24, 4/12 outside the verbatim text; <=1 parenthetical per non-verbatim paragraph; no banned words) 
- v3: bridges only at section ends (every bridge in the last paragraph of its section, at most three, verbatim paragraphs excepted) 
- v3: every cross-reference resolves 
- v3: same figures and the same main-text table as v1; Figure 1 slot 1.4 in 
- v3: the fourteen appendix tables are all input, numbered in the order v3 first cites them, and each is cited from the main text or another table 
- v3: claim fills recomputed (mKNN/CKA calibrated means, text genuine count and GPT-2 S p, joint-sensitivity range, power at s=1, star depth) 

Every number in v3 is a fill read from a result file (`rebuttal/results/phaseE_v3_fills.json`) or a token that v1 already carries; the abstract, S1, S2 and the two methodology paragraphs are checked verbatim against `main_local.tex`.
