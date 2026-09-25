# Sweep report — v1 (`main_iclr2027.tex`) and v2 (`main_iclr2027_v2.tex`)

`rebuttal/scripts/sweep_freeze.py`, run on 2026-09-18: **170/170** checks pass (no failures).

v1 keeps its 160 checks (data, tool, prose, final pass). The v2 block adds the following checks on the parallel file:

- v2: abstract identical to v1; statements, bibliography and appendix identical to v1 
- v2: no number appears in v2 that is not in v1 (prose, boxes, equations, captions) 
- v2: every provenance comment of v2 exists in v1, so every number traces to the same result file 
- v2: same figures and tables as v1 (identical \includegraphics and \input sets) 
- v2: thesis verbatim exactly twice (abstract, S8), no short form 
- v2: skeleton of eight numbered sections and twelve subsections, seven definitions, seven numbered equations, the hypothesis box, four finding boxes and the definitions-at-a-glance box 
- v2: the v1 prose rules hold on the v2 skeleton (<=25 number groups in S1-S8, <=2 per paragraph, none in parentheses, results rules in S6-S7, only 49/72, 18/24, 4/12 in S1 and S8, <=1 parenthetical, no banned words) 
- v2: exactly three bridges (end of S5, end of the depth subsection, S6.5), each in the last paragraph of its unit, distinct wording 
- v2: every cross-reference resolves 
- v2: the shared appendix tables are still numbered in the order v2 first cites them 

Both files read every number from the same result files; v2 introduces no number that v1 does not carry and no provenance comment that v1 does not have.
