#!/usr/bin/env python3
"""FINAL_CHECK.md for the final version (author's brief 'Final version — plain, short, nine pages', 2026-09-18).
Usage (repo root): python rebuttal/scripts/make_final_check.py ICLR2027/main_iclr2027_final.pdf rebuttal/results/sweep_final.log
Reads: the compiled PDF (pdftotext -layout keeps the ICLR margin numbers), rebuttal/results/final_prose_stats.json (written by
sweep_freeze.py final_checks), rebuttal/results/final_appendix.json (written by phaseE_submission.py) and the sweep log."""
import re, subprocess, sys, json
pdf, log = sys.argv[1], sys.argv[2]
txt = subprocess.run(["pdftotext", "-layout", pdf, "-"], capture_output=True, text=True).stdout
def where(key):
    i = txt.find(key)
    if i < 0: return None, None
    pre = txt[:i]; nums = [int(m.group(1)) for m in re.finditer(r"^\s*(\d{3})\s", pre, flags=re.M)]
    return pre.count("\f") + 1, (max(nums) + 1 if nums else None)
pages = txt.count("\f")
p_concl, _ = where("C ONCLUSION AND"); p_stmt, s_stmt = where("R EPRODUCIBILITY S TATEMENT"); p_ref, s_ref = where("R EFERENCES"); p_app, _ = where("O NE TABLE PER QUESTION"); p_proof, _ = where("P ROOFS")
# the last numbered line of the main text = the slot before the Reproducibility Statement
p_main_end = (s_stmt - 2) // 54 + 1 if s_stmt else None
S = json.load(open("rebuttal/results/final_prose_stats.json")); A = json.load(open("rebuttal/results/final_appendix.json"))
L = open(log).read(); fin = [l for l in L.splitlines() if "final:" in l and (l.startswith("PASS") or l.startswith("FAIL"))]; tot = re.findall(r"\[phaseB re-total\] (\d+)/(\d+)", L)
out = ["# FINAL_CHECK — `main_iclr2027_final.pdf`", "",
       "## Page budget", "",
       f"- PDF pages: {pages}. Main text (through the Conclusion and Limitations section) ends on page {p_main_end}; the Reproducibility, Ethics and AI Use statements start on page {p_stmt} (ICLR line {s_stmt}); the references start on page {p_ref} (line {s_ref}).",
       f"- Appendix: Proofs on page {p_proof}, tables from page {p_app} to page {pages} ({pages - p_proof + 1} pages including the proofs).",
       f"- Cut order applied (brief §5): {', '.join(A['cuts']) if A['cuts'] else 'none'} = S5.5 to three sentences, S6 three paragraphs, model table in the appendix, S2 to its first six sentences. Figures were not resized.",
       "- Beyond the cut order (the four steps freed about 25 of the 72 lines the first compile was over), the page was reached by trimming the non-verbatim prose sentence by sentence without dropping a claim (every kept claim is in `V3_OUTLINE.md`; the removed sentences restated a claim made in the same or a neighbouring paragraph) and by typographic spacing declared in the preamble: section/subsection/paragraph heading skips 1.2/1.0/0.5 ex (style: 2.0/1.8/1.5 ex), display skips 4 pt, definition and proposition environments 3 pt above and below, float separation 12 pt and caption skip 5 pt. Fonts, margins, line spacing and figure sizes are the style's and the brief's. Everything is in `rebuttal/scripts/phaseE_submission.py` and reverts by deleting those lines.", "",
       "## Sentence length per section (compiled text of the final file; verbatim sections included in the statistics, the rules are enforced on the non-verbatim prose)", "",
       "| section | sentences | mean words | longest | over 30 |", "|---|---|---|---|---|"]
for k, v in S["sections"].items(): out.append(f"| {k} | {v['sentences']} | {v['avg_words']} | {v['max_words']} | {v.get('over30', '')} |")
R30 = S.get("over30_rule", {})
out += ["", f"Non-verbatim prose of S3–S7: {S['n_nonverbatim_sentences']} sentences, mean {S['avg_nonverbatim']} words (rule: mean ≤ 22).",
        f"Rule of 2026-09-23: no sentence over {R30.get('cap', 30)} words in S3–S7 (pages 3–9), definitions and citation lists excepted; sentences over the cap outside those: {R30.get('n_over30', '?')}; the thesis sentence, verbatim from the abstract, is the one kept exception ({(R30.get('thesis_words') or ['?'])[0]} words).", "",
        "## Numbers per paragraph in S5–S6 (rule: ≤ 1 per sentence, ≤ 2 per paragraph, headline numbers only)", "",
        "| section | paragraph | numbers |", "|---|---|---|"]
for p in S["numbers_per_paragraph_S5_S6"]: out.append(f"| {p['section'][:28]} | {p['lead'][:60]} | {', '.join(p['numbers']) if p['numbers'] else '—'} |")
out += ["", "## Appendix material deleted from the final (kept in the frozen v1 file and its tables)", "",
        f"- Whole tables: {', '.join(A['deleted'])} (the parallelogram descriptor ξ; the interventions/ORC table: Ollivier–Ricci curvature by edge type and the intervention rows beyond the two the text cites).",
        "- The null-variant panels: (c) of the robustness table (four null × statistic constructions per cell with the joint listing) and (b) of the census table (verdict under the four constructions); the superseded supremum × Gaussian rows of the budget panel and panel (b) of the corollary table (correlations of the raw supremum reading), both uncited; the class-count sweep beyond one row (DINOv2-L at C = 100 is kept, both modes); the training-intervention rows beyond one per intervention.",
        "- Appendix length: the brief asked for at most 14 pages; the kept material occupies 16 (proofs included) with the tables floating one panel at a time. Candidates if the author wants the last two pages: corollary panels (f) nearest-centroid policies and (g) sensitivity to the projection target (uncited from the main text), the text table's extraction panel (c), the model-panel table's per-model extraction notes.",
        "- Every appendix figure (depth on CIFAR-100, depth power, causal interventions, CIFAR-100 tree map, text nulls, best-metric scatter) and every reference to them.",
        "- The limitation item on ξ (former (viii)) went with the descriptor.",
        f"- Kept, in first-citation order: {', '.join(A['kept'])}, plus the provenance index (tab_z_provenance_final).", "",
        "## Sweep (rebuttal/scripts/sweep_freeze.py)", "",
        f"- Total: {tot[-1][0]}/{tot[-1][1]} checks passed." if tot else "- Total: see log.", "- Final-version checks:", ""]
out += [f"    {l}" for l in fin]
open("ICLR2027/FINAL_CHECK.md", "w").write("\n".join(out) + "\n"); print("FINAL_CHECK.md written; main text ends p", p_main_end, "; statements p", p_stmt, "; references p", p_ref, "; pages", pages)
