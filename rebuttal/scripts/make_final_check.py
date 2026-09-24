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
p_concl, _ = where("C ONCLUSION AND"); p_stmt, s_stmt = where("R EPRODUCIBILITY S TATEMENT"); p_ref, s_ref = where("R EFERENCES"); p_app, _ = where("A DDITIONAL RESULTS"); p_proof, _ = where("P ROOFS")
# the last numbered line of the main text = the slot before the Reproducibility Statement
p_main_end = (s_stmt - 2) // 54 + 1 if s_stmt else None
S = json.load(open("rebuttal/results/final_prose_stats.json")); A = json.load(open("rebuttal/results/final_appendix.json"))
L = open(log).read(); fin = [l for l in L.splitlines() if "final:" in l and (l.startswith("PASS") or l.startswith("FAIL"))]; tot = re.findall(r"\[phaseB re-total\] (\d+)/(\d+)", L)
out = ["# FINAL_CHECK — `main_iclr2027_final.pdf`", "",
       "## Page budget", "",
       f"- PDF pages: {pages}. Main text (through the Conclusion and Limitations section) ends on page {p_main_end}; the Reproducibility, Ethics and AI Use statements start on page {p_stmt} (ICLR line {s_stmt}); the references start on page {p_ref} (line {s_ref}).",
       f"- Appendix: Proofs on page {p_proof}, implementation and additional results from page {p_app} to page {pages} ({pages - p_proof + 1} pages including the proofs).",
       f"- Cut order applied (brief §5): {', '.join(A['cuts']) if A['cuts'] else 'none'} = S5.5 in the appendix behind a one-sentence pointer, S6 in one paragraph since 2026-09-24, model table in the appendix, S2 to its first six sentences. Figures were not resized.",
       "- Main text without tables (brief of 2026-09-24): the census table and the published-reading table are inputted in the appendix, each beside the question it answers, and the published-reading table keeps every column including the supremum. Their place in Section 5.1 is taken by a figure, `fig_premise_final`, with the 24 sample-level cells and the four published datasets.",
       "- Template compliance (brief of 2026-09-24): the preamble no longer overrides the style's spacing. The display-skip block, `\\textfloatsep`, `\\abovecaptionskip`, `\\parskip` and `\\floatsep` are gone and the main text no longer sets `\\raggedbottom`, so the style's `\\parskip .5pc`, `\\parindent 0pt` and `\\flushbottom` apply. Page 9 was recovered by moving content to the appendix in the author's order: the limitations (four kept in S7, the ten in Appendix B), S6 to one paragraph, the WordNet, DBpedia, HierarCaps and text readings of S5.4 to one sentence each, the frames, leaf-label and MERU paragraphs of S5.3 to one sentence each with a pointer, and the class-set and neural-collapse paragraphs of S5.2 to Appendix C. Every moved sentence is verbatim in the appendix and figures were not resized.",
       "- What still departs from the style, all of it typographic and reversible in `rebuttal/scripts/phaseE_submission.py`: the section, subsection and paragraph heading skips (0.3/0.3/0 ex before and 0.2/0.2/-1 em after, against the style's 2.0/1.8/1.5 ex and 1.5/0.8/-1 em); `\\topsep`, `\\partopsep` and `\\parskip` zeroed inside the definition and proposition boxes only; the author's `\\vspace{2pt}` between Figure 1 and its caption; and, after `\\appendix`, `\\raggedbottom`, the float-placement fractions and counters and `\\arraystretch 0.92`.",
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
# ---- cited bibliography entries and their sources (final accuracy pass, 2026-09-23): DBLP key from the biburl, else the DOI, else the publisher URL
import glob
bib = open("ICLR2027/iclr2027/references.bib").read(); tex = open("ICLR2027/iclr2027/main_iclr2027_final.tex").read() + "".join(open(f).read() for f in sorted(glob.glob("ICLR2027/iclr2027/appendix_tables/final/*.tex")) + sorted(glob.glob("ICLR2027/iclr2027/tab_*.tex")))   # the appendix tables are \input files
cited = sorted({k.strip() for mm in re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", tex) for k in mm.group(1).split(",")})
def bib_entry(k):
    mm = re.search(r"@\w+\s*\{\s*" + re.escape(k) + r"\s*,", bib); st = mm.start(); d = 0; j = st + len(mm.group(0).split("{")[0])
    while True:
        if bib[j] == "{": d += 1
        elif bib[j] == "}":
            d -= 1
            if d == 0: return bib[st:j + 1]
        j += 1
def source(e):
    f = lambda name: (re.search(r"\n\s*" + name + r"\s*=\s*[{\"]([^}\"]*)[}\"]", e) or [None, None])[1]
    bu = f("biburl")
    if bu and "dblp.org/rec/" in bu: return "DBLP " + bu.split("dblp.org/rec/")[1].replace(".bib", "")
    if f("doi"): return "DOI https://doi.org/" + f("doi")
    if f("url"): return "URL " + f("url")
    if f("eprint"): return "arXiv " + f("eprint")
    return "NO SOURCE"
out += ["", "## Cited bibliography entries and their sources (DBLP key, else DOI, else publisher URL)", "", "| key | source |", "|---|---|"]
for k in cited: out.append(f"| {k} | {source(bib_entry(k))} |")
out += ["", f"{len(cited)} entries cited; without a source: {sum(1 for k in cited if source(bib_entry(k)) == 'NO SOURCE')}."]
open("ICLR2027/FINAL_CHECK.md", "w").write("\n".join(out) + "\n"); print("FINAL_CHECK.md written; main text ends p", p_main_end, "; statements p", p_stmt, "; references p", p_ref, "; pages", pages)
