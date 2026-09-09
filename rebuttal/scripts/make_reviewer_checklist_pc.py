#!/usr/bin/env python3
"""REVIEWER_CHECKLIST_positive_control.md: the three questions of the second review answered from the main text alone,
with the ICLR margin line numbers of the sentences that answer them, read from the compiled PDF (pdftotext -layout keeps
the margin numbers). Usage: python make_reviewer_checklist_pc.py <path/to/main_iclr2027.pdf>"""
import re, subprocess, sys, json
pdf = sys.argv[1]
txt = subprocess.run(["pdftotext", "-layout", "-l", "9", pdf, "-"], capture_output=True, text=True).stdout
# map every text line to the nearest margin number (ICLR numbers every line; pdftotext sometimes drops the number on wrapped lines)
lines = txt.split("\n"); numbered = []; last = None
for l in lines:
    m = re.match(r"\s*(\d{3})\s+(.*)", l)
    if m: last = int(m.group(1)); numbered.append((last, m.group(2)))
    elif l.strip():
        if re.fullmatch(r"\s*\d{3}\s*", l): last = int(l.strip()); continue
        numbered.append((last, l.strip()))
flat = " ".join(t for _, t in numbered)
def find(phrase):
    """line number of the first numbered line whose text (joined with the next) contains the phrase (ligature- and hyphen-tolerant)."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace("ﬁ", "fi").replace("ﬂ", "fl"))
    target = norm(phrase)
    for i in range(len(numbered)):
        window = norm("".join(t for _, t in numbered[i:i + 3]))
        if target in window: return numbered[i][0]
    return None
M = json.load(open("rebuttal/results/positive_control_memo.json"))
r9b = M["r9b"]; ndet = r9b["ndet_s1"]; lo = min(r9b["ratio_real"].values()); hi = max(r9b["ratio_real"].values())
J = M["r11_joint"]; B = M["r11_bootbh"]
Q = [("Q1. The depth test has no positive control on real data; how should the reader interpret 4 of 12?",
      [("Section 4, 'On real clouds the test is conservative and weak'", "We measured the test on the real ImageNet clouds themselves", f"On the real clouds with the hub arrangement replaced by an implanted two-level tree the test raises no false alarm (none of 60 zero-strength runs) and detects a clean tree in {ndet} of 12 backbones; the rest are missed because the within-cluster spread of the real clouds is {lo:.1f}–{hi:.1f} times their between-hub spread while the synthetic sweep covered ratios below one, and shrinking the offsets into that range makes every backbone detectable. The test is conservative and weak at ImageNet's noise level: its positive verdicts stand, 4 of 12 is a lower bound, and the other eight are 'not detected', never 'not hierarchical'."),
       ("Section 4, 'Depth is certified on ImageNet and not on CIFAR-100'", "The other eight are not detected", "The certified set (ViT-S/B/L, DINOv2-L) is unchanged; the wording for the rest is 'not detected'."),
       ("Figure 4, panel (b)", "implanted two-level tree of strength", "z against implant strength for every backbone, family colors; dashed: the same clouds with their within-cluster spread shrunk into the validated range."),
       ("Limitations (i)", "conservative and weak at ImageNet", "The limitation states the weakness and that the one trained positive control is inconclusive.")]),
     ("Q2. Does the test's weakness change the abstract's claim about MERU and about depth?",
      [("Abstract", "no detected depth", "'No additional depth' became 'no detected depth' everywhere: MERU shows the same clustering and no detected depth; the test is described as conservative and weak."),
       ("Section 4, 'Imposing the geometry does not create depth'", "conservative and weak at this noise level", "Neither MERU nor its twin is detected as more hierarchical than a matched star; training in hyperbolic space neither creates the clustering nor adds detectable depth."),
       ("Section 6, 'Measure before imposing'", "a positive verdict certifies and a negative one says nothing", "The practical reading of the test's asymmetry.")]),
     ("Q3. How many of the 49 genuine cells survive when centroid-resampling and estimator noise are combined with the null?",
      [("Section 4, 'The count survives resampling'", "The count survives resampling", f"Folding the bootstrap s.d. and the estimator s.d. into the null, or repeating the census on 30 resampled centroid sets, leaves {min(J[0],B[0])}–{max(J[0],B[0])} of the 72 cells genuine and {min(J[1],B[1])}–{max(J[1],B[1])} of the 24 on ImageNet and CIFAR-100; the cells that move are on the transfer sets."),
       ("Table 1 caption", "With centroid-resampling and estimator noise folded", "The joint count in one clause, with the appendix table pointer.")])]
out = ["# Reviewer checklist — positive-control pass", "", "The three questions of the second review, each answered from the main text alone; line numbers are the ICLR margin numbers of the compiled PDF.", ""]
missing = []
for q, items in Q:
    out.append(f"## {q}"); out.append("")
    for where, phrase, answer in items:
        ln = find(phrase)
        if ln is None: missing.append(phrase)
        out.append(f"- **{where}** (line {ln if ln else '?'}): {answer}")
    out.append("")
open("ICLR2027/REVIEWER_CHECKLIST_positive_control.md", "w").write("\n".join(out)); print("\n".join(out)); print("missing anchors:", missing)
