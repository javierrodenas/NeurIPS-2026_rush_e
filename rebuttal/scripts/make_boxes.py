#!/usr/bin/env python3
"""main_iclr2027_boxes.tex: the submission with every Definition in a light-blue box and Proposition 1 in a light-amber box
(author's brief, 2026-09-23). The submission file is not touched: this writes a separate file next to it, to be compiled on its own.
Boxes: tcolorbox, thin left rule, no frame, 3pt padding, the same font; the title ("Definition k (name)", "Proposition 1") is the
theorem header of amsthm, bold, inside the box.
Usage (repo root): python rebuttal/scripts/make_boxes.py [<tectonic> <build dir>]"""
import re, sys, subprocess, shutil, os
from pathlib import Path
TEX = Path("ICLR2027/iclr2027"); SRC = TEX / "main_iclr2027_final.tex"; DST = TEX / "main_iclr2027_boxes.tex"
t = SRC.read_text()
PRE = r"""
% ---- boxed statements (author's brief, 2026-09-23): definitions in light blue, Proposition 1 in light amber; the submission file
% itself is unchanged. Thin left rule, no frame, 3pt padding, same font as the body.
\usepackage[most]{tcolorbox}
\tcbset{statementbox/.style={enhanced, breakable, boxrule=0pt, frame hidden, arc=0pt, outer arc=0pt, sharp corners,
  left=2pt, right=2pt, top=2pt, bottom=2pt, boxsep=0pt, borderline west={1.2pt}{0pt}{#1}, before skip=2pt, after skip=2pt,
  before upper={\setlength{\topsep}{0pt}\setlength{\partopsep}{0pt}\setlength{\parskip}{0pt}}}}
\newtcolorbox{defbox}{statementbox=blue!55!black, colback=blue!4}
\newtcolorbox{propbox}{statementbox=orange!70!black, colback=orange!8}
"""
anchor = "\\begin{document}"
assert t.count(anchor) == 1, "one \\begin{document} expected"
t = t.replace(anchor, PRE.strip("\n") + "\n\n" + anchor)
nd = t.count("\\begin{definition}"); npr = t.count("\\begin{proposition}")
t = t.replace("\\begin{definition}", "\\begin{defbox}\n\\begin{definition}").replace("\\end{definition}", "\\end{definition}\n\\end{defbox}")
t = t.replace("\\begin{proposition}", "\\begin{propbox}\n\\begin{proposition}").replace("\\end{proposition}", "\\end{proposition}\n\\end{propbox}")
# ---- S5.5 to the appendix, with a one-sentence pointer at the end of S5.4 (author's brief, 2026-09-23)
i5 = t.index("\\subsection{What is shared is local}"); j5 = t.index("\\section{", i5)
sub = t[i5:j5].rstrip() + "\n"
body = re.search(r"\\paragraph\{[^}]*\}(.*)", sub, re.S).group(0).strip()
t = t[:i5] + t[j5:]
PTR = "\n\\paragraph{Models share neighborhoods, not metrics.} Permutation-calibrated agreement across models survives, and Appendix~\\ref{app:local} gives the reading. % exp21b_local_global_K200.csv\n"
k4 = t.index("\\section{Implications for Hyperbolic Representation Learning}")
t = t[:k4] + PTR.lstrip("\n") + "\n" + t[k4:]
anchor_app = "\\input{appendix_tables/final/tab_q13_local}"
assert t.count(anchor_app) == 1, "the local-agreement table is inputted once"
t = t.replace(anchor_app, body + "\n" + anchor_app)
DST.write_text(t); print(f"{DST} written: {nd} definitions boxed in blue, {npr} proposition(s) in amber; S5.5 moved to the appendix with a pointer in S5.4")
if len(sys.argv) >= 3:
    tectonic, build = sys.argv[1], Path(sys.argv[2]); shutil.rmtree(build, ignore_errors=True); shutil.copytree(TEX, build)
    r = subprocess.run([tectonic, "-X", "compile", "--keep-logs", "-Z", "shell-escape", str(build / DST.name)], capture_output=True, text=True)
    pdf = build / (DST.stem + ".pdf")
    if not pdf.exists():
        print("compile failed:"); print("\n".join(l for l in (r.stdout + r.stderr).split("\n") if "error" in l.lower())[:1500]); sys.exit(1)
    txt = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
    pages = txt.count("\f") + 1
    def where(key):
        i = txt.find(key)
        return (txt[:i].count("\f") + 1) if i >= 0 else None
    print(f"pages {pages} | Conclusion p {where('C ONCLUSION')} | Reproducibility statement p {where('R EPRODUCIBILITY')} | References p {where('R EFERENCES')}")
    out = Path("ICLR2027/main_iclr2027_boxes.pdf"); shutil.copy(pdf, out); print("pdf ->", out)
