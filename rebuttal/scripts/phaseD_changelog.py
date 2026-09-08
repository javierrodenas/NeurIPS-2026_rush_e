#!/usr/bin/env python3
"""Lists, for CHANGELOG §20, every number that left the main-text prose in the prose pass and where it now lives
(generated table / caption label, or appendix prose), by comparing rebuttal/results/phaseD_old_main_body.tex with the
current main text and the generated tables. Prints a Markdown table."""
import re, glob
from pathlib import Path
TEX = Path('ICLR2027/iclr2027'); R = Path('rebuttal/results')
def strip(s):
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s); s = re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", s); s = re.sub(r"\\(S)?\\?ref\{[^}]*\}", "", s); return re.sub(r"\\label\{[^}]*\}", "", s)
old = strip(open(R/'phaseD_old_main_body.tex').read()); T = open(TEX/'main_iclr2027.tex').read()
new_main = strip(T[T.index("\\begin{abstract}"):T.index("\\subsubsection*{Ethics Statement}")]); app = T[T.index("\\appendix"):]
tables = {f: open(f).read() for f in glob.glob(str(TEX/'appendix_tables'/'*.tex')) + [str(TEX/'tab_census.tex')]}
def label(txt):
    m = re.search(r"\\label\{([^}]*)\}", txt); return m.group(1) if m else "?"
NUM = r"[-+]?\d+\.\d+|(?<![\w.])\d+(?![\w.])"
def toks(s): return set(re.findall(r"[-+]?\d+\.\d+", s)) | {t for t in re.findall(r"(?<![\w.\-])\d{2,4}(?![\w.])", s) if not (1900 < int(t) < 2100)}
gone = sorted(toks(old) - toks(new_main), key=lambda t: (len(t), t))
rows = []
for t in gone:
    homes = [label(txt) for f, txt in tables.items() if re.search(rf"(?<![\d.]){re.escape(t.lstrip('+-'))}(?![\d])", txt)]
    in_app = bool(re.search(rf"(?<![\d.]){re.escape(t.lstrip('+-'))}(?![\d])", app))
    # context in the old prose
    m = re.search(rf".{{0,60}}{re.escape(t)}.{{0,40}}", old); ctx = m.group(0).replace("\n", " ") if m else ""
    rows.append((t, ", ".join(sorted(set(homes))[:4]) + (" + appendix prose" if in_app else ""), ctx))
print("| number (old prose) | now lives in | old context |"); print("|---|---|---|")
for t, h, c in rows: print(f"| {t} | {h or 'MISSING'} | …{c.strip()}… |")
print(f"\n{len(rows)} numbers left the prose; {sum(1 for _, h, _ in rows if not h)} without a home")
