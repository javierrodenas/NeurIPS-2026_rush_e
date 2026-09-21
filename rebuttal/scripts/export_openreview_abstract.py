#!/usr/bin/env python3
"""Write the OpenReview abstract as plain text from the submission file (the abstract on OpenReview is the tex abstract, by the
author's rule). Run from the repo root: python rebuttal/scripts/export_openreview_abstract.py"""
import re
SRC = 'ICLR2027/iclr2027/main_iclr2027_final.tex'; DST = 'ICLR2027/OPENREVIEW_abstract.txt'
t = open(SRC).read(); a = t[t.index('\\begin{abstract}') + len('\\begin{abstract}'):t.index('\\end{abstract}')]
a = re.sub(r'(?m)(?<!\\)%.*$', '', a)                       # provenance comments
a = a.replace('$\\delta$', 'δ').replace('\\delta', 'δ').replace('$', '')
a = re.sub(r'\\emph\{([^}]*)\}', r'\1', a); a = re.sub(r'\\textbf\{([^}]*)\}', r'\1', a); a = a.replace('\\%', '%').replace('~', ' ').replace('---', '—').replace('--', '–')
a = ' '.join(a.split()); assert '\\' not in a and '{' not in a, a
open(DST, 'w').write(a + '\n'); print(f"{DST}: {len(a.split())} words"); print(a)
