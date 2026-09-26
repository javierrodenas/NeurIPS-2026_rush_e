#!/usr/bin/env bash
# Rebuild the final version end to end (tables, Table 1, figures, body, provenance index, compile, sweep, PDF export, QA pages,
# FINAL_CHECK.md). Run the experiment merges first (expR73/expR74/expR75 --merge). Usage (repo root):
#   bash rebuttal/scripts/final_rebuild.sh <tectonic binary> <build dir>
set -euo pipefail
ROOT=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e; TECTONIC=${1:?tectonic}; BUILD=${2:?build dir}
cd "$ROOT"
(cd ICLR2027/iclr2027 && python3 gen_appendix_final.py | tail -n 1 && python3 gen_main_table.py | tail -n 1 && CENSUS_SRC=expR75_census_centered_haar.csv FINAL_ONLY=1 python3 gen_main_table.py | tail -n 1)
(cd ICLR2027/figures && python3 make_figs_final.py | grep -c written)
FINAL_CUTS=s55,s6,table,s2 python3 rebuttal/scripts/phaseE_submission.py | tail -n 1 | cut -c1-80
(cd ICLR2027/iclr2027 && python3 gen_provenance.py main_iclr2027_final.tex tab_z_provenance_final.tex final | tail -n 1)
rm -rf "$BUILD" && cp -r ICLR2027/iclr2027 "$BUILD"
(cd "$BUILD" && "$TECTONIC" -X compile main_iclr2027_final.tex 2>&1 | grep -E "error|Overfull|Underfull|undefined|multiply|Warning" | sort | uniq -c || true)
pdfinfo "$BUILD/main_iclr2027_final.pdf" | grep Pages
pdftotext -layout "$BUILD/main_iclr2027_final.pdf" "$BUILD/final.txt"; echo "?? count: $(grep -c '??' "$BUILD/final.txt" || true)"
python3 - "$BUILD/final.txt" <<'EOF'
import re, sys
txt = open(sys.argv[1]).read()
for key in ('C ONCLUSION', 'R EPRODUCIBILITY', 'R EFERENCES', 'P ROOFS'):
    i = txt.index(key); pre = txt[:i]; nums = [int(m.group(1)) for m in re.finditer(r'^\s*(\d{3})\s', pre, flags=re.M)]
    print(f"{key:18s} slot {max(nums)+1} page {pre.count(chr(12))+1}")
EOF
cp "$BUILD/main_iclr2027_final.pdf" ICLR2027/main_iclr2027_final.pdf   # before the sweep, which now reads the PDF for the page-9 check (2026-09-26)
python3 rebuttal/scripts/make_supp_readme.py
python3 rebuttal/scripts/make_supplement.py   # before the sweep, which reads the zip (final audit, 2026-09-26)
python3 rebuttal/scripts/sweep_freeze.py > rebuttal/results/sweep_final.log 2>&1 || true
grep -n "FAIL\|phaseB re-total" rebuttal/results/sweep_final.log | cut -c1-200 | tail -n 5
python3 rebuttal/scripts/make_submission.py "$TECTONIC"
rm -rf ICLR2027/qa_pages_final && mkdir -p ICLR2027/qa_pages_final && pdftoppm -r 100 -png ICLR2027/main_iclr2027_final.pdf ICLR2027/qa_pages_final/p && echo "qa pages: $(ls ICLR2027/qa_pages_final | wc -l)"
python3 rebuttal/scripts/make_final_check.py ICLR2027/main_iclr2027_final.pdf rebuttal/results/sweep_final.log | cut -c1-120
