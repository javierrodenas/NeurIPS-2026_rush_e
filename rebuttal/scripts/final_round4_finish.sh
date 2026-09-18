#!/usr/bin/env bash
# Fourth-review round (2026-09-18): once the expR73 shards are done, merge them and rebuild the final version end to end.
# Usage (repo root): bash rebuttal/scripts/final_round4_finish.sh <tectonic binary> <build dir>
set -euo pipefail
ROOT=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e; TECTONIC=${1:?tectonic}; BUILD=${2:?build dir}
cd "$ROOT"
python3 rebuttal/scripts/expR73_transfer_bootstrap_record.py --merge | tail -n 3
python3 - <<'EOF'
import pandas as pd
s = pd.read_csv('rebuttal/results/expR73_transfer_bootstrap_record_summary.csv'); assert len(s) == 24 and (s.n_boot == 30).all(), s[['model','dataset','n_boot']]
print(s.groupby('dataset')[['excess_boot_sd','frac_boot_negative']].agg(['max','min']).round(4).to_string())
EOF
(cd ICLR2027/iclr2027 && python3 gen_appendix_final.py | tail -n 1 && python3 gen_main_table.py | tail -n 1)
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
python3 rebuttal/scripts/sweep_freeze.py > rebuttal/results/sweep_final.log 2>&1 || true
grep -n "FAIL\|phaseB re-total" rebuttal/results/sweep_final.log | cut -c1-200 | tail -n 5
cp "$BUILD/main_iclr2027_final.pdf" ICLR2027/main_iclr2027_final.pdf
rm -rf ICLR2027/qa_pages_final && mkdir -p ICLR2027/qa_pages_final && pdftoppm -r 100 -png ICLR2027/main_iclr2027_final.pdf ICLR2027/qa_pages_final/p && echo "qa pages: $(ls ICLR2027/qa_pages_final | wc -l)"
python3 rebuttal/scripts/make_final_check.py ICLR2027/main_iclr2027_final.pdf rebuttal/results/sweep_final.log | cut -c1-120
