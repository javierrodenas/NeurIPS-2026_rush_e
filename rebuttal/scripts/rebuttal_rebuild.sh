#!/usr/bin/env bash
# Parallel track (brief of 2026-09-20): build main_iclr2027_rebuttal.tex from the frozen submission file plus the parallel-track
# paragraphs whose result files exist (phaseE_rebuttal.py), compile it, export ICLR2027/main_iclr2027_rebuttal.pdf and run the sweep
# (rebuttal_checks: the frozen text and numbers unchanged, every new number re-derived from its CSV).
# Usage (repo root): bash rebuttal/scripts/rebuttal_rebuild.sh <tectonic binary> <build dir>
set -euo pipefail
ROOT=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e; TECTONIC=${1:?tectonic}; BUILD=${2:?build dir}
cd "$ROOT"
python3 rebuttal/scripts/phaseE_rebuttal.py | cut -c1-200
rm -rf "$BUILD" && cp -r ICLR2027/iclr2027 "$BUILD"
(cd "$BUILD" && "$TECTONIC" -X compile main_iclr2027_rebuttal.tex 2>&1 | grep -E "error|undefined|multiply" | sort | uniq -c || true)
pdfinfo "$BUILD/main_iclr2027_rebuttal.pdf" | grep Pages
pdftotext -layout "$BUILD/main_iclr2027_rebuttal.pdf" "$BUILD/rebuttal.txt"; echo "?? count: $(grep -c '??' "$BUILD/rebuttal.txt" || true)"
cp "$BUILD/main_iclr2027_rebuttal.pdf" ICLR2027/main_iclr2027_rebuttal.pdf
python3 rebuttal/scripts/sweep_freeze.py > rebuttal/results/sweep_rebuttal.log 2>&1 || true
grep -n "FAIL\|checks passed" rebuttal/results/sweep_rebuttal.log | cut -c1-200 | tail -n 6
