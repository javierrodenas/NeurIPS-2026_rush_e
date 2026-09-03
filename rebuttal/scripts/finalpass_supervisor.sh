#!/bin/bash
# Detached supervisor for the final-version pass: survives the Claude session closing.
# Finishes the ImageNet cache extraction (skips completed models), then R1 (expR39b)
# and R2 (exp21b). Idles politely while the in-session processes are alive.
R=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e
C=/media/HDD_4TB_2/javi/Platonic/results/practical_tasks_cache
cd $R
log() { echo "$(date +%H:%M:%S) $*"; }
while true; do
  ntr=$(ls $C/*_imagenet_train.npz 2>/dev/null | wc -l)
  nex=$(ls $C/*_imagenet_exp5sub100.npz 2>/dev/null | wc -l)
  r1=$(python3 - <<'P'
import csv
from pathlib import Path
f=Path('rebuttal/results/expR39b_census20_cache.csv')
print(sum(1 for r in csv.DictReader(open(f)) if r['dataset']=='imagenet') if f.exists() else 0)
P
)
  r2=$(python3 - <<'P'
import csv
from pathlib import Path
f=Path('rebuttal/results/exp21b_local_global_K200.csv')
print(sum(1 for _ in csv.DictReader(open(f))) if f.exists() else 0)
P
)
  log "state: train $ntr/12, exp5sub $nex/12, R1 $r1/12, R2 $r2/66"
  if [ "$ntr" = 12 ] && [ "$nex" = 12 ] && [ "$r1" = 12 ] && [ "$r2" = 66 ]; then log "ALL DONE"; break; fi
  # 1) extraction: only if no extractor is running
  if ! pgrep -f "extract_imagenet_cache_loca[l].py --models" >/dev/null; then
    if [ "$ntr" != 12 ] || [ "$nex" != 12 ]; then
      log "no extractor alive; resuming extraction"
      python3 rebuttal/scripts/extract_imagenet_cache_local.py --models i21k_t i21k_s i21k_b i21k_l dinov1_b clip_b siglip_b --device cuda:0 >> rebuttal/results/extractA.log 2>&1 &
      python3 rebuttal/scripts/extract_imagenet_cache_local.py --models dinov2_s dinov2_b dinov2_l dinov2_g clip_l --device cuda:1 >> rebuttal/results/extractB.log 2>&1 &
      wait
    fi
  fi
  # 2) R1: run when something new might be available and no other copy is alive
  if [ "$r1" != 12 ] && ! pgrep -f "expR39b_census20_cach[e].py" >/dev/null; then
    python3 rebuttal/scripts/expR39b_census20_cache.py >> rebuttal/results/expR39b.log 2>&1
  fi
  # 3) R2: needs all 12 exp5sub files; run once, no other copy alive
  if [ "$nex" = 12 ] && [ "$r2" != 66 ] && ! pgrep -f "exp21b_local_global_K20[0].py" >/dev/null; then
    log "launching exp21b"
    python3 rebuttal/scripts/exp21b_local_global_K200.py >> rebuttal/results/exp21b.log 2>&1
  fi
  sleep 300
done
