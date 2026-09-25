#!/bin/bash
# Detached supervisor v2 (final pass): train-only ImageNet cache extraction (100 img/class),
# DINOv2-G split in two shards (one per GPU), then R1 (expR39b) and R2 (exp21b).
# Survives the Claude session closing (launched with setsid). Idempotent: every step skips
# work whose output already exists.
R=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e
cd $R
X="python3 rebuttal/scripts/extract_imagenet_cache_local.py --sets train"
log() { echo "$(date +%H:%M:%S) $*"; }
r1_rows() { python3 - <<'P'
import csv
from pathlib import Path
f=Path('rebuttal/results/expR39b_census20_cache.csv')
print(sum(1 for r in csv.DictReader(open(f)) if r['dataset']=='imagenet') if f.exists() else 0)
P
}
log "supervisor v2 start"
( $X --models i21k_s i21k_b i21k_l dinov1_b clip_b siglip_b clip_l --device cuda:0; $X --models dinov2_g --shard 1/2 --device cuda:0 ) >> rebuttal/results/extractA.log 2>&1 &
PA=$!
( $X --models dinov2_b dinov2_l --device cuda:1; $X --models dinov2_g --shard 0/2 --device cuda:1 ) >> rebuttal/results/extractB.log 2>&1 &
PB=$!
( while true; do n=$(r1_rows); [ "$n" = 12 ] && break; python3 rebuttal/scripts/expR39b_census20_cache.py >> rebuttal/results/expR39b.log 2>&1; sleep 600; done ) &
PR=$!
wait $PA; log "stream A (GPU0) finished"
wait $PB; log "stream B (GPU1) finished"
$X --models dinov2_g --merge 2 >> rebuttal/results/extractB.log 2>&1; log "dinov2_g shards merged"
python3 rebuttal/scripts/expR39b_census20_cache.py >> rebuttal/results/expR39b.log 2>&1
wait $PR; log "R1 complete ($(r1_rows)/12 ImageNet cells)"
n=$(ls /media/HDD_4TB_2/javi/Platonic/results/practical_tasks_cache/*_imagenet_train.npz | wc -l)
if [ "$n" = 12 ]; then
  log "launching exp21b (R2)"; python3 rebuttal/scripts/exp21b_local_global_K200.py > rebuttal/results/exp21b.log 2>&1; log "R2 complete"
else log "R2 NOT launched: only $n/12 train caches"; fi
log "ALL DONE"
