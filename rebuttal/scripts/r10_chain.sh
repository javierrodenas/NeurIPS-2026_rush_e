#!/bin/bash
R=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
log() { echo "$(date +%H:%M:%S) $*"; }
log "R10 start: expR55b (leaf frame, anisotropic star, 10 seeds), 4 parts by K"
P=rebuttal/scripts/expR55b_depth_power_leafframe.py
python3 $P --part k6 --K 6 > rebuttal/results/expR55b_k6.log 2>&1 &
python3 $P --part k12 --K 12 > rebuttal/results/expR55b_k12.log 2>&1 &
python3 $P --part k20 --K 20 > rebuttal/results/expR55b_k20.log 2>&1 &
python3 $P --part k30 --K 30 > rebuttal/results/expR55b_k30.log 2>&1 &
wait
python3 $P --merge > rebuttal/results/expR55b_merge.log 2>&1
python3 $P --figure >> rebuttal/results/expR55b_merge.log 2>&1
log "R10 DONE"
