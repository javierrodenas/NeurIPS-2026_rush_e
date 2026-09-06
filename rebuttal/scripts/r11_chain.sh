#!/bin/bash
# R11: expR60 class-count sweep under the record (3 models x 2 modes, 6 workers), then merge.
cd /media/HDD_4TB_2/javi/NeurIPS-2026_rush_e
export PLATONIC_ROOT=/media/HDD_4TB_2/javi/Platonic OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
L=rebuttal/results; echo "$(date +%H:%M:%S) R11 start: expR60 (Haar x p99.9 x 200), 6 parts" >> $L/r11_chain.log
for p in dinov2_l_random dinov2_l_coherent dinov2_g_random dinov2_g_coherent clip_l_random clip_l_coherent; do
  python3 rebuttal/scripts/expR60_c_sweep_record.py --part $p > $L/expR60_$p.log 2>&1 &
done
wait
python3 rebuttal/scripts/expR60_c_sweep_record.py --merge > $L/expR60_merge.log 2>&1
echo "$(date +%H:%M:%S) R11 DONE" >> $L/r11_chain.log
