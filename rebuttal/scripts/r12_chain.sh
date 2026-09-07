#!/bin/bash
# R12: restructuring reruns R7 (expR62 sample-level, 4 parts) and R8 (expR63 MERU, 3 workers over 6 models), then merges.
cd /media/HDD_4TB_2/javi/NeurIPS-2026_rush_e
export PLATONIC_ROOT=/media/HDD_4TB_2/javi/Platonic OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
L=rebuttal/results; echo "$(date +%H:%M:%S) R12 start: expR62 (4 parts) + expR63 (3 workers)" >> $L/r12_chain.log
for k in 0 1 2 3; do python3 rebuttal/scripts/expR62_samplelevel_record.py --part $k > $L/expR62_part$k.log 2>&1 & done
(python3 rebuttal/scripts/expR63_meru_record.py --part meru_s; python3 rebuttal/scripts/expR63_meru_record.py --part clip_s) > $L/expR63_s.log 2>&1 &
(python3 rebuttal/scripts/expR63_meru_record.py --part meru_b; python3 rebuttal/scripts/expR63_meru_record.py --part clip_b) > $L/expR63_b.log 2>&1 &
(python3 rebuttal/scripts/expR63_meru_record.py --part meru_l; python3 rebuttal/scripts/expR63_meru_record.py --part clip_l) > $L/expR63_l.log 2>&1 &
wait
python3 rebuttal/scripts/expR62_samplelevel_record.py --merge > $L/expR62_merge.log 2>&1
python3 rebuttal/scripts/expR63_meru_record.py --merge > $L/expR63_merge.log 2>&1
echo "$(date +%H:%M:%S) R12 DONE" >> $L/r12_chain.log
