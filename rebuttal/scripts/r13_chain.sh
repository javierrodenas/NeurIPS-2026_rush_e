#!/bin/bash
# R13 (positive-control pass, Phase A): R9 = expR64 implanted depth (12 parts, one per backbone) then R11 = expR66 joint
# sensitivity (12 parts); a 6-worker queue, then the merges. Detached: setsid nohup bash r13_chain.sh &
cd /media/HDD_4TB_2/javi/NeurIPS-2026_rush_e
export PLATONIC_ROOT=/media/HDD_4TB_2/javi/Platonic OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
L=rebuttal/results; echo "$(date +%H:%M:%S) R13 start: expR64 x12 + expR66 x12, 6 workers" >> $L/r13_chain.log
{ for m in i21k_t i21k_s i21k_b i21k_l dinov1_b dinov2_s dinov2_b dinov2_l dinov2_g clip_b clip_l siglip_b; do echo "expR64_implanted_depth.py --part $m expR64_$m.log"; done
  for k in 0 1 2 3 4 5 6 7 8 9 10 11; do echo "expR66_joint_sensitivity.py --part $k expR66_part$k.log"; done; } \
  | xargs -P 6 -L 1 bash -c 'python3 rebuttal/scripts/$0 $1 $2 > rebuttal/results/$3 2>&1'
python3 rebuttal/scripts/expR64_implanted_depth.py --merge > $L/expR64_merge.log 2>&1
python3 rebuttal/scripts/expR66_joint_sensitivity.py --merge > $L/expR66_merge.log 2>&1
echo "$(date +%H:%M:%S) R13 DONE" >> $L/r13_chain.log
