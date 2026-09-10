#!/bin/bash
# R16 (final pass, Phase A, CPU): A1 = expR69 (12 parts) then A5 = expR72 (3 parts); 6 workers; merges.
cd /media/HDD_4TB_2/javi/NeurIPS-2026_rush_e
export PLATONIC_ROOT=/media/HDD_4TB_2/javi/Platonic OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
L=rebuttal/results; echo "$(date +%H:%M:%S) R16 start: expR69 x12 + expR72 x3, 6 workers" >> $L/r16_chain.log
{ for m in i21k_t i21k_s i21k_b i21k_l dinov1_b dinov2_s dinov2_b dinov2_l dinov2_g clip_b clip_l siglip_b; do echo "expR69_depth_haarhubs.py $m expR69_$m.log"; done
  for d in imagenet cifar100 dtd; do echo "expR72_budget_record.py $d expR72_$d.log"; done; } | xargs -P 6 -L 1 bash -c 'python3 rebuttal/scripts/$0 --part $1 > rebuttal/results/$2 2>&1'
python3 rebuttal/scripts/expR69_depth_haarhubs.py --merge > $L/expR69_merge.log 2>&1
python3 rebuttal/scripts/expR72_budget_record.py --merge > $L/expR72_merge.log 2>&1
echo "$(date +%H:%M:%S) R16 DONE" >> $L/r16_chain.log
