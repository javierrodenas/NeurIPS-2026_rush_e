#!/bin/bash
# R15 (positive-control pass, Phase B runs): R9b = expR64b on the frame of record (12 parts: depth + census + tight) and
# R12 = expR64b on the pre-registered balanced frame (12 parts: depth only); 7-worker queue, then the merges.
cd /media/HDD_4TB_2/javi/NeurIPS-2026_rush_e
export PLATONIC_ROOT=/media/HDD_4TB_2/javi/Platonic OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
L=rebuttal/results; echo "$(date +%H:%M:%S) R15 start: expR64b wn30 x12 + wn30bal x12, 7 workers" >> $L/r15_chain.log
{ for m in i21k_t i21k_s i21k_b i21k_l dinov1_b dinov2_s dinov2_b dinov2_l dinov2_g clip_b clip_l siglip_b; do echo "wn30 $m"; done
  for m in i21k_t i21k_s i21k_b i21k_l dinov1_b dinov2_s dinov2_b dinov2_l dinov2_g clip_b clip_l siglip_b; do echo "wn30bal $m"; done; } \
  | xargs -P 7 -L 1 bash -c 'python3 rebuttal/scripts/expR64b_implanted_depth_v2.py --frame $0 --part $1 > rebuttal/results/expR64b_$0_$1.log 2>&1'
python3 rebuttal/scripts/expR64b_implanted_depth_v2.py --frame wn30 --merge > $L/expR64b_wn30_merge.log 2>&1
python3 rebuttal/scripts/expR64b_implanted_depth_v2.py --frame wn30bal --merge > $L/expR64b_wn30bal_merge.log 2>&1
echo "$(date +%H:%M:%S) R15 DONE" >> $L/r15_chain.log
