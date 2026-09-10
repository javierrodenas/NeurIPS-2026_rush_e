#!/bin/bash
# R14: after R13, the tight variant of R9 (real clusters shrunk to within/between = 0.6; s in {0, 0.5, 1}, 2 seeds; 12 backbones), then merge.
cd /media/HDD_4TB_2/javi/NeurIPS-2026_rush_e
export PLATONIC_ROOT=/media/HDD_4TB_2/javi/Platonic OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
L=rebuttal/results; while ! grep -q "R13 DONE" $L/r13_chain.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) R14 start: expR64 --tight x12, 6 workers" >> $L/r14_chain.log
for m in i21k_t i21k_s i21k_b i21k_l dinov1_b dinov2_s dinov2_b dinov2_l dinov2_g clip_b clip_l siglip_b; do echo "$m"; done | xargs -P 6 -I{} bash -c 'python3 rebuttal/scripts/expR64_implanted_depth.py --part {} --tight > rebuttal/results/expR64_tight_{}.log 2>&1'
python3 rebuttal/scripts/expR64_implanted_depth.py --merge > $L/expR64_merge.log 2>&1
echo "$(date +%H:%M:%S) R14 DONE" >> $L/r14_chain.log
