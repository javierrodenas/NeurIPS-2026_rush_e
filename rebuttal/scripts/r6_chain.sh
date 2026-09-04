#!/bin/bash
# R6 chain (detached): vision census at 200 replicates (3 CPU parts) + text census at 200 replicates
# (2 GPU parts), then merge both. Idempotent: parts resume from their own CSVs.
R=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1   # 5 processes on 8 shared cores: no BLAS oversubscription
log() { echo "$(date +%H:%M:%S) $*"; }
log "R6 start"
python3 rebuttal/scripts/expR39c_census200_cache.py --part in1 --datasets imagenet --models i21k_t i21k_s i21k_b i21k_l dinov1_b dinov2_s > rebuttal/results/expR39c_in1.log 2>&1 &
python3 rebuttal/scripts/expR39c_census200_cache.py --part in2 --datasets imagenet --models dinov2_b dinov2_l dinov2_g clip_b clip_l siglip_b > rebuttal/results/expR39c_in2.log 2>&1 &
python3 rebuttal/scripts/expR39c_census200_cache.py --part tr --datasets cifar100 cifar10 dtd fashionmnist mnist > rebuttal/results/expR39c_tr.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 python3 rebuttal/scripts/expR48b_text_census200_bs1.py --part a --models gpt2 gpt2_m gpt2_l gpt2_xl pythia_410m pythia_1b pythia_2b8 olmo_1b > rebuttal/results/expR48b_a.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 rebuttal/scripts/expR48b_text_census200_bs1.py --part b --models bge_base bge_large gte_base gte_large gte_qwen2 e5_base e5_large > rebuttal/results/expR48b_b.log 2>&1 &
wait
log "parts finished"
python3 rebuttal/scripts/expR39c_census200_cache.py --merge >> rebuttal/results/expR39c_merge.log 2>&1
python3 rebuttal/scripts/expR48b_text_census200_bs1.py --merge >> rebuttal/results/expR48b_merge.log 2>&1
log "R6 DONE"
