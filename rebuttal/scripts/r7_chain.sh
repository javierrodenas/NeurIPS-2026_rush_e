#!/bin/bash
R=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
log() { echo "$(date +%H:%M:%S) $*"; }
log "R7 start (expR40b p99.9 census, 200 replicates)"
python3 rebuttal/scripts/expR40b_p999census200.py --part in1 --datasets imagenet --models i21k_t i21k_s i21k_b i21k_l dinov1_b dinov2_s > rebuttal/results/expR40b_in1.log 2>&1 &
python3 rebuttal/scripts/expR40b_p999census200.py --part in2 --datasets imagenet --models dinov2_b dinov2_l dinov2_g clip_b clip_l siglip_b > rebuttal/results/expR40b_in2.log 2>&1 &
python3 rebuttal/scripts/expR40b_p999census200.py --part tr --datasets cifar100 cifar10 dtd fashionmnist mnist > rebuttal/results/expR40b_tr.log 2>&1 &
wait
python3 rebuttal/scripts/expR40b_p999census200.py --merge >> rebuttal/results/expR40b_merge.log 2>&1
log "R7 DONE"
