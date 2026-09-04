#!/bin/bash
# Review-response Phase A, queues 3-5 (detached): waits for the 2x2 census chain (r8), then runs
# the cosine census (A3), the ImageNet bootstrap (A1), the depth power sweep and the depth variants (A2),
# and the cut-free tree map (A4). Idempotent: every part resumes from its own CSV.
R=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
log() { echo "$(date +%H:%M:%S) $*"; }
until grep -q "R8 DONE" rebuttal/results/r8_chain.log 2>/dev/null; do sleep 120; done
log "queue 3: cosine census (vision 3 parts + text)"
C=rebuttal/scripts/expR57_census_cosine.py
python3 $C --part in1 --datasets imagenet --models i21k_t i21k_s i21k_b i21k_l dinov1_b dinov2_s > rebuttal/results/expR57_in1.log 2>&1 &
python3 $C --part in2 --datasets imagenet --models dinov2_b dinov2_l dinov2_g clip_b clip_l siglip_b > rebuttal/results/expR57_in2.log 2>&1 &
python3 $C --part tr --datasets cifar100 cifar10 dtd fashionmnist mnist > rebuttal/results/expR57_tr.log 2>&1 &
python3 $C --text --part txt > rebuttal/results/expR57_txt.log 2>&1 &
python3 rebuttal/scripts/expR58_treemap_cutfree.py > rebuttal/results/expR58.log 2>&1 &
wait
python3 $C --merge >> rebuttal/results/expR57_merge.log 2>&1
log "queue 3 merged"
log "queue 4: ImageNet bootstrap (4 parts) + depth power sweep (4 parts by K)"
B=rebuttal/scripts/expR59_imagenet_bootstrap.py; P=rebuttal/scripts/expR55_depth_power.py
python3 $B --part b1 --models i21k_t i21k_s i21k_b > rebuttal/results/expR59_b1.log 2>&1 &
python3 $B --part b2 --models i21k_l dinov1_b dinov2_s > rebuttal/results/expR59_b2.log 2>&1 &
python3 $B --part b3 --models dinov2_b dinov2_l dinov2_g > rebuttal/results/expR59_b3.log 2>&1 &
python3 $B --part b4 --models clip_b clip_l siglip_b > rebuttal/results/expR59_b4.log 2>&1 &
python3 $P --part k6 --K 6 > rebuttal/results/expR55_k6.log 2>&1 &
python3 $P --part k12 --K 12 > rebuttal/results/expR55_k12.log 2>&1 &
wait
python3 $P --part k20 --K 20 > rebuttal/results/expR55_k20.log 2>&1 &
python3 $P --part k30 --K 30 > rebuttal/results/expR55_k30.log 2>&1 &
log "queue 5: depth variants on the real backbones (3 parts)"
V=rebuttal/scripts/expR56_depth_variants.py
python3 $V --part v1 --models i21k_t i21k_s i21k_b i21k_l > rebuttal/results/expR56_v1.log 2>&1 &
python3 $V --part v2 --models dinov1_b dinov2_s dinov2_b dinov2_l > rebuttal/results/expR56_v2.log 2>&1 &
python3 $V --part v3 --models dinov2_g clip_b clip_l siglip_b > rebuttal/results/expR56_v3.log 2>&1 &
wait
python3 $B --merge >> rebuttal/results/expR59_merge.log 2>&1
python3 $P --merge >> rebuttal/results/expR55_merge.log 2>&1
python3 $P --figure >> rebuttal/results/expR55_merge.log 2>&1
python3 $V --merge >> rebuttal/results/expR56_merge.log 2>&1
log "R9 DONE"
