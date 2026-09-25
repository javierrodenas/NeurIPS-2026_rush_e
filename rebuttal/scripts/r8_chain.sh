#!/bin/bash
# Review-response Phase A, queues 1-2 (detached): the 2x2 census at 200 replicates.
R=/media/HDD_4TB_2/javi/NeurIPS-2026_rush_e; cd $R
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
log() { echo "$(date +%H:%M:%S) $*"; }
V=rebuttal/scripts/expR52_census_haar_p999_200.py; T=rebuttal/scripts/expR53_text_haar_p999_200.py
vis() { # $1 null $2 stat $3 tag
  python3 $V --null $1 --stat $2 --part in1 --datasets imagenet --models i21k_t i21k_s i21k_b i21k_l dinov1_b dinov2_s > rebuttal/results/$3_in1.log 2>&1 &
  python3 $V --null $1 --stat $2 --part in2 --datasets imagenet --models dinov2_b dinov2_l dinov2_g clip_b clip_l siglip_b > rebuttal/results/$3_in2.log 2>&1 &
  python3 $V --null $1 --stat $2 --part tr --datasets cifar100 cifar10 dtd fashionmnist mnist > rebuttal/results/$3_tr.log 2>&1 &
}
txt() { # $1 null $2 stat $3 tag
  CUDA_VISIBLE_DEVICES=0 python3 $T --null $1 --stat $2 --part a --models gpt2 gpt2_m gpt2_l gpt2_xl pythia_410m pythia_1b pythia_2b8 olmo_1b > rebuttal/results/$3_a.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python3 $T --null $1 --stat $2 --part b --models bge_base bge_large gte_base gte_large gte_qwen2 e5_base e5_large > rebuttal/results/$3_b.log 2>&1 &
}
log "queue 1: record (haar x p999), vision + text"
vis haar p999 expR52; txt haar p999 expR53; wait
python3 $V --null haar --stat p999 --merge >> rebuttal/results/expR52_merge.log 2>&1
python3 $T --null haar --stat p999 --merge >> rebuttal/results/expR53_merge.log 2>&1
log "queue 1 merged"
log "queue 2: haar x sup (vision, text) + gauss x p999 (text)"
vis haar sup expR54; txt haar sup expR53hs; wait
python3 $V --null haar --stat sup --merge >> rebuttal/results/expR54_merge.log 2>&1
python3 $T --null haar --stat sup --merge >> rebuttal/results/expR53_merge.log 2>&1
txt gauss p999 expR53gp; wait
python3 $T --null gauss --stat p999 --merge >> rebuttal/results/expR53_merge.log 2>&1
log "R8 DONE"
