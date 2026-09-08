# Memo — positive-control pass (Phase A). Numbers only.

## R9 — implanted depth on real ImageNet centroids (`expR64_implanted_depth.csv`, 300 depth runs: 12 backbones × 5 strengths × 5 implant seeds; partition rand6 = 6 super-hubs of 5, RandomState(0); WordNet 6-cut is nested but 10/5/11/2/1/1, run as `wn6` for seed 0)

| backbone | s* (first s with z ≤ −2 in ≥ 4/5 seeds) | hits at s = 0 / 0.25 / 0.5 / 0.75 / 1 (of 5) | mean z at s = 1 | census excess s = 0 → s = 1 (null s.d.) | real z here / B34 |
|---|---|---|---|---|---|
| ViT-T | none | 0/0/0/0/0 | -0.7 | -0.0089 → -0.0094 (0.0051) | -1.48 / -1.48 |
| ViT-S | none | 0/0/0/0/1 | -2.0 | -0.0184 → -0.0200 (0.0043) | -2.36 / -2.36 |
| ViT-B | none | 0/0/0/0/0 | -0.5 | -0.0173 → -0.0208 (0.0041) | -3.62 / -3.62 |
| ViT-L | 1 | 0/0/0/0/5 | -2.6 | -0.0227 → -0.0266 (0.0043) | -4.04 / -4.04 |
| DINO-B | none | 0/0/0/0/0 | -1.3 | -0.0205 → -0.0180 (0.0042) | -0.94 / -0.94 |
| DINOv2-S | none | 0/0/0/0/0 | -1.4 | -0.0163 → -0.0151 (0.0037) | +0.66 / +0.66 |
| DINOv2-B | none | 0/0/0/0/0 | -0.4 | -0.0259 → -0.0274 (0.0030) | -1.19 / -1.19 |
| DINOv2-L | none | 0/0/0/0/0 | -0.8 | -0.0286 → -0.0323 (0.0029) | -2.39 / -2.39 |
| DINOv2-G | none | 0/0/0/0/0 | +0.9 | -0.0280 → -0.0301 (0.0028) | -1.78 / -1.78 |
| CLIP-B | none | 0/0/0/0/3 | -2.2 | -0.0211 → -0.0216 (0.0053) | -1.95 / -1.95 |
| CLIP-L | 0.75 | 0/0/0/5/5 | -3.3 | -0.0224 → -0.0225 (0.0049) | -0.83 / -0.83 |
| SigLIP-B | 1 | 0/0/0/1/5 | -2.7 | -0.0204 → -0.0198 (0.0048) | -0.90 / -0.90 |

- Backbones declared hierarchical at s = 0 (any seed): **0**; max z at s = 0: +0.50.
- Power by s (fraction of runs with z ≤ −2, all backbones): s=0: 0.00, s=0.25: 0.00, s=0.5: 0.00, s=0.75: 0.10, s=1: 0.32.
- s* of the four backbones certified in the real data: ViT-S none, ViT-B none, ViT-L 1, DINOv2-L none; of the other eight: ViT-T none, DINO-B none, DINOv2-S none, DINOv2-B none, DINOv2-G none, CLIP-B none, CLIP-L 0.75, SigLIP-B 1.
- Census excess across s (implant seed 0): max spread 0.0076, i.e. at most 1.85 null s.d.; minimum r over all implanted clouds 172/200.
- WordNet 6-cut partition (seed 0): power at s = 1 0.333333, at s = 0.5 0.
- Untouched cloud through the same code vs Table B34: max |Δz| = 0.00.

- Within/between spread of the real clouds at K = 30 (RMS of within-cluster offsets over point-weighted RMS of hub displacements; the synthetic power sweep covered 0.1–0.6): ViT-T 1.55, ViT-S 1.93, ViT-B 1.96, ViT-L 1.88, DINO-B 2.12, DINOv2-S 3, DINOv2-B 3.71, DINOv2-L 3.88, DINOv2-G 3.92, CLIP-B 1.33, CLIP-L 1.49, SigLIP-B 1.54.
- Frame cluster sizes (classes per WordNet-30 cluster): [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 10, 11, 12, 13, 16, 17, 18, 22, 23, 31, 67, 71, 158, 175, 316].
- Tight variant (same implant after shrinking the real offsets to within/between = 0.6; s ∈ {0, 0.5, 1}, 2 seeds): ViT-T z -0.6/-4.7/-10.2, hits at s=1 2/2, at s=0 0/2; ViT-S z -0.2/-7.0/-10.5, hits at s=1 2/2, at s=0 0/2; ViT-B z -1.0/-7.7/-11.1, hits at s=1 2/2, at s=0 0/2; ViT-L z -1.6/-8.7/-12.3, hits at s=1 2/2, at s=0 0/2; DINO-B z -1.6/-9.4/-11.4, hits at s=1 2/2, at s=0 0/2; DINOv2-S z -0.2/-6.6/-11.9, hits at s=1 2/2, at s=0 0/2; DINOv2-B z -0.0/-8.5/-11.0, hits at s=1 2/2, at s=0 0/2; DINOv2-L z +0.2/-7.4/-10.9, hits at s=1 2/2, at s=0 0/2; DINOv2-G z +0.2/-9.1/-10.7, hits at s=1 2/2, at s=0 0/2; CLIP-B z -1.7/-6.5/-12.5, hits at s=1 2/2, at s=0 0/2; CLIP-L z -2.4/-9.8/-13.4, hits at s=1 2/2, at s=0 2/2; SigLIP-B z -2.1/-8.9/-11.8, hits at s=1 2/2, at s=0 2/2.

## R11 — joint sensitivity of the genuine count (`expR66_joint_sensitivity_summary.csv`; 72 cells, 30 resamples each, 50 Haar replicates per resample)

- Record (Haar × p99.9 × 200, BH): **49/72**, 18/24 on ImageNet + CIFAR-100.
- z_joint = excess / sqrt(sd_null² + sd_boot² + sd_est²) ≤ −2: **42/72**, 18/24. Record-genuine cells that drop: CLIP-B/cifar10, CLIP-L/cifar10, ViT-T/dtd, DINOv2-L/fashionmnist, ViT-S/fashionmnist, SigLIP-B/fashionmnist, SigLIP-B/mnist; non-record cells that pass: none.
- Bootstrap-BH (genuine under BH in ≥ 27 of 30 resamples): **47/72**, 17/24. Record-genuine cells that drop: CLIP-B/cifar10, ViT-T/dtd, CLIP-L/imagenet; newly genuine: DINOv2-S/fashionmnist.
- Bootstrap s.d. of the excess: max 0.0026 over all cells, 0.0011 on ImageNet.

## R10 — fine-tuned ViT-B/16 with injected hierarchy (`expR65_hier_finetune.csv`; census subset of 100 images/class, 2 epochs, same batches for both objectives)

| model | census excess | r/200 (p) | depth | z |
|---|---|---|---|---|
| frozen | -0.0162 | 200 (0.005) | -0.0335 | -3.62 |
| ce | -0.0153 | 200 (0.005) | -0.0303 | -2.33 |
| hier | -0.0162 | 200 (0.005) | -0.0295 | -2.77 |

- Final training-batch accuracy: ce 0.96, hier 0.94 (1549 steps). Positive-control criterion (hier certified and ce not, or hier's z clearly below ce's): **False**.
