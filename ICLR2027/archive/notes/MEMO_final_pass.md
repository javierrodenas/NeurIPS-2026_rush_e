# Memo — final pass (Phase A). Numbers only.

## A1 — depth test with spectrum-matched (Haar-resampled) star hubs (`expR69_depth_haarhubs.csv`; WordNet-30, 10 star seeds, hub null 10×3; rank r_star = star seeds with excess B ≤ real, p = (1+r)/11, resolution 1/11)

| backbone | real excess B | Gaussian star: star / depth / z / r_star | Haar-hub star: star / depth / z / r_star |
|---|---|---|---|
| ViT-T | -0.0173 | +0.0003 / -0.0176 / -1.48 / 0/10 | +0.0011 / -0.0184 / -1.45 / 0/10 |
| ViT-S | -0.0247 | +0.0034 / -0.0281 / -2.36 / 0/10 | +0.0010 / -0.0257 / -2.04 / 0/10 |
| ViT-B | -0.0335 | -0.0000 / -0.0335 / -3.62 / 0/10 | +0.0012 / -0.0347 / -3.01 / 0/10 |
| ViT-L | -0.0336 | -0.0005 / -0.0332 / -4.04 / 0/10 | +0.0000 / -0.0337 / -3.73 / 0/10 |
| DINO-B | -0.0091 | -0.0002 / -0.0089 / -0.94 / 0/10 | +0.0002 / -0.0092 / -1.02 / 0/10 |
| DINOv2-S | +0.0089 | +0.0012 / +0.0077 / +0.66 / 9/10 | +0.0007 / +0.0083 / +0.71 / 10/10 |
| DINOv2-B | -0.0137 | +0.0019 / -0.0157 / -1.19 / 0/10 | +0.0018 / -0.0155 / -1.18 / 0/10 |
| DINOv2-L | -0.0274 | +0.0003 / -0.0277 / -2.39 / 0/10 | -0.0007 / -0.0267 / -2.21 / 0/10 |
| DINOv2-G | -0.0284 | +0.0006 / -0.0290 / -1.78 / 0/10 | -0.0021 / -0.0263 / -1.50 / 0/10 |
| CLIP-B | -0.0159 | +0.0005 / -0.0164 / -1.95 / 0/10 | +0.0012 / -0.0171 / -1.48 / 1/10 |
| CLIP-L | -0.0072 | +0.0006 / -0.0078 / -0.83 / 0/10 | +0.0018 / -0.0090 / -0.74 / 2/10 |
| SigLIP-B | -0.0097 | +0.0008 / -0.0106 / -0.90 / 0/10 | +0.0024 / -0.0121 / -0.78 / 1/10 |

- Certified (z ≤ −2) under the Gaussian star: ViT-S, ViT-B, ViT-L, DINOv2-L (4/12); under the Haar-hub star: ViT-S, ViT-B, ViT-L, DINOv2-L (4/12); intersection: ViT-S, ViT-B, ViT-L, DINOv2-L.
- Haar-hub star on the R9b implanted clouds: false alarms at s = 0: 0 of 60; power at s = 1: 1/60 = 0.02.

## A2 — ViTs supervised on ImageNet-1k leaf labels only (`expR70_inet1k_supervised.csv`; census cache protocol)

| model | IN excess (r/200) | C100 excess (r/200) | depth z Gaussian star / Haar-hub star | ρ_WN (shuffle) | C100 superclass ARI max / cosine-complete |
|---|---|---|---|---|---|
| DeiT-B (IN-1k) | -0.0073 (200) | -0.0257 (200) | -1.26 / -1.21 | +0.079 (+0.008) | 0.56 / 0.38 |
| ViT-B (augreg IN-1k) | -0.0070 (199) | -0.0117 (187) | -3.03 / -2.65 | +0.516 (+0.010) | 0.48 / 0.42 |
| ViT-B | -0.0162 (200) | +0.0057 (43) | -3.62 / -3.01 | +0.480 (-0.003) | 0.61 / 0.45 |

- Reference ρ_WN of the i21k ViTs in exp3: ViT-T +0.521, ViT-S +0.502, ViT-B +0.489, ViT-L +0.534.

## A3 — corollary correlations on the calibrated reading (`expR68_corollary_excess.csv`; Pearson r, Fisher-z 95% CI, n = 10 backbones; dm = family-demeaned)

| gain | dataset | old (raw δ, Table B5) | raw δ̂₉₉.₉ | record excess | depth z |
|---|---|---|---|---|---|
| NC_adv | imagenet | -0.83 [-0.97, -0.65] | -0.84 [-0.96, -0.45] (dm -0.83) | -0.73 [-0.93, -0.19] (dm -0.61) | -0.28 [-0.77, +0.42] |
| NC_adv | cifar100 | -0.87 [-0.97, -0.78] | -0.92 [-0.98, -0.68] (dm -0.58) | -0.79 [-0.95, -0.33] (dm -0.21) | +0.52 [-0.16, +0.87] |
| NC_adv | cifar10 | -0.63 [-0.97, -0.06] | -0.63 [-0.90, +0.00] (dm -0.31) | -0.67 [-0.91, -0.06] (dm -0.34) | n/a |
| NC_adv | dtd | -0.46 [-0.90, +0.22] | -0.67 [-0.91, -0.07] (dm -0.67) | -0.76 [-0.94, -0.26] (dm -0.60) | n/a |
| FS_adv | imagenet | -0.57 [-0.91, +0.14] | -0.86 [-0.97, -0.50] (dm -0.56) | -0.21 [-0.74, +0.48] (dm +0.35) | +0.27 [-0.43, +0.77] |
| FS_adv | cifar100 | -0.67 [-0.95, -0.15] | -0.62 [-0.90, +0.02] (dm -0.51) | -0.79 [-0.95, -0.31] (dm -0.80) | -0.26 [-0.76, +0.45] |
| FS_adv | cifar10 | -0.78 [-0.93, -0.23] | -0.78 [-0.95, -0.31] (dm -0.64) | -0.78 [-0.95, -0.30] (dm -0.66) | n/a |
| FS_adv | dtd | -0.71 [-0.94, -0.32] | -0.87 [-0.97, -0.53] (dm -0.94) | -0.98 [-1.00, -0.91] (dm -0.97) | n/a |
| FS_best_adv | imagenet | — | -0.83 [-0.96, -0.43] (dm -0.50) | -0.08 [-0.68, +0.58] (dm +0.47) | +0.13 [-0.54, +0.70] |
| FS_best_adv | cifar100 | — | -0.77 [-0.94, -0.27] (dm -0.75) | -0.75 [-0.94, -0.23] (dm -0.65) | +0.22 [-0.48, +0.75] |
| FS_best_adv | cifar10 | — | -0.63 [-0.90, -0.01] (dm -0.55) | -0.62 [-0.90, +0.02] (dm -0.56) | n/a |
| FS_best_adv | dtd | — | -0.81 [-0.95, -0.36] (dm -0.78) | -0.90 [-0.98, -0.63] (dm -0.84) | n/a |

- NC H−R on the record excess, CI excluding 0: imagenet yes, cifar100 yes, cifar10 yes, dtd yes; FS H−R: imagenet no, cifar100 yes, cifar10 yes, dtd yes; any depth-z correlation with CI excluding 0: False.

## A4 — MERU radii (`expR71_meru_radii.csv`)

- Learned curvature c = 0.100 (all three sizes). Spatial norm × √c: median 0.258–0.279, 95th percentile 0.263–0.287 across the six image clouds; Lorentz/Euclidean pairwise distance ratio at the median 0.9972–0.9991 (class centroids 0.9988–0.9997).

## A5 — quadruple-budget sweep under the record (`expR72_budget_record.csv`; {ViT-L, DINOv2-L, CLIP-B} × {ImageNet, CIFAR-100, DTD}; Haar × p99.9 × 200 at each budget)

| cell | excess by budget | drift (≥ 10⁵) | drift / s.d. at 5×10⁵ | signs stable |
|---|---|---|---|---|
| CLIP-B / cifar100 | 10000:-0.0186 50000:-0.0195 100000:-0.0188 500000:-0.0195 1000000:-0.0194 2000000:-0.0194 | 0.0006 | 0.10 | True |
| CLIP-B / dtd | 10000:-0.0243 50000:-0.0235 100000:-0.0242 500000:-0.0240 1000000:-0.0243 2000000:-0.0242 | 0.0003 | 0.05 | True |
| CLIP-B / imagenet | 10000:-0.0063 50000:-0.0060 100000:-0.0059 500000:-0.0065 1000000:-0.0064 2000000:-0.0064 | 0.0007 | 0.13 | True |
| DINOv2-L / cifar100 | 10000:-0.0348 50000:-0.0348 100000:-0.0348 500000:-0.0348 1000000:-0.0347 2000000:-0.0347 | 0.0001 | 0.07 | True |
| DINOv2-L / dtd | 10000:-0.0418 50000:-0.0419 100000:-0.0421 500000:-0.0420 1000000:-0.0422 2000000:-0.0420 | 0.0002 | 0.06 | True |
| DINOv2-L / imagenet | 10000:-0.0131 50000:-0.0136 100000:-0.0137 500000:-0.0135 1000000:-0.0136 2000000:-0.0136 | 0.0002 | 0.38 | True |
| ViT-L / cifar100 | 10000:-0.0208 50000:-0.0208 100000:-0.0206 500000:-0.0205 1000000:-0.0205 2000000:-0.0205 | 0.0002 | 0.03 | True |
| ViT-L / dtd | 10000:-0.0305 50000:-0.0308 100000:-0.0312 500000:-0.0311 1000000:-0.0313 2000000:-0.0313 | 0.0002 | 0.05 | True |
| ViT-L / imagenet | 10000:-0.0161 50000:-0.0161 100000:-0.0160 500000:-0.0157 1000000:-0.0159 2000000:-0.0159 | 0.0002 | 0.07 | True |

- imagenet: max drift 0.0007 = 0.38 s.d. (all budgets 0.0007), signs stable True; cifar100: max drift 0.0006 = 0.10 s.d. (all budgets 0.0009), signs stable True; dtd: max drift 0.0003 = 0.06 s.d. (all budgets 0.0008), signs stable True. 'Budget-stable' on ImageNet (drift below the excess's own s.d.): **True**.

## A6 — normalized effect size, excess / δ_null (fraction of the null reading)

- ImageNet centroids (12 cells): -33% to -5% (genuine cells -33% to -11%); all 72 cells -84% to +14%.
- Sample level (24 cells): -39% to +7% (genuine cells -39% to -7%).
- DINOv2-G / CIFAR-100: -57% (excess -0.0395 over a null of 0.0696).
