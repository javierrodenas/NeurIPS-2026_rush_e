# Memo — final pass (Phase A). Numbers only.

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

## A6 — normalized effect size, excess / δ_null (fraction of the null reading)

- ImageNet centroids (12 cells): -33% to -5% (genuine cells -33% to -11%); all 72 cells -84% to +14%.
- Sample level (24 cells): -39% to +7% (genuine cells -39% to -7%).
- DINOv2-G / CIFAR-100: -57% (excess -0.0395 over a null of 0.0696).
