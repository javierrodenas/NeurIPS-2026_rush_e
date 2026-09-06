# calibrated_delta — the paper's instrument as one script

Reads a matrix of class centroids and returns the calibrated reading used throughout *Is There a Platonic
Tree?*: the raw four-point statistic (99.9th percentile of the four-point defect, the reading of record; the
supremum with `--stat sup`), its **excess over a spectrum-matched null** (Haar construction, exact sample
spectrum, the null of record; Gaussian coefficients with `--null gauss`) with a percentile rank and a
left-tail p-value, and (with superclass labels) the **matched-star depth test** (anisotropic star, 10 seeds;
`--star iso` for the isotropic one). Estimator, null constructions and seeds are copied verbatim from the scripts that
produced the paper's tables, so the same matrix always gives the same numbers.

Requirements: `numpy`, `scipy` (the demo and the check also use `scikit-learn`).

## Usage

```
python calibrated_delta.py centroids.npy [--labels superclasses.npy] [--reps 200] [--null haar|gauss] [--stat p999|sup] [--star aniso|iso] [--json out.json]
```

* `centroids.npy` — `n × d` array, one row per class (float32/64).
* `--labels` — optional integer array of length `n` (superclass of each class); enables the depth test.
* `--reps` — number of spectrum-null replicates (paper: 200). The tool reports the uncorrected left-tail `p`;
  in the paper a cell is *genuine* when its Benjamini–Hochberg-corrected `p` across the census is ≤ 0.05.
* `--json` — also write the results as JSON.

Output fields: `delta` (raw statistic, mean over 10 quadruple seeds of 5×10⁵ quadruples), `null_mean`,
`null_sd`, `excess = delta − null_mean`, `r_above` (null replicates above the real value),
`p_left = (1 + #{null ≤ real}) / (reps + 1)`, `genuine_uncorrected`; with labels: `excessB_real` (excess under
the hub-randomizing Haar null), `excessB_star` (same for a matched star: Gaussian hubs with the real hub
radius and, per cluster, a Haar sample with the cluster's own covariance), `depth_excess = excessB_real −
excessB_star` (negative = hierarchy above the labelled clusters beyond the star) and `z_depth`. The test
certifies hierarchy *above* its frame only: clusters below the labels are invisible to it by construction.

What the numbers mean: a **negative excess** below the null certifies beyond-null (clustered) structure —
clustering included; a pure star already produces it. Only a **negative depth** beyond the noise indicates
hierarchy above the labelled clusters.

## Reproduction check (two cells of the paper)

```
python run_checks.py          # ~15 CPU-minutes; writes rebuttal/results/tool_check.json
```

Reproduces Table 1 (census of record: Haar null, 99.9th-percentile statistic, 200 replicates) for ViT-L /
CIFAR-100 and DINOv2-L / ImageNet (excess to 3 dp, `r` and `p` exactly) and the anisotropic-star depth test
of Table B34 for the same two cells.
`rebuttal/scripts/sweep_freeze.py` compares the JSON with the released CSVs at every freeze. Paths to
the feature caches are read from `PLATONIC_ROOT` / `PLATONIC_RESULTS` (defaults in the script). The
labels used for the depth test are saved next to the tool as `example_labels_*.npy`.

## Demo (no data needed)

```
python demo.py
```

Builds three synthetic clouds (n = 60, d = 32, fixed seeds) — an iid Gaussian cloud, a 6-cluster
star and a two-level 6×5 hierarchy — and runs the instrument with 50 replicates. Expected readings:
the random cloud is not below the null; the star and the hierarchy are, with the depth test (frame = the
6 top-level clusters) separating them only if the hierarchy lies *above* that frame; a hierarchy built
below the frame is invisible to it by construction (Table B35). Runs in about 2 min.
