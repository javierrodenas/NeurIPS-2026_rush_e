# calibrated_delta — the paper's instrument as one script

Reads a matrix of class centroids and returns the calibrated tree-likeness reading used throughout
*Is There a Platonic Tree?*: the raw normalized Gromov δ, its **excess over a spectrum-matched Gaussian
null** with a percentile rank and a left-tail p-value, and (with superclass labels) the **matched-star
depth test**. Estimator, null constructions and seeds are copied verbatim from the scripts that
produced the paper's tables, so the same matrix always gives the same numbers.

Requirements: `numpy`, `scipy` (the demo and the check also use `scikit-learn`).

## Usage

```
python calibrated_delta.py centroids.npy [--labels superclasses.npy] [--reps 200] [--json out.json]
```

* `centroids.npy` — `n × d` array, one row per class (float32/64).
* `--labels` — optional integer array of length `n` (superclass of each class); enables the depth test.
* `--reps` — number of spectrum-null replicates (paper: 200; genuine := `p ≤ 0.05`, i.e. `r ≥ 191/200`).
* `--json` — also write the results as JSON.

Output fields: `delta` (raw δ_norm, mean over 10 quadruple seeds of 5×10⁵ quadruples), `null_mean`,
`null_sd`, `excess = delta − null_mean`, `r_above` (null replicates above the real value),
`p_left = (1 + #{null ≤ real}) / (reps + 1)`, `genuine`; with labels: `excessB_real` (excess under the
hub-randomizing Haar null), `excessB_star` (same for a matched star with the real hub radius, within-
superclass spread and cluster sizes), `depth_excess = excessB_real − excessB_star` (negative = more
tree-like than a matched star) and `z_depth`.

What the numbers mean: a **negative excess** with `p ≤ 0.05` certifies structure beyond the second
moments of the cloud — clustering included; a pure star already produces it. Only a **negative depth**
beyond the noise indicates hierarchy beyond the hubs.

## Reproduction check (two cells of the paper)

```
python run_checks.py          # ~15 CPU-minutes; writes rebuttal/results/tool_check.json
```

Reproduces Table 1 (census of record, 200 replicates) for ViT-L / CIFAR-100 and DINOv2-L / ImageNet
(excess to 3 dp, `r` and `p` exactly) and Table B29 (matched-star depth test) for the same two cells.
`rebuttal/scripts/sweep_freeze.py` compares the JSON with the released CSVs at every freeze. Paths to
the feature caches are read from `PLATONIC_ROOT` / `PLATONIC_RESULTS` (defaults in the script). The
labels used for the depth test are saved next to the tool as `example_labels_*.npy`.

## Demo (no data needed)

```
python demo.py
```

Builds three synthetic clouds (n = 60, d = 32, fixed seeds) — an iid Gaussian cloud, a 6-cluster
star and a two-level 6×5 hierarchy — and runs the instrument with 50 replicates. Expected verdicts:
the random cloud is not genuine (p > 0.05); the star is genuine with a positive depth (no hierarchy
beyond its matched star); the hierarchy is genuine with a negative depth (more tree-like than its
matched star). At n = 60 the depth test is underpowered (K = 6 hubs), so the demo reads the sign of the
depth, not its z; the paper's calibration (Tables B25/B29) uses n = 100–1000. Runs in about 1.5 min.
