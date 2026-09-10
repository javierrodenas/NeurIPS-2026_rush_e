#!/usr/bin/env python3
"""expR71 (final pass, A4): MERU embedding radii against the curvature scale.

From the cached MERU embeddings (Platonic/results/meru_cache/{meru_s,b,l}_{cifar100,imagenet}_img.npz: 'proj' = space-like
hyperboloid coordinates, 'curv' = the learned curvature c): the spatial norm ||x|| times sqrt(c) (median, 95th percentile; the
Lorentz distance reduces to the Euclidean one when this is small against 1), and the ratio of Lorentz to Euclidean pairwise
distances, median over 2e5 random pairs, plus the same ratio on the class centroids (the objects of the census). Output:
expR71_meru_radii.csv."""
import sys, numpy as np, pandas as pd
from pathlib import Path
CACHE = Path("/media/HDD_4TB_2/javi/Platonic/results/meru_cache"); OUT = Path(__file__).resolve().parents[1] / "results"
def lorentz_pairs(X, c, idx_a, idx_b):
    """Lorentz distance between space-like coordinates x, y on the hyperboloid of curvature c: acosh(-<x,y>_L * c) / sqrt(c) with x0 = sqrt(1/c + ||x||^2)."""
    xa, xb = X[idx_a], X[idx_b]; ta = np.sqrt(1.0 / c + (xa ** 2).sum(1)); tb = np.sqrt(1.0 / c + (xb ** 2).sum(1))
    inner = (xa * xb).sum(1) - ta * tb; arg = np.maximum(-inner * c, 1.0 + 1e-7); return np.arccosh(arg) / np.sqrt(c)
rows = []
for m in ("meru_s", "meru_b", "meru_l"):
    for ds in ("cifar100", "imagenet"):
        f = CACHE / f"{m}_{ds}_img.npz"
        if not f.exists(): continue
        d = np.load(f); X = d["proj"].astype(np.float64); y = d["labels"]; c = float(d["curv"])
        r = np.linalg.norm(X, axis=1) * np.sqrt(c); rng = np.random.RandomState(0); n = len(X); a = rng.randint(0, n, 200_000); b = rng.randint(0, n, 200_000); ok = a != b; a, b = a[ok], b[ok]
        dl = lorentz_pairs(X, c, a, b); de = np.linalg.norm(X[a] - X[b], axis=1); ratio = dl / de
        C = np.stack([X[y == k].mean(0) for k in range(int(y.max()) + 1)]); iu = np.triu_indices(len(C), 1)
        dlc = lorentz_pairs(C, c, iu[0], iu[1]); dec = np.linalg.norm(C[iu[0]] - C[iu[1]], axis=1)
        rows.append(dict(model=m, dataset=ds, curv=c, n=n, d=X.shape[1], radius_sqrtc_median=float(np.median(r)), radius_sqrtc_p95=float(np.percentile(r, 95)), radius_sqrtc_max=float(r.max()),
                         lorentz_over_euclid_median=float(np.median(ratio)), lorentz_over_euclid_p95=float(np.percentile(ratio, 95)), centroid_radius_sqrtc_median=float(np.median(np.linalg.norm(C, axis=1) * np.sqrt(c))),
                         centroid_lorentz_over_euclid_median=float(np.median(dlc / dec))))
        print(f"{m} {ds}: c={c:.3f}, ||x||*sqrt(c) median {rows[-1]['radius_sqrtc_median']:.3f} p95 {rows[-1]['radius_sqrtc_p95']:.3f}; Lorentz/Euclid median {rows[-1]['lorentz_over_euclid_median']:.4f} (centroids {rows[-1]['centroid_lorentz_over_euclid_median']:.4f})")
pd.DataFrame(rows).to_csv(OUT / "expR71_meru_radii.csv", index=False); print("DONE expR71")
