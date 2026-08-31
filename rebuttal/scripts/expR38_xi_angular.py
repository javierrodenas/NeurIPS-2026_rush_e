#!/usr/bin/env python3
"""expR38 (fresh-review Q2): xi on L2-normalized centroids with spherical geodesic
distance. Does DINOv2's most-negative xi-excess survive in the angular geometry,
i.e. is the delta/xi disagreement on ImageNet a norm-structure effect?
12 models, ImageNet centroid store; spectrum null built on the normalized cloud
and re-normalized (20 reps); xi_stats as in exp12. Output: expR38_xi_angular.csv."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); OUT = Path(__file__).resolve().parents[1]/"results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
def xi_stats(D, n_tri=20000, seed=0):
    # vectorized version of exp12's estimator: same statistic, batched sampling
    n = len(D); rng = np.random.RandomState(seed); diam = D.max()
    xs = []
    for a0 in range(0, n_tri, 2000):
        B = min(2000, n_tri - a0)
        b = rng.randint(0, n, B); c = rng.randint(0, n, B)
        ok = b != c; b, c = b[ok], c[ok]
        S = D[:, b] + D[:, c] + np.abs(D[:, b] - D[:, c])
        S[b, np.arange(len(b))] = np.inf
        S[c, np.arange(len(b))] = np.inf
        m = S.argmin(0)
        a = rng.randint(0, n, len(b))
        ok = (a != b) & (a != c) & (a != m); a, b, c, m = a[ok], b[ok], c[ok], m[ok]
        dam = D[a, m]; ok = dam > 1e-9; a, b, c, m, dam = a[ok], b[ok], c[ok], m[ok], dam[ok]
        xi = (dam**2 + D[b, c]**2/4 - (D[a, b]**2 + D[a, c]**2)/2) / (2*dam)
        xs.append(xi / diam)
    xs = np.concatenate(xs)
    return float(np.mean(xs)), float((xs < 0).mean())
def geoD(X):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = np.clip(Xn @ Xn.T, -1.0, 1.0)
    D = np.arccos(G); np.fill_diagonal(D, 0.0)
    return D
def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu
rows=[]
for m in MODELS:
    t0=time.time()
    C = np.load(ROOT/f"results/centroids/imagenet_train/{m}.npy").astype(np.float32)
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    D = geoD(Cn)
    xr, fneg = xi_stats(D)
    x_sd = float(np.std([xi_stats(D, seed=s)[0] for s in range(1,6)], ddof=1))
    nulls = [xi_stats(geoD(specnull(Cn, r)))[0] for r in range(20)]
    nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
    z = (xr-nm)/max(np.sqrt(nsd**2+x_sd**2),1e-9)
    rows.append(dict(model=m, xi_geo=xr, frac_neg=fneg, null_mean=nm, null_sd=nsd,
                     excess=xr-nm, z=z, time_s=time.time()-t0))
    print(f"{m:9s} xi_geo {xr:+.4f} fneg {fneg:.2f} null {nm:+.4f} exc {xr-nm:+.4f} z {z:+.1f} ({rows[-1]['time_s']:.0f}s)")
    pd.DataFrame(rows).to_csv(OUT/"expR38_xi_angular.csv", index=False)
print("Done expR38")
