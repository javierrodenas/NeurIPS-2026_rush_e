#!/usr/bin/env python3
"""expR43 (round-7 P3): does the curvature-sign estimator xi confound norm
structure with curvature? References with NO curvature but heterogeneous norms:
iid Gaussian (baseline), Gaussian with lognormal radial rescaling (sigma 0.3/0.6/1.0),
core+shell mixture; plus a sphere with radial jitter. n=1000, d=768; 5 seeds."""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
OUT = Path(__file__).resolve().parents[1]/"results"
def xi_stats(D, n_tri=20000, seed=0):
    n = len(D); rng = np.random.RandomState(seed); diam = D.max(); xs = []
    for a0 in range(0, n_tri, 2000):
        B = min(2000, n_tri - a0); b = rng.randint(0, n, B); c = rng.randint(0, n, B)
        ok = b != c; b, c = b[ok], c[ok]
        S = D[:, b] + D[:, c] + np.abs(D[:, b] - D[:, c]); S[b, np.arange(len(b))] = np.inf; S[c, np.arange(len(b))] = np.inf
        m = S.argmin(0); a = rng.randint(0, n, len(b))
        ok = (a != b) & (a != c) & (a != m); a, b, c, m = a[ok], b[ok], c[ok], m[ok]
        dam = D[a, m]; ok = dam > 1e-9; a, b, c, m, dam = a[ok], b[ok], c[ok], m[ok], dam[ok]
        xi = (dam**2 + D[b, c]**2/4 - (D[a, b]**2 + D[a, c]**2)/2) / (2*dam); xs.append(xi/diam)
    xs = np.concatenate(xs); return float(np.mean(xs)), float((xs < 0).mean())
def refs(seed, n=1000, d=768):
    rng = np.random.RandomState(seed); G = rng.randn(n, d)
    out = {"gaussian": G}
    for s in (0.3, 0.6, 1.0):
        out[f"gauss_lognorm_r{s}"] = G * np.exp(s*rng.randn(n))[:, None]
    core = rng.randn(n//2, d)*0.3; shell = rng.randn(n - n//2, d); shell = 3*shell/np.linalg.norm(shell, axis=1, keepdims=True)*np.sqrt(d)
    out["core_shell"] = np.vstack([core, shell])
    U = G/np.linalg.norm(G, axis=1, keepdims=True); out["sphere"] = U*np.sqrt(d)
    out["sphere_radial_jitter"] = U*np.sqrt(d)*np.exp(0.3*rng.randn(n))[:, None]
    return out
rows=[]
for seed in range(5):
    for name, X in refs(seed).items():
        D = squareform(pdist(X.astype(np.float32), "euclidean")); xi, fneg = xi_stats(D, seed=seed)
        rows.append(dict(reference=name, seed=seed, xi=xi, frac_neg=fneg))
        print(f"{name:22s} s{seed} xi {xi:+.4f} frac_neg {fneg:.2f}")
pd.DataFrame(rows).to_csv(OUT/"expR43_xi_norm_references.csv", index=False); print("Done expR43")
