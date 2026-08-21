#!/usr/bin/env python3
"""
ICLR EXP 26 — matched nulls for the curvature-sign estimator xi (W3, round 3).

The reviewer's point is exactly the paper's own: if raw delta was uninterpretable
without a matched null, raw xi is too. For the 12 vision models on ImageNet
centroids: xi_real, spectrum-matched null xi (3 reps), excess and z. Also the
iid-Gaussian reference for context. Resolves the DINOv2xImageNet three-descriptor
tension: does its negative xi survive its own null?

Output: exp26_xi_nulls.csv + log.
"""
import sys, time
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
sys.path.insert(0, str(ROOT/"rebuttal/scripts"))
from exp12_curvature_sign import xi_stats   # exact same estimator

MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b",
          "dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]

def cents(m):
    d = np.load(CACHE/f"{m}_imagenet_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(1000)])

def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu

def xi_of(X):
    D = squareform(pdist(X, "euclidean"))
    out = xi_stats(D)
    # xi_stats returns (mean, median, frac_neg) per exp12 convention
    return out

def main():
    csv_path = OUT/"exp26_xi_nulls.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["model"] for r in rows}
    for m in MODELS:
        if m in done: continue
        t0 = time.time()
        C = cents(m)
        xr, xmed, fneg = xi_of(C)
        nulls = [xi_of(specnull(C, rep)) for rep in range(3)]
        nm = float(np.mean([n[0] for n in nulls])); nsd = float(np.std([n[0] for n in nulls]))
        nfneg = float(np.mean([n[2] for n in nulls]))
        z = (xr-nm)/max(nsd, 1e-9)
        rows.append(dict(model=m, xi=xr, frac_neg=fneg, xi_null=nm, xi_null_sd=nsd,
                         null_frac_neg=nfneg, excess=xr-nm, z=z, time_s=time.time()-t0))
        print(f"{m:9s} xi {xr:+.4f} (fneg {fneg:.2f}) | null {nm:+.4f}±{nsd:.4f} "
              f"(fneg {nfneg:.2f}) | exc {xr-nm:+.4f} z={z:+.1f} | {rows[-1]['time_s']:.0f}s")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done")

if __name__ == "__main__":
    main()
