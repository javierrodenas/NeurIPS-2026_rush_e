#!/usr/bin/env python3
"""
ICLR EXP 20 — formalizing "beyond null noise" (W6 of the mock review).

For each of the 72 vision cells (12 models x 6 datasets): delta_norm real
(10 quad-seeds) and FIVE spectrum-null replicates (5 quad-seeds each),
storing null mean, null s.d., z = (real - null_mean)/null_sd, and the
Holm-corrected significance verdict at alpha=0.05 over the 72 cells
(computed downstream in the analysis script; here we store the raw z).

Output: rebuttal/results/exp20_null_ztable.csv + exp20.log (incremental).
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b",
          "dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
DATASETS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]

def delta_norm(X, n_quads=500_000, n_seeds=10):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, n_quads) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))

def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu

def cents(model, ds):
    d = np.load(CACHE/f"{model}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

def main():
    csv_path = OUT/"exp20_null_ztable.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"]) for r in rows}
    for ds in DATASETS:
        for m in MODELS:
            if (m, ds) in done: continue
            t0 = time.time()
            C = cents(m, ds)
            dr, dr_sd = delta_norm(C)
            nulls = [delta_norm(specnull(C, rep), n_seeds=5)[0] for rep in range(5)]
            nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
            noise = float(np.sqrt(nsd**2 + dr_sd**2))
            z = (dr - nm)/max(noise, 1e-9)
            rows.append(dict(model=m, dataset=ds, n=len(C), delta=dr, delta_sd=dr_sd,
                             null_mean=nm, null_sd=nsd, excess=dr-nm, z=z,
                             time_s=time.time()-t0))
            print(f"{m:9s} {ds:12s} delta {dr:.4f}±{dr_sd:.4f} null {nm:.4f}±{nsd:.4f} "
                  f"exc {dr-nm:+.4f} z={z:+.1f} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done")

if __name__ == "__main__":
    main()
