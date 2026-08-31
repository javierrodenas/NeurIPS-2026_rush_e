#!/usr/bin/env python3
"""expR39 (post-revision requirement): the full 72-cell vision census re-run with
20 spectrum-null replicates (5 seeds each) and percentile ranks. Transfer sets
from the canonical per-image cache; ImageNet from the centroid store (the four
supervised-ViT centroids there differ from the census cache; rows flagged).
Output: expR39_census20.csv."""
import os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = Path(__file__).resolve().parents[1]/"results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
DATASETS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]
def delta_norm(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
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
def cents(m, ds):
    if ds == "imagenet":
        return np.load(ROOT/f"results/centroids/imagenet_train/{m}.npy").astype(np.float32), (m.startswith("i21k"))
    d = np.load(CACHE/f"{m}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)]), False
def main():
    csv_path = OUT/"expR39_census20.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"]) for r in rows}
    for ds in DATASETS:
        for m in MODELS:
            if (m, ds) in done: continue
            t0 = time.time()
            C, flagged = cents(m, ds)
            dr, dr_sd = delta_norm(C, 10)
            nulls = [delta_norm(specnull(C, r), 5)[0] for r in range(20)]
            nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
            pr = float(np.mean([n > dr for n in nulls]))
            z = (dr-nm)/max(np.sqrt(nsd**2+dr_sd**2), 1e-9)
            rows.append(dict(model=m, dataset=ds, delta=dr, delta_sd=dr_sd, null_mean=nm,
                             null_sd=nsd, excess=dr-nm, z=z, frac_null_above=pr,
                             store_centroids=int(flagged), time_s=time.time()-t0))
            print(f"{m:9s} {ds:12s} exc {dr-nm:+.4f} z {z:+.1f} above {pr:.2f}{' [store]' if flagged else ''} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done expR39")
if __name__ == "__main__":
    main()
