#!/usr/bin/env python3
"""expR39b (final pass, R1): the 20-replicate vision census on ONE centroid source.

expR39_census20.csv used the centroid *store* for ImageNet, whose supervised-ViT centroids
differ from the census cache (the daggered rows). This re-run computes the 12 ImageNet
cells from the per-image census cache ({m}_imagenet_train.npz, 100 img/class; regenerated
locally by extract_imagenet_cache_local.py with the original extraction pipeline) and
copies the 60 transfer rows VERBATIM from expR39_census20.csv, which are cache-based
already. Estimator, null construction and seeds are identical to expR39: real delta with
10 quadruple seeds, 20 spectrum-null replicates (seeds 300+rep) with 5 quadruple seeds
each, 5x10^5 quadruples per seed.

Output: rebuttal/results/expR39b_census20_cache.csv
"""
import os, sys, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
CACHE = ROOT/"results/practical_tasks_cache"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]

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
    """exp20_null_ztable.py's cents(): census cache for every dataset, ImageNet included."""
    d = np.load(CACHE/f"{m}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

def main():
    csv_path = OUT/"expR39b_census20_cache.csv"
    if csv_path.exists():
        rows = pd.read_csv(csv_path).to_dict("records")
    else:
        # 60 transfer rows copied verbatim from expR39_census20.csv (cache-based already)
        src = pd.read_csv(OUT/"expR39_census20.csv")
        rows = src[src.dataset != "imagenet"].to_dict("records")
        assert len(rows) == 60, len(rows)
        print("copied 60 transfer rows verbatim from expR39_census20.csv (cache-based already)")
    done = {(r["model"], r["dataset"]) for r in rows}
    for m in MODELS:
        if (m, "imagenet") in done: continue
        if not (CACHE/f"{m}_imagenet_train.npz").exists():
            print(f"{m}: cache not extracted yet, skipping this pass"); continue
        t0 = time.time()
        C = cents(m, "imagenet")
        dr, dr_sd = delta_norm(C, 10)
        nulls = [delta_norm(specnull(C, r), 5)[0] for r in range(20)]
        nm, nsd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
        pr = float(np.mean([n > dr for n in nulls]))
        z = (dr-nm)/max(np.sqrt(nsd**2+dr_sd**2), 1e-9)
        rows.append(dict(model=m, dataset="imagenet", delta=dr, delta_sd=dr_sd, null_mean=nm,
                         null_sd=nsd, excess=dr-nm, z=z, frac_null_above=pr,
                         store_centroids=0, time_s=time.time()-t0))
        print(f"{m:9s} imagenet exc {dr-nm:+.4f} z {z:+.1f} above {pr:.2f} ({rows[-1]['time_s']:.0f}s)")
        pd.DataFrame(rows).drop_duplicates(subset=["model","dataset"], keep="first").to_csv(csv_path, index=False)
    # summary counts for the text (recomputed, never typed)
    df = pd.read_csv(csv_path); HIER = {"imagenet","cifar100","cifar10","dtd"}
    sign = int((df.excess < 0).sum())
    below_all = int((df.frac_null_above == 1.0).sum())
    h = df[df.dataset.isin(HIER)]
    print(f"SUMMARY: {sign}/72 sign-negative | {below_all}/72 below every one of 20 replicates "
          f"| hierarchical {int((h.frac_null_above==1.0).sum())}/48 below every replicate")
    print("Done expR39b")

if __name__ == "__main__":
    main()
