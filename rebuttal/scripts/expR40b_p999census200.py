#!/usr/bin/env python3
"""expR40b: the p99.9 robustness census at 200 spectrum-null replicates (same protocol as the census of record).

Copy of expR40_p999census.py with two changes: range(200) replicates instead of 20, and the census cache
as centroid source for every dataset (expR40 read ImageNet from the centroid store); statistic unchanged:
the 99.9th percentile of the four-point defect over the same 5e5 sampled quadruples per seed. Same
seeds (300+rep), same 5 quadruple seeds per replicate, same estimator (real delta with 10 seeds,
5x10^5 quadruples per seed), same centroid source (the census cache for every dataset, ImageNet
included). Stored per cell: frac_null_above (= r/200), r_above = #{null > real}, and the
left-tail add-one p-value p_left = (1 + #{null <= real}) / 201. "Genuine" in the paper is
p_left <= 0.05, i.e. r_above >= 191 of 200.

Parallel by dataset: `--part TAG --datasets ... [--models ...]` writes
expR40b_p999census200.part_TAG.csv; `--merge` concatenates all parts into
expR40b_p999census200.csv in canonical (dataset, model) order.
"""
import os, sys, time, argparse, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
CACHE = ROOT/"results/practical_tasks_cache"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
DATASETS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]
N_REP = 200

def delta_norm(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D)
    out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(np.percentile((S[:,2]-S[:,1])/2, 99.9)/diam)   # p99.9 statistic (expR40)
    return float(np.mean(out)), float(np.std(out))

def specnull(C, rep):
    mu = C.mean(0); Cc = C - mu
    U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep)
    G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C))
    return (G*S)@Vt + mu

def cents(m, ds):
    d = np.load(CACHE/f"{m}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

def run(part, datasets, models):
    csv_path = OUT/f"expR40b_p999census200.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"]) for r in rows}
    for ds in datasets:
        for m in models:
            if (m, ds) in done: continue
            t0 = time.time()
            C = cents(m, ds)
            dr, dr_sd = delta_norm(C, 10)
            nulls = np.array([delta_norm(specnull(C, r), 5)[0] for r in range(N_REP)])
            nm, nsd = float(nulls.mean()), float(nulls.std(ddof=1))
            r_above = int((nulls > dr).sum())
            p_left = (1 + int((nulls <= dr).sum())) / (N_REP + 1)
            z = (dr-nm)/max(np.sqrt(nsd**2+dr_sd**2), 1e-9)
            rows.append(dict(model=m, dataset=ds, delta=dr, delta_sd=dr_sd, null_mean=nm, null_sd=nsd,
                             excess=dr-nm, z=z, frac_null_above=r_above/N_REP, r_above=r_above, p_left=p_left,
                             store_centroids=0, time_s=time.time()-t0))
            print(f"{m:9s} {ds:12s} exc {dr-nm:+.4f} r {r_above}/{N_REP} p {p_left:.4f} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part}")

def merge():
    parts = sorted(glob.glob(str(OUT/"expR40b_p999census200.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model","dataset"], keep="first")
    df["_d"] = df.dataset.map({d: i for i, d in enumerate(DATASETS)}); df["_m"] = df.model.map({m: i for i, m in enumerate(MODELS)})
    df = df.sort_values(["_d","_m"]).drop(columns=["_d","_m"])
    assert len(df) == 72, len(df)
    df.to_csv(OUT/"expR40b_p999census200.csv", index=False)
    HIER = {"imagenet","cifar100","cifar10","dtd"}; h = df[df.dataset.isin(HIER)]
    print(f"SUMMARY: {int((df.excess<0).sum())}/72 sign-negative | genuine (p_left<=0.05): {int((df.p_left<=0.05).sum())}/72 "
          f"| hierarchical {int((h.p_left<=0.05).sum())}/48 | flat {int((df[~df.dataset.isin(HIER)].p_left<=0.05).sum())}/24 "
          f"| below every replicate (r=200): {int((df.r_above==200).sum())}/72")
    print("merged expR40b_p999census200.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default=None); ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--merge", action="store_true")
    A = ap.parse_args()
    if A.merge: merge()
    else:
        assert A.part, "--part TAG required"
        run(A.part, A.datasets, A.models)
