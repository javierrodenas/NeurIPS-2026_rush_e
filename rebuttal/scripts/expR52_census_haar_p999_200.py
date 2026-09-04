#!/usr/bin/env python3
"""expR52 (review-response, A1): the vision census under the 2x2 of {null construction} x {statistic},
200 replicates, census cache for every dataset (as expR39c).

  --null gauss : Gaussian coefficients recombined with the real singular values (expR39c/expR40b)
  --null haar  : Haar-rotated coefficients, exact sample spectrum (expR46 `haar` branch)
  --stat sup   : supremum of the four-point defect over 5e5 sampled quadruples per seed (expR39c)
  --stat p999  : 99.9th percentile of the same defects (expR40b)

Seeds: null replicate rep uses RandomState(300+rep) for BOTH constructions (the brief fixes 300+rep;
expR46 used 700+rep for its Haar branch), real value over quadruple seeds 0..9, each replicate over
seeds 0..4. Per cell: raw statistic +- s.d. over quadruple seeds, null mean and s.d., excess,
r_above = #{null > real}, p_left = (1 + #{null <= real})/201. At merge time: Benjamini-Hochberg across
the 72 cells (p_bh, genuine_bh = p_bh <= 0.05) and the uncorrected genuine = p_left <= 0.05.

Output: expR52_census_haar_p999_200.csv (the census of record), expR54_census_haar_sup_200.csv
(--null haar --stat sup). Parallel by dataset with --part TAG; --merge assembles the parts.
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
NAMES = {("haar","p999"): "expR52_census_haar_p999_200", ("haar","sup"): "expR54_census_haar_sup_200",
         ("gauss","p999"): "expR52_census_gauss_p999_200", ("gauss","sup"): "expR52_census_gauss_sup_200"}

def make_delta(stat):
    def delta_norm(X, n_seeds):
        D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D); out = []
        for s in range(n_seeds):
            rng = np.random.RandomState(s)
            i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
            ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
            S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
            dfc = (S[:,2]-S[:,1])/2
            out.append((dfc.max() if stat == "sup" else np.percentile(dfc, 99.9))/diam)
        return float(np.mean(out)), float(np.std(out))
    return delta_norm

def svd(M): mu = M.mean(0); U, S, Vt = np.linalg.svd(M-mu, full_matrices=False); return mu, U, S, Vt
def null_gauss(M, rep):
    mu, U, S, Vt = svd(M); rng = np.random.RandomState(300+rep)
    G = rng.randn(len(M), len(S)).astype(np.float32); G /= G.std(0, keepdims=True)*np.sqrt(len(M)); return (G*S)@Vt+mu
def null_haar(M, rep):
    mu, U, S, Vt = svd(M); rng = np.random.RandomState(300+rep)
    Z = rng.randn(len(M), len(M)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt+mu).astype(np.float32)
NULLS = {"gauss": null_gauss, "haar": null_haar}

def cents(m, ds):
    d = np.load(CACHE/f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

def bh(p):
    p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p); ranked = p[order]*n/np.arange(1, n+1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty(n); out[order] = np.minimum(adj, 1.0); return out

def run(null, stat, part, datasets, models):
    delta_norm = make_delta(stat); nf = NULLS[null]; base = NAMES[(null, stat)]
    csv_path = OUT/f"{base}.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"]) for r in rows}
    for ds in datasets:
        for m in models:
            if (m, ds) in done: continue
            t0 = time.time(); C = cents(m, ds)
            dr, dr_sd = delta_norm(C, 10)
            nulls = np.array([delta_norm(nf(C, r), 5)[0] for r in range(N_REP)])
            nm, nsd = float(nulls.mean()), float(nulls.std(ddof=1))
            r_above = int((nulls > dr).sum()); p_left = (1 + int((nulls <= dr).sum()))/(N_REP+1)
            rows.append(dict(model=m, dataset=ds, null=null, stat=stat, n=len(C), d=C.shape[1], delta=dr, delta_sd=dr_sd,
                             null_mean=nm, null_sd=nsd, excess=dr-nm, z=(dr-nm)/max(np.sqrt(nsd**2+dr_sd**2), 1e-9),
                             frac_null_above=r_above/N_REP, r_above=r_above, p_left=p_left, time_s=time.time()-t0))
            print(f"{m:9s} {ds:12s} {null}/{stat} exc {dr-nm:+.4f} r {r_above}/{N_REP} p {p_left:.4f} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part} ({null}/{stat})")

def merge(null, stat):
    base = NAMES[(null, stat)]
    parts = sorted(glob.glob(str(OUT/f"{base}.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model","dataset"], keep="first")
    df["_d"] = df.dataset.map({d: i for i, d in enumerate(DATASETS)}); df["_m"] = df.model.map({m: i for i, m in enumerate(MODELS)})
    df = df.sort_values(["_d","_m"]).drop(columns=["_d","_m"]); assert len(df) == 72, len(df)
    df["genuine"] = df.p_left <= 0.05; df["p_bh"] = bh(df.p_left.values); df["genuine_bh"] = df.p_bh <= 0.05
    df.to_csv(OUT/f"{base}.csv", index=False)
    top = df[df.dataset.isin(["imagenet","cifar100"])]; rest = df[~df.dataset.isin(["imagenet","cifar100"])]
    print(f"SUMMARY {base}: sign-neg {int((df.excess<0).sum())}/72 | genuine {int(df.genuine.sum())}/72 (BH {int(df.genuine_bh.sum())}/72) "
          f"| IN+C100 BH {int(top.genuine_bh.sum())}/24 | other four BH {int(rest.genuine_bh.sum())}/48")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", default="haar", choices=["gauss","haar"]); ap.add_argument("--stat", default="p999", choices=["sup","p999"])
    ap.add_argument("--part", default=None); ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--merge", action="store_true")
    A = ap.parse_args()
    if A.merge: merge(A.null, A.stat)
    else:
        assert A.part, "--part TAG required"; run(A.null, A.stat, A.part, A.datasets, A.models)
