#!/usr/bin/env python3
"""expR57 (review-response, A3): the census on COSINE geometry, Haar x p99.9, 200 replicates.

Geometry (fixed here): centroids L2-normalized; distance = spherical geodesic arccos(cos), as the angular
xi table (expR38) uses. Null: the Haar construction (exact sample spectrum, seeds 300+rep) built on the
NORMALIZED cloud, then re-normalized row-wise before the geodesic distances are taken (as expR38).
Statistic: 99.9th percentile of the four-point defect over 5e5 sampled quadruples per seed (real: seeds
0..9; each replicate: seeds 0..4 for vision, 0..2 for text). Stored per cell: raw, null mean/s.d.,
excess, r_above, p_left; Benjamini-Hochberg at merge (72 vision cells; 15 text models).

Vision: 72 cells from the census cache (--part by dataset). Text: the 15 padding-free embeddings cached by
expR53 (results/text_cache). Outputs: expR57_census_cosine_haar_p999_200.csv, expR57_text_cosine_haar_p999_200.csv.
"""
import os, sys, time, argparse, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
CACHE = ROOT/"results/practical_tasks_cache"; TCACHE = ROOT/"results/text_cache"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
DATASETS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]
TEXT = ["gpt2","gpt2_m","gpt2_l","gpt2_xl","pythia_410m","pythia_1b","pythia_2b8","olmo_1b","bge_base","bge_large","gte_base","gte_large","gte_qwen2","e5_base","e5_large"]
N_REP = 200

def normalize(X): return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
def geo_D(Xn):
    G = np.clip(Xn @ Xn.T, -1.0, 1.0); D = np.arccos(G); np.fill_diagonal(D, 0.0); return D
def delta_p999(Xn, n_seeds):
    D = geo_D(Xn); diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s); i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(np.percentile((S[:,2]-S[:,1])/2, 99.9)/diam)
    return float(np.mean(out)), float(np.std(out))
def null_haar_normalized(Xn, rep):
    mu = Xn.mean(0); U, S, Vt = np.linalg.svd(Xn-mu, full_matrices=False); rng = np.random.RandomState(300+rep)
    Z = rng.randn(len(Xn), len(Xn)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return normalize(((Q[:, :len(S)]*S)@Vt+mu).astype(np.float32))
def bh(p):
    p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p); ranked = p[order]*n/np.arange(1, n+1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty(n); out[order] = np.minimum(adj, 1.0); return out
def cell(X, n_seed_null):
    Xn = normalize(X.astype(np.float32)); dr, dr_sd = delta_p999(Xn, 10)
    nulls = np.array([delta_p999(null_haar_normalized(Xn, r), n_seed_null)[0] for r in range(N_REP)])
    nm, nsd = float(nulls.mean()), float(nulls.std(ddof=1)); r_above = int((nulls > dr).sum()); p_left = (1 + int((nulls <= dr).sum()))/(N_REP+1)
    return dict(delta=dr, delta_sd=dr_sd, null_mean=nm, null_sd=nsd, excess=dr-nm, z=(dr-nm)/max(np.sqrt(nsd**2+dr_sd**2),1e-9),
                frac_null_above=r_above/N_REP, r_above=r_above, p_left=p_left)
def cents(m, ds):
    d = np.load(CACHE/f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])

def run_vision(part, datasets, models):
    csv_path = OUT/f"expR57_census_cosine_haar_p999_200.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"]) for r in rows}
    for ds in datasets:
        for m in models:
            if (m, ds) in done: continue
            t0 = time.time(); res = cell(cents(m, ds), 5)
            rows.append(dict(model=m, dataset=ds, metric="cosine_geodesic", null="haar", stat="p999", **res, time_s=time.time()-t0))
            print(f"{m:9s} {ds:12s} cos exc {res['excess']:+.4f} r {res['r_above']}/{N_REP} p {res['p_left']:.4f} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done vision part {part}")
def run_text(part, models):
    csv_path = OUT/f"expR57_text_cosine_haar_p999_200.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {r["model"] for r in rows}
    for m in models:
        if m in done: continue
        f = TCACHE/f"{m}_photo_bs1.npz"
        if not f.exists(): print(f"{m}: no cached embeddings (run expR53 first)"); continue
        t0 = time.time(); res = cell(np.load(f)["X"], 3)
        rows.append(dict(model=m, metric="cosine_geodesic", null="haar", stat="p999", **res, time_s=time.time()-t0))
        print(f"{m:12s} cos exc {res['excess']:+.4f} r {res['r_above']}/{N_REP} p {res['p_left']:.4f} ({rows[-1]['time_s']:.0f}s)")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done text part {part}")
def merge():
    for base, key, order in (("expR57_census_cosine_haar_p999_200", ["model","dataset"], None), ("expR57_text_cosine_haar_p999_200", ["model"], TEXT)):
        parts = sorted(glob.glob(str(OUT/f"{base}.part_*.csv")))
        if not parts: continue
        df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=key)
        if order: df["_o"] = df.model.map({m: i for i, m in enumerate(order)}); df = df.sort_values("_o").drop(columns=["_o"])
        else:
            df["_d"] = df.dataset.map({d: i for i, d in enumerate(DATASETS)}); df["_m"] = df.model.map({m: i for i, m in enumerate(MODELS)})
            df = df.sort_values(["_d","_m"]).drop(columns=["_d","_m"])
        df["genuine"] = df.p_left <= 0.05; df["p_bh"] = bh(df.p_left.values); df["genuine_bh"] = df.p_bh <= 0.05
        df.to_csv(OUT/f"{base}.csv", index=False)
        print(f"SUMMARY {base}: {len(df)} rows | sign-neg {int((df.excess<0).sum())} | genuine {int(df.genuine.sum())} (BH {int(df.genuine_bh.sum())})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default=None); ap.add_argument("--text", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=DATASETS); ap.add_argument("--models", nargs="+", default=None); ap.add_argument("--merge", action="store_true")
    A = ap.parse_args()
    if A.merge: merge()
    elif A.text: run_text(A.part, A.models or TEXT)
    else: run_vision(A.part, A.datasets, A.models or MODELS)
