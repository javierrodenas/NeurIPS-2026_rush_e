#!/usr/bin/env python3
"""expR62 (R7 of the restructuring brief) -- the sample-level reading of expR37 (Appendix "Sample-level features") under the
census of record: the instrument on ~1000 stratified training images per cell (CIFAR-100: 10/class; DTD: 22/class; seed 0, the
exact subsets of expR37), all 12 backbones, Haar spectrum-matched null on the sample cloud, 99.9th-percentile statistic, 200
replicates (real: 10 quadruple seeds; replicate: 5), BH across the 24 cells at merge. For the same clouds the raw supremum
delta_norm (10 seeds), the statistic the latent-hyperbolicity papers report, is stored alongside.
Estimator/null code: iclr2027/tool/calibrated_delta.py (the code that reproduces Table 1).

    python expR62_samplelevel_record.py --part <k>   (k in 0..3: models 3k..3k+2, both datasets)
    python expR62_samplelevel_record.py --merge      -> expR62_samplelevel_record.csv
"""
import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "iclr2027" / "tool"))
import calibrated_delta as cd
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic")); CACHE = ROOT / "results/practical_tasks_cache"
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))
MODELS = ["i21k_t", "i21k_s", "i21k_b", "i21k_l", "dinov1_b", "dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g", "clip_b", "clip_l", "siglip_b"]
PER = {"cifar100": 10, "dtd": 22}

def subset(m, ds):
    d = np.load(CACHE / f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    rng = np.random.RandomState(0); idx = []
    for c in range(int(y.max()) + 1):
        ii = np.where(y == c)[0]; idx += list(rng.choice(ii, size=min(PER[ds], len(ii)), replace=False))
    return X[np.array(idx)]

def run_part(k):
    f = OUT / f"expR62_samplelevel_record.part_{k}.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []; done = {(r["model"], r["dataset"]) for r in rows}
    for m in MODELS[3*k:3*k+3]:
        for ds in ("cifar100", "dtd"):
            if (m, ds) in done: continue
            t0 = time.time(); Xs = subset(m, ds)
            c = cd.census(Xs, 200, "haar", "p999"); sup, sup_sd = cd.delta_stat(Xs, 10, "sup")
            rows.append(dict(model=m, dataset=ds, n=int(len(Xs)), d=int(Xs.shape[1]), delta_999=c["delta"], delta_sd=c["delta_sd"], null_mean=c["null_mean"], null_sd=c["null_sd"],
                             excess=c["excess"], z=c["z"], r_above=c["r_above"], p_left=c["p_left"], delta_sup=sup, delta_sup_sd=sup_sd, time_s=time.time() - t0))
            pd.DataFrame(rows).to_csv(f, index=False)
            print(f"{m:9s} {ds:9s} n={len(Xs)} d999 {c['delta']:.4f} null {c['null_mean']:.4f} exc {c['excess']:+.4f} r={c['r_above']} p={c['p_left']:.3f} | sup {sup:.4f} ({time.time()-t0:.0f}s)")
    print(f"PART DONE {k}")

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(adj, 1); return out

def merge():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("expR62_samplelevel_record.part_*.csv"))], ignore_index=True)
    df["order"] = df.model.map({m: i for i, m in enumerate(MODELS)}); df = df.sort_values(["dataset", "order"]).drop(columns="order")
    df["p_bh"] = bh(df.p_left); df["genuine_bh"] = df.p_bh <= 0.05
    old = pd.read_csv(OUT / "expR37_sample_level.csv")[["model", "dataset", "delta", "excess", "z"]].rename(columns={"delta": "delta_sup_expR37", "excess": "excess_gauss_sup_expR37", "z": "z_expR37"})
    df = df.merge(old, on=["model", "dataset"], how="left"); df.to_csv(OUT / "expR62_samplelevel_record.csv", index=False)
    print(f"merged {len(df)} cells; genuine BH {int(df.genuine_bh.sum())}/{len(df)}: {list(df[df.genuine_bh].apply(lambda r: f'{r.model}/{r.dataset}', axis=1))}")
    print(f"raw sup band {df.delta_sup.min():.3f}..{df.delta_sup.max():.3f} (expR37 {df.delta_sup_expR37.min():.3f}..{df.delta_sup_expR37.max():.3f}); d999 {df.delta_999.min():.3f}..{df.delta_999.max():.3f}")
    print(f"excess {df.excess.min():+.4f}..{df.excess.max():+.4f}; |z|<2 in {int((df.z.abs()<2).sum())}/{len(df)}; sign-negative {int((df.excess<0).sum())}/{len(df)}")
    print(df[["model", "dataset", "delta_sup", "delta_999", "excess", "z", "r_above", "p_left", "p_bh"]].to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", type=int); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else run_part(A.part)
