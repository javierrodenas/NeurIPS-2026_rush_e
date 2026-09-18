#!/usr/bin/env python3
"""expR74 (fifth review, B.1): the decoupling control of the depth test on the 12 ImageNet backbones.

The real hubs (the mean of each WordNet-30 cluster, the frame of expR56) are kept exactly and each cluster's offsets are
rotated by an independent Haar rotation of R^d (10 seeds), which preserves every within-cluster spectrum and the hub
arrangement but destroys the coupling between hubs and offsets. The decoupled cloud then goes through the unchanged
depth test of the record (expR56: anisotropic matched star, hub null, 10 star seeds). If a certified backbone no longer
fires, its verdict came from the coupling rather than from the hub arrangement.
Output: expR74_decoupling.csv (one row per model x seed) and expR74_decoupling_summary.csv (per model: real z from expR56
K=30 aniso, decoupled z mean/sd, fraction of seeds with z <= -2).
Usage: python expR74_decoupling.py --part <name> --models m1 m2 ...; then --merge."""
import os, sys, time, argparse, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expR56_depth_variants as R56
OUT = R56.OUT; MODELS = R56.MODELS; N_SEEDS = 10

def decouple(C, sup, seed):
    rng = np.random.RandomState(50_000 + seed); K = sup.max() + 1; d = C.shape[1]; X = np.empty_like(C)
    for k in range(K):
        m = sup == k; hub = C[m].mean(0); off = C[m] - hub
        Z = rng.randn(d, d); Q, R = np.linalg.qr(Z); Q = Q * np.sign(np.diag(R))
        X[m] = hub + off @ Q      # rows rotated by an independent Haar rotation; the cluster mean (the hub) is unchanged
    return X.astype(np.float32)

def run(part, models):
    csv_path = OUT/f"expR74_decoupling.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["seed"]) for r in rows}
    _, inet = R56.frames(); sup = np.asarray(inet[30]).astype(int)
    for m in models:
        C = R56.cents(m, "imagenet")
        for seed in range(N_SEEDS):
            if (m, seed) in done: continue
            t0 = time.time(); res = R56.depth_test(decouple(C, sup, seed), sup, "aniso")
            rows.append(dict(model=m, seed=seed, K=30, variant="aniso_decoupled", **res, time_s=time.time()-t0))
            print(f"{m:9s} seed {seed} depth {res['depth_excess']:+.4f} z {res['z_depth']:+.2f} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part}")

def merge():
    parts = sorted(glob.glob(str(OUT/"expR74_decoupling.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model","seed"]).sort_values(["model","seed"])
    df.to_csv(OUT/"expR74_decoupling.csv", index=False)
    real = pd.read_csv(OUT/"expR56_depth_variants.csv"); real = real[(real.dataset == "imagenet") & (real.K == 30) & (real.variant == "aniso")].set_index("model")
    s = []
    for m in MODELS:
        g = df[df.model == m]
        if len(g) == 0: continue
        s.append(dict(model=m, n_seeds=len(g), real_z=float(real.loc[m, "z_depth"]), real_certified=bool(real.loc[m, "z_depth"] <= -2), dec_z_mean=g.z_depth.mean(), dec_z_sd=g.z_depth.std(ddof=1), dec_z_min=g.z_depth.min(), dec_z_max=g.z_depth.max(),
                      dec_depth_mean=g.depth_excess.mean(), frac_certified=(g.z_depth <= -2).mean()))
        print(f"{m:9s} real z {s[-1]['real_z']:+.2f} | decoupled z {s[-1]['dec_z_mean']:+.2f} +- {s[-1]['dec_z_sd']:.2f} | certified {s[-1]['frac_certified']:.1f}")
    pd.DataFrame(s).to_csv(OUT/"expR74_decoupling_summary.csv", index=False); print("Done expR74 merge")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default=None); ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else:
        assert A.part; run(A.part, A.models)
