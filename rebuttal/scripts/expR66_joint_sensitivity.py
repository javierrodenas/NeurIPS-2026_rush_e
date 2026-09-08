#!/usr/bin/env python3
"""expR66 (positive-control pass, R11): joint sensitivity of the genuine count.

For every vision cell (12 backbones x 6 datasets, the census cache): 30 centroid resamples (RandomState(b) choice with
replacement of the cached training images per class, exactly as expR59), each read with the record statistic (p99.9,
10 quadruple seeds) against N_REP = 50 Haar null replicates (seeds 300+rep, 5 quadruple seeds; 50 rather than 200 so that
the 30 x 72 censuses fit in hours -- with 50 replicates the smallest attainable p is 1/51 = 0.0196, which passes
Benjamini-Hochberg at the census's genuine count). Row b = -1 is the unresampled cloud under the same 50 replicates.
Merge:
  * z_joint = excess / sqrt(sd_null^2 + sd_boot^2 + sd_est^2) per cell, with excess, sd_null (200 replicates) and
    sd_est (10 quadruple seeds) from the census of record (expR52) and sd_boot the s.d. of the excess over the 30
    resamples; count of cells with z_joint <= -2 (all 72; ImageNet + CIFAR-100).
  * bootstrap-BH: per resample b, BH over the 72 cells on p_b = (1 + #{null <= real}) / 51; a cell is robustly genuine
    when genuine in >= 27 of 30 resamples; count, and the list of record-genuine cells that drop.
Output: expR66_joint_sensitivity.csv (per cell x resample), expR66_joint_sensitivity_summary.csv (per cell), memo numbers.

    python expR66_joint_sensitivity.py --part <k>   (k in 0..11: parts 0-5 = two ImageNet cells each; 6-11 = ten transfer cells each)
    python expR66_joint_sensitivity.py --merge
"""
import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import calibrated_delta as cd
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic")); CACHE = ROOT / "results/practical_tasks_cache"
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))
MODELS = ["i21k_t", "i21k_s", "i21k_b", "i21k_l", "dinov1_b", "dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g", "clip_b", "clip_l", "siglip_b"]
DS = ["imagenet", "cifar100", "cifar10", "dtd", "fashionmnist", "mnist"]
N_BOOT, N_REP = 30, 50
CELLS = [(m, "imagenet") for m in MODELS] + [(m, d) for d in DS[1:] for m in MODELS]
PARTS = [CELLS[2*k:2*k+2] for k in range(6)] + [CELLS[12+10*k:22+10*k] for k in range(6)]   # 6 x 2 ImageNet cells, 6 x 10 transfer cells

def run_part(k):
    f = OUT / f"expR66_joint_sensitivity.part_{k}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["model"], r["dataset"], int(r["b"])) for r in rows}
    for m, ds in PARTS[k]:
        d = np.load(CACHE / f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
        idx = [np.where(y == c)[0] for c in range(int(y.max()) + 1)]
        for b in range(-1, N_BOOT):
            if (m, ds, b) in done: continue
            t0 = time.time()
            if b < 0: C = np.stack([X[ii].mean(0) for ii in idx])
            else:
                rng = np.random.RandomState(b); C = np.stack([X[rng.choice(ii, size=len(ii), replace=True)].mean(0) for ii in idx])
            c = cd.census(C, N_REP, "haar", "p999")
            rows.append(dict(model=m, dataset=ds, b=b, n=c["n"], delta_999=c["delta"], delta_sd=c["delta_sd"], null_mean=c["null_mean"], null_sd=c["null_sd"],
                             excess=c["excess"], r_above=c["r_above"], p_left=c["p_left"], n_rep=N_REP, time_s=time.time() - t0))
            pd.DataFrame(rows).to_csv(f, index=False)
            if b % 10 == 0 or b < 0: print(f"{m:9s} {ds:12s} b={b:3d} exc {c['excess']:+.4f} r={c['r_above']}/{N_REP} ({time.time()-t0:.0f}s)")
    print(f"PART DONE {k}")

def bh(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); r = p[o] * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(r[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(adj, 1); return out

def merge():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("expR66_joint_sensitivity.part_*.csv"))], ignore_index=True)
    df.to_csv(OUT / "expR66_joint_sensitivity.csv", index=False)
    rec = pd.read_csv(OUT / "expR52_census_haar_p999_200.csv").set_index(["model", "dataset"])
    boot = df[df.b >= 0]
    # bootstrap-BH per resample
    gen_b = {}
    for b, g in boot.groupby("b"):
        g = g.set_index(["model", "dataset"]); pb = pd.Series(bh(g.p_left.values), index=g.index); gen_b[b] = pb <= 0.05
    GB = pd.DataFrame(gen_b)                         # cells x resamples
    out = []
    for (m, ds), g in boot.groupby(["model", "dataset"]):
        r = rec.loc[(m, ds)]; sd_boot = float(g.excess.std(ddof=1)); sd_null = float(r["null_sd"]); sd_est = float(r["delta_sd"])
        zj = float(r["excess"]) / max(np.sqrt(sd_null ** 2 + sd_boot ** 2 + sd_est ** 2), 1e-9)
        ng = int(GB.loc[(m, ds)].sum()) if (m, ds) in GB.index else 0
        out.append(dict(model=m, dataset=ds, excess=float(r["excess"]), sd_null=sd_null, sd_boot=sd_boot, sd_est=sd_est, z_joint=zj, joint_genuine=zj <= -2,
                        genuine_bh_record=bool(r["genuine_bh"]), n_boot=int(len(g)), n_boot_genuine=ng, boot_bh_genuine=ng >= 27,
                        boot_excess_mean=float(g.excess.mean()), boot_frac_negative=float((g.excess < 0).mean())))
    S = pd.DataFrame(out).sort_values(["dataset", "model"]); S.to_csv(OUT / "expR66_joint_sensitivity_summary.csv", index=False)
    top = S.dataset.isin(["imagenet", "cifar100"])
    print(f"cells: {len(S)}; record genuine {int(S.genuine_bh_record.sum())}/{len(S)} ({int((S.genuine_bh_record & top).sum())}/24)")
    print(f"z_joint <= -2: {int(S.joint_genuine.sum())}/{len(S)} ({int((S.joint_genuine & top).sum())}/24); record-genuine cells that drop: {list(S[S.genuine_bh_record & ~S.joint_genuine].apply(lambda r: f'{r.model}/{r.dataset}', axis=1))}")
    print(f"bootstrap-BH genuine in >= 27/30: {int(S.boot_bh_genuine.sum())}/{len(S)} ({int((S.boot_bh_genuine & top).sum())}/24); record-genuine cells that drop: {list(S[S.genuine_bh_record & ~S.boot_bh_genuine].apply(lambda r: f'{r.model}/{r.dataset}', axis=1))}; newly genuine: {list(S[~S.genuine_bh_record & S.boot_bh_genuine].apply(lambda r: f'{r.model}/{r.dataset}', axis=1))}")
    print(f"sd_boot: max {S.sd_boot.max():.4f} (ImageNet max {S[S.dataset=='imagenet'].sd_boot.max():.4f}); sd_null max {S.sd_null.max():.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", type=int); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else run_part(A.part)
