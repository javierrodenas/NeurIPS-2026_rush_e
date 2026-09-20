#!/usr/bin/env python3
"""expR66c (author's decision of 2026-09-20: the record is the centered Haar null): the joint sensitivity of the genuine count,
exactly expR66 with the null's Gaussian matrix column-centered before the QR (Q orthogonal to the all-ones vector, as in
expR75) and the record read from expR75_census_centered_haar.csv. 30 centroid resamples per cell, p99.9 over 10 quadruple
seeds, 50 null replicates (5 seeds each), BH over the 72 cells per resample. Output: expR66c_joint_sensitivity.csv (cell x
resample) and expR66c_joint_sensitivity_summary.csv (per cell: z_joint, joint_genuine, n_boot_genuine, boot_bh_genuine).
    python expR66c_joint_sensitivity.py --part <k>   (k in 0..11, the parts of expR66);  python expR66c_joint_sensitivity.py --merge"""
import os, sys, time, argparse
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool")); sys.path.insert(0, str(HERE))
import calibrated_delta as cd
import expR66_joint_sensitivity as R66
OUT, CACHE, MODELS, DS, PARTS, N_BOOT, N_REP = R66.OUT, R66.CACHE, R66.MODELS, R66.DS, R66.PARTS, R66.N_BOOT, R66.N_REP

def haarnull_centered(C, rep):
    mu = C.mean(0); U, S, Vt = np.linalg.svd(C - mu, full_matrices=False)
    rng = np.random.RandomState(300 + rep); Z = rng.randn(len(C), len(C)); Z -= Z.mean(0, keepdims=True)
    Q, R = np.linalg.qr(Z); Q = Q * np.sign(np.diag(R))
    return ((Q[:, :len(S)] * S) @ Vt + mu).astype(np.float32)
cd.haarnull = haarnull_centered   # cd.census looks the null up by name at call time

def run_part(k):
    f = OUT / f"expR66c_joint_sensitivity.part_{k}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []
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

def merge():
    import glob
    parts = sorted(glob.glob(str(OUT / "expR66c_joint_sensitivity.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model", "dataset", "b"]).sort_values(["dataset", "model", "b"])
    df.to_csv(OUT / "expR66c_joint_sensitivity.csv", index=False)
    rec = pd.read_csv(OUT / "expR75_census_centered_haar.csv").set_index(["model", "dataset"])
    boot = df[df.b >= 0]; nb = boot.b.max() + 1
    bh_gen = {}
    for b in range(nb):
        g = boot[boot.b == b].set_index(["model", "dataset"]); p = g.p_left.values; q = R66.bh(p)
        for k, ok in zip(g.index, q <= 0.05): bh_gen.setdefault(k, []).append(bool(ok))
    out = []
    for (m, ds), g in boot.groupby(["model", "dataset"]):
        r = rec.loc[(m, ds)]; sd_boot = float(g.excess.std(ddof=1)); sd_null = float(r["null_sd"]); sd_est = float(r["delta_sd"]); ex = float(r["excess"])
        zj = ex / max(np.sqrt(sd_null**2 + sd_boot**2 + sd_est**2), 1e-9); nbg = int(sum(bh_gen[(m, ds)]))
        out.append(dict(model=m, dataset=ds, excess=ex, sd_null=sd_null, sd_boot=sd_boot, sd_est=sd_est, z_joint=zj, joint_genuine=bool(zj <= -2), genuine_bh_record=bool(r["genuine_bh"]),
                        n_boot=int(len(g)), n_boot_genuine=nbg, boot_bh_genuine=bool(nbg >= 27), boot_excess_mean=float(g.excess.mean()), boot_frac_negative=float((g.excess < 0).mean())))
    S = pd.DataFrame(out).sort_values(["dataset", "model"]); S.to_csv(OUT / "expR66c_joint_sensitivity_summary.csv", index=False)
    top = S[S.dataset.isin(["imagenet", "cifar100"])]
    print(f"SUMMARY expR66c: record {int(S.genuine_bh_record.sum())}/72 ({int(top.genuine_bh_record.sum())}/24) | z_joint<=-2: {int(S.joint_genuine.sum())}/72 ({int(top.joint_genuine.sum())}/24) | genuine in >=27/30 resamples: {int(S.boot_bh_genuine.sum())}/72 ({int(top.boot_bh_genuine.sum())}/24)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", type=int); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else: run_part(A.part)
