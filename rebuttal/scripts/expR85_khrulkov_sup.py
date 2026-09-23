#!/usr/bin/env python3
"""expR85 (twelfth review, CPU): the published statistic itself, calibrated.

Khrulkov et al. (2020) read delta_rel = 2 delta / diam with delta the exact supremum over all quadruples (min-max product). expR78
reproduced that reading on class-balanced batches of 1500 ResNet-34 features and calibrated the record instrument (99.9th percentile)
on the same clouds. Here the SAME batches (RandomState(t), as expR78) are calibrated with THEIR statistic: the exact delta_rel of the
real batch against the exact delta_rel of the same 200 centered-Haar replicates the record uses (seeds 0..199), giving the excess and
the rank of the supremum reading. Per trial: delta_rel_real, null mean and s.d., excess, r_above (replicates above the real value),
left-tail p = (1 + K - r) / (K + 1). One exact delta_rel takes about 17 s on 1500 points in float32 under an 8-process load (34 s in float64), so a trial costs about an hour; run in shards.
Usage: python expR85_khrulkov_sup.py --dataset cifar10 --trials 0 1 2 3 4 ; python expR85_khrulkov_sup.py --merge"""
import os, sys, time, argparse, glob
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import expR78_khrulkov_replication as R78, expR75_census_centered_haar as R75
OUT = R78.OUT; N_REP = R78.N_REP

def delta_rel_sup32(X):
    """expR78.delta_rel_theirs with the min-max product in float32 and 8-row chunks (the memory-bound step): twice as fast under an
    8-process load; the exact delta differs from the float64 value by about 2e-6 (cifar10 trial 0: 5.533531 vs 5.533529 before the
    2/diam normalization), far below the 3 decimals reported."""
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X)).astype(np.float32); p = 0; row = D[p, :][None, :]; col = D[:, p][:, None]; XY = 0.5 * (row + col - D); n = len(D); maxmin = np.empty_like(XY)
    for i in range(0, n, 8): maxmin[i:i + 8] = np.max(np.minimum(XY[i:i + 8, :, None], XY[None, :, :]), axis=1)
    return 2 * float(np.max(maxmin - XY)) / float(D.max())

def batch(ds, t):
    d = np.load(R78.FEAT / f"resnet34_{ds}.npz"); X = d["features"].astype(np.float64); y = d["labels"]; C = int(y.max()) + 1; per = R78.N_BATCH // C
    rng = np.random.RandomState(t); idx = np.concatenate([rng.choice(np.where(y == c)[0], min(per, int((y == c).sum())), replace=False) for c in range(C)])
    if len(idx) < R78.N_BATCH: idx = np.concatenate([idx, rng.choice(np.setdiff1d(np.arange(len(y)), idx), R78.N_BATCH - len(idx), replace=False)])
    return X[idx]

def run(ds, trials):
    f = OUT / f"expR85_khrulkov_sup.part_{ds}_{trials[0]}-{trials[-1]}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []; done = {int(r["trial"]) for r in rows}
    for t in trials:
        if t in done: continue
        Xb = batch(ds, t); t0 = time.time(); dr = delta_rel_sup32(Xb)
        nulls = np.array([delta_rel_sup32(R75.null_haar_centered(Xb.astype(np.float32), r)) for r in range(N_REP)])
        r_above = int((nulls > dr).sum()); p_left = (1 + N_REP - r_above) / (N_REP + 1)
        rows.append(dict(dataset=ds, trial=t, n=len(Xb), n_rep=N_REP, delta_rel_sup=dr, null_mean=float(nulls.mean()), null_sd=float(nulls.std(ddof=1)), excess_sup=float(dr - nulls.mean()), r_above=r_above, p_left=p_left, time_s=time.time() - t0))
        print(f"{ds:12s} trial {t} delta_rel {dr:.4f} null {nulls.mean():.4f} +- {nulls.std(ddof=1):.4f} excess {dr - nulls.mean():+.4f} r {r_above}/{N_REP} p {p_left:.3f} ({time.time() - t0:.0f}s)")
        pd.DataFrame(rows).to_csv(f, index=False)
    print("PART DONE", ds, trials)

def merge():
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "expR85_khrulkov_sup.part_*.csv")))]).drop_duplicates(subset=["dataset", "trial"]).sort_values(["dataset", "trial"]); df.to_csv(OUT / "expR85_khrulkov_sup.csv", index=False)
    S = df.groupby("dataset").agg(n_trials=("trial", "count"), delta_rel_sup_mean=("delta_rel_sup", "mean"), null_mean=("null_mean", "mean"), excess_sup_mean=("excess_sup", "mean"), excess_sup_sd=("excess_sup", "std"), r_above_mean=("r_above", "mean"), r_above_median=("r_above", "median"), p_left_max=("p_left", "max"), p_left_min=("p_left", "min")).reset_index()
    S.to_csv(OUT / "expR85_khrulkov_sup_summary.csv", index=False); print(S.round(4).to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset"); ap.add_argument("--trials", nargs="+", type=int, default=list(range(R78.N_TRIALS))); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else run(A.dataset, A.trials)
