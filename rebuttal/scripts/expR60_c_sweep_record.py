#!/usr/bin/env python3
"""expR60 -- the class-count sweep of exp19 (Table B3 / Appendix "Class-count vs hierarchy depth") re-run under the
census of record: Haar spectrum-matched null x 99.9th-percentile statistic x 200 replicates (calibrated_delta.py,
the same code that reproduces Table 1). Same class subsets as exp19 (random: RandomState(seed).choice; WordNet-coherent:
greedy min mean WordNet distance from a random seed class), same seeds per C (5 for C<=200, 2 for C=500, 1 for C=1000),
same three table models. The NC advantage columns are copied from exp19_c_sweep.csv (the task pipeline is unchanged).

    python expR60_c_sweep_record.py --part <model>_<mode>     # one worker
    python expR60_c_sweep_record.py --merge                   # merge the parts -> expR60_c_sweep_record.csv
"""
import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "iclr2027" / "tool"))
import calibrated_delta as cd
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))
CACHE = ROOT / "results/practical_tasks_cache"
WND = np.load(ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy")
MODELS = ["dinov2_l", "dinov2_g", "clip_l"]
CS = [10, 20, 50, 100, 200, 500, 1000]
N_REP = 200

def coherent_subset(C, seed):
    rng = np.random.RandomState(seed)
    cur = [int(rng.randint(1000))]; cand = set(range(1000)) - set(cur)
    while len(cur) < C:
        cl = np.array(sorted(cand)); best = cl[np.argmin(WND[np.ix_(cl, cur)].mean(1))]
        cur.append(int(best)); cand.discard(int(best))
    return np.array(cur)

def configs(model, mode):
    for C in CS:
        if C == 1000 and mode != "random": continue
        seeds = range(5) if C <= 200 else range(2) if C < 1000 else range(1)
        for seed in seeds: yield C, seed

def run_part(part):
    model, mode = part.rsplit("_", 1)
    f = OUT / f"expR60_c_sweep_record.part_{part}.csv"
    rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["C"], r["seed"]) for r in rows}
    d = np.load(CACHE / f"{model}_imagenet_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    cents = np.stack([X[y == c].mean(0) for c in range(1000)])
    for C, seed in configs(model, mode):
        if (C, seed) in done: continue
        t0 = time.time()
        cls = np.arange(1000) if C == 1000 else np.random.RandomState(seed).choice(1000, C, replace=False) if mode == "random" else coherent_subset(C, seed)
        c = cd.census(cents[cls], N_REP, "haar", "p999")
        rows.append(dict(model=model, C=C, mode=mode, seed=seed, delta_999=c["delta"], delta_sd=c["delta_sd"], null_mean=c["null_mean"], null_sd=c["null_sd"],
                         excess=c["excess"], r_above=c["r_above"], p_left=c["p_left"], time_s=time.time() - t0))
        pd.DataFrame(rows).to_csv(f, index=False)
        print(f"{model:9s} C={C:4d} {mode:8s} s{seed} | d999 {c['delta']:.4f} null {c['null_mean']:.4f} exc {c['excess']:+.4f} r={c['r_above']} p={c['p_left']:.3f} | {time.time()-t0:.0f}s")
    print(f"PART DONE {part}")

def merge():
    parts = sorted(OUT.glob("expR60_c_sweep_record.part_*.csv")); df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    e19 = pd.read_csv(OUT / "exp19_c_sweep.csv")[["model", "C", "mode", "seed", "delta", "excess", "nc_adv_pp", "nc_R", "nc_H"]]
    e19 = e19.rename(columns={"delta": "delta_sup_exp19", "excess": "excess_sup_exp19"})
    df = df.merge(e19, on=["model", "C", "mode", "seed"], how="left").sort_values(["model", "C", "mode", "seed"])
    df.to_csv(OUT / "expR60_c_sweep_record.csv", index=False)
    print(f"merged {len(df)} rows ({len(parts)} parts); missing NC: {int(df.nc_adv_pp.isna().sum())}")
    for m in MODELS:
        for C in CS:
            for mode in ("random", "coherent"):
                g = df[(df.model == m) & (df.C == C) & (df["mode"] == mode)]
                if len(g): print(f"  {m:9s} C={C:4d} {mode:8s}: exc {g.excess.mean():+.4f} (sup-exp19 {g.excess_sup_exp19.mean():+.4f}) r={g.r_above.mean():.0f} p<=0.05 in {int((g.p_left<=0.05).sum())}/{len(g)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part"); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else: run_part(A.part)
