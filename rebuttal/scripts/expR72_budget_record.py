#!/usr/bin/env python3
"""expR72 (final pass, A5): the quadruple-budget sweep of expR31 under the census of record.

Same 9 cells ({ViT-L, DINOv2-L, CLIP-B} x {ImageNet, CIFAR-100, DTD}), budgets n_quads in {1e4, 5e4, 1e5, 5e5, 1e6, 2e6}:
the record statistic (p99.9 of the sampled defects, 10 quadruple seeds) against 200 Haar null replicates (seeds 300+rep,
5 quadruple seeds) at EVERY budget (calibrated_delta.census with n_quads). Merge: drift of the excess across budgets per cell
against the excess's own s.d. at 5e5 (sqrt(null_sd^2 + delta_sd^2)).

    python expR72_budget_record.py --part <dataset>     # one dataset, three models
    python expR72_budget_record.py --merge              # -> expR72_budget_record.csv, _summary.csv
"""
import os, sys, time, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
import calibrated_delta as cd
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic")); CACHE = ROOT / "results/practical_tasks_cache"
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))
MODELS = ["i21k_l", "dinov2_l", "clip_b"]; DS = ["imagenet", "cifar100", "dtd"]; BUDGETS = [10_000, 50_000, 100_000, 500_000, 1_000_000, 2_000_000]

def cents(m, ds):
    d = np.load(CACHE / f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y == c].mean(0) for c in range(int(y.max()) + 1)])

def run_part(ds):
    f = OUT / f"expR72_budget_record.part_{ds}.csv"; rows = pd.read_csv(f).to_dict("records") if f.exists() else []
    done = {(r["model"], int(r["n_quads"])) for r in rows}
    for m in MODELS:
        C = cents(m, ds)
        for nq in BUDGETS:
            if (m, nq) in done: continue
            t0 = time.time(); c = cd.census(C, 200, "haar", "p999", n_quads=nq)
            rows.append(dict(model=m, dataset=ds, n_quads=nq, n=c["n"], delta_999=c["delta"], delta_sd=c["delta_sd"], null_mean=c["null_mean"], null_sd=c["null_sd"], excess=c["excess"], r_above=c["r_above"], p_left=c["p_left"], time_s=time.time() - t0))
            pd.DataFrame(rows).to_csv(f, index=False); print(f"{m:9s} {ds:9s} n_quads={nq:8d}: d999 {c['delta']:.4f} null {c['null_mean']:.4f} exc {c['excess']:+.4f} r={c['r_above']} ({time.time()-t0:.0f}s)")
    print(f"PART DONE {ds}")

def merge():
    df = pd.concat([pd.read_csv(p) for p in sorted(OUT.glob("expR72_budget_record.part_*.csv"))], ignore_index=True); df.to_csv(OUT / "expR72_budget_record.csv", index=False)
    rec = pd.read_csv(OUT / "expR52_census_haar_p999_200.csv").set_index(["model", "dataset"]); out = []
    for (m, ds), g in df.groupby(["model", "dataset"]):
        g = g.sort_values("n_quads"); ref = g[g.n_quads == 500_000].iloc[0]; sd = float(np.sqrt(ref.null_sd ** 2 + ref.delta_sd ** 2))
        hi = g[g.n_quads >= 100_000]
        out.append(dict(model=m, dataset=ds, excess_5e5=float(ref.excess), excess_record=float(rec.loc[(m, ds), "excess"]), sd_5e5=sd, drift_all=float(g.excess.max() - g.excess.min()), drift_ge1e5=float(hi.excess.max() - hi.excess.min()),
                        drift_ge1e5_over_sd=float((hi.excess.max() - hi.excess.min()) / sd), sign_stable=bool((g.excess < 0).all() or (g.excess > 0).all()), r_min=int(g.r_above.min()), r_max=int(g.r_above.max()),
                        excess_by_budget=" ".join(f"{int(r.n_quads)}:{r.excess:+.4f}" for r in g.itertuples())))
    S = pd.DataFrame(out); S.to_csv(OUT / "expR72_budget_record_summary.csv", index=False); print(S.round(4).to_string())
    for ds in DS:
        s = S[S.dataset == ds]; print(f"{ds}: max drift (>=1e5) {s.drift_ge1e5.max():.4f} = {s.drift_ge1e5_over_sd.max():.2f} s.d.; max drift (all budgets) {s.drift_all.max():.4f}; signs stable {bool(s.sign_stable.all())}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part"); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else run_part(A.part)
