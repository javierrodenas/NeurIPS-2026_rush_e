#!/usr/bin/env python3
"""expR75 (fifth review, B.2): the vision census under the record with a CENTERED Haar null.

The record (expR52) draws X' = Q S V^T + mu with Q the Q-factor of a Gaussian n x n matrix; the columns of Q are not
orthogonal to the all-ones vector, so the centered spectrum of X' matches the real one only up to a centering term of
order 1/sqrt(n). Here the columns of Z are centered before the QR, so Q is orthogonal to 1, X' has exactly the mean mu
and its centered spectrum is exact. Everything else is the record: 99.9th-percentile statistic over 10 quadruple seeds,
200 null replicates with 5 seeds each, BH over the 72 cells. Output: expR75_census_centered_haar.csv and
expR75_census_centered_haar_summary.csv (verdict changes against expR52).
Usage: python expR75_census_centered_haar.py --part <name> --datasets d1 ... [--models ...]; then --merge."""
import os, sys, time, argparse, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import expR52_census_haar_p999_200 as R52
OUT = R52.OUT; MODELS = R52.MODELS; DATASETS = R52.DATASETS; N_REP = 200

def null_haar_centered(M, rep):
    mu, U, S, Vt = R52.svd(M); rng = np.random.RandomState(300+rep)
    Z = rng.randn(len(M), len(M)); Z -= Z.mean(0, keepdims=True)            # columns orthogonal to 1
    Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt+mu).astype(np.float32)

def run(part, datasets, models):
    delta_norm = R52.make_delta("p999"); csv_path = OUT/f"expR75_census_centered_haar.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"]) for r in rows}
    for ds in datasets:
        for m in models:
            if (m, ds) in done: continue
            t0 = time.time(); C = R52.cents(m, ds); dr, dr_sd = delta_norm(C, 10)
            nulls = np.array([delta_norm(null_haar_centered(C, r), 5)[0] for r in range(N_REP)])
            nm, nsd = float(nulls.mean()), float(nulls.std(ddof=1)); r_above = int((nulls > dr).sum()); p_left = (1 + int((nulls <= dr).sum()))/(N_REP+1)
            rows.append(dict(model=m, dataset=ds, null="haar_centered", stat="p999", n=len(C), d=C.shape[1], delta=dr, delta_sd=dr_sd, null_mean=nm, null_sd=nsd, excess=dr-nm,
                             z=(dr-nm)/max(np.sqrt(nsd**2+dr_sd**2), 1e-9), r_above=r_above, p_left=p_left, time_s=time.time()-t0))
            print(f"{m:9s} {ds:12s} exc {dr-nm:+.4f} r {r_above}/{N_REP} p {p_left:.4f} ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part}")

def merge():
    parts = sorted(glob.glob(str(OUT/"expR75_census_centered_haar.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model","dataset"], keep="first")
    df["_d"] = df.dataset.map({d: i for i, d in enumerate(DATASETS)}); df["_m"] = df.model.map({m: i for i, m in enumerate(MODELS)})
    df = df.sort_values(["_d","_m"]).drop(columns=["_d","_m"]); assert len(df) == 72, len(df)
    df["genuine"] = df.p_left <= 0.05; df["p_bh"] = R52.bh(df.p_left.values); df["genuine_bh"] = df.p_bh <= 0.05
    rec = pd.read_csv(OUT/"expR52_census_haar_p999_200.csv").set_index(["model","dataset"])
    df["record_excess"] = [rec.loc[(m, d), "excess"] for m, d in zip(df.model, df.dataset)]; df["record_genuine_bh"] = [bool(rec.loc[(m, d), "genuine_bh"]) for m, d in zip(df.model, df.dataset)]
    df["verdict_changed"] = df.genuine_bh != df.record_genuine_bh
    df.to_csv(OUT/"expR75_census_centered_haar.csv", index=False)
    ch = df[df.verdict_changed]
    s = dict(n_cells=len(df), genuine_bh=int(df.genuine_bh.sum()), record_genuine_bh=int(df.record_genuine_bh.sum()), verdict_changes=int(df.verdict_changed.sum()),
             changed_cells=";".join(f"{m}/{d}:{'gained' if g else 'lost'}" for m, d, g in zip(ch.model, ch.dataset, ch.genuine_bh)), max_abs_excess_shift=float((df.excess - df.record_excess).abs().max()),
             mean_excess_shift=float((df.excess - df.record_excess).mean()), sign_negative=int((df.excess < 0).sum()))
    pd.DataFrame([s]).to_csv(OUT/"expR75_census_centered_haar_summary.csv", index=False)
    print(f"SUMMARY centered Haar: genuine {s['genuine_bh']}/72 (record {s['record_genuine_bh']}/72), verdict changes {s['verdict_changes']} [{s['changed_cells']}], max |excess shift| {s['max_abs_excess_shift']:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default=None); ap.add_argument("--datasets", nargs="+", default=DATASETS); ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else:
        assert A.part; run(A.part, A.datasets, A.models)
