#!/usr/bin/env python3
"""expR73 (fourth review): centroid bootstrap of CIFAR-100 and DTD under the census of record (Haar x p99.9 x 200 replicates).

For each of the 12 backbones and each of the two transfer sets: 30 resamples (RandomState(b)) of the cached training images per
class with replacement at the census image budget (the full cached split, as expR52 builds the centroids), centroids rebuilt, then
the record protocol: real p99.9 statistic over 10 quadruple seeds of 500k quadruples, and 200 Haar null replicates (seeds 300+rep,
5 quadruple seeds each, as expR59). Row b = -1 is the unresampled reference under the same protocol. Stored per row: delta, null
mean, excess; the summary (--merge) gives the bootstrap mean and s.d. of the excess and the fraction of sign-negative resamples per
cell. Output: expR73_transfer_bootstrap_record.csv, expR73_transfer_bootstrap_record_summary.csv.
Usage: python expR73_transfer_bootstrap_record.py --part <name> --models m1 m2 ... [--datasets cifar100 dtd]; then --merge."""
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
DATASETS = ["cifar100", "dtd"]; N_BOOT, N_REP = 30, 200

def delta_p999(X, n_seeds):
    D = squareform(pdist(X, "euclidean")); diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s); i, j, k, l = (rng.randint(0, n, 500_000) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1)
        out.append(np.percentile((S[:,2]-S[:,1])/2, 99.9)/diam)
    return float(np.mean(out)), float(np.std(out))
def null_haar(M, rep):
    mu = M.mean(0); U, S, Vt = np.linalg.svd(M-mu, full_matrices=False); rng = np.random.RandomState(300+rep)
    Z = rng.randn(len(M), len(M)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt+mu).astype(np.float32)
def excess(C):
    dr, dr_sd = delta_p999(C, 10); nulls = np.array([delta_p999(null_haar(C, r), 5)[0] for r in range(N_REP)])
    return dr, dr_sd, float(nulls.mean()), float(nulls.std(ddof=1)), dr-float(nulls.mean()), int((nulls > dr).sum())

def run(part, models, datasets):
    csv_path = OUT/f"expR73_transfer_bootstrap_record.part_{part}.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["dataset"], r["b"]) for r in rows}
    for ds in datasets:
        for m in models:
            d = np.load(CACHE/f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
            idx = [np.where(y==c)[0] for c in range(int(y.max())+1)]
            for b in range(-1, N_BOOT):
                if (m, ds, b) in done: continue
                t0 = time.time()
                if b < 0: C = np.stack([X[ii].mean(0) for ii in idx])
                else:
                    rng = np.random.RandomState(b); C = np.stack([X[rng.choice(ii, size=len(ii), replace=True)].mean(0) for ii in idx])
                dr, dr_sd, nm, nsd, ex, r_above = excess(C)
                rows.append(dict(model=m, dataset=ds, b=b, n_classes=len(idx), n_per_class=int(np.mean([len(ii) for ii in idx])), delta=dr, delta_sd=dr_sd, null_mean=nm, null_sd=nsd, excess=ex, r_above=r_above, n_rep=N_REP, time_s=time.time()-t0))
                print(f"{m:9s} {ds:8s} b{b:3d} delta {dr:.4f} null {nm:.4f} exc {ex:+.4f} r {r_above}/{N_REP} ({rows[-1]['time_s']:.0f}s)")
                pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Done part {part}")

def merge():
    parts = sorted(glob.glob(str(OUT/"expR73_transfer_bootstrap_record.part_*.csv")))
    df = pd.concat([pd.read_csv(p) for p in parts]).drop_duplicates(subset=["model","dataset","b"]).sort_values(["dataset","model","b"])
    df.to_csv(OUT/"expR73_transfer_bootstrap_record.csv", index=False)
    s = []
    for ds in DATASETS:
        for m in MODELS:
            g = df[(df.model==m)&(df.dataset==ds)&(df.b>=0)]; ref = df[(df.model==m)&(df.dataset==ds)&(df.b<0)]
            if len(g)==0: continue
            s.append(dict(model=m, dataset=ds, n_boot=len(g), n_per_class=int(g.n_per_class.iloc[0]), excess_ref=float(ref.excess.iloc[0]) if len(ref) else np.nan, excess_boot_mean=g.excess.mean(),
                          excess_boot_sd=g.excess.std(ddof=1), delta_boot_sd=g.delta.std(ddof=1), frac_boot_negative=(g.excess<0).mean(), frac_boot_below_all=(g.r_above==N_REP).mean()))
            print(f"{m:9s} {ds:8s} ref {s[-1]['excess_ref']:+.4f} | boot mean {s[-1]['excess_boot_mean']:+.4f} sd {s[-1]['excess_boot_sd']:.4f} | frac neg {s[-1]['frac_boot_negative']:.2f}")
    pd.DataFrame(s).to_csv(OUT/"expR73_transfer_bootstrap_record_summary.csv", index=False); print("Done expR73 merge")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--part", default=None); ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    if A.merge: merge()
    else:
        assert A.part; run(A.part, A.models, A.datasets)
