#!/usr/bin/env python3
"""expR84 (tenth review, CPU): the within-model ceiling of the dendrogram agreements.

For each of the 12 ImageNet backbones, the 30 bootstrap centroid sets of expR59 (RandomState(b), the 100 cached training images per
class resampled with replacement, centroids rebuilt) are turned into dendrograms under the selected configuration of the treemap
(cosine distances, average linkage, ImageNet; asserted from exp23_config_diagnostics.json as in the appendix generator), and every
pair of resamples of the same model is compared with the three measures of expR58 (ARI at the 30-cluster cut, Pearson correlation of
the cophenetic distances, agreement on 10^4 random class triplets with RandomState(0), the same triplets as expR58). The mean over the
435 pairs is the within-model ceiling of each measure: the agreement two trees of the same model reach under resampling noise, the
reference against which the cross-model 0.77/0.76 triplet agreement of Table q6 is read. The unresampled reference set against each
resample is stored too (kind ref_vs_boot). Output: expR84_tree_ceiling.csv (pairs) and expR84_tree_ceiling_summary.csv (per model:
mean, s.d. and min per measure over the bootstrap pairs; plus the ALL row: mean and min over models).
Usage: python expR84_tree_ceiling.py [--models m1 m2 ...] ; --merge"""
import os, sys, time, json, argparse, itertools, glob
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import cophenet, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import expR58_treemap_cutfree as R58
OUT = R58.OUT; CACHE = R58.CACHE; MODELS = R58.MODELS; N_BOOT = 30; CUT = R58.CUT["imagenet"]

def selected_config():
    diag = json.load(open(OUT / "exp23_config_diagnostics.json")); adm = {k: v for k, v in diag.items() if k.startswith("imagenet|") and v["maxfrac"] <= 0.5}
    metric, link = max(adm, key=lambda k: adm[k]["cpcc"]).split("|")[1:]; assert (metric, link) == ("cosine", "average"), (metric, link)
    return metric, link

def resamples(m):
    d = np.load(CACHE / f"{m}_imagenet_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    idx = [np.where(y == c)[0] for c in range(int(y.max()) + 1)]
    yield -1, np.stack([X[ii].mean(0) for ii in idx])
    for b in range(N_BOOT):
        rng = np.random.RandomState(b); yield b, np.stack([X[rng.choice(ii, size=len(ii), replace=True)].mean(0) for ii in idx])   # as expR59

def run(models):
    metric, link = selected_config(); rows = []
    for m in models:
        f = OUT / f"expR84_tree_ceiling.part_{m}.csv"
        if f.exists(): print("skip", m); continue
        t0 = time.time(); coph, cuts, fm = {}, {}, {}; T = None
        for b, C in resamples(m):
            Z = R58.tree(C, metric, link); cv = cophenet(Z); coph[b] = cv; cuts[b] = fcluster(Z, CUT, criterion="maxclust")
            if T is None: T = R58.triplets(len(C), np.random.RandomState(0))
            fm[b] = R58.first_merge(squareform(cv), T)
        rows_m = []
        for a, b in itertools.combinations(range(N_BOOT), 2):
            rows_m.append(dict(model=m, kind="boot_pair", a=a, b=b, ari_cut=adjusted_rand_score(cuts[a], cuts[b]), coph_corr=float(np.corrcoef(coph[a], coph[b])[0, 1]), triplet_agree=float((fm[a] == fm[b]).mean())))
        for b in range(N_BOOT):
            rows_m.append(dict(model=m, kind="ref_vs_boot", a=-1, b=b, ari_cut=adjusted_rand_score(cuts[-1], cuts[b]), coph_corr=float(np.corrcoef(coph[-1], coph[b])[0, 1]), triplet_agree=float((fm[-1] == fm[b]).mean())))
        pd.DataFrame(rows_m).to_csv(f, index=False); rows += rows_m
        d = pd.DataFrame([r for r in rows_m if r["kind"] == "boot_pair"])
        print(f"{m:9s} boot pairs {len(d)}: ARI@30 {d.ari_cut.mean():.3f}  coph {d.coph_corr.mean():.3f}  triplets {d.triplet_agree.mean():.3f} (min {d.triplet_agree.min():.3f})  ({time.time() - t0:.0f}s)")
    print("Done", models)

def merge():
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / "expR84_tree_ceiling.part_*.csv")))]).drop_duplicates(subset=["model", "kind", "a", "b"]); df.to_csv(OUT / "expR84_tree_ceiling.csv", index=False)
    S = []
    for m in MODELS:
        d = df[(df.model == m) & (df.kind == "boot_pair")]; r = df[(df.model == m) & (df.kind == "ref_vs_boot")]
        if not len(d): continue
        row = dict(model=m, n_pairs=len(d))
        for k in ("ari_cut", "coph_corr", "triplet_agree"): row.update({f"{k}_mean": float(d[k].mean()), f"{k}_sd": float(d[k].std()), f"{k}_min": float(d[k].min()), f"{k}_ref_mean": float(r[k].mean()) if len(r) else np.nan})
        S.append(row)
    S = pd.DataFrame(S); allrow = dict(model="ALL", n_pairs=int(S.n_pairs.sum()))
    for k in ("ari_cut", "coph_corr", "triplet_agree"): allrow.update({f"{k}_mean": float(S[f"{k}_mean"].mean()), f"{k}_sd": float(S[f"{k}_mean"].std()), f"{k}_min": float(S[f"{k}_mean"].min()), f"{k}_ref_mean": float(S[f"{k}_ref_mean"].mean())})
    S = pd.concat([S, pd.DataFrame([allrow])]); S.to_csv(OUT / "expR84_tree_ceiling_summary.csv", index=False)
    print(S.round(3).to_string()); print(f"SUMMARY expR84: within-model triplet ceiling mean over models {allrow['triplet_agree_mean']:.3f}, lowest backbone {allrow['triplet_agree_min']:.3f}; models done {len(S) - 1} of 12")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=MODELS); ap.add_argument("--merge", action="store_true"); A = ap.parse_args()
    merge() if A.merge else run(A.models)
