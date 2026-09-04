#!/usr/bin/env python3
"""expR58 (review-response, A4): the tree map without a cut.

exp22/exp23 store only the ARI/cophenetic summaries, so the dendrograms are rebuilt here from the same
centroids (census cache, ImageNet 1000 classes and CIFAR-100 100 classes) with the same six configurations
{euclid, cosine} x {average, complete, ward} (cosine-Ward: Ward on the L2-normalized centroids, as exp23).
For every pair of the 12 models and every configuration:
  (i)  cross-model cophenetic correlation: Pearson correlation between the two trees' cophenetic distance
       vectors over all class pairs;
  (ii) triplet agreement: 10^4 random class triplets (RandomState(0)), fraction on which the two trees agree
       about which pair merges first (smallest cophenetic distance);
  plus the ARI at the paper's cut (30 / 20 clusters) for reference.
Summaries per configuration: DINOv2-B/L/G vs the supervised+contrastive block, and within-block means.
Output: expR58_treemap_cutfree.csv (per pair) and expR58_treemap_cutfree_summary.csv.
"""
import os, sys, itertools
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
from scipy.cluster.hierarchy import linkage, cophenet, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
CACHE = ROOT/"results/practical_tasks_cache"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
BLOCK = ["i21k_t","i21k_s","i21k_b","i21k_l","clip_b","clip_l","siglip_b"]; BIG = ["dinov2_b","dinov2_l","dinov2_g"]
CUT = {"imagenet": 30, "cifar100": 20}
N_TRIPLETS = 10_000

def cents(m, ds):
    d = np.load(CACHE/f"{m}_{ds}_train.npz"); X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(int(y.max())+1)])
def tree(C, metric, link):
    if metric == "cosine": C = C/np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    if link == "ward": Z = linkage(C, method="ward")
    else: Z = linkage(pdist(C, "cosine" if metric == "cosine" else "euclidean"), method=link)
    return Z
def triplets(n, rng):
    T = np.stack([rng.choice(n, 3, replace=False) for _ in range(N_TRIPLETS)])
    return T
def first_merge(Dsq, T):
    a, b, c = T[:,0], T[:,1], T[:,2]
    d = np.stack([Dsq[a,b], Dsq[a,c], Dsq[b,c]], 1)
    return d.argmin(1)

def main():
    rows, summ = [], []
    for ds in ("imagenet", "cifar100"):
        Cs = {m: cents(m, ds) for m in MODELS}; n = len(next(iter(Cs.values())))
        rng = np.random.RandomState(0); T = triplets(n, rng)
        for metric in ("euclid", "cosine"):
            for link in ("average", "complete", "ward"):
                coph, cuts, fm = {}, {}, {}
                for m in MODELS:
                    Z = tree(Cs[m], metric, link); cv = cophenet(Z); coph[m] = cv
                    cuts[m] = fcluster(Z, CUT[ds], criterion="maxclust"); fm[m] = first_merge(squareform(cv), T)
                vals = {}
                for a, b in itertools.combinations(MODELS, 2):
                    r = dict(dataset=ds, metric=metric, linkage=link, model_a=a, model_b=b,
                             ari_cut=adjusted_rand_score(cuts[a], cuts[b]),
                             coph_corr=float(np.corrcoef(coph[a], coph[b])[0,1]),
                             triplet_agree=float((fm[a]==fm[b]).mean()))
                    rows.append(r); vals[(a,b)] = vals[(b,a)] = r
                def mean_over(pairs, k): return float(np.mean([vals[p][k] for p in pairs]))
                big_blk = [(x,y) for x in BIG for y in BLOCK]; blk_blk = list(itertools.combinations(BLOCK, 2))
                s = dict(dataset=ds, metric=metric, linkage=link)
                for k in ("ari_cut", "coph_corr", "triplet_agree"):
                    s[f"{k}_big_vs_block"] = mean_over(big_blk, k); s[f"{k}_within_block"] = mean_over(blk_blk, k)
                summ.append(s)
                print(f"{ds:9s} {metric:6s}-{link:8s}: ARI@cut big/blk {s['ari_cut_big_vs_block']:.2f}/{s['ari_cut_within_block']:.2f} | coph-corr {s['coph_corr_big_vs_block']:.2f}/{s['coph_corr_within_block']:.2f} | triplets {s['triplet_agree_big_vs_block']:.2f}/{s['triplet_agree_within_block']:.2f}")
    pd.DataFrame(rows).to_csv(OUT/"expR58_treemap_cutfree.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT/"expR58_treemap_cutfree_summary.csv", index=False)
    print("Done expR58")

if __name__ == "__main__":
    main()
