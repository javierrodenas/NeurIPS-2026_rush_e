#!/usr/bin/env python3
"""
REBUTTAL EXP 3 — quantitative semantic correctness + ORC bridge analysis
(answers pux6 W1/Q1 and RJje's "internal tension in the ORC argument").

Part A (ImageNet, per model):
  spearman_wn   Spearman correlation between the 1000x1000 centroid distance
                matrix and the WordNet tree-distance matrix (ground-truth
                hierarchy alignment; upper-tri entries).
  spearman_shuf same after shuffling class identity (control ~ 0).

Part B (CIFAR-100, per model):
  ARI/NMI between the 20 clusters obtained by cutting the average-linkage
  dendrogram of the 100 class centroids and the TRUE 20 superclasses.
  + random-labels control.

Part C (ORC bridge, per model, ImageNet + CIFAR-100):
  ORC per edge of the k=10 centroid kNN graph, split into
  within-superclass edges vs across-superclass (bridge) edges.
  Superclasses: CIFAR-100 -> true 20; ImageNet -> 30 WordNet clusters.
  Reports mean ORC and fraction of negative edges per edge type.

Output: rebuttal/results/exp3_alignment.csv, exp3_orc_bridge.csv
"""
import os, sys, time
sys.stdout.reconfigure(line_buffering=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from scipy.optimize import linear_sum_assignment
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
OUT = ROOT / "rebuttal/results"
OUT.mkdir(parents=True, exist_ok=True)
WN_DIST = ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy"

PARADIGMS = {
    "i21k_t":"Supervised","i21k_s":"Supervised","i21k_b":"Supervised","i21k_l":"Supervised",
    "dinov1_b":"SSL","dinov2_s":"SSL","dinov2_b":"SSL","dinov2_l":"SSL","dinov2_g":"SSL",
    "clip_b":"Contrastive","clip_l":"Contrastive","siglip_b":"Contrastive",
}
PER_CLASS = 100

COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],
 5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],
 10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],
 14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],
 18:[8,13,48,58,90],19:[41,69,81,85,89]}
C100_SUPER = np.zeros(100, dtype=int)
for c, fs in COARSE.items():
    for f in fs: C100_SUPER[f] = c


def load_centroids(model, ds):
    if ds == "imagenet":
        d = np.load(CACHE / f"{model}_imagenet_fulltrain.npz")
        X_full = d["features"].astype(np.float32); y_full = d["labels"].astype(np.int64)
        rng = np.random.RandomState(0)
        sel = []
        for c in range(int(y_full.max()) + 1):
            ic = np.where(y_full == c)[0]
            sel.extend(rng.choice(ic, min(PER_CLASS, len(ic)), replace=False).tolist())
        X = X_full[np.array(sel)]; y = y_full[np.array(sel)]
    else:
        dd = np.load(CACHE / f"{model}_{ds}_train.npz")
        X = dd["features"].astype(np.float32); y = dd["labels"].astype(np.int64)
    n_classes = int(y.max()) + 1
    return np.stack([X[y == c].mean(0) for c in range(n_classes)])


def orc_edges(cents, k=10):
    """Returns list of (u, v, kappa). k=10 kNN graph, W1 via linear assignment."""
    D = squareform(pdist(cents, 'euclidean'))
    n = len(cents)
    nn = np.argsort(D, axis=1)[:, 1:k+1]
    edges = set()
    for u in range(n):
        for v in nn[u]:
            edges.add((min(u, int(v)), max(u, int(v))))
    out = []
    for (u, v) in edges:
        Nu, Nv = nn[u], nn[v]
        M = D[np.ix_(Nu, Nv)]
        r, c = linear_sum_assignment(M)
        W1 = M[r, c].mean()
        out.append((u, v, 1.0 - W1 / max(D[u, v], 1e-12)))
    return out


def main():
    Dw = np.load(WN_DIST)
    iu = np.triu_indices(1000, 1)
    wn_vec = Dw[iu]
    wn30 = AgglomerativeClustering(n_clusters=30, metric="precomputed",
                                   linkage="average").fit_predict(Dw)

    align_rows, orc_rows = [], []
    rng = np.random.RandomState(11)

    for model in PARADIGMS:
        t0 = time.time()
        # Part A: WordNet alignment on ImageNet
        cents_in = load_centroids(model, "imagenet")
        D_in = squareform(pdist(cents_in, 'euclidean'))
        r_wn = spearmanr(D_in[iu], wn_vec).statistic
        perm = rng.permutation(1000)
        r_shuf = spearmanr(D_in[np.ix_(perm, perm)][iu], wn_vec).statistic
        # gaussian control
        G = rng.randn(*cents_in.shape).astype(np.float32)
        r_gauss = spearmanr(squareform(pdist(G))[iu], wn_vec).statistic

        # Part B: CIFAR-100 dendrogram vs true superclasses
        cents_c1 = load_centroids(model, "cifar100")
        Z = linkage(cents_c1, method="average")
        cut = fcluster(Z, t=20, criterion="maxclust")
        ari = adjusted_rand_score(C100_SUPER, cut)
        nmi = normalized_mutual_info_score(C100_SUPER, cut)
        ari_rand = adjusted_rand_score(C100_SUPER, rng.permutation(cut))

        align_rows.append(dict(model=model, paradigm=PARADIGMS[model],
                               spearman_wn=r_wn, spearman_shuf=r_shuf,
                               spearman_gauss=r_gauss,
                               c100_ari=ari, c100_nmi=nmi, c100_ari_rand=ari_rand))
        pd.DataFrame(align_rows).to_csv(OUT / "exp3_alignment.csv", index=False)

        # Part C: ORC bridge, ImageNet (WordNet-30 superclasses) + CIFAR-100
        for ds, cents, sup in [("imagenet", cents_in, wn30),
                               ("cifar100", cents_c1, C100_SUPER)]:
            ed = orc_edges(cents)
            kap = np.array([e[2] for e in ed])
            within = np.array([sup[e[0]] == sup[e[1]] for e in ed])
            orc_rows.append(dict(
                model=model, paradigm=PARADIGMS[model], dataset=ds,
                n_edges=len(ed), frac_within=float(within.mean()),
                orc_within=float(kap[within].mean()),
                orc_across=float(kap[~within].mean()) if (~within).sum() else float("nan"),
                fneg_within=float((kap[within] < 0).mean()),
                fneg_across=float((kap[~within] < 0).mean()) if (~within).sum() else float("nan"),
            ))
        pd.DataFrame(orc_rows).to_csv(OUT / "exp3_orc_bridge.csv", index=False)
        o = orc_rows[-2]
        print(f"{model:10s} ({time.time()-t0:.0f}s) wn_rho={r_wn:+.3f} "
              f"(shuf {r_shuf:+.3f}) ARI={ari:.3f} (rand {ari_rand:+.3f}) | "
              f"IN orc w/a={o['orc_within']:+.3f}/{o['orc_across']:+.3f} "
              f"fneg w/a={o['fneg_within']:.3f}/{o['fneg_across']:.3f}", flush=True)

    print("Done ->", OUT)


if __name__ == "__main__":
    main()
