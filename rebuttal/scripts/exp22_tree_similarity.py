#!/usr/bin/env python3
"""
ICLR EXP 22 — direct tree-vs-tree comparison across models (the map).

Per model: dendrogram over the shared class centroids (average linkage on
Euclidean centroid distances). Pairwise hierarchy similarity with two CLASSIC
measures (nothing invented): ARI between k-cut partitions (Hubert-Arabie) and
cophenetic correlation (Sokal-Rohlf). References included as extra rows:
ImageNet -> WordNet tree; CIFAR-100 -> the true 20-superclass partition.
Permutation null for ARI ~ 0 by construction (checked).

Outputs: exp22_tree_similarity_{imagenet,cifar100}.npz + fig_tree_similarity.{pdf,png}
"""
import sys, itertools
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, cophenet
from sklearn.metrics import adjusted_rand_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/javi/Platonic"); CACHE = ROOT/"results/practical_tasks_cache"
OUT = ROOT/"rebuttal/results"
MODELS = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b",
          "dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
SH = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
      "dinov2_s":"Dv2-S","dinov2_b":"Dv2-B","dinov2_l":"Dv2-L","dinov2_g":"Dv2-G",
      "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP",
      "wordnet":"WordNet","superclasses":"Supercl."}
FINE2COARSE = np.array([4,1,14,8,0,6,7,7,18,3,3,14,9,18,7,11,3,9,7,11,6,11,5,10,7,6,13,15,3,15,
 0,11,1,10,12,14,16,9,11,5,5,19,8,8,15,13,14,17,18,10,16,4,17,4,2,0,17,4,18,17,10,3,2,12,12,16,
 12,1,9,19,2,10,0,1,16,12,9,13,15,13,16,19,2,4,6,19,5,5,8,19,18,1,2,15,6,0,17,8,14,13])

def cents(m, ds, n_cls):
    d = np.load(CACHE/f"{m}_{ds}_train.npz")
    X = d["features"].astype(np.float32); y = d["labels"].astype(np.int64)
    return np.stack([X[y==c].mean(0) for c in range(n_cls)])

def run(ds, n_cls, k_cut, ref_name):
    trees, coph_v = {}, {}
    for m in MODELS:
        Z = linkage(pdist(cents(m, ds, n_cls)), method="average")
        trees[m] = Z; coph_v[m] = cophenet(Z)
    names = MODELS + [ref_name]
    cuts = {m: fcluster(trees[m], k_cut, criterion="maxclust") for m in MODELS}
    if ds == "imagenet":
        WND = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
        Zr = linkage(squareform(WND, checks=False), method="average")
        cuts[ref_name] = fcluster(Zr, k_cut, criterion="maxclust")
        coph_v[ref_name] = cophenet(Zr)
    else:
        cuts[ref_name] = FINE2COARSE + 1
        coph_v[ref_name] = None
    n = len(names)
    ari = np.eye(n); cop = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        a, b = names[i], names[j]
        ari[i,j] = ari[j,i] = adjusted_rand_score(cuts[a], cuts[b])
        if coph_v[a] is not None and coph_v[b] is not None:
            cop[i,j] = cop[j,i] = np.corrcoef(coph_v[a], coph_v[b])[0,1]
        else:
            cop[i,j] = cop[j,i] = np.nan
    np.savez(OUT/f"exp22_tree_similarity_{ds}.npz", ari=ari, coph=cop,
             models=names, k_cut=k_cut)
    return names, ari

names_in, ari_in = run("imagenet", 1000, 30, "wordnet")
names_c1, ari_c1 = run("cifar100", 100, 20, "superclasses")

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6),
                         gridspec_kw={"width_ratios": [1, 1]})
for ax, (names, M, title) in zip(axes, [
        (names_in, ari_in, "(a) ImageNet, ARI @ 30 clusters"),
        (names_c1, ari_c1, "(b) CIFAR-100, ARI @ 20 clusters")]):
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    labs = [SH[m] for m in names]
    ax.set_xticks(range(len(names)), labs, rotation=45, ha="right", fontsize=6.5)
    ax.set_yticks(range(len(names)), labs, fontsize=6.5)
    for i in range(len(names)):
        for j in range(len(names)):
            v = M[i,j]
            ax.text(j, i, f"{v:.2f}"[1:] if 0<=v<1 else f"{v:.0f}",
                    ha="center", va="center", fontsize=4.6,
                    color="white" if v < 0.5 else "black")
    ax.set_title(title, fontsize=9)
    for pos in [3.5, 4.5, 8.5, 11.5]:
        ax.axhline(pos, color="w", lw=0.8); ax.axvline(pos, color="w", lw=0.8)
fig.colorbar(im, ax=axes, shrink=0.8, label="ARI between tree cuts")
fig.savefig(OUT.parent.parent/"rebuttal/results/fig_tree_similarity.pdf", bbox_inches="tight")
fig.savefig(OUT.parent.parent/"rebuttal/results/fig_tree_similarity.png", dpi=170, bbox_inches="tight")

print("CIFAR-100 verdict (where DINOv2 form is strongest):")
idx = {m: i for i, m in enumerate(names_c1)}
d2 = ["dinov2_s","dinov2_b","dinov2_l","dinov2_g"]
others = [m for m in MODELS if m not in d2 and m != "dinov1_b"]
xf = [ari_c1[idx[a], idx[b]] for a in d2 for b in others]
wn = [ari_c1[idx[a], idx["superclasses"]] for a in d2]
ow = [ari_c1[idx[a], idx["superclasses"]] for a in others]
win = [ari_c1[idx[a], idx[b]] for a, b in itertools.combinations(d2, 2)]
sup_pairs = [ari_c1[idx[a], idx[b]] for a, b in itertools.combinations(others, 2)]
print(f"  DINOv2 vs otros:        {np.mean(xf):.3f} (range {min(xf):.2f}-{max(xf):.2f})")
print(f"  DINOv2 dentro familia:  {np.mean(win):.3f}")
print(f"  DINOv2 vs superclases:  {np.mean(wn):.3f}")
print(f"  otros vs superclases:   {np.mean(ow):.3f}")
print(f"  otros entre si:         {np.mean(sup_pairs):.3f}")
EOF_MARKER_NOT_USED = True
