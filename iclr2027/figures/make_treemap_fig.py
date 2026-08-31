#!/usr/bin/env python3
"""Tree-map figure: naive configuration (Euclidean, average linkage) vs the criterion-selected
configuration, ImageNet and CIFAR-100. Sources: exp23_treemap_controls.npz (per-configuration
pairwise ARI between dendrogram cuts of the 12 vision models)."""
import os, sys, importlib
from pathlib import Path
import numpy as np, numpy.core as _core
sys.modules.setdefault("numpy._core", _core)                      # npz pickled under numpy 2
for _s in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core."+_s, importlib.import_module("numpy.core."+_s))
    except Exception: pass
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
z = np.load(RES/"exp23_treemap_controls.npz", allow_pickle=True)
S = {"imagenet": z["summary_in"].item(), "cifar100": z["summary_c1"].item()}
NAMES = ["ViT-T","ViT-S","ViT-B","ViT-L","DINO-B","Dv2-S","Dv2-B","Dv2-L","Dv2-G","CLIP-B","CLIP-L","SigLIP"]
PANELS = [("imagenet","('euclid', 'average')","(a) ImageNet, naive\nEuclidean, average"),
          ("imagenet","('cosine', 'average')","(b) ImageNet, selected\ncosine, average"),
          ("cifar100","('euclid', 'average')","(c) CIFAR-100, naive\nEuclidean, average"),
          ("cifar100","('cosine', 'complete')","(d) CIFAR-100, selected\ncosine, complete")]
fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.95))
for ax,(ds,cfg,title) in zip(axes, PANELS):
    M = np.asarray(S[ds][cfg]["ari"], dtype=float)
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    for b in (3.5, 8.5): ax.axhline(b, color="w", lw=0.8); ax.axvline(b, color="w", lw=0.8)
    sup = [0,1,2,3,9,10,11]; big = [6,7,8]
    v = np.mean([M[i,j] for i in big for j in sup])
    ax.set_title(title + f"\nDv2-B/L/G vs block: {v:.2f}", fontsize=6, pad=3)
    ax.set_xticks(range(12)); ax.set_xticklabels(NAMES, rotation=90, fontsize=4.8)
    ax.set_yticks(range(12)); ax.set_yticklabels(NAMES if ax is axes[0] else [], fontsize=4.8)
    ax.tick_params(length=1.5, pad=1)
cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.015, pad=0.03, shrink=0.8)
cb.set_label("ARI between tree cuts", fontsize=6); cb.ax.tick_params(labelsize=5.5)
fig.subplots_adjust(left=0.075, right=0.865, top=0.80, bottom=0.2, wspace=0.12)
out = HERE.parents[0] / "iclr2027" / "figures"
fig.savefig(out/"fig_treemap_controls.pdf"); fig.savefig(out/"fig_treemap_controls.png", dpi=200)
fig.savefig(HERE/"fig_treemap_controls.pdf")
print("fig_treemap_controls written to", out)
