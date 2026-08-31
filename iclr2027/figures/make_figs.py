#!/usr/bin/env python3
"""ICLR 2027 figures: (A) excess-basis geometry panel, (B) best-metric scatter.
Data: rebuttal/results/exp11_null_per_dataset.csv (excess), exp2_metric_controls.csv (tasks).
"""
import csv, os, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(os.environ.get("PLATONIC_RESULTS", Path(__file__).resolve().parents[2] / "rebuttal/results"))
OUT = Path(__file__).resolve().parent

NAME = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L",
        "dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B",
        "dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
        "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
PARA = {"i21k":"Supervised","i21k_":"Supervised","dinov":"SSL","clip_":"Contrastive","sigli":"Contrastive"}
COL  = {"Supervised":"#4C72B0","SSL":"#DD8452","Contrastive":"#55A868"}
MARK = {"imagenet":"o","cifar100":"s","cifar10":"D","dtd":"^","fashionmnist":"v","mnist":"x"}
ORDER = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b",
         "dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]

def para(m): return PARA[m[:4]] if m.startswith("i21k") else PARA[m[:5]]

d20 = list(csv.DictReader(open(RES/"exp20_null_ztable.csv")))
exc = {(r["model"], r["dataset"]): float(r["excess"]) for r in d20}
dlt = {(r["model"], r["dataset"]): float(r["delta"]) for r in d20}

# ---------- Figure A: excess panel ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.35),
                               gridspec_kw={"width_ratios": [1.55, 1]})
xs = np.arange(len(ORDER))
for k, m in enumerate(ORDER):
    for ds, mk in MARK.items():
        if (m, ds) in exc:
            ax1.scatter(k, exc[(m, ds)], marker=mk, s=16, color=COL[para(m)],
                        alpha=0.45 if ds == "imagenet" else 0.9,
                        edgecolors="none", zorder=3)
ax1.axhline(0, color="k", lw=0.8, zorder=1)
ax1.set_xticks(xs); ax1.set_xticklabels([NAME[m] for m in ORDER], rotation=60, ha="right", fontsize=6.5)
ax1.set_ylabel(r"tree excess  $\hat\delta_{\rm real}-\hat\delta_{\rm null}$", fontsize=8)
ax1.set_title("(a) 69/72 cells below their matched null", fontsize=7.5)
ax1.tick_params(labelsize=7)
hd = [plt.Line2D([], [], marker=mk, ls="", color="gray", ms=4.5, label=ds)
      for ds, mk in MARK.items()]
ax1.legend(handles=hd, fontsize=5.5, ncol=3, frameon=False, loc="lower left", columnspacing=0.8, handletextpad=0.3)

fams = {"DINOv2 (SSL)": ["dinov2_s","dinov2_b","dinov2_l","dinov2_g"],
        "ViT (Sup.)":   ["i21k_t","i21k_s","i21k_b","i21k_l"]}
for fam, ms in fams.items():
    col = COL["SSL"] if "DINO" in fam else COL["Supervised"]
    tr = [np.mean([exc[(m, ds)] for ds in ["cifar100","cifar10","dtd"]]) for m in ms]
    im = [exc[(m, "imagenet")] for m in ms]
    ax2.plot(range(len(ms)), tr, "-o", color=col, ms=4, lw=1.4, label=f"{fam}, transfer")
    ax2.plot(range(len(ms)), im, "--o", color=col, ms=4, lw=1.1, alpha=0.45,
             label=f"{fam}, ImageNet")
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xticks(range(4)); ax2.set_xticklabels(["1\n(smallest)", "2", "3", "4\n(largest)"], fontsize=6.5)
ax2.set_xlabel("model scale $\\rightarrow$", fontsize=8)
ax2.set_title("(b) excess deepens with scale", fontsize=7.5)
ax2.tick_params(labelsize=7)
ax2.set_ylim(-0.115, 0.08)
ax2.legend(fontsize=5.5, frameon=False, loc="upper right", handlelength=1.6)
fig.tight_layout()
for o in (OUT, OUT.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_excess_panel.pdf"); fig.savefig(o/"fig_excess_panel.png", dpi=200)

# ---------- Figure B: best-metric scatter ----------
m2 = list(csv.DictReader(open(RES/"exp2_metric_controls.csv")))
HIER = {"imagenet","cifar100","cifar10","dtd"}
pts = []
for r in m2:
    k = (r["model"], r["dataset"])
    if k not in dlt: continue
    fsR, fsH, fsC = (float(r[f"FS_{a}"]) for a in ["R","H","COS"])
    ncR, ncH, ncC = (float(r[f"NC_{a}"]) for a in ["R","H","COS"])
    pts.append(dict(m=r["model"], ds=r["dataset"], delta=dlt[k],
                    fs=(max(fsH, fsC)-fsR)*100, nc=(max(ncH, ncC)-ncR)*100))

def pear(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return np.corrcoef(x, y)[0, 1]

fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
for ax, key, lab in [(axes[0], "fs", "FS"), (axes[1], "nc", "NC")]:
    for p in pts:
        ax.scatter(p["delta"], p[key], marker=MARK[p["ds"]], s=18,
                   color=COL[para(p["m"])], alpha=0.85, edgecolors="none")
    ax.axhline(0, color="k", lw=0.8)
    r_all = pear([p["delta"] for p in pts], [p[key] for p in pts])
    ph = [p for p in pts if p["ds"] in HIER]
    r_h = pear([p["delta"] for p in ph], [p[key] for p in ph])
    ax.set_xlabel(r"$\hat\delta$ (per model$\times$dataset)", fontsize=7.5)
    ax.set_ylabel(f"best $-$ Euclidean (pp), {lab}", fontsize=7)
    ax.set_title(f"{lab}: r = {r_all:+.2f} pooled / {r_h:+.2f} hierarchical", fontsize=7.5)
    ax.tick_params(labelsize=7)
hp = [plt.Line2D([], [], marker="o", ls="", color=c, ms=5, label=p) for p, c in COL.items()]
hm = [plt.Line2D([], [], marker=mk, ls="", color="gray", ms=4.5, label=ds) for ds, mk in MARK.items()]
fig.legend(handles=hp+hm, fontsize=5.8, ncol=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handletextpad=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for o in (OUT, OUT.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_bestmetric_scatter.pdf"); fig.savefig(o/"fig_bestmetric_scatter.png", dpi=200)
print("figs written")
