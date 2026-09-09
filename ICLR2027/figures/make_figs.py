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
plt.style.use(str(OUT / "style.mplstyle"))

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

_src = RES/"expR52_census_haar_p999_200.csv"                            # census of record (Phase B): Haar x p99.9, BH
if not _src.exists(): _src = RES/"expR39c_census200_cache.csv"
if not _src.exists(): _src = RES/"exp20_null_ztable.csv"
print("census source:", _src.name)
d20 = list(csv.DictReader(open(_src)))
exc = {(r["model"], r["dataset"]): float(r["excess"]) for r in d20}
def _p(r):
    if "p_left" in r: return float(r["p_left"])
    if "frac_null_above" in r: return (1 + 20 - round(20*float(r["frac_null_above"]))) / 21
    return 0.0
gen = {(r["model"], r["dataset"]): (str(r["genuine_bh"]) == "True" if "genuine_bh" in r else _p(r) <= 0.05) for r in d20}   # genuine = BH-corrected p <= 0.05
dlt = {(r["model"], r["dataset"]): float(r["delta"]) for r in d20}

# ---------- Figure A: excess panel ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 1.12),
                               gridspec_kw={"width_ratios": [1.55, 1]})
xs = np.arange(len(ORDER))
for k, m in enumerate(ORDER):
    for ds, mk in MARK.items():
        if (m, ds) in exc:
            _g = gen[(m, ds)]
            ax1.scatter(k, exc[(m, ds)], marker=mk, s=16,
                        facecolors=COL[para(m)] if _g else "white", edgecolors=COL[para(m)], linewidths=0.8,
                        alpha=0.45 if ds == "imagenet" else 0.9, zorder=3)   # hollow = not genuine (p > 0.05)
ax1.axhline(0, color="k", lw=0.8, zorder=1)
ax1.set_xticks(xs); ax1.set_xticklabels([NAME[m] for m in ORDER], rotation=60, ha="right", fontsize=6.5)
ax1.set_ylabel(r"excess  $\hat\delta_{99.9}^{\rm real}-\hat\delta_{99.9}^{\rm null}$", fontsize=8)
_neg = sum(1 for r in d20 if float(r["excess"]) < 0); _gen = sum(gen.values())
ax1.set_title(f"(a) clustered structure: {_neg}/72 below the null, {_gen} genuine", fontsize=7.5)
ax1.tick_params(labelsize=7)
hd = [plt.Line2D([], [], marker=mk, ls="", color="gray", ms=4.5, label=ds)
      for ds, mk in MARK.items()]

fams = {"DINOv2 (SSL)": ["dinov2_s","dinov2_b","dinov2_l","dinov2_g"],
        "ViT (Sup.)":   ["i21k_t","i21k_s","i21k_b","i21k_l"]}
for fam, ms in fams.items():
    col = COL["SSL"] if "DINO" in fam else COL["Supervised"]
    for ds in ["cifar100","cifar10","dtd"]:
        ax2.plot(range(len(ms)), [exc[(m, ds)] for m in ms], "-", marker=MARK[ds],
                 color=col, ms=3.5, lw=1.0, label=None)
    ax2.plot(range(len(ms)), [exc[(m, "imagenet")] for m in ms], "--o", color=col,
             ms=3.5, lw=1.0, alpha=0.45, label=None)
hd2 = [plt.Line2D([], [], marker=MARK[ds], ls="-", color="gray", ms=3.5, lw=1.0, label=ds)
       for ds in ["cifar100","cifar10","dtd"]] +       [plt.Line2D([], [], marker="o", ls="--", color="gray", ms=3.5, lw=1.0, alpha=0.6, label="imagenet")] +       [plt.Line2D([], [], ls="-", color=COL["SSL"], lw=1.6, label="DINOv2"),
       plt.Line2D([], [], ls="-", color=COL["Supervised"], lw=1.6, label="ViT")]
ax2.axhline(0, color="k", lw=0.8)
ax2.set_xticks(range(4)); ax2.set_xticklabels(["1\n(smallest)", "2", "3", "4\n(largest)"], fontsize=6.5)
ax2.set_xlabel("model scale $\\rightarrow$", fontsize=8)
ax2.set_title("(b) per-dataset excess vs scale", fontsize=7.5)
ax2.tick_params(labelsize=7)
ax2.set_ylim(-0.175, 0.09)
hfam = [plt.Line2D([], [], marker="s", ls="", color=c, ms=5, label=p) for p, c in COL.items()]
fig.legend(handles=hd + hfam + hd2[3:4], fontsize=5.8, ncol=10, frameon=False,
           loc="lower center", bbox_to_anchor=(0.5, -0.02), columnspacing=0.7, handletextpad=0.3)
fig.tight_layout(rect=(0, 0.10, 1, 1))
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

fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.5))
for ax, key, lab in [(axes[0], "fs", "FS"), (axes[1], "nc", "NC")]:
    for p in pts:
        ax.scatter(p["delta"], p[key], marker=MARK[p["ds"]], s=18,
                   color=COL[para(p["m"])], alpha=0.85, edgecolors="none")
    ax.axhline(0, color="k", lw=0.8)
    r_all = pear([p["delta"] for p in pts], [p[key] for p in pts])
    ph = [p for p in pts if p["ds"] in HIER]
    r_h = pear([p["delta"] for p in ph], [p[key] for p in ph])
    r_ds = [pear([p["delta"] for p in ph if p["ds"] == d], [p[key] for p in ph if p["ds"] == d]) for d in HIER]
    ax.set_xlabel(r"$\hat\delta$ (per model$\times$dataset)", fontsize=7.5)
    ax.set_ylabel(f"best $-$ Euclidean (pp), {lab}", fontsize=7)
    ax.set_title(lab, fontsize=8)
    print(f"CAPTION DATA {lab}: within-dataset r {max(r_ds):+.2f}..{min(r_ds):+.2f}; pooled {r_all:+.2f}; hier-only {r_h:+.2f}")
    ax.tick_params(labelsize=7)
hp = [plt.Line2D([], [], marker="o", ls="", color=c, ms=5, label=p) for p, c in COL.items()]
hm = [plt.Line2D([], [], marker=mk, ls="", color="gray", ms=4.5, label=ds) for ds, mk in MARK.items()]
fig.legend(handles=hp+hm, fontsize=5.8, ncol=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.9, handletextpad=0.3)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for o in (OUT, OUT.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_bestmetric_scatter.pdf"); fig.savefig(o/"fig_bestmetric_scatter.png", dpi=200)
print("figs written")
