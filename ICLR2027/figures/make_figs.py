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

from matplotlib.ticker import MaxNLocator, FuncFormatter
def _fmt2(v, pos=None):
    s = f"{v:.2f}"; return "0.00" if s in ("-0.00", "0.00") else s
def tidy(ax, decimals=True, n=3):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=2))
    if decimals: ax.yaxis.set_major_formatter(FuncFormatter(_fmt2))
    ax.tick_params(labelsize=7)

NAME = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L",
        "dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B",
        "dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
        "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
PARA = {"i21k":"Supervised","i21k_":"Supervised","dinov":"SSL","clip_":"Contrastive","sigli":"Contrastive"}
import sys as _sys; _sys.path.insert(0, str(OUT))
from palette import FAMILY_COLORS as _FC
COL  = {"Supervised": _FC["supervised"], "SSL": _FC["ssl"], "Contrastive": _FC["contrastive"]}   # family colors from the shared palette
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

# ---------- Figure A: the census of record as a horizontal dot plot, one panel per dataset (post-freeze redraw ordered by the author, 2026-09-18) ----------
import sys as _sys2; _sys2.path.insert(0, str(OUT))
from palette import FAMILY_COLORS as _FC2, color as _fam_color
DSO = ["imagenet", "cifar100", "cifar10", "dtd", "fashionmnist", "mnist"]
DSL = {"imagenet": "ImageNet", "cifar100": "CIFAR-100", "cifar10": "CIFAR-10", "dtd": "DTD", "fashionmnist": "FMNIST", "mnist": "MNIST"}
GRAY = _FC2["null"]; Y = np.arange(len(ORDER))[::-1]                       # ViT-T at the top, SigLIP-B at the bottom
fig, axes = plt.subplots(1, 6, figsize=(5.5, 1.88), sharey=True, gridspec_kw={"width_ratios": [1.35, 1, 1, 1, 1, 1]})
for ax, ds in zip(axes, DSO):
    for b in (7.5, 2.5): ax.axhline(b, color=GRAY, lw=0.5, zorder=1)             # thin separators between the three families
    ax.axvline(0.0, color="k", lw=0.7, zorder=2)
    for k, m in enumerate(ORDER):
        v, g, c = exc[(m, ds)], gen[(m, ds)], _fam_color(m)
        ax.scatter(v, Y[k], s=15, facecolors=c if g else "white", edgecolors=c, linewidths=0.9, zorder=3, clip_on=False)
    ax.set_xlim(-0.14, 0.03)
    if ds == "imagenet": ax.set_xticks([-0.10, -0.05, 0.0]); ax.set_xticklabels(["−0.10", "−0.05", "0.00"], fontsize=7)   # intermediate tick: the small ImageNet excesses read as -0.01 to -0.02, not as zero
    else: ax.set_xticks([-0.10, 0.0]); ax.set_xticklabels(["−0.10", "0.00"], fontsize=7)
    ax.set_ylim(-0.6, len(ORDER) - 0.4); ax.set_title(DSL[ds], fontsize=7, pad=3); ax.tick_params(labelsize=7, length=2, pad=1.5)
    for sp in ("left", "right", "top"): ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
axes[0].set_yticks(Y); axes[0].set_yticklabels([NAME[m] for m in ORDER], fontsize=7)
fig.text(0.56, 0.115, "excess of the reading of record over its matched null", ha="center", va="bottom", fontsize=7)
_hd = [plt.Line2D([], [], marker="o", ls="", color="k", ms=4, label="filled: genuine cell"), plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=4, label="hollow: not genuine")]
_hd += [plt.Line2D([], [], marker="o", ls="", color=_FC2[k], ms=4, label=l) for k, l in [("supervised", "supervised"), ("ssl", "self-supervised"), ("contrastive", "contrastive")]]
fig.legend(handles=_hd, frameon=False, loc="lower center", ncol=5, fontsize=7, handlelength=1.0, handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.12, right=0.995, top=0.905, bottom=0.30, wspace=0.10)
for o in (OUT, OUT.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_excess_panel.pdf"); fig.savefig(o/"fig_excess_panel.png", dpi=200)
_neg = sum(1 for k in exc if exc[k] < 0); print(f"dot plot: {_neg}/72 below the null, {sum(gen.values())} genuine; x range {min(exc.values()):+.3f}..{max(exc.values()):+.3f}")

# ---------- Figure B: best-metric scatter ----------
m2 = list(csv.DictReader(open(RES/"exp2_metric_controls.csv")))
HIER = {"imagenet","cifar100","cifar10","dtd"}
pts = []
for r in m2:
    k = (r["model"], r["dataset"])
    if k not in dlt: continue
    fsR, fsH, fsC = (float(r[f"FS_{a}"]) for a in ["R","H","COS"])
    ncR, ncH, ncC = (float(r[f"NC_{a}"]) for a in ["R","H","COS"])
    pts.append(dict(m=r["model"], ds=r["dataset"], delta=exc[k],
                    fs=(max(fsH, fsC)-fsR)*100, nc=(max(ncH, ncC)-ncR)*100))   # x = the record excess (A3 of the final pass); the raw reading is in expR68

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
    ax.set_xlabel(r"record excess (per model$\times$dataset)", fontsize=8)
    ax.set_ylabel(f"best $-$ Euclidean (pp), {lab}", fontsize=8)
    ax.set_title(lab, fontsize=8); tidy(ax)
    print(f"CAPTION DATA {lab}: within-dataset r {max(r_ds):+.2f}..{min(r_ds):+.2f}; pooled {r_all:+.2f}; hier-only {r_h:+.2f}")
hp = [plt.Line2D([], [], marker="o", ls="", color=c, ms=5, label=p) for p, c in COL.items()]
hm = [plt.Line2D([], [], marker=mk, ls="", color="gray", ms=4.5, label=ds) for ds, mk in MARK.items()]
fig.legend(handles=hp+hm, fontsize=7, ncol=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.6, handletextpad=0.25, handlelength=1.0)
fig.tight_layout(rect=(0, 0, 1, 0.92))
for o in (OUT, OUT.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_bestmetric_scatter.pdf"); fig.savefig(o/"fig_bestmetric_scatter.png", dpi=200)
print("figs written")
