#!/usr/bin/env python3
"""Figure 2 (overview), in the style of the scatter figures of Huh et al. (2024) and Groger et al. (2026): white background, no grid,
labeled points. Redrawn on the author's order after the freeze (2026-09-18).
(a) two thirds of the width: x = dimension d (log axis), y = delta_norm (the raw reading delta_999 of the 12 ImageNet backbones,
    expR52_census_haar_p999_200.csv), one filled marker per backbone in its family color with the model name beside it (adjustText),
    its matched-null mean as a hollow marker of the same color joined by a thin vertical segment, and the iid-Gaussian curve of the
    calibration table (exp1_delta_controls.csv, variant gauss, supremum statistic) as a dashed gray line labeled
    'structureless cloud, isotropic'; legend inside: filled = raw reading, hollow = matched null.
(b) one third: the two cells with equal raw reading of phaseC_fig2b.json (ViT-T on DTD images, expR62_samplelevel_record.csv;
    SigLIP-B on CIFAR-10 centroids, expR52) in the same language, names beside the points."""
import csv, os, sys, importlib, json
from pathlib import Path
import numpy as np, numpy.core as _core
sys.modules.setdefault("numpy._core", _core)
for _s in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core."+_s, importlib.import_module("numpy.core."+_s))
    except Exception: pass
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter, FixedLocator, NullLocator
from adjustText import adjust_text
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
plt.style.use(str(HERE / "style.mplstyle"))
sys.path.insert(0, str(HERE))
from palette import FAMILY_COLORS, color as fam_color
def _fmt2(v, pos=None):
    s = f"{v:.2f}"; return "0.00" if s in ("-0.00", "0.00") else s
def tidy(ax, n=3):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=2)); ax.yaxis.set_major_formatter(FuncFormatter(_fmt2)); ax.tick_params(labelsize=7)

census = list(csv.DictReader(open(RES/"expR52_census_haar_p999_200.csv")))
d_in = {r["model"]: (float(r["delta"]), float(r["null_mean"])) for r in census if r["dataset"] == "imagenet"}
e1 = list(csv.DictReader(open(RES/"exp1_delta_controls.csv")))
DIMS = {r["model"]: int(r["d"]) for r in e1 if r["model"] in d_in and r["variant"] == "real"}
DIMS.update({m: d for m, d in {"i21k_t":192,"i21k_s":384,"i21k_b":768,"i21k_l":1024,"dinov1_b":768,"dinov2_s":384,"dinov2_b":768,"dinov2_l":1024,"dinov2_g":1536,"clip_b":512,"clip_l":768,"siglip_b":768}.items() if m not in DIMS})
gauss = sorted({int(r["d"]): float(r["delta_max"]) for r in e1 if r["variant"] == "gauss"}.items())
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
GRAY = FAMILY_COLORS["null"]

fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.3), gridspec_kw={"width_ratios": [2, 1]})
# ---------------- (a)
ax = axes[0]
ax.plot([d for d, _ in gauss], [v for _, v in gauss], "--", color=GRAY, lw=0.9, zorder=1)
ax.text(gauss[0][0] * 1.05, gauss[0][1] + 0.003, "structureless cloud, isotropic", fontsize=7, color=GRAY, ha="left", va="bottom")
# backbones sharing a dimension are spread within a few per cent of d (log axis) so their segments do not overlap; label offsets in points
SPREAD = {"i21k_s": -0.05, "dinov2_s": 0.05, "i21k_b": -0.09, "dinov1_b": -0.045, "dinov2_b": 0.0, "clip_l": 0.045, "siglip_b": 0.09, "i21k_l": -0.05, "dinov2_l": 0.05}
OFF = {"i21k_t": (5, -4, "left"), "i21k_s": (-5, 1, "right"), "dinov2_s": (-5, -1, "right"), "clip_b": (-5, 2, "right"), "i21k_b": (-5, -1, "right"), "dinov1_b": (5, 4, "left"),
       "dinov2_b": (-5, -3, "right"), "clip_l": (0, 15, "center"), "siglip_b": (5, 6, "left"), "i21k_l": (5, 8, "left"), "dinov2_l": (0, -8, "center"), "dinov2_g": (5, -2, "left")}
for m, (raw, null) in d_in.items():
    x = DIMS[m] * (1 + SPREAD.get(m, 0.0)); c = fam_color(m); dx, dy, ha = OFF[m]
    ax.plot([x, x], [null, raw], "-", color=c, lw=0.7, alpha=0.8, zorder=2)
    ax.scatter(x, raw, s=18, color=c, zorder=4, clip_on=False)
    ax.scatter(x, null, s=18, facecolors="white", edgecolors=c, linewidths=0.9, zorder=3, clip_on=False)
    ax.annotate(NM[m], (x, raw), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=c, ha=ha, va="center", zorder=5)
ax.set_xscale("log"); ax.set_xlim(150, 2300); ax.xaxis.set_major_locator(FixedLocator([192, 384, 768, 1536])); ax.xaxis.set_minor_locator(NullLocator())
ax.set_xticklabels(["192", "384", "768", "1536"], fontsize=7); ax.set_xlabel("dimension $d$"); ax.set_ylabel(r"$\delta_{\mathrm{norm}}$")
ax.set_ylim(0.012, 0.118); tidy(ax, n=4)
ax.set_title("(a) the null moves with the dimension; the gap is the reading")
hd = [plt.Line2D([], [], marker="o", ls="", color="k", ms=4, label="filled: raw reading"), plt.Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=4, label="hollow: matched null")]
ax.legend(handles=hd, frameon=False, loc="lower left", fontsize=7, handlelength=1.0, handletextpad=0.3, labelspacing=0.2, borderaxespad=0.1)
# ---------------- (b)
ax = axes[1]
D = json.load(open(RES/"phaseC_fig2b.json")); assert D.get("mode") == "sample", D
sl = {(r["model"], r["dataset"]): r for r in csv.DictReader(open(RES/"expR62_samplelevel_record.csv"))}
s_m, s_ds = D["sample_cell"]; c_m, c_ds = D["class_cell"]; cc = {(r["model"], r["dataset"]): r for r in census}[(c_m, c_ds)]
DSN = {"dtd": "DTD", "cifar100": "CIFAR-100", "cifar10": "CIFAR-10", "imagenet": "ImageNet"}
cells = [(NM[s_m] + ", " + DSN[s_ds] + " images", fam_color(s_m), float(sl[(s_m, s_ds)]["delta_999"]), float(sl[(s_m, s_ds)]["null_mean"]), (-6, -10, "left")),
         (NM[c_m] + ",\n" + DSN[c_ds] + "\ncentroids", fam_color(c_m), float(cc["delta"]), float(cc["null_mean"]), (5, 8, "left"))]
XPOS = [0.0, 1.3]
print(f"(b) {s_m}/{s_ds} raw {cells[0][2]:.3f} null {cells[0][3]:.3f} | {c_m}/{c_ds} raw {cells[1][2]:.3f} null {cells[1][3]:.3f}")
for i, (lab, col, raw, null, (dx, dy, ha)) in enumerate(cells):
    x = XPOS[i]
    ax.plot([x, x], [null, raw], "-", color=col, lw=0.7, alpha=0.8, zorder=2)
    ax.scatter(x, raw, s=18, color=col, zorder=4, clip_on=False)
    ax.scatter(x, null, s=18, facecolors="white", edgecolors=col, linewidths=0.9, zorder=3, clip_on=False)
    ax.annotate(lab, (x, raw), xytext=(dx, dy), textcoords="offset points", fontsize=7, color=col, ha=ha, va="center", linespacing=1.1)
ax.set_xlim(-0.5, 3.3); ax.set_xticks([]); ax.set_ylabel(r"$\delta_{\mathrm{norm}}$")
lo, hi = min(c[2] for c in cells) - 0.022, max(c[3] for c in cells) + 0.012; ax.set_ylim(lo, hi); tidy(ax, n=4)
ax.set_title("(b) same reading, opposite verdict")
for a in axes:
    for sp in ("top", "right"): a.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.085, right=0.99, top=0.86, bottom=0.25, wspace=0.28)
for o in (HERE, HERE.parent/"iclr2027"/"figures"):
    fig.savefig(o/"fig_overview.pdf"); fig.savefig(o/"fig_overview.png", dpi=200)
print("fig_overview written")
