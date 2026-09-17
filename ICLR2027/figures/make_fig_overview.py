#!/usr/bin/env python3
"""Figure 2 (overview; final pass, post-freeze redraw ordered by the author on 2026-09-17), two panels at the same total height:
(a) the 12 ImageNet backbones by dimension d: each backbone's raw reading delta_999 (filled, family color) and its matched-null mean
    (hollow, gray) joined by a thin vertical segment whose length is the excess (expR52_census_haar_p999_200.csv; dimensions from
    exp1_delta_controls.csv);
(b) the same drawing for the two cells with equal raw reading chosen in phaseC_fig2b.json: a sample-level cell (ViT-T on DTD images,
    expR62_samplelevel_record.csv) and a class-level cell (SigLIP-B on CIFAR-10 centroids, expR52): two filled points at the same
    height, their hollow nulls at very different heights, segments joining each pair."""
import csv, os, sys, importlib, json
from pathlib import Path
import numpy as np, numpy.core as _core
sys.modules.setdefault("numpy._core", _core)
for _s in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core."+_s, importlib.import_module("numpy.core."+_s))
    except Exception: pass
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
plt.style.use(str(HERE / "style.mplstyle"))
sys.path.insert(0, str(HERE))
from palette import FAMILY_COLORS, color as fam_color
def _fmt2(v, pos=None):
    s = f"{v:.2f}"; return "0.00" if s in ("-0.00", "0.00") else s
def tidy(ax, n=3):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=2)); ax.yaxis.set_major_formatter(FuncFormatter(_fmt2)); ax.tick_params(labelsize=7)

src = RES/"expR52_census_haar_p999_200.csv"; print("census source:", src.name)
census = list(csv.DictReader(open(src)))
d_in = {r["model"]: (float(r["delta"]), float(r["null_mean"])) for r in census if r["dataset"] == "imagenet"}
DIMS = {}
for r in csv.DictReader(open(RES/"exp1_delta_controls.csv")):
    if r["model"] in d_in and r["variant"] == "real": DIMS[r["model"]] = int(r["d"])
DIMS.update({m: d for m, d in {"i21k_t":192,"i21k_s":384,"i21k_b":768,"i21k_l":1024,"dinov1_b":768,"dinov2_s":384,"dinov2_b":768,"dinov2_l":1024,"dinov2_g":1536,"clip_b":512,"clip_l":768,"siglip_b":768}.items() if m not in DIMS})
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
GRAY = FAMILY_COLORS["null"]

fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.3), gridspec_kw={"width_ratios": [1.35, 1]})
# ---- (a): backbones by dimension; backbones sharing a dimension are spread within +-7% of d so every segment is visible
ax = axes[0]
by_d = {}
for m in NM: by_d.setdefault(DIMS[m], []).append(m)
for d, ms in by_d.items():
    offs = np.linspace(-0.07, 0.07, len(ms)) * d if len(ms) > 1 else [0.0]
    for m, o in zip(ms, offs):
        raw, null = d_in[m]; x = d + o
        ax.plot([x, x], [null, raw], "-", color=GRAY, lw=0.7, zorder=1)
        ax.scatter(x, raw, s=18, color=fam_color(m), zorder=3)
        ax.scatter(x, null, s=26, facecolors="none", edgecolors=GRAY, linewidths=0.9, zorder=4)   # ring above the dot: visible even when the null sits at the reading
ax.set_xlabel("dimension $d$"); ax.set_ylabel(r"raw $\hat\delta_{99.9}$ (ImageNet)"); ax.set_title("(a) raw readings and their matched nulls")
ax.set_xlim(80, 1700); ax.set_xticks([192, 384, 768, 1024, 1536]); tidy(ax)
# ---- (b): the two cells with equal raw reading (decision recorded in phaseC_fig2b.json)
ax = axes[1]
D = json.load(open(RES/"phaseC_fig2b.json")); assert D.get("mode") == "sample", D
sl = {(r["model"], r["dataset"]): r for r in csv.DictReader(open(RES/"expR62_samplelevel_record.csv"))}
s_m, s_ds = D["sample_cell"]; c_m, c_ds = D["class_cell"]
cc = {(r["model"], r["dataset"]): r for r in census}[(c_m, c_ds)]
DSN = {"dtd": "DTD", "cifar100": "CIFAR-100", "cifar10": "CIFAR-10", "imagenet": "ImageNet"}
cells = [(NM[s_m] + "\n" + DSN[s_ds] + "\nimages", fam_color(s_m), float(sl[(s_m, s_ds)]["delta_999"]), float(sl[(s_m, s_ds)]["null_mean"])),
         (NM[c_m] + "\n" + DSN[c_ds] + "\ncentroids", fam_color(c_m), float(cc["delta"]), float(cc["null_mean"]))]
print(f"(b) {s_m}/{s_ds} raw {cells[0][2]:.3f} null {cells[0][3]:.3f} | {c_m}/{c_ds} raw {cells[1][2]:.3f} null {cells[1][3]:.3f}")
for i, (lab, col, raw, null) in enumerate(cells):
    ax.plot([i, i], [null, raw], "-", color=GRAY, lw=0.7, zorder=1)
    ax.scatter(i, raw, s=18, color=col, zorder=3)
    ax.scatter(i, null, s=26, facecolors="none", edgecolors=GRAY, linewidths=0.9, zorder=4)
ax.set_xticks([0, 1]); ax.set_xticklabels([c[0] for c in cells], fontsize=8, linespacing=1.15); ax.set_xlim(-0.75, 1.75)
ax.set_ylabel(r"raw $\hat\delta_{99.9}$"); ax.set_title("(b) same raw reading, opposite verdicts"); tidy(ax)
# ---- one legend below both panels
hd = [plt.Line2D([], [], marker="o", ls="", color="k", ms=4, label="filled: the raw reading"),
      plt.Line2D([], [], marker="o", ls="", mfc="none", mec=GRAY, ms=5, label="hollow: a structureless cloud of the same shape")]
hd += [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLORS[k], ms=4, label=l) for k, l in [("supervised", "sup."), ("ssl", "SSL"), ("contrastive", "contr.")]]
fig.legend(handles=hd, frameon=False, loc="lower center", ncol=5, fontsize=7, handlelength=1.0, handletextpad=0.3, columnspacing=0.9, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.44, wspace=0.40)
for o in (HERE, HERE.parent/"iclr2027"/"figures"):
    fig.savefig(o/"fig_overview.pdf"); fig.savefig(o/"fig_overview.png", dpi=200)
print("fig_overview written")
