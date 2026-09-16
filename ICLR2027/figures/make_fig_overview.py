#!/usr/bin/env python3
"""Figure 2 (overview, final pass 6.2), three panels:
(a) raw delta_norm vs d: iid-Gaussian curve (exp1_delta_controls.csv, variant gauss) with the
    12 models' raw ImageNet delta overlaid at their d (family colors);
(b) same raw, opposite verdict: paired bars raw/excess for BGE-base (expR48, padding-free)
    and the vision cell with the closest raw delta (picked from the census data);
(c) the map: DINOv2-B/L/G-vs-block mean ARI, naive vs selected, ImageNet and CIFAR-100
    (exp23_treemap_controls.npz, big_vs_sup)."""
import csv, os, sys, importlib
from pathlib import Path
import numpy as np, numpy.core as _core
sys.modules.setdefault("numpy._core", _core)
for _s in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core."+_s, importlib.import_module("numpy.core."+_s))
    except Exception: pass
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
plt.style.use(str(HERE / "style.mplstyle"))
sys.path.insert(0, str(HERE))
from palette import FAMILY_COLORS, color as fam_color

from matplotlib.ticker import MaxNLocator, FuncFormatter
def _fmt2(v, pos=None):
    s = f"{v:.2f}"; return "0.00" if s in ("-0.00", "0.00") else s
def tidy(ax, decimals=True, n=3):
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n, min_n_ticks=2))
    if decimals: ax.yaxis.set_major_formatter(FuncFormatter(_fmt2))
    ax.tick_params(labelsize=7)

src = RES/"expR52_census_haar_p999_200.csv"
if not src.exists(): src = RES/"expR39c_census200_cache.csv"
if not src.exists(): src = RES/"exp20_null_ztable.csv"
print("census source:", src.name)
census = list(csv.DictReader(open(src)))
d_in = {r["model"]: (float(r["delta"]), float(r["excess"]), float(r["null_mean"])) for r in census if r["dataset"]=="imagenet"}
DIMS = {"i21k_t":192,"i21k_s":384,"i21k_b":768,"i21k_l":1024,"dinov1_b":768,"dinov2_s":384,
        "dinov2_b":768,"dinov2_l":1024,"dinov2_g":1536,"clip_b":512,"clip_l":768,"siglip_b":768}

fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.3), gridspec_kw={"width_ratios":[1.2,1,1]})
# (a) gauss curve + raw deltas
for m, d in DIMS.items():
    if m in d_in:
        axes[0].scatter(d, d_in[m][2], s=14, facecolors="none", edgecolors=FAMILY_COLORS["null"], linewidths=0.8, zorder=2)
        axes[0].plot([d, d], [d_in[m][2], d_in[m][0]], "-", color=FAMILY_COLORS["null"], lw=0.5, zorder=1)
        axes[0].scatter(d, d_in[m][0], s=16, color=fam_color(m), zorder=3)
axes[0].scatter([], [], s=14, facecolors="none", edgecolors=FAMILY_COLORS["null"], label="matched null")
axes[0].set_xlabel("dimension $d$"); axes[0].set_ylabel(r"raw $\hat\delta_{99.9}$ (ImageNet)")
axes[0].set_title("(a) raw readings vs their\nspectrum-matched nulls")
_fam = [plt.Line2D([], [], marker="o", ls="", color=FAMILY_COLORS[k], ms=4, label=l) for k, l in
        [("supervised", "sup."), ("ssl", "SSL"), ("contrastive", "contr.")]]
axes[0].set_xlim(80, 2450)
_h0 = axes[0].get_legend_handles_labels()[0] + _fam
# (b) same raw, opposite verdict: BGE-base vs the closest vision cell, or (restructuring, decided by the numbers in
#     phaseC_fig2b.json) a sample-level cell within null noise vs the class-level cell with the same raw value
import json as _json
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
      "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
      "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
DSN = {"imagenet":"IN","cifar100":"C100","cifar10":"C10","dtd":"DTD","fashionmnist":"FMN","mnist":"MN"}
_dec = RES/"phaseC_fig2b.json"
if _dec.exists() and _json.load(open(_dec)).get("mode") == "sample":
    D = _json.load(open(_dec)); sl = {(r["model"], r["dataset"]): r for r in csv.DictReader(open(RES/"expR62_samplelevel_record.csv"))}
    s_m, s_ds = D["sample_cell"]; c_m, c_ds = D["class_cell"]
    a_d, a_e = float(sl[(s_m, s_ds)]["delta_999"]), float(sl[(s_m, s_ds)]["excess"])
    cc = {(r["model"], r["dataset"]): r for r in census}[(c_m, c_ds)]; v_name, v_ds, v_d, v_e = c_m, c_ds, float(cc["delta"]), float(cc["excess"])
    a_col, a_lab = fam_color(s_m), f"{NM[s_m]} {DSN[s_ds]}\nimages"; v_lab = f"{NM[v_name]} {DSN[v_ds]}\ncentroids"
    print(f"(b) sample-level pair: {s_m}/{s_ds} raw {a_d:.3f} exc {a_e:+.3f} vs class {c_m}/{c_ds} raw {v_d:.3f} exc {v_e:+.3f}")
else:
    _t = RES/"expR53_text_haar_p999_200.csv"
    if not _t.exists(): _t = RES/"expR48b_text_census200_bs1.csv"
    t48 = {r["model"]: r for r in csv.DictReader(open(_t))}
    a_d, a_e = float(t48["bge_base"]["delta"]), float(t48["bge_base"]["excess"])
    cand = min(census, key=lambda r: abs(float(r["delta"]) - a_d))
    v_name, v_ds, v_d, v_e = cand["model"], cand["dataset"], float(cand["delta"]), float(cand["excess"])
    a_col, a_lab, v_lab = fam_color("bge_base"), "BGE-base", f"{NM[v_name]} ({DSN[v_ds]})"
    print(f"closest vision cell to BGE-base raw {a_d:.3f}: {v_name}/{v_ds} raw {v_d:.3f} exc {v_e:+.3f}")
X = np.arange(2)
axes[1].bar(X-0.16, [a_d, v_d], width=0.3, color="white",
            edgecolor=[a_col, fam_color(v_name)], linewidth=1.2, label="raw")
axes[1].bar(X+0.16, [a_e, v_e], width=0.3,
            color=[a_col, fam_color(v_name)], label="excess")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xticks(X); axes[1].set_xticklabels([a_lab, v_lab], fontsize=7)
axes[1].set_title("(b) same raw value,\nopposite verdict")
_h1 = axes[1].get_legend_handles_labels()[0]
# (c) the map before/after
z = np.load(RES/"exp23_treemap_controls.npz", allow_pickle=True)
zi, zc = z["summary_in"].item(), z["summary_c1"].item()
import json as _json
diag = _json.load(open(RES/"exp23_config_diagnostics.json"))
def sel(ds):
    adm = {k: v for k, v in diag.items() if k.startswith(ds+"|") and v["maxfrac"] <= 0.5}
    m, l = max(adm, key=lambda k: adm[k]["cpcc"]).split("|")[1:]
    return f"('{m}', '{l}')"
vals = [zi["('euclid', 'average')"]["big_vs_sup"], zi[sel("imagenet")]["big_vs_sup"],
        zc["('euclid', 'average')"]["big_vs_sup"], zc[sel("cifar100")]["big_vs_sup"]]
pos = [0, 0.35, 1.0, 1.35]
cols = [FAMILY_COLORS["null"], FAMILY_COLORS["ssl"], FAMILY_COLORS["null"], FAMILY_COLORS["ssl"]]
axes[2].bar(pos, vals, width=0.3, color=cols)
for x, v in zip(pos, vals): axes[2].text(x, v+0.012, f"{v:.2f}", ha="center", fontsize=7)
axes[2].set_xticks([0.175, 1.175]); axes[2].set_xticklabels(["ImageNet", "CIFAR-100"])
axes[2].set_ylabel("DINOv2 vs block (ARI)")
axes[2].set_title("(c) the island is the\nclustering step's artifact")
import matplotlib.patches as mpatches
_h2 = [mpatches.Patch(color=FAMILY_COLORS["null"], label="naive"), mpatches.Patch(color=FAMILY_COLORS["ssl"], label="selected")]
for ax in axes: tidy(ax)
axes[2].set_ylim(0, max(vals) * 1.35)
fig.legend(handles=_h0 + _h1 + _h2, frameon=False, loc="lower center", ncol=8, fontsize=7, handlelength=1.1, handletextpad=0.3, columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.44, wspace=0.55)
for o in (HERE, HERE.parent/"iclr2027"/"figures"):
    fig.savefig(o/"fig_overview.pdf"); fig.savefig(o/"fig_overview.png", dpi=200)
print("fig_overview written")
