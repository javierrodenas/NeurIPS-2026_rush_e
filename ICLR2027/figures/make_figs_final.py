#!/usr/bin/env python3
"""Final-version figures (author's brief 'Final version — plain, short, nine pages', 2026-09-18), bar language for every data figure:
light dashed grid behind the axes, family palette, 8 pt labels, model names as tick labels, filled = passes the test, hatched = does
not, the threshold or the null as a dashed line or a gray band, numbers only on bars that pass, short legends inside the panel.
Writes fig_overview_final, fig_excess_final, fig_depth_final, fig_treemap_final and the appendix fig_implant_final (pdf+png) next to the other figures; the v1/v2/v3
figure files are untouched. Data: expR52_census_haar_p999_200.csv, expR62_samplelevel_record.csv, exp1_delta_controls.csv,
expR56_depth_variants.csv, expR64b_wn30.csv, exp23_treemap_controls.npz, phaseC_fig2b.json."""
import csv, os, sys, json, math, importlib
from pathlib import Path
import numpy as np, numpy.core as _core
sys.modules.setdefault("numpy._core", _core)
for _s in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core."+_s, importlib.import_module("numpy.core."+_s))
    except Exception: pass
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
plt.style.use(str(HERE / "style.mplstyle"))
plt.rcParams.update({"xtick.labelsize": 8, "ytick.labelsize": 8, "axes.labelsize": 8, "legend.fontsize": 8, "axes.titlesize": 8, "hatch.linewidth": 0.6})
sys.path.insert(0, str(HERE))
from palette import FAMILY_COLORS, color as fam_color, BAND, BAND_ALPHA, BAND_EDGE, BAND_LW
GRAY = FAMILY_COLORS["null"]; LIGHT = "#C9C9C9"
LEG = dict(frameon=True, fancybox=True, shadow=True, framealpha=0.95, facecolor="white", edgecolor="#DDDDDD")   # legends: white rounded box with a soft shadow (brief of 2026-09-24)
M = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
DIMS = {"i21k_t":192,"i21k_s":384,"i21k_b":768,"i21k_l":1024,"dinov1_b":768,"dinov2_s":384,"dinov2_b":768,"dinov2_l":1024,"dinov2_g":1536,"clip_b":512,"clip_l":768,"siglip_b":768}
DSO = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]; DSL = {"imagenet":"ImageNet","cifar100":"CIFAR-100","cifar10":"CIFAR-10","dtd":"DTD","fashionmnist":"FMNIST","mnist":"MNIST"}
census = list(csv.DictReader(open(RES/"expR75_census_centered_haar.csv")))   # the record since 2026-09-20: centered Haar null (Q orthogonal to the all-ones vector)
by = {(r["model"], r["dataset"]): r for r in census}
def save(fig, name):
    for o in (HERE, HERE.parent/"iclr2027"/"figures"): fig.savefig(o/f"{name}.pdf"); fig.savefig(o/f"{name}.png", dpi=200)
    print(name, "written")
def wilson(p, n, z=1.96):
    """95 per cent interval for a proportion; n is the run count of expR81 (50 per backbone, 200 for the DINOv2 family)."""
    den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h
def whisker(ax, x, lo, hi, lw=0.7, color="0.25", zorder=6):
    ax.plot([x, x], [lo, hi], "-", color=color, lw=lw, solid_capstyle="butt", zorder=zorder)
def hbar(ax, y, w, color, passes, height=0.72, zorder=3):
    if passes: return ax.barh(y, w, height, color=color, edgecolor="white", linewidth=0.8, zorder=zorder)
    return ax.barh(y, w, height, facecolor="white", edgecolor=color, hatch="////", linewidth=0.6, zorder=zorder)
def bar(ax, x, h, color, passes, width=0.38, zorder=3):
    if passes: return ax.bar(x, h, width, color=color, edgecolor="white", linewidth=0.8, zorder=zorder)
    return ax.bar(x, h, width, facecolor="white", edgecolor=color, hatch="////", linewidth=0.6, zorder=zorder)


# ---------------- Figure 2 (author's brief, 2026-09-24, 2x2): the instrument in one figure. The same x axis in the four panels:
# the 12 ImageNet backbones by family with a one-position gap; color means family throughout.
FAMS = [("supervised", ["i21k_t", "i21k_s", "i21k_b", "i21k_l"], 2.0), ("DINO", ["dinov1_b"], 1.4), ("DINOv2", ["dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g"], 2.0),
        ("contrastive", ["clip_b", "clip_l", "siglip_b"], 0.0)]   # DINO and DINOv2 keep their own gap, under one family name
FAMLAB = [("supervised", ["i21k_t", "i21k_s", "i21k_b", "i21k_l"]), ("self-supervised", ["dinov1_b", "dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g"]),
          ("contrastive", ["clip_b", "clip_l", "siglip_b"])]   # the family names in words (author's brief, 2026-09-24)
PLANT = "#222222"   # the planted three-level hierarchy: black, dashed, triangles (author's brief, 2026-09-24)
XPOS, XFAM, XSEP, _x = {}, [], [], 0.0
for _fname, _ms, _gap in FAMS:
    _start = _x
    for _m in _ms: XPOS[_m] = _x; _x += 1.0
    XFAM.append((_fname, (_start + _x - 1.0) / 2))
    if _gap: XSEP.append(_x - 1.0 + _gap / 2); _x += _gap
XLIM = (-0.9, _x - 1.0 + 0.9)
def famaxis(ax, names=True):
    ax.set_xlim(*XLIM)
    for xs_ in XSEP: ax.axvline(xs_, color="white", lw=1.4, zorder=0)
    ax.set_xticks([XPOS[m] for m in M]); ax.set_xticklabels([NM[m] for m in M] if names else [], fontsize=6.5, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=2, width=0.6, color="#BBBBBB", pad=1.5)
    if names:
        for fname, ms_ in FAMLAB:
            ax.annotate(fname, xy=((XPOS[ms_[0]] + XPOS[ms_[-1]]) / 2, 0), xytext=(0, -34), xycoords=("data", "axes fraction"), textcoords="offset points", ha="center", va="top", fontsize=6.8, color="0.25")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
def curve(ax, vals, style, color=None, marker="o", ms=4.4, filled=None, zorder=3):
    """One curve per family: the models joined, with their markers on top."""
    for _fname, ms_, _ in FAMS:
        ax.plot([XPOS[m] for m in ms_], [vals[m] for m in ms_], ls=style, color=(color or fam_color(ms_[0])), lw=(1.5 if style == "-" else 1.0), zorder=zorder)
    for m in M:
        c = color or fam_color(m)
        if filled is None or filled[m]: ax.plot(XPOS[m], vals[m], marker, color=c, ms=ms, mec="white", mew=0.5, zorder=zorder + 1)
        else: ax.plot(XPOS[m], vals[m], marker, mfc="white", mec=c, ms=ms, mew=1.3, zorder=zorder + 1)
IN = {m: by[(m, "imagenet")] for m in M}
D74 = {r["model"]: r for r in csv.DictReader(open(RES/"expR74_decoupling_summary.csv"))}
D81 = {r["model"]: r for r in csv.DictReader(open(RES/"expR81_deep_per_backbone_summary.csv"))}
DOT = (0, (1, 1.6))
fig, axes = plt.subplots(2, 2, figsize=(5.5, 3.45))
# (a) the raw reading and a random cloud of the same shape
ax = axes[0][0]
_gauss = {int(r["d"]): float(r["delta_max"]) for r in csv.DictReader(open(RES/"exp1_delta_controls.csv")) if r["variant"] == "gauss"}   # iid Gaussian: the dimension alone, no spectrum (Table 4)
for _fname, ms_, _ in FAMS:
    ax.plot([XPOS[m] for m in ms_], [_gauss[DIMS[m]] for m in ms_], "--", color="#C9C9C9", lw=0.8, zorder=1)
for m in M: ax.plot(XPOS[m], _gauss[DIMS[m]], "s", mfc="white", mec="#B6B6B6", ms=3.6, mew=0.9, zorder=2)
curve(ax, {m: float(IN[m]["null_mean"]) for m in M}, DOT, color=GRAY, marker="D", ms=3.4, zorder=2)
curve(ax, {m: float(IN[m]["delta"]) for m in M}, "-", filled={m: IN[m]["genuine_bh"] == "True" for m in M}, ms=4.6, zorder=4)
ax.set_ylabel(r"$\delta_{\mathrm{norm}}$"); ax.set_title("(a) raw reading, random ball and random cloud", pad=3); famaxis(ax, names=False)
# (b) the excess, filled when genuine, with the two-null-s.d. mark
ax = axes[0][1]
for m in M:
    e = float(IN[m]["excess"]); sd = float(IN[m]["null_sd"])
    bar(ax, XPOS[m], e, fam_color(m), IN[m]["genuine_bh"] == "True", width=0.74)
    ax.plot([XPOS[m] - 0.48, XPOS[m] + 0.48], [-2 * sd, -2 * sd], "-", color="0.25", lw=1.1, solid_capstyle="butt", zorder=5)
ax.axhline(0, color="0.35", lw=0.6, zorder=1); ax.set_ylim(top=0.003)
ax.set_ylabel("excess"); ax.set_title("(b) excess over the random cloud", pad=3); famaxis(ax, names=False)
# (c) the hierarchy test: the real cloud, the real cloud randomized, and a planted hierarchy randomized
ax = axes[1][0]
ax.axhline(-2, color="k", lw=0.8, ls="--", zorder=1)
curve(ax, {m: float(D74[m]["dec_z_mean"]) for m in M}, DOT, color=GRAY, marker="D", ms=3.4, zorder=2)
curve(ax, {m: float(D81[m]["dec_z_mean"]) for m in M}, (0, (4, 1.8)), color=PLANT, marker="^", ms=4.4, zorder=3)
curve(ax, {m: float(D74[m]["real_z"]) for m in M}, "-", ms=4.6, zorder=5)
ax.set_ylabel("hierarchy test $z$"); ax.set_title("(c) hierarchy test, real and planted", pad=3); famaxis(ax)
# (d) the power for the planted hierarchy, intact and randomized
ax = axes[1][1]
for m in M:
    p_int, p_dec = float(D81[m]["power"]), float(D81[m]["dec_power"])   # pd is pandas: do not shadow it
    ax.bar(XPOS[m] - 0.2, p_int, 0.38, color=LIGHT, edgecolor="white", linewidth=0.8, zorder=3)
    bar(ax, XPOS[m] + 0.2, p_dec, fam_color(m), p_dec >= 0.8, width=0.38)
    whisker(ax, XPOS[m] + 0.2, *wilson(p_dec, int(float(D81[m]["dec_runs"]))))   # 95 per cent interval (author's brief, 2026-09-25)
ax.axhline(0.8, color="k", lw=0.8, ls="--", zorder=2)
ax.set_ylim(0, 1.1); ax.set_yticks([0, 0.5, 0.8, 1]); ax.set_yticklabels(["0", "0.5", "0.8", "1"])
ax.set_ylabel("power"); ax.set_title("(d) power for the planted hierarchy", pad=3); famaxis(ax)
hd = [plt.Line2D([], [], marker="o", color="k", ls="-", lw=1.5, ms=4.6, mec="white", mew=0.5, label="model"),
      plt.Line2D([], [], marker="D", color=GRAY, ls=DOT, lw=1.0, ms=3.4, mec="white", mew=0.5, label="random cloud (a) / real, randomized (c)"),
      plt.Line2D([], [], marker="s", mfc="white", mec="#B6B6B6", ls="--", color="#C9C9C9", lw=0.8, ms=3.6, mew=0.9, label="random ball, same dimension (a)"),
      plt.Line2D([], [], marker="^", color=PLANT, ls=(0, (4, 1.8)), lw=1.5, ms=4.4, mec="white", mew=0.5, label="planted hierarchy, randomized (c)"),
      Patch(color="k", label="genuine / power $\\geq0.8$"),
      Patch(facecolor="white", edgecolor="k", hatch="////", label="not genuine (b) / no power (d)"),
      Patch(color=LIGHT, label="power intact (d)")]
fig.legend(handles=hd, **LEG, loc="lower center", ncol=3, handlelength=1.5, handletextpad=0.45, columnspacing=1.1, bbox_to_anchor=(0.5, -0.12))
fig.subplots_adjust(left=0.085, right=0.995, top=0.95, bottom=0.30, wspace=0.24, hspace=0.42)
_covered = [m for m in M if float(D81[m]["dec_power"]) >= 0.8]
json.dump({"legend": [h.get_label() for h in hd], "implanted_alignment_curve": False, "panel_b": "decoupled_power_per_backbone", "n_covered": len(_covered), "covered": _covered}, open(RES/"final_fig4.json", "w"), indent=1)
save(fig, "fig_instrument_final")
print(f"instrument: genuine {sum(IN[m]['genuine_bh'] == 'True' for m in M)}/12, certified {sum(float(D74[m]['real_z']) <= -2 for m in M)}/12, power>=0.8 in {len(_covered)}/12")


# ---------------- Figure 3 (author's brief, 2026-09-24, late): the premise where it is read, in the language of Figure 4(a).
SL = {(r["model"], r["dataset"]): r for r in csv.DictReader(open(RES/"expR62_samplelevel_record.csv"))}
K78 = {r["dataset"]: r for r in csv.DictReader(open(RES/"expR78_khrulkov_replication_summary.csv"))}
K85 = list(csv.DictReader(open(RES/"expR85_khrulkov_sup.csv")))
fig = plt.figure(figsize=(5.5, 1.95))
_gs3 = fig.add_gridspec(1, 2, width_ratios=[2.6, 3.0], wspace=0.34)
# (a) the 24 sample-level cells as bars, two columns, the 12 backbones on y
_ax3 = _gs3[0].subgridspec(1, 2, wspace=0.12).subplots(sharey=True)
Y3 = np.arange(len(M))[::-1]
for ax, (ds, lab) in zip(_ax3, (("cifar100", "CIFAR-100"), ("dtd", "DTD"))):
    sd = np.median([float(SL[(m, ds)]["null_sd"]) for m in M])
    ax.axvspan(-2*sd, 2*sd, facecolor=BAND, alpha=BAND_ALPHA, edgecolor=BAND_EDGE, lw=BAND_LW, zorder=0); ax.axvline(0, color="k", lw=0.6, zorder=1)
    for _i, m in enumerate(M): hbar(ax, Y3[_i], float(SL[(m, ds)]["excess"]), fam_color(m), SL[(m, ds)]["genuine_bh"] == "True")
    ax.set_yticks(Y3); ax.set_yticklabels([NM[m] for m in M]); ax.set_ylim(-0.7, len(M) - 0.3)
    ax.set_xlim(-0.030, 0.014); ax.set_xticks([-0.02, 0]); ax.set_xticklabels(["\u22120.02", "0"])
    ax.set_title(lab, fontsize=7.2); ax.tick_params(axis="y", length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
_ax3[0].annotate("(a) per-image features", xy=(0, 1), xytext=(0, 15), xycoords="axes fraction", textcoords="offset points", ha="left", va="baseline", fontsize=8)
# (b) the published reading against a random cloud of the same shape, on their statistic
axb3 = fig.add_subplot(_gs3[1])
DK3 = [("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100"), ("cub", "CUB-200"), ("miniimagenet", "MiniImageNet")]
_col3 = fam_color("i21k_b")   # ResNet-34, a supervised backbone
YK = np.arange(len(DK3))[::-1]
_inside = []
for _i, (_d, _lab) in enumerate(DK3):
    _rows = [r for r in K85 if r["dataset"] == _d]
    _nul = sum(float(r["null_mean"]) for r in _rows) / len(_rows); _sd = sum(float(r["null_sd"]) for r in _rows) / len(_rows)
    _ours = float(K78[_d]["ours_raw_mean"]); _in = _ours >= _nul - 2 * _sd; _inside.append(_in)
    axb3.barh(YK[_i], 4 * _sd, 0.62, left=_nul - 2 * _sd, facecolor=BAND, alpha=BAND_ALPHA, edgecolor=BAND_EDGE, lw=BAND_LW, zorder=0)
    axb3.plot(float(K78[_d]["theirs"]), YK[_i] + 0.26, "*", color="0.30", ms=7.0, mec="white", mew=0.5, zorder=4)
    if _in: axb3.plot(_ours, YK[_i] - 0.10, "o", mfc="white", mec=_col3, ms=4.6, mew=1.2, zorder=5)
    else: axb3.plot(_ours, YK[_i] - 0.10, "o", color=_col3, ms=4.6, mec="white", mew=0.6, zorder=5)
    axb3.annotate(f"{_ours:.2f}", (_ours, YK[_i] - 0.10), xytext=(5, -0.5), textcoords="offset points", fontsize=6.3, color=_col3, va="center", ha="left", zorder=6)
axb3.set_yticks(YK); axb3.set_yticklabels([l for _, l in DK3], fontsize=6.5); axb3.set_ylim(-0.7, len(DK3) - 0.3)
axb3.set_xlabel(r"$\delta_{\mathrm{rel}}$ on their statistic", labelpad=1); axb3.tick_params(axis="y", length=0)
axb3.annotate("(b) the values of Khrulkov et al., calibrated", xy=(0, 1), xytext=(0, 15), xycoords="axes fraction", textcoords="offset points", ha="left", va="baseline", fontsize=8)
for sp in ("top", "right"): axb3.spines[sp].set_visible(False)
fig.text(0.215, 0.085, "excess over the matched null", ha="center", va="center", fontsize=8)
hd = [Patch(color="k", label="genuine (a) / below the band (b)"), Patch(facecolor="white", edgecolor="k", hatch="////", label="not genuine (a)"),
      plt.Line2D([], [], marker="o", mfc="white", mec="k", ls="", ms=4.6, mew=1.2, label="inside the band (b)"),
      plt.Line2D([], [], marker="*", color="0.30", ls="", ms=7.0, mec="white", mew=0.5, label="published value (b)"),
      Patch(facecolor=BAND, edgecolor=BAND_EDGE, lw=BAND_LW, label="random cloud, $\\pm2$ s.d.")]
fig.legend(handles=hd, **LEG, loc="lower center", ncol=3, handlelength=1.3, handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.5, -0.21))
fig.subplots_adjust(left=0.115, right=0.995, top=0.87, bottom=0.235)
_gen24 = sum(SL[(m, ds)]["genuine_bh"] == "True" for m in M for ds in ("cifar100", "dtd"))
json.dump({"sample_genuine": _gen24, "n_cells": 24, "inside_band": [d for (d, _), _in in zip(DK3, _inside) if _in],
           "published": {d: float(K78[d]["theirs"]) for d, _ in DK3}, "ours_sup": {d: float(K78[d]["ours_raw_mean"]) for d, _ in DK3}}, open(RES/"final_fig_premise.json", "w"), indent=1)
save(fig, "fig_premise_final")
print(f"premise: {_gen24}/24 genuine; inside the band: {[d for (d, _), _in in zip(DK3, _inside) if _in]}")


# ---------------- Figure 3: six narrow panels of twelve horizontal bars (excess), filled when genuine, hatched when not, the null band around zero
fig = plt.figure(figsize=(5.5, 2.2))
_gs = fig.add_gridspec(1, 2, width_ratios=[4.25, 1.6], wspace=0.44)
axg = _gs[0].subgridspec(1, 6, wspace=0.12).subplots(sharey=True)
Y = np.arange(len(M))[::-1]
for ax, ds in zip(axg, DSO):
    sd = np.median([float(by[(m, ds)]["null_sd"]) for m in M])
    ax.axvspan(-2*sd, 2*sd, facecolor=BAND, alpha=BAND_ALPHA, edgecolor=BAND_EDGE, lw=BAND_LW, zorder=0); ax.axvline(0, color="k", lw=0.6, zorder=1)
    for i, m in enumerate(M):
        r = by[(m, ds)]; hbar(ax, Y[i], float(r["excess"]), fam_color(m), str(r["genuine_bh"]) == "True")
    ax.set_yticks(Y); ax.set_yticklabels([NM[m] for m in M]); ax.set_ylim(-0.7, len(M) - 0.3)
    ax.set_xlim(-0.145, 0.03); ax.set_xticks([-0.12, -0.06, 0]); ax.set_xticklabels(["\u22120.12", "", "0"])   # the middle tick keeps its gridline; its label crowded the 8 pt ticks (style brief, 2026-09-24)
    ax.set_title(DSL[ds], fontsize=7.2); ax.tick_params(axis="y", length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
axg[0].annotate("(a) class centroids, per dataset", xy=(0, 1), xytext=(0, 24), xycoords="axes fraction", textcoords="offset points", ha="left", va="baseline", fontsize=8)
# ---- (b) the 15 text models on the ImageNet class names, in the language of (a) (author's brief, 2026-09-24)
TXT = {r["model"]: r for r in csv.DictReader(open(RES/"expR53_text_haar_p999_200.csv"))}
TXTORD = [("gpt2", "GPT-2 S"), ("gpt2_m", "GPT-2 M"), ("gpt2_l", "GPT-2 L"), ("gpt2_xl", "GPT-2 XL"), ("pythia_410m", "Pythia-410M"), ("pythia_1b", "Pythia-1B"),
          ("pythia_2b8", "Pythia-2.8B"), ("olmo_1b", "OLMo-1B"), ("bge_base", "BGE-base"), ("bge_large", "BGE-large"), ("gte_base", "GTE-base"), ("gte_large", "GTE-large"),
          ("gte_qwen2", "GTE-Qwen2-1.5B"), ("e5_base", "E5-base"), ("e5_large", "E5-large")]
LM, EMB = FAMILY_COLORS["causal_lm"], GRAY
axb = fig.add_subplot(_gs[1])
_sdT = np.median([float(TXT[m]["null_sd"]) for m, _ in TXTORD])
axb.axvspan(-2*_sdT, 2*_sdT, facecolor=BAND, alpha=BAND_ALPHA, edgecolor=BAND_EDGE, lw=BAND_LW, zorder=0); axb.axvline(0, color="k", lw=0.6, zorder=1)
YT = np.arange(len(TXTORD))[::-1]
for _i, (m, _lab) in enumerate(TXTORD):
    hbar(axb, YT[_i], float(TXT[m]["excess"]), LM if _i < 8 else EMB, TXT[m]["genuine_bh"] == "True")
axb.set_yticks(YT); axb.set_yticklabels([l for _, l in TXTORD], fontsize=5.6); axb.set_ylim(-0.7, len(TXTORD) - 0.3)
axb.tick_params(axis="y", length=0); axb.set_xlabel("excess", labelpad=1)
_tb = axb.annotate("(b) text models, ImageNet class names", xy=(0, 1), xytext=(0, 24), xycoords="axes fraction", textcoords="offset points", ha="left", va="baseline", fontsize=8)   # measured below, then replaced by the axes title
for sp in ("top", "right"): axb.spines[sp].set_visible(False)
json.dump({"text_genuine": sum(TXT[m]["genuine_bh"] == "True" for m, _ in TXTORD), "n_text": len(TXTORD), "order": [m for m, _ in TXTORD],
           "excess": {m: float(TXT[m]["excess"]) for m, _ in TXTORD}, "labels": {m: l for m, l in TXTORD}}, open(RES/"final_fig_text.json", "w"), indent=1)
fig.text(0.33, 0.115, "excess over the matched null", ha="center", va="center", fontsize=8)
hd = [Patch(color="k", label="genuine"), Patch(facecolor="white", edgecolor="k", hatch="////", label="not genuine"), Patch(facecolor=BAND, edgecolor=BAND_EDGE, lw=BAND_LW, label="\u00b12 null s.d."),
      Patch(color=LM, label="causal LM (b)"), Patch(color=EMB, label="embedder (b)")]
fig.legend(handles=hd, **LEG, loc="lower center", ncol=5, handlelength=1.2, handletextpad=0.4, columnspacing=1.0, bbox_to_anchor=(0.5, -0.03))
fig.subplots_adjust(left=0.10, right=0.995, top=0.83, bottom=0.27)
# (b) fills the width its left-aligned label used to reach, and the label becomes the centred axes title (author's brief, 2026-09-26); height, y labels and (a) unchanged
fig.canvas.draw(); _x1 = fig.transFigure.inverted().transform(_tb.get_window_extent())[1, 0]; _tb.remove()
_pb = axb.get_position(); axb.set_position([_pb.x0, _pb.y0, _x1 - _pb.x0, _pb.height])
axb.set_title("(b) text models, ImageNet class names", fontsize=8, pad=24)
save(fig, "fig_excess_final")
print(f"text panel: {sum(TXT[m]['genuine_bh'] == 'True' for m, _ in TXTORD)}/{len(TXTORD)} genuine on the class names")

# ---------------- the former Figure 4 (depth bars and power bars) is now panels (c) and (d) of Figure 2 (brief of 2026-09-24)
# ---------------- Appendix figure (ninth review): the two-level implant detection curves, formerly Figure 4b
d64 = pd.read_csv(RES/"expR64b_wn30.csv"); dep = d64[(d64.kind == "depth") & (d64.partition == "rand6") & (d64.s != "real")].copy(); dep["s"] = dep.s.astype(float)
tg = d64[(d64.kind == "depth") & (d64.partition == "rand6_t06")].copy(); tg["s"] = tg.s.astype(float)
pr = dep.groupby("s").z.apply(lambda z: (z <= -2).mean()); pt = tg.groupby("s").z.apply(lambda z: (z <= -2).mean())
fig, ax = plt.subplots(figsize=(2.75, 1.9))
l1, = ax.plot(pr.index, pr.values, "-o", color="k", ms=4, mec="white", mew=0.8, lw=1.0, label="real spread", zorder=3)
l2, = ax.plot(pt.index, pt.values, "--s", color=GRAY, ms=4, mec="white", mew=0.8, lw=1.0, label="shrunk spread", zorder=3)
l3 = None   # priority 1c (expR80): the planted-alignment curve enters only when the decision rule of the brief is met
if (RES/"expR80_decision.csv").exists() and (RES/"expR80_implanted_alignment_summary.csv").exists():
    d80 = list(csv.DictReader(open(RES/"expR80_decision.csv")))[0]
    if d80["rule_power_ge_0_8_fa_le_0_05"] == "True":
        s80 = list(csv.DictReader(open(RES/"expR80_implanted_alignment_summary.csv")))
        l3, = ax.plot([float(r["s"]) for r in s80], [float(r["detection_rate"]) for r in s80], ":^", color=FAMILY_COLORS["supervised"], ms=4, mec="white", mew=0.8, lw=1.0, label="planted alignment", zorder=3)
ax.annotate("no false alarms at $s{=}0$ (real spread)", (0, pr.loc[0.0]), xytext=(12, 5), textcoords="offset points", fontsize=8, ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color="k", lw=0.6))
ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.set_xlabel("planted-tree strength $s$", labelpad=1)
ax.set_ylim(-0.04, 1.08); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0.0", "0.5", "1.0"]); ax.set_ylabel("detection rate")
ax.set_title("planted two-level tree on the real clouds")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(handles=[l1, l2] + ([l3] if l3 is not None else []), **LEG, loc="center right", handlelength=1.4, handletextpad=0.4)
json.dump({"real": {str(float(k)): float(v) for k, v in pr.items()}, "shrunk": {str(float(k)): float(v) for k, v in pt.items()}, "implanted_alignment_curve": l3 is not None}, open(RES/"final_fig_implant.json", "w"), indent=1)   # read by the builder (caption)
fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.22)
save(fig, "fig_implant_final")
print(f"implant: detection real {dict((float(k), round(float(v), 3)) for k, v in pr.items())}; shrunk {dict((float(k), round(float(v), 3)) for k, v in pt.items())}")

# ---------------- Figure 5 (consolidated pass, 2026-09-22): (a) mean ARI at the 30-cut with the other eleven, naive (hollow) against selected (filled), joined;
# (b) triplet agreement under the selected configuration (expR58) against the within-model ceiling band of expR84 (range over resample pairs, line at the mean) and the chance level 1/3 (dashed);
# (c) sibling triplets under cosine and Euclidean distance (CIFAR-100, exp10). The two ARI matrices become an appendix figure (fig_treemap_matrices_final).
z = np.load(RES/"exp23_treemap_controls.npz", allow_pickle=True); S = z["summary_in"].item()
diag = json.load(open(RES/"exp23_config_diagnostics.json")); adm = {k: v for k, v in diag.items() if k.startswith("imagenet|") and v["maxfrac"] <= 0.5}
sel = max(adm, key=lambda k: adm[k]["cpcc"]).split("|")[1:]; SELKEY = "('{}', '{}')".format(*sel); assert SELKEY == "('cosine', 'average')", SELKEY
NAMES = [NM[m] for m in M]; sup = [0, 1, 2, 3, 9, 10, 11]; big = [6, 7, 8]
e10 = {r["model"]: r for r in csv.DictReader(open(RES/"exp10_local_vs_global.csv"))}
Mn = np.asarray(S["('euclid', 'average')"]["ari"], dtype=float); Ms = np.asarray(S[SELKEY]["ari"], dtype=float)
vals = [float(np.mean([Mx[i, j] for i in big for j in sup])) for Mx in (Mn, Ms)]
mean_other = lambda Mx, i: float(np.mean([Mx[i, j] for j in range(len(M)) if j != i]))
p58 = pd.read_csv(RES/"expR58_treemap_cutfree.csv"); p58 = p58[(p58.dataset == "imagenet") & (p58.metric == "cosine") & (p58.linkage == "average")]; assert len(p58) == 66, len(p58)
trip = {m: float(p58[(p58.model_a == m) | (p58.model_b == m)].triplet_agree.mean()) for m in M}
c84 = pd.read_csv(RES/"expR84_tree_ceiling.csv"); c84 = c84[c84.kind == "boot_pair"]
band = {m: (float(c84[c84.model == m].triplet_agree.min()), float(c84[c84.model == m].triplet_agree.max()), float(c84[c84.model == m].triplet_agree.mean())) for m in M}
ys = {}; _y = 0.0   # shared y axis of (a) and (b): backbones top to bottom in the order of M, a small gap between families (layout brief, 2026-09-22)
for i, m in enumerate(M):
    if i in (4, 9): _y -= 0.7
    ys[m] = _y; _y -= 1.0
YLIM = (min(ys.values()) - 0.7, 0.7)
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.2), gridspec_kw={"width_ratios": [1.25, 1.05, 1.35]})   # the names of (a) get their room from the margins (2026-09-24)
ax = axes[0]
for i, m in enumerate(M):
    y = ys[m]; a, b = mean_other(Mn, i), mean_other(Ms, i)
    ax.plot([a, b], [y, y], color=fam_color(m), lw=0.9, zorder=2); ax.plot(a, y, "o", mfc="white", mec=fam_color(m), ms=4.5, mew=1.0, zorder=3); ax.plot(b, y, "o", color=fam_color(m), ms=5, mec="white", mew=0.8, zorder=4)
ax.set_yticks([ys[m] for m in M]); ax.set_yticklabels(NAMES, fontsize=7); ax.set_ylim(*YLIM); ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 0.75); ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75"]); ax.set_xlabel("mean ARI with the other eleven", labelpad=1); ax.set_title("(a) the island is the cut", pad=3)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax = axes[1]
for i, m in enumerate(M):
    y = ys[m]; lo, hi, mu = band[m]
    ax.barh(y, hi - lo, 0.62, left=lo, facecolor=BAND, alpha=BAND_ALPHA, edgecolor=BAND_EDGE, lw=BAND_LW, zorder=0); ax.plot([mu, mu], [y - 0.32, y + 0.32], color="0.35", lw=0.8, zorder=2); ax.plot(trip[m], y, "o", color=fam_color(m), ms=5, mec="white", mew=0.8, zorder=4)
ax.set_yticks([ys[m] for m in M]); ax.set_yticklabels([]); ax.set_ylim(*YLIM); ax.tick_params(axis="y", length=0)
ax.axvline(1 / 3, color="k", lw=0.8, ls="--", zorder=2)   # chance for the three-way triplet choice (which of the three pairs merges first); caption brief of 2026-09-23
ax.set_xlim(0.25, 1.0); ax.set_xticks([1 / 3, 0.6, 0.8, 1.0]); ax.set_xticklabels(["0.33", "0.6", "0.8", "1.0"]); ax.set_xlabel("triplet agreement", labelpad=1); ax.set_title("(b) topology, chance and ceiling", pad=3)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax = axes[2]; tri = {}
for i, m in enumerate(M):
    c, e = float(e10[m]["c100_sibtrip_c"]), float(e10[m]["c100_sibtrip_e"]); tri[m] = (c, e)
    ax.bar(i - 0.2, c, 0.4, color=fam_color(m), edgecolor="white", linewidth=0.8, zorder=3); ax.bar(i + 0.2, e, 0.4, facecolor="white", edgecolor=fam_color(m), hatch="////", linewidth=0.6, zorder=3)
ax.axhline(0.5, color="k", lw=0.8, ls="--", zorder=2)
ax.set_xticks(range(len(M))); ax.set_xticklabels(NAMES, rotation=90, fontsize=7); ax.set_xlim(-0.7, len(M) - 0.3); ax.tick_params(axis="x", length=0)
# the names of (c) are black (author's note, 2026-09-24); the bars carry the family color
ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0.0", "0.5", "1.0"]); ax.set_ylabel("triplet agreement", labelpad=2)
ax.set_title("(c) sibling triplets, CIFAR-100", pad=3)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
hd = [plt.Line2D([], [], marker="o", mfc="white", mec="k", ls="", ms=4.5, mew=1.0, label="naive"), plt.Line2D([], [], marker="o", color="k", ls="", ms=5, mec="white", mew=0.8, label="corrected"), Patch(facecolor=BAND, edgecolor=BAND_EDGE, lw=BAND_LW, label="within-model ceiling"),
      Patch(color="k", label="cosine"), Patch(facecolor="white", edgecolor="k", hatch="////", label="Euclidean"), plt.Line2D([], [], ls="--", color="k", lw=0.8, label="chance")]
fig.legend(handles=hd, **LEG, loc="lower center", ncol=6, handlelength=1.2, handletextpad=0.4, columnspacing=0.9, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.11, right=0.99, top=0.93, bottom=0.34, wspace=0.45)
save(fig, "fig_treemap_final")
json.dump({"naive_dinov2_vs_block": vals[0], "selected_dinov2_vs_block": vals[1], "sibtrip_c100": {m: {"cosine": tri[m][0], "euclid": tri[m][1]} for m in M}, "mean_ari_other": {m: [mean_other(Mn, i), mean_other(Ms, i)] for i, m in enumerate(M)}, "triplet_selected": trip, "ceiling_band": band}, open(RES/"final_fig5_values.json", "w"), indent=1)
d2 = {m: tri[m][0] - tri[m][1] for m in M}; print(f"treemap: DINOv2-B/L/G vs block naive {vals[0]:.2f}, selected {vals[1]:.2f}; triplet cosine-Euclid gap: DINOv2 {min(d2[m] for m in ('dinov2_b','dinov2_l','dinov2_g')):.2f}..{max(d2[m] for m in ('dinov2_b','dinov2_l','dinov2_g')):.2f}")
# ---------------- Appendix figure: the two ARI matrices, formerly Figure 5ab
cmap = LinearSegmentedColormap.from_list("family_seq", ["#FFFFFF", "#9FBFD6", FAMILY_COLORS["supervised"], "#123B57"])
W, H = 4.2, 2.2; s = 1.45; bot = 0.62
fig = plt.figure(figsize=(W, H)); axes = [fig.add_axes([0.55/W, bot/H, s/W, s/H]), fig.add_axes([(0.55+s+0.12)/W, bot/H, s/W, s/H])]; cax = fig.add_axes([(0.55+2*s+0.20)/W, bot/H, 0.07/W, s/H])
for ax, (Mx, title) in zip(axes, [(Mn, "(a) naive: Euclidean, average"), (Ms, "(b) selected: cosine, average")]):
    im = ax.imshow(Mx, vmin=0, vmax=1, cmap=cmap); ax.grid(False); ax.add_patch(Rectangle((4.5, 4.5), 4, 4, fill=False, edgecolor=FAMILY_COLORS["ssl"], lw=1.0, zorder=4))
    ax.set_title(title, pad=3); ax.set_xticks(range(12)); ax.set_xticklabels(NAMES, rotation=90); ax.set_yticks(range(12)); ax.set_yticklabels(NAMES if ax is axes[0] else []); ax.tick_params(length=1.5, pad=1)
    for sp in ax.spines.values(): sp.set_edgecolor("0.62")
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8, length=1.5, pad=1); cb.outline.set_visible(False); cax.set_title("ARI", fontsize=8, pad=3)
save(fig, "fig_treemap_matrices_final")

# ================= appendix figures of the reduction (author's brief, 2026-09-24): five tables that show a trend become figures
# in the style of Figure 2. Every value is read from the same result file the table used.
# ---------------- the quadruple budget: the excess against the budget, one line per cell
B72 = pd.read_csv(RES/"expR72_budget_record.csv")
fig, ax = plt.subplots(figsize=(3.3, 1.9))
DSL72 = {"imagenet": "IN", "cifar100": "C100", "dtd": "DTD"}; MK72 = {"imagenet": "o", "cifar100": "s", "dtd": "^"}; LS72 = {"imagenet": "-", "cifar100": "--", "dtd": ":"}
for m in sorted(B72.model.unique()):
    for ds in ("imagenet", "cifar100", "dtd"):
        d_ = B72[(B72.model == m) & (B72.dataset == ds)].sort_values("n_quads")
        if not len(d_): continue
        ax.plot(d_.n_quads, d_.excess, LS72[ds], marker=MK72[ds], color=fam_color(m), lw=1.0, ms=3.4, mec="white", mew=0.5, zorder=3)
ax.set_xscale("log"); ax.set_xlabel("sampled quadruples per seed", labelpad=1); ax.set_ylabel("excess")
ax.set_title("the reading does not move with the budget", pad=3)
hd = [plt.Line2D([], [], marker=MK72[d], color="k", ls=LS72[d], lw=1.0, ms=3.4, mec="white", mew=0.5, label=DSL72[d]) for d in ("imagenet", "cifar100", "dtd")]
hd += [plt.Line2D([], [], color=fam_color(m), lw=2.0, label=NM[m]) for m in ("i21k_l", "dinov2_l", "clip_b")]
ax.legend(handles=hd, **LEG, loc="center right", ncol=2, handlelength=1.4, handletextpad=0.4, columnspacing=0.9, fontsize=6.2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.17, right=0.99, top=0.90, bottom=0.24)
_drift = float((B72[B72.n_quads >= 100000].groupby(["model", "dataset"]).excess.max() - B72[B72.n_quads >= 100000].groupby(["model", "dataset"]).excess.min()).max())
json.dump({"cells": int(B72.groupby(["model", "dataset"]).ngroups), "budgets": sorted(int(x) for x in B72.n_quads.unique()), "max_drift_above_1e5": round(_drift, 4)}, open(RES/"final_fig_budget.json", "w"), indent=1)
save(fig, "fig_budget_final")
print(f"budget: {B72.groupby(['model','dataset']).ngroups} cells, drift above 1e5 at most {_drift:.4f}")

# ---------------- the power per backbone against the noise level of the cloud
P81 = pd.read_csv(RES/"expR81_deep_per_backbone_summary.csv").set_index("model")
R64 = pd.read_csv(RES/"expR64b_wn30_summary.csv").set_index("model").ratio_real
fig, ax = plt.subplots(figsize=(3.3, 1.9))
for m in M:
    ax.plot(R64[m], float(P81.loc[m, "power"]), "o", mfc="white", mec=fam_color(m), ms=4.2, mew=1.1, zorder=3)
    ax.plot(R64[m], float(P81.loc[m, "dec_power"]), "o", color=fam_color(m), ms=4.6, mec="white", mew=0.6, zorder=4)
    ax.plot([R64[m], R64[m]], [float(P81.loc[m, "power"]), float(P81.loc[m, "dec_power"])], "-", color=fam_color(m), lw=0.7, alpha=0.6, zorder=2)
    whisker(ax, R64[m], *wilson(float(P81.loc[m, "dec_power"]), int(P81.loc[m, "dec_runs"])))   # 95 per cent interval (author's brief, 2026-09-25)
ax.axhline(0.8, color="k", lw=0.8, ls="--", zorder=1)
ax.set_ylim(-0.05, 1.08); ax.set_yticks([0, 0.5, 0.8, 1]); ax.set_yticklabels(["0", "0.5", "0.8", "1"])
ax.set_xlabel("noise level of the cloud (within/between spread)", labelpad=1); ax.set_ylabel("power")
ax.set_title("the power follows the backbone, not its noise level", pad=3)
hd = [plt.Line2D([], [], marker="o", color="k", ls="", ms=4.6, mec="white", mew=0.6, label="clusters rotated"),
      plt.Line2D([], [], marker="o", mfc="white", mec="k", ls="", ms=4.2, mew=1.1, label="intact"),
      plt.Line2D([], [], color="k", lw=0.8, ls="--", label="power 0.8")]
ax.legend(handles=hd, **LEG, loc="lower right", ncol=1, handlelength=1.4, handletextpad=0.4, fontsize=6.2)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.17, right=0.99, top=0.90, bottom=0.26)
_cov81 = [m for m in M if float(P81.loc[m, "dec_power"]) >= 0.8]
json.dump({"covered": _cov81, "n_covered": len(_cov81), "ratio_covered": [round(float(R64[m]), 2) for m in _cov81],
           "ratio_blind": [round(float(R64[m]), 2) for m in M if m not in _cov81]}, open(RES/"final_fig_power.json", "w"), indent=1)
save(fig, "fig_power_final")
print(f"power: {len(_cov81)}/12 backbones at decoupled power 0.8 or more")

# ---------------- the zero-cost gains per backbone
E2 = {(r["model"], r["dataset"]): r for r in csv.DictReader(open(RES/"exp2_metric_controls.csv"))}
M10 = [m for m in M if m not in ("dinov1_b", "siglip_b")]; HIER = ["imagenet", "cifar100", "cifar10", "dtd"]
fig, ax = plt.subplots(figsize=(5.5, 1.9))
_best = {}
for m in M10:
    fsH = [100 * (float(E2[(m, ds)]["FS_H"]) - float(E2[(m, ds)]["FS_R"])) for ds in HIER]
    fsC = [100 * (float(E2[(m, ds)]["FS_COS"]) - float(E2[(m, ds)]["FS_R"])) for ds in HIER]
    best = sum(max(h, c) for h, c in zip(fsH, fsC)) / len(HIER); gap = sum(fsH) / 4 - sum(fsC) / 4
    met = "H" if gap > 0.15 else ("cos" if gap < -0.15 else "either"); _best[m] = (round(best, 2), met)
    ax.bar(XPOS[m], best, 0.74, color=fam_color(m), edgecolor="white", linewidth=0.8, zorder=3)
    ax.annotate(met, (XPOS[m], best), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=5.8, color="0.25", zorder=5)
ax.axhline(0, color="0.35", lw=0.6, zorder=1)
ax.set_ylabel("advantage (pp)"); ax.set_title("the best zero-cost metric, against the Euclidean one", pad=3); famaxis(ax)
ax.set_ylim(0, max(v[0] for v in _best.values()) * 1.25)
json.dump({"best_pp": {m: _best[m][0] for m in M10}, "metric": {m: _best[m][1] for m in M10}, "datasets": HIER}, open(RES/"final_fig_gains.json", "w"), indent=1)
save(fig, "fig_gains_final")
print("gains:", {NM[m]: _best[m] for m in M10})
