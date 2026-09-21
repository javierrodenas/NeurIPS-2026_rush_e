#!/usr/bin/env python3
"""Final-version figures (author's brief 'Final version — plain, short, nine pages', 2026-09-18), bar language for every data figure:
light dashed grid behind the axes, family palette, 8 pt labels, model names as tick labels, filled = passes the test, hatched = does
not, the threshold or the null as a dashed line or a gray band, numbers only on bars that pass, short legends inside the panel.
Writes fig_overview_final, fig_excess_final, fig_depth_final and fig_treemap_final (pdf+png) next to the other figures; the v1/v2/v3
figure files are untouched. Data: expR52_census_haar_p999_200.csv, expR62_samplelevel_record.csv, exp1_delta_controls.csv,
expR56_depth_variants.csv, expR64b_wn30.csv, exp23_treemap_controls.npz, phaseC_fig2b.json."""
import csv, os, sys, json, importlib
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
from palette import FAMILY_COLORS, color as fam_color
GRAY = FAMILY_COLORS["null"]; LIGHT = "#C9C9C9"
M = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
DIMS = {"i21k_t":192,"i21k_s":384,"i21k_b":768,"i21k_l":1024,"dinov1_b":768,"dinov2_s":384,"dinov2_b":768,"dinov2_l":1024,"dinov2_g":1536,"clip_b":512,"clip_l":768,"siglip_b":768}
DSO = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]; DSL = {"imagenet":"ImageNet","cifar100":"CIFAR-100","cifar10":"CIFAR-10","dtd":"DTD","fashionmnist":"FMNIST","mnist":"MNIST"}
census = list(csv.DictReader(open(RES/"expR75_census_centered_haar.csv")))   # the record since 2026-09-20: centered Haar null (Q orthogonal to the all-ones vector)
by = {(r["model"], r["dataset"]): r for r in census}
def save(fig, name):
    for o in (HERE, HERE.parent/"iclr2027"/"figures"): fig.savefig(o/f"{name}.pdf"); fig.savefig(o/f"{name}.png", dpi=200)
    print(name, "written")
def bar(ax, x, h, color, passes, width=0.38, zorder=3):
    if passes: return ax.bar(x, h, width, color=color, edgecolor=color, linewidth=0.6, zorder=zorder)
    return ax.bar(x, h, width, facecolor="white", edgecolor=color, hatch="////", linewidth=0.6, zorder=zorder)

# ---------------- Figure 2: raw reading next to its matched null, one pair of bars per backbone ordered by dimension
order = sorted(M, key=lambda m: (DIMS[m], M.index(m)))
gauss = {int(r["d"]): float(r["delta_max"]) for r in csv.DictReader(open(RES/"exp1_delta_controls.csv")) if r["variant"] == "gauss"}
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.75), gridspec_kw={"width_ratios": [3.2, 1]})
ax = axes[0]; X = np.arange(len(order))
for i, m in enumerate(order):
    r = by[(m, "imagenet")]
    ax.bar(i - 0.2, float(r["delta"]), 0.38, color=fam_color(m), zorder=3)
    ax.bar(i + 0.2, float(r["null_mean"]), 0.38, color=LIGHT, zorder=3)
# the isotropic Gaussian reference steps with the dimension groups
xs, ys = [], []
for i, m in enumerate(order):
    xs += [i - 0.5, i + 0.5]; ys += [gauss[DIMS[m]], gauss[DIMS[m]]]
ax.plot(xs, ys, "--", color=GRAY, lw=0.9, zorder=4)
ax.set_xticks(X); ax.set_xticklabels([NM[m] for m in order], rotation=60, ha="right"); ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_ylim(0, 0.12); ax.set_yticks([0, 0.04, 0.08, 0.12]); ax.set_yticklabels(["0.00", "0.04", "0.08", "0.12"]); ax.set_ylabel(r"$\delta_{\mathrm{norm}}$ (ImageNet)")
ax.set_title("(a) raw reading and matched null, backbones by dimension")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
hd = [Patch(color="k", label="raw reading"), Patch(color=LIGHT, label="matched null"), plt.Line2D([], [], ls="--", color=GRAY, label="isotropic Gaussian")]
ax.set_ylim(0, 0.15); ax.set_yticks([0, 0.04, 0.08, 0.12]); ax.set_yticklabels(["0.00", "0.04", "0.08", "0.12"])
ax.legend(handles=hd, frameon=False, loc="upper right", ncol=2, handlelength=1.2, handletextpad=0.4, columnspacing=1.0, labelspacing=0.2, borderaxespad=0.1)
ax = axes[1]
D = json.load(open(RES/"final_fig2b.json")); assert D.get("mode") == "sample", D   # fifth review: same dataset, same reading, opposite verdict
sl = {(r["model"], r["dataset"]): r for r in csv.DictReader(open(RES/"expR62_samplelevel_record.csv"))}
s_m, s_ds = D["sample_cell"]; c_m, c_ds = D["class_cell"]; cc = by[(c_m, c_ds)]; E = D["expected"]; ss = sl[(s_m, s_ds)]
assert s_ds == c_ds and abs(float(ss["delta_999"]) - E["sample_delta"]) < 5e-4 and abs(float(ss["excess"]) - E["sample_excess"]) < 5e-4 and (ss["genuine_bh"] == "True") == E["sample_genuine"]
assert abs(float(cc["delta"]) - E["class_delta"]) < 5e-4 and abs(float(cc["excess"]) - E["class_excess"]) < 5e-4 and (cc["genuine_bh"] == "True") == E["class_genuine"]
cells = [(NM[s_m] + "\nimages", fam_color(s_m), float(sl[(s_m, s_ds)]["delta_999"]), float(sl[(s_m, s_ds)]["null_mean"])),
         (NM[c_m] + "\ncentroids", fam_color(c_m), float(cc["delta"]), float(cc["null_mean"]))]
for i, (lab, col, raw, null) in enumerate(cells):
    ax.bar(i - 0.2, raw, 0.38, color=col, zorder=3); ax.bar(i + 0.2, null, 0.38, color=LIGHT, zorder=3)
ax.set_xticks([0, 1]); ax.set_xticklabels([c[0] for c in cells], linespacing=1.1); ax.set_xlim(-0.7, 1.7)
ax.set_ylim(0, 0.16); ax.set_yticks([0, 0.05, 0.10, 0.15]); ax.set_yticklabels(["0.00", "0.05", "0.10", "0.15"])
ax.set_title("(b) " + DSL[s_ds] + ": same reading,\nopposite verdict", linespacing=1.1)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.subplots_adjust(left=0.09, right=0.995, top=0.88, bottom=0.34, wspace=0.30)
save(fig, "fig_overview_final")

# ---------------- Figure 3: six narrow panels of twelve horizontal bars (excess), filled when genuine, hatched when not, the null band around zero
def hbar(ax, y, w, color, passes, height=0.72, zorder=3):
    if passes: return ax.barh(y, w, height, color=color, edgecolor=color, linewidth=0.6, zorder=zorder)
    return ax.barh(y, w, height, facecolor="white", edgecolor=color, hatch="////", linewidth=0.6, zorder=zorder)
fig, axg = plt.subplots(1, 6, figsize=(5.5, 2.05), sharey=True)
Y = np.arange(len(M))[::-1]
for ax, ds in zip(axg, DSO):
    sd = np.median([float(by[(m, ds)]["null_sd"]) for m in M])
    ax.axvspan(-2*sd, 2*sd, color=GRAY, alpha=0.18, lw=0, zorder=0); ax.axvline(0, color="k", lw=0.6, zorder=1)
    for i, m in enumerate(M):
        r = by[(m, ds)]; hbar(ax, Y[i], float(r["excess"]), fam_color(m), str(r["genuine_bh"]) == "True")
    ax.set_yticks(Y); ax.set_yticklabels([NM[m] for m in M]); ax.set_ylim(-0.7, len(M) - 0.3)
    ax.set_xlim(-0.145, 0.03); ax.set_xticks([-0.12, -0.06, 0]); ax.set_xticklabels(["−0.12", "−0.06", "0"])
    ax.set_title(DSL[ds]); ax.tick_params(axis="y", length=0)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.text(0.5, 0.135, "excess over the matched null", ha="center", va="center", fontsize=8)
hd = [Patch(color="k", label="genuine"), Patch(facecolor="white", edgecolor="k", hatch="////", label="not genuine"), Patch(color=GRAY, alpha=0.3, label="±2 null s.d.")]
fig.legend(handles=hd, frameon=False, loc="lower center", ncol=3, handlelength=1.2, handletextpad=0.4, columnspacing=1.5, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.10, right=0.995, top=0.91, bottom=0.25, wspace=0.12)
save(fig, "fig_excess_final")

# ---------------- Figure 4: (a) twelve bars of depth z, filled when certified, dashed line at -2, z printed inside the certified bars; (b) detection curves; one legend below both panels
dv = pd.read_csv(RES/"expR56_depth_variants.csv"); an = {r.model: float(r.z_depth) for r in dv[(dv.dataset == "imagenet") & (dv.K == 30) & (dv.variant == "aniso")].itertuples()}
d64 = pd.read_csv(RES/"expR64b_wn30.csv"); dep = d64[(d64.kind == "depth") & (d64.partition == "rand6") & (d64.s != "real")].copy(); dep["s"] = dep.s.astype(float)
tg = d64[(d64.kind == "depth") & (d64.partition == "rand6_t06")].copy(); tg["s"] = tg.s.astype(float)
pr = dep.groupby("s").z.apply(lambda z: (z <= -2).mean()); pt = tg.groupby("s").z.apply(lambda z: (z <= -2).mean())
fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.9), gridspec_kw={"width_ratios": [1.6, 1]})
ax = axes[0]
ax.axhline(-2, color="k", lw=0.8, ls="--", zorder=2); ax.axhline(0, color=GRAY, lw=0.5, zorder=1)
for i, m in enumerate(M):
    z = an[m]; cert = z <= -2; bar(ax, i, z, fam_color(m), cert, width=0.76)
    if cert: ax.text(i, z / 2, f"{z:+.1f}".replace("-", "−").replace("+", ""), ha="center", va="center", rotation=90, fontsize=8, color="white", zorder=5)
ax.set_xticks(range(len(M))); ax.set_xticklabels([NM[m] for m in M], rotation=60, ha="right"); ax.set_xlim(-0.7, len(M) - 0.3)
ax.set_ylim(-4.6, 1.2); ax.set_yticks([-4, -2, 0]); ax.set_yticklabels(["−4", "−2", "0"]); ax.set_ylabel("depth test $z$")
ax.set_title("(a) depth above the WordNet superclasses, ImageNet"); ax.tick_params(axis="x", length=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax = axes[1]
l1, = ax.plot(pr.index, pr.values, "-o", color="k", ms=3, lw=1.0, label="real spread", zorder=3)
l2, = ax.plot(pt.index, pt.values, "--s", color=GRAY, ms=3, lw=1.0, label="shrunk spread", zorder=3)
l3 = None   # priority 1c (expR80): the implanted-alignment curve enters the submission only when the decision rule of the brief is met
if (RES/"expR80_decision.csv").exists() and (RES/"expR80_implanted_alignment_summary.csv").exists():
    d80 = list(csv.DictReader(open(RES/"expR80_decision.csv")))[0]
    if d80["rule_power_ge_0_8_fa_le_0_05"] == "True":
        s80 = list(csv.DictReader(open(RES/"expR80_implanted_alignment_summary.csv")))
        l3, = ax.plot([float(r["s"]) for r in s80], [float(r["detection_rate"]) for r in s80], ":^", color=FAMILY_COLORS["supervised"], ms=3, lw=1.0, label="implanted alignment", zorder=3)
ax.annotate("no false alarms at $s{=}0$ (real spread)", (0, pr.loc[0.0]), xytext=(12, 5), textcoords="offset points", fontsize=8, ha="left", va="bottom", arrowprops=dict(arrowstyle="-", color="k", lw=0.6))   # seventh review: the label sits on the real-spread curve
ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.set_xlabel("implant strength $s$", labelpad=1)
ax.set_ylim(-0.04, 1.08); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(["0.0", "0.5", "1.0"]); ax.set_ylabel("detection rate")
ax.set_title("(b) false alarms and power on real clouds")
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
hd = [Patch(color="k", label="certified ($z\\leq-2$)"), Patch(facecolor="white", edgecolor="k", hatch="////", label="not detected"), l1, l2] + ([l3] if l3 is not None else [])
json.dump({"legend": [h.get_label() for h in hd], "implanted_alignment_curve": l3 is not None}, open(RES/"final_fig4.json", "w"), indent=1)   # read by the sweep (priority 1c)
fig.legend(handles=hd, frameon=False, loc="lower center", ncol=len(hd), handlelength=1.4, handletextpad=0.4, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.09, right=0.99, top=0.89, bottom=0.40, wspace=0.35)
save(fig, "fig_depth_final")
print(f"depth: certified {sum(v <= -2 for v in an.values())}/12; detection real {dict((float(k), round(float(v), 3)) for k, v in pr.items())}; shrunk {dict((float(k), round(float(v), 3)) for k, v in pt.items())}")

# ---------------- Figure 5: (a)(b) two ARI matrices, sequential palette, rectangle around the DINOv2 block; (c) sibling triplets under cosine and Euclidean distance per backbone
z = np.load(RES/"exp23_treemap_controls.npz", allow_pickle=True); S = z["summary_in"].item()
diag = json.load(open(RES/"exp23_config_diagnostics.json")); adm = {k: v for k, v in diag.items() if k.startswith("imagenet|") and v["maxfrac"] <= 0.5}
sel = max(adm, key=lambda k: adm[k]["cpcc"]).split("|")[1:]; SELKEY = "('{}', '{}')".format(*sel); assert SELKEY == "('cosine', 'average')", SELKEY
cmap = LinearSegmentedColormap.from_list("family_seq", ["#FFFFFF", "#9FBFD6", FAMILY_COLORS["supervised"], "#123B57"])
NAMES = [NM[m] for m in M]; sup = [0, 1, 2, 3, 9, 10, 11]; big = [6, 7, 8]
e10 = {r["model"]: r for r in csv.DictReader(open(RES/"exp10_local_vs_global.csv"))}
W, H = 5.5, 2.5; s = 1.4; bot = 0.72                       # matrices of side s; panel (c) shares their bottom and height
fig = plt.figure(figsize=(W, H))
axes = [fig.add_axes([0.42/W, bot/H, s/W, s/H]), fig.add_axes([(0.42+s+0.10)/W, bot/H, s/W, s/H])]
cax = fig.add_axes([(0.42+2*s+0.16)/W, bot/H, 0.07/W, s/H]); axc = fig.add_axes([(0.42+2*s+0.78)/W, bot/H, (W-(0.42+2*s+0.78)-0.04)/W, s/H])
vals = []
for ax, (key, title) in zip(axes, [("('euclid', 'average')", "(a) naive: Euclidean, average"), (SELKEY, "(b) selected: cosine, average")]):
    Mx = np.asarray(S[key]["ari"], dtype=float); im = ax.imshow(Mx, vmin=0, vmax=1, cmap=cmap); ax.grid(False)
    v = float(np.mean([Mx[i, j] for i in big for j in sup])); vals.append(v)
    ax.add_patch(Rectangle((4.5, 4.5), 4, 4, fill=False, edgecolor=FAMILY_COLORS["ssl"], lw=1.0, zorder=4))
    ax.set_title(title, pad=3); ax.set_xticks(range(12)); ax.set_xticklabels(NAMES, rotation=90); ax.set_yticks(range(12)); ax.set_yticklabels(NAMES if ax is axes[0] else [])
    ax.tick_params(length=1.5, pad=1)
    for sp in ax.spines.values(): sp.set_edgecolor("0.62")
cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8, length=1.5, pad=1); cb.outline.set_visible(False); cax.set_title("ARI", fontsize=8, pad=3)
ax = axc; tri = {}
for i, m in enumerate(M):
    c, e = float(e10[m]["c100_sibtrip_c"]), float(e10[m]["c100_sibtrip_e"]); tri[m] = (c, e)
    ax.bar(i - 0.2, c, 0.4, color=fam_color(m), zorder=3); ax.bar(i + 0.2, e, 0.4, facecolor="white", edgecolor=fam_color(m), hatch="////", linewidth=0.6, zorder=3)
ax.axhline(0.5, color="k", lw=0.8, ls="--", zorder=2)
ax.set_xticks(range(len(M))); ax.set_xticklabels(NAMES, rotation=90); ax.set_xlim(-0.7, len(M) - 0.3); ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.5, 1.0]); ax.set_yticklabels(["0.0", "0.5", "1.0"]); ax.set_ylabel("triplet agreement", labelpad=2)
ax.set_title("(c) where the tree lives:\nsibling triplets, CIFAR-100", pad=3, linespacing=1.1)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
hd = [Patch(color="k", label="cosine"), Patch(facecolor="white", edgecolor="k", hatch="////", label="Euclidean"), plt.Line2D([], [], ls="--", color="k", lw=0.8, label="chance")]
ax.legend(handles=hd, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.50), ncol=3, handlelength=1.2, handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
save(fig, "fig_treemap_final")
json.dump({"naive_dinov2_vs_block": vals[0], "selected_dinov2_vs_block": vals[1], "sibtrip_c100": {m: {"cosine": tri[m][0], "euclid": tri[m][1]} for m in M}}, open(RES/"final_fig5_values.json", "w"), indent=1)
d2 = {m: tri[m][0] - tri[m][1] for m in M}; print(f"treemap: DINOv2-B/L/G vs block naive {vals[0]:.2f}, selected {vals[1]:.2f}; triplet cosine-Euclid gap: DINOv2 {min(d2[m] for m in ('dinov2_b','dinov2_l','dinov2_g')):.2f}..{max(d2[m] for m in ('dinov2_b','dinov2_l','dinov2_g')):.2f}, others max {max(abs(d2[m]) for m in M if not m.startswith('dinov2')):.2f}")
