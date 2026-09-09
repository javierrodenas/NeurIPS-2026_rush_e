#!/usr/bin/env python3
"""Depth-test figure (Phase B, B3): (a) real vs matched-star excess B per backbone with z, CIFAR-100 (K=20) and
ImageNet (K=30), isotropic and anisotropic stars side by side (expR56_depth_variants.csv); (b) power of the test on
synthetic hierarchies: leaf-frame anisotropic star (expR55b) as solid lines, top-level-frame isotropic star (expR55)
dotted, one line per K, pooled over hierarchy levels/anisotropy; star false alarms in gray."""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
plt.style.use(str(HERE / "style.mplstyle"))
sys.path.insert(0, str(HERE))
from palette import FAMILY_COLORS, color as fam_color
M = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"Dv2-S","dinov2_b":"Dv2-B",
      "dinov2_l":"Dv2-L","dinov2_g":"Dv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP"}
dv = pd.read_csv(RES/"expR56_depth_variants.csv")
fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
for ax, (ds, K, title) in zip(axes[:2], [("cifar100", 20, "(a) CIFAR-100, $K{=}20$"), ("imagenet", 30, "(b) ImageNet, $K{=}30$")]):
    x = np.arange(len(M))
    for i, m in enumerate(M):
        iso = dv[(dv.model==m)&(dv.dataset==ds)&(dv.K==K)&(dv.variant=="iso")].iloc[0]
        an = dv[(dv.model==m)&(dv.dataset==ds)&(dv.K==K)&(dv.variant=="aniso")].iloc[0]
        c = fam_color(m)
        ax.plot([i, i], [iso.excessB_real, iso.excessB_star], "-", color=FAMILY_COLORS["null"], lw=0.6, zorder=1)
        ax.scatter(i, iso.excessB_real, s=16, color=c, zorder=3)
        ax.scatter(i, iso.excessB_star, s=16, facecolors="none", edgecolors=FAMILY_COLORS["null"], linewidths=0.8, zorder=2)
        ax.scatter(i, an.excessB_star, s=14, marker="s", color=FAMILY_COLORS["null"], zorder=2)
    ax.axhline(0, color="k", lw=0.6)
    lo, hi = ax.get_ylim(); band = lo - 0.18*(hi-lo); ax.set_ylim(band - 0.06*(hi-lo), hi)
    for i, m in enumerate(M):
        an = dv[(dv.model==m)&(dv.dataset==ds)&(dv.K==K)&(dv.variant=="aniso")].iloc[0]
        ax.text(i, band, f"{an.z_depth:+.1f}", ha="center", va="bottom", fontsize=4.8, color="k", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([NM[m] for m in M], rotation=90, fontsize=5.5)
    ax.set_title(title); ax.set_ylabel("excess B (hub null)")
hd = [plt.Line2D([], [], marker="o", ls="", color="k", ms=4, label="real"),
      plt.Line2D([], [], marker="o", ls="", mfc="none", mec=FAMILY_COLORS["null"], ms=4, label="isotropic star"),
      plt.Line2D([], [], marker="s", ls="", color=FAMILY_COLORS["null"], ms=4, label="anisotropic star ($z$ below)")]
fig.legend(handles=hd, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.40, 1.0), fontsize=5.5, handletextpad=0.3, columnspacing=1.2)
# (c) power
ax = axes[2]; cols = {6:"#4C72B0", 12:"#55A868", 20:"#DD8452", 30:"#8172B2"}
for f, ls, lab in [(RES/"expR55b_depth_power_leafframe.csv", "-", "leaf frame"), (RES/"expR55_depth_power.csv", ":", "top frame")]:
    if not f.exists(): continue
    d = pd.read_csv(f); h = d[(d.level!="star")&(d.n==1000)]; st = d[(d.level=="star")&(d.n==1000)]
    for K in (6,12,20,30):
        hh = h[h.K==K]
        if len(hh)==0: continue
        ax.plot([0.1,0.3,0.6], [(hh[hh.ratio==r].z<=-2).mean() for r in (0.1,0.3,0.6)], ls, marker="o" if ls=="-" else None, ms=3, color=cols[K], lw=1.1,
                label=f"K={K}" if ls=="-" else None)
    if ls == "-":
        ax.plot([0.1,0.3,0.6], [(st[st.ratio==r].z<=-2).mean() for r in (0.1,0.3,0.6)], "--", color=FAMILY_COLORS["null"], lw=1.0, label="false alarms, $n{=}1000$")
        st100 = d[(d.level=="star")&(d.n==100)]
        ax.plot([0.1,0.3,0.6], [(st100[st100.ratio==r].z<=-2).mean() for r in (0.1,0.3,0.6)], "--", color="k", lw=1.0, label="false alarms, $n{=}100$")
ax.set_ylim(-0.03, 1.03); ax.set_xticks([0.1,0.3,0.6]); ax.set_xlabel("within/between noise ratio"); ax.set_ylabel("power ($z\\leq-2$), $n{=}1000$")
ax.set_title("(c) synthetic hierarchies"); ax.legend(frameon=False, fontsize=5.3, loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=1.4, labelspacing=0.3)
fig.tight_layout(w_pad=0.6, rect=[0, 0, 1, 0.93])
for o in (HERE, HERE.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_depth_test.pdf"); fig.savefig(o/"fig_depth_test.png", dpi=200)
print("fig_depth_test written")

# ---- main-text figure: (a) ImageNet K=30 real vs stars, (b) power; appendix figure: CIFAR-100 K=20 alone ----
def panel_real(ax, ds, K, title):
    x = np.arange(len(M))
    for i, m in enumerate(M):
        iso = dv[(dv.model==m)&(dv.dataset==ds)&(dv.K==K)&(dv.variant=="iso")].iloc[0]
        an = dv[(dv.model==m)&(dv.dataset==ds)&(dv.K==K)&(dv.variant=="aniso")].iloc[0]
        ax.plot([i, i], [iso.excessB_real, iso.excessB_star], "-", color=FAMILY_COLORS["null"], lw=0.6, zorder=1)
        ax.scatter(i, iso.excessB_real, s=16, color=fam_color(m), zorder=3)
        ax.scatter(i, iso.excessB_star, s=16, facecolors="none", edgecolors=FAMILY_COLORS["null"], linewidths=0.8, zorder=2)
        ax.scatter(i, an.excessB_star, s=14, marker="s", color=FAMILY_COLORS["null"], zorder=2)
    ax.axhline(0, color="k", lw=0.6)
    lo, hi = ax.get_ylim(); band = lo - 0.18*(hi-lo); ax.set_ylim(band - 0.06*(hi-lo), hi)
    for i, m in enumerate(M):
        an = dv[(dv.model==m)&(dv.dataset==ds)&(dv.K==K)&(dv.variant=="aniso")].iloc[0]
        ax.text(i, band, f"{an.z_depth:+.1f}", ha="center", va="bottom", fontsize=4.8, color="k" if an.z_depth > -2 else "#B22222", fontweight="normal" if an.z_depth > -2 else "bold", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels([NM[m] for m in M], rotation=90, fontsize=5.5); ax.set_title(title); ax.set_ylabel("excess B (hub null)")

def panel_power(ax, title):
    for f, ls in [(RES/"expR55b_depth_power_leafframe.csv", "-"), (RES/"expR55_depth_power.csv", ":")]:
        if not f.exists(): continue
        d = pd.read_csv(f); h = d[(d.level!="star")&(d.n==1000)]; st = d[(d.level=="star")&(d.n==1000)]
        for K in (6,12,20,30):
            hh = h[h.K==K]
            if len(hh)==0: continue
            ax.plot([0.1,0.3,0.6], [(hh[hh.ratio==r].z<=-2).mean() for r in (0.1,0.3,0.6)], ls, marker="o" if ls=="-" else None, ms=3, color=cols[K], lw=1.1, label=f"K={K}" if ls=="-" else None)
        if ls == "-":
            ax.plot([0.1,0.3,0.6], [(st[st.ratio==r].z<=-2).mean() for r in (0.1,0.3,0.6)], "--", color=FAMILY_COLORS["null"], lw=1.0, label="false alarms, $n{=}1000$")
            st100 = d[(d.level=="star")&(d.n==100)]
            ax.plot([0.1,0.3,0.6], [(st100[st100.ratio==r].z<=-2).mean() for r in (0.1,0.3,0.6)], "--", color="k", lw=1.0, label="false alarms, $n{=}100$")
    ax.set_ylim(-0.03, 1.03); ax.set_xticks([0.1,0.3,0.6]); ax.set_xlabel("within/between noise ratio"); ax.set_ylabel("power ($z\\leq-2$), $n{=}1000$")
    ax.set_title(title); ax.legend(frameon=False, fontsize=5.3, loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=1.4, labelspacing=0.3)

def panel_implant(ax, title):
    """z of the depth test on the real ImageNet clouds with the hub arrangement replaced by an implanted two-level tree of strength s
    (expR64b, frame of record); dashed: the same clouds with their within-cluster offsets shrunk to within/between = 0.6 (tight variant)."""
    d = pd.read_csv(RES/"expR64b_wn30.csv"); dep = d[(d.kind=="depth")&(d.partition=="rand6")&(d.s!="real")].copy(); dep["s"] = dep.s.astype(float)
    tg = d[(d.kind=="depth")&(d.partition=="rand6_t06")].copy(); tg["s"] = tg.s.astype(float)
    for m in M:
        g = dep[dep.model==m].groupby("s").z; c = fam_color(m)
        ax.plot(g.mean().index, g.mean().values, "-o", color=c, ms=2.2, lw=0.8, zorder=3)
        ax.fill_between(g.min().index, g.min().values, g.max().values, color=c, alpha=0.10, lw=0)
        if len(tg):
            t = tg[tg.model==m].groupby("s").z.mean(); ax.plot(t.index, t.values, "--", color=c, lw=0.6, alpha=0.7, zorder=2)
    ax.axhline(-2, color="k", lw=0.7, ls="--"); ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("implant strength $s$"); ax.set_ylabel("depth test $z$"); ax.set_title(title); ax.set_xticks([0, 0.25, 0.5, 0.75, 1])

if (RES/"expR64b_wn30.csv").exists():
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.38), gridspec_kw={"width_ratios": [1.25, 1]})
    panel_real(axes[0], "imagenet", 30, "(a) ImageNet, WordNet $K{=}30$ ($n{=}1000$)"); panel_implant(axes[1], "(b) implanted depth on the real clouds")
    fig.legend(handles=hd, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.30, 1.0), fontsize=5.5, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(w_pad=0.8, rect=[0, 0, 1, 0.92])
    for o in (HERE, HERE.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_depth_main.pdf"); fig.savefig(o/"fig_depth_main.png", dpi=200)
    fig, ax = plt.subplots(1, 1, figsize=(2.8, 1.9)); panel_power(ax, "synthetic hierarchies, leaf frame"); fig.tight_layout()
    for o in (HERE, HERE.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_depth_power_app.pdf"); fig.savefig(o/"fig_depth_power_app.png", dpi=200)
    print("fig_depth_main (real + implant) and fig_depth_power_app written")
else:
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.5), gridspec_kw={"width_ratios": [1.25, 1]})
    panel_real(axes[0], "imagenet", 30, "(a) ImageNet, WordNet $K{=}30$ ($n{=}1000$)"); panel_power(axes[1], "(b) synthetic hierarchies, leaf frame")
    fig.legend(handles=hd, frameon=False, loc="upper center", ncol=3, bbox_to_anchor=(0.30, 1.0), fontsize=5.5, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout(w_pad=0.8, rect=[0, 0, 1, 0.92])
    for o in (HERE, HERE.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_depth_main.pdf"); fig.savefig(o/"fig_depth_main.png", dpi=200)
fig, ax = plt.subplots(1, 1, figsize=(3.0, 1.9))
panel_real(ax, "cifar100", 20, "CIFAR-100, $K{=}20$ ($n{=}100$; unvalidated regime)")
ax.legend(handles=hd, frameon=False, loc="lower right", fontsize=5.2, handletextpad=0.3, labelspacing=0.25)
fig.tight_layout()
for o in (HERE, HERE.parent/"iclr2027"/"figures"): fig.savefig(o/"fig_depth_cifar100.pdf"); fig.savefig(o/"fig_depth_cifar100.png", dpi=200)
print("fig_depth_main + fig_depth_cifar100 written")
