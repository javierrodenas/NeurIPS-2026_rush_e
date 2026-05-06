#!/usr/bin/env python3
"""
Generate the "money figure": ORC vs Hyperbolic Retrieval Advantage scatter.
Saves to figures/fig_hyp_advantage.pdf and figures/fig_hyp_advantage.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_csv("/media/HDD_4TB_2/javi/Platonic/results/hyperbolic_advantage.csv")

# Convert imp_ret_H_p10 to percentage points
df["imp_pp"] = df["imp_ret_H_p10"] * 100  # already H - R

# ── Paradigm mapping ──────────────────────────────────────────────────
paradigm_colors = {
    "self-supervised": "#2077B4",   # blue
    "supervised":      "#2CA02C",   # green
    "contrastive":     "#D62728",   # red
    "contrastive-hyp": "#D62728",   # red (MERU is contrastive, hyperbolic variant)
}

paradigm_labels = {
    "self-supervised": "Self-supervised",
    "supervised":      "Supervised",
    "contrastive":     "Contrastive",
    "contrastive-hyp": "Contrastive (hyp.)",
}

# ── Compute correlation ───────────────────────────────────────────────
mask = df["orc_mean"].notna() & df["imp_pp"].notna()
r, p = stats.pearsonr(df.loc[mask, "orc_mean"], df.loc[mask, "imp_pp"])
print(f"Pearson r = {r:.3f}, p = {p:.6f}")

# ── Figure setup ──────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family":       "serif",
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   7.5,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":      42,   # TrueType for NeurIPS
    "ps.fonttype":       42,
})

fig, ax = plt.subplots(figsize=(5, 3.5))

# ── Region shading ────────────────────────────────────────────────────
ORC_THRESH = 0.45
xlim = (0.27, 0.70)
ylim = (-7, 20)

ax.axhspan(ylim[0], ylim[1], xmin=0, xmax=(ORC_THRESH - xlim[0]) / (xlim[1] - xlim[0]),
           color="#FFCCCC", alpha=0.18, zorder=0)
ax.axhspan(ylim[0], ylim[1], xmin=(ORC_THRESH - xlim[0]) / (xlim[1] - xlim[0]), xmax=1,
           color="#CCFFCC", alpha=0.18, zorder=0)

# ── Reference lines ──────────────────────────────────────────────────
ax.axhline(0, color="0.4", ls="--", lw=0.8, zorder=1)
ax.axvline(ORC_THRESH, color="0.4", ls="--", lw=0.8, zorder=1)

# ── Scatter points ────────────────────────────────────────────────────
# Plot each paradigm separately for legend
plotted_paradigms = {}
for _, row in df.iterrows():
    p_key = row["paradigm"]
    c = paradigm_colors.get(p_key, "gray")
    lbl = paradigm_labels.get(p_key, p_key)
    if lbl not in plotted_paradigms:
        plotted_paradigms[lbl] = True
        ax.scatter(row["orc_mean"], row["imp_pp"], c=c, s=42, edgecolors="white",
                   linewidths=0.5, zorder=5, label=lbl)
    else:
        ax.scatter(row["orc_mean"], row["imp_pp"], c=c, s=42, edgecolors="white",
                   linewidths=0.5, zorder=5)

# ── Regression line ───────────────────────────────────────────────────
x_fit = np.linspace(xlim[0], xlim[1], 200)
slope, intercept = np.polyfit(df.loc[mask, "orc_mean"], df.loc[mask, "imp_pp"], 1)
ax.plot(x_fit, slope * x_fit + intercept, color="0.35", lw=1.0, ls="-", alpha=0.6, zorder=2)

# ── Label key models ──────────────────────────────────────────────────
label_specs = {
    "dinov2_l":  {"text": "DINOv2-L\n(+17.6 pp)", "offset": (-55, 8)},
    "dinov2_g":  {"text": "DINOv2-G\n(+13.1 pp)", "offset": (8, -18)},
    "dinov2_b":  {"text": "DINOv2-B\n(+11.2 pp)", "offset": (-62, 5)},
    "clip_b":    {"text": "CLIP-B",               "offset": (-45, -14)},
    "i21k_t":    {"text": "ViT-T",                "offset": (8, 3)},
}

for _, row in df.iterrows():
    if row["model"] in label_specs:
        spec = label_specs[row["model"]]
        ax.annotate(
            spec["text"],
            xy=(row["orc_mean"], row["imp_pp"]),
            xytext=spec["offset"],
            textcoords="offset points",
            fontsize=7,
            color="0.15",
            arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
            zorder=10,
        )

# ── Region labels ─────────────────────────────────────────────────────
ax.text(0.36, ylim[1] - 1.5, "Euclidean\nsufficient", fontsize=7.5,
        color="#AA3333", alpha=0.55, ha="center", va="top", style="italic")
ax.text(0.58, ylim[1] - 1.5, "Hyperbolic\nadvantage", fontsize=7.5,
        color="#227722", alpha=0.55, ha="center", va="top", style="italic")

# ── Annotation box ────────────────────────────────────────────────────
p_str = f"$p$ < 0.001" if p < 0.001 else f"$p$ = {p:.3f}"
stat_text = f"$r$ = {r:.3f}\n{p_str}"
ax.text(0.97, 0.03, stat_text, transform=ax.transAxes, fontsize=8,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="0.7", alpha=0.9))

# ── Axes ──────────────────────────────────────────────────────────────
ax.set_xlim(xlim)
ax.set_ylim(ylim)
ax.set_xlabel("Mean Ollivier–Ricci curvature (ORC)")
ax.set_ylabel("Retrieval P@10 improvement (pp)\n(Hyperbolic $-$ Euclidean)")
ax.set_title("ORC predicts when hyperbolic distances help", fontweight="bold", pad=8)

# Legend
ax.legend(loc="lower right", frameon=True, framealpha=0.9, edgecolor="0.7",
          borderpad=0.4, handletextpad=0.4, bbox_to_anchor=(0.62, 0.0))

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────
out_dir = "/media/HDD_4TB_2/javi/NeurIPS_2026/figures"
fig.savefig(f"{out_dir}/fig_hyp_advantage.pdf")
fig.savefig(f"{out_dir}/fig_hyp_advantage.png")
print(f"Saved to {out_dir}/fig_hyp_advantage.{{pdf,png}}")
plt.close()
