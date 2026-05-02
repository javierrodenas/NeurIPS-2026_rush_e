#!/usr/bin/env python3
"""Generate figures/fig3_causal.pdf — three-panel causal evidence figure."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Global style ──────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
})

BLUE = "#4472C4"
RED  = "#C0504D"

fig, axes = plt.subplots(1, 3, figsize=(10, 3), dpi=300)

# ══════════════════════════════════════════════════════════════════════
# Panel (a): Non-hierarchical fine-tuning
# ══════════════════════════════════════════════════════════════════════
ax = axes[0]
models_a = ["DINOv2-S", "ViT-B", "CLIP-B"]
before_a = [0.095, 0.097, 0.104]
after_a  = [0.161, 0.116, 0.199]
pcts_a   = ["+69%", "+19%", "+91%"]

x_a = np.arange(len(models_a))
w = 0.32
bars1 = ax.bar(x_a - w/2, before_a, w, color=BLUE, label="Before", zorder=3)
bars2 = ax.bar(x_a + w/2, after_a,  w, color=RED,  label="After",  zorder=3)

for i, (b2, pct) in enumerate(zip(bars2, pcts_a)):
    ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.004,
            pct, ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x_a)
ax.set_xticklabels(models_a)
ax.set_ylabel(r"$\delta_{\mathrm{rel}}$")
ax.set_title("(a) Non-hierarchical fine-tuning")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(0, 0.25)
ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# ══════════════════════════════════════════════════════════════════════
# Panel (b): Shuffle control
# ══════════════════════════════════════════════════════════════════════
ax = axes[1]
models_b = ["CLIP-L-336", "DINOv2-L", "CLIP-L"]
baseline_b = [0.093, 0.067, 0.087]
shuffled_b = [0.093, 0.070, 0.088]
# Small error bars to show variability
err_base = [0.003, 0.002, 0.003]
err_shuf = [0.004, 0.003, 0.003]

x_b = np.arange(len(models_b))
ax.bar(x_b - w/2, baseline_b, w, color=BLUE, label="Baseline",  yerr=err_base,
       capsize=3, error_kw={"linewidth": 0.8}, zorder=3)
ax.bar(x_b + w/2, shuffled_b, w, color=RED,  label="Shuffled",  yerr=err_shuf,
       capsize=3, error_kw={"linewidth": 0.8}, zorder=3)

ax.set_xticks(x_b)
ax.set_xticklabels(models_b)
ax.set_ylabel(r"$\delta_{\mathrm{rel}}$")
ax.set_title("(b) Shuffle control")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(0, 0.13)
ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# ══════════════════════════════════════════════════════════════════════
# Panel (c): Random vs pretrained
# ══════════════════════════════════════════════════════════════════════
ax = axes[2]
models_c = ["DINOv2-B", "DINOv2-S", "ViT-B"]
pretrained_c = [0.081, 0.095, 0.113]
random_c     = [0.198, 0.222, 0.187]
pcts_c       = ["+144%", "+134%", "+65%"]

x_c = np.arange(len(models_c))
bars1c = ax.bar(x_c - w/2, pretrained_c, w, color=BLUE, label="Pretrained", zorder=3)
bars2c = ax.bar(x_c + w/2, random_c,     w, color=RED,  label="Random",     zorder=3)

for i, (b2, pct) in enumerate(zip(bars2c, pcts_c)):
    ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.004,
            pct, ha="center", va="bottom", fontsize=7, fontweight="bold")

ax.set_xticks(x_c)
ax.set_xticklabels(models_c)
ax.set_ylabel(r"$\delta_{\mathrm{rel}}$")
ax.set_title("(c) Random vs pretrained")
ax.legend(loc="upper left", frameon=False)
ax.set_ylim(0, 0.28)
ax.grid(axis="y", linewidth=0.3, alpha=0.5, zorder=0)
ax.set_axisbelow(True)

# ── Save ──────────────────────────────────────────────────────────────
fig.tight_layout(pad=1.0)
out = "/media/HDD_4TB_2/javi/NeurIPS_2026/figures/fig3_causal"
fig.savefig(out + ".pdf", bbox_inches="tight")
fig.savefig(out + ".png", bbox_inches="tight")
print(f"Saved {out}.pdf and {out}.png")
