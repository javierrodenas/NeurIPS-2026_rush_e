#!/usr/bin/env python3
"""Schematic of the parameter-free Poincare projection (committed diagram)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent
rng = np.random.RandomState(3)

# synthetic 2-D feature cloud with 3 loose clusters
centers = np.array([[1.8, 0.6], [-0.9, 1.6], [-0.4, -1.7]])
X = np.concatenate([c + 0.55*rng.randn(26, 2) for c in centers])
mu = X.mean(0)
Z = X - mu
norms = np.linalg.norm(Z, axis=1)
p95 = np.percentile(norms, 95)
s = 2*np.arctanh(1/np.sqrt(2)) / p95 / 2  # s so that tanh(0.5*s*p95)=1/sqrt2
r_new = np.tanh(0.5*s*norms)
P = (Z/norms[:, None]) * r_new[:, None]

cols = np.repeat(["#4C72B0", "#DD8452", "#55A868"], 26)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
# left: feature space
ax1.scatter(Z[:, 0], Z[:, 1], s=16, c=cols, alpha=0.85, edgecolors="none")
ax1.scatter(0, 0, marker="+", s=90, c="k", lw=1.6, zorder=5)
ax1.annotate(r"$\mu$", (0, 0), textcoords="offset points", xytext=(7, 5), fontsize=11)
th = np.linspace(0, 2*np.pi, 200)
ax1.plot(p95*np.cos(th), p95*np.sin(th), "--", color="gray", lw=1.1)
ax1.annotate("95th-percentile norm", (p95*0.72, p95*0.72), fontsize=8, color="gray")
i = int(np.argmax(norms))
ax1.annotate(r"$x-\mu$", (Z[i, 0], Z[i, 1]), textcoords="offset points",
             xytext=(6, -10), fontsize=10)
ax1.plot([0, Z[i, 0]], [0, Z[i, 1]], color="k", lw=0.9, alpha=0.6)
ax1.set_title("centered feature space", fontsize=10)
ax1.set_aspect("equal"); ax1.axis("off")

# right: Poincare disk
ax2.plot(np.cos(th), np.sin(th), "k-", lw=1.4)
r95 = 1/np.sqrt(2)
ax2.plot(r95*np.cos(th), r95*np.sin(th), "--", color="gray", lw=1.1)
ax2.annotate(r"$\|\cdot\|_{p95} \mapsto 1/\sqrt{2}$", (r95*0.30, r95*1.04),
             fontsize=9, color="gray")
ax2.scatter(P[:, 0], P[:, 1], s=16, c=cols, alpha=0.85, edgecolors="none")
ax2.scatter(0, 0, marker="+", s=90, c="k", lw=1.6, zorder=5)
ax2.plot([0, P[i, 0]], [0, P[i, 1]], color="k", lw=0.9, alpha=0.6)
ax2.annotate(r"$\phi(x)$", (P[i, 0], P[i, 1]), textcoords="offset points",
             xytext=(6, -10), fontsize=10)
ax2.set_title("Poincaré ball (unit disk)", fontsize=10)
ax2.set_aspect("equal"); ax2.axis("off")

fig.suptitle(r"$\phi(x)=\tanh\!\left(\frac{1}{2}\|s(x-\mu)\|\right)\cdot"
             r"\frac{s(x-\mu)}{\|s(x-\mu)\|}$"
             "\n directions preserved; norms compressed through tanh", fontsize=10, y=1.04)
fig.tight_layout()
fig.savefig(OUT/"fig_expmap.pdf", bbox_inches="tight")
fig.savefig(OUT/"fig_expmap.png", dpi=170, bbox_inches="tight")
print("expmap figure written")
