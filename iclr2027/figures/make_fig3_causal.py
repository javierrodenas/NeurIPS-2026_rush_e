#!/usr/bin/env python3
"""Fig. 3 (ablations tying the form to learning), regenerated at ICLR width from the
stored ablation results: (a) non-hierarchical fine-tuning (analysis4_finetuning.csv), (b) delta across depth
(e1_delta_by_layer.csv). The hierarchical fine-tuning arm is intentionally not drawn
(it collapses effective rank and is not used as evidence); the label-shuffle control
is reported in the text."""
import csv, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
plt.style.use(str(HERE / "style.mplstyle"))
import sys; sys.path.insert(0, str(HERE))
from palette import color as fam_color
RES = Path(os.environ.get("PLATONIC_ABLATIONS", "/media/HDD_4TB_2/javi/Platonic/results"))
NAME = {"dinov2_s":"Dv2-S","dinov2_b":"Dv2-B","i21k_b":"ViT-B","clip_b":"CLIP-B","clip_b_vision":"CLIP-B"}
LEG = {"dinov2_b":"DINOv2-B","clip_b_vision":"CLIP-B"}

fig, ax = plt.subplots(1, 2, figsize=(5.5, 1.45))
# (b) non-hierarchical fine-tuning
r = list(csv.DictReader(open(RES/"analysis4_finetuning.csv")))
ft = {a["model"]: float(a["pct_change"]) for a in r}
ms = ["i21k_b","dinov2_s","clip_b"]
pct = [ft[m] for m in ms]
ax[0].bar(range(3), pct, color=[fam_color(m) for m in ms], width=0.6)   # family colors, not red
for i,v in enumerate(pct): ax[0].text(i, v+3, f"+{v:.0f}%", ha="center", fontsize=6.5, fontweight="bold")
ax[0].set_xticks(range(3)); ax[0].set_xticklabels([NAME[m] for m in ms], fontsize=6.5)
ax[0].set_ylabel(r"$\Delta\delta$ vs pretrained (%)", fontsize=6.5); ax[0].set_ylim(0, max(pct)*1.25)
ax[0].set_title("(a) Non-hierarchical fine-tuning", fontsize=7)
# (c) depth
r = list(csv.DictReader(open(RES/"e1_delta_by_layer.csv")))
for m, col in [("dinov2_b", fam_color("dinov2_b")), ("clip_b_vision", fam_color("clip_b"))]:
    rows = sorted([a for a in r if a["model"]==m and int(a["layer"])>=1], key=lambda a:int(a["layer"]))
    xs = [int(a["layer"]) for a in rows]; ys = [float(a["delta_normalized"]) for a in rows]
    ax[1].plot(xs, ys, "-o", color=col, ms=3, lw=1.2, label=LEG[m])
    print(m, "layers", xs, "delta first->last", round(ys[0],3), "->", round(ys[-1],3), f"({100*(ys[-1]-ys[0])/ys[0]:+.0f}%)")
ax[1].set_xlabel("transformer layer", fontsize=6.5); ax[1].set_ylabel(r"$\delta$", fontsize=6.5)
ax[1].set_title("(b) $\\delta$ falls across depth", fontsize=7); ax[1].legend(fontsize=6, frameon=False)
for a in ax: a.tick_params(labelsize=6.5)
fig.tight_layout(pad=0.6)
out = HERE.parent/"iclr2027"/"figures"
fig.savefig(out/"fig3_causal.pdf"); fig.savefig(out/"fig3_causal.png", dpi=200); fig.savefig(HERE/"fig3_causal.pdf")
print("fig3_causal regenerated ->", out)
