#!/usr/bin/env python3
"""Text-census figure: excess per text model with combined null+estimator error
bars. Source: expR48_text_census_bs1.csv (padding-free extraction, 20 null replicates)."""
import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
_FIGDIR = Path(__file__).resolve().parents[2] / "ICLR2027" / "figures"
plt.style.use(str(_FIGDIR / "style.mplstyle"))
sys.path.insert(0, str(_FIGDIR))
from palette import FAMILY_COLORS, color as fam_color
RES = Path(os.environ.get("PLATONIC_RESULTS", Path(__file__).resolve().parents[2] / "rebuttal/results"))
_src = RES / "expR53_text_haar_p999_200.csv"                            # text census of record (Phase B)
if not _src.exists(): _src = RES / "expR48b_text_census200_bs1.csv"
rows = list(csv.DictReader(open(_src)))
order = ["gpt2","gpt2_m","gpt2_l","gpt2_xl","pythia_410m","pythia_1b","pythia_2b8",
         "olmo_1b","olmo_7b","bge_base","bge_large","gte_base","gte_large",
         "e5_base","e5_large","gte_qwen2"]
NAME = {"gpt2":"GPT-2 S","gpt2_m":"GPT-2 M","gpt2_l":"GPT-2 L","gpt2_xl":"GPT-2 XL",
        "pythia_410m":"Pythia-410M","pythia_1b":"Pythia-1B","pythia_2b8":"Pythia-2.8B",
        "olmo_1b":"OLMo-1B","olmo_7b":"OLMo-7B","bge_base":"BGE-b","bge_large":"BGE-l",
        "gte_base":"GTE-b","gte_large":"GTE-l","e5_base":"E5-b","e5_large":"E5-l",
        "gte_qwen2":"GTE-Qwen2"}
d = {r["model"]: r for r in rows}
order = [m for m in order if m in d]
exc = [float(d[m]["excess"]) for m in order]
err = [np.sqrt(float(d[m]["null_sd"])**2 + float(d[m]["delta_sd"])**2) for m in order]
col = [fam_color(m) for m in order]                       # causal LMs purple, embedders brown
def _p(r):
    return float(r["p_left"]) if "p_left" in r else (1 + 20 - round(20*float(r["frac_null_above"]))) / 21
at_null = [(str(d[m]["genuine_bh"]) != "True") if "genuine_bh" in d[m] else _p(d[m]) > 0.05 for m in order]   # hollow = not genuine (BH p > 0.05)

fig, ax = plt.subplots(figsize=(5.5, 1.6))
x = np.arange(len(order))
bars = ax.bar(x, exc, yerr=err, capsize=2, error_kw=dict(lw=0.8))
for b, c, hollow in zip(bars, col, at_null):   # hollow = at null (not below every replicate)
    b.set_edgecolor(c); b.set_linewidth(1.0)
    b.set_facecolor("white" if hollow else c)
ax.axhline(0, color="k", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in order], rotation=55, ha="right", fontsize=6.5)
ax.set_ylabel(r"excess  $\hat\delta_{99.9}^{\rm real}-\hat\delta_{99.9}^{\rm null}$", fontsize=7)
ax.tick_params(labelsize=7)
import matplotlib.patches as mpatches
hd = [mpatches.Patch(facecolor=FAMILY_COLORS["causal_lm"], label="causal LM"),
      mpatches.Patch(facecolor=FAMILY_COLORS["embedder"], label="text embedder"),
      mpatches.Patch(facecolor="white", edgecolor="k", label="hollow: not genuine (BH $p>0.05$)")]
ax.legend(handles=hd, frameon=False, loc="lower right", ncol=1, handlelength=1.2, borderaxespad=0.3)
fig.tight_layout()
out = Path(__file__).parent / "figures"
fig.savefig(out/"fig_text_nulls.pdf"); fig.savefig(out/"fig_text_nulls.png", dpi=200)
print("fig_text_nulls regenerated with", len(order), "models")
