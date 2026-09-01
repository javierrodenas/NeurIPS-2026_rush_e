#!/usr/bin/env python3
"""Text-census figure: excess per text model with combined null+estimator error
bars. Source: expR48_text_census_bs1.csv (padding-free extraction, 20 null replicates)."""
import csv, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
RES = Path(os.environ.get("PLATONIC_RESULTS", Path(__file__).resolve().parents[2] / "rebuttal/results"))
rows = list(csv.DictReader(open(RES / "expR48_text_census_bs1.csv")))   # padding-free, 20 replicates
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
FAM = {"gpt2":"#DD8452","gpt2_m":"#DD8452","gpt2_l":"#DD8452","gpt2_xl":"#DD8452","pythia_410m":"#4C72B0","pythia_1b":"#4C72B0","pythia_2b8":"#4C72B0","olmo_1b":"#55A868","olmo_7b":"#55A868","bge_base":"#8172B3","bge_large":"#8172B3","gte_base":"#8172B3","gte_large":"#8172B3","e5_base":"#8172B3","e5_large":"#8172B3","gte_qwen2":"#937860"}
col = [FAM[m] for m in order]

fig, ax = plt.subplots(figsize=(5.5, 2.1))
x = np.arange(len(order))
ax.bar(x, exc, yerr=err, color=col, capsize=2, error_kw=dict(lw=0.8))
ax.axhline(0, color="k", lw=0.9)
ax.set_xticks(x); ax.set_xticklabels([NAME[m] for m in order], rotation=55, ha="right", fontsize=6.5)
ax.set_ylabel(r"tree excess  $\delta_{\rm real}-\delta_{\rm null}$", fontsize=7)
ax.tick_params(labelsize=7)
ax.annotate("genuine form emerges\nwith scale (GPT-2)", xy=(2.6, -0.038),
            xytext=(0.2, 0.030), fontsize=6.5, va="center", arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("classic sentence embedders: at null", xy=(11.4, 0.038), fontsize=6.5, ha="center")
qi = order.index("gte_qwen2")
ax.annotate("LLM-backboned:\nnot robust (App.)", xy=(qi, 0.004), xytext=(qi-2.0, -0.040), fontsize=6, ha="center", va="center", arrowprops=dict(arrowstyle="->", lw=0.7))
fig.tight_layout()
out = Path(__file__).parent / "figures"
fig.savefig(out/"fig_text_nulls.pdf"); fig.savefig(out/"fig_text_nulls.png", dpi=200)
print("fig_text_nulls regenerated with", len(order), "models")
