#!/usr/bin/env python3
"""Text-census figure: excess per text model with combined null+estimator error
bars. Source: exp18_text_nulls.csv."""
import csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

rows = list(csv.DictReader(open("/home/javi/Platonic/rebuttal/results/exp18_text_nulls.csv")))
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
col = (["#DD8452"]*4 + ["#4C72B0"]*3 + ["#55A868"]*2 + ["#8172B3"]*6 + ["#937860"])[:len(order)]

fig, ax = plt.subplots(figsize=(8.8, 3.0))
x = np.arange(len(order))
ax.bar(x, exc, yerr=err, color=col, capsize=2, error_kw=dict(lw=0.8))
ax.axhline(0, color="k", lw=0.9)
ax.set_xticks(x, [NAME[m] for m in order], rotation=45, ha="right", fontsize=7)
ax.set_ylabel(r"tree excess  $\delta_{\rm real}-\delta_{\rm null}$", fontsize=8)
ax.tick_params(labelsize=7)
ax.annotate("genuine form emerges with scale\n(GPT-2 and OLMo)", xy=(3, -0.044),
            xytext=(4.4, -0.038), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.8))
ax.annotate("classic sentence embedders: at null", xy=(11.5, 0.014), fontsize=7, ha="center")
fig.tight_layout()
out = Path(__file__).parent / "figures"
fig.savefig(out/"fig_text_nulls.pdf"); fig.savefig(out/"fig_text_nulls.png", dpi=170)
print("fig_text_nulls regenerated with", len(order), "models")
