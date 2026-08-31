#!/usr/bin/env python3
"""Appendix tables for the restructured (form-not-content) paper, generated
from the night-run CSVs. Complements gen_appendix.py; nothing hand-typed."""
import csv
from pathlib import Path

RES = Path("/home/javi/Platonic/rebuttal/results")
OUT = Path(__file__).parent / "appendix_tables"
OUT.mkdir(exist_ok=True)

def load(p, root=RES):
    return list(csv.DictReader(open(root/p)))

NAME = {"gpt2":"GPT-2 S","gpt2_m":"GPT-2 M","gpt2_l":"GPT-2 L","gpt2_xl":"GPT-2 XL",
        "pythia_410m":"Pythia-410M","pythia_1b":"Pythia-1B","pythia_2b8":"Pythia-2.8B",
        "olmo_1b":"OLMo-1B","olmo_7b":"OLMo-7B",
        "bge_base":"BGE-base","bge_large":"BGE-large","gte_base":"GTE-base",
        "gte_large":"GTE-large","gte_qwen2":"GTE-Qwen2-1.5B","e5_base":"E5-base",
        "e5_large":"E5-large",
        "i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L",
        "dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B",
        "dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B",
        "clip_l":"CLIP-L","siglip_b":"SigLIP-B"}

# ---- B1: text-model nulls (exp18) ----
_order = ["gpt2","gpt2_m","gpt2_l","gpt2_xl","pythia_410m","pythia_1b","pythia_2b8",
          "olmo_1b","olmo_7b","bge_base","bge_large","gte_base","gte_large",
          "e5_base","e5_large","gte_qwen2"]
rows = sorted(load("exp18_text_nulls.csv"), key=lambda r: _order.index(r["model"]))
lines = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\begin{tabular}{llrcccc}", r"\toprule",
         r"model & type & $d$ & $\delta_{\text{norm}}$ & null & excess & $z$ \\", r"\midrule"]
for r in rows:
    lines.append(f'{NAME[r["model"]]} & {"causal LM" if r["kind"]=="causal" else "embedder"} & '
                 f'{int(float(r["d"]))} & {float(r["delta"]):.3f}$\\pm${float(r["delta_sd"]):.3f} & '
                 f'{float(r["null_mean"]):.3f}$\\pm${float(r["null_sd"]):.3f} & '
                 f'{float(r["excess"]):+.3f} & '
                 f'{float(r["excess"])/max((float(r["null_sd"])**2+float(r["delta_sd"])**2)**0.5,1e-9):+.1f} \\\\')
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Text models under the calibrated protocol (1000 ImageNet class prompts): "
          r"$\delta_{\text{norm}}$, spectrum-matched null (3 reps) and excess. "
          r"Source: \texttt{exp18\_text\_nulls.csv}.}", r"\label{tab:b1-textnulls}", r"\end{table}"]
(OUT/"tab_b1_textnulls.tex").write_text("\n".join(lines)+"\n"); print("b1")

# ---- B2: vision z-table (exp20), grid of z ----
z = {(r["model"], r["dataset"]): r for r in load("exp20_null_ztable.csv")}
DS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]
DSH = {"imagenet":"IN","cifar100":"C100","cifar10":"C10","dtd":"DTD","fashionmnist":"FMNIST","mnist":"MNIST"}
M12 = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l"+"c"*6+"}", r"\toprule",
         "model & " + " & ".join(DSH[d] for d in DS) + r" \\", r"\midrule"]
for m in M12:
    cells = []
    for d in DS:
        r = z[(m,d)]; ex = float(r["excess"]); zz = float(r["z"])
        s = f"{ex:+.3f} ({zz:+.1f})"
        cells.append(rf"\textbf{{{s}}}" if ex > 0 else s)
    lines.append(NAME[m] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Per-cell excess and $z$-score (in parentheses; $z=(\delta_{\text{real}}-"
          r"\bar\delta_{\text{null}})/\sqrt{\sigma_{\text{null}}^2+\sigma_{\text{est}}^2}$, "
          r"5 null replicates). Bold: the three sign-positive cells. "
          r"Source: \texttt{exp20\_null\_ztable.csv}.}", r"\label{tab:b2-ztable}", r"\end{table}"]
(OUT/"tab_b2_ztable.tex").write_text("\n".join(lines)+"\n"); print("b2")

# ---- B3: C-sweep summary (exp19), dinov2_l + clip_l ----
c19 = load("exp19_c_sweep.csv")
from statistics import mean
def agg(m,C,mode,f):
    v=[float(r[f]) for r in c19 if r["model"]==m and int(r["C"])==C and r["mode"]==mode]
    return mean(v) if v else None
lines = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\begin{tabular}{lr|ccc|ccc}", r"\toprule",
         r" & & \multicolumn{3}{c|}{random subsets (span the hierarchy)} & \multicolumn{3}{c}{WordNet-coherent (siblings)} \\",
         r"model & $C$ & $\delta_{\text{norm}}$ & excess & NC adv & $\delta_{\text{norm}}$ & excess & NC adv \\", r"\midrule"]
for m in ["dinov2_l","dinov2_g","clip_l"]:
    for C in [10,20,50,100,200,500,1000]:
        dr,er,nr = agg(m,C,"random","delta"),agg(m,C,"random","excess"),agg(m,C,"random","nc_adv_pp")
        dc,ec,nc = agg(m,C,"coherent","delta"),agg(m,C,"coherent","excess"),agg(m,C,"coherent","nc_adv_pp")
        right = f"{dc:.3f} & {ec:+.3f} & {nc:+.2f}" if dc is not None else "--- & --- & ---"
        lines.append(f"{NAME[m]} & {C} & {dr:.3f} & {er:+.3f} & {nr:+.2f} & {right} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}",
          r"\caption{Class-count vs hierarchy-depth deconfound on ImageNet (advantages in pp; "
          r"means over subset seeds; $C\le30$ uses exact quadruple enumeration). "
          r"Source: \texttt{exp19\_c\_sweep.csv}.}", r"\label{tab:b3-csweep}", r"\end{table}"]
(OUT/"tab_b3_csweep.tex").write_text("\n".join(lines)+"\n"); print("b3")

# ---- B4: t-sweep ----
ts = load("night/t_sweep.csv")
lines = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\begin{tabular}{l"+"c"*len(ts)+"}", r"\toprule",
         "target $t$ & " + " & ".join(f'{float(r["t"]):.2f}' for r in ts) + r" \\", r"\midrule",
         "FS adv.\\ mean (pp) & " + " & ".join(f'{float(r["FS_adv_mean"]):+.2f}' for r in ts) + r" \\",
         "NC adv.\\ mean (pp) & " + " & ".join(f'{float(r["NC_adv_mean"]):+.2f}' for r in ts) + r" \\",
         r"\% cells FS $>0$ & " + " & ".join(f'{100*float(r["FS_pos_frac"]):.0f}' for r in ts) + r" \\",
         r"\bottomrule", r"\end{tabular}",
         r"\caption{Sensitivity to the projection target $t$ (36 hierarchical cells). The headline "
         r"$t=1/\sqrt2\approx0.71$ was fixed a priori and is conservative for FS. "
         r"Source: \texttt{night/t\_sweep.csv}.}", r"\label{tab:b4-tsweep}", r"\end{table}"]
(OUT/"tab_b4_tsweep.tex").write_text("\n".join(lines)+"\n"); print("b4")

# ---- B5: correlation CIs ----
ci = load("night/correlation_cis.csv")
lines = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\begin{tabular}{llcc}", r"\toprule",
         r"task & dataset & $r$ & CI95 \\", r"\midrule"]
for r in ci:
    lines.append(f'{r["task"].replace("_adv","")} & {r["dataset"]} & {float(r["r"]):+.2f} & '
                 f'[{float(r["ci_lo"]):+.2f}, {float(r["ci_hi"]):+.2f}] \\\\')
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Bootstrap CI95 for the correlations between $\delta_{\text{norm}}$ and the "
          r"projection advantage (within-dataset: 10k resamples of the 10 models; pooled: cluster "
          r"bootstrap over datasets). Source: \texttt{night/correlation\_cis.csv}.}",
          r"\label{tab:b5-cis}", r"\end{table}"]
(OUT/"tab_b5_cis.tex").write_text("\n".join(lines)+"\n"); print("b5")

# ---- B6: ORC bridges ----
br = load("night/orc_bridges.csv")
lines = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\begin{tabular}{llcccc}", r"\toprule",
         r"model & dataset & ORC within & ORC across & \%neg within & \%neg across \\", r"\midrule"]
for r in br:
    lines.append(f'{NAME[r["model"]]} & {DSH.get(r["dataset"],r["dataset"])} & '
                 f'{float(r["orc_within"]):.2f} & {float(r["orc_across"]):.2f} & '
                 f'{100*float(r["fneg_within"]):.1f} & {100*float(r["fneg_across"]):.1f} \\\\')
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Edge-type ORC on the centroid kNN graph (30 WordNet superclasses): negative "
          r"curvature concentrates on between-cluster bridges. Source: \texttt{night/orc\_bridges.csv}.}",
          r"\label{tab:b6-bridges}", r"\end{table}"]
(OUT/"tab_b6_bridges.tex").write_text("\n".join(lines)+"\n"); print("b6")
print("done")


# ---- B7: tree-map metric/linkage controls (exp23 + config diagnostics) ----
import numpy as np, json
diag = json.load(open(RES/"exp23_config_diagnostics.json"))
SELECTED = {"imagenet": "cosine-average", "cifar100": "cosine-complete"}
lines = [r"\begin{table}[H]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"dataset & configuration & max frac.\ & CPCC & Dv2-B/L/G vs block & Dv2-S+DINO-B vs block & within block \\",
         r"\midrule"]
for ds, tag in [("imagenet","summary_in"), ("cifar100","summary_c1")]:
    z = np.load(RES/"exp23_treemap_controls.npz", allow_pickle=True)[tag].item()
    for key in ["('euclid', 'average')","('euclid', 'complete')","('euclid', 'ward')",
                "('cosine', 'average')","('cosine', 'complete')","('cosine', 'ward')"]:
        v = z[key]
        cfg = key.replace("('","").replace("')","").replace("', '","-")
        metric, link = cfg.split("-")
        d = diag[f"{ds}|{metric}|{link}"]
        deg = d["maxfrac"] > 0.5
        mark = r" (selected)" if (cfg == SELECTED[ds] and not deg) else (" (deg.)" if deg else "")
        lines.append(f"{ds} & {cfg}{mark} & {d['maxfrac']:.2f} & {d['cpcc']:.3f} & "
                     f"{v['big_vs_sup']:.2f} & {v['small_vs_sup']:.2f} & {v['sup_vs_sup']:.2f} \\\\")
    lines.append(r"\midrule")
lines[-1] = r"\bottomrule"
lines += [r"\end{tabular}",
          r"\caption{Tree-map configurations with the two quantities that drive the selection "
          r"criterion of \S5: the degeneracy diagnostic (largest-cluster fraction at the reference "
          r"cut, worst model) and the mean cophenetic fidelity (CPCC) of each model's dendrogram to "
          r"its own distance matrix. Configurations with max-cluster fraction $>0.5$ are excluded; "
          r"among the rest, highest CPCC selects the marked row. Mean pairwise ARIs at the reference "
          r"cut shown for all configurations. Sources: \texttt{exp23\_treemap\_controls.npz}, "
          r"\texttt{exp23\_config\_diagnostics.json}.}",
          r"\label{tab:b7-treemapcontrols}", r"\end{table}"]
(OUT/"tab_b7_treemap.tex").write_text("\n".join(lines)+"\n"); print("b7")

# ---- B8: validation-split metric selection (exp24) ----
r24 = load("exp24_val_metric_selection.csv")
HIER = {"imagenet","cifar100","cifar10","dtd"}
from statistics import mean as _mean
def pol(rows, col): return _mean(float(r[col]) for r in rows)
h24 = [r for r in r24 if r["dataset"] in HIER]
d20g = {(r["model"],r["dataset"]): float(r["delta"]) for r in load("exp20_null_ztable.csv")}
FLATg = {"fashionmnist","mnist"}
def dgate_adv(r):
    if d20g[(r["model"],r["dataset"])] >= 0.10 or r["dataset"] in FLATg: return 0.0
    return float(r["adv_rule"])

lines = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"policy (test advantage over Euclidean, pp) & all 60 cells & hierarchical (40) \\",
         r"\midrule",
         f"always cosine & {pol(r24,'adv_cos'):+.2f} & {pol(h24,'adv_cos'):+.2f} \\\\",
         f"objective rule (zero validation) & {pol(r24,'adv_rule'):+.2f} & {pol(h24,'adv_rule'):+.2f} \\\\",
         f"$\\delta$-gated objective rule & {_mean(dgate_adv(r) for r in r24):+.2f} & {_mean(dgate_adv(r) for r in h24):+.2f} \\\\",
         f"validation-picked per cell & {pol(r24,'adv_val'):+.2f} & {pol(h24,'adv_val'):+.2f} \\\\",
         f"test oracle (reference) & {pol(r24,'adv_oracle'):+.2f} & {pol(h24,'adv_oracle'):+.2f} \\\\",
         r"\bottomrule", r"\end{tabular}",
         r"\caption{Metric-selection policies evaluated on held-out episodes (500 validation / "
         r"500 test per cell, paper protocol). Validation picking retains the oracle advantage; "
         r"the zero-validation objective rule modestly beats always-cosine. "
         r"Source: \texttt{exp24\_val\_metric\_selection.csv}.}",
         r"\label{tab:b8-valpick}", r"\end{table}"]
(OUT/"tab_b8_valpick.tex").write_text("\n".join(lines)+"\n"); print("b8")


# ---- B9: NC metric-selection policies with the delta gate (exp2 + exp20) ----
e2p = {(r["model"],r["dataset"]): r for r in load("exp2_metric_controls.csv")}
d20p = {(r["model"],r["dataset"]): float(r["delta"]) for r in load("exp20_null_ztable.csv")}
FLATp = {"fashionmnist","mnist"}
def famp(m): return "ssl" if m.startswith("dinov") else "sup" if m.startswith("i21k") else "con"
pols = {"always cosine": [], "always Poincar\\'e": [], "gated rule (flat$\\to$R)": [], "$\\delta$-gated rule": []}
splits = {"all": [], "flat": [], "hier": []}
recs = []
for (m,ds), r in e2p.items():
    if (m,ds) not in d20p: continue
    R_,H_,C_ = float(r["NC_R"]), float(r["NC_H"]), float(r["NC_COS"])
    obj = C_ if famp(m) in ("ssl","sup") else H_
    row = {"always cosine": (C_-R_)*100, "always Poincar\\'e": (H_-R_)*100,
           "gated rule (flat$\\to$R)": ((R_ if ds in FLATp else obj)-R_)*100,
           "$\\delta$-gated rule": ((R_ if (d20p[(m,ds)]>=0.10 or ds in FLATp) else obj)-R_)*100,
           "flat": ds in FLATp}
    recs.append(row)
from statistics import mean as _m
lines = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\begin{tabular}{lccc}", r"\toprule",
         r"NC policy (advantage over Euclidean, pp) & all 60 & flat (20) & hierarchical (40) \\",
         r"\midrule"]
for pol in ["always cosine","always Poincar\\'e","gated rule (flat$\\to$R)","$\\delta$-gated rule"]:
    a=_m(x[pol] for x in recs); f=_m(x[pol] for x in recs if x["flat"]); h=_m(x[pol] for x in recs if not x["flat"])
    lines.append(f"{pol} & {a:+.2f} & {f:+.2f} & {h:+.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{NC metric-selection policies. The gate's value is avoiding the Poincar\'e "
          r"projection on flat-label data; cosine is a safe NC default everywhere. "
          r"Sources: \texttt{exp2\_metric\_controls.csv}, \texttt{exp20\_null\_ztable.csv}.}",
          r"\label{tab:b9-ncpolicies}", r"\end{table}"]
(OUT/"tab_b9_ncpolicies.tex").write_text("\n".join(lines)+"\n"); print("b9")

# ---- B11: ground-truth recovery per model x configuration (C100) ----
r28 = load("exp28_recovery_per_config.csv")
CONFS = ["euclid-average","euclid-ward","cosine-average","cosine-complete","cosine-ward"]
HEAD = ["euclid-avg (naive, degen.)","euclid-ward","cosine-avg (degen.\\ C100)","cosine-complete $\\leftarrow$ sel.","cosine-ward"]
lines = [r"\begin{table}[H]", r"\centering", r"\scriptsize",
         r"\begin{tabular}{l" + "c"*5 + "}", r"\toprule",
         "model & " + " & ".join(HEAD) + r" \\", r"\midrule"]
for r in r28:
    lines.append(NAME[r["model"]] + " & " + " & ".join(f'{float(r[c]):.2f}' for c in CONFS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Recovery of the true CIFAR-100 superclasses (ARI of the 20-cluster cut) per "
          r"model and configuration, the quantity behind the corrected picture of \S5. Degenerate "
          r"configurations marked; the criterion-selected configuration is cosine-complete. "
          r"Source: \texttt{exp28\_recovery\_per\_config.csv}.}",
          r"\label{tab:b11-recovery}", r"\end{table}"]
(OUT/"tab_b11_recovery.tex").write_text("\n".join(lines)+"\n"); print("b11")
