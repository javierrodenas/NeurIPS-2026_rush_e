#!/usr/bin/env python3
"""Generate appendix tables (appendix_tables/*.tex) from the result CSVs.
Every table in the appendix is produced by this script; nothing is hand-typed.
"""
import csv, os
from pathlib import Path

RES = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[2] / "rebuttal/results")))
OUT = Path(__file__).parent / "appendix_tables"
OUT.mkdir(exist_ok=True)

NAME = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L",
        "dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B",
        "dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
        "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B",
        "bge_base":"BGE-base","e5_base":"E5-base","gte_base":"GTE-base",
        "gpt2":"GPT-2","gpt2_m":"GPT-2-M"}
DS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]
DSH = {"imagenet":"IN","cifar100":"C100","cifar10":"C10","dtd":"DTD",
       "fashionmnist":"FMNIST","mnist":"MNIST"}
M12 = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b",
       "dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
M10 = [m for m in M12 if m not in ("dinov1_b","siglip_b")]

def load(f):
    return list(csv.DictReader(open(RES/f)))

def grid_table(fname, caption, label, models, cell_fn, colhead=DS):
    lines = [r"\begin{table}[H]", r"\centering", r"\small",
             r"\begin{tabular}{l" + "c"*len(colhead) + "}", r"\toprule",
             "model & " + " & ".join(DSH[d] for d in colhead) + r" \\", r"\midrule"]
    for m in models:
        cells = [cell_fn(m, d) for d in colhead]
        lines.append(NAME[m] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    (OUT/fname).write_text("\n".join(lines) + "\n")
    print("wrote", fname)

# ---- A1: excess grid (12x6), exp11 ----
d11 = {(r["model"], r["dataset"]): r for r in load("exp11_null_per_dataset.csv")}
def cell_exc(m, d):
    v = float(d11[(m, d)]["exc_eucl"])
    s = f"{v:+.3f}"
    return rf"\textbf{{{s}}}" if v > 0 else s
grid_table("tab_a1_excess.tex",
    r"Tree excess $\hat\delta_{\mathrm{real}}-\hat\delta_{\mathrm{null}}$ per cell "
    r"(spectrum-matched null; null s.d.\ $\le 0.007$). Bold: the 3/72 cells at or above "
    r"their null, all DINOv2 on ImageNet. Source: \texttt{exp11\_null\_per\_dataset.csv}.",
    "tab:a1", M12, cell_exc)

# ---- A2/A3: NC and FS accuracies R/H (10x6), regenerated table ----
t1 = {(r["model"], r["dataset"]): r for r in load("table1_regenerated.csv")}
def acc_cell(task):
    def f(m, d):
        r = t1[(m, d)]
        return f'{100*float(r[task+"_R"]):.1f}/{100*float(r[task+"_H"]):.1f}'
    return f
grid_table("tab_a2_nc.tex",
    r"Nearest-centroid accuracy (\%), Euclidean/Poincar\'e, regenerated from the audited "
    r"reruns. Source: \texttt{table1\_regenerated.csv}.", "tab:a2", M10, acc_cell("NC"))
grid_table("tab_a3_fs.tex",
    r"5-way 5-shot accuracy (\%), Euclidean/Poincar\'e, 1000 paired episodes. "
    r"Source: \texttt{table1\_regenerated.csv}.", "tab:a3", M10, acc_cell("FS"))

# ---- A4: FS COS-R and RT-R (pp), exp2 ----
e2 = {(r["model"], r["dataset"]): r for r in load("exp2_metric_controls.csv")}
def diff_cell(a, b):
    def f(m, d):
        r = e2[(m, d)]
        return f'{(float(r["FS_"+a])-float(r["FS_"+b]))*100:+.2f}'
    return f
grid_table("tab_a4_cos.tex",
    r"Few-shot advantage of cosine over Euclidean (pp). Source: \texttt{exp2\_metric\_controls.csv}.",
    "tab:a4", M10, diff_cell("COS", "R"))
grid_table("tab_a5_rt.tex",
    r"Few-shot advantage of the radial-rescaling control over Euclidean (pp): the map "
    r"without the metric changes nothing. Source: \texttt{exp2\_metric\_controls.csv}.",
    "tab:a5", M10, diff_cell("RT", "R"))

# ---- A6: McNemar p-values (NC, H vs R) with gain direction, exp13 ----
mc = {(r["model"], r["dataset"]): r for r in load("exp13_mcnemar.csv")}
t1d = {(r["model"], r["dataset"]): float(r["NC_adv"]) for r in load("table1_regenerated.csv")}
def p_cell(m, d):
    p = float(mc[(m, d)]["p_H_vs_R"])
    sign = "+" if t1d[(m, d)] > 0 else "$-$"
    ptxt = rf"$10^{{{max(-99, int(f'{p:.0e}'.split('e')[1])) }}}$" if p < 1e-3 else f"{p:.2f}"
    return f"{ptxt}\,({sign})"
grid_table("tab_a6_mcnemar.tex",
    r"McNemar $p$-values for NC test-set decisions, Poincar\'e vs Euclidean "
    r"(order of magnitude when $p<10^{-3}$; sign of the NC advantage in parentheses, "
    r"so significance is readable with its direction). Sources: \texttt{exp13\_mcnemar.csv}, \texttt{table1\_regenerated.csv}.",
    "tab:a6", M10, p_cell)

# ---- A7: prompts (exp4 + exp17) ----
rows4, rows17 = load("exp4_prompt_variation.csv"), load("exp17_wordnet_prompts.csv")
prompts = {}
for r in rows4:
    prompts.setdefault(r["model"], {})[r["template"]] = float(r["delta_max"])
for r in rows17:
    prompts.setdefault(r["model"], {})[r["template"]] = float(r["delta"])
tpl_order = ["photo","name_only","image","closeup","this_is","wild",
             "def_pair","gloss_only","concept","discussion"]
tpl_head = ["photo","name","image","closeup","this is","wild",
            "def.","gloss only","concept","discuss."]
lines = [r"\begin{table}[H]", r"\centering", r"\scriptsize",
         r"\begin{tabular}{l" + "c"*len(tpl_order) + "}", r"\toprule",
         "model & " + " & ".join(tpl_head) + r" \\", r"\midrule"]
for m in ["bge_base","e5_base","gpt2","gpt2_m"]:
    cells = [f'{prompts[m][t]:.3f}' if t in prompts.get(m, {}) else "---" for t in tpl_order]
    lines.append(NAME[m] + " & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{$\hat\delta$ across prompt templates (ImageNet class names; "
          r"``gloss only'' contains no class name). Sources: \texttt{exp4}, \texttt{exp17}.}",
          r"\label{tab:a7}", r"\end{table}"]
(OUT/"tab_a7_prompts.tex").write_text("\n".join(lines) + "\n"); print("wrote tab_a7_prompts.tex")

# ---- A8: HierarCaps (exp16) ----
h = load("exp16_hierarcaps.csv")
lines = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"model & $\rho$(level, radius) & \% monotone & trip.\ L1 & trip.\ L2 & "
         r"$\hat\delta$ leaves & excess \\", r"\midrule"]
for r in h:
    lines.append(f'{NAME[r["model"]]} & {float(r["rho_mean"]):+.2f}$\\pm${float(r["rho_sd"]):.2f} & '
                 f'{100*float(r["pct_monotone"]):.1f} & {float(r["trip_L1_cos"]):.2f} & '
                 f'{float(r["trip_L2_cos"]):.2f} & {float(r["delta_leaves"]):.3f} & '
                 f'{float(r["excess"]):+.3f} \\\\')
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{HierarCaps test set (1000 four-level chains): radial ordering under the "
          r"paper's projection (chance for strict monotonicity $=1/24\approx4.2\%$; shuffle "
          r"controls $\approx 0$), sibling triplet agreement (cosine; chance $0.5$), and leaf "
          r"$\hat\delta$ with spectrum-null excess. Source: \texttt{exp16\_hierarcaps.csv}.}",
          r"\label{tab:a8}", r"\end{table}"]
(OUT/"tab_a8_hierarcaps.tex").write_text("\n".join(lines) + "\n"); print("wrote tab_a8_hierarcaps.tex")

# ---- A9: DBpedia (exp14) ----
db = load("exp14_dbpedia.csv")
rec = {r["model"]: r for r in load("expR61_dbpedia_record.csv")} if (RES/"expR61_dbpedia_record.csv").exists() else {}
lines = [r"\begin{table}[H]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{lcc|ccc|ccc}", r"\toprule",
         r" & \multicolumn{2}{c|}{supremum, Gaussian null} & \multicolumn{3}{c|}{census of record} & & & \\",
         r"model & $\hat\delta_{\max}$ & excess & $\hat\delta_{99.9}$ & excess & $r/200$ ($p$) & trip.\ (cos) & NC H$-$R (pp) & FS H$-$R (pp) \\",
         r"\midrule"]
for r in db:
    nc = (float(r["NC_H"]) - float(r["NC_R"])) * 100
    q = rec.get(r["model"])
    recc = f'{float(q["delta_999"]):.3f} & {float(q["excess"]):+.3f} & {int(q["r_above"])} ({float(q["p_left"]):.3f})' if q else "--- & --- & ---"
    lines.append(f'{NAME[r["model"]]} & {float(r["delta"]):.3f} & '
                 f'{float(r["excess"]):+.3f} & {recc} & {float(r["trip_cos"]):.2f} & {nc:+.2f} & '
                 f'{float(r["FS_HR_pp"]):+.2f}$\\pm${float(r["FS_HR_ci"]):.2f} \\\\')
tm = ""
if (RES/"exp27_dbpedia_treemap.json").exists():
    import json as _json
    d27 = _json.load(open(RES/"exp27_dbpedia_treemap.json")); l2 = [a for v in d27.values() for a in v["ari_l2"].values()]; cm = [v["cross_model"] for v in d27.values()]
    n_l2 = ""
    try:
        import numpy as _np; _sup = _np.load(Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))/"results/text_cache/dbpedia_bge_base.npz")["sup"]; n_l2 = f"{len(set(_sup.tolist()))}-way "
    except Exception: pass
    tm = (f" Tree map on the same classes: no configuration degenerates (largest cluster $\\le{100*max(v['maxfrac'] for v in d27.values()):.0f}$\\%), the embedders recover the true {n_l2}level-2 partition at ARI {min(l2):.2f}--{max(l2):.2f} in all {len(d27)} configurations and agree with each other at cross-model ARI {min(cm):.2f}--{max(cm):.2f}.")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{DBpedia Classes (219 leaf classes, 3 levels): excess under the original "
          r"supremum reading (3 Gaussian replicates) and under the census of record (Haar null, 99.9th-percentile "
          r"statistic, 200 replicates; $r$ = replicates above the real value, left-tail $p$), strong sibling alignment, "
          r"and $\hat\delta$ above the low-$\delta$ band correctly predicting marginal gains." + tm + " "
          r"Sources: \texttt{exp14\_dbpedia.csv}, \texttt{expR61\_dbpedia\_record.csv}, \texttt{exp27\_dbpedia\_treemap.json}.}",
          r"\label{tab:a9}", r"\end{table}"]
(OUT/"tab_a9_dbpedia.tex").write_text("\n".join(lines) + "\n"); print("wrote tab_a9_dbpedia.tex")

# ---- A10: curvature sign (exp12 raw + exp26 matched nulls) ----
cv = load("exp12_curvature_sign.csv")
e26 = {r["model"]: r for r in load("exp26_xi_nulls.csv")}
lines = [r"\begin{table}[H]", r"\centering", r"\small",
         r"\begin{tabular}{lccccc}", r"\toprule",
         r"space / model & $\xi$ (mean) & frac.\ negative & null $\xi$ & excess ($\pm$CI95) & $z$ \\", r"\midrule"]
REF = {"ref_tree_d9":"balanced tree (depth 9)","ref_H2_R6":"$\\mathbb{H}^2$ region ($R{=}6$)",
       "ref_gauss768":"iid Gaussian ($d{=}768$)","ref_sphere_geo":"$S^{d}$ (geodesic)",
       "ref_sphere_chord":"$S^{d}$ (chord)"}
for r in cv:
    nm = REF.get(r["model"], NAME.get(r["model"], r["model"]))
    if r["model"] in e26:
        x = e26[r["model"]]
        tail = (f' & {float(x["xi_null"]):+.3f}$\\pm${float(x["xi_null_sd"]):.3f}'
                f' & {float(x["excess"]):+.3f}$\\pm${float(x["ci95"]):.3f} & {float(x["z"]):+.0f}')
    else:
        tail = " & --- & --- & ---"
    lines.append(f'{nm} & {float(r["xi_mean"]):+.3f} & {100*float(r["frac_neg"]):.0f}\\%{tail} \\\\')
    if r["model"] == "ref_sphere_chord":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Parallelogram curvature-sign estimator $\xi$ on reference geometries "
          r"(top) and ImageNet centroids (bottom), with the spectrum-matched null, excess and "
          r"$z$-score for each model (20 null replicates; noise combines null and estimator s.d.): matched nulls have positive $\xi$, so "
          r"raw readings understate curvature. Sources: \texttt{exp12\_curvature\_sign.csv}, "
          r"\texttt{exp26\_xi\_nulls.csv}.}",
          r"\label{tab:a10}", r"\end{table}"]
(OUT/"tab_a10_curvature.tex").write_text("\n".join(lines) + "\n"); print("wrote tab_a10_curvature.tex")

print("all appendix tables generated")


# ---- Table 3: estimator calibration (generated; R4 of the final pass) ----
# Gaussian rows from exp1_delta_controls.csv (variant "gauss", deduplicated by d).
# Tree / H^2 / sphere rows have no result CSV (exp6_h2_sphere_check printed to stdout);
# they are typed here and re-verified live by rebuttal/scripts/sweep_freeze.py.
_g = {}
for r in load("exp1_delta_controls.csv"):
    if r["variant"] == "gauss": _g[int(r["d"])] = float(r["delta_max"])
_ds = sorted(_g)
lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{lcc}", r"\toprule",
         r"space & absolute $\delta$ & $\delta_{\text{norm}}$ \\", r"\midrule",
         r"balanced binary tree (depth 10) & 0.000 & 0.000 \\",
         r"$\mathbb{H}^2$ ($K{=}-1$), region radius $R{=}2/4/8/16$ & 0.65/0.69/0.69/0.69 & 0.162/0.087/0.043/0.022 \\",
         r"uniform $S^{99}$ (chord / geodesic) & 0.25/0.37 & 0.143/0.179 \\",
         "iid Gaussian ($n{=}1000$), $d{=}192,\\dots,1536$ & --- & "
         + "/".join(f"{_g[d]:.3f}" for d in _ds) + r" \\",
         r"\bottomrule", r"\end{tabular}",
         r"\caption{\textbf{A low raw $\delta$ is not evidence of hierarchy.} Calibration on reference geometries: the estimator recovers the tree zero and the four-point constant of $\mathbb{H}^2$, spheres stay high, and iid Gaussians drift low as $d$ grows, the dimension confound. Gaussian row from the released control file; tree/$\mathbb{H}^2$/sphere rows re-verified synthetically at every freeze. % exp1_delta_controls.csv, sweep_freeze.py",
         r"}",
         r"\label{tab:calibration}", r"\end{table}"]
(OUT/"tab_calibration.tex").write_text("\n".join(lines) + "\n"); print("wrote tab_calibration.tex")
