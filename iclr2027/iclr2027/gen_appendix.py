#!/usr/bin/env python3
"""Generate appendix tables (appendix_tables/*.tex) from the result CSVs.
Every table in the appendix is produced by this script; nothing is hand-typed.
"""
import csv
from pathlib import Path

RES = Path("/home/javi/Platonic/rebuttal/results")
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
    lines = [r"\begin{table}[h]", r"\centering", r"\small",
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

# ---- A6: McNemar p-values (NC, H vs R), exp13 ----
mc = {(r["model"], r["dataset"]): r for r in load("exp13_mcnemar.csv")}
def p_cell(m, d):
    p = float(mc[(m, d)]["p_H_vs_R"])
    return rf"$10^{{{max(-99, int(f'{p:.0e}'.split('e')[1])) }}}$" if p < 1e-3 else f"{p:.2f}"
grid_table("tab_a6_mcnemar.tex",
    r"McNemar $p$-values for NC test-set decisions, Poincar\'e vs Euclidean "
    r"(order of magnitude when $p<10^{-3}$). Source: \texttt{exp13\_mcnemar.csv}.",
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
lines = [r"\begin{table}[h]", r"\centering", r"\scriptsize",
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
lines = [r"\begin{table}[h]", r"\centering", r"\small",
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
lines = [r"\begin{table}[h]", r"\centering", r"\small",
         r"\begin{tabular}{lcccccc}", r"\toprule",
         r"model & $\hat\delta$ & excess & trip.\ (cos) & NC H$-$R (pp) & FS H$-$R (pp) & CI95 \\",
         r"\midrule"]
for r in db:
    nc = (float(r["NC_H"]) - float(r["NC_R"])) * 100
    lines.append(f'{NAME[r["model"]]} & {float(r["delta"]):.3f} & '
                 f'{float(r["excess"]):+.3f} & {float(r["trip_cos"]):.2f} & {nc:+.2f} & '
                 f'{float(r["FS_HR_pp"]):+.2f} & $\\pm${float(r["FS_HR_ci"]):.2f} \\\\')
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{DBpedia Classes (219 leaf classes, 3 levels): genuine tree excess and "
          r"strong sibling alignment, yet $\hat\delta$ above the low-$\delta$ band correctly "
          r"predicts marginal gains. Source: \texttt{exp14\_dbpedia.csv}.}",
          r"\label{tab:a9}", r"\end{table}"]
(OUT/"tab_a9_dbpedia.tex").write_text("\n".join(lines) + "\n"); print("wrote tab_a9_dbpedia.tex")

# ---- A10: curvature sign (exp12) ----
cv = load("exp12_curvature_sign.csv")
lines = [r"\begin{table}[h]", r"\centering", r"\small",
         r"\begin{tabular}{lcc}", r"\toprule",
         r"space / model & $\xi$ (mean) & frac.\ negative \\", r"\midrule"]
REF = {"ref_tree_d9":"balanced tree (depth 9)","ref_H2_R6":"$\\mathbb{H}^2$ region ($R{=}6$)",
       "ref_gauss768":"iid Gaussian ($d{=}768$)","ref_sphere_geo":"$S^{d}$ (geodesic)",
       "ref_sphere_chord":"$S^{d}$ (chord)"}
for r in cv:
    nm = REF.get(r["model"], NAME.get(r["model"], r["model"]))
    lines.append(f'{nm} & {float(r["xi_mean"]):+.3f} & {100*float(r["frac_neg"]):.0f}\\% \\\\')
    if r["model"] == "ref_sphere_chord":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Parallelogram curvature-sign estimator $\xi$ on reference geometries "
          r"(top) and ImageNet centroids (bottom). Source: \texttt{exp12\_curvature\_sign.csv}.}",
          r"\label{tab:a10}", r"\end{table}"]
(OUT/"tab_a10_curvature.tex").write_text("\n".join(lines) + "\n"); print("wrote tab_a10_curvature.tex")

print("all appendix tables generated")
