#!/usr/bin/env python3
"""Main-text census table (Table 1, final pass): 7 columns, every value read from the released
result files. Sources: expR39b_census20_cache.csv (homogeneous 20-replicate census on the census
cache; until it exists, expR39_census20.csv is used for layout and a warning printed) and
exp3_alignment.csv (WordNet alignment). The columns removed from the old 10-column table
(p99.9, DTD, best-R, metric) move to appendix_tables/tab_census_extra.tex, generated here from
expR34_p999_imagenet.csv and exp2_metric_controls.csv."""
import csv, os, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
def load(f): return list(csv.DictReader(open(RES / f)))
NAME = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
        "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
        "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
FAM = {"i21k":"sup.","dinov":"SSL","clip":"contr.","sigli":"contr."}
def fam(m): return FAM["dinov" if m.startswith("dinov") else m[:5] if m.startswith("sigli") else m[:4] if m.startswith(("i21k","clip")) else m]
ORDER = list(NAME)
src = "expR52_census_haar_p999_200.csv"                  # census of record (Phase B): Haar null x p99.9 statistic, BH
if not (RES/src).exists():
    src = "expR39c_census200_cache.csv"; print(f"WARNING: expR52 not found; layout run from {src}")
cen = {(r["model"], r["dataset"]): r for r in load(src)}
N = 200
def rank(r): return int(r["r_above"])
def pleft(r): return float(r["p_left"])
def genuine(r): return str(r.get("genuine_bh", str(pleft(r) <= 0.05))) == "True"
RMIN = 191
d3 = {r["model"]: r for r in load("exp3_alignment.csv")}
SLF = {(r["model"], r["dataset"]): r for r in load("expR62_samplelevel_record.csv")} if (RES/"expR62_samplelevel_record.csv").exists() else {}   # sample-level reading (restructuring, R7)
SL = {}   # final pass (brief 5): Table 1 back to \small; the sample-level columns live in the appendix table tab:q3-sample
rows = []
for m in ORDER:
    r = cen[(m, "imagenet")]
    rows.append((NAME[m], fam(m), float(r["delta"]),
                 float(r["excess"]), rank(r), pleft(r), genuine(r),
                 float(cen[(m, "cifar100")]["excess"]), genuine(cen[(m, "cifar100")]),
                 float(cen[(m, "cifar10")]["excess"]), genuine(cen[(m, "cifar10")]),
                 float(d3[m]["spearman_wn"])))
sl_head = r" & \multicolumn{2}{c}{sample-level exc.} \\" if SL else ""
lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize" if SL else r"\small", r"\setlength{\tabcolsep}{2.6pt}" if SL else r"\setlength{\tabcolsep}{3.4pt}",
         r"\begin{tabular}{l l c c c c c c" + (" | c c" if SL else "") + "}", r"\toprule"]
if SL: lines.append(r" & & \multicolumn{6}{c|}{class level (centroids)} & \multicolumn{2}{c}{sample level (images)} \\ \cmidrule(lr){3-8}\cmidrule(lr){9-10}")
lines += [r"model & family & $\hat\delta_{99.9}$ IN & excess IN & $r$/200 ($p$) & exc.\ C100 & exc.\ C10 & $\rho_{\text{WN}}$" + (r" & exc.\ C100 & exc.\ DTD" if SL else "") + r" \\", r"\midrule"]
def pf(p): return f"{p:.3f}".lstrip("0") if p < 1 else "1"
for i, (n, f, dl, ex, rk, pv, g, e100, g100, e10, g10, rho) in enumerate(rows):
    if i in (4, 9): lines.append(r"\midrule")
    exs = f"${ex:+.3f}" + ("" if g else r"^{\circ}") + "$"
    if ex > 0: exs = r"\textbf{" + exs + "}"
    rp = f"{rk} ({pf(pv)})"
    c100 = f"${e100:+.3f}" + ("" if g100 else r"^{\circ}") + "$"; c10 = f"${e10:+.3f}" + ("" if g10 else r"^{\circ}") + "$"
    if e100 > 0: c100 = r"\textbf{" + c100 + "}"
    if e10 > 0: c10 = r"\textbf{" + c10 + "}"
    slc = ""
    if SL:
        m = ORDER[i]; cells = []
        for ds in ("cifar100", "dtd"):
            a = SL[(m, ds)]; gg = str(a["genuine_bh"]) == "True"; v = float(a["excess"])
            cells.append(f"${v:+.3f}" + ("" if gg else r"^{\circ}") + "$")
        slc = " & " + " & ".join(cells)
    lines.append(f"{n} & {f} & ${dl:.3f}$ & {exs} & {rp} & {c100} & {c10} & ${rho:+.2f}${slc} \\\\")
sl_n = sum(1 for k, a in SLF.items() if str(a["genuine_bh"]) == "True")
JS = load("expR66_joint_sensitivity_summary.csv") if (RES/"expR66_joint_sensitivity_summary.csv").exists() else []
js_txt = ""
if JS:
    a, b = sum(r["joint_genuine"] == "True" for r in JS), sum(r["boot_bh_genuine"] == "True" for r in JS)
    js_txt = f"Under image resampling and estimator noise the genuine count is {min(a,b)}--{max(a,b)} of 72 (Table~\\ref{{tab:q8-robust}}). "
sl_txt = (f"The same reading on per-image features is genuine in {sl_n} of 24 cells (Table~\\ref{{tab:q3-sample}}). " if SLF else "")
lines += [r"\bottomrule", r"\end{tabular}",
  r"\caption{\textbf{Clustered structure is the rule at the class level.} "
  r"One row per backbone. $\hat\delta_{99.9}$: raw ImageNet reading; excess: $\hat\delta_{99.9}$ minus the mean of 200 Haar spectrum-matched null replicates; $r$/200: replicates above "
  r"the real value, with its left-tail $p$; genuine: Benjamini--Hochberg-corrected $p\le0.05$ over the 72 cells; $^{\circ}$: not genuine; bold: less clustered than the null. "
  r"$\rho_{\text{WN}}$: Spearman correlation of inter-centroid and WordNet distances. "
  + sl_txt + js_txt
  + r"Other datasets and the four null$\times$statistic verdicts: Table~\ref{tab:q1-census}; task columns: Table~\ref{tab:q9-corollary}. % " + src + ", exp3_alignment.csv" + (", expR62_samplelevel_record.csv" if SLF else "") + (", expR66_joint_sensitivity_summary.csv" if JS else "") + "\n}",
  r"\label{tab:census}", r"\end{table}"]
open(HERE/"tab_census.tex", "w").write("\n".join(lines) + "\n")
# the final version (main_iclr2027_final.tex) inputs the same table with a caption of one takeaway and one sentence on what is shown
short = [r"\caption{\textbf{Clustered structure is the rule at the class level.} Per backbone: raw ImageNet reading $\hat\delta_{99.9}$, excess over the Haar null mean, rank $r$/200 with left-tail $p$, "
         r"genuine = BH $p\le0.05$ over 72 cells ($^{\circ}$: not genuine; bold: above the null), the CIFAR-100 and CIFAR-10 excess, and $\rho_{\text{WN}}$, the Spearman correlation with WordNet distances. % " + src + ", exp3_alignment.csv" + "\n}", r"\label{tab:census}", r"\end{table}"]
assert lines[-2] == r"\label{tab:census}" and lines[-1] == r"\end{table}"
open(HERE/"tab_census_final.tex", "w").write("\n".join(lines[:-3] + short) + "\n")
for r in rows: print(r[0], f"{r[1]} d={r[2]:.3f} exc={r[3]:+.3f} r={r[4]}/{N} p={r[5]:.3f} G={r[6]} c100={r[7]:+.3f} c10={r[9]:+.3f} rho={r[11]:+.2f}")
print("wrote tab_census.tex from", src)

# The columns that once formed tab_census_extra (supremum IN, DTD/FMNIST/MNIST excess, best-R, metric) are generated by gen_appendix_final.py (tab:q1-census, tab:q9-corollary).
