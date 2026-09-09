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
SL = {(r["model"], r["dataset"]): r for r in load("expR62_samplelevel_record.csv")} if (RES/"expR62_samplelevel_record.csv").exists() else {}   # sample-level reading (restructuring, R7)
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
sl_n = sum(1 for k, a in SL.items() if str(a["genuine_bh"]) == "True")
JS = load("expR66_joint_sensitivity_summary.csv") if (RES/"expR66_joint_sensitivity_summary.csv").exists() else []
js_txt = ""
if JS:
    a, b = sum(r["joint_genuine"] == "True" for r in JS), sum(r["boot_bh_genuine"] == "True" for r in JS)
    js_txt = f"With centroid-resampling and estimator noise folded into the null, or the census repeated on 30 resampled centroid sets, the genuine count is {min(a,b)}--{max(a,b)} of 72 (Table~\\ref{{tab:b38-joint}}). "
sl_txt = (f"Sample level: the same reading on $\\approx$1000 stratified training images per cell, BH over those 24 cells: genuine in {sl_n} of 24 (full table in Appendix~\\ref{{app:sample}}). " if SL else "")
lines += [r"\bottomrule", r"\end{tabular}",
  r"\caption{\textbf{Clustered structure is the rule at the class level; the sample-level reading sits within null noise in most cells.} "
  r"One row per backbone. $\hat\delta_{99.9}$: raw ImageNet reading; excess: $\hat\delta_{99.9}$ minus the mean of 200 Haar spectrum-matched null replicates; $r$/200: replicates above "
  r"the real value, with the left-tail $p=(1+\#\{\text{null}\le\text{real}\})/201$; genuine: Benjamini--Hochberg-corrected $p\le0.05$ over the 72 cells; $^{\circ}$: not genuine; bold: less clustered than the null. "
  r"$\rho_{\text{WN}}$: Spearman correlation of inter-centroid and WordNet distances. "
  + sl_txt + js_txt
  + r"Cross-dataset magnitudes are not comparable. The other datasets, the supremum reading and the task columns are in Table~\ref{tab:census-extra}; all four null$\times$statistic verdicts per cell in Table~\ref{tab:b21-2x2}. % " + src + ", exp3_alignment.csv" + (", expR62_samplelevel_record.csv" if SL else "") + (", expR66_joint_sensitivity_summary.csv" if JS else "") + "\n}",
  r"\label{tab:census}", r"\end{table}"]
open(HERE/"tab_census.tex", "w").write("\n".join(lines) + "\n")
for r in rows: print(r[0], f"{r[1]} d={r[2]:.3f} exc={r[3]:+.3f} r={r[4]}/{N} p={r[5]:.3f} G={r[6]} c100={r[7]:+.3f} c10={r[9]:+.3f} rho={r[11]:+.2f}")
print("wrote tab_census.tex from", src)

# ---- appendix: the columns moved out of Table 1 ----
_p9 = "expR54_census_haar_sup_200.csv" if (RES/"expR54_census_haar_sup_200.csv").exists() else "expR39c_census200_cache.csv"
d34 = {r["model"]: r for r in load(_p9) if r.get("dataset", "imagenet") == "imagenet"}
d2 = {(r["model"], r["dataset"]): r for r in load("exp2_metric_controls.csv")}
HIER = ("imagenet", "cifar100", "cifar10", "dtd")
lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{l c c c c c c}", r"\toprule",
         r"model & sup.\ IN ($r$) & exc.\ DTD & exc.\ FMNIST & exc.\ MNIST & best$-$R (pp) & metric \\",
         r"\midrule"]
for i, m in enumerate(ORDER):
    if i in (4, 9): lines.append(r"\midrule")
    p9 = f"${float(d34[m]['excess']):+.3f}$ ({int(d34[m]['r_above'])})" if m in d34 else "--"
    cells = [f"${float(cen[(m, ds)]['excess']):+.3f}" + ("" if genuine(cen[(m, ds)]) else r"^{\circ}") + "$" for ds in ("dtd", "fashionmnist", "mnist")]
    if all((m, ds) in d2 for ds in HIER):
        fsH = [100*(float(d2[(m,ds)]["FS_H"])-float(d2[(m,ds)]["FS_R"])) for ds in HIER]
        fsC = [100*(float(d2[(m,ds)]["FS_COS"])-float(d2[(m,ds)]["FS_R"])) for ds in HIER]
        best = st.mean(max(h, c) for h, c in zip(fsH, fsC))
        gap = st.mean(fsH) - st.mean(fsC)
        bests = f"${best:+.2f}$"; met = "H" if gap > 0.15 else ("cos" if gap < -0.15 else "either")
    else:
        bests, met = "---", "---"
    lines.append(f"{NAME[m]} & {p9} & " + " & ".join(cells) + f" & {bests} & {met} \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
  r"\caption{Census columns moved out of Table~\ref{tab:census}. sup.\ IN: ImageNet excess of the supremum statistic under the same Haar null "
  r"(200 replicates; $r$ = replicates above the real value; Table~\ref{tab:b21-2x2}); exc.\ DTD/FMNIST/MNIST: per-dataset record "
  r"excess on the remaining sets; best$-$R: few-shot advantage of the best zero-cost metric over Euclidean, "
  r"mean over the four hierarchical datasets; metric: which of Poincar\'e (H) or cosine collects it "
  r"(either: within $0.15$pp; ---: outside the 10-model task grid). % " + _p9 + ", exp2_metric_controls.csv, " + src + "\n}",
  r"\label{tab:census-extra}", r"\end{table}"]
(HERE/"appendix_tables"/"tab_census_extra.tex").write_text("\n".join(lines) + "\n")
print("wrote tab_census_extra.tex")
