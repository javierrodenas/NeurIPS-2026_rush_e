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
rows = []
for m in ORDER:
    r = cen[(m, "imagenet")]
    rows.append((NAME[m], fam(m), float(r["delta"]),
                 float(r["excess"]), rank(r), pleft(r), genuine(r),
                 float(cen[(m, "cifar100")]["excess"]), genuine(cen[(m, "cifar100")]),
                 float(cen[(m, "cifar10")]["excess"]), genuine(cen[(m, "cifar10")]),
                 float(d3[m]["spearman_wn"])))
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.4pt}",
         r"\begin{tabular}{l l c c c c c c}", r"\toprule",
         r"model & family & $\hat\delta_{99.9}$ IN & excess IN & $r$/200 ($p$) & exc.\ C100 & exc.\ C10 & $\rho_{\text{WN}}$ \\",
         r"\midrule"]
def pf(p): return f"{p:.3f}".lstrip("0") if p < 1 else "1"
for i, (n, f, dl, ex, rk, pv, g, e100, g100, e10, g10, rho) in enumerate(rows):
    if i in (4, 9): lines.append(r"\midrule")
    exs = f"${ex:+.3f}" + ("" if g else r"^{\circ}") + "$"
    if ex > 0: exs = r"\textbf{" + exs + "}"
    rp = f"{rk} ({pf(pv)})"
    c100 = f"${e100:+.3f}" + ("" if g100 else r"^{\circ}") + "$"; c10 = f"${e10:+.3f}" + ("" if g10 else r"^{\circ}") + "$"
    if e100 > 0: c100 = r"\textbf{" + c100 + "}"
    if e10 > 0: c10 = r"\textbf{" + c10 + "}"
    lines.append(f"{n} & {f} & ${dl:.3f}$ & {exs} & {rp} & {c100} & {c10} & ${rho:+.2f}$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
  r"\caption{\textbf{The census of record: beyond-null (clustered) structure in most cells and every family.} "
  r"One row per backbone (supervised / self-supervised / contrastive blocks). $\hat\delta_{99.9}$: raw ImageNet reading of the 99.9th-percentile "
  r"four-point statistic; excess: $\hat\delta_{99.9}$ minus the mean of 200 Haar-rotated spectrum-matched null replicates; $r$/200: replicates above "
  r"the real value, with the left-tail $p=(1+\#\{\text{null}\le\text{real}\})/201$; a cell is \emph{genuine} when its Benjamini--Hochberg-corrected "
  r"$p\le0.05$ over the 72 cells; $^{\circ}$: not genuine; bold: sign-positive, i.e.\ less clustered than the null. "
  r"$\rho_{\text{WN}}$: Spearman correlation of inter-centroid and WordNet distances. "
  r"Cross-dataset magnitudes are not comparable (\S\ref{sec:form}). DTD, the flat datasets, the supremum reading and the task columns are in "
  r"Appendix~\ref{app:tables} (Table~\ref{tab:census-extra}); all four null$\times$statistic verdicts per cell in Table~\ref{tab:b21-2x2}. % " + src + ", exp3_alignment.csv" + "\n}",
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
