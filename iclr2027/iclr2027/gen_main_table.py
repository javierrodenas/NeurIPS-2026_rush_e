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
src = "expR39c_census200_cache.csv"                      # R6: 200 replicates, census of record
if not (RES/src).exists():
    src = "expR39b_census20_cache.csv"; print(f"WARNING: expR39c not found; layout run from {src} (20 replicates)")
cen = {(r["model"], r["dataset"]): r for r in load(src)}
N = 200 if "r_above" in next(iter(cen.values())) else 20
def rank(r): return int(r["r_above"]) if "r_above" in r else round(20*float(r["frac_null_above"]))
def pleft(r): return float(r["p_left"]) if "p_left" in r else (1 + 20 - rank(r)) / 21
RMIN = 191 if N == 200 else 20
d3 = {r["model"]: r for r in load("exp3_alignment.csv")}
rows = []
for m in ORDER:
    r = cen[(m, "imagenet")]
    rows.append((NAME[m], fam(m), float(r["delta"]),
                 float(r["excess"]), rank(r), pleft(r),
                 float(cen[(m, "cifar100")]["excess"]), pleft(cen[(m, "cifar100")]),
                 float(cen[(m, "cifar10")]["excess"]), pleft(cen[(m, "cifar10")]),
                 float(d3[m]["spearman_wn"])))
lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.4pt}",
         r"\begin{tabular}{l l c c c c c c}", r"\toprule",
         r"model & family & $\delta_{\text{norm}}$ IN & excess IN & $r$/" + str(N) + r" ($p$) & exc.\ C100 & exc.\ C10 & $\rho_{\text{WN}}$ \\",
         r"\midrule"]
def pf(p): return f"{p:.3f}".lstrip("0") if p < 1 else "1"
for i, (n, f, dl, ex, rk, pv, e100, p100, e10, p10, rho) in enumerate(rows):
    if i in (4, 9): lines.append(r"\midrule")
    exs = f"${ex:+.3f}$"
    if ex > 0: exs = r"\textbf{" + exs + "}"
    rp = f"{rk} ({pf(pv)})"
    c100 = f"${e100:+.3f}" + (r"^{\circ}" if p100 > 0.05 else "") + "$"; c10 = f"${e10:+.3f}" + (r"^{\circ}" if p10 > 0.05 else "") + "$"
    if e100 > 0: c100 = r"\textbf{" + c100 + "}"
    if e10 > 0: c10 = r"\textbf{" + c10 + "}"
    lines.append(f"{n} & {f} & ${dl:.3f}$ & {exs} & {rp} & {c100} & {c10} & ${rho:+.2f}$ \\\\")
lines += [r"\bottomrule", r"\end{tabular}",
  r"\caption{\textbf{The calibrated census: genuine tree-like structure in nearly every vision cell.} "
  r"One row per backbone (supervised / self-supervised / contrastive blocks). $\delta_{\text{norm}}$: raw ImageNet reading; "
  r"excess: $\delta_{\text{norm}}$ minus the spectrum-matched null mean; $r$/" + str(N) + r": number of the " + str(N) + r" null replicates above the real value, "
  r"with the left-tail $p=(1+\#\{\text{null}\le\text{real}\})/" + str(N+1) + r"$; a cell is \emph{genuine} when $p\le0.05$ ($r\ge" + str(RMIN) + r"$); "
  r"bold: sign-positive, i.e.\ less tree-like than the null; $^{\circ}$ on a transfer-set excess: not genuine ($p>0.05$). "
  r"$\rho_{\text{WN}}$: Spearman correlation of inter-centroid and WordNet distances. "
  r"Cross-dataset magnitudes are not comparable (\S\ref{sec:form}). DTD, the flat datasets, the p99.9 robustness "
  r"statistic and the task columns are in Appendix~\ref{app:tables} (Table~\ref{tab:census-extra}). % " + src + ", exp3_alignment.csv" + "\n}",
  r"\label{tab:census}", r"\end{table}"]
open(HERE/"tab_census.tex", "w").write("\n".join(lines) + "\n")
for r in rows: print(r[0], f"{r[1]} d={r[2]:.3f} exc={r[3]:+.3f} r={r[4]}/{N} p={r[5]:.3f} c100={r[6]:+.3f} c10={r[8]:+.3f} rho={r[10]:+.2f}")
print("wrote tab_census.tex from", src)

# ---- appendix: the columns moved out of Table 1 ----
d34 = {r["model"]: r for r in load("expR34_p999_imagenet.csv")}
d2 = {(r["model"], r["dataset"]): r for r in load("exp2_metric_controls.csv")}
HIER = ("imagenet", "cifar100", "cifar10", "dtd")
lines = [r"\begin{table}[H]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{l c c c c c c}", r"\toprule",
         r"model & p99.9 IN & exc.\ DTD & exc.\ FMNIST & exc.\ MNIST & best$-$R (pp) & metric \\",
         r"\midrule"]
for i, m in enumerate(ORDER):
    if i in (4, 9): lines.append(r"\midrule")
    p9 = f"${float(d34[m]['excess']):+.3f}$" if m in d34 else "--"
    cells = [f"${float(cen[(m, ds)]['excess']):+.3f}$" for ds in ("dtd", "fashionmnist", "mnist")]
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
  r"\caption{Census columns moved out of Table~\ref{tab:census}. p99.9 IN: ImageNet excess under the "
  r"supremum-robust statistic (20 replicates; Table~\ref{tab:b18-p999}); exc.\ DTD/FMNIST/MNIST: per-dataset "
  r"excess on the remaining sets; best$-$R: few-shot advantage of the best zero-cost metric over Euclidean, "
  r"mean over the four hierarchical datasets; metric: which of Poincar\'e (H) or cosine collects it "
  r"(either: within $0.15$pp; ---: outside the 10-model task grid). % expR34_p999_imagenet.csv, exp2_metric_controls.csv, " + src + "\n}",
  r"\label{tab:census-extra}", r"\end{table}"]
(HERE/"appendix_tables"/"tab_census_extra.tex").write_text("\n".join(lines) + "\n")
print("wrote tab_census_extra.tex")
