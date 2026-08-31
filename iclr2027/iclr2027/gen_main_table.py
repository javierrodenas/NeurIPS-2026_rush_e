#!/usr/bin/env python3
"""Main-text census table (Table 2): one row per vision backbone, every value read from the
released result files. Sources: exp20_null_ztable.csv (raw delta, excess, z), exp26_xi_nulls.csv
(curvature-sign excess), exp3_alignment.csv (WordNet alignment), exp28_recovery_per_config.csv
(CIFAR-100 superclass recovery under the selected cosine-complete configuration),
exp2_metric_controls.csv (best zero-cost metric vs Euclidean, few-shot, hierarchical datasets)."""
import csv, os, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
def load(f): return list(csv.DictReader(open(RES / f)))
NAME = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
        "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
        "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
ORDER = list(NAME)
HIER_TR = ("cifar100","cifar10","dtd"); HIER = ("imagenet",)+HIER_TR
d20 = {(r["model"],r["dataset"]): r for r in load("exp20_null_ztable.csv")}
d26 = {r["model"]: r for r in load("exp26_xi_nulls.csv")}
d3  = {r["model"]: r for r in load("exp3_alignment.csv")}
d28 = {r["model"]: r for r in load("exp28_recovery_per_config.csv")}
d34 = {r["model"]: r for r in load("expR34_p999_imagenet.csv")}
d2  = {(r["model"],r["dataset"]): r for r in load("exp2_metric_controls.csv")}
rows = []
for m in ORDER:
    r = d20[(m,"imagenet")]
    exc_tr = [float(d20[(m,ds)]["excess"]) for ds in HIER_TR]
    xi = float(d26[m]["excess"])
    rho = float(d3[m]["spearman_wn"])
    ari = float(d28[m]["cosine-complete"])
    if all((m,ds) in d2 for ds in HIER):
        fsH = [100*(float(d2[(m,ds)]["FS_H"])-float(d2[(m,ds)]["FS_R"])) for ds in HIER]
        fsC = [100*(float(d2[(m,ds)]["FS_COS"])-float(d2[(m,ds)]["FS_R"])) for ds in HIER]
        best = st.mean(max(h,c) for h,c in zip(fsH,fsC))
        gap = st.mean(fsH) - st.mean(fsC)
        metric = "H" if gap > 0.15 else ("cos" if gap < -0.15 else "either")
    else:
        best, metric = None, "---"   # backbone outside the 10-model task grid
    p9 = float(d34[m]["excess"]) if m in d34 else None
    rows.append((NAME[m], float(r["delta"]), float(r["excess"]), float(r["z"]), p9, exc_tr, xi, rho, ari, best, metric))
lines = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2pt}",
         r"\begin{tabular}{l c c c c c c c c c c c}", r"\toprule",
         r"model & $\delta_{\text{norm}}$ IN & exc.\ IN sup.\ ($z$) & p99.9 & exc.\ C100 & exc.\ C10 & exc.\ DTD & $\xi$-exc.\ IN & $\rho_{\text{WN}}$ & ARI$_{20}$ & best$-$R & metric \\",
         r"\midrule"]
for i,(n,dl,ex,z,p9,tr,xi,rho,ari,best,met) in enumerate(rows):
    if i in (4,9): lines.append(r"\midrule")
    exs = f"${ex:+.3f}$ (${z:+.1f}$)".replace("0.", ".")
    if ex > 0: exs = r"\textbf{" + exs + "}"
    bests = f"${best:+.2f}$" if best is not None else "---"
    trs = " & ".join(f"${v:+.3f}$".replace("0.", ".") for v in tr)
    p9s = f"${p9:+.3f}$".replace("0.", ".") if p9 is not None else "--"
    lines.append(f"{n} & ${dl:.3f}$ & {exs} & {p9s} & {trs} & ${xi:+.3f}$ & ${rho:+.2f}$ & ${ari:.2f}$ & {bests} & {met} \\\\".replace("$0.", "$.").replace("$-0.", "$-.").replace("$+0.", "$+."))
lines += [r"\bottomrule", r"\end{tabular}",
  r"\caption{Calibrated census of the 12 vision backbones (supervised / self-supervised / contrastive). "
  r"$\delta_{\text{norm}}$: raw ImageNet reading; excess IN: excess over the spectrum-matched null with $z$ (bold: sign-positive, at null); p99.9: ImageNet excess under the supremum-robust statistic (20 replicates, Table~\ref{tab:b18-p999}; centroid-store caveat there); exc.\ C100/C10/DTD: per-dataset excess on the transfer sets (means across datasets are not comparable, \S\ref{sec:form}); "
  r"$\xi$-excess: curvature-sign excess on ImageNet (negative: hyperbolic-leaning); $\rho_{\text{WN}}$: Spearman correlation with WordNet distances; ARI$_{20}$: recovery of the 20 CIFAR-100 superclasses (selected cosine-complete configuration); "
  r"best$-$R: few-shot advantage of the best zero-cost metric over Euclidean, mean over the four hierarchical datasets; metric: which of Poincar\'e (H) or cosine collects it (either: within $0.15$pp; ---: outside the 10-model task grid). Per-cell values in Appendix~\ref{app:tables}.}",
  r"\label{tab:census}", r"\end{table}"]
open(HERE/"tab_census.tex","w").write("\n".join(lines)+"\n")
for r in rows: print(r[0], f"d={r[1]:.3f} exc={r[2]:+.3f} z={r[3]:+.1f} p9={r[4]}", "tr=" + "/".join(f"{v:+.3f}" for v in r[5]), f"xi={r[6]:+.3f} rho={r[7]:+.2f} ari={r[8]:.2f} best={r[9]} {r[10]}")
print("wrote tab_census.tex")
