#!/usr/bin/env python3
"""Consolidated appendix (final pass, brief 4.3): fourteen tables, one per question, generated from the same result files as the
forty-four tables they replace (snapshot in rebuttal/results/final_pass_old_appendix/). Every table is one float; a question whose
panels exceed a page continues under the same number (\\ContinuedFloat). Nothing is hand-typed except the model-panel parameter
counts (tab_a0 heritage) and the tree/H^2/sphere calibration rows, both re-verified by sweep_freeze.py. Each file carries a
'% prov:' line with its result files for the provenance index (gen_provenance.py). Run from anywhere; writes appendix_tables/tab_q*.tex
and checks that every decimal token of the old appendix survives (the same check runs in sweep_freeze.py)."""
import pandas as pd
import csv, os, re, json, statistics as st, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
ABL = Path(os.environ.get("PLATONIC_ABLATIONS", "/media/HDD_4TB_2/javi/Platonic/results"))
OUT = HERE / "appendix_tables"; OUT.mkdir(exist_ok=True)
import numpy.core as _nc; sys.modules.setdefault("numpy._core", _nc)
for _s in ("multiarray", "umath", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core." + _s, __import__("numpy.core." + _s, fromlist=["_"]))
    except Exception: pass

def load(f): return list(csv.DictReader(open(RES / f)))
def ex(f): return (RES / f).exists()
NAME = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B",
        "dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B",
        "gpt2":"GPT-2 S","gpt2_m":"GPT-2 M","gpt2_l":"GPT-2 L","gpt2_xl":"GPT-2 XL","pythia_410m":"Pythia-410M","pythia_1b":"Pythia-1B","pythia_2b8":"Pythia-2.8B",
        "olmo_1b":"OLMo-1B","olmo_7b":"OLMo-7B","bge_base":"BGE-base","bge_large":"BGE-large","gte_base":"GTE-base","gte_large":"GTE-large",
        "gte_qwen2":"GTE-Qwen2-1.5B","e5_base":"E5-base","e5_large":"E5-large","deit_b":"DeiT-B (IN-1k)","vit_b_in1k":"ViT-B (augreg, IN-1k)"}
M12 = ["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
M10 = [m for m in M12 if m not in ("dinov1_b","siglip_b")]
DS = ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]; DSH = {"imagenet":"IN","cifar100":"C100","cifar10":"C10","dtd":"DTD","fashionmnist":"FMNIST","mnist":"MNIST"}
DSL = {"imagenet":"ImageNet","cifar100":"CIFAR-100","cifar10":"CIFAR-10","dtd":"DTD","fashionmnist":"FMNIST","mnist":"MNIST"}
HIER = ["imagenet","cifar100","cifar10","dtd"]; TOP = {"imagenet","cifar100"}
def pfmt(p): return f"{p:.3f}".lstrip("0") if p < 1 else "1"
def bh(p):
    p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p); ranked = p[order]*n/np.arange(1, n+1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty(n); out[order] = np.minimum(adj, 1.0); return out
def gb(a): return str(a["genuine_bh"]) == "True"
def fam(m): return "ViT" if m.startswith("i21k") else ("DINOv2" if m.startswith("dinov2") else "CLIP")
def pear(x, y): return float(np.corrcoef(x, y)[0, 1])
def demean(vals, fams):
    mu = {f: st.mean(v for v, g in zip(vals, fams) if g == f) for f in set(fams)}; return [v - mu[g] for v, g in zip(vals, fams)]

class Table:
    """One appendix table: panels (title, colspec, header lines, rows, midrules) under one caption; parts continue the number."""
    def __init__(self, fname, label, size=r"\scriptsize", colsep="2.5pt"):
        self.fname, self.label, self.size, self.colsep = fname, label, size, colsep; self.parts = [[]]; self.prov = []
    def panel(self, title, colspec, header, rows, mids=(), size=None, colsep=None, note=None):
        L = []
        if title: L.append(r"\noindent\textbf{" + title + r"}\par\vspace{1.5pt}")
        if size: L.append(size)
        if colsep: L.append(r"\setlength{\tabcolsep}{" + colsep + "}")
        L += [r"\begin{tabular}{" + colspec + "}", r"\toprule"] + list(header) + [r"\midrule"]
        for i, r in enumerate(rows):
            if i in mids: L.append(r"\midrule")
            L.append(r)
        L += [r"\bottomrule", r"\end{tabular}"]
        if note: L.append(r"\par\vspace{1pt}{" + note + "}")
        L.append(r"\par\vspace{5pt}")
        if size or colsep: L = [r"\begingroup"] + L + [r"\endgroup"]
        self.parts[-1] += L
    def newpart(self): self.parts.append([])
    def write(self, caption, contd=None):
        L = ["% prov: " + ", ".join(dict.fromkeys(self.prov))]
        for i, P in enumerate(self.parts):
            L += [r"\begin{table}[H]" + (r"\ContinuedFloat" if i else ""), r"\centering", self.size, r"\setlength{\tabcolsep}{" + self.colsep + "}"]
            L += P
            if i == 0: L += [r"\caption{" + caption + "\n}", r"\label{" + self.label + "}"]     # a caption may end in a '% source' comment: the brace closes on its own line
            else: L += [r"\caption{(continued) " + (contd[i-1] if contd and i-1 < len(contd) else "") + "\n}"]
            L.append(r"\end{table}")
        (OUT / self.fname).write_text("\n".join(L) + "\n"); print("wrote", self.fname, f"({len(self.parts)} part(s))")

# ======================================================================================================================
# Q10: calibration on reference geometries and synthetic clusters (cited first, from S3)
# ======================================================================================================================
def q_calibration():
    T = Table("tab_q10_calibration.tex", "tab:q10-calibration", size=r"\small", colsep="4pt"); T.prov += ["exp1_delta_controls.csv", "expR42_star_calibration.csv"]
    g = {int(r["d"]): float(r["delta_max"]) for r in load("exp1_delta_controls.csv") if r["variant"] == "gauss"}; ds = sorted(g)
    rows = [r"balanced binary tree (depth 10) & 0.000 & 0.000 \\",
            r"$\mathbb{H}^2$ ($K{=}-1$), region radius $R{=}2/4/8/16$ & 0.65/0.69/0.69/0.69 & 0.162/0.087/0.043/0.022 \\",
            r"uniform $S^{99}$ (chord / geodesic) & 0.25/0.37 & 0.143/0.179 \\",
            "iid Gaussian ($n{=}1000$), $d{=}192,\\dots,1536$ & --- & " + "/".join(f"{g[d]:.3f}" for d in ds) + r" \\"]
    T.panel("(a) Reference geometries.", "lcc", [r"space & absolute $\delta$ & $\delta_{\text{norm}}$ \\"], rows)
    rows2 = []
    if ex("expR42_star_calibration.csv"):
        R = load("expR42_star_calibration.csv"); CN = {"star30_tight":"30-cluster star, tight (0.1)","star30_mid":"30-cluster star, mid (0.3)","star30_loose":"30-cluster star, loose (0.6)","hier6x5_mid":"2-level hierarchy $6\\times5$ (0.3)"}
        for cfg in CN:
            sub = [a for a in R if a["config"] == cfg]
            if not sub: continue
            m = lambda k: st.mean(float(a[k]) for a in sub)
            rows2.append(f"{CN[cfg]} & ${m('delta'):.3f}$ & ${m('excessA_gauss'):+.3f}$ & ${m('excessA_haar'):+.3f}$ & ${m('excessB_gauss'):+.3f}$ & ${m('excessB_haar'):+.3f}$ \\\\")
        T.panel("(b) Synthetic clusters: what each null certifies.", "lc|cc|cc",
                [r"& & \multicolumn{2}{c|}{spectrum null A} & \multicolumn{2}{c}{hub-randomizing null B} \\", r"synthetic cloud ($n{=}1000$, $d{=}768$) & $\hat\delta$ & Gaussian & Haar & Gaussian & Haar \\"], rows2)
    T.write(r"\textbf{A low raw $\delta$ is not evidence of hierarchy, and the spectrum excess certifies clustering, not depth.} "
            r"(a) Calibration on reference geometries: the estimator recovers the tree zero and the four-point constant of $\mathbb{H}^2$, spheres stay high, and iid Gaussians drift low as $d$ grows, the dimension confound; the ordering trees $<$ hyperbolic $<$ spherical holds for hyperbolic regions of radius $R\ge4$. Gaussian row: the sampled supremum from the released control file, the statistic the curvature rule uses; tree/$\mathbb{H}^2$/sphere rows re-verified synthetically at every freeze. "
            r"(b) Excess of synthetic clouds (means over seeds; within/between noise ratio in parentheses) under the spectrum null A, with Gaussian coefficients and with Haar-rotated coefficients (exact sample spectrum), and under the hub-randomizing null B, which keeps every cluster and resamples the hub configuration. A pure star sits far below null A under either construction, so null A certifies clustering, not depth; null B also reports the star as hierarchical, since a near-regular simplex of hubs already minimizes $\delta$, which is why the depth test of Table~\ref{tab:q4-depth} is read against a matched star rather than against null B. Under the Haar construction the two-level hierarchy sits $2$--$3\times$ below the star; under the plain Gaussian null the mid-radius star scores $-0.104$ and the $6{\times}5$ hierarchy $-0.14$, a fraction of either's excess. % exp1_delta_controls.csv, expR42_star_calibration.csv, sweep_freeze.py")

# ======================================================================================================================
# Q1: the vision census, four constructions, verdict per cell, cosine reading, extra backbones
# ======================================================================================================================
def q_census():
    T = Table("final/tab_q01_census_final.tex" if FINAL else "tab_q01_census.tex", "tab:q1-census", colsep="1.4pt"); T.prov += (["expR75_census_centered_haar.csv"] if FINAL else []) + ["expR52_census_haar_p999_200.csv", "expR54_census_haar_sup_200.csv", "expR40b_p999census200.csv", "expR39c_census200_cache.csv", "expR57_census_cosine_haar_p999_200.csv", "expR57_text_cosine_haar_p999_200.csv", "exp10_local_vs_global.csv", "expR45_convnet_rows.csv", "exp20_null_ztable.csv"]
    rec = load("expR75_census_centered_haar.csv" if FINAL else "expR52_census_haar_p999_200.csv"); by = {(a["model"], a["dataset"]): a for a in rec}   # final: the record is the centered Haar null (author's decision, 2026-09-20)
    r20 = {(a["model"], a["dataset"]): a for a in load("exp20_null_ztable.csv")}
    comp = sum(1 for a in rec if (a["model"], a["dataset"]) in r20); agree = sum((float(a["excess"]) < 0) == (float(r20[(a["model"], a["dataset"])]["excess"]) < 0) for a in rec if (a["model"], a["dataset"]) in r20)
    gen = sum(gb(a) for a in rec); genh = sum(gb(a) for a in rec if a["dataset"] in TOP)
    rows = []
    for m in M12:
        cs = []
        for d in DS:
            a = by[(m, d)]; cs += [f"${float(a['excess']):+.3f}" + ("" if gb(a) else r"^{\circ}") + "$", f"{int(a['r_above'])}", pfmt(float(a["p_left"]))]
        rows.append(NAME[m] + " & " + " & ".join(cs) + r" \\")
    T.panel("(a) The census of record per cell: Haar spectrum-matched null" + (" with $Q$ orthogonal to the all-ones vector" if FINAL else "") + ", 99.9th-percentile statistic, 200 replicates.", "l" + "ccc"*6,
            [" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DSH[d]}}}" for d in DS) + r" \\", r"model & " + " & ".join([r"exc.\ & $r$ & $p$"]*6) + r" \\"], rows, mids=(4, 9))
    # (b) four verdicts + normalized excess + supremum column
    four = ([("Hc", "expR75_census_centered_haar.csv")] if FINAL else []) + [("Hp", "expR52_census_haar_p999_200.csv"), ("Hs", "expR54_census_haar_sup_200.csv"), ("Gp", "expR40b_p999census200.csv"), ("Gs", "expR39c_census200_cache.csv")]
    V = {}
    for tag, f in four:
        rr = load(f); pb = bh([float(a["p_left"]) for a in rr]); V[tag] = {(a["model"], a["dataset"]): (float(a["excess"]), pb[i] <= 0.05, int(a["r_above"])) for i, a in enumerate(rr)}
    counts = {tag: sum(v[1] for v in V[tag].values()) for tag, _ in four}
    fr = {k: float(a["excess"]) / float(a["null_mean"]) for k, a in by.items()}
    rows = []
    for m in M12:
        cs = ["".join(("$\\bullet$" if V[tag][(m, d)][1] else "$\\circ$") for tag, _ in four) + f" ${100*fr[(m,d)]:+.0f}\\%$" for d in DS]
        sup = V["Hs"][(m, "imagenet")]; cs.append(f"${sup[0]:+.3f}$ ({sup[2]})")
        rows.append(NAME[m] + " & " + " & ".join(cs) + r" \\")
    im_ = [fr[k] for k in fr if k[1] == "imagenet"]
    T.panel("(b) Verdict under the " + ("five" if FINAL else "four") + " null$\\times$statistic constructions and the excess as a fraction of the null reading.", "l" + "c"*6 + "|c",
            ["model & " + " & ".join(DSH[d] for d in DS) + r" & sup.\ IN exc.\ ($r$) \\"], rows, mids=(4, 9), colsep="3.5pt")
    T.newpart()
    # (c) cosine census
    cv = load("expR57_census_cosine_haar_p999_200.csv"); cby = {(a["model"], a["dataset"]): a for a in cv}
    cagree = sum(gb(cby[k]) == gb(by[k]) for k in cby if k in by); cgen = sum(gb(a) for a in cv); ctop = sum(gb(a) for a in cv if a["dataset"] in TOP); cneg = sum(float(a["excess"]) < 0 for a in cv)
    rows = []
    for m in M12:
        cs = []
        for d in DS:
            a = cby[(m, d)]; cs += [f"${float(a['excess']):+.3f}" + ("" if gb(a) else r"^{\circ}") + "$", f"{int(a['r_above'])}"]
        rows.append(NAME[m] + " & " + " & ".join(cs) + r" \\")
    T.panel("(c) The same census on cosine geometry (L2-normalized centroids, geodesic distances, Haar null on the normalized cloud): robustness, not the record.", "l" + "cc"*6,
            [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DSH[d]}}}" for d in DS) + r" \\", r"model & " + " & ".join([r"exc.\ & $r$"]*6) + r" \\"], rows, mids=(4, 9), colsep="2.4pt")
    txt = ""
    if ex("expR57_text_cosine_haar_p999_200.csv"):
        ct = load("expR57_text_cosine_haar_p999_200.csv"); txt = f" Text under the same cosine protocol (padding-free embeddings): genuine {sum(gb(a) for a in ct)}/15: " + ", ".join(a["model"].replace("_", "-") for a in ct if gb(a)) + "."
    trip = ""
    if ex("exp10_local_vs_global.csv"):
        e10 = {a["model"]: a for a in load("exp10_local_vs_global.csv")}
        trip = f" The angular tree: sibling-triplet agreement on CIFAR-100 for DINOv2-L is {float(e10['dinov2_l']['c100_sibtrip_c']):.2f} under cosine against {float(e10['dinov2_l']['c100_sibtrip_e']):.2f} under Euclidean distance."
    # (d) extra backbones
    xtxt = ""
    if ex("expR45_convnet_rows.csv"):
        XR = load("expR45_convnet_rows.csv"); XN = {"barlow_r50":"Barlow Twins (ResNet-50)","byol_r50":"BYOL (ResNet-50)","mae_b":"MAE-B (ViT)","ijepa_h":"I-JEPA-H (ViT)"}
        rows = [f"{XN.get(a['model'], a['model'])} & {a['d']} & ${float(a['delta']):.3f}$ & ${float(a['excessA']):+.3f}$ (${float(a['zA']):+.1f}$) & {round(20*float(a['rankA']))}/20 \\\\" for a in XR]
        T.panel("(d) Backbones outside the ViT panel on the ImageNet centroid store (spectrum null, 20 replicates).", "lcccc", [r"model & $d$ & $\hat\delta$ & excess ($z$) & reps above \\"], rows, size=r"\footnotesize", colsep="4pt")
        xtxt = " (d) Two self-supervised ResNet-50 backbones read the same clustered structure as the ViTs."
    T.write(r"\textbf{Clustered structure is the rule at the class level under every construction of the null and statistic, and the verdict does not depend on the metric.} "
            r"(a) Per cell: excess of $\hat\delta_{99.9}$ over the null mean, $r$ = replicates above the real value, $p=(1+\#\{\text{null}\le\text{real}\})/201$; genuine = Benjamini--Hochberg-corrected $p\le0.05$ over the 72 cells ($^{\circ}$: not genuine). "
            + f"Genuine cells: {gen}/72 overall, {genh}/24 on ImageNet+CIFAR-100; sign agreement with the original 5-replicate Gaussian$\\times$supremum census on the {comp} comparable cells: {agree}/{comp}. "
            + ((r"(b) Five symbols per cell in the order centered Haar$\times$p99.9 (the record, $Q\perp\mathbf{1}$), uncentered Haar$\times$p99.9, Haar$\times$supremum, Gaussian$\times$p99.9, Gaussian$\times$supremum, $\bullet$ = genuine under that construction; "
            + f"genuine counts {counts['Hc']}/72, {counts['Hp']}/72, {counts['Hs']}/72, {counts['Gp']}/72, {counts['Gs']}/72. The percentage is the record excess divided by the mean null reading (ImageNet {100*min(im_):+.0f}\\% to {100*max(im_):+.0f}\\%; all cells {100*min(fr.values()):+.0f}\\% to {100*max(fr.values()):+.0f}\\%); the last column is the ImageNet excess of the supremum statistic under the same Haar null with its rank. "
            r"The centered Haar construction reproduces the centered spectrum exactly, the uncentered one differs by a term of order $1/\sqrt{n}$ that moves the verdict only on the ten-class datasets; the Gaussian one does not at small $n$ and inflates excesses for low-rank clouds (Table~\ref{tab:q8-robust}). The supremum is decided by a few extreme quadruples and disagrees with the 99.9th percentile in both directions on DINOv2-S/B/G ImageNet. ") if FINAL else (r"(b) Four symbols per cell in the order Haar$\times$p99.9 (the record), Haar$\times$supremum, Gaussian$\times$p99.9, Gaussian$\times$supremum, $\bullet$ = genuine under that construction; "
            + f"genuine counts {counts['Hp']}/72, {counts['Hs']}/72, {counts['Gp']}/72, {counts['Gs']}/72. The percentage is the record excess divided by the mean null reading (ImageNet {100*min(im_):+.0f}\\% to {100*max(im_):+.0f}\\%; all cells {100*min(fr.values()):+.0f}\\% to {100*max(fr.values()):+.0f}\\%); the last column is the ImageNet excess of the supremum statistic under the same Haar null with its rank. "
            r"The Haar construction reproduces the sample spectrum exactly; the Gaussian one does not at small $n$ and inflates excesses for low-rank clouds (Table~\ref{tab:q8-robust}). The supremum is decided by a few extreme quadruples and disagrees with the 99.9th percentile in both directions on DINOv2-S/B/G ImageNet. "))
            + r" % expR52_census_haar_p999_200.csv, expR54_census_haar_sup_200.csv, expR40b_p999census200.csv, expR39c_census200_cache.csv",
            contd=[f"(c) Cosine reading, BH over 72 cells: sign-negative {cneg}/72, genuine {cgen}/72 and {ctop}/24 on ImageNet+CIFAR-100, verdict agreement with the Euclidean record {cagree}/72 cells." + txt + trip + xtxt + r" % expR57_census_cosine_haar_p999_200.csv, expR57_text_cosine_haar_p999_200.csv, exp10_local_vs_global.csv, expR45_convnet_rows.csv"])

# ======================================================================================================================
# Q8: robustness — budgets, null constructions, bootstrap, joint sensitivity, class count
# ======================================================================================================================
def q_robust():
    T = Table("tab_q08_robust.tex", "tab:q8-robust", colsep="2.4pt"); T.prov += ["expR72_budget_record.csv", "expR72_budget_record_summary.csv", "expR31_quad_sweep.csv", "expR32_centroid_bootstrap.csv", "expR46_null_variants.csv", "expR59_imagenet_bootstrap_summary.csv", "expR66_joint_sensitivity_summary.csv", "expR60_c_sweep_record.csv"]
    NM = {"i21k_l":"ViT-L","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B"}; DSS = {"imagenet":"IN","cifar100":"C100","dtd":"DTD"}
    CELLS = [("i21k_l","imagenet"),("dinov2_l","imagenet"),("clip_b","imagenet"),("i21k_l","cifar100"),("dinov2_l","cifar100"),("clip_b","cifar100"),("i21k_l","dtd"),("dinov2_l","dtd"),("clip_b","dtd")]
    B72 = {(a["model"], a["dataset"]): a for a in load("expR72_budget_record_summary.csv")} if ex("expR72_budget_record_summary.csv") else {}
    D72 = load("expR72_budget_record.csv") if ex("expR72_budget_record.csv") else []
    r31 = load("expR31_quad_sweep.csv"); r32 = load("expR32_centroid_bootstrap.csv")
    BUD = [10000, 50000, 100000, 500000, 1000000, 2000000]
    rows = []
    for (m, ds) in CELLS:
        rec_ = [a for a in D72 if a["model"] == m and a["dataset"] == ds]
        if rec_:
            v = {int(a["n_quads"]): float(a["excess"]) for a in rec_}; s = B72[(m, ds)]
            rows.append(f"{NM[m]} & {DSS[ds]} & record & " + " & ".join(f"${v[b]:+.4f}$" for b in BUD) + f" & ${float(s['drift_ge1e5']):.4f}$ & {float(s['drift_ge1e5_over_sd']):.2f} \\\\")
        mx = [a for a in r31 if a["model"] == m and a["dataset"] == ds and a["stat"] == "max"]
        p9 = [a for a in r31 if a["model"] == m and a["dataset"] == ds and a["stat"] == "p999" and a["n_quads"] == "500000"]
        if mx:
            rows.append(f" & & supremum, Gaussian & " + " & ".join(f"${float(a['excess']):+.3f}$" for a in sorted(mx, key=lambda a: int(a["n_quads"]))) + f" & \\multicolumn{{2}}{{l}}{{p99.9 @$5{{\\times}}10^5$: " + (f"${float(p9[0]['excess']):+.3f}$" if p9 else "--") + r"} \\")
    seen = set()
    for a in r32:
        k = (a["model"], a["dataset"])
        if k in seen or int(a["b"]) < 0: continue
        sub = [float(x["excess"]) for x in r32 if (x["model"], x["dataset"]) == k and int(x["b"]) >= 0]
        if len(sub) >= 10:
            seen.add(k); rows.append(f"{NM[a['model']]} & {DSS[a['dataset']]} & \\multicolumn{{9}}{{l}}{{centroid bootstrap: excess ${st.mean(sub):+.4f}$, s.d.\\ ${st.pstdev(sub):.4f}$ ({len(sub)} resamples of the per-class images, {a['n_per_class']} images/class)}} \\\\")
    T.panel("(a) Quadruple budget: the excess at $10^4$ to $2{\\times}10^6$ sampled quadruples per seed, under the record (Haar null, p99.9, 200 replicates) and under the superseded supremum$\\times$Gaussian protocol.", "llccccccc|cc",
            [r"model & data & protocol & $10^4$ & $5{\times}10^4$ & $10^5$ & $5{\times}10^5$ & $10^6$ & $2{\times}10^6$ & drift ($\ge10^5$) & drift/s.d. \\"], rows, colsep="2.2pt")
    # (b) ImageNet bootstrap
    bmax = ""
    if ex("expR59_imagenet_bootstrap_summary.csv"):
        B = load("expR59_imagenet_bootstrap_summary.csv")
        rows = [f"{NAME.get(a['model'], a['model'])} & ${float(a['excess_ref']):+.4f}$ & ${float(a['excess_boot_mean']):+.4f}$ & ${float(a['excess_boot_sd']):.4f}$ & {float(a['frac_boot_negative']):.2f} \\\\" for a in B]
        T.panel("(b) ImageNet centroid bootstrap under the record: 30 resamples of the 100 training images per class (20 null replicates per resample).", "lcccc", [r"model & excess (reference) & bootstrap mean & bootstrap s.d. & fraction negative \\"], rows, mids=(4, 9), size=r"\footnotesize", colsep="4pt")
        bmax = f"{max(float(a['excess_boot_sd']) for a in B):.4f}"
    T.newpart()
    # (c) null variants
    if ex("expR46_null_variants.csv"):
        R = load("expR46_null_variants.csv"); DSv = [d for d in ["imagenet","cifar100","cifar10"] if any(a["dataset"] == d for a in R)]; by = {(a["model"], a["dataset"]): a for a in R}
        rows = []
        for m in M12:
            cs = []
            for d in DSv:
                a = by.get((m, d)); cs += ["--","--","--"] if a is None else [f"${float(a[k]):+.3f}$" for k in ("excess_gauss","excess_haar","excess_pcperm")]
            rows.append(NAME[m] + " & " + " & ".join(cs) + r" \\")
        T.panel("(c) Three constructions of the spectrum-matched null (10 replicates each): Gaussian coefficients, Haar-rotated coefficients (exact sample spectrum), PC-permutation (exact per-component marginals).", "l" + "|ccc"*len(DSv),
                [" & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DSH[d]}}}" for d in DSv) + r" \\", "model & " + " & ".join(["Gauss. & Haar & PC-perm."]*len(DSv)) + r" \\"], rows, mids=(4, 9), colsep="3pt")
    # (d) joint sensitivity
    jt = ""
    if ex("expR66_joint_sensitivity_summary.csv"):
        J = {(r["model"], r["dataset"]): r for r in load("expR66_joint_sensitivity_summary.csv")}
        rows = []
        for m in M12:
            cs = []
            for d in DS:
                r = J[(m, d)]; g = r["genuine_bh_record"] == "True"; jg = r["joint_genuine"] == "True"; bg = r["boot_bh_genuine"] == "True"
                cs.append(f"${float(r['z_joint']):+.1f}$ ({r['n_boot_genuine']})" + ("" if g else r"$^{\circ}$") + (r"$^{\dagger}$" if (g and not (jg and bg)) else "") + (r"$^{\ddagger}$" if (not g and (jg or bg)) else ""))
            rows.append(NAME[m] + " & " + " & ".join(cs) + r" \\")
        T.panel("(d) Joint sensitivity of the genuine count: $z_{\\text{joint}}$ against null, bootstrap and estimator noise, and (in parentheses) the number of the 30 centroid resamples in which the cell is genuine.", "l" + "c"*6, ["model & " + " & ".join(DSH[d] for d in DS) + r" \\"], rows, mids=(4, 9), colsep="2.6pt")
        tot = lambda k: sum(1 for r in J.values() if r[k] == "True"); tot2 = lambda k: sum(1 for (m, d), r in J.items() if r[k] == "True" and d in TOP)
        jt = f" Counts: record {tot('genuine_bh_record')}/72 ({tot2('genuine_bh_record')}/24); $z_{{\\text{{joint}}}}\\le-2$: {tot('joint_genuine')}/72 ({tot2('joint_genuine')}/24); genuine in $\\ge27$ of 30 resamples: {tot('boot_bh_genuine')}/72 ({tot2('boot_bh_genuine')}/24)."
    T.newpart()
    # (e) class-count sweep
    if ex("expR60_c_sweep_record.csv"):
        c60 = load("expR60_c_sweep_record.csv")
        def agg(m, C, mode, f):
            v = [float(r[f]) for r in c60 if r["model"] == m and int(r["C"]) == C and r["mode"] == mode and r[f] not in ("", "nan")]; return st.mean(v) if v else None
        def below(m, C, mode):
            v = [float(r["p_left"]) <= 0.05 for r in c60 if r["model"] == m and int(r["C"]) == C and r["mode"] == mode]; return f"{sum(v)}/{len(v)}" if v else "---"
        rows = []; mids = []
        for m in ["dinov2_l","dinov2_g","clip_l"]:
            for C in [10,20,50,100,200,500,1000]:
                dr, er, nr = agg(m,C,"random","delta_999"), agg(m,C,"random","excess"), agg(m,C,"random","nc_adv_pp")
                dc, ec, nc = agg(m,C,"coherent","delta_999"), agg(m,C,"coherent","excess"), agg(m,C,"coherent","nc_adv_pp")
                if dr is None: continue
                right = f"{dc:.3f} & {ec:+.3f} & {below(m,C,'coherent')} & {nc:+.2f}" if dc is not None else "--- & --- & --- & ---"
                rows.append(f"{NAME[m]} & {C} & {dr:.3f} & {er:+.3f} & {below(m,C,'random')} & {nr:+.2f} & {right} \\\\")
            mids.append(len(rows))
        T.panel("(e) Class count against hierarchy depth: ImageNet subsets of $C$ classes, random (spanning the hierarchy) or WordNet-coherent (siblings, effectively flat), under the record.", "lr|cccc|cccc",
                [r" & & \multicolumn{4}{c|}{random subsets (span the hierarchy)} & \multicolumn{4}{c}{WordNet-coherent (siblings)} \\", r"model & $C$ & $\hat\delta_{99.9}$ & excess & below & NC adv & $\hat\delta_{99.9}$ & excess & below & NC adv \\"], rows, mids=tuple(mids[:-1]), size=r"\footnotesize", colsep="3.5pt")
    a5 = ""
    if B72:
        di = max(float(B72[k]["drift_ge1e5_over_sd"]) for k in B72 if k[1] == "imagenet"); dc_ = max(float(B72[k]["drift_ge1e5_over_sd"]) for k in B72 if k[1] == "cifar100"); dd = max(float(B72[k]["drift_ge1e5_over_sd"]) for k in B72 if k[1] == "dtd")
        a5 = f" Under the record every sign holds at every budget and the drift of the excess from $10^5$ upward is at most {di:.2f} of the excess's own spread (null and estimator s.d.\\ combined at $5{{\\times}}10^5$) on ImageNet, {dc_:.2f} on CIFAR-100 and {dd:.2f} on DTD, so the excess is budget-stable; the supremum rows show the drift of up to $0.02$ on ImageNet that the superseded protocol carried."
    T.write(r"\textbf{The excess is budget-stable, its size is partly a property of the null construction, and the genuine count survives resampling and estimator noise.} "
            r"(a) Excess against the quadruple budget for 9 cells; drift = range of the record excess over budgets $\ge10^5$, divided by the excess's spread in the last column." + a5
            + r" (b) Bootstrap of the ImageNet excess over resampled images per class: the bootstrap s.d.\ is at most " + bmax + r" and every resample of every backbone is sign-negative."
            + r" % expR72_budget_record.csv, expR31_quad_sweep.csv, expR32_centroid_bootstrap.csv, expR59_imagenet_bootstrap_summary.csv",
            contd=[r"(c) Excess under three constructions of the spectrum null: Gaussian and PC-permutation agree, the exact-sample-spectrum construction yields smaller excesses for low-effective-rank clouds on the small-$C$ sets (most visibly DINOv2 on CIFAR-100), while signs and the ImageNet readings are stable. "
                   r"(d) Per cell: $z_{\text{joint}} = \text{excess}/\sqrt{\sigma_{\text{null}}^2+\sigma_{\text{boot}}^2+\sigma_{\text{est}}^2}$ (200-replicate null s.d., bootstrap s.d.\ over 30 centroid resamples, quadruple-seed s.d.) and the number of resamples in which the cell is genuine under BH over the 72 cells (50 Haar replicates per resample); $^{\circ}$: not genuine in the record; $^{\dagger}$: record-genuine but failing one of the two criteria ($z_{\text{joint}}\le-2$, genuine in $\ge27$ of 30); $^{\ddagger}$: not record-genuine but passing one." + jt
                   + r" % expR46_null_variants.csv, expR66_joint_sensitivity_summary.csv",
                   r"(e) Class-count control under the record (means over subset seeds: 5 for $C\le200$, 2 for $C{=}500$, 1 for $C{=}1000$); ``below'' = subset seeds whose real value lies below the null at uncorrected $p\le0.05$; NC adv = prototype-classifier advantage of the Poincar\'{e} readout in pp. % expR60_c_sweep_record.csv"])

# ======================================================================================================================
# Q3: the sample-level census
# ======================================================================================================================
def q_sample():
    T = Table("final/tab_q03_sample_final.tex" if FINAL else "tab_q03_sample.tex", "tab:q3-sample", colsep="2.4pt"); T.prov += ["expR62_samplelevel_record.csv"]
    R = load("expR62_samplelevel_record.csv"); by = {(a["model"], a["dataset"]): a for a in R}
    rows = []
    for m in M12:
        cs = []
        for ds in ("cifar100", "dtd"):
            a = by[(m, ds)]; g = gb(a)
            cs += [f"${float(a['delta_sup']):.3f}$", f"${float(a['delta_999']):.3f}$", f"${float(a['excess']):+.4f}" + ("" if g else r"^{\circ}") + "$", f"${100*float(a['excess'])/float(a['null_mean']):+.0f}\\%$", f"{int(a['r_above'])} ({float(a['p_left']):.3f})"]
        rows.append(NAME[m] + " & " + " & ".join(cs) + r" \\")
    T.panel("(a) The census cells at the sample level." if FINAL else "", "lccccc|ccccc", [r"& \multicolumn{5}{c|}{CIFAR-100 (10 images/class, $n{=}1000$)} & \multicolumn{5}{c}{DTD (22 images/class, $n{=}1034$)} \\",
            r"model & sup.\ $\hat\delta$ & $\hat\delta_{99.9}$ & excess & exc./null & $r/200$ ($p$) & sup.\ $\hat\delta$ & $\hat\delta_{99.9}$ & excess & exc./null & $r/200$ ($p$) \\"], rows, mids=(4, 9))
    if FINAL and ex("expR78_khrulkov_replication_summary.csv"):   # priority 2 (brief of 2026-09-21): a published reading reproduced and calibrated
        S78 = pd.read_csv(RES / "expR78_khrulkov_replication_summary.csv").set_index("dataset"); T.prov += ["expR78_khrulkov_replication.csv", "expR78_khrulkov_replication_summary.csv"]
        DN78 = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cub": "CUB-200", "miniimagenet": "MiniImageNet"}; order78 = [d for d in DN78 if d in S78.index]
        rows78 = [f"{DN78[d]} & {S78.loc[d].theirs:.2f} & {S78.loc[d].ours_raw_mean:.3f} $\\pm$ {S78.loc[d].ours_raw_sd:.3f} & ${S78.loc[d].excess_mean:+.4f}$ $\\pm$ {S78.loc[d].excess_sd:.4f} & {S78.loc[d].r_above_mean:.0f}/200 & {S78.loc[d].p_left_max:.3f} \\\\" for d in order78]
        allin = len(order78) == 4 and bool(S78.loc[order78].within_range.all())
        T.panel("(b) A published reading reproduced and calibrated: the setting of Khrulkov et al. (2020), Table 1, ResNet-34 row.", "lccccc", [r"dataset & their $\delta_{\text{rel}}$ & our $\delta_{\text{rel}}$ & excess & rank $r$ & largest $p$ \\"], rows78, size=r"\footnotesize", colsep="4pt",
                note=r"(b) Penultimate features of an ImageNet-pretrained ResNet-34 (torchvision, no substitution) on class-balanced batches of 1500 points; their estimator (exact $\delta$ on the batch, $\delta_{\text{rel}}=2\delta/\text{diam}$; mean $\pm$ s.d.\ over 10 batches) next to their published value, and on the same clouds the record instrument: excess over the centered Haar null (200 replicates, 99.9th-percentile statistic), the mean rank of the reading among the replicates and the largest left-tail $p$ over the 10 batches. "
                     + ("Every reproduced raw value is within 0.03 of the published one." if allin else "Not every reproduced raw value is within 0.03 of the published one; no claim is drawn."))
    n_gen = sum(gb(a) for a in R); DSL2 = {"cifar100":"CIFAR-100","dtd":"DTD"}
    gen_names = "; ".join(", ".join(NAME[a["model"]] for a in R if a["dataset"] == ds and gb(a)) + f" on {DSL2[ds]}" for ds in ("cifar100","dtd") if any(a["dataset"] == ds and gb(a) for a in R))
    sup_lo, sup_hi = min(float(a["delta_sup"]) for a in R), max(float(a["delta_sup"]) for a in R)
    exc_gen = min(float(a["excess"]) for a in R if gb(a)); fr = [float(a["excess"])/float(a["null_mean"]) for a in R if gb(a)]
    T.write(r"\textbf{The sample-level reading, the object of prior latent-hyperbolicity claims, sits within null noise in most cells.} "
            r"The instrument on $\approx$1000 stratified training images per cell (per-image features, not centroids): sup.\ $\hat\delta$ is the raw supremum $\delta_{\text{norm}}$ the literature reports; $\hat\delta_{99.9}$, excess, exc./null, $r$/200 and $p$ follow Table~\ref{tab:census} (Haar spectrum-matched null on the sample cloud, 99.9th-percentile statistic, 200 replicates); $^{\circ}$: not genuine under Benjamini--Hochberg over the 24 cells. "
            + f"The raw supremum lies in the band {sup_lo:.3f}--{sup_hi:.3f}; the excess is not genuine in {len(R)-n_gen} of {len(R)} cells and genuine in {n_gen} ({gen_names}), where it reaches at most ${exc_gen:+.3f}$, that is {100*min(fr):+.0f}\\% to {100*max(fr):+.0f}\\% of the null reading." + r" % expR62_samplelevel_record.csv")

# ======================================================================================================================
# Q4: the depth test — real backbones (both stars, both frames, K sweep), leaf-label ViTs, MERU, the trained control
# ======================================================================================================================
FINAL = False   # set by the final block below: q_depth/q_wordnet then write appendix_tables/final/*_final.tex
def q_depth():
    T = Table("final/tab_q04_depth_final.tex" if FINAL else "tab_q04_depth.tex", "tab:q4-depth", colsep="2.2pt"); zf = "+.2f" if FINAL else "+.1f"; T.prov += ["expR56_depth_variants.csv", "expR69_depth_haarhubs_summary.csv", "expR64b_wn30bal_summary.csv", "expR70_inet1k_supervised.csv", "expR52_census_haar_p999_200.csv", "expR63_meru_record.csv", "expR65_hier_finetune.csv"]
    dv = load("expR56_depth_variants.csv"); g2 = {(a["model"], a["dataset"], int(a["K"]), a["variant"]): a for a in dv}
    H = {a["model"]: a for a in load("expR69_depth_haarhubs_summary.csv")} if ex("expR69_depth_haarhubs_summary.csv") else {}
    BAL = {a["model"]: a for a in load("expR64b_wn30bal_summary.csv")} if ex("expR64b_wn30bal_summary.csv") else {}
    def cell(a): return "--" if a is None else f"${float(a['depth_excess']):+.3f}$ ({float(a['z_depth']):{zf}})"
    rows = []
    for m in M12:
        if FINAL:
            zonly = lambda a: "--" if a is None else f"${float(a['z_depth']):{zf}}$"
            cs = [zonly(g2.get((m,"cifar100",20,"iso"))), cell(g2.get((m,"cifar100",20,"aniso"))), zonly(g2.get((m,"imagenet",30,"iso"))), zonly(g2.get((m,"imagenet",10,"aniso"))), cell(g2.get((m,"imagenet",30,"aniso"))), zonly(g2.get((m,"imagenet",60,"aniso")))]
        else: cs = [cell(g2.get(k)) for k in [(m,"cifar100",20,"iso"),(m,"cifar100",20,"aniso"),(m,"imagenet",30,"iso"),(m,"imagenet",30,"aniso")]]
        h = H.get(m); cs += ["--"]*3 if h is None else [f"${float(h['star_haar']):+.4f}$", f"${float(h['depth_haar']):+.4f}$ ({float(h['z_haar']):+.2f})", f"{int(h['r_star_haar'])}/10"]
        b = BAL.get(m); cs.append("--" if b is None else f"${float(b['real_z']):{zf}}$")
        rows.append(NAME[m] + " & " + " & ".join(cs) + r" \\")
    DEC = {}
    if FINAL:   # two narrower panels: the stars with Gaussian hubs and the frames, then the star with Haar-resampled hubs
        split_ = lambda r: r[:-3].split(" & ")
        rows_a = [" & ".join([split_(r)[k] for k in (0, 1, 2, 3, 4, 5, 6, 10)]) + r" \\" for r in rows]; rows_h = [" & ".join([split_(r)[k] for k in (0, 7, 8, 9)]) + r" \\" for r in rows]
        T.panel("(a) The 12 backbones under the stars with Gaussian hubs: depth = excess B of the real centroids minus that of a matched star (negative = more hierarchical above the frame), with $z$ against the combined spread; the WordNet cut at $K{=}10$, 30 and 60 superclasses and the balanced frame.", "lcc@{\\hspace{5pt}}cccc@{\\hspace{5pt}}c",
                [r" & \multicolumn{2}{c}{C100, $K{=}20$} & \multicolumn{4}{c}{ImageNet, anisotropic star} & bal.\ frame \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}\cmidrule(lr){8-8}",
                 r"model & iso.\ $z$ & aniso.\ depth ($z$) & iso.\ $K{=}30$ $z$ & $K{=}10$ $z$ & $K{=}30$ depth ($z$) & $K{=}60$ $z$ & real $z$ \\"], rows_a, mids=(4, 9), colsep="2.2pt")
        DEC = {a["model"]: a for a in load("expR74_decoupling_summary.csv")} if ex("expR74_decoupling_summary.csv") else {}
        if DEC: T.prov.append("expR74_decoupling_summary.csv")
        dcol = lambda m: ("" if not DEC else (" & --" if m not in DEC else f" & ${float(DEC[m]['dec_z_mean']):+.2f}$ $\\pm$ ${float(DEC[m]['dec_z_sd']):.2f}$ ({int(round(10*float(DEC[m]['frac_certified'])))}/10)"))
        rows_h = [r[:-3] + dcol(M12[ri]) + r" \\" for ri, r in enumerate(rows_h)]
        T.panel("(a$'$) The same backbones under the star whose hubs are a Haar resample of the real hubs (ImageNet, $K{=}30$)" + ("; last column: the decoupling control, the real hubs kept and each cluster's offsets rotated by an independent Haar rotation (mean $z$ $\\pm$ s.d.\\ over 10 seeds; seeds certified at $z\\le-2$)." if DEC else "."), "lccc" + ("c" if DEC else ""),
                [r"model & star B & depth ($z$) & $r_{\text{star}}$" + (r" & decoupled $z$ (cert.)" if DEC else "") + r" \\"], rows_h, mids=(4, 9), colsep="4pt")
    else:
        T.panel("(a) The 12 backbones: depth = excess B of the real centroids minus that of a matched star (negative = more hierarchical above the frame), with $z$ against the combined spread.", "lcc@{\\hspace{5pt}}cc@{\\hspace{5pt}}ccc@{\\hspace{5pt}}c",
                [r" & \multicolumn{2}{c}{C100, $K{=}20$, Gaussian hubs} & \multicolumn{2}{c}{IN, $K{=}30$, Gaussian hubs} & \multicolumn{3}{c}{IN, $K{=}30$, Haar-resampled hubs} & bal.\ frame \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(lr){9-9}",
                 r"model & iso.\ star & aniso.\ star & iso.\ star & aniso.\ star & star B & depth ($z$) & $r_{\text{star}}$ & real $z$ \\"], rows, mids=(4, 9), colsep="1.8pt")
    def summ(ds, K, v):
        rows_ = [a for a in dv if a["dataset"] == ds and int(a["K"]) == K and a["variant"] == v]
        return f"{sum(float(a['z_depth'])<=-2 for a in rows_)}/{len(rows_)} at $z\\le-2$, {sum(float(a['z_depth'])>=2 for a in rows_)}/{len(rows_)} at $z\\ge+2$"
    ks = " ".join(f"CIFAR-100 $K{{=}}{K}$: iso {summ('cifar100',K,'iso')}; aniso {summ('cifar100',K,'aniso')}." for K in (5,10,20)) + " " + " ".join(f"ImageNet $K{{=}}{K}$: iso {summ('imagenet',K,'iso')}; aniso {summ('imagenet',K,'aniso')}." for K in (10,30,60))
    hits_ = [a for a in dv if a["dataset"] == "imagenet" and int(a["K"]) == 30 and a["variant"] == "aniso" and float(a["z_depth"]) <= -2]
    iso_c = [float(a["z_depth"]) for a in dv if a["dataset"] == "cifar100" and int(a["K"]) == 20 and a["variant"] == "iso"]
    cert_ = f" Certified on ImageNet ($n{{=}}1000$, the validated regime): {len(hits_)} of 12 backbones ({', '.join(NAME[a['model']] for a in hits_)}) with $z$ from ${max(float(a['z_depth']) for a in hits_):{zf}}$ to ${min(float(a['z_depth']) for a in hits_):{zf}}$; the withdrawn isotropic star reached $z{{=}}{{{max(iso_c):{zf}}}}$ on CIFAR-100."
    htxt = ""
    if H:
        ch = [m for m in M12 if H[m]["cert_haar"] == "True"]; zh = [float(H[m]["z_haar"]) for m in ch]
        htxt = f" With the hubs of the matched star drawn as a Haar resample of the real hubs (exact hub spectrum, 10 star seeds) the certified set is {', '.join(NAME[m] for m in ch)}, with $z$ from ${max(zh):+.2f}$ to ${min(zh):+.2f}$; $r_{{\\text{{star}}}}$ counts the star seeds whose excess B lies at or below the real one, a rank whose resolution is one part in eleven and which does not separate the certified backbones from the others."
    btxt = ""
    if BAL:
        cb = [m for m in M12 if float(BAL[m]["real_z"]) <= -2]; btxt = f" Balanced frame (complete-linkage WordNet cut with the smallest variance of cluster sizes, chosen before any depth run): certified {', '.join(NAME[m] for m in cb)}."
    # (b) leaf-label ViTs
    ltxt = ""
    if ex("expR70_inet1k_supervised.csv"):
        L = {a["model"]: a for a in load("expR70_inet1k_supervised.csv")}; rec = {(a["model"], a["dataset"]): a for a in load("expR52_census_haar_p999_200.csv")}
        rows = []
        for m in ("deit_b", "vit_b_in1k"):
            a = L[m]; rows.append(f"{NAME[m].replace(' (', ', ').replace(')', '')} & ${float(a['excess_in']):+.4f}$ ({int(a['r_in'])}) & ${float(a['excess_c100']):+.4f}$ ({int(a['r_c100'])}) & ${float(a['depth_gauss']):+.4f}$ ({float(a['z_gauss']):+.2f}) & ${float(a['depth_haar']):+.4f}$ ({float(a['z_haar']):+.2f}) \\\\")
        rb = rec[("i21k_b","imagenet")]; rc = rec[("i21k_b","cifar100")]; hb = H.get("i21k_b")
        rows.append(f"ViT-B, augreg, IN-21k (census) & ${float(rb['excess']):+.4f}$ ({int(rb['r_above'])}) & ${float(rc['excess']):+.4f}$ ({int(rc['r_above'])}) & " + (f"${float(hb['depth_gauss']):+.4f}$ ({float(hb['z_gauss']):+.2f}) & ${float(hb['depth_haar']):+.4f}$ ({float(hb['z_haar']):+.2f})" if hb else "-- & --") + r" \\")
        T.panel("(b) Two ViT-B/16 supervised on ImageNet-1k leaf labels only, through the census protocol, against the IN-21k census backbone.", "lcccc",
                [r"model & IN excess ($r$) & C100 excess ($r$) & depth, Gaussian hubs ($z$) & depth, Haar hubs ($z$) \\"], rows, colsep="4pt")
        ltxt = " (b) Supervision on leaf labels can produce the depth (the augreg recipe is certified under both stars) and need not (DeiT-B is not detected): the hierarchical label set of IN-21k is not what makes the census ViTs deep; their WordNet alignment is in Table~\\ref{tab:q7-wordnet}."
    T.newpart()
    # (c) MERU
    mtxt = ""
    if ex("expR63_meru_record.csv"):
        R = load("expR63_meru_record.csv"); by = {(a["model"], a["dataset"], a["modality"]): a for a in R}
        Mm = [m for m in ["meru_s","clip_s","meru_b","clip_b","meru_l","clip_l"] if any(a["model"] == m for a in R)]
        MN = {"meru_s":"MERU ViT-S","clip_s":"CLIP ViT-S","meru_b":"MERU ViT-B","clip_b":"CLIP ViT-B","meru_l":"MERU ViT-L","clip_l":"CLIP ViT-L"}
        def mc(a):
            if a is None: return "-- & -- & --"
            return f"${float(a['excess']):+.3f}" + ("" if gb(a) else r"^{\circ}") + f"$ ({int(a['r_above'])}) & ${float(a['delta_999_native']):.3f}$/${float(a['delta_999']):.3f}$ & ${float(a['z_depth']):{zf}}$"
        rows = []; mids = []
        for mod, lab in (("image","images"),("text","prompts")):
            for m in Mm: rows.append(MN[m] + f" & {lab} & " + mc(by.get((m,"cifar100",mod))) + " & " + mc(by.get((m,"imagenet",mod))) + r" \\")
            mids.append(len(rows))
        T.panel("(c) A hyperbolic backbone and its Euclidean twin under the census of record and the depth test.", "llccc@{\\hspace{8pt}}ccc",
                [r" & & \multicolumn{3}{c}{CIFAR-100 ($K{=}20$, $n{=}100$: depth unvalidated)} & \multicolumn{3}{c}{ImageNet ($K{=}30$, $n{=}1000$: depth validated)} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
                 r"model & input & exc.\ ($r$) & $\hat\delta_{99.9}$ nat./Eucl. & depth $z$ & exc.\ ($r$) & $\hat\delta_{99.9}$ nat./Eucl. & depth $z$ \\"], rows, mids=(mids[0],))
        im_ = [a for a in R if a["dataset"] == "imagenet" and a["modality"] == "image"]; mer = [a for a in im_ if a["family"] == "meru"]; cli = [a for a in im_ if a["family"] == "clip"]
        rng = lambda L_, k: (max(float(a[k]) for a in L_), min(float(a[k]) for a in L_))
        me, ce, mz, cz = rng(mer,"excess"), rng(cli,"excess"), rng(mer,"z_depth"), rng(cli,"z_depth"); natgap = max(abs(float(a["delta_999_native"])-float(a["delta_999"])) for a in mer)
        rad = ""
        if ex("expR71_meru_radii.csv"):
            Rr = load("expR71_meru_radii.csv"); T.prov.append("expR71_meru_radii.csv")
            rad = f" Its embeddings sit in the near-flat regime of the hyperboloid: learned curvature $c{{=}}{float(Rr[0]['curv']):.2f}$, spatial norm times $\\sqrt{{c}}$ at the median {min(float(a['radius_sqrtc_median']) for a in Rr):.3f}--{max(float(a['radius_sqrtc_median']) for a in Rr):.3f} (95th percentile {min(float(a['radius_sqrtc_p95']) for a in Rr):.3f}--{max(float(a['radius_sqrtc_p95']) for a in Rr):.3f}), and Lorentz-to-Euclidean pairwise distance ratio {min(float(a['lorentz_over_euclid_median']) for a in Rr):.4f}--{max(float(a['lorentz_over_euclid_median']) for a in Rr):.4f} at the median ({min(float(a['centroid_lorentz_over_euclid_median']) for a in Rr):.4f}--{max(float(a['centroid_lorentz_over_euclid_median']) for a in Rr):.4f} on the class centroids), so the control tests the training objective, not a curved geometry."
        mtxt = (r" (c) MERU \citep{desai2023meru} embeds images and text on the Lorentz hyperboloid with an entailment objective; its released twin is a CLIP baseline with the same ViT, RedCaps data and recipe minus the hyperbolic lift and entailment loss. Both pass through the instrument unchanged: class centroids of the projected embeddings (MERU: space-like hyperboloid coordinates; CLIP: unit vectors), Euclidean distances; exc.\ ($r$): excess over the Haar spectrum-matched null with the number of 200 replicates above the real value ($^{\circ}$: not genuine under BH over the 24 cells); nat./Eucl.: the same statistic in the model's own metric (Lorentz distance between tangent-space-mean centroids for MERU, angular for CLIP) against the Euclidean one; depth $z$: the anisotropic matched-star test of panel (a) (10 star seeds), validated at $n{=}1000$ only. "
                + f"On ImageNet images the excess runs from ${me[0]:+.3f}$ to ${me[1]:+.3f}$ for MERU and from ${ce[0]:+.3f}$ to ${ce[1]:+.3f}$ for its twin across ViT-S/B/L; MERU's Lorentz-metric $\\hat\\delta_{{99.9}}$ differs from its Euclidean value by at most ${natgap:.4f}$; depth $z$ runs from ${min(mz):{zf}}$ to ${max(mz):{zf}}$ for MERU and from ${min(cz):{zf}}$ to ${max(cz):{zf}}$ for CLIP, so neither is more hierarchical than a matched star and MERU is never ahead of its twin." + rad + r" MERU's hierarchy is a generic$\to$specific (text$\supset$image) partial order, not a class taxonomy.")
    # (d) the trained positive control (expR77; the earlier two-pass control of expR65 is superseded)
    ftxt = ""
    if FINAL and ex("expR77_positive_control.csv"):
        P77 = pd.read_csv(RES / "expR77_positive_control.csv").set_index("model"); T.prov += ["expR77_positive_control.csv", "expR77_positive_control_verdict.json"]
        LB = {"frozen": "frozen (census)", "ce_seed0": "leaf cross-entropy, seed 0", "hier_seed0": "leaf + hierarchical, seed 0", "ce_seed1": "leaf cross-entropy, seed 1", "hier_seed1": "leaf + hierarchical, seed 1"}
        rows = [f"{LB[m]} & ${P77.loc[m, 'excess']:+.4f}$ ({int(P77.loc[m, 'r_above'])}) & ${P77.loc[m, 'z_wn30']:+.2f}$ & ${P77.loc[m, 'zdec_mean_wn30']:+.2f}$ ({P77.loc[m, 'dec_frac_cert_wn30']:.1f}) & ${P77.loc[m, 'z_wn30bal']:+.2f}$ & ${P77.loc[m, 'zdec_mean_wn30bal']:+.2f}$ ({P77.loc[m, 'dec_frac_cert_wn30bal']:.1f}) \\\\" for m in ("frozen", "ce_seed0", "hier_seed0", "ce_seed1", "hier_seed1") if m in P77.index]
        T.panel("(d) A trained positive control: the census ViT-B/16 fine-tuned on the full ImageNet-1k training set with leaf cross-entropy alone and with an added hierarchical cross-entropy at the WordNet 30/6/2 cuts (weights 1/2/4), 2 seeds, identical batches within a seed; census excess under the centered Haar null, depth test and decoupling control on both frames.", "lccccc",
                [r"ViT-B/16 & excess ($r$) & $z$, WordNet-30 & dec.\ $z$ (cert.) & $z$, balanced & dec.\ $z$ (cert.) \\"], rows, size=r"\footnotesize", colsep="3pt")
        _c1 = int(round(P77.loc["ce_seed1", "dec_frac_cert_wn30"] * 10)) if "ce_seed1" in P77.index else None
        ftxt = (r" (d) Five epochs, AdamW, lr $10^{-5}$ encoder and $10^{-3}$ heads, batch 256, seeds 0 and 1; centroids of the census subset; decoupling over 10 seeds (fraction certified in parentheses). The hierarchical model is certified on both frames and keeps firing once decoupled in both seeds; the leaf-CE model and the frozen checkpoint are certified intact, the frozen checkpoint does not fire once decoupled on the frame of record"
                + (f" and the leaf-CE model fires there in 0 of 10 decoupled seeds for seed 0 and {_c1} of 10 for seed 1" if _c1 is not None else " and the leaf-CE model does not either") + ", and the frozen checkpoint fires in 7 of 10 decoupled seeds on the balanced frame. "
                r"The pre-set criterion (leaf-CE and frozen not certified intact) is not met, because ViT-B is certified by alignment; the discriminating comparison is the decoupled one. % expR77_positive_control.csv, expR77_positive_control_verdict.json")
    rtxt = ""
    if FINAL and ex("expR82_radial_control_summary.csv"):   # eighth review (1e): the radial control, once merged over the twelve backbones
        S82 = pd.read_csv(RES / "expR82_radial_control_summary.csv")
        if S82.model.nunique() == 12 and len(S82) == 24 and bool((S82.dec_runs == 10).all()):   # every backbone, both transforms, the ten decoupling seeds
            T.prov += ["expR82_radial_control.csv", "expR82_radial_control_summary.csv"]; rows = []
            for m in M12:
                a_ = S82[(S82.model == m) & (S82["transform"] == "l2norm")].iloc[0]; b_ = S82[(S82.model == m) & (S82["transform"] == "deradial")].iloc[0]
                rows.append(NAME[m] + " & " + " & ".join(f"${x:+.2f}$" for x in (a_.z_aniso, a_.z_haarhubs)) + f" & ${a_.dec_z_mean:+.2f}$ ({a_.dec_frac_certified:.1f}) & " + " & ".join(f"${x:+.2f}$" for x in (b_.z_aniso, b_.z_haarhubs)) + f" & ${b_.dec_z_mean:+.2f}$ ({b_.dec_frac_certified:.1f}) \\\\")
            T.panel("(e) Radial control: the depth test on the same clouds after L2-normalizing every centroid and after removing the radial component of every offset (its projection onto the hub direction), under both matched stars, with the decoupling control on each transformed cloud.", "lccc|ccc",
                    [r"& \multicolumn{3}{c|}{L2-normalized} & \multicolumn{3}{c}{radial component removed} \\", r"model & $z$ aniso & $z$ Haar-hub & decoupled $z$ (cert.) & $z$ aniso & $z$ Haar-hub & decoupled $z$ (cert.) \\"], rows, mids=(4, 9), size=r"\footnotesize", colsep="3pt")
            c_ = S82[S82.model.isin(["i21k_s", "i21k_b", "i21k_l", "dinov2_l"])]
            rtxt = f" (e) The four certified backbones keep $z\\le-2$ under both stars once the radial component is removed ({int((c_[c_['transform'] == 'deradial'].z_aniso <= -2).sum())} of 4), and {int((c_[c_['transform'] == 'l2norm'].z_aniso <= -2).sum())} of 4 under full L2 normalization; the decoupled transformed clouds fire in {int(round(c_[c_['transform'] == 'deradial'].dec_frac_certified.sum() * 10))} of 40 runs. % expR82_radial_control_summary.csv"
    T.write((r"\textbf{The depth test certifies in 4 of 12 ImageNet backbones under both matched stars that clusters are oriented toward their hubs, not that the hubs of the frame form a hierarchy, and no hub hierarchy is found in 9 of 12 backbones where the decoupled control detects an implanted one, and the test is blind in the other 3: none fires once cluster orientations are randomized, whereas a three-level hierarchy implanted at the same noise level is detected and survives that randomization (Table~\ref{tab:q5-power}); imposing the geometry does not create the structure, and leaf-label supervision can produce it or not.} " if (FINAL and DEC) else r"\textbf{Hierarchy above the superclasses is certified in four ImageNet backbones under both matched stars, imposing the geometry does not create it, and leaf-label supervision can produce it or not.} ") +
            r"(a) Isotropic star: Gaussian clouds with each superclass's RMS spread (the earlier, withdrawn construction); anisotropic star: within each superclass a Haar sample with the cloud's own covariance, hubs Gaussian at the real hub radius; Haar-resampled hubs: the same clusters with the hubs drawn as a Haar resample of the real hubs. Hub-randomizing Haar null, 10 star seeds; frames: CIFAR-100 coarse labels, ImageNet WordNet cut. The test is validated at $n{=}1000$ only (Table~\ref{tab:q5-power}); CIFAR-100 readings ($n{=}100$) are reported without certification."
            + cert_ + htxt + btxt + " K sweep: " + ks + ltxt
            + r" % expR56_depth_variants.csv, expR69_depth_haarhubs_summary.csv, expR64b_wn30bal_summary.csv, expR70_inet1k_supervised.csv",
            contd=[mtxt.strip() + ftxt + rtxt + r" % expR63_meru_record.csv, expR71_meru_radii.csv, expR65_hier_finetune.csv"])

# ======================================================================================================================
# Q5: power of the depth test — synthetic hierarchies, implanted trees on real clouds, both stars
# ======================================================================================================================
def q_power():
    T = Table("final/tab_q05_power_final.tex" if FINAL else "tab_q05_power.tex", "tab:q5-power", colsep="2pt"); zf = "+.2f" if FINAL else "+.1f"; T.prov += ["expR55_depth_power.csv", "expR55b_depth_power_leafframe.csv", "expR64b_wn30.csv", "expR64b_wn30_summary.csv", "expR64b_wn30bal_summary.csv", "expR67_frame_choice.json", "expR69_depth_haarhubs_summary.csv"]
    f1, f2 = RES/"expR55_depth_power.csv", RES/"expR55b_depth_power_leafframe.csv"
    def _summ(f):
        d = pd.read_csv(f); h = d[d.level != "star"]; s = d[d.level == "star"]; out = []
        for n in (100, 1000):
            for K in (6, 12, 20, 30):
                hh = h[(h.n == n) & (h.K == K)]; ss = s[(s.n == n) & (s.K == K)]
                if len(hh) == 0: continue
                out.append((n, K, [(hh[hh.ratio == r].z <= -2).mean() if (hh.ratio == r).any() else float("nan") for r in (0.1, 0.3, 0.6)], (ss.z <= -2).mean() if len(ss) else float("nan"), (ss.z >= 2).mean() if len(ss) else float("nan")))
        return out
    rows = []; mids = []
    for label, f in (("top-level, isotropic star", f1), ("leaf clusters, anisotropic star", f2)):
        if not f.exists(): continue
        for n, K, pw, fa1, fa2 in _summ(f): rows.append(f"{label} & {n} & {K} & " + " & ".join("--" if p != p else f"{p:.2f}" for p in pw) + f" & {fa1:.2f} & {fa2:.2f} \\\\")
        mids.append(len(rows))
    T.panel("(a) Synthetic hierarchies at $d{=}768$: two- and three-level hierarchies (pooled) and pure stars, within/between noise ratios 0.1/0.3/0.6, isotropic and anisotropic clusters pooled, 5 seeds.", "llcccccc",
            [r"frame & $n$ & $K$ & power @ ratio 0.1 & 0.3 & 0.6 & star $z\le-2$ & star $z\ge+2$ \\"], rows, mids=tuple(mids[:-1]), size=r"\footnotesize", colsep="4pt")
    bar_ = ""
    if f2.exists():
        d2 = pd.read_csv(f2); h2 = d2[d2.level != "star"]; s2 = d2[d2.level == "star"]; d1 = pd.read_csv(f1)
        p03 = (h2[(h2.n == 1000) & (h2.ratio <= 0.3)].z <= -2).mean(); p06 = (h2[(h2.n == 1000) & (h2.ratio == 0.6)].z <= -2).mean()
        fa1000 = ((s2[s2.n == 1000].z <= -2).mean(), (s2[s2.n == 1000].z >= 2).mean()); fa100 = (s2[(s2.n == 100) & (s2.K >= 12)].z <= -2).mean()
        top_fa = (d1[d1.level == "star"].z.abs() >= 2).mean(); top_pw = (d1[d1.level != "star"].z <= -2).mean()
        bar_ = (f" Pre-set bar: power $\\ge0.8$ for two- and three-level hierarchies at noise ratios $\\le0.3$ for both $n$, and false alarms $\\le5$\\% in each direction. Leaf frame: met at $n{{=}}1000$ (power {p03:.2f} at ratios $\\le0.3$ and {p06:.2f} at $0.6$; false alarms {100*fa1000[0]:.0f}\\% and {100*fa1000[1]:.0f}\\%), failed at $n{{=}}100$, where $K\\ge12$ (under ten points per cluster) gives {100*fa100:.0f}\\% false alarms at $z\\le-2$. Top-level frame: power {top_pw:.2f}, false alarms {100*top_fa:.0f}\\% pooled.")
    # (b) implanted
    ptxt = ""
    if ex("expR64b_wn30_summary.csv"):
        A = load("expR64b_wn30_summary.csv"); B = {r["model"]: r for r in load("expR64b_wn30bal_summary.csv")} if ex("expR64b_wn30bal_summary.csv") else {}
        Hs = {r["model"]: r for r in load("expR69_depth_haarhubs_summary.csv")} if ex("expR69_depth_haarhubs_summary.csv") else {}
        fc = json.load(open(RES/"expR67_frame_choice.json")) if ex("expR67_frame_choice.json") else None
        hb = lambda r: f"{r['hits_s0']}/{r['hits_s05']}/{r['hits_s1']}"
        rows = []
        for i, r in enumerate(A):
            ss = r["s_star"] if r["s_star"] not in ("", "nan") else "--"
            tz = f"${float(r['tight_z_s0']):{zf}}$/${float(r['tight_z_s05']):{zf}}$/${float(r['tight_z_s1']):{zf}}$" if r.get("tight_z_s1") not in (None, "", "nan") else "--"
            b = B.get(r["model"]); bb = (f"${float(b['real_z']):{zf}}$ & {hb(b)} & {b['s_star'] if b['s_star'] not in ('', 'nan') else '--'}") if b else "-- & -- & --"
            h = Hs.get(r["model"]); hh = (f"{int(h['fa_s0_haar'])}/{int(h['hits_s1_haar'])} & ${float(h['z_mean_s1_haar']):{zf}}$") if h else "-- & --"
            rows.append(f"{NAME[r['model']]} & ${float(r['real_z']):{zf}}$ & {float(r['ratio_real']):.1f} & {hb(r)} & {ss} & ${float(r['z_mean_s1']):{zf}}$ & {tz} & {hh} & {bb} \\\\")
        T.panel("(b) Implanted two-level trees on the real ImageNet centroids: the hub arrangement replaced at strength $s$, every within-cluster offset kept; hits = implant seeds with $z\\le-2$ at $s{=}0/0.5/1$, $\\bar z_1$ = mean $z$ at $s{=}1$, w/b = within/between spread.", "lcccccc|cc|ccc",
                [r" & \multicolumn{6}{c|}{WordNet-30 frame of record, Gaussian-hub star} & \multicolumn{2}{c|}{Haar-hub star} & \multicolumn{3}{c}{balanced frame (pre-specified)} \\",
                 r"model & real $z$ & w/b & hits & $s^*$ & $\bar z_1$ & tight $z$ at $s{=}0/0.5/1$ & hits $s{=}0/1$ & $\bar z_1$ & real $z$ & hits & $s^*$ \\"], rows, mids=(4, 9), colsep="1.8pt")
        dep = [a for a in load("expR64b_wn30.csv") if a["kind"] == "depth" and a["partition"] == "rand6" and a["s"] != "real"]
        pw = {s: sum(float(a["z"]) <= -2 for a in dep if float(a["s"]) == s)/max(1, sum(1 for a in dep if float(a["s"]) == s)) for s in (0.0, 0.25, 0.5, 0.75, 1.0)}
        fa0 = sum(float(a["z"]) <= -2 for a in dep if float(a["s"]) == 0.0); n0 = sum(1 for a in dep if float(a["s"]) == 0.0)
        ratio_lo, ratio_hi = min(float(r["ratio_real"]) for r in A), max(float(r["ratio_real"]) for r in A)
        fctxt = (f" Balanced frame: the $K{{=}}30$ agglomerative cut of the WordNet distance matrix ({fc['chosen']} linkage) with the smallest variance of cluster sizes, chosen and logged before any depth run; sizes {fc['chosen_sizes'][0]}--{fc['chosen_sizes'][-1]} classes against {min(fc['sizes']['average'])}--{max(fc['sizes']['average'])} for the frame of record." if fc else "")
        htx = ""
        if Hs:
            fa = sum(int(h["fa_s0_haar"]) for h in Hs.values()); n_ = sum(int(h["n_s0"]) for h in Hs.values()); hi = sum(int(h["hits_s1_haar"]) for h in Hs.values()); n1 = sum(int(h["n_s1"]) for h in Hs.values())
            htx = f" Under the Haar-hub star the same implants give {fa} of {n_} false alarms at $s{{=}}0$ and {hi} of {n1} detections at $s{{=}}1$ (power {hi/max(1,n1):.2f})."
        ptxt = (r" (b) Six super-hubs of five, hub $= s\cdot$super-hub $+ (1-s+0.4s)\cdot$own Gaussian draw, hub cloud rescaled to the real RMS radius; the depth test of Table~\ref{tab:q4-depth} runs unchanged, 5 implant seeds per $s$. hits: implant seeds with $z\le-2$ (of 5); $s^*$: first $s$ with $\ge4$ of 5; within/between: RMS of the within-cluster offsets over the point-weighted RMS of the hub displacements"
                + f" ({ratio_lo:.1f} to {ratio_hi:.1f} on the real clouds, against 0.1--0.6 in the synthetic sweep); tight: the same implant after shrinking the offsets to within/between $=0.6$ (2 seeds). Power (fraction of runs with $z\\le-2$) by $s$: " + ", ".join(f"$s{{=}}{s:g}$: {v:.2f}" for s, v in pw.items()) + f"; false alarms at $s{{=}}0$: {fa0} of {n0}." + htx + fctxt)
    ctxt80 = ""; ctxt79 = ""
    if FINAL and ex("expR80_decision.csv") and ex("expR80_implanted_alignment.csv"):   # priority 1c (brief of 2026-09-21): in the appendix whatever the decision rule said; its outcome is stated
        dec80 = load("expR80_decision.csv")[0]; D80 = pd.read_csv(RES / "expR80_implanted_alignment.csv"); D80["hit"] = D80.z_depth <= -2; T.prov += ["expR80_implanted_alignment.csv", "expR80_decision.csv"]
        SS = sorted(D80.s.unique()); pm = D80.groupby(["model", "s"]).hit.mean().unstack(); pooled = D80.groupby("s").hit.mean()
        rows = [NAME[m] + " & " + " & ".join(f"{pm.loc[m, s]:.1f}" for s in SS) + r" \\" for m in M12 if m in pm.index] + [r"\midrule pooled (12 backbones $\times$ 5 seeds) & " + " & ".join(f"{pooled[s]:.2f}" for s in SS) + r" \\"]
        T.panel("(c) Implanted hub alignment on the real ImageNet clouds: from the decoupled cloud of the decoupling control, each cluster's principal axis rotated toward its hub direction by a fraction $s$ of the angle (hubs and within-cluster spectra unchanged); detection rate at $z\\le-2$ over 5 seeds per backbone and pooled.", "l" + "c" * len(SS),
                [r"model & " + " & ".join(f"$s{{=}}{s:g}$" for s in SS) + r" \\"], rows, mids=(4, 9), size=r"\footnotesize", colsep="6pt")
        n1 = int((D80.s == 1.0).sum()); h1 = int(D80[D80.s == 1.0].hit.sum()); n0 = int((D80.s == 0.0).sum()); h0 = int(D80[D80.s == 0.0].hit.sum()); met80 = dec80["rule_power_ge_0_8_fa_le_0_05"] == "True"
        ctxt80 = (f" (c) Implanted alignment is detected in {h1} of {n1} runs at full strength ({100 * h1 / n1:.0f}\\%) and in {h0} of {n0} at zero; " + ("the pre-set bar of 0.8 power at full strength with false alarms at or below 0.05 is met" if met80 else "the pre-set bar of 0.8 power at full strength is not met")
                  + ", and the power is backbone-dependent: every seed for the supervised ViTs and CLIP-B, none for DINOv2-S/B/L. % expR80_implanted_alignment.csv, expR80_decision.csv")
    if FINAL and ex("expR79_synthetic_deep_poincare.csv"):   # priority 1b (brief of 2026-09-21): a deep hierarchy at the real noise level, and the WordNet Poincare embeddings
        E79 = pd.read_csv(RES / "expR79_synthetic_deep_poincare.csv"); T.prov += ["expR79_synthetic_deep_poincare.csv"]
        CL = {"synthetic_deep_vitl_spectrum": "three-level hierarchy, ViT-L spectrum", "synthetic_flat_vitl_spectrum": "flat control, ViT-L spectrum", "wordnet_poincare_d10": r"WordNet Poincar\'e, $d{=}10$", "wordnet_poincare_d50": r"WordNet Poincar\'e, $d{=}50$"}
        rows = [f"{CL.get(r.cloud, r.cloud)} & {int(r.seed)} & {int(r.dim)} & ${r.excess:+.4f}$ ({int(r.r_above)}) & ${r.z:+.2f}$ & ${r.zdec_mean:+.2f}$ $\\pm$ {r.zdec_sd:.2f} ({r.dec_frac_cert:.1f}) \\\\" for _, r in E79.iterrows()]
        T.panel("(d) A deep hierarchy at the real noise level: synthetic clouds with ViT-L's real ImageNet spectrum and a three-level implanted hierarchy (nested 2/6/30 cuts of the frame) at ViT-L's real within/between ratio, a flat two-level control (30 iid hubs, same spectrum and ratio), 5 seeds each, and the WordNet Poincar\\'e embeddings of Nickel and Kiela trained on the transitive closure of the tree over the 1000 ImageNet leaves, read with Euclidean distances on the ball coordinates.", "lccccc",
                [r"cloud & seed & $d$ & excess ($r$) & depth $z$ & decoupled $z$ (cert.) \\"], rows, mids=(5, 10), size=r"\footnotesize", colsep="4pt")
        deep = E79[E79.cloud == "synthetic_deep_vitl_spectrum"]; flat = E79[E79.cloud == "synthetic_flat_vitl_spectrum"]
        ctxt79 = (f" (d) Census excess under the centered Haar null (rank in parentheses), depth test with the matched anisotropic star at $K{{=}}30$ (10 star seeds), decoupling control (mean $\\pm$ s.d.\\ over 10 seeds; fraction certified in parentheses). The deep hierarchy fires in {int((deep.z <= -2).sum())} of {len(deep)} seeds and the flat control in {int((flat.z <= -2).sum())} of {len(flat)}; "
                  f"the decoupling control fires in {int((deep.zdec_mean <= -2).sum())} of {len(deep)} deep seeds, deeper than the intact cloud; the Poincar\\'e embeddings fire at no dimension. % expR79_synthetic_deep_poincare.csv")
    ctxt81 = ""
    if FINAL and ex("expR81_deep_per_backbone_summary.csv"):   # closing pass (2026-09-22): power per backbone at its own spectrum and ratio
        S81 = pd.read_csv(RES / "expR81_deep_per_backbone_summary.csv").set_index("model"); T.prov += ["expR81_deep_per_backbone.csv", "expR81_deep_per_backbone_summary.csv"]
        rows = [f"{NAME[m]} & {S81.loc[m, 'ratio']:.2f} & {int(S81.loc[m, 'n_seeds'])} & {S81.loc[m, 'power']:.2f} & ${S81.loc[m, 'z_mean']:+.2f}$ & {S81.loc[m, 'dec_power']:.2f} & ${S81.loc[m, 'dec_z_mean']:+.2f}$ \\\\" for m in M12 if m in S81.index]
        T.panel("(e) Power per backbone for a deep hierarchy at the backbone's own noise level: the three-level synthetic hierarchy of (d) built with each backbone's real ImageNet spectrum and within/between ratio; depth test at $K{=}30$ (10 star seeds) on the intact cloud and the decoupling control (10 seeds) on each.", "lcccccc",
                [r"model & within/between & seeds & power & mean $z$ & decoupled power & decoupled mean $z$ \\"], rows, mids=(4, 9), size=r"\footnotesize", colsep="4pt")
        cov = [NAME[m] for m in M12 if m in S81.index and S81.loc[m, "dec_power"] >= 0.8]; covi = [NAME[m] for m in M12 if m in S81.index and S81.loc[m, "power"] >= 0.8]   # ninth review: the decoupled power is the criterion of the text
        ctxt81 = f" (e) Decoupled power $\\ge0.8$ in {len(cov)} backbones ({', '.join(cov)}) and below in the other {12 - len(cov)}, the criterion of the text; intact power $\\ge0.8$ in {len(covi)} ({', '.join(covi)}); DINOv2 with 20 seeds, the rest with 5. % expR81_deep_per_backbone_summary.csv"
    ctxt83 = ""
    if FINAL and ex("expR83_flat_balanced_summary.csv"):   # ninth review (2026-09-22): the balanced frame's false alarms on the flat control, matched and mismatched hub assignment
        S83 = pd.read_csv(RES / "expR83_flat_balanced_summary.csv").set_index("built"); T.prov += ["expR83_flat_balanced.csv", "expR83_flat_balanced_summary.csv"]
        E79f = pd.read_csv(RES / "expR79_synthetic_deep_poincare.csv"); E79f = E79f[E79f.cloud == "synthetic_flat_vitl_spectrum"]; fa79 = int(round(E79f.dec_frac_cert.sum() * 10))
        LBL = {"balanced": "matched hubs (balanced frame)", "wn30": "mismatched hubs (WordNet-30 frame)"}
        rows = [f"{LBL[b]} & {int(S83.loc[b, 'intact_fired'])}/{int(S83.loc[b, 'n_intact'])} & ${S83.loc[b, 'intact_z_mean']:+.2f}$ & {int(S83.loc[b, 'decoupled_fired'])}/{int(S83.loc[b, 'n_decoupled'])} & ${S83.loc[b, 'dec_z_mean']:+.2f}$ & ${S83.loc[b, 'dec_z_max']:+.2f}$ to ${S83.loc[b, 'dec_z_min']:+.2f}$ \\\\" for b in ("balanced", "wn30") if b in S83.index]
        T.panel("(f) False alarms of the balanced frame: the flat control of (d), ViT-L's spectrum and ratio, 5 seeds, read on the balanced thirty-superclass frame intact and decoupled (10 seeds each), with the flat hubs assigned by the balanced frame itself or by the WordNet-30 frame of record.", "lcc|ccc",
                [r"flat control & intact fired & mean $z$ & decoupled fired & mean $z$ & range \\"], rows, size=r"\footnotesize", colsep="3pt")
        ctxt83 = f" (f) With matched hubs the balanced frame fires once decoupled in {int(S83.loc['balanced', 'decoupled_fired'])} of {int(S83.loc['balanced', 'n_decoupled'])} runs, against {fa79} of {10 * len(E79f)} on the frame of record; a flat cloud clustered by the other frame fires on it intact in {int(S83.loc['wn30', 'intact_fired'])} of {int(S83.loc['wn30', 'n_intact'])}, which is frame mismatch, not depth. % expR83_flat_balanced_summary.csv, expR79_synthetic_deep_poincare.csv"
    T.write((r"\textbf{The depth test has full power on synthetic hierarchies at the leaf frame and none at the top-level frame used on real backbones.} " if FINAL else r"\textbf{The depth test meets its bar on synthetic hierarchies at ImageNet's class count, raises no false alarm on real clouds under either star, and misses implanted depth at the real within-cluster spread.} ") +
            r"(a) Power = fraction of hierarchy runs with $z\le-2$; star columns = false-alarm rates in each direction. Top-level frame: the test receives the $K$ super-cluster labels, as the earlier isotropic test did on the real backbones; its hub null keeps every frame cluster intact and only rearranges the $K$ hubs, so hierarchy below the frame is invisible by construction. Leaf frame: the test receives the finest cluster labels with the anisotropic matched star (configurations with more leaves than points are infeasible and omitted)." + bar_ + ptxt + ctxt80 + ctxt79 + ctxt81 + ctxt83
            + r" % expR55_depth_power.csv, expR55b_depth_power_leafframe.csv, expR64b_wn30.csv, expR64b_wn30_summary.csv, expR64b_wn30bal_summary.csv, expR67_frame_choice.json, expR69_depth_haarhubs_summary.csv")

# ======================================================================================================================
# Q11: training interventions and ORC edge types
# ======================================================================================================================
def q_interventions():
    T = Table("tab_q11_interventions.tex", "tab:q11-interventions", size=r"\footnotesize", colsep="4pt"); T.prov += ["analysis4_finetuning.csv", "e1_delta_by_layer.csv", "e5_random_control.csv", "night/arch_matched_triplet.csv", "night/orc_bridges.csv"]
    NM = {"dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","i21k_b":"ViT-B","clip_b":"CLIP-B","clip_b_vision":"CLIP-B","dinov2_l":"DINOv2-L","dinov1_b":"DINO-B"}
    rows = []; mids = []
    if (ABL/"analysis4_finetuning.csv").exists():
        for a in csv.DictReader(open(ABL/"analysis4_finetuning.csv")):
            rows.append(f"non-hierarchical fine-tuning & {NM.get(a['model'], a['model'])} & ${float(a['delta_before']):.3f}$ & ${float(a['delta_after']):.3f}$ & ${float(a['pct_change']):+.0f}\\%$ \\\\")
        mids.append(len(rows))
    if (ABL/"e1_delta_by_layer.csv").exists():
        L = list(csv.DictReader(open(ABL/"e1_delta_by_layer.csv")))
        for m in ("dinov2_b", "clip_b_vision"):
            rr = sorted([a for a in L if a["model"] == m and int(a["layer"]) >= 1], key=lambda a: int(a["layer"]))
            if rr: rows.append(f"across depth (layer {rr[0]['layer']} $\\to$ {rr[-1]['layer']}) & {NM[m]} & ${float(rr[0]['delta_normalized']):.3f}$ & ${float(rr[-1]['delta_normalized']):.3f}$ & ${100*(float(rr[-1]['delta_normalized'])-float(rr[0]['delta_normalized']))/float(rr[0]['delta_normalized']):+.0f}\\%$ \\\\")
        mids.append(len(rows))
    if (ABL/"e5_random_control.csv").exists():
        E = list(csv.DictReader(open(ABL/"e5_random_control.csv"))); by = {(a["model"], a["mode"]): float(a["delta_normalized"]) for a in E}
        for m in dict.fromkeys(a["model"] for a in E):
            if (m, "pretrained") in by and (m, "random") in by: rows.append(f"random weights (pretrained $\\to$ random) & {NM.get(m, m)} & ${by[(m,'pretrained')]:.3f}$ & ${by[(m,'random')]:.3f}$ & ${100*(by[(m,'random')]-by[(m,'pretrained')])/by[(m,'pretrained')]:+.0f}\\%$ \\\\")
        mids.append(len(rows))
    T.panel("(a) Interventions on the raw reading, within architecture.", "llccc", [r"intervention & model & $\delta$ before & $\delta$ after & change \\"], rows, mids=tuple(mids[:-1]))
    if ex("night/arch_matched_triplet.csv"):
        tr = load("night/arch_matched_triplet.csv")
        rows = [f"{DSL.get(a['dataset'], a['dataset'])} & ${float(a['i21k_b']):+.3f}$ & ${float(a['dinov1_b']):+.3f}$ & ${float(a['clip_b']):+.3f}$ \\\\" for a in tr]
        T.panel("(b) Matched architecture, three objectives: excess of ViT-B/16 supervised, DINO and CLIP.", "lccc", [r"dataset & ViT-B (supervised) & DINO-B & CLIP-B \\"], rows)
    br = load("night/orc_bridges.csv")
    rows = [f"{NAME[a['model']]} & {DSH.get(a['dataset'], a['dataset'])} & {float(a['orc_within']):.2f} & {float(a['orc_across']):.2f} & {100*float(a['fneg_within']):.1f} & {100*float(a['fneg_across']):.1f} \\\\" for a in br]
    T.panel("(c) Ollivier--Ricci curvature by edge type on the centroid kNN graph (30 WordNet superclasses).", "llcccc", [r"model & dataset & ORC within & ORC across & \%neg within & \%neg across \\"], rows)
    T.write(r"\textbf{The raw reading is learned, and negative curvature lives on the bridges between superclasses.} "
            r"(a) Fine-tuning on a task with no class hierarchy raises $\delta$, $\delta$ falls across transformer depth, and random-weight networks read a higher raw $\delta$ with near-Gaussian features, so they are not counted as evidence; label shuffling leaves $\delta$ unchanged, as a null-invariant quantity should (Figure~\ref{fig:causal}). "
            r"(b) The three objectives on one architecture; the family separations emerge with scale rather than at fixed size. "
            r"(c) Negative curvature concentrates on between-cluster bridges while the mean ORC measures within-cluster compactness; DINOv2 on ImageNet inverts the pattern and recovers it on CIFAR-100. % analysis4_finetuning.csv, e1_delta_by_layer.csv, e5_random_control.csv, night/arch_matched_triplet.csv, night/orc_bridges.csv")

# ======================================================================================================================
# Q6: the tree map — six configurations x three measures, degeneracy, CPCC
# ======================================================================================================================
def q_treemap():
    T = Table("tab_q06_treemap.tex", "tab:q6-treemap", colsep="3pt"); T.prov += ["exp22_tree_similarity_imagenet.npz", "exp23_treemap_controls.npz", "exp23_config_diagnostics.json", "expR58_treemap_cutfree_summary.csv"]
    diag = json.load(open(RES/"exp23_config_diagnostics.json"))
    SEL = {}
    for ds in ("imagenet", "cifar100"):
        adm = {k: v for k, v in diag.items() if k.startswith(ds + "|") and v["maxfrac"] <= 0.5}; SEL[ds] = max(adm, key=lambda k: adm[k]["cpcc"]).split("|", 1)[1].replace("|", "-")
    assert SEL == {"imagenet": "cosine-average", "cifar100": "cosine-complete"}, SEL
    cf = {(a["dataset"], a["metric"], a["linkage"]): a for a in load("expR58_treemap_cutfree_summary.csv")} if ex("expR58_treemap_cutfree_summary.csv") else {}
    rows = []; rows2 = []; mids = []; ADM = {}
    for ds, tag in [("imagenet", "summary_in"), ("cifar100", "summary_c1")]:
        z = np.load(RES/"exp23_treemap_controls.npz", allow_pickle=True)[tag].item()
        for key in ["('euclid', 'average')","('euclid', 'complete')","('euclid', 'ward')","('cosine', 'average')","('cosine', 'complete')","('cosine', 'ward')"]:
            v = z[key]; cfg = key.replace("('", "").replace("')", "").replace("', '", "-"); metric, link = cfg.split("-"); d = diag[f"{ds}|{metric}|{link}"]; deg = d["maxfrac"] > 0.5
            mark = r" (selected)" if (cfg == SEL[ds] and not deg) else (" (deg.)" if deg else "")
            if not deg and cfg != SEL[ds]: ADM.setdefault(ds, []).append((v["big_vs_sup"], v["sup_vs_sup"]))
            c = cf.get((ds, metric, link)); cc = (f"{float(c['ari_cut_big_vs_block']):.2f} / {float(c['ari_cut_within_block']):.2f} & {float(c['coph_corr_big_vs_block']):.2f} / {float(c['coph_corr_within_block']):.2f} & {float(c['triplet_agree_big_vs_block']):.2f} / {float(c['triplet_agree_within_block']):.2f}") if c else "-- & -- & --"
            rows.append(f"{DSL[ds]} & {cfg}{mark} & {d['maxfrac']:.2f} & {d['cpcc']:.3f} & {v['big_vs_sup']:.2f} & {v['small_vs_sup']:.2f} & {v['sup_vs_sup']:.2f} \\\\")
            rows2.append(f"{DSL[ds]} & {cfg}{mark} & {cc} \\\\") if c else None
        mids.append(len(rows))
    T.panel("(a) Selection diagnostics and mean pairwise ARI at the reference cut.", "llcc|ccc", [r" & & & & \multicolumn{3}{c}{mean pairwise ARI at the reference cut} \\",
            r"dataset & configuration & max frac.\ & CPCC & B/L/G vs block & S, DINO-B vs block & within \\"], rows, mids=(mids[0],), colsep="3pt")
    T.panel("(b) Three measures of DINOv2-B/L/G against the supervised+contrastive block, each written as vs block / within block.", "ll|ccc",
            [r"dataset & configuration & ARI at the cut & cophenetic corr. & triplet agreement \\"], rows2, mids=(mids[0],), colsep="3pt")
    z22 = np.load(RES/"exp22_tree_similarity_imagenet.npz", allow_pickle=True); ari22 = z22["ari"]; nm22 = list(z22["models"]); ix = {m: i for i, m in enumerate(nm22)}
    SUP = ["i21k_t","i21k_s","i21k_b","i21k_l","clip_b","clip_l","siglip_b"]; D2 = ["dinov2_s","dinov2_b","dinov2_l","dinov2_g"]; VIT = SUP[:4]; CON = SUP[4:]
    naive_max = max(ari22[ix[a], ix[b]] for a in VIT for b in CON); island_max = max(ari22[ix[a], ix[b]] for a in D2 for b in SUP); chain = diag["imagenet|euclid|average"]["maxfrac"]
    zin = np.load(RES/"exp23_treemap_controls.npz", allow_pickle=True)["summary_in"].item()["('cosine', 'average')"]
    miss = []
    for (ds_, m_, l_), c in cf.items():
        if ds_ != "imagenet" or diag[f"imagenet|{m_}|{l_}"]["maxfrac"] > 0.5: continue
        for a, b in (("ari_cut_big_vs_block", "ari_cut_within_block"), ("coph_corr_big_vs_block", "coph_corr_within_block"), ("triplet_agree_big_vs_block", "triplet_agree_within_block")): miss.append(1 - float(c[a]) / float(c[b]))
    med_miss = float(np.median(miss)) if miss else float("nan"); json.dump({"median_missing_admissible_imagenet": med_miss, "n": len(miss)}, open(RES / "final_pass_island_gap.json", "w"))
    adm_in = ADM.get("imagenet", []); adm_txt = (f" Under the other {len(adm_in)} admissible ImageNet configurations the DINOv2-vs-block agreement is {min(a for a, _ in adm_in):.2f}--{max(a for a, _ in adm_in):.2f} against {min(b for _, b in adm_in):.2f}--{max(b for _, b in adm_in):.2f} within the block; over the admissible configurations and the three measures the median fraction of within-block agreement missing is {med_miss:.2f}, about a third." if adm_in else "")
    ctxt84 = ""
    if ex("expR84_tree_ceiling_summary.csv"):   # tenth review (2026-09-22): the within-model ceiling of the three measures
        S84 = pd.read_csv(RES / "expR84_tree_ceiling_summary.csv").set_index("model"); T.prov += ["expR84_tree_ceiling.csv", "expR84_tree_ceiling_summary.csv"]
        rows = [f"{NAME[m] if m != 'ALL' else 'mean over backbones'} & {S84.loc[m, 'ari_cut_mean']:.2f} $\\pm$ {S84.loc[m, 'ari_cut_sd']:.2f} & {S84.loc[m, 'coph_corr_mean']:.2f} $\\pm$ {S84.loc[m, 'coph_corr_sd']:.2f} & {S84.loc[m, 'triplet_agree_mean']:.2f} $\\pm$ {S84.loc[m, 'triplet_agree_sd']:.2f} & {S84.loc[m, 'triplet_agree_min']:.2f} \\\\" for m in M12 + ["ALL"] if m in S84.index]
        T.panel("(c) Within-model ceiling under the selected ImageNet configuration: the three measures of (b) between pairs of the 30 bootstrap centroid sets of each backbone (mean $\\pm$ s.d.\\ over the 435 pairs) and the lowest triplet agreement over pairs.", "lccc|c", [r"model & ARI at the cut & cophenetic corr. & triplet agreement & lowest triplet \\"], rows, mids=(4, 9, 12), size=r"\footnotesize", colsep="4pt")
        c_sel = cf.get(("imagenet", "cosine", "average"))
        ctxt84 = f" (c) Two resamples of the same model agree on {S84.loc['ALL', 'triplet_agree_mean']:.2f} of the triplets on average, {S84.loc['ALL', 'triplet_agree_min']:.2f} for the lowest backbone: the ceiling against which the cross-model {float(c_sel['triplet_agree_big_vs_block']):.2f} and the within-block {float(c_sel['triplet_agree_within_block']):.2f} are read. % expR84_tree_ceiling_summary.csv"
    T.write(r"\textbf{The categorical island is a chaining artifact of the naive cut; a moderate gap remains under every admissible configuration and closes only in triplet agreement under cosine-average.} "
            r"(a) Six configurations (metric $\times$ linkage) per dataset with the two quantities that drive the selection criterion of \S\ref{sec:content}: the degeneracy diagnostic (largest-cluster fraction at the reference cut, worst model) and the mean cophenetic fidelity (CPCC) of each model's dendrogram to its own distance matrix; configurations with max-cluster fraction $>0.5$ are excluded and, among the rest, the highest CPCC selects the marked row; then the mean pairwise ARI between the cuts at 30 (ImageNet) or 20 (CIFAR-100) clusters. (b) Dendrograms rebuilt from the census cache: ARI at the cut, Pearson correlation between the two trees' cophenetic distance vectors, and agreement on which pair merges first over $10^4$ random class triplets. "
            + f"Under the naive ImageNet configuration, cross-family ARI between the supervised ViTs and the contrastive VLMs reaches {naive_max:.2f} while every DINOv2-vs-block pair stays at or below {island_max:.2f}, and the worst model's largest cluster holds {100*chain:.0f}\\% of the classes. Under the selected configuration the DINOv2 family agrees with the block at {zin['big_vs_sup']:.2f} against {zin['sup_vs_sup']:.2f} within the block." + adm_txt + ctxt84
            + r" % exp22_tree_similarity_imagenet.npz, exp23_treemap_controls.npz, exp23_config_diagnostics.json, expR58_treemap_cutfree_summary.csv")

# ======================================================================================================================
# Q7: WordNet alignment, superclass recovery, DBpedia, HierarCaps
# ======================================================================================================================
def q_wordnet():
    T = Table("final/tab_q07_wordnet_final.tex" if FINAL else "tab_q07_wordnet.tex", "tab:q7-wordnet", colsep="2.6pt"); T.prov += ["exp3_alignment.csv", "exp28_recovery_per_config.csv", "exp8_p1_recovery.csv", "exp1_delta_controls.csv", "exp8_p6_pooling.csv", "expR70_inet1k_supervised.csv", "exp14_dbpedia.csv", "expR61_dbpedia_record.csv", "exp27_dbpedia_treemap.json", "exp16_hierarcaps.csv"]
    e3 = {a["model"]: a for a in load("exp3_alignment.csv")}; r28 = {a["model"]: a for a in load("exp28_recovery_per_config.csv")}
    CONFS = ["euclid-average","euclid-ward","cosine-average","cosine-complete","cosine-ward"]
    e1 = {}
    for a in load("exp1_delta_controls.csv"): e1.setdefault(a["model"], {})[a["variant"]] = float(a["delta_max"])
    p6 = {a["model"]: float(a["rho_mean"]) for a in load("exp8_p6_pooling.csv") if a["m_imgs"] == "1"}
    L70 = {a["model"]: a for a in load("expR70_inet1k_supervised.csv")} if ex("expR70_inet1k_supervised.csv") else {}
    rows = []; rows2 = []
    for m in M12:
        a = e3[m]; g = e1.get(m, {})
        rows.append(f"{NAME[m]} & ${float(a['spearman_wn']):+.2f}$ & ${float(a['spearman_shuf']):+.2f}$ & ${float(a['spearman_gauss']):+.2f}$ & " + (f"${p6[m]:+.2f}$" if m in p6 else "--") + " & " + (f"{g['grp_wordnet']:.3f} / {g['grp_random']:.3f}" if "grp_wordnet" in g else "--") + r" \\")
        rows2.append(NAME[m] + " & " + " & ".join(f"{float(r28[m][c]):.2f}" for c in CONFS) + r" \\")
    for m in ("deit_b", "vit_b_in1k"):
        if m in L70:
            a = L70[m]; rows.append(f"{NAME[m]} & ${float(a['spearman_wn']):+.2f}$ & ${float(a['spearman_shuf']):+.2f}$ & -- & -- & -- \\\\")
            rows2.append(NAME[m] + " & " + " & ".join(f"{float(a['ari_'+c]):.2f}" for c in CONFS) + r" \\")
    HEAD = ["euclid-avg$^{\\dagger}$","euclid-ward","cosine-avg$^{\\dagger}$","cosine-complete$^{*}$","cosine-ward"]
    T.panel("(a) Alignment with WordNet on ImageNet.", "lcccc|c",
            [r" & \multicolumn{4}{c|}{Spearman $\rho$, inter-centroid vs WordNet distance} & raw $\hat\delta$ of \\",
             r"model & $\rho_{\text{WN}}$ & shuffle & Gaussian & single image & WordNet / random groups \\"], rows, mids=(4, 9, 12), size=r"\footnotesize", colsep="4pt")
    T.panel("(b) Recovery of the CIFAR-100 superclasses: ARI of the 20-cluster cut per configuration ($^{\\dagger}$: degenerate on CIFAR-100, euclid-average being the naive configuration; $^{*}$: criterion-selected).", "lccccc",
            ["model & " + " & ".join(HEAD) + r" \\"], rows2, mids=(4, 9, 12), size=r"\footnotesize", colsep="4pt")
    ADM = ["euclid-ward", "cosine-complete", "cosine-ward"]; D2 = ["dinov2_s","dinov2_b","dinov2_l","dinov2_g"]
    wins = sum(float(r28["i21k_b"][c]) > float(r28[m][c]) for c in ADM for m in D2); d2v = [float(r28[m][c]) for c in ADM for m in D2]
    ari_max = max(float(a["ari"]) for a in load("exp8_p1_recovery.csv")); grp = sum(1 for m, d in e1.items() if d.get("grp_wordnet", 9) < d.get("grp_random", 0)); rho1 = max(p6.values())
    atxt = (f" Paired within each of the {len(ADM)} admissible configurations, the label-supervised ViT-B recovers more than each DINOv2 model in {wins} of {len(ADM)*len(D2)} comparisons; DINOv2 itself recovers the superclasses at ARI {min(d2v):.2f}--{max(d2v):.2f}, and the best cut over models and configurations reaches {ari_max:.2f} against zero for permuted labels. "
            f"Circularity controls: WordNet-coherent groupings read a lower raw $\\delta$ than random ones in {grp} of 12 models, and single-image distances still correlate with WordNet at up to ${rho1:+.2f}$.")
    ltxt = ""
    if L70:
        ltxt = f" The two leaf-label ViT-B/16 of Table~\\ref{{tab:q4-depth}}: the augreg recipe is aligned with WordNet like the IN-21k ViTs ($\\rho_{{\\text{{WN}}}}$ {float(L70['vit_b_in1k']['spearman_wn']):+.2f}) while DeiT-B is nearly unaligned ({float(L70['deit_b']['spearman_wn']):+.2f}) yet recovers the CIFAR-100 superclasses at ARI up to {float(L70['deit_b']['ari_max']):.2f}, so WordNet alignment is recipe-dependent even among leaf-supervised ViTs."
    T.newpart()
    # (c) DBpedia
    db = load("exp14_dbpedia.csv"); rec = {a["model"]: a for a in load("expR61_dbpedia_record.csv")} if ex("expR61_dbpedia_record.csv") else {}
    rows = []
    for a in db:
        nc = (float(a["NC_H"]) - float(a["NC_R"])) * 100; q = rec.get(a["model"])
        recc = f"{float(q['delta_999']):.3f} & {float(q['excess']):+.3f} & {int(q['r_above'])} ({float(q['p_left']):.3f})" if q else "--- & --- & ---"
        sup_exc = float(q['excess_haar_sup']) if (FINAL and q) else float(a['excess'])   # final: the supremum read against the Haar null of the record (expR61), not the superseded Gaussian one (exp14)
        rows.append(f"{NAME[a['model']]} & {float(a['delta']):.3f} & {sup_exc:+.3f} & {recc} & {float(a['trip_cos']):.2f} & {nc:+.2f} & {float(a['FS_HR_pp']):+.2f}$\\pm${float(a['FS_HR_ci']):.2f} \\\\")
    T.panel("(c) DBpedia Classes (219 leaf classes, 3 levels), a text hierarchy independent of WordNet; gains in pp.", "lcc|ccc|ccc",
            [(r" & \multicolumn{2}{c|}{supremum, Haar} & \multicolumn{3}{c|}{census of record} & & & \\" if FINAL else r" & \multicolumn{2}{c|}{supremum, Gaussian} & \multicolumn{3}{c|}{census of record} & & & \\"), r"model & $\hat\delta_{\max}$ & excess & $\hat\delta_{99.9}$ & excess & $r$ ($p$) & trip.\ & NC H$-$R & FS H$-$R \\"], rows, colsep="2.5pt")
    tm = ""
    if ex("exp27_dbpedia_treemap.json"):
        d27 = json.load(open(RES/"exp27_dbpedia_treemap.json")); l2 = [x for v in d27.values() for x in v["ari_l2"].values()]; cm = [v["cross_model"] for v in d27.values()]
        n_l2 = ""
        try:
            sup_ = np.load(Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))/"results/text_cache/dbpedia_bge_base.npz")["sup"]; n_l2 = f"{len(set(sup_.tolist()))}-way "
        except Exception: pass
        tm = f" Tree map on the same classes: no configuration degenerates (largest cluster $\\le{100*max(v['maxfrac'] for v in d27.values()):.0f}$\\%), the embedders recover the true {n_l2}level-2 partition at ARI {min(l2):.2f}--{max(l2):.2f} in all {len(d27)} configurations and agree with each other at cross-model ARI {min(cm):.2f}--{max(cm):.2f}; the configuration the criterion selects there (Euclidean, average linkage) is the one that degenerates on ImageNet, which is why the criterion is applied per dataset."
    # (c) HierarCaps
    h = load("exp16_hierarcaps.csv")
    rows = [f"{NAME[a['model']]} & {float(a['rho_mean']):+.2f}$\\pm${float(a['rho_sd']):.2f} & {100*float(a['pct_monotone']):.1f} & {float(a['trip_L1_cos']):.2f} & {float(a['trip_L2_cos']):.2f} & {float(a['delta_leaves']):.3f} & {float(a['excess']):+.3f} \\\\" for a in h]
    T.panel("(d) HierarCaps (1000 four-level caption chains): a multi-level text hierarchy with no class set.", "lcccccc", [r"model & $\rho$(level, radius) & \% monotone & trip.\ L1 & trip.\ L2 & $\hat\delta$ leaves & excess \\"], rows, size=r"\footnotesize", colsep="4pt")
    T.write(r"\textbf{Alignment with the human taxonomy follows supervision and recipe, the superclasses are recovered by every family, and the recovery is not WordNet circularity.} "
            r"(a) $\rho_{\text{WN}}$: Spearman correlation between inter-centroid and WordNet tree distances on ImageNet, with the label-shuffle and Gaussian-cloud controls; single image: the same correlation on single-image distances (one image per class); WordNet / random groups: raw $\hat\delta$ of WordNet-coherent against random class groupings. (b) ARI of the 20-cluster cut against the true CIFAR-100 superclasses per configuration (degenerate configurations marked; the criterion-selected one is cosine-complete)." + atxt + ltxt
            + r" % exp3_alignment.csv, exp28_recovery_per_config.csv, exp8_p1_recovery.csv, exp1_delta_controls.csv, exp8_p6_pooling.csv, expR70_inet1k_supervised.csv",
            contd=[r"(c) Excess under the original supremum reading (3 Gaussian replicates) and under the census of record (Haar null, 99.9th-percentile statistic, 200 replicates; $r$ = replicates above the real value, left-tail $p$), sibling-triplet agreement, and the metric gains, marginal as the diagnostic predicts." + tm
                   + r" (d) Radial ordering of the four levels under the paper's projection (chance for strict monotonicity $=1/24\approx4.2\%$; shuffle controls $\approx0$), sibling triplet agreement (cosine; chance $0.5$), and leaf $\hat\delta$ with spectrum-null excess. % exp14_dbpedia.csv, expR61_dbpedia_record.csv, exp27_dbpedia_treemap.json, exp16_hierarcaps.csv"])

# ======================================================================================================================
# Q2: the text census of record with template and extraction robustness
# ======================================================================================================================
def q_text():
    T = Table("tab_q02_text.tex", "tab:q2-text", colsep="2.5pt"); T.prov += ["expR53_text_haar_p999_200.csv", "expR53_text_haar_sup_200.csv", "expR53_text_gauss_p999_200.csv", "expR48b_text_census200_bs1.csv", "expR57_text_cosine_haar_p999_200.csv", "exp18_text_nulls.csv", "exp4_prompt_variation.csv", "exp17_wordnet_prompts.csv", "expR49_template_nulls_bs1.csv", "expR47_extraction_variance.csv"]
    rows_ = load("expR53_text_haar_p999_200.csv"); by = {a["model"]: a for a in rows_}
    tfour = [("Hp","expR53_text_haar_p999_200.csv"),("Hs","expR53_text_haar_sup_200.csv"),("Gp","expR53_text_gauss_p999_200.csv"),("Gs","expR48b_text_census200_bs1.csv")]
    TV = {}
    for tag, tf in tfour:
        if ex(tf):
            tr = load(tf); pb = bh([float(a["p_left"]) for a in tr]); TV[tag] = {a["model"]: pb[i] <= 0.05 for i, a in enumerate(tr)}
    def tcode(m): return "".join(("$\\bullet$" if TV[t].get(m, False) else "$\\circ$") for t, _ in tfour if t in TV)
    kind = {a["model"]: a for a in load("exp18_text_nulls.csv")}
    cos = {a["model"]: a for a in load("expR57_text_cosine_haar_p999_200.csv")} if ex("expR57_text_cosine_haar_p999_200.csv") else {}
    ORDER = ["gpt2","gpt2_m","gpt2_l","gpt2_xl","pythia_410m","pythia_1b","pythia_2b8","olmo_1b","olmo_7b","bge_base","bge_large","gte_base","gte_large","gte_qwen2","e5_base","e5_large"]
    rows = []
    for m in ORDER:
        a = by.get(m); k = kind.get(m)
        if a is None and k is None: continue
        ty = ("causal LM" if k and k["kind"] == "causal" else "embedder"); d = int(float(k["d"])) if k else "--"
        if a is None: rows.append(f"{NAME[m]} & {ty} & {d} & \\multicolumn{{8}}{{l}}{{not re-extracted (exceeds the local GPU)}} \\\\"); continue
        c = cos.get(m); cc = (f"${float(c['excess']):+.3f}" + ("" if gb(c) else r"^{\circ}") + f"$ & {int(c['r_above'])}") if c else "-- & --"
        rows.append(f"{NAME[m]} & {ty} & {d} & ${float(a['delta']):.3f}$ & ${float(a['excess']):+.3f}" + ("" if gb(a) else r"^{\circ}") + f"$ & ${100*float(a['excess'])/float(a['null_mean']):+.0f}\\%$ & {int(a['r_above'])} & {pfmt(float(a['p_left']))} & {tcode(m)} & {cc} \\\\")
    T.panel("(a) The census of record on the 1000 ImageNet class prompts: padding-free extraction, Haar null, 99.9th-percentile statistic, 200 replicates.", "llrccccccc|cc",
            [r" & & & \multicolumn{6}{c|}{record} & \multicolumn{2}{c}{cosine} \\", r"model & type & $d$ & $\hat\delta_{99.9}$ & excess & exc./null & $r$/200 & $p$ & 2$\times$2 & exc.\ & $r$ \\"], rows, mids=(4, 9), colsep="3.5pt")
    ngen = sum(gb(a) for a in rows_)
    # (b) templates
    rows4, rows17 = load("exp4_prompt_variation.csv"), load("exp17_wordnet_prompts.csv"); prompts = {}
    for r in rows4: prompts.setdefault(r["model"], {})[r["template"]] = float(r["delta_max"])
    for r in rows17: prompts.setdefault(r["model"], {})[r["template"]] = float(r["delta"])
    TPL = ["photo","name_only","image","closeup","this_is","wild","def_pair","gloss_only","concept","discussion"]
    f49 = RES/"expR49_template_nulls_bs1.csv"; t49 = {(a["model"], a["template"]): a for a in load("expR49_template_nulls_bs1.csv")} if f49.exists() else {}
    G4 = ["gpt2","gpt2_m","gpt2_l","gpt2_xl"]
    rows = []
    for t in TPL:
        cs = [f"{prompts[m][t]:.3f}" if t in prompts.get(m, {}) else "---" for m in ["bge_base","e5_base","gpt2","gpt2_m"]]
        for m in G4:
            a = t49.get((m, t)); cs.append("--" if a is None else f"${float(a['excess']):+.3f}$" + (r"\rlap{$^*$}" if abs(float(a["z"])) >= 2 else ""))
        for m in G4:
            a = t49.get((m, t)); cs.append("--" if a is None else f"${100*float(a['excess'])/float(a['null_mean']):+.0f}\\%$")
        rows.append(t.replace("_", r"\_") + " & " + " & ".join(cs) + r" \\")
    if t49:
        mean_ = [st.mean(float(t49[(m, t)]["excess"]) for t in TPL if (m, t) in t49) for m in G4]; nstar = [sum(abs(float(t49[(m, t)]["z"])) >= 2 for t in TPL if (m, t) in t49) for m in G4]; ntpl = [sum((m, t) in t49 for t in TPL) for m in G4]
        rows.append(r"\midrule mean over templates & & & & & " + " & ".join(f"${v:+.3f}$" for v in mean_) + " & " + " & ".join("" for _ in G4) + r" \\")
        rows.append(r"templates at $|z|\ge2$ & & & & & " + " & ".join(f"{a} of {b}" for a, b in zip(nstar, ntpl)) + " & " + " & ".join("" for _ in G4) + r" \\")
    T.panel("(b) Prompt templates: raw $\\hat\\delta$ per template and, for the four GPT-2 sizes, the template-conditioned excess over the spectrum null (3 replicates; $^*$: $|z|\\ge2$) and its fraction of the null.", "l" + "cccc" + "|cccc|cccc",
            [r" & \multicolumn{4}{c|}{raw $\hat\delta$} & \multicolumn{4}{c|}{GPT-2 excess (padding-free)} & \multicolumn{4}{c}{GPT-2 excess / null} \\", r"template & BGE-b & E5-b & GPT-2 S & GPT-2 M & S & M & L & XL & S & M & L & XL \\"], rows, colsep="1.8pt")
    allp = {m: list(v.values()) for m, v in prompts.items()}
    rng_lm = max(max(v) - min(v) for m, v in allp.items() if "gpt" in m); rng_emb = max(max(v) - min(v) for m, v in allp.items() if "gpt" not in m)
    # (c) extraction
    xt = ""
    if ex("expR47_extraction_variance.csv"):
        R = load("expR47_extraction_variance.csv"); rows = []
        for bs in (1, 16, 32):
            cs = []
            for m in ("gpt2_m", "olmo_1b"):
                for fp in (0, 1):
                    a = [r for r in R if r["model"] == m and int(r["fp16"]) == fp and int(r["batch"]) == bs]; cs.append(f"${float(a[0]['delta']):.4f}$" if a else "--")
            rows.append(f"{bs}{' (no padding)' if bs == 1 else ''} & " + " & ".join(cs) + r" \\")
        T.panel("(c) Extraction: the same 1000 prompts, only batch size and precision vary.", "lcc|cc", [r"& \multicolumn{2}{c|}{GPT-2 M} & \multicolumn{2}{c}{OLMo-1B} \\", r"batch size & fp32 $\hat\delta$ & fp16 $\hat\delta$ & fp32 $\hat\delta$ & fp16 $\hat\delta$ \\"], rows, size=r"\footnotesize", colsep="4pt")
        xt = r" (c) Precision is irrelevant (fp16 equals fp32 to four decimals); for GPT-2, left-padded batching changes the hidden states (mean cosine to the padding-free extraction $0.98$) and moves $\hat\delta$ by up to $0.018$, while OLMo handles left padding correctly and is invariant, which is why padding-free extraction is the record."
    s, m_ = by["gpt2"], by["gpt2_m"]
    T.write(r"\textbf{In text, clustered structure depends on recipe and scale: genuine at every Pythia scale and at the larger GPT-2 sizes, at the margin for GPT-2 S, absent for GPT-2 M and for the sentence embedders on this probe.} "
            r"(a) $r$ = replicates above the real value, $p$ the left-tail add-one $p$-value; genuine = BH-corrected $p\le0.05$ over the 15 models ($^{\circ}$: not genuine); exc./null: excess as a fraction of the null reading; the 2$\times$2 column gives the verdicts under Haar$\times$p99.9, Haar$\times$supremum, Gaussian$\times$p99.9, Gaussian$\times$supremum ($\bullet$ genuine); cosine: the same census on cosine geometry. "
            + f"Genuine {ngen}/15; GPT-2 S is genuine at $p={pfmt(float(s['p_left']))}$ with excess ${float(s['excess']):+.3f}$ and GPT-2 M is not ($p={pfmt(float(m_['p_left']))}$, ${float(m_['excess']):+.3f}$), while L and XL are genuine under every construction. GTE-base has the highest effective rank and is not genuine, so genuineness is not a proxy for anisotropy; the only LLM-backboned embedder (GTE-Qwen2-1.5B) is not genuine. "
            + f"(b) Causal-LM values are template-sensitive (range of raw $\\hat\\delta$ up to {rng_lm:.3f} for causal LMs against $\\le{rng_emb:.3f}$ for embedders; ``gloss only'' contains no class name) but the scale ordering is template-robust: with only 3 null replicates per cell the starred S and M cells are borderline, as the 200-replicate record in (a) shows, whereas the L/XL cells are not in doubt." + xt
            + r" % expR53_text_haar_p999_200.csv, expR53_text_haar_sup_200.csv, expR53_text_gauss_p999_200.csv, expR48b_text_census200_bs1.csv, expR57_text_cosine_haar_p999_200.csv, exp18_text_nulls.csv, exp4_prompt_variation.csv, exp17_wordnet_prompts.csv, expR49_template_nulls_bs1.csv, expR47_extraction_variance.csv")

# ======================================================================================================================
# Q13: local agreement, permutation-calibrated
# ======================================================================================================================
def q_local():
    T = Table("tab_q13_local.tex", "tab:q13-local", size=r"\footnotesize", colsep="3pt"); T.prov += ["exp21b_local_global_K200.csv"]
    R = load("exp21b_local_global_K200.csv"); assert len(R) == 66
    TAGS = [("knn_R","mutual-kNN ($k{=}10$), Euclidean"),("knn_H","mutual-kNN, Poincar\\'e"),("cka_R","linear CKA, Euclidean"),("cka_H","linear CKA, Poincar\\'e")]
    rows = []
    for tag, name in TAGS:
        g = lambda k: st.mean(float(a[f"{tag}_{k}"]) for a in R); fr = sum(float(a[f"{tag}_p"]) < 0.05 for a in R)
        rows.append(f"{name} & ${g('raw'):.3f}$ & ${g('null_mean'):.3f}$ & ${g('tau95'):.3f}$ & ${g('cal'):.3f}$ & ${g('nullcentered'):.3f}$ & {fr}/66 \\\\")
    T.panel("", "lcccccc", [r"measure & raw & null mean & $\tau_{0.05}$ & calibrated & null-centered & pairs $p<0.05$ \\"], rows)
    T.write(r"\textbf{Cross-model agreement survives permutation calibration.} Means over the 66 model pairs of the 12 vision backbones on the 1000 ImageNet class centroids (census cache). Calibration of \citet{groger2026aristotelian} with $K{=}200$ permutations of the class correspondence and $\alpha{=}0.05$, scalar (no layer search): $\tau_{0.05}$ is the $\lceil0.95(K{+}1)\rceil$-th order statistic of the observed score and its nulls (eq.\ 9), $p=(1+\#\{\text{null}\ge\text{obs}\})/(K{+}1)$ (eq.\ 10), calibrated $=\max\{(\text{obs}-\tau_{0.05})/(1-\tau_{0.05}),0\}$ (eq.\ 12); ``null-centered'' is the earlier raw-minus-null-mean variant, kept for continuity. The mKNN null mean equals the analytic chance level $k/(n{-}1)=10/999$. % exp21b_local_global_K200.csv")

# ======================================================================================================================
# Q9: the corollary — tasks, gains, correlations on raw and calibrated readings, policies
# ======================================================================================================================
def q_corollary():
    T = Table("tab_q09_corollary.tex", "tab:q9-corollary", colsep="2.6pt"); T.prov += ["table1_regenerated.csv", "exp2_metric_controls.csv", "exp2b_normalized_stack.csv", "exp13_mcnemar.csv", "night/correlation_cis.csv", "exp20_null_ztable.csv", "expR68_corollary_excess.csv", "exp24_val_metric_selection.csv", "night/t_sweep.csv"]
    t1 = {(r["model"], r["dataset"]): r for r in load("table1_regenerated.csv")}; e2 = {(r["model"], r["dataset"]): r for r in load("exp2_metric_controls.csv")}
    b2 = {(r["model"], r["dataset"]): r for r in load("exp2b_normalized_stack.csv")}; mc = {(r["model"], r["dataset"]): r for r in load("exp13_mcnemar.csv")}
    rows = []; mids = []
    for m in M10:
        for d in DS:
            r = t1[(m, d)]; e = e2[(m, d)]; p = float(mc[(m, d)]["p_H_vs_R"]); sign = "+" if float(r["NC_adv"]) > 0 else "$-$"
            ptxt = rf"$10^{{{max(-99, int(f'{p:.0e}'.split('e')[1]))}}}$" if p < 1e-3 else f"{p:.2f}"
            hn = f'{100*float(b2[(m, d)]["FS_HN_COS_diff"]):+.2f}' if (m, d) in b2 else "--"
            rows.append(f"{NAME[m] if d == 'imagenet' else ''} & {DSH[d]} & {100*float(r['NC_R']):.1f} & {100*float(r['NC_H']):.1f} & {100*float(r['FS_R']):.1f} & {100*float(r['FS_H']):.1f} & {(float(e['FS_COS'])-float(e['FS_R']))*100:+.2f} & {(float(e['FS_RT'])-float(e['FS_R']))*100:+.2f} & {hn} & {ptxt}\\,({sign}) \\\\")
        mids.append(len(rows))
    T.panel("(a) Task accuracies and zero-cost metric gains per cell (10 backbones with cached per-dataset features $\\times$ 6 datasets).", "llcccccccc",
            [r" & & \multicolumn{2}{c}{NC acc.\ (\%)} & \multicolumn{2}{c}{FS acc.\ (\%)} & \multicolumn{3}{c}{FS advantage over Euclidean (pp)} & McNemar \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-9}",
             r"model & data & Eucl. & Poinc. & Eucl. & Poinc. & cosine & radial map & Poinc.$_{\text{N}}$ $-$ cos & NC H vs R \\"], rows, mids=tuple(mids[:-1]), size=r"\scriptsize\renewcommand{\arraystretch}{0.88}", colsep="3pt")
    T.newpart()
    # (b) correlations: raw supremum delta (B5 + B12) and the calibrated reading (expR68)
    d20 = {(r["model"], r["dataset"]): float(r["delta"]) for r in load("exp20_null_ztable.csv")}
    ci = {(r["task"], r["dataset"]): r for r in load("night/correlation_cis.csv")}
    def bestadv(m, ds): a = e2[(m, ds)]; return 100*(max(float(a["FS_H"]), float(a["FS_COS"])) - float(a["FS_R"]))
    GN = {"NC_adv": "NC H$-$R", "FS_adv": "FS H$-$R", "FS_best_adv": "FS best$-$R"}
    GF = {"NC_adv": lambda m, ds: float(t1[(m, ds)]["NC_adv"]), "FS_adv": lambda m, ds: float(t1[(m, ds)]["FS_adv"]), "FS_best_adv": bestadv}
    C68 = {(r["predictor"], r["gain"], r["dataset"]): r for r in load("expR68_corollary_excess.csv")} if ex("expR68_corollary_excess.csv") else {}
    rows = []; rows2 = []; mids = []; mids2 = []
    for gname in GN:
        for ds in HIER:
            xs = [d20[(m, ds)] for m in M10]; ys = [GF[gname](m, ds) for m in M10]; fs = [fam(m) for m in M10]
            wf = {f: pear([x for x, g in zip(xs, fs) if g == f], [y for y, g in zip(ys, fs) if g == f]) for f in ["ViT", "DINOv2", "CLIP"]}
            c = ci.get((gname, ds)); cit = f"[{float(c['ci_lo']):+.2f}, {float(c['ci_hi']):+.2f}]" if c else "--"
            def c68(pred):
                r = C68.get((pred, gname, ds)); return (f"${float(r['r']):+.2f}$ [{float(r['ci_lo']):+.2f}, {float(r['ci_hi']):+.2f}] & ${float(r['r_demeaned']):+.2f}$" if r else "-- & --")
            rows.append(f"{GN[gname] if ds == 'imagenet' else ''} & {DSL[ds]} & ${pear(xs, ys):+.2f}$ {cit} & ${pear(demean(xs, fs), demean(ys, fs)):+.2f}$ & ${wf['ViT']:+.2f}$ & ${wf['DINOv2']:+.2f}$ & ({wf['CLIP']:+.0f}) \\\\")
            rows2.append(f"{GN[gname] if ds == 'imagenet' else ''} & {DSL[ds]} & {c68('raw')} & {c68('excess')} & " + (lambda r: f"${float(r['r']):+.2f}$ [{float(r['ci_lo']):+.2f}, {float(r['ci_hi']):+.2f}]" if r else "n/a")(C68.get(("depth_z", gname, ds))) + r" \\")
        c = ci.get((gname, "POOLED(cluster-bs)"))
        if c: rows.append(f" & pooled (cluster bootstrap) & ${float(c['r']):+.2f}$ [{float(c['ci_lo']):+.2f}, {float(c['ci_hi']):+.2f}] & -- & -- & -- & -- \\\\")
        mids.append(len(rows)); mids2.append(len(rows2))
    T.panel("(b) Correlation between the raw supremum $\\delta_{\\text{norm}}$ and the metric gain within each dataset ($n{=}10$ backbones), with bootstrap CI95 and the family-identity control (demeaned: both variables centered within family; within-family $r$ for ViT and DINOv2, sign only for CLIP).", "llccccc",
            [r"gain & dataset & $r$ [CI95] & demeaned & ViT & DINOv2 & CLIP \\"], rows, mids=tuple(mids[:-1]), colsep="4pt")
    CAL = (rows2, tuple(mids2[:-1]))
    # (c) best metric per backbone
    rows = []
    for m in M10:
        fsH = [100*(float(e2[(m, ds)]["FS_H"]) - float(e2[(m, ds)]["FS_R"])) for ds in HIER]; fsC = [100*(float(e2[(m, ds)]["FS_COS"]) - float(e2[(m, ds)]["FS_R"])) for ds in HIER]
        best = st.mean(max(h, c) for h, c in zip(fsH, fsC)); gap = st.mean(fsH) - st.mean(fsC); met = "H" if gap > 0.15 else ("cos" if gap < -0.15 else "either")
        rows.append(f"{NAME[m]} & ${best:+.2f}$ & {met} \\\\")
    T.panel("(c) Best zero-cost metric per backbone: few-shot advantage over Euclidean, mean over the four hierarchical datasets, and which metric collects it (either: within $0.15$pp).", "lcc", [r"model & best$-$R (pp) & metric \\"], rows, size=r"\footnotesize", colsep="5pt")
    T.newpart()
    T.panel("(d) The same correlations on the raw $\\hat\\delta_{99.9}$, the record excess and the depth-test $z$ (Fisher-$z$ CI95; dm: family-demeaned $r$).", "llcc|cc|c",
            [r" & & \multicolumn{2}{c|}{raw $\hat\delta_{99.9}$} & \multicolumn{2}{c|}{record excess} & depth $z$ \\", r"gain & dataset & $r$ [CI95] & dm & $r$ [CI95] & dm & $r$ [CI95] \\"], CAL[0], mids=CAL[1], colsep="3pt")
    # (d) policies
    r24 = load("exp24_val_metric_selection.csv"); HS = set(HIER); h24 = [r for r in r24 if r["dataset"] in HS]; FLAT = {"fashionmnist", "mnist"}
    def pol(rr, col): return st.mean(float(r[col]) for r in rr)
    def dgate(r): return 0.0 if (d20[(r["model"], r["dataset"])] >= 0.10 or r["dataset"] in FLAT) else float(r["adv_rule"])
    rows = [f"always cosine & {pol(r24,'adv_cos'):+.2f} & {pol(h24,'adv_cos'):+.2f} \\\\", f"objective rule (zero validation) & {pol(r24,'adv_rule'):+.2f} & {pol(h24,'adv_rule'):+.2f} \\\\",
            f"$\\delta$-gated objective rule & {st.mean(dgate(r) for r in r24):+.2f} & {st.mean(dgate(r) for r in h24):+.2f} \\\\", f"validation-picked per cell & {pol(r24,'adv_val'):+.2f} & {pol(h24,'adv_val'):+.2f} \\\\", f"test oracle (reference) & {pol(r24,'adv_oracle'):+.2f} & {pol(h24,'adv_oracle'):+.2f} \\\\"]
    T.panel("(e) Few-shot metric-selection policies on held-out episodes (500 validation / 500 test per cell).", "lcc", [r"policy (test advantage over Euclidean, pp) & all 60 cells & hierarchical (40) \\"], rows, size=r"\footnotesize", colsep="5pt")
    famp = lambda m: "ssl" if m.startswith("dinov") else "sup" if m.startswith("i21k") else "con"
    recs = []
    for (m, ds), r in e2.items():
        if (m, ds) not in d20: continue
        R_, H_, C_ = float(r["NC_R"]), float(r["NC_H"]), float(r["NC_COS"]); obj = C_ if famp(m) in ("ssl", "sup") else H_
        recs.append({"always cosine": (C_-R_)*100, "always Poincar\\'e": (H_-R_)*100, "gated rule (flat$\\to$R)": ((R_ if ds in FLAT else obj)-R_)*100, "$\\delta$-gated rule": ((R_ if (d20[(m, ds)] >= 0.10 or ds in FLAT) else obj)-R_)*100, "flat": ds in FLAT})
    rows = [f"{p} & {st.mean(x[p] for x in recs):+.2f} & {st.mean(x[p] for x in recs if x['flat']):+.2f} & {st.mean(x[p] for x in recs if not x['flat']):+.2f} \\\\" for p in ["always cosine", "always Poincar\\'e", "gated rule (flat$\\to$R)", "$\\delta$-gated rule"]]
    T.panel("(f) Nearest-centroid metric-selection policies.", "lccc", [r"NC policy (advantage over Euclidean, pp) & all 60 & flat (20) & hierarchical (40) \\"], rows, size=r"\footnotesize", colsep="5pt")
    ts = load("night/t_sweep.csv")
    rows = ["FS adv.\\ mean (pp) & " + " & ".join(f'{float(r["FS_adv_mean"]):+.2f}' for r in ts) + r" \\", "NC adv.\\ mean (pp) & " + " & ".join(f'{float(r["NC_adv_mean"]):+.2f}' for r in ts) + r" \\", r"\% cells FS $>0$ & " + " & ".join(f'{100*float(r["FS_pos_frac"]):.0f}' for r in ts) + r" \\"]
    T.panel("(g) Sensitivity to the projection target $t$ (36 hierarchical cells).", "l" + "c"*len(ts), ["target $t$ & " + " & ".join(f'{float(r["t"]):.2f}' for r in ts) + r" \\"], rows, size=r"\footnotesize", colsep="5pt")
    # caption numbers of the family control (B12)
    xs6 = lambda ds: [d20[(m, ds)] for m in M10 if not m.startswith("dinov2")]; ys6 = lambda ds: [float(t1[(m, ds)]["NC_adv"]) for m in M10 if not m.startswith("dinov2")]
    no_d2 = {ds: pear(xs6(ds), ys6(ds)) for ds in HIER}
    pts = [(d20[(m, d)], bestadv(m, d), fam(m), d) for m in M10 for d in DS]; ph = [p for p in pts if p[3] in HS]
    pooled_all = pear([p[0] for p in pts], [p[1] for p in pts]); pooled_dm = pear(demean([p[0] for p in pts], [p[2] for p in pts]), demean([p[1] for p in pts], [p[2] for p in pts]))
    pooled_h = pear([p[0] for p in ph], [p[1] for p in ph]); pooled_hdm = pear(demean([p[0] for p in ph], [p[2] for p in ph]), demean([p[1] for p in ph], [p[2] for p in ph]))
    surv = ""
    if C68:
        nc_ok = [DSL[ds] for ds in HIER if float(C68[("excess","NC_adv",ds)]["ci_hi"]) < 0]; fs_ok = [DSL[ds] for ds in HIER if float(C68[("excess","FS_adv",ds)]["ci_hi"]) < 0]
        dz = any(float(r["ci_hi"]) < 0 or float(r["ci_lo"]) > 0 for (p, g, d), r in C68.items() if p == "depth_z" and d != "pooled")
        surv = f" On the record excess the nearest-centroid prediction survives (CI95 excluding zero) on {', '.join(nc_ok)} and the few-shot prediction on {', '.join(fs_ok)}; the depth verdict predicts {'one gain' if dz else 'no gain'} in any dataset."
    T.write(r"\textbf{The calibrated reading predicts the zero-cost gain within datasets, the depth verdict predicts nothing, and cosine is the safe default.} "
            r"(a) Nearest-centroid (NC) and 5-way 5-shot (FS, 1000 paired episodes) accuracy under Euclidean and Poincar\'e readouts, regenerated from the audited reruns; FS advantage of cosine and of the radial map with Euclidean distances (the map without the metric changes nothing) over Euclidean, and of the Poincar\'e readout over cosine after L2 normalization; McNemar $p$ for the NC test-set decisions, Poincar\'e vs Euclidean (order of magnitude when $p<10^{-3}$; sign of the NC advantage in parentheses). % table1_regenerated.csv, exp2_metric_controls.csv, exp2b_normalized_stack.csv, exp13_mcnemar.csv",
            contd=[r"(b) Within-dataset correlations of the raw supremum reading; bootstrap CI95 from 10k resamples of the 10 models, pooled rows by cluster bootstrap over datasets; family control: both variables centered within family before correlating, and within-family correlations (ViT, DINOv2: 4 models each; CLIP has two, so only its sign is shown). "
                   + f"Without DINOv2 (6 backbones) the raw NC correlation holds on ImageNet and CIFAR-100 (${no_d2['imagenet']:+.2f}$/${no_d2['cifar100']:+.2f}$) but not on CIFAR-10/DTD (${no_d2['cifar10']:+.2f}$/${no_d2['dtd']:+.2f}$); pooled cross-dataset correlations of the best-metric advantage do not survive the family control (${pooled_all:+.2f}\\to{pooled_dm:+.2f}$ over all 60 cells, ${pooled_h:+.2f}\\to{pooled_hdm:+.2f}$ on hierarchical cells) and are family-driven." + surv
                   + r" (c) The gain per backbone and which readout collects it. % night/correlation_cis.csv, exp20_null_ztable.csv, exp2_metric_controls.csv",
                   r"(d) The calibrated reading: the raw 99.9th-percentile statistic, the record excess and the depth-test $z$ of Table~\ref{tab:q4-depth} as predictors. (e) Validation picking retains the oracle advantage and the zero-validation objective rule modestly beats always-cosine. (f) The gate's value is avoiding the Poincar\'e projection on flat-label data; cosine is a safe NC default everywhere. (g) The headline $t=1/\sqrt2\approx0.71$ was fixed a priori and is conservative for FS. % expR68_corollary_excess.csv, exp24_val_metric_selection.csv, night/t_sweep.csv"])

# ======================================================================================================================
# Q12: the parallelogram defect xi
# ======================================================================================================================
def q_xi():
    T = Table("tab_q12_xi.tex", "tab:q12-xi", size=r"\footnotesize", colsep="4pt"); T.prov += ["exp12_curvature_sign.csv", "exp26_xi_nulls.csv", "expR38_xi_angular.csv", "expR43_xi_norm_references.csv"]
    cv = load("exp12_curvature_sign.csv"); e26 = {r["model"]: r for r in load("exp26_xi_nulls.csv")}
    REF = {"ref_tree_d9":"balanced tree (depth 9)","ref_H2_R6":"$\\mathbb{H}^2$ region ($R{=}6$)","ref_gauss768":"iid Gaussian ($d{=}768$)","ref_sphere_geo":"$S^{d}$ (geodesic)","ref_sphere_chord":"$S^{d}$ (chord)"}
    rows = []; mids = []
    for r in cv:
        nm = REF.get(r["model"], NAME.get(r["model"], r["model"]))
        if r["model"] in e26:
            x = e26[r["model"]]; tail = f' & {float(x["xi_null"]):+.3f}$\\pm${float(x["xi_null_sd"]):.3f} & {float(x["excess"]):+.3f}$\\pm${float(x["ci95"]):.3f} & {float(x["z"]):+.0f}'
        else: tail = " & --- & --- & ---"
        rows.append(f'{nm} & {float(r["xi_mean"]):+.3f} & {100*float(r["frac_neg"]):.0f}\\%{tail} \\\\')
        if r["model"] == "ref_sphere_chord": mids.append(len(rows))
    T.panel("(a) $\\xi$ on reference geometries and ImageNet centroids, with the spectrum-matched null (20 replicates).", "lccccc", [r"space / model & $\xi$ (mean) & frac.\ negative & null $\xi$ & excess ($\pm$CI95) & $z$ \\"], rows, mids=tuple(mids))
    if ex("expR38_xi_angular.csv"):
        by = {a["model"]: a for a in load("expR38_xi_angular.csv")}
        rows = [f"{NAME[m]} & ${float(by[m]['xi_geo']):+.3f}$ & ${float(by[m]['frac_neg']):.2f}$ & ${float(by[m]['excess']):+.4f}$ & ${float(by[m]['z']):+.1f}$ \\\\" if m in by else NAME[m] + r" & -- & -- & -- & -- \\" for m in M12]
        T.panel("(b) $\\xi$ on the angular geometry: centroids L2-normalized, spherical geodesic distances, null on the normalized cloud.", "lcccc", [r"model & $\xi_{\text{geo}}$ & frac.\ neg.\ & excess & $z$ \\"], rows, mids=(4, 9))
    if ex("expR43_xi_norm_references.csv"):
        R = load("expR43_xi_norm_references.csv"); RN = {"gaussian":"iid Gaussian","gauss_lognorm_r0.3":"Gaussian, lognormal radii ($\\sigma{=}0.3$)","gauss_lognorm_r0.6":"Gaussian, lognormal radii ($\\sigma{=}0.6$)","gauss_lognorm_r1.0":"Gaussian, lognormal radii ($\\sigma{=}1.0$)","core_shell":"core + shell mixture","sphere":"uniform sphere","sphere_radial_jitter":"sphere with radial jitter"}
        rows = []
        for k in RN:
            sub = [a for a in R if a["reference"] == k]
            if sub: rows.append(f"{RN[k]} & ${st.mean(float(a['xi']) for a in sub):+.4f}$ & ${st.mean(float(a['frac_neg']) for a in sub):.2f}$ \\\\")
        T.panel("(c) $\\xi$ on references with heterogeneous norms but no curvature ($n{=}1000$, $d{=}768$, 5 seeds).", "lcc", [r"reference (no curvature unless stated) & $\xi$ & frac.\ negative \\"], rows)
    T.write(r"\textbf{$\xi$ reads norm structure, not curvature.} (a) Matched nulls have positive $\xi$ (noise combines null and estimator s.d.), so raw readings understate curvature. (b) The deep negative $\xi$-excess of DINOv2 on ImageNet is a norm-structure effect for S/B/L, whose angular $\xi$-excess turns sphere-leaning positive, while DINOv2-G and CLIP-B/L remain negative on the sphere. (c) Norm dispersion alone drives $\xi$ negative (lognormal radii, radial jitter on a sphere, core+shell), so the estimator is reported only as a norm-structure descriptor. % exp12_curvature_sign.csv, exp26_xi_nulls.csv, expR38_xi_angular.csv, expR43_xi_norm_references.csv")

# ======================================================================================================================
# Q14: the model panel and extraction
# ======================================================================================================================
def q_panel():
    T = Table("tab_q14_panel.tex", "tab:q14-panel", size=r"\scriptsize", colsep="4pt"); T.prov += ["(model panel: stated in the generator; parameter counts from the model cards)"]
    P = [("Supervised ViT (vision)", ["ViT-T (i21k) & 5M", "ViT-S (i21k) & 22M", "ViT-B (i21k) & 86M", "ViT-L (i21k) & 307M"]),
         ("SSL (vision)", ["DINO-B & 86M", "DINOv2-S & 22M", "DINOv2-B & 86M", "DINOv2-L & 307M", "DINOv2-G & 1.1B"]),
         ("Contrastive (vision)", ["CLIP-B & 86M", "CLIP-L & 307M", "SigLIP-B & 86M"]),
         ("Causal LM", ["GPT-2 S & 117M", "GPT-2 M & 345M", "GPT-2 L & 774M", "GPT-2 XL & 1.5B", "Pythia-410M & 410M", "Pythia-1B & 1.0B", "Pythia-2.8B & 2.8B", "OLMo-1B & 1.0B", "OLMo-7B & 7.0B"]),
         ("Text embedder", ["BGE-base & 110M", "BGE-large & 335M", "GTE-base & 110M", "GTE-large & 335M", "GTE-Qwen2-1.5B & 1.5B", "E5-base & 110M", "E5-large & 335M"]),
         ("Controls (vision)", ["DeiT-B/16 (IN-1k, no distillation) & 86M", "ViT-B/16 augreg (IN-1k) & 86M", "MERU ViT-S/B/L and CLIP twins & 22M/86M/307M", "Barlow Twins, BYOL (ResNet-50) & 24M"])]
    rows = []; mids = []
    for ty, ms in P:
        for x in ms: rows.append(f"{ty} & {x} \\\\")
        mids.append(len(rows))
    T.panel("", "llr", [r"type & model & params \\"], rows, mids=(mids[2], mids[4]))
    T.write(r"\textbf{The model panel.} 12 vision backbones (all in the census; the 10 with cached per-dataset features, all but DINO-B and SigLIP-B, form the task grid), 9 causal LMs, 7 text embedders, and the vision models used only as controls. "
            r"Class centroids average $100$ cached training images per class on ImageNet and the full cached split, $80$--$6{,}000$ images per class, on the other datasets. Feature extraction: vision backbones use the encoder's default pooled embedding (timm, \texttt{num\_classes=0}, CLS or global pool per architecture); causal LMs use the last-token hidden state, extracted one prompt at a time (padding-free: left-padded batching alters GPT-2's hidden states, Table~\ref{tab:q2-text}); text embedders use their recommended pooling (CLS for BGE, mean for GTE/E5, last token for GTE-Qwen2); CLIP/SigLIP text towers use their text encoders.")

# ======================================================================================================================
def conservation_check(verbose=True):
    """Every decimal token of the old appendix (snapshot) survives in the new appendix tables or the main text, except Table 13
    (the 3-replicate original text extraction, removed by the brief) and the 'original extraction' columns of the old Table 14."""
    OLD = RES / "final_pass_old_appendix"
    order = [s for s in open(OLD / "INPUT_ORDER.txt").read().split() if s not in ("tab_b1_textnulls", "tab_b39_finetune")]   # B39 (the two-pass control of expR65) is superseded by the trained positive control of expR77 (closing pass, 2026-09-22)
    old_tokens = {}
    for s in order:
        t = open(OLD / f"{s}.tex").read(); t = re.sub(r"(?m)(?<!\\)%.*$", "", t)
        if s == "tab_b23_text20":
            t = re.sub(r" & \$[-+0-9.]+\$ & \$[-+0-9.]+\$ \\\\", r" \\\\", t); t = re.sub(r"Three verdicts change.*?not used as evidence\)\.", "", t, flags=re.S); t = re.sub(r"Sign agreement with the 3-replicate original: \d+/\d+\.", "", t)
        for tok in re.findall(r"(?<![\w.])[-+]?\d+\.\d+", t): old_tokens.setdefault(tok, set()).add(s)
    new_all = "".join(open(f).read() for f in sorted(OUT.glob("tab_q*.tex")) if "final" not in f.name) + open(HERE / "tab_census.tex").read()
    if (HERE / "main_iclr2027.tex").exists(): new_all += open(HERE / "main_iclr2027.tex").read()
    new_nums = [float(x) for x in re.findall(r"(?<![\w.])[-+]?\d+\.\d+|(?<![\w.])\d+(?![\w.])", new_all.replace("{=}", "="))]
    missing = []
    for tok, srcs in sorted(old_tokens.items()):
        v = float(tok); prec = len(tok.split(".")[1]); tol = 0.5 * 10 ** (-prec) + 1e-12
        if tok in new_all: continue
        if any(abs(abs(v) - abs(u)) <= tol or (abs(v) > 0 and abs(round(u, prec) - abs(v)) <= tol) for u in new_nums): continue
        missing.append((tok, sorted(srcs)))
    if verbose:
        for tok, srcs in missing: print("   NUMBER LOST:", tok, srcs)
        print(f"conservation: {len(old_tokens)-len(missing)}/{len(old_tokens)} old decimal tokens found")
    return missing


# ======================================================================================================================
# Q8 (final version): robustness = budget, resampling, joint sensitivity, the two cited class-count rows and the training rows of S5.2
# ======================================================================================================================
def q_robust_final():
    (OUT / "final").mkdir(exist_ok=True)   # the final version's copies of the appendix tables live in appendix_tables/final/ (phaseE_submission.py fills the rest)
    T = Table("final/tab_q08_robust_final.tex", "tab:q8-robust", colsep="2.4pt"); T.prov += ["expR72_budget_record.csv", "expR72_budget_record_summary.csv", "expR59_imagenet_bootstrap_summary.csv", "expR73_transfer_bootstrap_record_summary.csv", "expR66_joint_sensitivity_summary.csv", "expR60_c_sweep_record.csv", "analysis4_finetuning.csv", "e1_delta_by_layer.csv"]
    NM = {"i21k_l":"ViT-L","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B"}; DSS = {"imagenet":"IN","cifar100":"C100","dtd":"DTD"}
    CELLS = [("i21k_l","imagenet"),("dinov2_l","imagenet"),("clip_b","imagenet"),("i21k_l","cifar100"),("dinov2_l","cifar100"),("clip_b","cifar100"),("i21k_l","dtd"),("dinov2_l","dtd"),("clip_b","dtd")]
    B72 = {(a["model"], a["dataset"]): a for a in load("expR72_budget_record_summary.csv")}; D72 = load("expR72_budget_record.csv")
    BUD = [10000, 50000, 100000, 500000, 1000000, 2000000]; rows = []
    for (m, ds) in CELLS:
        rec_ = [a for a in D72 if a["model"] == m and a["dataset"] == ds]
        if rec_:
            v = {int(a["n_quads"]): float(a["excess"]) for a in rec_}; s = B72[(m, ds)]
            rows.append(f"{NM[m]} & {DSS[ds]} & " + " & ".join(f"${v[b]:+.4f}$" for b in BUD) + f" & ${float(s['drift_ge1e5']):.4f}$ & {float(s['drift_ge1e5_over_sd']):.2f} \\\\")
    T.panel("(a) Quadruple budget under the record (Haar null, p99.9, 200 replicates): the excess of 9 cells at $10^4$ to $2{\\times}10^6$ sampled quadruples per seed; drift = range of the excess over budgets $\\ge10^5$, also divided by the excess's null spread.", "llcccccc|cc",
            [r"model & data & $10^4$ & $5{\times}10^4$ & $10^5$ & $5{\times}10^5$ & $10^6$ & $2{\times}10^6$ & drift ($\ge10^5$) & drift/s.d. \\"], rows, colsep="2.6pt")
    B = [dict(a, dataset="imagenet") for a in load("expR59_imagenet_bootstrap_summary.csv")]; B73 = load("expR73_transfer_bootstrap_record_summary.csv") if ex("expR73_transfer_bootstrap_record_summary.csv") else []
    B = B + [a for a in B73 if a["dataset"] in ("cifar100", "dtd")]; DSN = {"imagenet": ("ImageNet", 100, 20), "cifar100": ("CIFAR-100", 500, 200), "dtd": ("DTD", 80, 200)}
    rows = []
    for ds in ("imagenet", "cifar100", "dtd"):
        rows += [f"{NAME.get(a['model'], a['model'])} & {DSN[ds][0]} & {DSN[ds][1]} & {DSN[ds][2]} & ${float(a['excess_ref']):+.4f}$ & ${float(a['excess_boot_mean']):+.4f}$ & ${float(a['excess_boot_sd']):.4f}$ & {float(a['frac_boot_negative']):.2f} \\\\" for a in B if a["dataset"] == ds]
    T.panel("(b) Centroid bootstrap under the record (Haar null, p99.9): 30 resamples with replacement of the cached training images per class at the census budget; ImageNet with 20 null replicates per resample" + (", CIFAR-100 and DTD with 200." if B73 else "; the transfer sets follow once expR73 completes."), "llcccccc", [r"model & dataset & images/class & null rep. & excess (reference) & bootstrap mean & bootstrap s.d. & fraction negative \\"], rows, mids=(12, 24), colsep="3pt")
    bmax = f"{max(float(a['excess_boot_sd']) for a in B):.4f}"; nds = len({a["dataset"] for a in B})
    fneg = min((float(a["frac_boot_negative"]) if float(a["excess_ref"]) < 0 else 1 - float(a["frac_boot_negative"])) for a in B)   # agreement of the resample sign with the record's sign, worst cell
    T.newpart()
    JS_FILE = "expR66c_joint_sensitivity_summary.csv" if ex("expR66c_joint_sensitivity_summary.csv") else "expR66_joint_sensitivity_summary.csv"   # final: regenerated under the centered record (expR66c)
    if JS_FILE.startswith("expR66c"): T.prov.append(JS_FILE)
    J = {(r["model"], r["dataset"]): r for r in load(JS_FILE)}; rows = []
    for m in M12:
        cs = []
        for d in DS:
            r = J[(m, d)]; g = r["genuine_bh_record"] == "True"; jg = r["joint_genuine"] == "True"; bg = r["boot_bh_genuine"] == "True"
            cs.append(f"${float(r['z_joint']):+.1f}$ ({r['n_boot_genuine']})" + ("" if g else r"$^{\circ}$") + (r"$^{\dagger}$" if (g and not (jg and bg)) else "") + (r"$^{\ddagger}$" if (not g and (jg or bg)) else ""))
        rows.append(NAME[m] + " & " + " & ".join(cs) + r" \\")
    T.panel("(c) Joint sensitivity of the genuine count: $z_{\\text{joint}}$ against null, bootstrap and estimator noise, and (in parentheses) the number of the 30 centroid resamples in which the cell is genuine.", "l" + "c"*6, ["model & " + " & ".join(DSH[d] for d in DS) + r" \\"], rows, mids=(4, 9), colsep="2.6pt")
    tot = lambda k: sum(1 for r in J.values() if r[k] == "True"); tot2 = lambda k: sum(1 for (m, d), r in J.items() if r[k] == "True" and d in TOP)
    jt = f" Counts: record {tot('genuine_bh_record')}/72 ({tot2('genuine_bh_record')}/24); $z_{{\\text{{joint}}}}\\le-2$: {tot('joint_genuine')}/72 ({tot2('joint_genuine')}/24); genuine in $\\ge27$ of 30 resamples: {tot('boot_bh_genuine')}/72 ({tot2('boot_bh_genuine')}/24)."
    c60 = load("expR60_c_sweep_record.csv")
    def agg(m, C, mode, f):
        v = [float(r[f]) for r in c60 if r["model"] == m and int(r["C"]) == C and r["mode"] == mode and r[f] not in ("", "nan")]; return st.mean(v) if v else None
    def below(m, C, mode):
        v = [float(r["p_left"]) <= 0.05 for r in c60 if r["model"] == m and int(r["C"]) == C and r["mode"] == mode]; return f"{sum(v)}/{len(v)}" if v else "---"
    rows = []
    for m in ["dinov2_l"]:
        for C in sorted({int(r["C"]) for r in c60}):
            dr, er, nr = agg(m,C,"random","delta_999"), agg(m,C,"random","excess"), agg(m,C,"random","nc_adv_pp"); dc, ec, nc = agg(m,C,"coherent","delta_999"), agg(m,C,"coherent","excess"), agg(m,C,"coherent","nc_adv_pp")
            if dr is None: continue
            right = f"{dc:.3f} & {ec:+.3f} & {below(m,C,'coherent')} & {nc:+.2f}" if dc is not None else "--- & --- & --- & ---"
            rows.append(f"{NAME[m]} & {C} & {dr:.3f} & {er:+.3f} & {below(m,C,'random')} & {nr:+.2f} & {right} \\\\")
    T.panel("(d) Class count: ImageNet subsets of $C$ classes, random (spanning the hierarchy) or WordNet-coherent (siblings, effectively flat), means over the subset seeds. The excess shrinks with the number of classes for coherent and random subsets alike, so magnitudes are not comparable across class counts; the verdict is what carries across datasets.", "lr|cccc|cccc",
            [r" & & \multicolumn{4}{c|}{random subsets (span the hierarchy)} & \multicolumn{4}{c}{WordNet-coherent (siblings)} \\", r"model & $C$ & $\hat\delta_{99.9}$ & excess & below & NC adv & $\hat\delta_{99.9}$ & excess & below & NC adv \\"], rows, size=r"\footnotesize", colsep="3.5pt")
    NMI = {"dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","i21k_b":"ViT-B","clip_b":"CLIP-B","clip_b_vision":"CLIP-B"}; rows = []
    if (ABL/"analysis4_finetuning.csv").exists():
        ft = list(csv.DictReader(open(ABL/"analysis4_finetuning.csv"))); a = next((x for x in ft if x["model"] == "dinov2_b"), ft[0])
        rows.append(f"non-hierarchical fine-tuning & {NMI.get(a['model'], a['model'])} & ${float(a['delta_before']):.3f}$ & ${float(a['delta_after']):.3f}$ & ${float(a['pct_change']):+.0f}\\%$ \\\\")
    if (ABL/"e1_delta_by_layer.csv").exists():
        L = list(csv.DictReader(open(ABL/"e1_delta_by_layer.csv")))
        for m in ("dinov2_b",):
            rr = sorted([a for a in L if a["model"] == m and int(a["layer"]) >= 1], key=lambda a: int(a["layer"]))
            if rr: rows.append(f"across depth (layer {rr[0]['layer']} $\\to$ {rr[-1]['layer']}) & {NMI[m]} & ${float(rr[0]['delta_normalized']):.3f}$ & ${float(rr[-1]['delta_normalized']):.3f}$ & ${100*(float(rr[-1]['delta_normalized'])-float(rr[0]['delta_normalized']))/float(rr[0]['delta_normalized']):+.0f}\\%$ \\\\")
    T.panel("(e) Training moves the raw reading within an architecture: one row per intervention. Statistic: the raw supremum $\\delta_{\\text{norm}}$, the largest four-point defect over sampled quadruples divided by the diameter, on 512 sampled points; fine-tuning row: 1000 CIFAR-100 test images before and after a non-hierarchical (colour) fine-tuning of DINOv2-S, $5{\\times}10^4$ quadruples; depth row: DINOv2-B hidden states per layer.", "llccc", [r"intervention & model & $\delta$ before & $\delta$ after & change \\"], rows, size=r"\footnotesize", colsep="4pt")   # seventh review: the statistic stated
    ctxt = ""
    _cb = r"(b) Bootstrap of the excess over resampled images per class on " + ("ImageNet, CIFAR-100 and DTD" if nds == 3 else "ImageNet") + r": the bootstrap s.d.\ is at most " + bmax + (r" and every resample of every cell keeps the sign of the record excess." if fneg == 1.0 else f" and every cell keeps the sign of its record excess in at least {fneg:.2f} of its resamples.")   # (b) goes to its own continued caption (twelfth review, 2026-09-23)
    T.write(r"\textbf{The excess is budget-stable, the genuine count survives resampling and estimator noise, the excess shrinks with the class count, and training moves it.} "
            r"(a) Excess against the quadruple budget for 9 cells; drift = range of the record excess over budgets $\ge10^5$, divided by the excess's spread in the last column; under the record every sign holds at every budget. " + _cb + r" % expR72_budget_record.csv, expR59_imagenet_bootstrap_summary.csv, expR73_transfer_bootstrap_record_summary.csv",
            contd=[r"(c) Per cell: $z_{\text{joint}} = \text{excess}/\sqrt{\sigma_{\text{null}}^2+\sigma_{\text{boot}}^2+\sigma_{\text{est}}^2}$ and the number of resamples in which the cell is genuine under BH over the 72 cells; $^{\circ}$: not genuine in the record; $^{\dagger}$: record-genuine but failing one of the two criteria ($z_{\text{joint}}\le-2$, genuine in $\ge27$ of 30); $^{\ddagger}$: not record-genuine but passing one." + jt
                   + r" (d) Class-count control under the record (means over subset seeds): the excess shrinks with the number of classes for coherent and random subsets alike, so magnitudes are not comparable across class counts and the verdict is what carries across datasets; ``below'' = subset seeds whose real value lies below the null at uncorrected $p\le0.05$; NC adv = prototype-classifier advantage of the Poincar\'{e} readout in pp. (e) Raw $\delta$ within architecture: fine-tuning on a task without class hierarchy raises it and it falls across transformer depth. % expR66_joint_sensitivity_summary.csv, expR60_c_sweep_record.csv, analysis4_finetuning.csv, e1_delta_by_layer.csv" + ctxt])

def main_khrulkov():
    """Main-text table of the Khrulkov replication (full-pass cleanup, 2026-09-22): the four rows of expR78 beside the S5.1 paragraph."""
    S = pd.read_csv(RES / "expR78_khrulkov_replication_summary.csv").set_index("dataset"); NMD = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cub": "CUB-200", "miniimagenet": "MiniImageNet"}
    S85 = pd.read_csv(RES / "expR85_khrulkov_sup_summary.csv").set_index("dataset") if ex("expR85_khrulkov_sup_summary.csv") else None   # twelfth review (5): their own statistic, the supremum, against the same replicates
    if S85 is not None: assert set(S85.index) == set(S.index) and bool((S85.n_trials == 10).all()), S85[["n_trials"]]
    sup = (lambda d: f" & ${S85.loc[d, 'excess_sup_mean']:+.4f}$ & {int(round(S85.loc[d, 'r_above_mean']))}/200, {S85.loc[d, 'p_left_max']:.3f}") if S85 is not None else (lambda d: "")
    rows = [f"{NMD[d]} & {S.loc[d, 'theirs']:.2f} & {S.loc[d, 'ours_raw_mean']:.3f} & ${S.loc[d, 'excess_mean']:+.4f}$ & {int(round(S.loc[d, 'r_above_mean']))}/200, {S.loc[d, 'p_left_max']:.3f}" + sup(d) + " \\\\" for d in ("cifar10", "cifar100", "cub", "miniimagenet")]
    hdr = (r"dataset & their $\delta_{\text{rel}}$ & ours, their estimator & excess & $r$/200, largest $p$ & excess (sup.) & $r$/200, largest $p$ (sup.) \\" if S85 is not None else r"dataset & their $\delta_{\text{rel}}$ & ours, their estimator & excess & $r$/200, largest $p$ \\")
    cap = (r"\caption{\textbf{A published reading reproduced and calibrated.} ResNet-34 setting of \citet{Khrulkov_2020_CVPR}: their relative hyperbolicity, ours with their estimator, then the excess and the rank of 200 with the largest left-tail $p$ for the record statistic (99.9th percentile) and for their own statistic, the supremum (sup.), against the same replicates. % expR78_khrulkov_replication_summary.csv, expR85_khrulkov_sup_summary.csv" if S85 is not None
           else r"\caption{\textbf{A published reading reproduced and calibrated.} ResNet-34 setting of \citet{Khrulkov_2020_CVPR}: their relative hyperbolicity, ours with their estimator, the excess, and the rank of 200 with the largest left-tail $p$. % expR78_khrulkov_replication_summary.csv")
    L = ["% prov: expR78_khrulkov_replication_summary.csv" + (", expR85_khrulkov_sup_summary.csv" if S85 is not None else ""), r"\begin{table}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{" + ("3.5pt" if S85 is not None else "5pt") + "}", r"\begin{tabular}{" + ("lcccccc" if S85 is not None else "lcccc") + "}", r"\toprule", hdr, r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", cap, "}", r"\label{tab:khrulkov}", r"\end{table}"]
    (HERE / "tab_khrulkov_final.tex").write_text("\n".join(L) + "\n"); print("wrote tab_khrulkov_final.tex from expR78_khrulkov_replication_summary.csv")

if __name__ == "__main__":
    for f in OUT.glob("tab_q*.tex"): f.unlink()
    q_calibration(); q_census(); q_robust(); q_sample(); q_depth(); q_power(); q_interventions(); q_treemap(); q_wordnet(); q_text(); q_local(); q_corollary(); q_xi(); q_panel(); q_robust_final()
    conservation_check()
    FINAL = True; q_depth(); q_wordnet(); q_power(); q_census(); q_sample(); main_khrulkov()   # the final version's copies (two-decimal z and the K sweep; DBpedia supremum under the Haar null; power table with two-decimal z)
