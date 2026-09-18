#!/usr/bin/env python3
"""Final version (author's brief 'Final version — plain, short, nine pages', 2026-09-18): fills phaseE_paper_final.tex.tmpl and writes
ICLR2027/iclr2027/main_iclr2027_final.tex. Verbatim from the author's main_local.tex: abstract, S1 with Figure 1, S2, the 'Gromov delta'
and 'Estimation and normalization' paragraphs, with the named edits the brief asks for (metaphors and the bridge sentence removed);
the edits are saved to rebuttal/results/final_verbatim_edits.json for the sweep. Every number in the prose is a fill read from the
result files (phaseE_fills.json + phaseE_v3_fills.json + the Figure 5 values). The appendix keeps the question tables the main text
cites, without figures, in citation order. Cuts for the page budget are applied in the brief's order through FINAL_CUTS (comma-separated
steps among: s55, s6, table, s2). Run from the repo root after phaseE_final.py, phaseE_v3.py, gen_appendix_final.py and make_figs_final.py."""
import json, re, os, sys, importlib
import numpy as np, pandas as pd
R = 'rebuttal/results/'; S = 'rebuttal/scripts/'; TEX = 'ICLR2027/iclr2027/'; P1 = TEX + 'main_iclr2027.tex'; PF = TEX + 'main_iclr2027_final.tex'
import numpy.core as _c; sys.modules.setdefault("numpy._core", _c)
for s_ in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core." + s_, importlib.import_module("numpy.core." + s_))
    except Exception: pass
F = json.load(open(R + 'phaseE_fills.json')); F.update(json.load(open(R + 'phaseE_v3_fills.json')))
v5 = json.load(open(R + 'final_fig5_values.json')); F['NAIVE_BIG'] = f"{v5['naive_dinov2_vs_block']:.2f}"; assert F['ARI_BIG'] == f"{v5['selected_dinov2_vs_block']:.2f}"
# ---- the words of the final prose, checked against the files they summarize
sl = pd.read_csv(R + 'expR62_samplelevel_record.csv'); assert int((~sl.genuine_bh).sum()) > 12, "'not genuine in most cells' (S5.1)"
assert 100 * abs((sl.excess / sl.null_mean)[sl.genuine_bh]).max() <= 40, "'at most two fifths of the null reading' (S5.1)"
c52 = pd.read_csv(R + 'expR52_census_haar_p999_200.csv'); assert int((c52.excess < 0).sum()) >= 68, "'negative in almost every cell' (S5.2)"
assert int(F['AGREE']) >= 60, "'agrees in almost every cell' (S5.4, cosine census)"
D9 = pd.read_csv(R + 'expR64b_wn30.csv'); dep = D9[(D9.kind == 'depth') & (D9.partition == 'rand6') & (D9.s != 'real')].copy(); dep['s'] = dep.s.astype(float)
assert (dep[dep.s == 1.0].z <= -2).mean() <= 0.10 and (dep[dep.s == 0.0].z <= -2).sum() == 0, "'detected in almost no run' / 'never fires' (S5.3)"
S9 = pd.read_csv(R + 'expR64b_wn30_summary.csv'); assert S9.ratio_real.min() > 1.0, "'within-cluster spread exceeds the between-hub spread' (S5.3)"
J = pd.read_csv(R + 'expR66_joint_sensitivity_summary.csv'); assert min(int(J.joint_genuine.sum()), int(J.boot_bh_genuine.sum())) > 36, "'leaves most of the count in place' (S5.2)"
zin = np.load(R + 'exp23_treemap_controls.npz', allow_pickle=True)["summary_in"].item(); na = zin["('euclid', 'average')"]; A = na['ari']
BLOCK = [0, 1, 2, 3, 9, 10, 11]   # model order of exp23: ViT-T/S/B/L, DINO-B, DINOv2-S/B/L/G, CLIP-B/L, SigLIP-B
assert A[4, BLOCK].mean() > 2 * A[[6, 7, 8]][:, BLOCK].mean(), "'DINO-B does not fall in the island' (S5.4): under the naive map DINO-B agrees with the block far more than DINOv2-B/L/G"
tx = pd.read_csv(R + 'expR53_text_haar_p999_200.csv').set_index('model')
assert tx.loc['gpt2', 'genuine_bh'] and not tx.loc['gpt2_m', 'genuine_bh'] and all(tx.loc[m, 'genuine_bh'] for m in ('gpt2_l', 'gpt2_xl', 'pythia_410m', 'pythia_1b', 'pythia_2b8', 'olmo_1b'))
assert not any(tx.loc[m, 'genuine_bh'] for m in ('bge_base', 'bge_large', 'gte_base', 'gte_large', 'gte_qwen2', 'e5_base', 'e5_large')), "'embedders absent on the class names' (S5.4)"
d61 = pd.read_csv(R + 'expR61_dbpedia_record.csv'); assert d61.genuine_bh.all(), "'embedders present on DBpedia' (S5.4)"
e21 = pd.read_csv(R + 'exp21b_local_global_K200.csv'); assert (e21.knn_R_p < 0.05).all() and (e21.cka_R_p < 0.05).all() and e21.knn_H_raw.mean() < e21.knn_R_raw.mean() and e21.cka_H_raw.mean() <= e21.cka_R_raw.mean() + 0.01, "S5.5 words"
# fourth review: the frame paragraph of S5.3, the class-count sentence of S5.2 and the recommendation of S6 are checked against the files
dv = pd.read_csv(R + 'expR56_depth_variants.csv'); im = dv[(dv.dataset == 'imagenet') & (dv.variant == 'aniso')]
cert = {K: set(im[(im.K == K) & (im.z_depth <= -2)].model) for K in (10, 30, 60)}
bal = pd.read_csv(R + 'expR64b_wn30bal_summary.csv'); cert['bal'] = set(bal[bal.real_z <= -2].model)
assert cert[30] == {'i21k_s', 'i21k_b', 'i21k_l', 'dinov2_l'} and cert['bal'] - cert[30] == {'i21k_t'} and cert[30] - cert['bal'] == {'i21k_s', 'dinov2_l'}, cert   # "ViT-T joins, ViT-S and DINOv2-L leave"
assert cert[10] != cert[30] and cert[60] != cert[30] and set.intersection(*cert.values()) == {'i21k_b', 'i21k_l'}, cert                        # "Only ViT-B and ViT-L are certified under every frame"
c60 = pd.read_csv(R + 'expR60_c_sweep_record.csv'); g = c60.groupby(['mode', 'C']).excess.mean()
for mode in ('coherent', 'random'):
    e = g[mode].sort_index(); assert e.iloc[0] < e.iloc[-1] < 0 and (e.diff().dropna() > 0).mean() >= 0.8, (mode, e.to_dict())   # "the excess shrinks with the number of classes for coherent and random subsets alike"
k = pd.read_csv(R + 'expR68_corollary_excess.csv') if os.path.exists(R + 'expR68_corollary_excess.csv') else None
e24 = pd.read_csv(R + 'exp24_val_metric_selection.csv'); pol = {c: e24[c].mean() for c in ('adv_cos', 'adv_rule', 'adv_val')}   # the advantages are stored in pp
assert pol['adv_cos'] > 0 and pol['adv_rule'] - pol['adv_cos'] <= 0.15 and pol['adv_val'] > pol['adv_rule'], pol    # "cosine everywhere ...; the objective rule adds little" (Table 13e: +0.23 vs +0.28 pp over 60 cells)
# ---- verbatim parts and the named edits
L = open(TEX + 'main_local.tex').read()
def between(a, b, s=L, strip=False): i = s.index(a); j = s.index(b, i); return s[i + (len(a) if strip else 0):j]
EDITS = [("a structureless cloud, a star of clusters without depth and a tree with depth cast the same shadow.", "a structureless cloud, a star of clusters without depth and a tree with depth all read the same."),
         ("Why is the shadow low?", "Why is the reading low?"),
         ("the geometric face of the width confounder of", "the geometric counterpart of the width confounder of"),
         ("it keeps the tail of the defects, the intent of the supremum, but it is set by hundreds of quadruples rather than by one.", "it keeps the tail of the defects, as the supremum does, but it is set by hundreds of quadruples rather than by one."),
         (" What a structureless cloud reads on it is the next question.", ""),
         # fourth and fifth reviews (2026-09-18): the abstract (the same one goes to OpenReview), S1 and S2 edits ordered by the author
         ("6 datasets and 16 text models, the premise does not survive:", "6 datasets and 15 text models, the raw reading is not evidence, and the calibrated reading is weak and model-dependent:"),
         ("We show that a low raw reading is what high dimension, an anisotropic spectrum and a maximum over sampled quadruples produce on their own, and we build",
          "We show that the reference level of the raw reading depends on dimension, spectrum and statistic, so a raw value cannot be called low on its own, and we build"),
         ("Class centroids do carry clustered structure in 49 of 72 cells, but a star already produces it; hierarchy above the superclasses is certified in only 4 of 12 ImageNet backbones, by a test",
          "Class centroids do carry clustered structure in 49 of 72 cells, 30 of 36 on datasets with 47 classes or more, but a star already produces it; hierarchy above the superclasses is certified in 4 of 12 ImageNet backbones, two of which survive every choice of frame, by a test"),
         ("but three artifacts push it down without any hierarchy.", "but its reference level depends on dimension, spectrum and statistic, so a raw value cannot be called low on its own."),
         ("Anisotropic spectra mimic low-dimensional behavior: the \\emph{spectrum confound}. And the supremum over sampled quadruples is a one-quadruple extreme that does not converge \\citep{fournier2015computing}: the \\emph{statistic confound}.",
          "Anisotropic spectra lower the effective dimension and raise the reading: the \\emph{spectrum confound}. And the supremum over sampled quadruples grows with the budget and does not converge \\citep{fournier2015computing}: the \\emph{statistic confound}."),
         ("so the premise does not survive calibration as stated. On class centroids, clustered structure is genuine in 49 of 72 cells, but a star already produces it. Hierarchy above the superclasses is certified in 4 of 12 ImageNet backbones by a test",
          "so the raw reading is not evidence and the calibrated reading is weak and model-dependent. On class centroids, clustered structure is genuine in 49 of 72 cells, 30 of 36 on datasets with 47 classes or more, but a star already produces it. Hierarchy above the superclasses is certified in 4 of 12 ImageNet backbones, two of which survive every choice of frame, by a test"),
         # page budget (fifth review): Figure 1 floats to the top of page 2 instead of leaving six blank lines at the foot of page 1 (placement only; the author's environment otherwise verbatim)
         ("\\begin{figure}[H]\n\\centering\n\\IfFileExists{figures/fig1_concept.pdf}", "\\begin{figure}[t]\n\\centering\n\\IfFileExists{figures/fig1_concept.pdf}"),
         ("we bring the idea to embedding clouds, where the reference must match dimension and spectrum.",
          "we bring the idea to embedding clouds, where the reference must match dimension and spectrum. The curvature itself has been studied as a representation tradeoff \\citep{sala2018representation} and as a mixed-curvature product to be learned \\citep{gu2019learning}, which presupposes a diagnostic of the kind we calibrate.")]
def edit(s):
    for a, b in EDITS:
        if a in s: s = s.replace(a, b)
    return s
ABSTRACT = edit(between("\\begin{abstract}", "\\end{abstract}")) + "\\end{abstract}"
assert "16 text models" not in ABSTRACT and "weak and model-dependent" in ABSTRACT and "30 of 36" in ABSTRACT and "two of which survive" in ABSTRACT and "cannot be called low" in ABSTRACT
# fifth review: fills and checks for the new sentences
F['FMNIST_GEN'] = str(int(c52[c52.dataset == 'fashionmnist'].genuine_bh.sum())); assert F['FMNIST_GEN'] == '7', F['FMNIST_GEN']
assert int(c52[c52.dataset.isin(['imagenet', 'cifar100', 'dtd'])].genuine_bh.sum()) == 30 and (c52[c52.dataset.isin(['imagenet', 'cifar100', 'dtd'])].n >= 47).all(), "'30 of 36 on the three datasets with 47 classes or more'"
hcells = e24[e24.dataset.isin(['imagenet', 'cifar100', 'cifar10', 'dtd'])]; assert len(hcells) == 40
F['POL_RULE_H'], F['POL_COS_H'] = f"{hcells.adv_rule.mean():+.2f}", f"{hcells.adv_cos.mean():+.2f}"; assert hcells.adv_rule.mean() > hcells.adv_cos.mean() > 0, "'the objective rule beats cosine'"
f2 = json.load(open(R + 'final_fig2b.json')); s_ = sl[(sl.model == f2['sample_cell'][0]) & (sl.dataset == f2['sample_cell'][1])].iloc[0]; c_ = c52[(c52.model == f2['class_cell'][0]) & (c52.dataset == f2['class_cell'][1])].iloc[0]
assert f2['sample_cell'][1] == f2['class_cell'][1] and abs(s_.delta_999 - c_.delta) < 0.001 and not s_.genuine_bh and c_.genuine_bh, "Figure 2(b): same dataset, same reading, opposite verdict"
wl = json.load(open(R + 'final_wordnet_levels.json')); F['WN_H'] = wl['h_word']
from math import comb; F['QUAD10'] = str(comb(10, 4)); assert F['QUAD10'] == '210'
json.dump({k: F[k] for k in ('FMNIST_GEN', 'POL_RULE_H', 'POL_COS_H', 'WN_H', 'QUAD10', 'NAIVE_BIG', 'C_LO', 'C_HI')}, open(R + 'final_fills.json', 'w'), indent=1)   # the fills of the final version, read by the sweep
ga = pd.read_csv(R + 'exp1_delta_controls.csv'); ga = ga[ga.variant == 'gauss'].sort_values('d')
assert F['C_LO'] == f"{(0.144/(2*float(ga.delta_max.iloc[0])))**2:.2f}" and F['C_HI'] == f"{(0.144/(2*float(ga.delta_max.iloc[-1])))**2:.1f}", "Khrulkov's rule recomputed on the supremum Gaussian band of Table 3 (exp1 delta_max is the sampled supremum)"
INTRO = edit(between("\\section{Introduction}", "\\section{Related Work}").rstrip())
RELATED = edit(between("\\section{Related Work}", "\\section{The Instrument}").rstrip()); assert "sala2018representation" in RELATED
GROMOV = edit(between("\\paragraph{Gromov $\\delta$.} ", "\\paragraph{Estimation and normalization.}", strip=True).rstrip())
ESTIM = edit(between("\\paragraph{Estimation and normalization.} ", "\\begin{figure}", strip=True).rstrip())
for a, _ in EDITS: assert a not in INTRO + GROMOV + ESTIM, a[:40]
json.dump({"edits": EDITS}, open(R + 'final_verbatim_edits.json', 'w'), indent=1)
CUTS = [c for c in os.environ.get("FINAL_CUTS", "").split(",") if c]
if "s2" in CUTS:   # S2 to one paragraph of six sentences: the first six sentences of the author's paragraph
    head, para = RELATED.split("\n\n", 1)[0], RELATED.split("\n\n", 1)[1]
    sents = re.split(r"(?<=[.])\s+(?=[A-Z\\])", para.strip()); RELATED = head + "\n\n" + " ".join(sents[:6])
body = open(S + 'phaseE_paper_final.tex.tmpl').read()
for k, v in [("%%ABSTRACT%%", ABSTRACT), ("%%INTRO%%", INTRO), ("%%RELATED%%", RELATED), ("%%GROMOV%%", GROMOV), ("%%ESTIMATION%%", ESTIM)]:
    assert body.count(k) == 1, k; body = body.replace(k, v)
if "s55" in CUTS: pass   # S5.5 is written with three sentences in the template (cut step 1 applied at the source)
if "s6" in CUTS: pass    # S6 has three paragraphs in the template (cut step 2)
if "table" in CUTS: pass # the model table is in the appendix (Table of the panel, cut step 3)
for k, v in F.items(): body = body.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{[A-Z0-9_]+\}\}", body); assert not left, left
# ---- appendix: the cited question tables of the v1 template, without figures, in the order the final text first cites them
v1t = open(S + 'phaseE_paper.tex.tmpl').read(); app = v1t[v1t.index("\\section{Appendix: one table per question}"):v1t.index("\\end{document}")]
for k, v in F.items(): app = app.replace("{{" + k + "}}", v)
head = app[:app.index("\\FloatBarrier")].replace("\\section{Appendix: one table per question}", "\\section{One table per question}")
blocks = [b for b in re.split(r"(?=\\FloatBarrier\n\\subsection\{)", app[app.index("\\FloatBarrier"):]) if b.strip()]
bytab = {}
for b in blocks:
    m = re.search(r"\\input\{appendix_tables/(tab_[a-z0-9_]+)\}", b); bytab[m.group(1)] = b
KEEP = ["tab_q10_calibration", "tab_q01_census", "tab_q08_robust", "tab_q03_sample", "tab_q04_depth", "tab_q05_power", "tab_q06_treemap", "tab_q07_wordnet", "tab_q02_text", "tab_q13_local", "tab_q09_corollary", "tab_q14_panel"]
DELETED = sorted(set(bytab) - set(KEEP) - {"tab_z_provenance"})
def clean_block(b):
    b = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}\n?", "", b, flags=re.S)                     # no appendix figure is cited from the main text
    b = b.replace("(Table~\\ref{tab:q4-depth}, Figure~\\ref{fig:depth}a)", "(Table~\\ref{tab:q4-depth}, Figure~\\ref{fig:depth}a)")
    b = b.replace(" are reported without certification (Figure~\\ref{fig:depth-c100}).", " are reported without certification (Table~\\ref{tab:q4-depth}).")
    b = b.replace("(Figure~\\ref{fig:bestmetric}; held-out selection and policy comparison in Table~\\ref{tab:q9-corollary})", "(Table~\\ref{tab:q9-corollary})")
    b = b.replace("(Figure~\\ref{fig:causal}; raw $\\delta$, within architecture)", "(raw $\\delta$, within architecture)")
    b = b.replace("\\paragraph{The calibrated reading predicts the gain within datasets.}", "\\paragraph{Both readings predict the gain within datasets.}")   # fifth review, S B.12
    return b
# tables regenerated for the final by gen_appendix_final.py (appendix_tables/final/*_final.tex): robustness (bootstrap under the record, class-count sweep), depth (two-decimal z, K sweep), wordnet (DBpedia supremum under the Haar null)
FINAL_SRC = {"tab_q08_robust": "tab_q08_robust_final", "tab_q04_depth": "tab_q04_depth_final", "tab_q07_wordnet": "tab_q07_wordnet_final", "tab_q05_power": "tab_q05_power_final"}
B8 = clean_block(bytab["tab_q08_robust"])
B8 = re.sub(r"\\paragraph\{Hierarchy depth, not class count\.\}.*?\n", "", B8)                    # the class-count paragraph of v1 claimed the opposite of the corrected caption
bytab["tab_q08_robust"] = B8
for k in KEEP:
    bytab[k] = clean_block(bytab[k]) if k != "tab_q08_robust" else bytab[k]
    if k in FINAL_SRC: bytab[k] = bytab[k].replace("\\input{appendix_tables/" + k + "}", "\\input{appendix_tables/" + FINAL_SRC[k] + "}")
labels = {}
for stem in KEEP:
    fn = FINAL_SRC.get(stem, stem)
    mm = re.search(r"\\label\{(tab:q[^}]*)\}", open(TEX + ("appendix_tables/final/" if stem in FINAL_SRC else "appendix_tables/") + fn + ".tex").read()); labels[mm.group(1)] = stem
main = body[:body.index("\\appendix")]; cited = []
for m in re.finditer(r"\\ref\{(tab:q[^}]*)\}", main):
    if m.group(1) not in cited: cited.append(m.group(1))
order = [labels[l] for l in cited if l in labels] + [s for s in KEEP if s not in [labels[l] for l in cited if l in labels]]
assert set(order) == set(KEEP), (order, KEEP)
prov_block = bytab["tab_z_provenance"].replace("appendix_tables/tab_z_provenance", "appendix_tables/final/tab_z_provenance_final")
# ---- the final's copies of the kept tables: appendix_tables/final/, one floating [tbp] table per panel so the appendix pages pack (the [H] parts of v1 left every page half empty)
FD = TEX + "appendix_tables/final/"; os.makedirs(FD, exist_ok=True)
# panels the brief deletes (null-variant panel, uncited supremum table) with the caption sentence that described them
DROP = {"tab_q01_census": ["(b) Verdict under the four null"], "tab_q09_corollary": ["(b) Correlation between the raw supremum"]}
CAPFIX = {"tab_q01_census": lambda c: re.sub(r"\s*\(b\) Four symbols per cell.*?(?=\s*%)", "", c, flags=re.S)}
DROPLINE = {"tab_q02_text": ["OLMo-7B"], "tab_q14_panel": ["OLMo-7B"]}   # fourth review: 15 text models; OLMo-7B was not extracted
REPL = {"tab_q14_panel": [("9 causal LMs", "8 causal LMs")],                                                                 # fifth review: Table 5 caption
        "tab_q09_corollary": [("\\textbf{The calibrated reading predicts the zero-cost gain within datasets,", "\\textbf{Both readings predict the zero-cost gain within datasets,")]}   # Table 13 title
def split_panels(src, drop=(), capfix=None, dropline=(), repl=()):
    for a, b in repl:
        assert src.count(a) == 1, a; src = src.replace(a, b)
    if dropline:
        n0 = len(src.split("\n")); src = "\n".join(l for l in src.split("\n") if not any(d in l for d in dropline)); assert len(src.split("\n")) == n0 - len(dropline), dropline
    prov = [l for l in src.split("\n") if l.startswith("% prov:")]; out = list(prov); first = True; dropped = 0
    blocks_ = re.findall(r"\\begin\{table\}\[(?:H|tbp)\](\\ContinuedFloat)?\n(.*?)\n\\end\{table\}", src, flags=re.S); assert blocks_, "no table block in the source"   # [tbp]: an already split copy (idempotent)
    for _, body in blocks_:
        L = body.split("\n"); hdr, rest = L[:3], L[3:]; assert hdr[0] == "\\centering", hdr
        ci = next(i for i, l in enumerate(rest) if l.startswith("\\caption{")); cap, plines = rest[ci:], rest[:ci]
        if capfix and first: cap = ("\n".join(cap)); cap2 = capfix(cap); assert cap2 != cap, "caption sentence of the dropped panel not found"; cap = cap2.split("\n")
        panels, cur = [], []
        for i, l in enumerate(plines):   # a panel starts at \begingroup (sized panel) or at a bold panel title not wrapped in a group
            if (l == "\\begingroup" or (l.startswith("\\noindent\\textbf{(") and (i == 0 or plines[i-1] != "\\begingroup"))) and cur: panels.append(cur); cur = []
            cur.append(l)
        if cur: panels.append(cur)
        isdrop = lambda P: any(("\\noindent\\textbf{" + d) in "\n".join(P) for d in drop)
        cap_ok = first or not isdrop(panels[0]); first_kept = True   # a continued block whose first panel is dropped loses its caption (it described that panel)
        for P in panels:
            if isdrop(P): dropped += 1; continue
            out += ["\\begin{table}[tbp]" + ("" if first else "\\ContinuedFloat")] + hdr + P + (cap if (first_kept and cap_ok) else ["\\caption{(continued)}"]) + ["\\end{table}"]; first = False; first_kept = False
    assert dropped == len(drop), (drop, dropped)
    return "\n".join(out) + "\n"
for stem in order:
    fn = FINAL_SRC.get(stem, stem)
    src = open((FD if stem in FINAL_SRC else TEX + "appendix_tables/") + fn + ".tex").read()
    open(FD + fn + ".tex", "w").write(split_panels(src, DROP.get(stem, ()), CAPFIX.get(stem), DROPLINE.get(stem, ()), REPL.get(stem, ())))
for f in os.listdir(FD):   # stale copies from earlier rounds are removed so the provenance index and the sweep see the current set only
    if f.endswith(".tex") and f[:-4] not in [FINAL_SRC.get(s, s) for s in order] + ["tab_z_provenance_final"]: os.remove(FD + f)
appendix = head + "".join(bytab[s] for s in order) + prov_block
appendix = appendix.replace("\\input{appendix_tables/tab_", "\\input{appendix_tables/final/tab_")   # the \FloatBarrier before every subsection stays (fourth review: no heading is left empty)
appendix = appendix.replace("\\section{One table per question}", "\\renewcommand{\\topfraction}{0.95}\\renewcommand{\\bottomfraction}{0.95}\\renewcommand{\\textfraction}{0.03}\\renewcommand{\\floatpagefraction}{0.85}\\setcounter{topnumber}{4}\\setcounter{bottomnumber}{4}\\setcounter{totalnumber}{6}\\setlength{\\floatsep}{8pt plus 2pt}\n"
                            "% final version: the tables float ([tbp], one float per panel) within their subsection (a \\FloatBarrier closes each one)\n\\section{One table per question}", 1)
assert "\\input{appendix_tables/final/tab_q08_robust_final}" in appendix and appendix.count("\\FloatBarrier") >= 12
body = body.replace("%%APPENDIX_QUESTIONS%%", appendix) + "\n\\end{document}\n"
T1 = open(P1).read(); pre = T1[:T1.index("\\begin{abstract}")]
pre = pre.replace("\\usepackage{array}\n", "\\usepackage{array}\n\\usepackage{amsthm}\n", 1)
pre = pre.replace("\\begin{document}\n", "\\newtheoremstyle{inline}{3pt}{3pt}{}{}{\\bfseries}{.}{ }{}\n\\newtheoremstyle{inlineit}{3pt}{3pt}{\\itshape}{}{\\bfseries}{.}{ }{}\n"
                  "\\theoremstyle{inline}\n\\newtheorem{definition}{Definition}\n\\theoremstyle{inlineit}\n\\newtheorem{proposition}{Proposition}\n"
                  "\\makeatletter\\g@addto@macro\\normalsize{\\setlength\\abovedisplayskip{3pt plus 1pt}\\setlength\\belowdisplayskip{3pt plus 1pt}\\setlength\\abovedisplayshortskip{2pt}\\setlength\\belowdisplayshortskip{2pt}}\n"
                  "% typographic only (page budget, 2026-09-18): the section headings and the run-in paragraph headings open with less white space than the style's default; fonts, margins and line spacing are the style's\n"
                  "\\renewcommand\\section{\\@startsection{section}{1}{\\z@}{-1.0ex plus -0.4ex minus -.2ex}{0.6ex plus 0.2ex minus 0.1ex}{\\large\\sc\\raggedright}}\n"
                  "\\renewcommand\\subsection{\\@startsection{subsection}{2}{\\z@}{-1.0ex plus -0.4ex minus -.2ex}{0.5ex plus .2ex}{\\normalsize\\sc\\raggedright}}\n"
                  "\\renewcommand\\paragraph{\\@startsection{paragraph}{4}{\\z@}{0.3ex plus 0.3ex minus .2ex}{-1em}{\\normalsize\\bf}}\\makeatother\n"
                  "\\setlength{\\textfloatsep}{8pt plus 2pt minus 2pt}\\setlength{\\abovecaptionskip}{3pt}\\setlength{\\parskip}{4pt plus 1pt minus 1pt}\n"
                  "% final version: classic structure, plain prose (author's brief of 2026-09-18); generated by rebuttal/scripts/phaseE_submission.py.\n\\begin{document}\n", 1)
open(PF, 'w').write(pre + body)
json.dump({"kept": order, "deleted": DELETED, "cuts": CUTS}, open(R + 'final_appendix.json', 'w'), indent=1)
print("final written ->", PF, "| appendix order:", order, "| deleted:", DELETED, "| cuts:", CUTS)
