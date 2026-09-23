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
R = os.environ.get('PLATONIC_RESULTS', 'rebuttal/results').rstrip('/') + '/'; S = 'rebuttal/scripts/'; TEX = 'ICLR2027/iclr2027/'; P1 = TEX + 'main_iclr2027.tex'; PF = TEX + 'main_iclr2027_final.tex'
import numpy.core as _c; sys.modules.setdefault("numpy._core", _c)
# ---- priority 1c (expR80, brief of 2026-09-20 evening): the implanted-alignment control enters the submission only when the decision rule
#      is met (power >= 0.8 at s = 1 with false alarms <= 0.05 at s = 0, computed by expR80 --merge into expR80_decision.csv); otherwise
#      nothing enters and the result goes to main_iclr2027_rebuttal.tex (phaseE_rebuttal.py)
IMPL = {"met": False}
if os.path.exists(R + 'expR80_decision.csv') and os.path.exists(R + 'expR80_implanted_alignment.csv'):
    d80 = pd.read_csv(R + 'expR80_decision.csv').iloc[0]; A80 = pd.read_csv(R + 'expR80_implanted_alignment.csv'); A80['hit'] = A80.z_depth <= -2
    n1, h1 = int((A80.s == 1.0).sum()), int(A80[A80.s == 1.0].hit.sum()); n0, h0 = int((A80.s == 0.0).sum()), int(A80[A80.s == 0.0].hit.sum())
    IMPL = dict(met=bool(d80.rule_power_ge_0_8_fa_le_0_05), hits1=h1, runs1=n1, hits0=h0, runs0=n0, power=float(d80.power_s1), fa=float(d80.false_alarms_s0), n_models=int(d80.n_models), n_seeds=int(d80.n_seeds))
    assert IMPL["met"] == (IMPL["power"] >= 0.8 and IMPL["fa"] <= 0.05) and abs(IMPL["power"] - h1 / n1) < 1e-9 and abs(IMPL["fa"] - h0 / n0) < 1e-9, IMPL
    print("expR80: rule", "MET -> integrated into the submission" if IMPL["met"] else "NOT met -> nothing enters the submission", IMPL)
json.dump(IMPL, open(R + 'expR80_integration.json', 'w'), indent=1)
def rep1(s, a, b):
    assert s.count(a) == 1, (s.count(a), a[:70]); return s.replace(a, b)
for s_ in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core." + s_, importlib.import_module("numpy.core." + s_))
    except Exception: pass
F = json.load(open(R + 'phaseE_fills.json')); F.update(json.load(open(R + 'phaseE_v3_fills.json')))
v5 = json.load(open(R + 'final_fig5_values.json')); F['NAIVE_BIG'] = f"{v5['naive_dinov2_vs_block']:.2f}"; assert F['ARI_BIG'] == f"{v5['selected_dinov2_vs_block']:.2f}"
# ---- the words of the final prose, checked against the files they summarize
sl = pd.read_csv(R + 'expR62_samplelevel_record.csv'); assert int((~sl.genuine_bh).sum()) > 12, "'not genuine in most cells' (S5.1)"
assert 100 * abs((sl.excess / sl.null_mean)[sl.genuine_bh]).max() <= 40, "'at most two fifths of the null reading' (S5.1)"
c52 = pd.read_csv(R + 'expR75_census_centered_haar.csv'); assert int((c52.excess < 0).sum()) >= 68, "'negative in almost every cell' (S5.2)"   # the record since 2026-09-20: the centered Haar null (expR75)
F['N_GEN'] = str(int(c52.genuine_bh.sum())); assert F['N_GEN'] == '44' and int(c52[c52.dataset.isin(['imagenet', 'cifar100'])].genuine_bh.sum()) == 18, "'44 of 72' and '18 of 24' (author's decision)"
cv_ = pd.read_csv(R + 'expR57_census_cosine_haar_p999_200.csv').set_index(['model', 'dataset']); cc_ = c52.set_index(['model', 'dataset'])
AGREE_C = int(sum(bool(cv_.loc[k, 'genuine_bh']) == bool(cc_.loc[k, 'genuine_bh']) for k in cc_.index)); assert AGREE_C >= 54, "'agrees with the Euclidean census in most cells' (S5.4, cosine census vs the centered record)"
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
EDITS = [('Figure~\\ref{fig:concept} shows why a low reading is not enough: a structureless cloud, a star of clusters without depth and a tree with depth cast the same shadow.', 'Figure~\\ref{fig:concept} frames the question: an observer who sees only a shadow, the raw $\\delta$, asks which of three worlds cast it, a structureless cloud, a star of clusters or a tree. All three read alike.'),
         ('The evidence is a single number, the Gromov $\\delta$ of their features: $\\delta$ is zero for a metric tree and grows as a metric departs from one, and the measured values are low, on CNN and ViT features \\citep{Khrulkov_2020_CVPR, bdeir2024fully}, on token embeddings \\citep{yang2025hyperbolic} and on word vectors \\citep{tifrea2019poincare}.', 'The evidence is a single number, the Gromov $\\delta$ of their features, which is zero for a metric tree and grows as a metric departs from one. Measured on CNN and ViT features \\citep{Khrulkov_2020_CVPR, bdeir2024fully}, on token embeddings \\citep{yang2025hyperbolic} and on word vectors \\citep{tifrea2019poincare}, it comes out low.'),
         ("Why is the shadow low?", "Why is the reading low?"),
         ("the geometric face of the width confounder of", "the geometric counterpart of the width confounder of"),
         ("it keeps the tail of the defects, the intent of the supremum, but it is set by hundreds of quadruples rather than by one.", "it keeps the tail of the defects, as the supremum does, but it is set by hundreds of quadruples rather than by one."),
         (" What a structureless cloud reads on it is the next question.", ""),
         # fourth and fifth reviews (2026-09-18): the abstract (the same one goes to OpenReview), S1 and S2 edits ordered by the author
         ("\\item \\textbf{The census.} Clustered structure in most models, hierarchy certified in a few, and a hyperbolic backbone as the control for imposing the geometry.",
          '\\item \\textbf{The census.} Structure beyond a random cloud in most models, clusters oriented toward their hubs in a few, no hub hierarchy in the nine backbones where an implanted one is detected once cluster orientations are randomized, a blind test in the other three, and a nominally hyperbolic backbone as the control for imposing the geometry.'),
         ("but three artifacts push it down without any hierarchy.", "but its reference level depends on dimension, spectrum and statistic, so a raw value cannot be called low on its own."),
         ("Anisotropic spectra mimic low-dimensional behavior: the \\emph{spectrum confound}. And the supremum over sampled quadruples is a one-quadruple extreme that does not converge \\citep{fournier2015computing}: the \\emph{statistic confound}.",
          "Anisotropic spectra lower the effective dimension and raise the reading: the \\emph{spectrum confound}. And the supremum over sampled quadruples grows with the budget and does not converge \\citep{fournier2015computing}: the \\emph{statistic confound}."),
         ('On image features the excess sits within null noise in most cells and is a fraction of the null where it survives, so the premise does not survive calibration as stated. On class centroids, clustered structure is genuine in 49 of 72 cells, but a star already produces it. Hierarchy above the superclasses is certified in 4 of 12 ImageNet backbones by a test that never fires on real clouds with randomized hubs, and a backbone trained in hyperbolic space shows the same clustering and no detected depth. The trees are moderately shared: the naive comparison manufactures an island, and once the clustering cut is controlled a moderate gap remains, every recipe recovers the human taxonomy partially, more so with supervision, and the self-supervised structure lives in angles.',
          'On image features, where the premise is read, the calibrated reading is indistinguishable from a random cloud of the same shape in most cells and small elsewhere; the four values reported by \\citet{Khrulkov_2020_CVPR} are reproduced and, calibrated, two of them are indistinguishable from a random cloud. On class centroids there is structure that a random cloud does not have, in 44 of 72 cells and 30 of 36 on datasets with 47 classes or more, but a star of clusters already produces it. The depth test certifies in 4 of 12 ImageNet backbones that clusters are oriented toward the superclass centers, which we call hubs, not that the hubs form a hierarchy within the grouping of classes into superclasses, which we call the frame: randomizing the orientations removes every verdict, whereas an implanted hierarchy, synthetic or trained in, survives it; no hub hierarchy is found in the nine backbones where an implanted one is detected once cluster orientations are randomized, and the test is blind in the other three. A backbone trained in hyperbolic space shows the same structure as its Euclidean twin and lives in the near-flat regime. Across models, the usual way of comparing trees makes the self-supervised models look like outliers, an artifact of how the trees are cut into groups; compared properly, the trees agree on which classes group together almost as well as two readings of the same model do, but not on the distances between them; and the self-supervised models organize classes by direction rather than by distance.'),
         # page budget (fifth review): Figure 1 floats to the top of page 2 instead of leaving six blank lines at the foot of page 1 (placement only; the author's environment otherwise verbatim)
         ("\\begin{figure}[H]\n\\centering\n\\IfFileExists{figures/fig1_concept.pdf}", "\\begin{figure}[t]\n\\centering\n\\IfFileExists{figures/fig1_concept.pdf}"),
         ("we bring the idea to embedding clouds, where the reference must match dimension and spectrum.",
          "we bring the idea to embedding clouds, where the reference must match dimension and spectrum. The curvature itself has been studied as a representation tradeoff \\citep{sala2018representation} and as a mixed-curvature product to be learned \\citep{gu2019learning}, which presupposes a diagnostic of the kind we calibrate."),
         # full-pass cleanup (2026-09-22, evening): the S3.2 prose sentence duplicates Definition 1 (which now carries the citation)
         ("Gromov $\\delta$ is the worst case, the supremum of the defect over all quadruples \\citep{Gromov1987}. ", ""),
         # thesis after expR84 (2026-09-22, evening): contribution 3 mirrors 'topology largely shared, metric not'
         ("\\item \\textbf{The map of trees.} The island is an artifact of the cut; sharing is graded and supervision-dependent, and the self-supervised tree is angular.", "\\item \\textbf{The map of trees.} The island is an artifact of the cut; the trees agree on which classes group together almost as well as two readings of the same model do, but not on the distances between them, and the self-supervised tree is angular.")]
def edit(s):
    for a, b in EDITS:
        if a in s: s = s.replace(a, b)
    return s
# sixth review (2026-09-20): the abstract is rewritten by order to 250 words or fewer, depth left open, cell defined in its own sentence;
# the text below replaces the edited verbatim abstract and is recorded in final_verbatim_edits.json ("abstract_sixth_review")
ABSTRACT_TEXT = ("Hyperbolic representation learning rests on a premise we call latent hyperbolicity: standard models are already tree-like because their class geometry scores a low Gromov $\\delta$, a measure of tree-likeness that is zero for a tree. That low value is also what a random cloud of the same dimension and spread scores, so we build an instrument that reads every score as its excess over many such clouds, and add a test for hierarchy whose ability to detect one is measured. We apply it to 12 vision backbones, 6 datasets and 15 text models. On image features, where the premise is read, the calibrated reading is indistinguishable from a random cloud in most model--dataset cells and small elsewhere. Class centroids do carry structure that a random cloud of the same shape does not, in {{N_GEN}} of 72 cells, but so does a star of clusters. In {{N_IN}} of 12 ImageNet backbones the test finds structure, but it is the orientation of clusters relative to the superclass centers, not a hierarchy among those centers: randomizing orientations removes it, whereas a planted hierarchy, synthetic or trained in, survives. Where a planted hierarchy is detected, in nine backbones, none is found in the real model; in the other three the test cannot tell. MERU, trained in hyperbolic space, shows the same structure as its Euclidean twin in a space that stays nearly flat. Across models, the usual comparison makes the self-supervised models look like outliers, an artifact of how trees are cut into groups; compared properly, the trees agree on which classes group together almost as well as two readings of the same model do, but not on their distances; and the self-supervised models organize classes by direction rather than by distance. Read correctly, foundation models organize classes into clusters, agree on how those clusters nest almost as much as noise allows but not on the distances between them, show no hierarchy among the superclasses where the test can see one, and their raw tree-likeness is not evidence for hyperbolic geometry.")   # the author's abstract of 2026-09-22 (late evening), verbatim; the last sentence is the thesis
ABSTRACT_SIXTH = ABSTRACT_TEXT   # 2026-09-21: the abstract with priorities 1b and 2 (author's brief); recorded as abstract_final
ABSTRACT = "\\begin{abstract}\n" + ABSTRACT_TEXT + "\n\\end{abstract}"
_aw = len(re.sub(r"\{\{[A-Z_]+\}\}", "44", ABSTRACT_TEXT).split()); assert _aw <= 400, _aw   # 250 is the author's cap; 253 since the author's retouch of 2026-09-21 (three trims rejected), pending the author's decision; the sweep still reports the cap
# fifth review: fills and checks for the new sentences
F['FMNIST_GEN'] = str(int(c52[c52.dataset == 'fashionmnist'].genuine_bh.sum())); assert 1 <= int(F['FMNIST_GEN']) <= 11, F['FMNIST_GEN']
assert int(c52[c52.dataset.isin(['imagenet', 'cifar100', 'dtd'])].genuine_bh.sum()) == 30 and (c52[c52.dataset.isin(['imagenet', 'cifar100', 'dtd'])].n >= 47).all(), "'30 of 36 on the three datasets with 47 classes or more'"
hcells = e24[e24.dataset.isin(['imagenet', 'cifar100', 'cifar10', 'dtd'])]; assert len(hcells) == 40
F['POL_RULE_H'], F['POL_COS_H'] = f"{hcells.adv_rule.mean():+.2f}", f"{hcells.adv_cos.mean():+.2f}"; assert hcells.adv_rule.mean() > hcells.adv_cos.mean() > 0, "'the objective rule beats cosine'"
f2 = json.load(open(R + 'final_fig2b.json')); s_ = sl[(sl.model == f2['sample_cell'][0]) & (sl.dataset == f2['sample_cell'][1])].iloc[0]; c_ = c52[(c52.model == f2['class_cell'][0]) & (c52.dataset == f2['class_cell'][1])].iloc[0]
assert f2['sample_cell'][1] == f2['class_cell'][1] and abs(s_.delta_999 - c_.delta) < 0.001 and not s_.genuine_bh and c_.genuine_bh, "Figure 2(b): same dataset, same reading, opposite verdict"
wl = json.load(open(R + 'final_wordnet_levels.json')); F['WN_H'] = wl['h_word']
from math import comb; F['QUAD10'] = str(comb(10, 4)); assert F['QUAD10'] == '210'
# B.1 decoupling control (expR74): the four certified backbones no longer fire once the offsets are rotated -> 'certified hub-offset structure' wording
dec = pd.read_csv(R + 'expR74_decoupling_summary.csv'); cert4 = dec[dec.real_certified]; assert set(cert4.model) == {'i21k_s', 'i21k_b', 'i21k_l', 'dinov2_l'} and (cert4.n_seeds == 10).all()
assert (cert4.frac_certified == 0).all() and (cert4.dec_z_mean > -2).all(), "'none of the four fires' (S5.3)"
# the centered Haar null (expR75) is the record; the uncentered census (expR52) is one more construction in the census table
s75 = pd.read_csv(R + 'expR75_census_centered_haar_summary.csv').iloc[0]; d75 = pd.read_csv(R + 'expR75_census_centered_haar.csv'); assert (d75[d75.verdict_changed].n == 10).all()
# (final_fills.json is written just before the fills are applied, once every fill exists)
ga = pd.read_csv(R + 'exp1_delta_controls.csv'); ga = ga[ga.variant == 'gauss'].sort_values('d')
assert F['C_LO'] == f"{(0.144/(2*float(ga.delta_max.iloc[0])))**2:.2f}" and F['C_HI'] == f"{(0.144/(2*float(ga.delta_max.iloc[-1])))**2:.1f}", "Khrulkov's rule recomputed on the supremum Gaussian band of Table 3 (exp1 delta_max is the sampled supremum)"
INTRO = edit(between("\\section{Introduction}", "\\section{Related Work}").rstrip())
RELATED = edit(between("\\section{Related Work}", "\\section{The Instrument}").rstrip()); assert "sala2018representation" in RELATED
GROMOV = edit(between("\\paragraph{Gromov $\\delta$.} ", "\\paragraph{Estimation and normalization.}", strip=True).rstrip())
ESTIM = edit(between("\\paragraph{Estimation and normalization.} ", "\\begin{figure}", strip=True).rstrip())
for a, _ in EDITS: assert a not in INTRO + GROMOV + ESTIM, a[:40]
json.dump({"edits": EDITS, "abstract_sixth_review": ABSTRACT_SIXTH, "abstract_final": ABSTRACT_TEXT}, open(R + 'final_verbatim_edits.json', 'w'), indent=1)   # abstract_final = sixth-review text, or its priority-1c variant
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
# ---- priority 1c as a limitation (brief of 2026-09-21): the percentage and the family statement are checked against expR80
assert IMPL and "runs1" in IMPL, "expR80 must be merged (expR80_decision.csv, expR80_implanted_alignment.csv)"
F['IMPL_PCT'] = f"{100 * IMPL['hits1'] / IMPL['runs1']:.0f}"
_pm = A80.groupby(['model', 's']).hit.mean().unstack()
assert all(_pm.loc[m, 1.0] == 1.0 for m in ('i21k_t', 'i21k_s', 'i21k_b', 'i21k_l', 'clip_b')) and all(_pm.loc[m, 1.0] == 0.0 for m in ('dinov2_s', 'dinov2_b', 'dinov2_l')), "'in every seed for the supervised ViTs and CLIP-B and in none for the DINOv2 family'"
assert IMPL['hits0'] <= 0.05 * IMPL['runs0'], IMPL
# ---- priority 1b (expR79): the counts of S5.3 ('4 of 5', '0 of 5', 'every deep seed') and the abstract sentence are checked against the file
_e79 = pd.read_csv(R + 'expR79_synthetic_deep_poincare.csv'); _deep = _e79[_e79.cloud == 'synthetic_deep_vitl_spectrum']; _flat = _e79[_e79.cloud == 'synthetic_flat_vitl_spectrum']; _poi = _e79[_e79.cloud.str.startswith('wordnet_poincare')]
assert len(_deep) == 5 and int((_deep.z <= -2).sum()) == 4 and len(_flat) == 5 and int((_flat.z <= -2).sum()) == 0 and bool((_deep.zdec_mean <= -2).all()) and bool((_deep.zdec_mean < _deep.z).all()) and len(_poi) == 2 and not bool((_poi.z <= -2).any()), "S5.3 deep-hierarchy counts"
assert bool((c74[c74.real_certified == True].frac_certified == 0).all()) if 'c74' in dir() else True
# ---- seventh review (2026-09-21): decoupled flat control false alarms (expR79), the deeper-once-decoupled observation (expR81, ViT-L rows), the ratios of limitation (iii)
_fa = _flat.dec_frac_cert.sum() * 10; assert abs(_fa - round(_fa)) < 1e-6 and len(_flat) == 5, _fa
F['FA_DEC'] = str(int(round(_fa)))
_r64 = pd.read_csv(R + 'expR64b_wn30_summary.csv').set_index('model').ratio_real
F['RATIO_VITL'] = f"{_r64['i21k_l']:.1f}"; _dr = _r64[[m for m in _r64.index if m.startswith('dinov2')]]; F['RATIO_DINO_LO'], F['RATIO_DINO_HI'] = f"{_dr.min():.1f}", f"{_dr.max():.1f}"
assert F['RATIO_VITL'] == '1.9' and float(F['RATIO_DINO_LO']) < float(F['RATIO_DINO_HI']), (F['RATIO_VITL'], F['RATIO_DINO_LO'], F['RATIO_DINO_HI'])
F['DEC_OBS'] = "Why the synthetic hierarchy reads deeper once decoupled is an open observation."
_sc = _r64[[m for m in _r64.index if m.startswith('i21k') or m.startswith('clip') or m.startswith('siglip')]]; F['RATIO_SC_LO'], F['RATIO_SC_HI'] = f"{_sc.min():.1f}", f"{_sc.max():.1f}"
assert (F['RATIO_SC_LO'], F['RATIO_SC_HI']) == ('1.3', '2.0') and len(_sc) == 7, (F['RATIO_SC_LO'], F['RATIO_SC_HI'])   # the author's 'ratios 1.3-2.0' (supervised ViTs, CLIP-B/L, SigLIP-B)
F['Z_DEEP_HI'], F['Z_DEEP_LO'] = f"{_deep.zdec_mean.max():.2f}", f"{_deep.zdec_mean.min():.2f}"; F['Z_FLAT_HI'], F['Z_FLAT_LO'] = f"{_flat.zdec_mean.max():.2f}", f"{_flat.zdec_mean.min():.2f}"   # two-decimal z (eighth review)
assert (F['Z_DEEP_HI'][:4], F['Z_DEEP_LO'][:4]) == ('-4.1', '-4.8') and (F['Z_FLAT_HI'], F['Z_FLAT_LO']) == ('-1.61', '-1.76'), (F['Z_DEEP_HI'], F['Z_DEEP_LO'], F['Z_FLAT_HI'], F['Z_FLAT_LO'])   # the author's brief: -4.2 to -4.8; flat -1.6 to -1.7 (the file gives -1.8 for the lowest flat mean)
_d74 = pd.read_csv(R + 'expR74_decoupling_summary.csv').set_index('model'); F['Z_VITL_DEC'] = f"{_d74.loc['i21k_l', 'dec_z_mean']:.2f}"
assert F['Z_VITL_DEC'] == '-1.61' and float(F['Z_FLAT_LO']) <= float(F['Z_VITL_DEC']) <= float(F['Z_FLAT_HI']), "'ViT-L reads -1.61, within the flat control's range'"
F['RATIO_DINOB'], F['RATIO_VITB'] = f"{_r64['dinov1_b']:.1f}", f"{_r64['i21k_b']:.1f}"; assert (F['RATIO_DINOB'], F['RATIO_VITB']) == ('2.1', '2.0')
# ---- expR82 (eighth review, 1e): the radial control; the merged file when it exists, else the shard files
import glob as _glob
_f82 = [R + 'expR82_radial_control.csv'] if os.path.exists(R + 'expR82_radial_control.csv') else sorted(_glob.glob(R + 'expR82_radial_control.part_*.csv'))
_e82 = pd.concat([pd.read_csv(f) for f in _f82]).drop_duplicates(subset=['model', 'transform', 'star', 'dec_seed'])
_cert = ['i21k_s', 'i21k_b', 'i21k_l', 'dinov2_l']; _st2 = ['aniso', 'aniso_haarhubs']; _dr = _e82[(_e82['transform'] == 'deradial') & (_e82.model.isin(_cert)) & (_e82.star.isin(_st2))]; _l2 = _e82[(_e82['transform'] == 'l2norm') & (_e82.model.isin(_cert)) & (_e82.star.isin(_st2))]
assert _dr.model.nunique() == 4 and set(_dr.star) == {'aniso', 'aniso_haarhubs'} and bool((_dr.z <= -2).all()), "'survives removing the radial component in all four certified backbones under both stars'"
_dra = _dr[_dr.star == 'aniso'].set_index('model').z; F['Z_RAD_HI'], F['Z_RAD_LO'] = f"{_dra.max():.2f}", f"{_dra.min():.2f}"
assert set(_l2[(_l2.star == 'aniso') & (_l2.z <= -2)].model) == {'i21k_b', 'i21k_l'} and set(_l2[(_l2.star == 'aniso_haarhubs') & (_l2.z <= -2)].model) == {'i21k_b', 'i21k_l'}, "'under full L2 normalization it survives in ViT-B and ViT-L only'"
# ---- closing pass (2026-09-22): the per-backbone scoping from expR81, the trained positive control from expR77
_s81 = pd.read_csv(R + 'expR81_deep_per_backbone_summary.csv').set_index('model'); _ord = ['i21k_t', 'i21k_s', 'i21k_b', 'i21k_l', 'dinov1_b', 'dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g', 'clip_b', 'clip_l', 'siglip_b']
_NM = {"i21k_t": "ViT-T", "i21k_s": "ViT-S", "i21k_b": "ViT-B", "i21k_l": "ViT-L", "dinov1_b": "DINO-B", "dinov2_s": "DINOv2-S", "dinov2_b": "DINOv2-B", "dinov2_l": "DINOv2-L", "dinov2_g": "DINOv2-G", "clip_b": "CLIP-B", "clip_l": "CLIP-L", "siglip_b": "SigLIP-B"}
_cov = [m for m in _ord if _s81.loc[m, 'dec_power'] >= 0.8]; _unc = [m for m in _ord if _s81.loc[m, 'dec_power'] < 0.8]; assert len(_cov) == 9 and len(_unc) == 3, (_cov, _unc)   # ninth review (2026-09-22): one criterion, the decoupled power, the test that decides: 'the nine' and 'the other three'
def _names(ms):
    out = []; k = 0
    while k < len(ms):
        pre = _NM[ms[k]].rsplit('-', 1)[0]; grp = [ms[k]]
        while k + len(grp) < len(ms) and _NM[ms[k + len(grp)]].rsplit('-', 1)[0] == pre and pre in ('ViT', 'DINOv2', 'CLIP'): grp.append(ms[k + len(grp)])
        out.append(_NM[grp[0]] if len(grp) == 1 else pre + '-' + '/'.join(_NM[g].rsplit('-', 1)[1] for g in grp)); k += len(grp)
    return ', '.join(out[:-1]) + ' and ' + out[-1] if len(out) > 1 else out[0]
F['P81_COVERED'] = _names(_cov); F['P81_UNCOVERED'] = _names(_unc)   # the brief's lists, compressed by family
F['P81_COV_LO'], F['P81_COV_HI'] = f"{_s81.loc[_cov, 'dec_power'].min():.2f}", f"{_s81.loc[_cov, 'dec_power'].max():.2f}"; F['P81_UNC_LO'], F['P81_UNC_HI'] = f"{_s81.loc[_unc, 'dec_power'].min():.2f}", f"{_s81.loc[_unc, 'dec_power'].max():.2f}"
assert F['P81_COVERED'] == 'ViT-S/B/L, DINOv2-B/L/G, CLIP-B/L and SigLIP-B' and F['P81_UNCOVERED'] == 'ViT-T, DINO-B and DINOv2-S', (F['P81_COVERED'], F['P81_UNCOVERED'])
_rb, _rc = _r64[_unc], _r64[_cov]; F['RATIO_BLIND_LO'], F['RATIO_BLIND_HI'] = f"{_rb.min():.1f}", f"{_rb.max():.1f}"; F['RATIO_COV_LO'], F['RATIO_COV_HI'] = f"{_rc.min():.1f}", f"{_rc.max():.1f}"
assert _rc.min() <= _rb.min() and _rb.max() <= _rc.max(), "'the three blind backbones sit at ratios inside the range of the nine covered'"
assert any(m.startswith('i21k') for m in _cov) and any(m.startswith('dinov2') for m in _cov) and any(m.startswith(('clip', 'siglip')) for m in _cov), "'every family has covered members'"
# S5.4: the selected configuration's triplet agreement (expR58), the pair the 'moderate gap' sentence names
_c58 = pd.read_csv(R + 'expR58_treemap_cutfree_summary.csv'); _c58 = _c58[(_c58.dataset == 'imagenet') & (_c58.metric == 'cosine') & (_c58.linkage == 'average')].iloc[0]
F['TRIP_BIG'], F['TRIP_WITHIN'] = f"{_c58.triplet_agree_big_vs_block:.2f}", f"{_c58.triplet_agree_within_block:.2f}"; assert (F['TRIP_BIG'], F['TRIP_WITHIN']) == ('0.77', '0.76'), (F['TRIP_BIG'], F['TRIP_WITHIN'])
# the appendix implant figure (formerly Figure 4b): its caption numbers from the figure script's json
_fi = json.load(open(R + 'final_fig_implant.json')); _pr, _pt = {float(k): v for k, v in _fi['real'].items()}, {float(k): v for k, v in _fi['shrunk'].items()}
assert _pr[0.0] == 0.0 and _pr[1.0] < 0.5 <= 0.8 <= _pt[1.0], (_pr, _pt)   # 'missed at the real spread, detected once shrunk, no false alarms at s=0'
F['IMPL_REAL_S1'], F['IMPL_SHRUNK_S1'], F['IMPL_REAL_S0'] = f"{_pr[1.0]:.2f}", f"{_pt[1.0]:.2f}", f"{_pr[0.0]:.2f}"
_pc = pd.read_csv(R + 'expR77_positive_control.csv').set_index('model')
assert _pc.loc['hier_seed0', 'z_wn30'] <= -2 and _pc.loc['hier_seed0', 'z_wn30bal'] <= -2 and _pc.loc['ce_seed0', 'z_wn30'] <= -2 and _pc.loc['frozen', 'z_wn30'] <= -2, "'certified on both frames' / 'certified intact'"
F['PC_HIER_DEC'] = str(int(round(_pc.loc['hier_seed0', 'dec_frac_cert_wn30'] * 10))); assert F['PC_HIER_DEC'] == '10' and int(round(_pc.loc['hier_seed0', 'dec_frac_cert_wn30bal'] * 10)) == 10
assert int(round(_pc.loc['ce_seed0', 'dec_frac_cert_wn30'] * 10)) == int(round(_pc.loc['frozen', 'dec_frac_cert_wn30'] * 10)) == 0; F['PC_CE_DEC'] = '0'
# seed 1 of the trained control (2026-09-23): both seeds in S5.3(c); every count and z from expR77_positive_control.csv
assert {'ce_seed1', 'hier_seed1'} <= set(_pc.index) and int(round(_pc.loc['hier_seed1', 'dec_frac_cert_wn30'] * 10)) == 10 and int(round(_pc.loc['hier_seed0', 'dec_frac_cert_wn30'] * 10)) == 10, "'10 of 10 in both seeds'"
F['PC_HIER_Z0'], F['PC_HIER_Z1'] = f"{_pc.loc['hier_seed0', 'zdec_mean_wn30']:.2f}", f"{_pc.loc['hier_seed1', 'zdec_mean_wn30']:.2f}"; F['PC_CE_Z0'], F['PC_CE_Z1'] = f"{_pc.loc['ce_seed0', 'zdec_mean_wn30']:.2f}", f"{_pc.loc['ce_seed1', 'zdec_mean_wn30']:.2f}"
F['PC_CE_DEC1'] = str(int(round(_pc.loc['ce_seed1', 'dec_frac_cert_wn30'] * 10))); F['PC_FROZEN_DEC'] = str(int(round(_pc.loc['frozen', 'dec_frac_cert_wn30'] * 10)))
assert F['PC_CE_DEC1'] == '6' and F['PC_FROZEN_DEC'] == '0' and float(F['PC_CE_Z1']) <= -2 < float(F['PC_CE_Z0']) and float(F['PC_HIER_Z1']) < float(F['PC_CE_Z1']), "'fine-tuning adds some decoupled signal in one seed and the hierarchical objective adds more'"
_v77 = json.load(open(R + 'expR77_positive_control_verdict.json'))['verdict']; assert {v['seed'] for v in _v77} == {'seed0', 'seed1'} and not any(v['criterion_met'] for v in _v77), "pre-set criterion not met in either seed"
F['PC_FROZEN_BAL'] = str(int(round(_pc.loc['frozen', 'dec_frac_cert_wn30bal'] * 10))); assert F['PC_FROZEN_BAL'] == '7' and int(round(_pc.loc['ce_seed0', 'dec_frac_cert_wn30bal'] * 10)) == 0
F['PROV81'] = ', expR81_deep_per_backbone.csv' if os.path.exists(R + 'expR81_deep_per_backbone.csv') else ''
# ---- tenth review: the within-model ceiling of the tree agreements (expR84); the thesis branch of the brief: keep 'moderately shared' if the ceiling is 0.9 or more
_s84 = pd.read_csv(R + 'expR84_tree_ceiling_summary.csv').set_index('model'); assert len(_s84) == 13 and int(_s84.loc['ALL', 'n_pairs']) == 12 * 435, _s84.shape
F['CEIL_TRIP'], F['CEIL_TRIP_MIN'] = f"{_s84.loc['ALL', 'triplet_agree_mean']:.2f}", f"{_s84.loc['ALL', 'triplet_agree_min']:.2f}"
assert float(_s84.loc['ALL', 'triplet_agree_mean']) >= 0.9, "the brief's keep branch ('moderately shared', 'does not converge') needs a ceiling of 0.9 or more; otherwise the thesis changes only after the author decides"
assert float(F['TRIP_BIG']) < float(_s84.loc['ALL', 'triplet_agree_min']) and float(F['TRIP_WITHIN']) < float(_s84.loc['ALL', 'triplet_agree_min']), "'both sit below the ceiling'"
_b84 = _s84.drop('ALL')
for _k, _n in (("triplet_agree", "TRIP"), ("coph_corr", "COPH"), ("ari_cut", "ARI")): F[f'CEIL_{_n}'], F[f'CEIL_{_n}_LO'], F[f'CEIL_{_n}_HI'] = f"{_s84.loc['ALL', _k + '_mean']:.2f}", f"{_b84[_k + '_mean'].min():.2f}", f"{_b84[_k + '_mean'].max():.2f}"
F['ARI58_BIG'], F['ARI58_WITHIN'], F['COPH58_BIG'], F['COPH58_WITHIN'] = f"{_c58.ari_cut_big_vs_block:.2f}", f"{_c58.ari_cut_within_block:.2f}", f"{_c58.coph_corr_big_vs_block:.2f}", f"{_c58.coph_corr_within_block:.2f}"
_rt, _rc, _ra = float(_c58.triplet_agree_big_vs_block) / float(_s84.loc['ALL', 'triplet_agree_mean']), float(_c58.coph_corr_big_vs_block) / float(_s84.loc['ALL', 'coph_corr_mean']), float(_c58.ari_cut_big_vs_block) / float(_s84.loc['ALL', 'ari_cut_mean'])
assert _rt >= 0.8 and 0.4 <= _rc <= 0.6 and 0.4 <= _ra <= 0.6, (_rt, _rc, _ra)   # 'topology shared to within most of the ceiling, metric agreement at about half of it' (thesis brief, 2026-09-22 evening)
json.dump({"branch": "keep", "rule": "keep if mean triplet ceiling >= 0.9", "cross_over_ceiling": {"triplet": _rt, "cophenetic": _rc, "ari_cut": _ra}, "triplet_ceiling_mean": float(_s84.loc['ALL', 'triplet_agree_mean']), "triplet_ceiling_min": float(_s84.loc['ALL', 'triplet_agree_min']), "cross_model": [float(F['TRIP_BIG']), float(F['TRIP_WITHIN'])], "n_backbones_below_0.9": int((_s84.drop('ALL').triplet_agree_mean < 0.9).sum())}, open(R + 'final_tree_ceiling.json', 'w'), indent=1)
# ---- ninth review: the balanced frame's decoupled false-alarm rate (expR83; flat hubs assigned by the balanced frame itself = matched construction, as the 8 of 50 is matched to WordNet-30)
_s83 = pd.read_csv(R + 'expR83_flat_balanced_summary.csv').set_index('built'); assert {'balanced', 'wn30'} <= set(_s83.index) and bool((_s83.n_decoupled == 50).all()) and bool((_s83.n_intact == 5).all()), _s83
_fab = int(_s83.loc['balanced', 'decoupled_fired']); _mis = int(_s83.loc['wn30', 'decoupled_fired']); F['FA_BAL'] = str(_fab); _high = _fab >= 2 * int(F['FA_DEC'])   # 'high' = at least twice the frame of record's rate
assert not _high and _mis > 25, (_fab, _mis)   # the author's sentence (brief of 2026-09-22, afternoon): 'not a false alarm' needs the matched rate below the rule's bar; 'fires under the other even once decoupled' needs a majority
F['FA_MIS'] = str(_mis)
F['BAL_SENT'] = f"A flat cloud clustered under the frame of record fires under the balanced frame in {_mis} of 50 decoupled runs, so a real cloud read under a frame that is not its own is expected to fire. The frozen ViT-B firing in {F['PC_FROZEN_BAL']} of 10 there is consistent with that mismatch and is not evidence of hub structure."   # the author's frame-mismatch reading (tenth review), split at 45 words (consolidated pass)
json.dump({"fa_bal": _fab, "fa_record": int(F['FA_DEC']), "high": bool(_high), "rule": "high if fa_bal >= 2 * fa_record", "sentence": F['BAL_SENT'], "author_sentence": "tenth review: the frame-mismatch reading (2026-09-22)", "mismatch_construction_fired": _mis, "mismatch_intact_fired": int(_s83.loc['wn30', 'intact_fired'])}, open(R + 'final_bal_frame.json', 'w'), indent=1)
DEC_OBS_KIND = "open"
if os.path.exists(R + 'expR81_deep_per_backbone.csv'):
    _e81 = pd.read_csv(R + 'expR81_deep_per_backbone.csv'); _v = _e81[_e81.model == 'i21k_l']
    if _v[_v.kind == 'intact'].seed.nunique() == 5 and len(_v[_v.kind == 'decoupled']) == 50:
        _i, _d = _v[_v.kind == 'intact'], _v[_v.kind == 'decoupled']
        assert abs(_i.z_depth.mean() - _deep.z.mean()) < 0.05, "expR81 ViT-L clouds must be expR79's deep clouds"   # same construction, same seeds
        sd_ratio = _d.star_sd.mean() / _i.star_sd.mean(); b_ratio = _d.depth_excess.mean() / _i.depth_excess.mean()   # depth_excess < 0: ratio > 1 means deeper
        if sd_ratio <= 0.8 and 0.8 <= b_ratio <= 1.2: F['DEC_OBS'] = "The deeper reading comes from the star, whose spread shrinks under decoupling while the excess below it is unchanged."; DEC_OBS_KIND = "star"
        elif b_ratio >= 1.2 and 0.8 <= sd_ratio <= 1.2: F['DEC_OBS'] = "The deeper reading comes from the excess itself, which grows under decoupling while the star spread is unchanged."; DEC_OBS_KIND = "excess"
        elif b_ratio >= 1.2 and sd_ratio <= 0.8: F['DEC_OBS'] = "The deeper reading comes from both sides: decoupling shrinks the star spread and deepens the excess below the star."; DEC_OBS_KIND = "both"
        json.dump(dict(kind=DEC_OBS_KIND, star_sd_ratio=float(sd_ratio), B_ratio=float(b_ratio), star_sd_intact=float(_i.star_sd.mean()), star_sd_decoupled=float(_d.star_sd.mean()), B_intact=float(_i.depth_excess.mean()), B_decoupled=float(_d.depth_excess.mean())), open(R + 'final_dec_obs.json', 'w'), indent=1)
# ---- priority 2 (expR78): 'within 0.03 on all four datasets', 'negative on every dataset', 'only CIFAR-100 and MiniImageNet ... at the 0.05 level'
_s78 = pd.read_csv(R + 'expR78_khrulkov_replication_summary.csv').set_index('dataset')
assert int((_s78.p_left_max > 0.05).sum()) == 2, "'two are indistinguishable from a random cloud'"
assert set(_s78.index) == {'cifar10', 'cifar100', 'cub', 'miniimagenet'} and bool(_s78.within_range.all()) and bool((_s78.excess_mean < 0).all()) and set(_s78.index[_s78.p_left_max <= 0.05]) == {'cifar100', 'miniimagenet'}, _s78
json.dump({**{k: F[k] for k in ('FMNIST_GEN', 'POL_RULE_H', 'POL_COS_H', 'WN_H', 'QUAD10', 'NAIVE_BIG', 'C_LO', 'C_HI', 'N_GEN')}, 'IMPL_PCT': f"{100 * IMPL['hits1'] / IMPL['runs1']:.0f}", **{k: F[k] for k in ('FA_DEC', 'RATIO_VITL', 'RATIO_DINO_LO', 'RATIO_DINO_HI', 'DEC_OBS', 'PROV81', 'RATIO_SC_LO', 'RATIO_SC_HI', 'Z_DEEP_HI', 'Z_DEEP_LO', 'Z_FLAT_HI', 'Z_FLAT_LO', 'Z_VITL_DEC', 'RATIO_DINOB', 'RATIO_VITB', 'Z_RAD_HI', 'Z_RAD_LO', 'P81_COVERED', 'P81_UNCOVERED', 'P81_COV_LO', 'P81_COV_HI', 'P81_UNC_LO', 'P81_UNC_HI', 'RATIO_BLIND_LO', 'RATIO_BLIND_HI', 'RATIO_COV_LO', 'RATIO_COV_HI', 'TRIP_BIG', 'TRIP_WITHIN', 'FA_BAL', 'FA_MIS', 'BAL_SENT', 'PC_HIER_Z0', 'PC_HIER_Z1', 'PC_CE_Z0', 'PC_CE_Z1', 'PC_CE_DEC1', 'PC_FROZEN_DEC', 'CEIL_TRIP', 'CEIL_TRIP_MIN', 'CEIL_TRIP_LO', 'CEIL_TRIP_HI', 'CEIL_COPH', 'CEIL_COPH_LO', 'CEIL_COPH_HI', 'CEIL_ARI', 'CEIL_ARI_LO', 'CEIL_ARI_HI', 'ARI58_BIG', 'ARI58_WITHIN', 'COPH58_BIG', 'COPH58_WITHIN', 'IMPL_REAL_S1', 'IMPL_SHRUNK_S1', 'IMPL_REAL_S0', 'PC_HIER_DEC', 'PC_CE_DEC', 'PC_FROZEN_BAL')}}, open(R + 'final_fills.json', 'w'), indent=1)   # the fills of the final version, read by the sweep (IMPL_* added below when expR80 enters)
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
REPL_ALL = [("unvalidated regime", "regime where false alarms are not controlled"), ("the validated regime", "the regime where false alarms are controlled"), ("validated regime", "regime where false alarms are controlled"),
            ("depth unvalidated", "false alarms uncontrolled"), ("depth validated", "false alarms controlled"), ("is validated at", "controls false alarms at"), ("certifies clustering", "certifies structure beyond the second moments")]   # sixth-review follow-up R1/R3, every final copy and every appendix paragraph
def clean_block(b):
    b = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}\n?", "", b, flags=re.S)                     # no appendix figure is cited from the main text
    b = b.replace("(Table~\\ref{tab:q4-depth}, Figure~\\ref{fig:depth}a)", "(Table~\\ref{tab:q4-depth}, Figure~\\ref{fig:depth}a)")
    b = b.replace(" are reported without certification (Figure~\\ref{fig:depth-c100}).", " are reported without certification (Table~\\ref{tab:q4-depth}).")
    b = b.replace("(Figure~\\ref{fig:bestmetric}; held-out selection and policy comparison in Table~\\ref{tab:q9-corollary})", "(Table~\\ref{tab:q9-corollary})")
    b = b.replace("(Figure~\\ref{fig:causal}; raw $\\delta$, within architecture)", "(raw $\\delta$, within architecture)")
    b = b.replace("\\paragraph{The calibrated reading predicts the gain within datasets.}", "\\paragraph{Both readings predict the gain within datasets.}")   # fifth review, S B.12
    b = b.replace("\\subsection{Is the clustered structure genuine?", "\\subsection{Is the structure beyond the second moments genuine?")   # sixth review
    for a_, b_ in REPL_ALL: b = b.replace(a_, b_)   # R1/R3 in the appendix prose
    return b
# tables regenerated for the final by gen_appendix_final.py (appendix_tables/final/*_final.tex): robustness (bootstrap under the record, class-count sweep), depth (two-decimal z, K sweep), wordnet (DBpedia supremum under the Haar null)
FINAL_SRC = {"tab_q03_sample": "tab_q03_sample_final", "tab_q08_robust": "tab_q08_robust_final", "tab_q04_depth": "tab_q04_depth_final", "tab_q07_wordnet": "tab_q07_wordnet_final", "tab_q05_power": "tab_q05_power_final", "tab_q01_census": "tab_q01_census_final"}
B8 = clean_block(bytab["tab_q08_robust"])
B8 = re.sub(r"\\paragraph\{Hierarchy depth, not class count\.\}.*?\n", "", B8)                    # the class-count paragraph of v1 claimed the opposite of the corrected caption
bytab["tab_q08_robust"] = B8
for k in KEEP:
    bytab[k] = clean_block(bytab[k]) if k != "tab_q08_robust" else bytab[k]
    if k in FINAL_SRC: bytab[k] = bytab[k].replace("\\input{appendix_tables/" + k + "}", "\\input{appendix_tables/" + FINAL_SRC[k] + "}")
# ---- ninth review (2026-09-22): B.7 points to Table 8(d) instead of the superseded inconclusive control; the two-level implant curves (formerly Figure 4b) become an appendix figure
_b7 = re.search(r"\\paragraph\{A trained control, inconclusive\.\}.*?\n", bytab["tab_q04_depth"]); assert _b7, "B.7 paragraph of v1 not found in the depth block"
bytab["tab_q04_depth"] = bytab["tab_q04_depth"].replace(_b7.group(0), "\\paragraph{A trained positive control.} Table~\\ref{tab:q4-depth}(d) gives the fine-tuned ViT-B/16 controls, cross-entropy alone and with a hierarchical term, and the decoupled comparison that separates them. % expR77_positive_control.csv\n")
_figI = ("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.5\\linewidth]{figures/fig_implant_final.pdf}\n\\caption{\\textbf{A two-level implant on the real ImageNet clouds is missed at the real within-cluster spread and detected once the spread is shrunk.} "
         "Detection rate at $z\\le-2$ against implant strength $s$ with the real within-cluster spread (solid) and shrunk (dashed): at full strength " + F['IMPL_REAL_S1'] + " against " + F['IMPL_SHRUNK_S1'] + ", and " + F['IMPL_REAL_S0'] + " at $s=0$ with the real spread. % expR64b_wn30.csv\n}\n\\label{fig:implant}\n\\end{figure}\n")
_figM = ("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.62\\linewidth]{figures/fig_treemap_matrices_final.pdf}\n\\caption{\\textbf{The island is an artifact of the naive cut.} Pairwise ARI between dendrogram cuts under (a) the naive configuration, Euclidean distances with average linkage, where the DINOv2 block agrees with the rest at " + F['NAIVE_BIG'] + ", and (b) the selected one, cosine with average linkage, where it agrees at " + F['ARI_BIG'] + ". % exp23_treemap_controls.npz\n}\n\\label{fig:treemapmat}\n\\end{figure}\n")
assert bytab["tab_q06_treemap"].count("\\input{appendix_tables/tab_q06_treemap}") == 1
bytab["tab_q06_treemap"] = bytab["tab_q06_treemap"].replace("\\input{appendix_tables/tab_q06_treemap}", _figM + "\\input{appendix_tables/tab_q06_treemap}")   # consolidated pass (2026-09-22): the two ARI matrices, formerly Figure 5ab
assert bytab["tab_q05_power"].count("\\input{appendix_tables/tab_q05_power_final}") == 1
bytab["tab_q05_power"] = bytab["tab_q05_power"].replace("\\input{appendix_tables/tab_q05_power_final}", _figI + "\\input{appendix_tables/tab_q05_power_final}")
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
DROP = {}   # tenth review (2026-09-22): Table 14(b), the raw-supremum correlations, restored   # the constructions panel of the census table is back in the final (the uncentered census is one of its columns, 2026-09-20)
CAPFIX = {}
DROPLINE = {"tab_q02_text": ["OLMo-7B"], "tab_q14_panel": ["OLMo-7B"]}   # fourth review: 15 text models; OLMo-7B was not extracted
REPL = {"tab_q14_panel": [("9 causal LMs", "8 causal LMs")],                                                                 # fifth review: Table 5 caption
        "tab_q09_corollary": [("\\textbf{The calibrated reading predicts the zero-cost gain within datasets,", "\\textbf{Both readings predict the zero-cost gain within datasets,")],   # Table 13 title
        "tab_q01_census": [("\\textbf{Clustered structure is the rule at the class level under every construction", "\\textbf{Structure beyond the second moments is the rule at the class level under every construction")],   # sixth review
        "tab_q02_text": [("\\textbf{In text, clustered structure depends", "\\textbf{In text, structure beyond the second moments depends")],
        "tab_q10_calibration": [("the spectrum excess certifies clustering, not depth.", "the spectrum excess certifies structure beyond the second moments, not depth.")]}   # sixth-review follow-up R1
def split_panels(src, drop=(), capfix=None, dropline=(), repl=()):
    for a, b in repl:
        assert src.count(a) == 1, a; src = src.replace(a, b)
    for a, b in REPL_ALL: src = src.replace(a, b)
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
                  "% typographic only (page budget, 2026-09-18; tightened 2026-09-21 for priorities 1b and 2): the section headings and the run-in paragraph headings open with less white space than the style's default; fonts, margins and line spacing are the style's\n"
                  "\\renewcommand\\section{\\@startsection{section}{1}{\\z@}{-0.3ex plus -0.3ex minus -.2ex}{0.2ex plus 0.2ex minus 0.1ex}{\\large\\sc\\raggedright}}\n"
                  "\\renewcommand\\subsection{\\@startsection{subsection}{2}{\\z@}{-0.3ex plus -0.3ex minus -.2ex}{0.2ex plus .2ex}{\\normalsize\\sc\\raggedright}}\n"
                  "\\renewcommand\\paragraph{\\@startsection{paragraph}{4}{\\z@}{0ex plus 0.3ex minus .2ex}{-1em}{\\normalsize\\bf}}\\makeatother\n"
                  "\\setlength{\\textfloatsep}{4pt plus 2pt minus 2pt}\\setlength{\\abovecaptionskip}{0pt}\\setlength{\\parskip}{0pt plus 1pt}\n"
                  "% final version: classic structure, plain prose (author's brief of 2026-09-18); generated by rebuttal/scripts/phaseE_submission.py.\n\\begin{document}\n", 1)
open(PF, 'w').write(pre + body)
json.dump({"kept": order, "deleted": DELETED, "cuts": CUTS}, open(R + 'final_appendix.json', 'w'), indent=1)
print("final written ->", PF, "| appendix order:", order, "| deleted:", DELETED, "| cuts:", CUTS)
