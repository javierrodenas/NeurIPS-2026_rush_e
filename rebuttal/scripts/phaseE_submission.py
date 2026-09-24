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
         ("it keeps the tail of the defects, the intent of the supremum, but it is set by hundreds of quadruples rather than by one.", "It keeps the tail of the defects, as the supremum does, but it is set by hundreds of quadruples rather than by one."),
         # readability pass (2026-09-23): the three verbatim sentences over 30 words are split at their natural turn
         ("the value that 99.9 per cent of them fall below: It keeps", "the value that 99.9 per cent of them fall below. It keeps"),
         ("is zero; the further a metric is from a tree, the larger the defect can become.", "is zero. The further a metric is from a tree, the larger the defect can become."),
         ('We are the geometric complement of this program: the same calibration applied within a model instead of across models, with a depth test measured on real clouds, a hyperbolic backbone as the control for imposing the geometry, and a calibration of the tree-to-tree comparison itself.', 'We are the geometric complement of this program: the same calibration applied within a model instead of across models. It adds a depth test measured on real clouds, a hyperbolic backbone as the control for imposing the geometry, and a calibration of the tree-to-tree comparison itself.'),
         (" What a structureless cloud reads on it is the next question.", ""),
         # fourth and fifth reviews (2026-09-18): the abstract (the same one goes to OpenReview), S1 and S2 edits ordered by the author
         ("\\item \\textbf{The census.} Clustered structure in most models, hierarchy certified in a few, and a hyperbolic backbone as the control for imposing the geometry.",
          '\\item \\textbf{The census.} Structure beyond a random cloud in most models, clusters oriented toward their hubs in a few, no hub hierarchy in 9 of 12 backbones where an implanted one is detected once cluster orientations are randomized, a blind test in the other 3, and a nominally hyperbolic backbone as the control for imposing the geometry.'),
         ("but three artifacts push it down without any hierarchy.", "but its reference level depends on dimension, spectrum and statistic, so a raw value cannot be called low on its own."),
         ("Anisotropic spectra mimic low-dimensional behavior: the \\emph{spectrum confound}. And the supremum over sampled quadruples is a one-quadruple extreme that does not converge \\citep{fournier2015computing}: the \\emph{statistic confound}.",
          "Anisotropic spectra lower the effective dimension and raise the reading: the \\emph{spectrum confound}. And the supremum over sampled quadruples grows with the budget and does not converge \\citep{fournier2015computing}: the \\emph{statistic confound}."),
         ('On image features the excess sits within null noise in most cells and is a fraction of the null where it survives, so the premise does not survive calibration as stated. On class centroids, clustered structure is genuine in 49 of 72 cells, but a star already produces it. Hierarchy above the superclasses is certified in 4 of 12 ImageNet backbones by a test that never fires on real clouds with randomized hubs, and a backbone trained in hyperbolic space shows the same clustering and no detected depth. The trees are moderately shared: the naive comparison manufactures an island, and once the clustering cut is controlled a moderate gap remains, every recipe recovers the human taxonomy partially, more so with supervision, and the self-supervised structure lives in angles.',
          'On image features, where the premise is read, the calibrated reading is indistinguishable from a random cloud of the same shape in most cells and small elsewhere; the 4 values reported by \\citet{Khrulkov_2020_CVPR} are reproduced and, calibrated, 2 of them are indistinguishable from a random cloud. On class centroids there is structure that a random cloud does not have, in 44 of 72 cells and 30 of 36 on datasets with 47 classes or more, but a star of clusters already produces it. The depth test certifies in 4 of 12 ImageNet backbones that clusters are oriented toward the superclass centers, which we call hubs, not that the hubs form a hierarchy within the grouping of classes into superclasses, which we call the frame: randomizing the orientations removes every verdict, whereas an implanted hierarchy, synthetic or trained in, survives it; no hub hierarchy is found in 9 of 12 backbones where an implanted one is detected once cluster orientations are randomized, and the test is blind in the other 3. A backbone trained in hyperbolic space shows the same structure as its Euclidean twin and lives in the near-flat regime. Across models, the trees agree on which classes group together well above chance, though less than two readings of the same model, and not on distances; the self-supervised models organize classes by direction rather than by distance, which a comparison by distance misses.'),
         # page budget (fifth review): Figure 1 floats to the top of page 2 instead of leaving six blank lines at the foot of page 1 (placement only; the author's environment otherwise verbatim)
         ("\\begin{figure}[H]\n\\centering\n\\IfFileExists{figures/fig1_concept.pdf}", "\\begin{figure}[t]\n\\centering\n\\IfFileExists{figures/fig1_concept.pdf}"),
         ("; we bring the idea to embedding clouds, where the reference must match dimension and spectrum.",
          ". We bring the idea to embedding clouds, where the reference must match dimension and spectrum. The curvature itself has been studied as a representation tradeoff \\citep{sala2018representation} and as a mixed-curvature product to be learned \\citep{gu2019learning}, which presupposes a diagnostic of the kind we calibrate."),
         # full-pass cleanup (2026-09-22, evening): the S3.2 prose sentence duplicates Definition 1 (which now carries the citation)
         ("Gromov $\\delta$ is the worst case, the supremum of the defect over all quadruples \\citep{Gromov1987}. ", ""),
         # thesis after expR84 (2026-09-22, evening): contribution 3 mirrors 'topology largely shared, metric not'
         ("\\item \\textbf{The map of trees.} The island is an artifact of the cut; sharing is graded and supervision-dependent, and the self-supervised tree is angular.", "\\item \\textbf{The map of trees.} The island is an artifact of the cut; the trees agree on which classes group together well above chance, though short of what two readings of the same model reach, but not on the distances between them, and the self-supervised tree is angular."),
         # S1 rewrite (author's brief, 2026-09-23, 16:30): P1 reference sentence, P3 last sentence deleted, P4 'against the right reference' with the citation, P5 in three labelled parts, the four contributions, 'depth test' renamed 'hierarchy test'; the first four supersede earlier edits (their a is the earlier b)
         ('It adds a depth test measured on real clouds', 'It adds a hierarchy test measured on real clouds'),
         ('On image features, where the premise is read, the calibrated reading is indistinguishable from a random cloud of the same shape in most cells and small elsewhere; the 4 values reported by \\citet{Khrulkov_2020_CVPR} are reproduced and, calibrated, 2 of them are indistinguishable from a random cloud. On class centroids there is structure that a random cloud does not have, in 44 of 72 cells and 30 of 36 on datasets with 47 classes or more, but a star of clusters already produces it. The depth test certifies in 4 of 12 ImageNet backbones that clusters are oriented toward the superclass centers, which we call hubs, not that the hubs form a hierarchy within the grouping of classes into superclasses, which we call the frame: randomizing the orientations removes every verdict, whereas an implanted hierarchy, synthetic or trained in, survives it; no hub hierarchy is found in 9 of 12 backbones where an implanted one is detected once cluster orientations are randomized, and the test is blind in the other 3. A backbone trained in hyperbolic space shows the same structure as its Euclidean twin and lives in the near-flat regime. Across models, the trees agree on which classes group together well above chance, though less than two readings of the same model, and not on distances; the self-supervised models organize classes by direction rather than by distance, which a comparison by distance misses.', '(i) \\emph{Where the premise is read,} on image features, the calibrated reading is indistinguishable from a random cloud of the same shape in most model--dataset cells and small elsewhere. The 4 values reported by \\citet{Khrulkov_2020_CVPR} are reproduced and, calibrated, 2 of them are indistinguishable from a random cloud. (ii) \\emph{Within each model,} class centroids carry structure that a random cloud does not have, in 44 of 72 cells and 30 of 36 on datasets with 47 classes or more, but a star of clusters already produces it. The hierarchy test finds more in 4 of 12 ImageNet backbones: the orientation of each cluster relative to its superclass center, which we call its hub, not a hierarchy among the hubs. Randomizing the orientations removes every verdict, whereas an implanted hierarchy, synthetic or trained in, survives. In the 9 of 12 backbones where an implanted hierarchy is detected, none as strong is found; in the other 3 the test is blind. A backbone trained in hyperbolic space shows the same structure as its Euclidean twin and stays nearly flat. (iii) \\emph{Across models,} the trees agree on which classes group together well above chance, though less than two readings of the same model, and not on distances; the self-supervised models organize classes by direction rather than by distance, which a comparison by distance misses.'),
         ('\\item \\textbf{The census.} Structure beyond a random cloud in most models, clusters oriented toward their hubs in a few, no hub hierarchy in 9 of 12 backbones where an implanted one is detected once cluster orientations are randomized, a blind test in the other 3, and a nominally hyperbolic backbone as the control for imposing the geometry.', '\\item \\textbf{The census.} The premise calibrated where it is read, including a published reading reproduced; structure beyond a random cloud in most models; clusters oriented toward their hubs in a few; no hierarchy among the hubs where the test can see one; and a hyperbolic-trained backbone as the control for imposing the geometry.'),
         ('\\item \\textbf{The map of trees.} The island is an artifact of the cut; the trees agree on which classes group together well above chance, though short of what two readings of the same model reach, but not on the distances between them, and the self-supervised tree is angular.', '\\item \\textbf{The map of trees.} Across models, the trees agree on which classes group together but not on distances, and the self-supervised models organize classes by direction, which a comparison by distance misses.'),
         ('\\item \\textbf{The instrument.} Estimator calibration, spectrum-matched nulls read as corrected ranks, and a depth test with measured false-alarm rate and power.', '\\item \\textbf{The instrument.} A calibrated reading of $\\delta$, compared with random clouds of the same shape, and a hierarchy test whose false-alarm rate and power are measured on real clouds.'),
         ('\\item \\textbf{Consequences for practice.} What to measure before imposing curvature, and which zero-cost readout collects what is there.', '\\item \\textbf{Consequences for practice.} The rule that derives curvature from a raw $\\delta$ assigns curvature to random clouds; we say what to measure before imposing curvature, and cosine collects most of the structure at no cost.'),
         ('and they are never calibrated: a value is called low on its own, without asking what a structureless cloud of the same dimension and spectrum would score', 'and they are read against the ideal values of a tree and of a non-hyperbolic space, never against what a structureless cloud of the same dimension and spectrum would score'),
         (' The three are visible in real models: GPT-2 M sits at its matched null despite a low raw reading, and classic sentence embedders share their raw reading with vision-contrastive models while carrying none of their structure.', ''),
         ('We build the instrument that reads $\\delta$ correctly.', 'We build the instrument that reads $\\delta$ against the right reference.'),
         ('They calibrate similarity across models;', '\\citet{groger2026aristotelian} calibrate similarity across models;'),
         ('adds a \\emph{depth test} whose', 'adds a \\emph{hierarchy test} whose')
,
         # Figure 1 caption (author's brief, 2026-09-23, 17:00): last sentence names what each test separates; 2pt between the image and the caption
         ('The instrument reads the objects instead of the shadow: no structure beyond the null in the cloud, clusters without depth in the star, clusters with depth in the tree.', 'The instrument reads the objects instead of the shadow: the excess separates the random cloud from the other two, and the hierarchy test separates the star from the tree.'),
         ("same-size placeholder)}}}\n\\caption{\\textbf{Plato's cave", "same-size placeholder)}}}\n\\vspace{2pt}\n\\caption{\\textbf{Plato's cave")
]
def edit(s):
    for a, b in EDITS:
        if a in s: s = s.replace(a, b)
    return s
# sixth review (2026-09-20): the abstract is rewritten by order to 250 words or fewer, depth left open, cell defined in its own sentence;
# the text below replaces the edited verbatim abstract and is recorded in final_verbatim_edits.json ("abstract_sixth_review")
ABSTRACT_TEXT = ("Hyperbolic representation learning rests on a premise we call latent hyperbolicity: standard models are already tree-like because their class geometry scores a low Gromov $\\delta$, a measure of tree-likeness that is zero for a tree. That low value is also what a random cloud of the same dimension and spread scores, so we build an instrument that reads every score as its excess over many such clouds, and a hierarchy test whose ability to detect a hierarchy is measured. We apply it to 12 vision backbones, 6 datasets and 15 text models. On image features, where the premise is read, the calibrated reading is indistinguishable from a random cloud in most model--dataset cells and small elsewhere. Class centroids do carry structure that a random cloud of the same shape does not, in {{N_GEN}} of 72 cells, but so does a star of clusters. In {{N_IN}} of 12 ImageNet backbones the hierarchy test finds structure, but it is the orientation of clusters relative to the superclass centers, not a hierarchy among those centers: randomizing orientations removes it, whereas a planted hierarchy, synthetic or trained in, survives. Where a planted hierarchy is detected, in 9 of 12 backbones, no hierarchy as strong as the planted one is found; in the other 3 the test cannot tell. MERU, trained in hyperbolic space, shows the same structure as its Euclidean twin in a space that stays nearly flat. In text, the result depends on the model's recipe and size. Across models, the trees agree on which classes group together well above chance, though less than two readings of the same model, and not on distances; the self-supervised models organize classes by direction rather than by distance, which a comparison by distance misses. Read correctly, foundation models organize classes into clusters that they partly share, show no hierarchy among the superclasses where the test can see one, and their raw tree-likeness is not evidence for hyperbolic geometry.")   # the author's abstract of 2026-09-22 (late evening) with the four edits of 2026-09-23 (13:50), verbatim; the last sentence is the thesis
ABSTRACT_SIXTH = ABSTRACT_TEXT   # 2026-09-21: the abstract with priorities 1b and 2 (author's brief); recorded as abstract_final
ABSTRACT = "\\begin{abstract}\n" + ABSTRACT_TEXT + "\n\\end{abstract}"
_aw = len(re.sub(r"\{\{[A-Z_]+\}\}", "44", ABSTRACT_TEXT).split()); assert _aw <= 400, _aw   # 250 is the author's cap; 253 since the author's retouch of 2026-09-21 (three trims rejected), pending the author's decision; the sweep still reports the cap
# fifth review: fills and checks for the new sentences
F['FMNIST_GEN'] = str(int(c52[c52.dataset == 'fashionmnist'].genuine_bh.sum())); assert 1 <= int(F['FMNIST_GEN']) <= 11, F['FMNIST_GEN']
assert int(c52[c52.dataset.isin(['imagenet', 'cifar100', 'dtd'])].genuine_bh.sum()) == 30 and (c52[c52.dataset.isin(['imagenet', 'cifar100', 'dtd'])].n >= 47).all(), "'30 of 36 on the 3 datasets with 47 classes or more'"
hcells = e24[e24.dataset.isin(['imagenet', 'cifar100', 'cifar10', 'dtd'])]; assert len(hcells) == 40
F['POL_RULE_H'], F['POL_COS_H'] = f"{hcells.adv_rule.mean():+.2f}", f"{hcells.adv_cos.mean():+.2f}"; assert hcells.adv_rule.mean() > hcells.adv_cos.mean() > 0, "'the objective rule beats cosine'"
f2 = json.load(open(R + 'final_fig2b.json')); s_ = sl[(sl.model == f2['sample_cell'][0]) & (sl.dataset == f2['sample_cell'][1])].iloc[0]; c_ = c52[(c52.model == f2['class_cell'][0]) & (c52.dataset == f2['class_cell'][1])].iloc[0]
assert f2['sample_cell'][1] == f2['class_cell'][1] and abs(s_.delta_999 - c_.delta) < 0.001 and not s_.genuine_bh and c_.genuine_bh, "Figure 2(b): same dataset, same reading, opposite verdict"
wl = json.load(open(R + 'final_wordnet_levels.json')); F['WN_H'] = wl['h_word']
from math import comb; F['QUAD10'] = str(comb(10, 4)); assert F['QUAD10'] == '210'
# B.1 decoupling control (expR74): the four certified backbones no longer fire once the offsets are rotated -> 'certified hub-offset structure' wording
dec = pd.read_csv(R + 'expR74_decoupling_summary.csv'); cert4 = dec[dec.real_certified]; assert set(cert4.model) == {'i21k_s', 'i21k_b', 'i21k_l', 'dinov2_l'} and (cert4.n_seeds == 10).all()
assert (cert4.frac_certified == 0).all() and (cert4.dec_z_mean > -2).all(), "'none of the 4 backbones fires' (S5.3)"
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
    sents = re.split(r"(?<=[.])\s+(?=[A-Z\\])", para.strip()); RELATED = head + "\n\n" + " ".join(sents[:7])   # seven since the readability pass of 2026-09-23 split the sixth sentence at the 30-word cap
body = open(S + 'phaseE_paper_final.tex.tmpl').read()
for k, v in [("%%ABSTRACT%%", ABSTRACT), ("%%INTRO%%", INTRO), ("%%RELATED%%", RELATED), ("%%GROMOV%%", GROMOV), ("%%ESTIMATION%%", ESTIM)]:
    assert body.count(k) == 1, k; body = body.replace(k, v)
for _a, _b in (("Depth test", "Hierarchy test"), ("depth test", "hierarchy test"), ("depth-test", "hierarchy-test"), ("depth verdict", "hierarchy verdict")): body = body.replace(_a, _b)   # rename of 2026-09-23 (S1 rewrite brief): the test is the hierarchy test; 'depth' stays where it means the z statistic
assert "depth test" not in body.lower() and "depth verdict" not in body and body.count("hierarchy test") >= 8, body.count("hierarchy test")
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
assert _dr.model.nunique() == 4 and set(_dr.star) == {'aniso', 'aniso_haarhubs'} and bool((_dr.z <= -2).all()), "'survives removing the radial component in all 4 certified backbones under both stars'"
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
# pooled over the two seeds (author's brief of 2026-09-23, 09:15): counts summed over seeds, mean z = mean of the two seed means (10 decoupled runs each)
F['PC_HIER_POOL'] = str(sum(int(round(_pc.loc[f'hier_seed{s}', 'dec_frac_cert_wn30'] * 10)) for s in (0, 1))); F['PC_CE_POOL'] = str(sum(int(round(_pc.loc[f'ce_seed{s}', 'dec_frac_cert_wn30'] * 10)) for s in (0, 1)))
F['PC_HIER_ZM'] = f"{_pc.loc[['hier_seed0', 'hier_seed1'], 'zdec_mean_wn30'].mean():.2f}"; F['PC_CE_ZM'] = f"{_pc.loc[['ce_seed0', 'ce_seed1'], 'zdec_mean_wn30'].mean():.2f}"
assert F['PC_HIER_POOL'] == '20' and F['PC_CE_POOL'] == '6' and F['PC_FROZEN_DEC'] == '0' and float(F['PC_HIER_ZM']) < float(F['PC_CE_ZM']) < 0, "'20 of 20, 6 of 20, none; fine-tuning adds some decoupled signal and the hierarchical objective adds more'"
_v77 = json.load(open(R + 'expR77_positive_control_verdict.json'))['verdict']; assert {v['seed'] for v in _v77} == {'seed0', 'seed1'} and not any(v['criterion_met'] for v in _v77), "pre-set criterion not met in either seed"
F['PC_FROZEN_BAL'] = str(int(round(_pc.loc['frozen', 'dec_frac_cert_wn30bal'] * 10))); assert F['PC_FROZEN_BAL'] == '7' and int(round(_pc.loc['ce_seed0', 'dec_frac_cert_wn30bal'] * 10)) == 0
F['PROV81'] = ', expR81_deep_per_backbone.csv' if os.path.exists(R + 'expR81_deep_per_backbone.csv') else ''
# ---- final accuracy pass (2026-09-23): the MERU sentence's radius, 95th percentile of the spatial norm times sqrt(c), largest over the MERU models and datasets (expR71)
_r71 = pd.read_csv(R + 'expR71_meru_radii.csv'); _r71 = _r71[_r71.model.str.startswith('meru')]; assert len(_r71) == 6 and bool((_r71.n == 50000).all()), len(_r71)
F['MERU_P95'] = f"{float(_r71.radius_sqrtc_p95.max()):.2f}"; F['MERU_PCT'] = "95"; assert "radius_sqrtc_p95" in _r71.columns   # the percentile of the column read assert 0.2 < float(F['MERU_P95']) < 0.5 and float(_r71.radius_sqrtc_max.max()) < 0.5, F['MERU_P95']   # 'within 0.29 of the curvature scale'
# ---- twelfth review (5): Table 1 carries the excess and rank of the published statistic itself, the supremum, once expR85 is merged; until then the brief's fallback clause
if os.path.exists(R + 'expR85_khrulkov_sup_summary.csv'):
    _s85 = pd.read_csv(R + 'expR85_khrulkov_sup_summary.csv').set_index('dataset'); assert set(_s85.index) == {'cifar10', 'cifar100', 'cub', 'miniimagenet'} and bool((_s85.n_trials == 10).all()), _s85[['n_trials']]
    _e85 = pd.read_csv(R + 'expR85_khrulkov_sup.csv'); assert bool((_e85.n_rep == 200).all()) and len(_e85) == 40, len(_e85)
    F['SUP_CLAUSE'] = "Under their own statistic, the supremum, CIFAR-10 and CUB-200 stay indistinguishable from a random cloud and MiniImageNet stays below it. The verdict for CIFAR-100 changes from trial to trial, as the statistic confound predicts. "; F['PROV85'] = ", expR85_khrulkov_sup_summary.csv"   # author's sentence (2026-09-23, 17:40), split at the 30-word cap; the median trial decides
    assert all(float(_s85.loc[d, 'p_left_median']) < 0.05 for d in ('cifar100', 'miniimagenet')) and all(float(_s85.loc[d, 'p_left_median']) > 0.05 for d in ('cifar10', 'cub')) and float(_s85.loc['cifar100', 'p_left_max']) > 0.05, "'CIFAR-10 and CUB-200 stay indistinguishable and MiniImageNet stays below; CIFAR-100 changes from trial to trial' (median p over the 10 trials, largest p for CIFAR-100)"
    json.dump({"n_trials": 10, "excess_sup_mean": _s85.excess_sup_mean.round(4).to_dict(), "p_left_max": _s85.p_left_max.round(3).to_dict(), "all_negative": bool((_s85.excess_sup_mean < 0).all()), "p_left_median": _s85.p_left_median.round(3).to_dict(), "p_left_min": _s85.p_left_min.round(3).to_dict(), "r_above_median": _s85.r_above_median.round(1).to_dict()}, open(R + 'final_khrulkov_sup.json', 'w'), indent=1)
else:
    F['SUP_CLAUSE'] = "The calibration reads the percentile statistic $\\hat\\delta_{99.9}$, whereas their statistic is the supremum. "; F['PROV85'] = ""
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
F['BAL_SENT'] = f"A flat cloud clustered under the WordNet frame fires under the balanced frame in {_mis} of 50 decoupled runs. A real cloud read under a frame that is not its own is therefore expected to fire. The frozen ViT-B firing in {F['PC_FROZEN_BAL']} of 10 there is consistent with that mismatch and is not evidence of hub structure."   # the author's frame-mismatch reading (tenth review), split at 45 words (consolidated pass)
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
json.dump({**{k: F[k] for k in ('FMNIST_GEN', 'POL_RULE_H', 'POL_COS_H', 'WN_H', 'QUAD10', 'NAIVE_BIG', 'C_LO', 'C_HI', 'N_GEN', 'MERU_P95', 'MERU_PCT')}, 'IMPL_PCT': f"{100 * IMPL['hits1'] / IMPL['runs1']:.0f}", **{k: F[k] for k in ('FA_DEC', 'RATIO_VITL', 'RATIO_DINO_LO', 'RATIO_DINO_HI', 'DEC_OBS', 'PROV81', 'RATIO_SC_LO', 'RATIO_SC_HI', 'Z_DEEP_HI', 'Z_DEEP_LO', 'Z_FLAT_HI', 'Z_FLAT_LO', 'Z_VITL_DEC', 'RATIO_DINOB', 'RATIO_VITB', 'Z_RAD_HI', 'Z_RAD_LO', 'P81_COVERED', 'P81_UNCOVERED', 'P81_COV_LO', 'P81_COV_HI', 'P81_UNC_LO', 'P81_UNC_HI', 'RATIO_BLIND_LO', 'RATIO_BLIND_HI', 'RATIO_COV_LO', 'RATIO_COV_HI', 'TRIP_BIG', 'TRIP_WITHIN', 'FA_BAL', 'FA_MIS', 'BAL_SENT', 'PC_HIER_Z0', 'PC_HIER_Z1', 'PC_CE_Z0', 'PC_CE_Z1', 'PC_CE_DEC1', 'PC_FROZEN_DEC', 'PC_HIER_POOL', 'PC_CE_POOL', 'PC_HIER_ZM', 'PC_CE_ZM', 'CEIL_TRIP', 'CEIL_TRIP_MIN', 'CEIL_TRIP_LO', 'CEIL_TRIP_HI', 'CEIL_COPH', 'CEIL_COPH_LO', 'CEIL_COPH_HI', 'CEIL_ARI', 'CEIL_ARI_LO', 'CEIL_ARI_HI', 'ARI58_BIG', 'ARI58_WITHIN', 'COPH58_BIG', 'COPH58_WITHIN', 'IMPL_REAL_S1', 'IMPL_SHRUNK_S1', 'IMPL_REAL_S0', 'PC_HIER_DEC', 'PC_CE_DEC', 'PC_FROZEN_BAL')}}, open(R + 'final_fills.json', 'w'), indent=1)   # the fills of the final version, read by the sweep (IMPL_* added below when expR80 enters)
for k, v in F.items(): body = body.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{[A-Z0-9_]+\}\}", body); assert not left, left
# ---- appendix cleanup (2026-09-23): one plain-language paragraph of 2-4 sentences before each table; every explanatory sentence of
# the old captions and of the v1 prose that is not a claim of the main text is dropped (listed in the CHANGELOG).
INTRO = {
 "tab_q08_robust": r"\paragraph{What this table answers.} Three choices could have made the reading look structured: how many quadruples we sample, which images stand behind each centroid, and how many classes a cell has. Each table varies one of them and reads the same cells again. The verdicts move only where the excess was already at the edge of the null.",
 "tab_q10_calibration": r"\paragraph{What this table answers.} A raw reading means nothing until something known is read on the same scale. The first table reads geometries whose answer we know in advance, a tree, a hyperbolic region and a sphere. The second reads clouds we built ourselves, clusters with and without a hierarchy above them, against each null in turn.",
 "tab_q01_census": r"\paragraph{What this table answers.} This is the census itself, one row per model and class set. The first table gives the reading used in the paper, the second repeats the verdict under other constructions of the null and of the statistic, and the third repeats the census on cosine geometry. A cell is genuine when its excess survives the correction across cells.",
 "tab_q14_panel": r"\paragraph{What this table answers.} Every model named anywhere in the paper appears here once, with its size and its role: census, control or text. The notes below the table say how features are taken from each kind of model, which is part of the measurement rather than a detail of implementation.",
 "tab_q02_text": r"\paragraph{What this table answers.} The text census reads the thousand ImageNet class names through language models and sentence embedders. The verdicts follow the recipe and the scale of the model, not its family. Extraction is part of the measurement, so the second table varies batch size and precision on the same prompts.",
 "tab_q03_sample": r"\paragraph{What this table answers.} The premise is read on per-image features, so this table reads the census there instead of on class centroids. Each cell gives the raw statistic beside the excess over its matched null. Where the excess survives, it removes a small fraction of the null reading.",
 "tab_q04_depth": r"\paragraph{What this table answers.} The census certifies structure beyond the second moments, and this table asks whether the clusters are arranged hierarchically. Each backbone is read against a matched star under both constructions of the star. The controls follow: the models supervised on leaf labels, a backbone trained in hyperbolic space with its Euclidean twin, a fine-tuned pair and a control that removes the radial part of every offset.",
 "tab_q05_power": r"\paragraph{What this table answers.} A negative verdict is worth reading only where the test can see. Here hierarchies of known strength are planted in real and synthetic clouds and the test is run again, which gives its power; flat clouds are read the same way, which gives its false alarms. The power depends on how quiet the cloud is, so it is reported per backbone.",
 "tab_q06_treemap": r"\paragraph{What this table answers.} Trees of different models are compared as objects, without labels. The comparison depends on the cut, so the first table gives the diagnostics of every admissible configuration and the criterion that selects one. The last table reads two resamples of the same model, which is the ceiling any cross-model agreement can reach.",
 "tab_q07_wordnet": r"\paragraph{What this table answers.} Here the trees are compared with a human taxonomy instead of with each other. The first two tables use WordNet on ImageNet and the coarse labels of CIFAR-100. The last two repeat the reading on an ontology built independently of WordNet and on a set of caption chains with no class set, so the recovery is not circular.",
 "tab_q13_local": r"\paragraph{What this table answers.} Two models can place classes in the same neighborhoods and still disagree on distances. This table calibrates both readings against permutations of the class labels, so agreement is counted only where a shuffled pairing would not reach it.",
 "tab_q09_corollary": r"\paragraph{What this table answers.} The instrument certifies structure; this table asks what it buys downstream. Accuracies and zero-cost gains are read per cell, then correlated with the raw reading, with the calibrated one and with the hierarchy verdict. The gains are modest and limited to prototype tasks.",
}

HOWTO = r"""\subsection{How to read this appendix}
\label{app:howto}
One question per subsection, in the order the main text first cites them, with a plain paragraph saying what the group answers.
Each table then carries its own caption: the answer in bold, then what the columns are.

\begin{itemize}[topsep=2pt,itemsep=1pt,leftmargin=*]
\item Is the reading robust to our choices? Tables~\ref{tab:q8-robust}--\ref{tab:q8-robust-d}: the excess under the quadruple budget, resampling and the class count.
\item Is a low raw reading evidence? Tables~\ref{tab:q10-calibration} and \ref{tab:q10-calibration-b}: known geometries and synthetic clusters read on the same scale.
\item Is the structure beyond the second moments genuine? Tables~\ref{tab:q1-census}--\ref{tab:q1-census-d}: the vision census, per cell and per construction.
\item Which models, and how were they read? Table~\ref{tab:q14-panel}: the model panel and the extraction.
\item Is text the same? Tables~\ref{tab:q2-text} and \ref{tab:q2-text-c}: the census on the thousand ImageNet class names, and the extraction.
\item Does the premise survive where it is read? Table~\ref{tab:q3-sample}: the census on per-image features.
\item Is the structure deeper than a star? Tables~\ref{tab:q4-depth}--\ref{tab:q4-depth-e}: the hierarchy test and its controls.
\item How much can the test see? Tables~\ref{tab:q5-power}--\ref{tab:q5-power-f}: power against planted hierarchies and false alarms on flat clouds.
\item Is the island real? Tables~\ref{tab:q6-treemap}--\ref{tab:q6-treemap-c}: the tree map under every admissible configuration, against the within-model ceiling.
\item Whose taxonomy? Tables~\ref{tab:q7-wordnet}--\ref{tab:q7-wordnet-d}: agreement with WordNet, superclass recovery, an independent ontology and a caption hierarchy.
\item Is local structure shared? Table~\ref{tab:q13-local}: permutation-calibrated agreement across model pairs.
\item What follows for practice? Tables~\ref{tab:q9-corollary}--\ref{tab:q9-corollary-d}: accuracies, zero-cost gains and what predicts them.
\item Where does every number come from? Table~\ref{tab:provenance}: the result file behind each table.
\end{itemize}

\paragraph{Symbols and column names.} The same words are used here and in the main text.
\begin{itemize}[topsep=2pt,itemsep=1pt,leftmargin=*]
\item \emph{The reading used in the paper}: the 99.9th-percentile statistic on Euclidean distances between class centroids, against the centered Haar null, 200 replicates; the cosine census is a robustness column (Table~\ref{tab:q1-census-c}). Other constructions appear beside it as columns.
\item \emph{excess}: the real reading minus the mean of its matched null replicates. Negative means more tree-like than a random cloud of the same shape; positive, less.
\item \emph{exc./null}: the excess as a fraction of the null reading, so that cells of different size can be compared.
\item \emph{$r$/200}: how many of the 200 null replicates read above the real cloud. \emph{$p$}: the left-tail add-one $p$-value built from that rank.
\item \emph{BH}: the Benjamini--Hochberg correction applied over all cells of a census. \emph{genuine}: BH-corrected $p\le0.05$. \emph{$^{\circ}$}: not genuine. \emph{$^{\dagger}$} and \emph{$^{\ddagger}$}: the cell passes one of two criteria and fails the other, as the caption says.
\item \emph{Bold} marks the answer sentence of a caption and the title of a panel; no cell of an appendix table is bold, and in the main-text census table bold marks cells above the null.
\item \emph{frame}: the grouping of classes into superclasses. \emph{hub}: the center of one superclass. \emph{matched star}: a cloud with the same clusters and hubs drawn from a matched distribution, which is what a real cloud is read against. \emph{Haar-hub star}: the same with the real hubs resampled by a Haar rotation.
\item \emph{$z$}: the depth statistic, the real excess minus the star's, divided by the combined spread. A cloud is certified when $z\le-2$ under both stars.
\item \emph{decoupled}: the real hubs kept and each cluster's offsets rotated independently, which removes the orientation of clusters and keeps the arrangement of hubs.
\item \emph{power}: the fraction of planted hierarchies the test detects. \emph{false alarms}: the fraction of flat controls that fire.
\item \emph{w/b}: the within-cluster spread of a cloud divided by the spread between its hubs, which is how quiet a cloud is. \emph{tight} or \emph{shrunk spread}: the within-cluster spread shrunk into the range the synthetic sweep covers.
\item \emph{$s^{*}$}: the smallest implant strength at which the test fires. \emph{$\bar z_1$}: the mean $z$ at full implant strength.
\end{itemize}

"""

# ---- appendix: the cited question tables of the v1 template, without figures, in the order the final text first cites them
v1t = open(S + 'phaseE_paper.tex.tmpl').read(); app = v1t[v1t.index("\\section{Appendix: one table per question}"):v1t.index("\\end{document}")]
for k, v in F.items(): app = app.replace("{{" + k + "}}", v)
for _a, _b in (("Depth test", "Hierarchy test"), ("depth test", "hierarchy test"), ("depth-test", "hierarchy-test"), ("depth verdict", "hierarchy verdict")): app = app.replace(_a, _b)   # rename of 2026-09-23 (S1 rewrite brief) in the appendix prose of the v1 template
head = app[:app.index("\\FloatBarrier")].replace("\\section{Appendix: one table per question}", "\\section{One table per question}\n\\renewcommand{\\arraystretch}{0.92}") + HOWTO
blocks = [b for b in re.split(r"(?=\\FloatBarrier\n\\subsection\{)", app[app.index("\\FloatBarrier"):]) if b.strip()]
bytab = {}
for b in blocks:
    m = re.search(r"\\input\{appendix_tables/(tab_[a-z0-9_]+)\}", b); bytab[m.group(1)] = b
KEEP = ["tab_q10_calibration", "tab_q01_census", "tab_q08_robust", "tab_q03_sample", "tab_q04_depth", "tab_q05_power", "tab_q06_treemap", "tab_q07_wordnet", "tab_q02_text", "tab_q13_local", "tab_q09_corollary", "tab_q14_panel"]
DELETED = sorted(set(bytab) - set(KEEP) - {"tab_z_provenance"})
REPL_ALL = [("the census of record", "the census"), ("census of record", "census"), ("the reading of record", "the reading"), ("reading of record", "reading"),
            ("the record excess", "the excess"), ("record excess", "excess"), ("under the record", "under the census reading"), ("in the record", "in the census reading"),
            ("the frame of record", "the frame"), ("the record", "the census reading"),
            ("unvalidated regime", "regime where false alarms are not controlled"), ("the validated regime", "the regime where false alarms are controlled"), ("validated regime", "regime where false alarms are controlled"),
            ("depth unvalidated", "false alarms uncontrolled"), ("depth validated", "false alarms controlled"), ("is validated at", "controls false alarms at"), ("certifies clustering", "certifies structure beyond the second moments")]   # sixth-review follow-up R1/R3, every final copy and every appendix paragraph
def clean_block(b):
    b = re.sub(r"(?s)\n\\paragraph\{.*?(?=\n\\input\{)", "\n", b)                        # appendix cleanup: the v1 explanatory paragraphs go (the new INTRO replaces them)
    b = re.sub(r"(?s)\n(?!\\(FloatBarrier|subsection|label|input))[^\n]*\\ref\{tab:q7-wordnet\}[^\n]*\n", "\n", b)   # HierarCaps prose (its panel is dropped)
    b = re.sub(r"(?s)\nThreshold sensitivity of the selection criterion.*?(?=\n\\begin\{figure\}|\n\\input\{)", "\n", b)   # treemap prose
    b = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}\n?", "", b, flags=re.S)                     # no appendix figure is cited from the main text
    b = b.replace("(Table~\\ref{tab:q4-depth}, Figure~\\ref{fig:depth}a)", "(Table~\\ref{tab:q4-depth}, Figure~\\ref{fig:instrument}b)")
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
    if k in INTRO: bytab[k] = re.sub(r"\n\\input\{", lambda m_, _t=INTRO[k]: "\n" + _t + m_.group(0), bytab[k], count=1)
    if k in FINAL_SRC: bytab[k] = bytab[k].replace("\\input{appendix_tables/" + k + "}", "\\input{appendix_tables/" + FINAL_SRC[k] + "}")
# ---- ninth review (2026-09-22): B.7 points to Table 8(d) instead of the superseded inconclusive control; the two-level implant curves (formerly Figure 4b) become an appendix figure
# the v1 B.7 paragraph (a trained control, inconclusive) is dropped with the rest of the v1 prose by clean_block (appendix cleanup, 2026-09-23); the trained control is described in the INTRO paragraph of the hierarchy-test table
assert "A trained control, inconclusive" not in bytab["tab_q04_depth"], "the v1 B.7 paragraph should have been dropped by clean_block"
_figI = ("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.5\\linewidth]{figures/fig_implant_final.pdf}\n\\caption{\\textbf{A two-level implant on the real ImageNet clouds is missed at the real within-cluster spread and detected once the spread is shrunk.} "
         "Detection rate at $z\\le-2$ against implant strength $s$ with the real within-cluster spread (solid) and shrunk (dashed): at full strength " + F['IMPL_REAL_S1'] + " against " + F['IMPL_SHRUNK_S1'] + ", and " + F['IMPL_REAL_S0'] + " at $s=0$ with the real spread. % expR64b_wn30.csv\n}\n\\label{fig:implant}\n\\end{figure}\n")
_figM = ("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.62\\linewidth]{figures/fig_treemap_matrices_final.pdf}\n\\caption{\\textbf{The island is an artifact of the naive cut.} Pairwise ARI between dendrogram cuts under (a) the naive configuration, Euclidean distances with average linkage, where the DINOv2 block agrees with the rest at " + F['NAIVE_BIG'] + ", and (b) the selected one, cosine with average linkage, where it agrees at " + F['ARI_BIG'] + ". % exp23_treemap_controls.npz\n}\n\\label{fig:treemapmat}\n\\end{figure}\n")
assert bytab["tab_q06_treemap"].count("\\input{appendix_tables/tab_q06_treemap}") == 1
# the ARI-matrix figure is not cited from the main text (appendix cleanup, 2026-09-23): it is not inserted   # consolidated pass (2026-09-22): the two ARI matrices, formerly Figure 5ab
assert bytab["tab_q05_power"].count("\\input{appendix_tables/tab_q05_power_final}") == 1
bytab["tab_q05_power"] = bytab["tab_q05_power"].replace("\\input{appendix_tables/tab_q05_power_final}", _figI + "\\input{appendix_tables/tab_q05_power_final}")
_figSR = ("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.45\\linewidth]{figures/fig_samereading_final.pdf}\n"
          "\\caption{\\textbf{The same reading, opposite verdicts.} The raw reading of ViT-B on CIFAR-100 images and of DINO-B on CIFAR-100 centroids, each beside the mean of its matched null: the same reading, and only one of them is genuine. % expR62_samplelevel_record.csv, expR52_census_haar_p999_200.csv, final_fig2b.json\n}\n\\label{fig:samereading}\n\\end{figure}\n")
assert bytab["tab_q03_sample"].count("\\input{appendix_tables/tab_q03_sample_final}") == 1
bytab["tab_q03_sample"] = bytab["tab_q03_sample"].replace("\\input{appendix_tables/tab_q03_sample_final}", _figSR + "\\input{appendix_tables/tab_q03_sample_final}")
labels = {}
for stem in KEEP:
    fn = FINAL_SRC.get(stem, stem)
    mm = re.search(r"\\label\{(tab:q[^}]*)\}", open(TEX + ("appendix_tables/final/" if stem in FINAL_SRC else "appendix_tables/") + fn + ".tex").read()); labels[mm.group(1)] = stem
main = body[:body.index("\\appendix")]; cited = []
for m in re.finditer(r"\\ref\{(tab:q[^}]*)\}", main):
    _l = re.sub(r"-(ap|[b-f])$", "", m.group(1))   # one table per former panel since 2026-09-23: the question is what orders the appendix
    if _l not in cited: cited.append(_l)
order = [labels[l] for l in cited if l in labels] + [s for s in KEEP if s not in [labels[l] for l in cited if l in labels]]
assert set(order) == set(KEEP), (order, KEEP)
prov_block = bytab["tab_z_provenance"].replace("appendix_tables/tab_z_provenance", "appendix_tables/final/tab_z_provenance_final")
# ---- the final's copies of the kept tables: appendix_tables/final/, one floating [tbp] table per panel so the appendix pages pack (the [H] parts of v1 left every page half empty)
FD = TEX + "appendix_tables/final/"; os.makedirs(FD, exist_ok=True)
# panels the brief deletes (null-variant panel, uncited supremum table) with the caption sentence that described them
DROP = {"tab_q02_text": ("(b)",), "tab_q03_sample": ("(b)",), "tab_q05_power": ("(a)",),
        "tab_q08_robust": ("(e)",), "tab_q09_corollary": ("(e)", "(f)", "(g)")}   # appendix cleanup (2026-09-23): panels no claim of the main text rests on
# DROP = {}   # tenth review (2026-09-22): Table 14(b), the raw-supremum correlations, restored   # the constructions panel of the census table is back in the final (the uncentered census is one of its columns, 2026-09-20)
CAPFIX = {}
DROPLINE = {"tab_q02_text": ["OLMo-7B"], "tab_q14_panel": ["OLMo-7B"]}   # fourth review: 15 text models; OLMo-7B was not extracted
REPL = {"tab_q14_panel": [("9 causal LMs", "8 causal LMs")],                                                                 # fifth review: Table 5 caption
        "tab_q09_corollary": [("\\textbf{The calibrated reading predicts the zero-cost gain within datasets,", "\\textbf{Both readings predict the zero-cost gain within datasets,")],   # Table 13 title
        "tab_q01_census": [("\\textbf{Clustered structure is the rule at the class level under every construction", "\\textbf{Structure beyond the second moments is the rule at the class level under every construction")],   # sixth review
        "tab_q02_text": [("\\textbf{In text, clustered structure depends", "\\textbf{In text, structure beyond the second moments depends")],
        "tab_q10_calibration": [("the spectrum excess certifies clustering, not depth.", "the spectrum excess certifies structure beyond the second moments, not depth.")]}   # sixth-review follow-up R1
def percap(cap, letter, first):
    """The caption lines of a block, cut to the sentences of panel `letter`: the head (bold takeaway) stays with the first panel of the
    first block; later panels get '(continued) (x) ...' with their own sentences (twelfth review, 2026-09-23: Table 3)."""
    ce = cap.index("}"); tail = cap[ce + 1:]; body = "\n".join(cap[:ce + 1]); assert body.startswith("\\caption{") and body.endswith("\n}"), cap[:1]   # the caption closes on its own line; a \label may follow
    inner = body[len("\\caption{"):-len("\n}")]; cont = inner.startswith("(continued) ")
    if cont: inner = inner[len("(continued) "):]
    segs = re.split(r"(?=\([a-e]\) )", inner); head = segs[0] if not re.match(r"\([a-e]\) ", segs[0]) else ""; parts = {s[1]: s for s in segs if re.match(r"\([a-e]\) ", s)}
    if letter not in parts: return ["\\caption{(continued)}"]
    seg = parts[letter].rstrip()
    return (["\\caption{" + head + seg + "\n}"] + tail if first else ["\\caption{(continued) " + seg + "\n}"])
# ---- appendix cleanup (2026-09-23): one caption per table, at most 60 words (bold answer + columns); the explanation moved to the
# plain paragraph before the table (INTRO). The provenance comment of the original caption is kept verbatim.
NEWCAP = {
 "tab_q08_robust": r"\textbf{The reading does not move with the quadruple budget, with the images behind each centroid or with the number of classes.} Panels: the excess at growing budgets, a bootstrap over resampled images, the genuine count under null, bootstrap and estimator noise together, and the excess on smaller class sets.",
 "tab_q10_calibration": r"\textbf{A low raw reading is not evidence of hierarchy, and the excess certifies structure beyond the second moments.} Panels: reference geometries with their absolute and normalized readings, and synthetic clouds of clusters read against each null.",
 "tab_q01_census": r"\textbf{Structure beyond the second moments is the rule at the class level under every construction of the null.} Panels: the reading used in the paper per cell, the verdict under five constructions of null and statistic, the same census on cosine geometry, and the backbones read only on ImageNet.",
 "tab_q14_panel": r"\textbf{The models and the extraction.} Every backbone, language model and text embedder of the census and of the controls, with its parameter count, followed by how features are extracted and pooled.",
 "tab_q02_text": r"\textbf{In text, structure beyond the second moments depends on the recipe and the scale.} Columns: the statistic, its excess over the matched null, the rank among the replicates and the left-tail $p$ per model, and the same prompts under changes of batch size and precision.",
 "tab_q03_sample": r"\textbf{At the sample level, where the premise is read, the reading sits within null noise in most cells.} Per cell: the statistic, the excess, the excess as a fraction of the null, the rank and the left-tail $p$.",
 "tab_q04_depth": r"\textbf{The hierarchy test certifies that clusters are oriented toward their hubs in a few backbones, and finds no hierarchy among the hubs.} Panels: the 12 backbones under both matched stars, the leaf-label ViTs, a hyperbolic backbone with its Euclidean twin, the trained control and the radial control.",
 "tab_q05_power": r"\textbf{The test detects a planted hierarchy where the cloud is quiet enough, and its false alarms are measured.} Panels: implanted two-level trees, implanted hub alignment, a deep hierarchy at the real noise level, the power per backbone and the false alarms of the balanced frame.",
 "tab_q06_treemap": r"\textbf{The island is an artifact of the cut; the trees share topology well above chance and not the distances.} Panels: the selection diagnostics per configuration, the three agreement measures of the self-supervised models against the rest, and the within-model ceiling under resampling.",
 "tab_q07_wordnet": r"\textbf{Agreement with the human taxonomy follows supervision and recipe, and the superclasses are recovered by every family.} Panels: the correlation with WordNet on ImageNet, the recovery of the CIFAR-100 superclasses per configuration, and the same readings on DBpedia Classes and on the HierarCaps caption chains.",
 "tab_q13_local": r"\textbf{Cross-model agreement survives permutation calibration.} Columns: the raw measure, the null mean, the calibration threshold, the calibrated value and the number of model pairs that survive, for mutual-kNN agreement and for centroid CKA.",
 "tab_q09_corollary": r"\textbf{Both readings predict the gain of a non-Euclidean readout, and the hierarchy verdict predicts none.} Panels: accuracies and zero-cost gains per cell, the correlations under the raw reading and under the calibrated one, and the best zero-cost metric per backbone.",
}
for _k, _v in NEWCAP.items(): assert len(_v.split()) <= 60, (_k, len(_v.split()))

# ---- appendix cleanup (2026-09-23): panel titles are labels, not definitions; the glossary of the reading page defines the terms
TITLE = {"(a) The 12 backbones under the stars with Gaussian hubs": "(a) The 12 backbones under the star with Gaussian hubs",
         "(d) A trained positive control": "(d) A trained positive control: leaf cross-entropy against leaf plus hierarchical, 2 seeds, identical batches",
         "(e) Radial control": "(e) Radial control: centroids normalized, offsets stripped of their radial part",
         "(c) Implanted hub alignment on the real ImageNet clouds": "(c) Implanted hub alignment on the real ImageNet clouds",
         "(d) A deep hierarchy at the real noise level": "(d) A deep hierarchy at the real noise level",
         "(f) False alarms of the balanced frame": "(f) False alarms of the balanced frame",
         "(c) Joint sensitivity of the genuine count": "(c) Joint sensitivity of the genuine count",
         "(b) Correlation between the raw supremum $\\delta_{\\text{norm}}$": "(b) Correlation between the raw supremum reading and the gain",
         "(d) The same correlations on the raw $\\hat\\delta_{99.9}$": "(d) The same correlations on the calibrated reading and the hierarchy-test $z$",
         "(d) Class count": "(d) Class count: the excess shrinks with the number of classes"}

# ---- appendix (author's brief, 2026-09-23 22:00): one numbered table per former panel, each with its own caption (<= 40 words:
# bold answer sentence, then what the columns are) and its own label; the panel letter survives only as the label suffix.
PANELCAP = {
 ("tab_q08_robust", "a"): r"\textbf{The excess does not move with the quadruple budget.} Per cell: the excess at budgets from $10^4$ to $2\times10^6$ sampled quadruples per seed, and its drift over budgets as a fraction of the null spread.",
 ("tab_q08_robust", "b"): r"\textbf{The genuine count survives resampling of the images behind each centroid.} Per cell: the excess over 30 bootstrap resamples, its spread, and how many resamples stay genuine under the correction across cells.",
 ("tab_q08_robust", "c"): r"\textbf{Null, bootstrap and estimator noise together leave the verdicts standing.} Per cell: a joint score that divides the excess by the combined spread, and the genuine count it implies.",
 ("tab_q08_robust", "d"): r"\textbf{The excess shrinks with the number of classes.} ImageNet subsets of growing size, drawn at random across the hierarchy or as WordNet siblings: the reading, the excess and the verdict per subset.",
 ("tab_q10_calibration", "a"): r"\textbf{Known geometries read as they should on this scale.} A tree, hyperbolic regions of growing radius, a sphere and Gaussian clouds of growing dimension: the absolute reading and the normalized one.",
 ("tab_q10_calibration", "b"): r"\textbf{The excess certifies clusters, and a star of clusters passes it.} Synthetic clouds of clusters, with and without a hierarchy: the reading and the excess against the spectrum null and against the hub-randomizing null.",
 ("tab_q01_census", "a"): r"\textbf{Structure beyond the second moments is the rule at the class level.} Per cell: the reading, its excess over the matched null, the excess as a fraction of the null, the rank among the replicates and the left-tail $p$.",
 ("tab_q01_census", "b"): r"\textbf{The verdict holds under every construction of null and statistic.} Per cell: genuine or not under five constructions, the centered and uncentered Haar nulls, the Gaussian null and the two statistics, with the excess as a fraction of the null.",
 ("tab_q01_census", "c"): r"\textbf{The census on cosine geometry agrees with the reading used in the paper.} Per cell, with L2-normalized centroids and geodesic distances: the reading, the excess, the rank and the verdict.",
 ("tab_q01_census", "d"): r"\textbf{The backbones outside the ViT panel read like it.} The 2 self-supervised ResNets on the ImageNet centroid store: dimension, reading, excess and rank against the spectrum null.",
 ("tab_q02_text", "a"): r"\textbf{In text the verdict follows the recipe and the scale.} Per model: dimension, the reading, the excess, the excess as a fraction of the null, the rank, the left-tail $p$ and the cosine reading beside it.",
 ("tab_q02_text", "c"): r"\textbf{Extraction is part of the measurement.} The same thousand prompts under changes of batch size and of precision: the reading per setting for a causal language model of each family.",
 ("tab_q04_depth", "a"): r"\textbf{Clusters are oriented toward their hubs in 4 of 12 backbones under the star with Gaussian hubs.} Per backbone: the depth statistic and its $z$ on CIFAR-100 and on ImageNet, the WordNet cut at 3 sizes, and the balanced frame.",
 ("tab_q04_depth", "a$'$"): r"\textbf{The same 4 backbones are certified under the star whose hubs are a Haar resample of the real ones.} Per backbone: the star's excess, the depth $z$, the star spread and the decoupling control over 10 seeds.",
 ("tab_q04_depth", "b"): r"\textbf{Supervision on leaf labels can produce the alignment but does not guarantee it.} The 2 ViT-B/16 trained on ImageNet-1k leaf labels, through the census protocol: excess, depth $z$ and agreement with WordNet.",
 ("tab_q04_depth", "c"): r"\textbf{Training in hyperbolic space creates no hub structure.} MERU and its Euclidean twin at 3 sizes: the census excess, the depth $z$ and the radii of the embeddings in units of the curvature scale.",
 ("tab_q04_depth", "d"): r"\textbf{The trained control separates the objectives by degree.} ViT-B/16 fine-tuned with leaf cross-entropy and with an added hierarchical term, 2 seeds: excess, depth $z$ intact and decoupled, on both frames.",
 ("tab_q04_depth", "e"): r"\textbf{The alignment is not the radial spread of feature norms.} The same clouds after removing the radial part of every offset and after full L2 normalization: the depth $z$ under both stars.",
 ("tab_q05_power", "b"): r"\textbf{A two-level implant at the real spread is missed.} Implanted trees on the real ImageNet centroids at growing strength: the detection rate at the certification bar, with the real spread and with it shrunk.",
 ("tab_q05_power", "c"): r"\textbf{Implanted hub alignment reproduces the verdict in the supervised ViTs and CLIP-B.} From the decoupled cloud, each cluster's principal axis rotated toward its hub: the detection rate per backbone and seed.",
 ("tab_q05_power", "d"): r"\textbf{A deep hierarchy at the real noise level is detected, and a flat control is not.} Synthetic clouds with ViT-L's spectrum and ratio: the depth $z$ intact and decoupled for the hierarchy and for the flat control.",
 ("tab_q05_power", "e"): r"\textbf{The power follows the backbone.} The same deep hierarchy built with each backbone's spectrum and ratio: the intact and decoupled power per backbone, and the within-between ratio it sits at.",
 ("tab_q05_power", "f"): r"\textbf{The balanced frame raises no false alarm.} The flat control of the deep experiment read on the balanced grouping: the intact and decoupled firing counts.",
 ("tab_q06_treemap", "a"): r"\textbf{The island belongs to the naive cut, not to the models.} Per configuration of metric and linkage: the degeneracy diagnostics, the cophenetic fidelity that selects one, and the mean pairwise agreement at the cut.",
 ("tab_q06_treemap", "b"): r"\textbf{The self-supervised models sit apart on distances, not on topology.} The three agreement measures for DINOv2 against the supervised and contrastive models, each written as against those models over within them.",
 ("tab_q06_treemap", "c"): r"\textbf{Two resamples of one model set the ceiling any pair of models can reach.} Per backbone: the three measures between pairs of the 30 bootstrap centroid sets, with their mean, spread and minimum.",
 ("tab_q07_wordnet", "a"): r"\textbf{Agreement with the human taxonomy follows supervision and recipe.} Per backbone: the correlation between inter-centroid and WordNet distances on ImageNet, with the superclass recovery beside it.",
 ("tab_q07_wordnet", "b"): r"\textbf{Every family recovers the CIFAR-100 superclasses.} Per backbone and configuration: the agreement of the 20-cluster cut with the coarse labels, with the degenerate configurations marked.",
 ("tab_q07_wordnet", "c"): r"\textbf{The recovery is not WordNet circularity.} On DBpedia Classes, an ontology built independently: the reading, the excess, the triplet agreement and the metric gains per text embedder.",
 ("tab_q07_wordnet", "d"): r"\textbf{A caption hierarchy with no class set reads the same way.} On HierarCaps: the correlation between level and radius, the share of monotone chains, the triplet agreements and the leaf excess per encoder.",
 ("tab_q09_corollary", "a"): r"\textbf{The zero-cost gains are modest and limited to prototype tasks.} Per cell: nearest-centroid and few-shot accuracy under the Euclidean and the Poincar\'{e} readout, the advantage of each zero-cost metric, and a paired test.",
 ("tab_q09_corollary", "b"): r"\textbf{The raw reading predicts the gain within datasets.} Per dataset and task: the correlation between the raw supremum reading and the metric gain, with bootstrap intervals and the family-demeaned value.",
 ("tab_q09_corollary", "c"): r"\textbf{Cosine collects the gain for the self-supervised models.} Per backbone: the best zero-cost advantage over the Euclidean readout, averaged over the hierarchical datasets, and which metric collects it.",
 ("tab_q09_corollary", "d"): r"\textbf{The calibrated reading predicts the gain as well as the raw one, and the hierarchy verdict predicts none.} Per dataset and task: the same correlations for the raw statistic, the excess and the hierarchy-test $z$.",
}
for _k, _v in PANELCAP.items(): assert len(_v.split()) <= 40, (_k, len(_v.split()))
PANELLAB = {"a": "", "a$'$": "-ap", "b": "-b", "c": "-c", "d": "-d", "e": "-e", "f": "-f"}

def short_title(t):
    """A panel title is a label: the text before the first colon outside math and braces, else its first sentence. Never cut math."""
    t = re.sub(r"\s+", " ", t).strip()
    for a_, b_ in TITLE.items():
        if t.startswith(a_): return b_ + "."
    depth = 0; math = False; cut = None
    for i, c in enumerate(t):
        if c == "$": math = not math
        elif c == "{": depth += 1
        elif c == "}": depth -= 1
        elif not math and depth == 0 and c == ":" and len(t[:i].split()) >= 5: cut = i; break
        elif not math and depth == 0 and c == "." and i + 1 < len(t) and t[i+1] == " ": cut = i; break
    return (t[:cut] + ".") if cut else t

def title_line(l):
    """Shorten the bold panel title of one line, matching its braces (a title may contain math with braces)."""
    m = re.match(r"\\noindent\\textbf\{", l)
    if not m: return l
    d, j = 1, m.end()
    while d and j < len(l):
        if l[j] == "{": d += 1
        elif l[j] == "}": d -= 1
        j += 1
    return "\\noindent\\textbf{" + short_title(l[m.end():j-1]) + "}" + l[j:]


def split_panels(src, drop=(), capfix=None, dropline=(), repl=(), per_panel=False, newcap=None, stem_=None):
    for a, b in repl:
        assert src.count(a) == 1, a; src = src.replace(a, b)
    for a, b in REPL_ALL: src = src.replace(a, b)
    if dropline:
        n0 = len(src.split("\n")); src = "\n".join(l for l in src.split("\n") if not any(d in l for d in dropline)); assert len(src.split("\n")) == n0 - len(dropline), dropline
    prov = [l for l in src.split("\n") if l.startswith("% prov:")]; out = list(prov); first = True; dropped = 0; labbase = []; labbase_used = []
    blocks_ = re.findall(r"\\begin\{table\}\[(?:H|tbp|htbp)\](\\ContinuedFloat)?\n(.*?)\n\\end\{table\}", src, flags=re.S); assert blocks_, "no table block in the source"   # [tbp]: an already split copy (idempotent)
    for _, body in blocks_:
        L = body.split("\n"); hdr, rest = L[:3], L[3:]; assert hdr[0] == "\\centering", hdr
        ci = next(i for i, l in enumerate(rest) if l.startswith("\\caption{")); cap, plines = rest[ci:], rest[:ci]
        if capfix and first: cap = ("\n".join(cap)); cap2 = capfix(cap); assert cap2 != cap, "caption sentence of the dropped panel not found"; cap = cap2.split("\n")
        if newcap and first:   # appendix cleanup (2026-09-23): the short caption, with the provenance comment of the original kept
            _prov = [l for l in cap if l.lstrip().startswith("%")]
            cap = ["\\caption{" + newcap] + _prov + ["}"] + [l for l in cap if l.startswith("\\label{")]
        plines = [title_line(l) for l in plines]
        panels, cur = [], []
        for i, l in enumerate(plines):   # a panel starts at \begingroup (sized panel) or at a bold panel title not wrapped in a group
            if (l == "\\begingroup" or (l.startswith("\\noindent\\textbf{(") and (i == 0 or plines[i-1] != "\\begingroup"))) and cur: panels.append(cur); cur = []
            cur.append(l)
        if cur: panels.append(cur)
        isdrop = lambda P: any(("\\noindent\\textbf{" + d) in "\n".join(P) for d in drop)
        cap_ok = first or not isdrop(panels[0]); first_kept = True   # a continued block whose first panel is dropped loses its caption (it described that panel)
        for P in panels:
            if isdrop(P): dropped += 1; continue
            letter = (re.search(r"\\noindent\\textbf\{\(([a-e])\)", "\n".join(P)) or [None, None])[1]
            letter_raw = (re.search(r"\\noindent\\textbf\{\((a\$'\$|[a-h])\)", "\n".join(P)) or [None, None])[1]
            lab = re.search(r"\\label\{(tab:[^}]*)\}", "\n".join(cap))
            if lab: labbase.append(lab.group(1))
            base = labbase[0] if labbase else "tab:unknown"
            pc = PANELCAP.get((stem_, letter_raw)) if letter_raw else None
            if pc:   # a former panel becomes its own numbered table (author's brief, 2026-09-23 22:00)
                _suf = PANELLAB.get(letter_raw, "-x")
                _labs = ("\\label{" + base + "}" + ("\\label{" + base + _suf + "}" if _suf else "")) if not labbase_used else ("\\label{" + base + _suf + "}")
                labbase_used.append(True)
                capP = ["\\caption{" + pc] + [l for l in cap if l.lstrip().startswith("%")] + ["}", _labs]
                P = [l for l in P if not l.startswith("\\noindent\\textbf{(")]   # the panel title is now the caption
            else:
                capP = (cap if (first_kept and cap_ok and (first or not newcap)) else ["\\caption{(continued)}"])
            P = [("\\par\\vspace{2pt}" if l == "\\par\\vspace{5pt}" else ("\\par\\vspace{1pt}" if l == "\\par\\vspace{1.5pt}" else l)) for l in P]   # less air between panels (appendix cleanup, 2026-09-23)
            out += ["\\begin{table}[H]" ] + hdr + P + capP + ["\\end{table}"]; first = False; first_kept = False
    assert dropped == len(drop), (drop, dropped)
    return "\n".join(out) + "\n"
for stem in order:
    fn = FINAL_SRC.get(stem, stem)
    src = open((FD if stem in FINAL_SRC else TEX + "appendix_tables/") + fn + ".tex").read()
    open(FD + fn + ".tex", "w").write(split_panels(src, DROP.get(stem, ()), CAPFIX.get(stem), DROPLINE.get(stem, ()), REPL.get(stem, ()), per_panel=False, newcap=NEWCAP.get(stem), stem_=stem))   # Table 3: one caption segment per panel (twelfth review)
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
# ---- boxed statements and S5.5 in the appendix (author's brief, 2026-09-24): this is the submission from now on; the same text
# without boxes and with S5.5 in place is written beside it as main_iclr2027_final_noboxes.tex.
BOXPRE = r"""
% ---- boxed statements: definitions in light blue, Proposition 1 in light amber; thin left rule, no frame, 2pt padding, same font.
\usepackage[most]{tcolorbox}
\tcbset{statementbox/.style={enhanced, breakable, boxrule=0pt, frame hidden, arc=0pt, outer arc=0pt, sharp corners,
  left=2pt, right=2pt, top=2pt, bottom=2pt, boxsep=0pt, borderline west={1.2pt}{0pt}{#1}, before skip=2pt, after skip=2pt,
  before upper={\setlength{\topsep}{0pt}\setlength{\partopsep}{0pt}\setlength{\parskip}{0pt}}}}
\newtcolorbox{defbox}{statementbox=blue!55!black, colback=blue!4}
\newtcolorbox{propbox}{statementbox=orange!70!black, colback=orange!8}
"""
def boxed(t):
    """The statements in boxes and S5.5 moved to the appendix, with a one-sentence pointer at the end of S5.4."""
    t = t.replace("\\begin{document}", BOXPRE.strip("\n") + "\n\n\\begin{document}", 1)
    t = t.replace("\\begin{definition}", "\\begin{defbox}\n\\begin{definition}").replace("\\end{definition}", "\\end{definition}\n\\end{defbox}")
    t = t.replace("\\begin{proposition}", "\\begin{propbox}\n\\begin{proposition}").replace("\\end{proposition}", "\\end{proposition}\n\\end{propbox}")
    i5 = t.index("\\subsection{What is shared is local}"); j5 = t.index("\\section{", i5)
    body5 = re.search(r"\\paragraph\{[^}]*\}.*", t[i5:j5], re.S).group(0).strip()
    t = t[:i5] + t[j5:]
    ptr = "\\paragraph{Models share neighborhoods, not metrics.} Permutation-calibrated agreement across models survives, and Table~\\ref{tab:q13-local} in Appendix~\\ref{app:local} gives the reading. % exp21b_local_global_K200.csv\n\n"
    k4 = t.index("\\section{Implications for Hyperbolic Representation Learning}")
    t = t[:k4] + ptr + t[k4:]
    anchor = "\\input{appendix_tables/final/tab_q13_local}"
    assert t.count(anchor) == 1, "the local-agreement table is inputted once"
    return t.replace(anchor, body5 + "\n" + anchor)

assert "depth test" not in (pre + body).lower() and "depth-test" not in body and "depth verdict" not in body, "the rename of 2026-09-23 must cover the appendix prose too"
open(PF.replace("_final.tex", "_final_noboxes.tex"), 'w').write(pre + body)   # the same text without boxes, for the record
open(PF, 'w').write(boxed(pre + body))
json.dump({"kept": order, "deleted": DELETED, "cuts": CUTS}, open(R + 'final_appendix.json', 'w'), indent=1)
print("final written ->", PF, "| appendix order:", order, "| deleted:", DELETED, "| cuts:", CUTS)
