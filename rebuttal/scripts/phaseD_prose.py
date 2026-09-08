#!/usr/bin/env python3
"""Prose pass (style of Groger et al.): fills phaseD_main_body.tex.tmpl with the numbers that stay in the prose (read from the
result files), splices it into main_iclr2027.tex between \\begin{abstract} and the Ethics Statement, applies the fixed-vocabulary
renames in the appendix, adds the appendix home of the one number that had none (Khrulkov's c ~ 0.33), and saves the previous
main body to rebuttal/results/phaseD_old_main_body.tex for the number-preservation check. Run from the repo root."""
import json, re
import pandas as pd
R = 'rebuttal/results/'; S = 'rebuttal/scripts/'
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
F = {}
sl = pd.read_csv(R + 'expR62_samplelevel_record.csv'); F['SL_WITHIN'] = str(int((~sl.genuine_bh).sum()))
d = pd.read_csv(R + 'expR55b_depth_power_leafframe.csv'); st = d[d.level == 'star']
F['FA100'] = f"{100*(st[(st.n == 100) & (st.K >= 12)].z <= -2).mean():.0f}"
dv = pd.read_csv(R + 'expR56_depth_variants.csv'); an = dv[(dv.dataset == 'imagenet') & (dv.K == 30) & (dv.variant == 'aniso')]
F['N_IN'] = str(int((an.z_depth <= -2).sum())); assert F['N_IN'] == "4"
import numpy as np, sys, importlib, numpy.core as _c
sys.modules.setdefault("numpy._core", _c)
for s_ in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core." + s_, importlib.import_module("numpy.core." + s_))
    except Exception: pass
z23 = np.load(R + 'exp23_treemap_controls.npz', allow_pickle=True)["summary_in"].item()["('cosine', 'average')"]
F['ARI_BIG'], F['ARI_BLOCK'] = f"{z23['big_vs_sup']:.2f}", f"{z23['sup_vs_sup']:.2f}"
r28 = pd.read_csv(R + 'exp28_recovery_per_config.csv').set_index('model'); ADM = ['euclid-ward', 'cosine-complete', 'cosine-ward']; D2 = ['dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g']
F['PAIRED'] = str(int(sum(r28.loc['i21k_b', c] > r28.loc[m, c] for c in ADM for m in D2)))
c52 = pd.read_csv(R + 'expR52_census_haar_p999_200.csv'); c57 = pd.read_csv(R + 'expR57_census_cosine_haar_p999_200.csv')
mg = c52.merge(c57, on=['model', 'dataset'], suffixes=('_e', '_c')); F['AGREE'] = str(int((mg.genuine_bh_e == mg.genuine_bh_c).sum()))
F['ARI_REC'] = f"{pd.read_csv(R + 'exp8_p1_recovery.csv').ari.max():.2f}"
e1 = pd.read_csv(R + 'exp1_delta_controls.csv'); g = e1.pivot_table(index='model', columns='variant', values='delta_max')
F['GRP'] = str(int((g['grp_wordnet'] < g['grp_random']).sum()))
p6 = pd.read_csv(R + 'exp8_p6_pooling.csv'); F['RHO1'] = f"{p6[p6.m_imgs == 1].rho_mean.max():+.2f}"
F['PAIRS'] = str(len(pd.read_csv(R + 'exp21b_local_global_K200.csv')))
ga = e1[e1.variant == 'gauss'].drop_duplicates('d').sort_values('d'); d_lo, d_hi = int(ga.d.iloc[0]), int(ga.d.iloc[-1])
F['C_LO'], F['C_HI'] = f"{(0.144/(2*float(ga.delta_max.iloc[0])))**2:.2f}", f"{(0.144/(2*float(ga.delta_max.iloc[-1])))**2:.1f}"; F['D_LO'], F['D_HI'] = str(d_lo), str(d_hi)
im = c52[c52.dataset == 'imagenet']; F['IN_EXC_HI'], F['IN_EXC_LO'] = f"{im.excess.max():+.3f}", f"{im.excess.min():+.3f}"
b = pd.read_csv(R + 'exp2b_normalized_stack.csv'); H = b[b.dataset.isin(['imagenet', 'cifar100', 'cifar10', 'dtd'])]; gg = H.FS_HN_COS_diff * 100; contr = H.paradigm.str.lower().str.startswith('contr')
F['GAIN_LO'], F['GAIN_HI'], F['GO_LO'], F['GO_HI'] = f"{gg[contr].min():+.1f}", f"{gg[contr].max():+.1f}", f"{gg[~contr].min():+.1f}", f"{gg[~contr].max():+.1f}"
print("fills:", F)
body = open(S + 'phaseD_main_body.tex.tmpl').read()
for k, v in F.items(): body = body.replace("{{" + k + "}}", v)
assert not re.findall(r"\{\{[A-Z0-9_]+\}\}", body)
p = 'ICLR2027/iclr2027/main_iclr2027.tex'; T = open(p).read()
i = T.index("\\begin{abstract}"); j = T.index("\\subsubsection*{Ethics Statement}")
import os
if not os.path.exists(R + 'phaseD_old_main_body.tex'): open(R + 'phaseD_old_main_body.tex', 'w').write(T[i:j])   # keep the pre-prose body for the number-preservation check
T = T[:i] + body + T[j:]
def rep(old, new, count=1):
    global T
    if T.count(old) == 0 and new in T: return   # already applied
    assert T.count(old) == count, f"{T.count(old)} matches: {old[:80]!r}"; T = T.replace(old, new)
# fixed-vocabulary renames in the appendix prose and captions
rep("\\paragraph{The form tracks training.}", "\\paragraph{Clustered structure tracks training.}")
rep("\\caption{\\textbf{The form is learned: training interventions move it.}", "\\caption{\\textbf{The raw reading is learned: training interventions move it.}")
rep("\\caption{\\textbf{In text, beyond-null structure depends on recipe and scale:", "\\caption{\\textbf{In text, clustered structure depends on recipe and scale:")
rep("and carry more beyond-null structure at small $C$", "and carry a larger excess at small $C$")
# the appendix home of Khrulkov's c ~ 0.33 (the one number the prose pass removed that had no table)
rep("\\paragraph{A diagnostic, not a policy.} Cosine is a safe default,",
    "\\paragraph{A diagnostic, not a policy.} \\citet{Khrulkov_2020_CVPR} estimate $c\\approx0.33$ from the raw reading of their datasets; the same rule on structureless Gaussian clouds gives the curvatures of Section~\\ref{sec:practice}. Cosine is a safe default,")
# the class-set sizes that left S3 (80-6,000 cached images per class on the transfer sets) live in the model-panel note
rep("Feature extraction: vision backbones use the encoder's default pooled embedding", "Class centroids average $100$ cached training images per class on ImageNet and the full cached split, $80$--$6{,}000$ images per class, on the other datasets. Feature extraction: vision backbones use the encoder's default pooled embedding")
open(p, 'w').write(T); print("prose pass applied")
