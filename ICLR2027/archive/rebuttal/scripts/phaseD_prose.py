#!/usr/bin/env python3
"""Prose pass (style of Groger et al.): fills phaseD_main_body.tex.tmpl with the numbers that stay in the prose (read from the
result files), splices it into main_iclr2027.tex between \\begin{abstract} and the Ethics Statement, applies the fixed-vocabulary
renames in the appendix, adds the appendix home of the one number that had none (Khrulkov's c ~ 0.33), and saves the previous
main body to rebuttal/results/phaseD_old_main_body.tex for the number-preservation check. Run from the repo root."""
import json, re, os as _os
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
# positive-control pass fills
import os as _os
if _os.path.exists(R + 'expR64b_wn30_summary.csv'):
    S9 = pd.read_csv(R + 'expR64b_wn30_summary.csv'); nd = int((S9.hits_s1 >= 4).sum()); F['R9B_NDET'] = str(nd); F['R9B_NDET_PHRASE'] = ('none of the twelve backbones' if nd == 0 else f'{nd} of 12 backbones')
    F['RATIO_LO'], F['RATIO_HI'] = f"{S9.ratio_real.min():.1f}", f"{S9.ratio_real.max():.1f}"
    D9 = pd.read_csv(R + 'expR64b_wn30.csv'); cert4 = ['i21k_s', 'i21k_b', 'i21k_l', 'dinov2_l']
    z0 = D9[(D9.kind == 'depth') & (D9.partition == 'rand6') & (D9.s != 'real') & (D9.model.isin(cert4))].copy(); z0['s'] = z0.s.astype(float); z0 = z0[z0.s == 0]
    F['CERT_S0'] = f"{int((z0.z <= -2).sum())} of {len(z0)}"
J = pd.read_csv(R + 'expR66_joint_sensitivity_summary.csv'); topJ = J.dataset.isin(['imagenet', 'cifar100'])
cnt = sorted([int(J.joint_genuine.sum()), int(J.boot_bh_genuine.sum())]); cntt = sorted([int((J.joint_genuine & topJ).sum()), int((J.boot_bh_genuine & topJ).sum())])
F['R11_LO'], F['R11_HI'], F['R11T_LO'], F['R11T_HI'] = str(cnt[0]), str(cnt[1]), str(cntt[0]), str(cntt[1])
u = (im.excess / im.null_sd).abs(); F['NULLU_LO'], F['NULLU_HI'] = f"{u.min():.0f}", f"{u.max():.0f}"
print("fills:", F)
body = open(S + 'phaseD_main_body.tex.tmpl').read()
for k, v in F.items(): body = body.replace("{{" + k + "}}", v)
DRY = _os.environ.get("DRY_OUT")
if DRY:   # layout rehearsal only: unknown numbers become 'X', the result goes to DRY_OUT, the repo tex is untouched
    body = re.sub(r"\{\{[A-Z0-9_]+\}\}", "3", body)
assert not re.findall(r"\{\{[A-Z0-9_]+\}\}", body)
p = 'ICLR2027/iclr2027/main_iclr2027.tex'; T = open(p).read()
i = T.index("\\begin{abstract}"); j = T.index("\\subsubsection*{Ethics Statement}")
import os
if not os.path.exists(R + 'phaseD_old_main_body.tex'): open(R + 'phaseD_old_main_body.tex', 'w').write(T[i:j])   # keep the pre-prose body for the number-preservation check
T = T[:i] + body + T[j:]
def rep(old, new, count=1):
    global T
    if new in T: return   # already applied (also covers insertions where old is a substring of new)
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
# positive-control pass: appendix
rep("\\caption{The nulls of the instrument and what each one preserves and destroys; ``genuine'' in the text always means beyond the first of these.}",
    "\\caption{The nulls of the instrument and what each one preserves and destroys; ``genuine'' in the text always means beyond the first of these. In plain terms, the spectrum null is a cloud with the same shape as the real one and no structure inside it; the hub null of Section~\\ref{sec:form} keeps every cluster of the real cloud and only rearranges where the clusters sit.}")
if _os.path.exists(R + 'expR65_hier_finetune.csv'):
    F65 = pd.read_csv(R + 'expR65_hier_finetune.csv').set_index('obj')
    r10 = ("\\paragraph{A trained control, inconclusive.} We fine-tuned the census ViT-B/16 for two passes over the images behind its ImageNet centroids with two objectives on the same batches: cross-entropy alone, and cross-entropy plus a hierarchical cross-entropy over the WordNet superclasses. "
           f"Both fine-tuned models keep the census excess of the frozen backbone (${F65.loc['ce','excess']:+.3f}$ and ${F65.loc['hier','excess']:+.3f}$ against ${F65.loc['frozen','excess']:+.3f}$) and both are certified by the depth test, with $z$ of ${F65.loc['ce','z_depth']:+.2f}$ and ${F65.loc['hier','z_depth']:+.2f}$ against ${F65.loc['frozen','z_depth']:+.2f}$ for the frozen backbone: the hierarchical term lowers $z$ relative to plain cross-entropy but neither reaches the frozen reading, so the control neither confirms nor refutes the test (Table~\\ref{{tab:b39-finetune}}). "
           "The full training set could not be read at training speed from the archive, so the schedule is two passes over the census subset. % expR65_hier_finetune.csv\n")
    rep("\\input{appendix_tables/tab_b30_meru}\n", "\\input{appendix_tables/tab_b30_meru}\n" + r10 + "\\input{appendix_tables/tab_b39_finetune}\n")
if _os.path.exists(R + 'expR64b_wn30_summary.csv'):
    rep("\\input{appendix_tables/tab_b35_power}\n", "\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.5\\linewidth]{figures/fig_depth_power_app.pdf}\n\\caption{\\textbf{Power of the depth test on synthetic hierarchies at ImageNet's class count.} Plotted: power ($z\\le-2$) on synthetic two- and three-level hierarchies with the frame at the leaf clusters (solid, one line per $K$) or at the top level (dotted), against the within/between noise ratio, with the false-alarm rates of pure stars dashed. % expR55b_depth_power_leafframe.csv, expR55_depth_power.csv\n}\n\\label{fig:depthpower}\n\\end{figure}\n\\input{appendix_tables/tab_b35_power}\n\\input{appendix_tables/tab_b37_implanted}\n")
    rep("\\input{appendix_tables/tab_b36_bootstrap}\n", "\\input{appendix_tables/tab_b36_bootstrap}\n\\input{appendix_tables/tab_b38_joint}\n")
open(DRY or p, 'w').write(T); print("prose pass applied" + (f" (DRY -> {DRY})" if DRY else ""))
