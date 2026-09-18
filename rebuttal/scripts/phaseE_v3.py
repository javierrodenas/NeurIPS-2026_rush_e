#!/usr/bin/env python3
"""Version v3 (classic structure, written from the claim lists): fills phaseE_paper_v3.tex.tmpl and writes
ICLR2027/iclr2027/main_iclr2027_v3.tex. Verbatim parts come from the author's main_local.tex (abstract, S1 with Figure 1, S2, the
'Gromov delta' and 'Estimation and normalization' paragraphs). Every number in the prose is a fill read from the result files
(rebuttal/results/phaseE_fills.json from phaseE_final.py plus the v3 fills computed here, saved to phaseE_v3_fills.json). The
appendix reuses the fourteen question sections of the v1 template, reordered to the order in which v3 first cites their tables.
Never touches main_iclr2027.tex. Run from the repo root after phaseE_final.py."""
import json, re, os, sys, importlib
import numpy as np, pandas as pd
R = 'rebuttal/results/'; S = 'rebuttal/scripts/'; TEX = 'ICLR2027/iclr2027/'; P1 = TEX + 'main_iclr2027.tex'; P3 = TEX + 'main_iclr2027_v3.tex'
import numpy.core as _c; sys.modules.setdefault("numpy._core", _c)
for s_ in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core." + s_, importlib.import_module("numpy.core." + s_))
    except Exception: pass
F = json.load(open(R + 'phaseE_fills.json'))
# ---- v3 fills (each traced to its file; the sweep recomputes the key ones)
sl = pd.read_csv(R + 'expR62_samplelevel_record.csv'); F['SL_WITHIN'] = str(int((~sl.genuine_bh).sum()))
c52 = pd.read_csv(R + 'expR52_census_haar_p999_200.csv'); F['NEG_COUNT'] = str(int((c52.excess < 0).sum()))
im = c52[c52.dataset == 'imagenet']; u = (im.excess / im.null_sd).abs(); F['NULLU_LO'], F['NULLU_HI'] = f"{u.min():.0f}", f"{u.max():.0f}"
s42 = pd.read_csv(R + 'expR42_star_calibration.csv'); st_ = s42[s42.config.str.startswith('star')].groupby('config').excessA_haar.mean(); F['STAR_DEEP'] = f"{st_.min():+.3f}"
D9 = pd.read_csv(R + 'expR64b_wn30.csv'); dep = D9[(D9.kind == 'depth') & (D9.partition == 'rand6') & (D9.s != 'real')].copy(); dep['s'] = dep.s.astype(float)
F['POWER_S1'] = f"{(dep[dep.s == 1.0].z <= -2).mean():.2f}"
L70 = pd.read_csv(R + 'expR70_inet1k_supervised.csv').set_index('model')
F['AUG_Z'] = f"{L70.loc['vit_b_in1k', 'z_gauss']:+.2f}"; F['AUG_RHO'] = f"{L70.loc['vit_b_in1k', 'spearman_wn']:+.2f}"; F['DEIT_RHO'] = f"{L70.loc['deit_b', 'spearman_wn']:+.2f}"; F['DEIT_ARI'] = f"{L70.loc['deit_b', 'ari_max']:.2f}"
assert L70.loc['vit_b_in1k', 'z_haar'] <= -2 and L70.loc['deit_b', 'z_gauss'] > -2 and L70.loc['deit_b', 'z_haar'] > -2
Rm = pd.read_csv(R + 'expR71_meru_radii.csv'); F['MERU_RAD_LO'], F['MERU_RAD_HI'] = f"{Rm.radius_sqrtc_median.min():.3f}", f"{Rm.radius_sqrtc_median.max():.3f}"
zin = np.load(R + 'exp23_treemap_controls.npz', allow_pickle=True)["summary_in"].item()
na = zin["('euclid', 'average')"]; F['NAIVE_BIG'], F['NAIVE_BLOCK'] = f"{na['big_vs_sup']:.2f}", f"{na['sup_vs_sup']:.2f}"
diag = json.load(open(R + 'exp23_config_diagnostics.json')); adm = [(zin[f"('{m}', '{l}')"]['big_vs_sup'], zin[f"('{m}', '{l}')"]['sup_vs_sup']) for m in ('euclid', 'cosine') for l in ('average', 'complete', 'ward') if diag[f"imagenet|{m}|{l}"]['maxfrac'] <= 0.5 and not (m == 'cosine' and l == 'average')]
assert len(adm) == 3, adm
F['ADM_LO'], F['ADM_HI'] = f"{min(a for a, _ in adm):.2f}", f"{max(a for a, _ in adm):.2f}"; F['ADM_BLK_LO'], F['ADM_BLK_HI'] = f"{min(b for _, b in adm):.2f}", f"{max(b for _, b in adm):.2f}"
cf = pd.read_csv(R + 'expR58_treemap_cutfree_summary.csv'); ca = cf[(cf.dataset == 'imagenet') & (cf.metric == 'cosine') & (cf.linkage == 'average')].iloc[0]
F['COPH_BIG'], F['COPH_BLOCK'] = f"{ca.coph_corr_big_vs_block:.2f}", f"{ca.coph_corr_within_block:.2f}"; F['TRIP_BIG'], F['TRIP_BLOCK'] = f"{ca.triplet_agree_big_vs_block:.2f}", f"{ca.triplet_agree_within_block:.2f}"
e10 = pd.read_csv(R + 'exp10_local_vs_global.csv').set_index('model')
F['TRIP_L_C'], F['TRIP_L_E'] = f"{e10.loc['dinov2_l', 'c100_sibtrip_c']:.2f}", f"{e10.loc['dinov2_l', 'c100_sibtrip_e']:.2f}"; F['TRIP_G_C'], F['TRIP_G_E'] = f"{e10.loc['dinov2_g', 'c100_sibtrip_c']:.2f}", f"{e10.loc['dinov2_g', 'c100_sibtrip_e']:.2f}"
e3 = pd.read_csv(R + 'exp3_alignment.csv').set_index('model')
rng = lambda ms: (f"{e3.loc[ms, 'spearman_wn'].min():+.2f}", f"{e3.loc[ms, 'spearman_wn'].max():+.2f}")
F['RHO_CON_LO'], F['RHO_CON_HI'] = rng(['clip_b', 'clip_l', 'siglip_b']); F['RHO_SUP_LO'], F['RHO_SUP_HI'] = rng(['i21k_t', 'i21k_s', 'i21k_b', 'i21k_l']); F['RHO_D2_LO'], F['RHO_D2_HI'] = rng(['dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g'])
d61 = pd.read_csv(R + 'expR61_dbpedia_record.csv'); assert d61.genuine_bh.all() and (d61.r_above == 200).all(); F['DBP_EXC_HI'], F['DBP_EXC_LO'] = f"{d61.excess.max():+.3f}", f"{d61.excess.min():+.3f}"
e14 = pd.read_csv(R + 'exp14_dbpedia.csv'); F['DBP_TRIP'] = f"{e14.trip_cos.mean():.2f}"; assert (e14.trip_cos.round(2) == e14.trip_cos.round(2).iloc[0]).all()
tx = pd.read_csv(R + 'expR53_text_haar_p999_200.csv').set_index('model'); F['TEXT_GEN'] = str(int(tx.genuine_bh.sum())); F['GPT2S_P'] = f"{tx.loc['gpt2', 'p_left']:.3f}".lstrip('0')
assert tx.loc['gpt2', 'genuine_bh'] and not tx.loc['gpt2_m', 'genuine_bh']
e21 = pd.read_csv(R + 'exp21b_local_global_K200.csv'); assert len(e21) == 66 and (e21.knn_R_p < 0.05).all() and (e21.cka_R_p < 0.05).all()
F['KNN_CAL'], F['CKA_CAL'], F['KNN_H'], F['CKA_H'] = f"{e21.knn_R_cal.mean():.3f}", f"{e21.cka_R_cal.mean():.3f}", f"{e21.knn_H_raw.mean():.3f}", f"{e21.cka_H_raw.mean():.3f}"
J = pd.read_csv(R + 'expR66_joint_sensitivity_summary.csv'); cnt = sorted([int(J.joint_genuine.sum()), int(J.boot_bh_genuine.sum())]); F['R11_LO'], F['R11_HI'] = str(cnt[0]), str(cnt[1])
cert4 = ['i21k_s', 'i21k_b', 'i21k_l', 'dinov2_l']; z0 = dep[(dep.s == 0.0) & dep.model.isin(cert4)]; F['CERT_S0'] = f"{int((z0.z <= -2).sum())} of {len(z0)}"
S9 = pd.read_csv(R + 'expR64b_wn30_summary.csv'); F['RATIO_LO'], F['RATIO_HI'] = f"{S9.ratio_real.min():.1f}", f"{S9.ratio_real.max():.1f}"
json.dump({k: F[k] for k in sorted(F)}, open(R + 'phaseE_v3_fills.json', 'w'), indent=1)
# ---- verbatim parts from the author's file
L = open(TEX + 'main_local.tex').read()
def between(a, b, s=L, strip=False): i = s.index(a); j = s.index(b, i); return s[i + (len(a) if strip else 0):j]
ABSTRACT = between("\\begin{abstract}", "\\end{abstract}") + "\\end{abstract}"
INTRO = between("\\section{Introduction}", "\\section{Related Work}").rstrip()
RELATED = between("\\section{Related Work}", "\\section{The Instrument}").rstrip()
GROMOV = between("\\paragraph{Gromov $\\delta$.} ", "\\paragraph{Estimation and normalization.}", strip=True).rstrip()
ESTIM = between("\\paragraph{Estimation and normalization.} ", "\\begin{figure}", strip=True).rstrip()
body = open(S + 'phaseE_paper_v3.tex.tmpl').read()
for k, v in [("%%ABSTRACT%%", ABSTRACT), ("%%INTRO%%", INTRO), ("%%RELATED%%", RELATED), ("%%GROMOV%%", GROMOV), ("%%ESTIMATION%%", ESTIM)]:
    assert body.count(k) == 1, k; body = body.replace(k, v)
for k, v in F.items(): body = body.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{[A-Z0-9_]+\}\}", body); assert not left, left
# ---- appendix: the question sections of the v1 template, in the order v3 first cites their tables
v1t = open(S + 'phaseE_paper.tex.tmpl').read(); app = v1t[v1t.index("\\section{Appendix: one table per question}"):v1t.index("\\end{document}")]
for k, v in F.items(): app = app.replace("{{" + k + "}}", v)
head = app[:app.index("\\FloatBarrier")]
blocks = re.split(r"(?=\\FloatBarrier\n\\subsection\{)", app[app.index("\\FloatBarrier"):]); blocks = [b for b in blocks if b.strip()]
bytab = {}
for b in blocks:
    m = re.search(r"\\input\{appendix_tables/(tab_[a-z0-9_]+)\}", b); bytab[m.group(1)] = b
labels = {}
for stem in bytab:
    if stem.startswith("tab_q"):
        mm = re.search(r"\\label\{(tab:q[^}]*)\}", open(TEX + "appendix_tables/" + stem + ".tex").read()); labels[mm.group(1)] = stem
main3 = body[:body.index("\\appendix")]; cited = []
for m in re.finditer(r"\\ref\{(tab:q[^}]*)\}", main3):
    if m.group(1) not in cited: cited.append(m.group(1))
order = [labels[l] for l in cited] + [s for s in bytab if s.startswith("tab_q") and s not in [labels[l] for l in cited]] + ["tab_z_provenance"]
assert len(order) == len(bytab) == 15, (len(order), len(bytab))
head = head.replace("\\section{Appendix: one table per question}", "\\section{One table per question}")
appendix = head + "".join(bytab[s] for s in order)
body = body.replace("%%APPENDIX_QUESTIONS%%", appendix) + "\n\\end{document}\n"
# ---- preamble of v1 plus amsthm
T1 = open(P1).read(); pre = T1[:T1.index("\\begin{abstract}")]
pre = pre.replace("\\usepackage{array}\n", "\\usepackage{array}\n\\usepackage{amsthm}\n", 1)
pre = pre.replace("\\begin{document}\n", "\\theoremstyle{definition}\n\\newtheorem{definition}{Definition}\n\\theoremstyle{plain}\n\\newtheorem{lemma}{Lemma}\n\\newtheorem{corollary}{Corollary}\n\\theoremstyle{remark}\n\\newtheorem*{remark}{Remark}\n"
                  "% v3: classic structure written from the claim lists (author's brief of 2026-09-18); generated by rebuttal/scripts/phaseE_v3.py. The submission file is main_iclr2027.tex.\n\\begin{document}\n", 1)
assert "\\newtheorem{lemma}" in pre
open(P3, 'w').write(pre + body); print("v3 written ->", P3, "| appendix order:", [labels.get(l, l) for l in cited])
