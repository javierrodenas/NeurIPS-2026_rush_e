#!/usr/bin/env python3
"""Final pass, Phase B: fills phaseE_paper.tex.tmpl (abstract to \\end{document}: main text, statements, appendix with one table
per question) with every number that stays in the prose, read from the result files, and writes main_iclr2027.tex = the
current preamble + the filled template. Run from the repo root AFTER ICLR2027/iclr2027/gen_appendix_final.py and gen_main_table.py,
and BEFORE gen_provenance.py (which indexes the tables the written tex inputs). The previous main body stays in
rebuttal/results/phaseD_old_main_body.tex for the number-preservation check of sweep_freeze.py."""
import json, re, os, sys, importlib
import numpy as np, pandas as pd
R = 'rebuttal/results/'; S = 'rebuttal/scripts/'; P = 'ICLR2027/iclr2027/main_iclr2027.tex'
import numpy.core as _c; sys.modules.setdefault("numpy._core", _c)
for s_ in ("multiarray", "numeric", "_multiarray_umath"):
    try: sys.modules.setdefault("numpy._core." + s_, importlib.import_module("numpy.core." + s_))
    except Exception: pass
W = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve", 20: "twenty", 24: "twenty-four", 30: "thirty", 60: "sixty", 72: "seventy-two"}
def w(n): return W.get(int(n), str(int(n)))
F = {}
# --- sample level (S4.1): the normalized excess of the genuine cells
sl = pd.read_csv(R + 'expR62_samplelevel_record.csv'); fr = (sl.excess / sl.null_mean)[sl.genuine_bh]
F['SL_FRAC_LO'], F['SL_FRAC_HI'] = f"{100*fr.abs().min():.0f}", f"{100*fr.abs().max():.0f}"
top = sl.loc[(sl.excess / sl.null_mean).idxmin()]; assert (top.model, top.dataset) == ("dinov2_g", "cifar100"), "the largest sample-level fraction is no longer DINOv2-G on CIFAR-100: fix S4.1"
F['SL_WITHIN'] = str(int((~sl.genuine_bh).sum()))
# --- depth test
dv = pd.read_csv(R + 'expR56_depth_variants.csv'); an = dv[(dv.dataset == 'imagenet') & (dv.K == 30) & (dv.variant == 'aniso')]
F['N_IN'] = str(int((an.z_depth <= -2).sum())); assert F['N_IN'] == "4"
F['N_C100'] = str(int((dv[(dv.dataset == 'cifar100') & (dv.K == 20) & (dv.variant == 'aniso')].z_depth <= -2).sum()))
d = pd.read_csv(R + 'expR55b_depth_power_leafframe.csv'); st_ = d[d.level == 'star']; F['FA100'] = f"{100*(st_[(st_.n == 100) & (st_.K >= 12)].z <= -2).mean():.0f}"
H = pd.read_csv(R + 'expR69_depth_haarhubs_summary.csv'); ch = H[H.cert_haar]; cg = H[H.cert_gauss]
assert sorted(ch.model) == sorted(cg.model) == sorted(["i21k_s", "i21k_b", "i21k_l", "dinov2_l"]), "the certified set differs between stars: fix S4.4"
F['HAAR_Z_HI'], F['HAAR_Z_LO'] = f"{ch.z_haar.max():.1f}", f"{ch.z_haar.min():.1f}"
F['FA_HAAR'] = f"{int(H.fa_s0_haar.sum())} of {int(H.n_s0.sum())}"
hits, n1 = int(H.hits_s1_haar.sum()), int(H.n_s1.sum()); F['POWER_HAAR'] = (f"{w(hits)} implant in {w(n1)}" if hits == 1 else f"{w(hits)} implants in {w(n1)}")
S9 = pd.read_csv(R + 'expR64b_wn30_summary.csv'); nd = int((S9.hits_s1 >= 4).sum()); F['R9B_NDET_PHRASE'] = ('none of the twelve backbones' if nd == 0 else f'{nd} of 12 backbones')
assert S9.ratio_real.min() >= 1.0 and S9.ratio_real.max() <= 4.0, "S4.4 says 'between one and four times': fix"
L70 = pd.read_csv(R + 'expR70_inet1k_supervised.csv').set_index('model')
assert L70.loc['vit_b_in1k', 'z_gauss'] <= -2 and L70.loc['vit_b_in1k', 'z_haar'] <= -2 and L70.loc['deit_b', 'z_gauss'] > -2 and L70.loc['deit_b', 'z_haar'] > -2, "leaf-label verdicts changed: fix S4.4/S5.4"
assert L70.loc['vit_b_in1k', 'spearman_wn'] > 0.4 and L70.loc['deit_b', 'spearman_wn'] < 0.15, "leaf-label alignment contrast changed: fix S5.4"
# --- MERU near-flat regime
Rm = pd.read_csv(R + 'expR71_meru_radii.csv'); F['MERU_RATIO'] = f"{Rm.lorentz_over_euclid_median.min():.3f}"
# --- tree map
z23 = np.load(R + 'exp23_treemap_controls.npz', allow_pickle=True)["summary_in"].item()["('cosine', 'average')"]
F['ARI_BIG'], F['ARI_BLOCK'] = f"{z23['big_vs_sup']:.2f}", f"{z23['sup_vs_sup']:.2f}"
zin = np.load(R + 'exp23_treemap_controls.npz', allow_pickle=True)["summary_in"].item(); zc1 = np.load(R + 'exp23_treemap_controls.npz', allow_pickle=True)["summary_c1"].item()
F['GAP_CC_IN'] = f"{zin[chr(40)+chr(39)+'cosine'+chr(39)+', '+chr(39)+'complete'+chr(39)+chr(41)]['big_vs_sup']:.2f}"; F['GAP_CW_C100'] = f"{zc1[chr(40)+chr(39)+'cosine'+chr(39)+', '+chr(39)+'ward'+chr(39)+chr(41)]['big_vs_sup']:.2f}"
r28 = pd.read_csv(R + 'exp28_recovery_per_config.csv').set_index('model'); ADM = ['euclid-ward', 'cosine-complete', 'cosine-ward']; D2 = ['dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g']
wins = int(sum(r28.loc['i21k_b', c] > r28.loc[m, c] for c in ADM for m in D2)); F['PAIRED'] = str(wins)
F['PAIRED_PHRASE'] = "all but one of twelve comparisons" if wins == 11 else ("every one of twelve comparisons" if wins == 12 else f"{wins} of 12 comparisons")
c52 = pd.read_csv(R + 'expR52_census_haar_p999_200.csv'); c57 = pd.read_csv(R + 'expR57_census_cosine_haar_p999_200.csv')
mg = c52.merge(c57, on=['model', 'dataset'], suffixes=('_e', '_c')); F['AGREE'] = str(int((mg.genuine_bh_e == mg.genuine_bh_c).sum()))
e1 = pd.read_csv(R + 'exp1_delta_controls.csv'); g = e1.pivot_table(index='model', columns='variant', values='delta_max')
grp = int((g['grp_wordnet'] < g['grp_random']).sum()); F['GRP'] = str(grp); F['GRP_PHRASE'] = "all but one of twelve models" if grp == 11 else ("every one of twelve models" if grp == 12 else f"{grp} of 12 models")
# --- curvature band and the normalized ImageNet excess (S6)
ga = e1[e1.variant == 'gauss'].drop_duplicates('d').sort_values('d'); d_lo, d_hi = int(ga.d.iloc[0]), int(ga.d.iloc[-1])
F['C_LO'], F['C_HI'] = f"{(0.144/(2*float(ga.delta_max.iloc[0])))**2:.2f}", f"{(0.144/(2*float(ga.delta_max.iloc[-1])))**2:.1f}"; F['D_LO'], F['D_HI'] = str(d_lo), str(d_hi)
im = c52[c52.dataset == 'imagenet']; fim = (im.excess / im.null_mean).abs(); F['IN_FRAC_LO'], F['IN_FRAC_HI'] = f"{100*fim.min():.0f}", f"{100*fim.max():.0f}"
assert (im.excess < 0).all(), "an ImageNet cell is sign-positive: 'removes x to y per cent' no longer holds"
b = pd.read_csv(R + 'exp2b_normalized_stack.csv'); Ht = b[b.dataset.isin(['cifar100', 'cifar10', 'dtd'])]; gg = Ht.FS_HN_COS_diff * 100; contr = Ht.paradigm.str.lower().str.startswith('contr')
F['GAIN_LO'], F['GAIN_HI'] = f"{gg[contr].min():+.1f}", f"{gg[contr].max():+.1f}"
Hh = b[b.dataset.isin(['imagenet', 'cifar100', 'cifar10', 'dtd'])]; go = (Hh.FS_HN_COS_diff * 100)[~Hh.paradigm.str.lower().str.startswith('contr')]
assert go.min() < 0 < go.max(), "the other families' Poincare-over-cosine effect is no longer inconsistent in sign: fix S6"
# --- appendix prose
c60 = pd.read_csv(R + 'expR60_c_sweep_record.csv'); c10 = c60[(c60.C == 10) & (c60['mode'] == 'random')].groupby('model').excess.mean()
F['C10_LO'], F['C10_HI'] = f"${c10.max():+.2f}$", f"${c10.min():+.2f}$"; F['DV2G_C1000'] = f"${c60[(c60.model == 'dinov2_g') & (c60.C == 1000) & (c60['mode'] == 'random')].excess.mean():+.3f}$"
F65 = pd.read_csv(R + 'expR65_hier_finetune.csv').set_index('obj')
for o, k in (("ce", "CE"), ("hier", "HIER"), ("frozen", "FROZEN")): F[f'FT_{k}_EXC'] = f"{F65.loc[o, 'excess']:+.3f}"; F[f'FT_{k}_Z'] = f"{F65.loc[o, 'z_depth']:+.2f}"
ABL = os.environ.get("PLATONIC_ABLATIONS", "/media/HDD_4TB_2/javi/Platonic/results")
a4 = pd.read_csv(os.path.join(ABL, 'analysis4_finetuning.csv')); F['FT_PCT_LO'], F['FT_PCT_HI'] = f"{a4['pct_change'].min():.0f}", f"{a4['pct_change'].max():.0f}"
hc = pd.read_csv(R + 'exp16_hierarcaps.csv'); tr = pd.concat([hc.trip_L1_cos, hc.trip_L2_cos]); F['HC_TRIP_LO'], F['HC_TRIP_HI'] = f"{tr.min():.2f}", f"{tr.max():.2f}"
x47 = pd.read_csv(R + 'expR47_extraction_variance.csv'); gm = x47[x47.model == 'gpt2_m']; base = gm[gm.batch == 1].delta.mean(); F['EXTR_MAX'] = f"{(gm.delta - base).abs().max():.3f}"
c68 = pd.read_csv(R + 'expR68_corollary_excess.csv').set_index(['predictor', 'gain', 'dataset'])
F['NC_EXC_IN'] = f"${c68.loc[('excess', 'NC_adv', 'imagenet'), 'r']:+.2f}$"; F['NC_RAW_IN'] = f"${c68.loc[('raw', 'NC_adv', 'imagenet'), 'r']:+.2f}$"
assert all(c68.loc[('excess', 'NC_adv', ds), 'ci_hi'] < 0 for ds in ('imagenet', 'cifar100', 'cifar10', 'dtd')), "NC-on-excess no longer survives on all four datasets: fix S6/A.12"
assert [ds for ds in ('imagenet', 'cifar100', 'cifar10', 'dtd') if c68.loc[('excess', 'FS_adv', ds), 'ci_hi'] < 0] == ['cifar100', 'cifar10', 'dtd'], "FS-on-excess survival set changed: fix S6/A.12"
assert not any((c68.loc[('depth_z', g_, ds), 'ci_hi'] < 0) or (c68.loc[('depth_z', g_, ds), 'ci_lo'] > 0) for g_ in ('NC_adv', 'FS_adv', 'FS_best_adv') for ds in ('imagenet', 'cifar100')), "the depth z predicts a gain: fix S6"
print("fills:", F)
body = open(S + 'phaseE_paper.tex.tmpl').read()
for k, v in F.items(): body = body.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{[A-Z0-9_]+\}\}", body); assert not left, left
T = open(P).read(); i = T.index("\\begin{abstract}")
open(P, 'w').write(T[:i] + body); print("phaseE applied ->", P)
json.dump(F, open(R + 'phaseE_fills.json', 'w'), indent=1)
