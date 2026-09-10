#!/usr/bin/env python3
"""Memo of the final pass (Phase A, A1-A6): numbers only. Writes ICLR2027/MEMO_final_pass.md and
rebuttal/results/final_pass_memo.json (the numbers sweep_freeze.py checks)."""
import json, os
import numpy as np, pandas as pd
R = 'rebuttal/results/'
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B","deit_b":"DeiT-B (IN-1k)","vit_b_in1k":"ViT-B (augreg IN-1k)"}
M = {}; L = ["# Memo — final pass (Phase A). Numbers only.", ""]
# ---- A1 ----
if os.path.exists(R + 'expR69_depth_haarhubs_summary.csv'):
    S = pd.read_csv(R + 'expR69_depth_haarhubs_summary.csv')
    M['a1'] = dict(cert_gauss=[m for m in S[S.cert_gauss].model], cert_haar=[m for m in S[S.cert_haar].model], cert_both=[m for m in S[S.cert_gauss & S.cert_haar].model],
                   fa_s0=int(S.fa_s0_haar.sum()), n_s0=int(S.n_s0.sum()), hits_s1=int(S.hits_s1_haar.sum()), n_s1=int(S.n_s1.sum()),
                   rows={r.model: dict(z_gauss=round(float(r.z_gauss), 2), r_gauss=int(r.r_star_gauss), z_haar=round(float(r.z_haar), 2), r_haar=int(r.r_star_haar), real_g=round(float(r.real_gauss), 4), star_g=round(float(r.star_gauss), 4), star_h=round(float(r.star_haar), 4), depth_g=round(float(r.depth_gauss), 4), depth_h=round(float(r.depth_haar), 4)) for r in S.itertuples()})
    L += ["## A1 — depth test with spectrum-matched (Haar-resampled) star hubs (`expR69_depth_haarhubs.csv`; WordNet-30, 10 star seeds, hub null 10×3; rank r_star = star seeds with excess B ≤ real, p = (1+r)/11, resolution 1/11)", "",
          "| backbone | real excess B | Gaussian star: star / depth / z / r_star | Haar-hub star: star / depth / z / r_star |", "|---|---|---|---|"]
    for r in S.itertuples(): L.append(f"| {NM[r.model]} | {r.real_gauss:+.4f} | {r.star_gauss:+.4f} / {r.depth_gauss:+.4f} / {r.z_gauss:+.2f} / {int(r.r_star_gauss)}/10 | {r.star_haar:+.4f} / {r.depth_haar:+.4f} / {r.z_haar:+.2f} / {int(r.r_star_haar)}/10 |")
    L += ["", f"- Certified (z ≤ −2) under the Gaussian star: {', '.join(NM[m] for m in M['a1']['cert_gauss'])} ({len(M['a1']['cert_gauss'])}/12); under the Haar-hub star: {', '.join(NM[m] for m in M['a1']['cert_haar']) or 'none'} ({len(M['a1']['cert_haar'])}/12); intersection: {', '.join(NM[m] for m in M['a1']['cert_both']) or 'none'}.",
          f"- Haar-hub star on the R9b implanted clouds: false alarms at s = 0: {M['a1']['fa_s0']} of {M['a1']['n_s0']}; power at s = 1: {M['a1']['hits_s1']}/{M['a1']['n_s1']} = {M['a1']['hits_s1']/max(1,M['a1']['n_s1']):.2f}.", ""]
# ---- A2 ----
if os.path.exists(R + 'expR70_inet1k_supervised.csv'):
    T = pd.read_csv(R + 'expR70_inet1k_supervised.csv').set_index('model'); e3 = pd.read_csv(R + 'exp3_alignment.csv').set_index('model')
    M['a2'] = {m: dict(excess_in=round(float(T.loc[m, 'excess_in']), 4), r_in=int(T.loc[m, 'r_in']), excess_c100=round(float(T.loc[m, 'excess_c100']), 4), r_c100=int(T.loc[m, 'r_c100']), z_gauss=round(float(T.loc[m, 'z_gauss']), 2), z_haar=round(float(T.loc[m, 'z_haar']), 2),
                       rho_wn=round(float(T.loc[m, 'spearman_wn']), 3), rho_shuf=round(float(T.loc[m, 'spearman_shuf']), 3), ari_max=round(float(T.loc[m, 'ari_max']), 2), ari_cos_complete=round(float(T.loc[m, 'ari_cosine-complete']), 2)) for m in T.index}
    L += ["## A2 — ViTs supervised on ImageNet-1k leaf labels only (`expR70_inet1k_supervised.csv`; census cache protocol)", "",
          "| model | IN excess (r/200) | C100 excess (r/200) | depth z Gaussian star / Haar-hub star | ρ_WN (shuffle) | C100 superclass ARI max / cosine-complete |", "|---|---|---|---|---|---|"]
    for m in T.index:
        v = M['a2'][m]; L.append(f"| {NM.get(m, m)} | {v['excess_in']:+.4f} ({v['r_in']}) | {v['excess_c100']:+.4f} ({v['r_c100']}) | {v['z_gauss']:+.2f} / {v['z_haar']:+.2f} | {v['rho_wn']:+.3f} ({v['rho_shuf']:+.3f}) | {v['ari_max']:.2f} / {v['ari_cos_complete']:.2f} |")
    L += ["", f"- Reference ρ_WN of the i21k ViTs in exp3: " + ", ".join(f"{NM[m]} {float(e3.loc[m, 'spearman_wn']):+.3f}" for m in ('i21k_t', 'i21k_s', 'i21k_b', 'i21k_l')) + ".", ""]
# ---- A3 ----
C = pd.read_csv(R + 'expR68_corollary_excess.csv'); old = pd.read_csv(R + 'night/correlation_cis.csv')
M['a3'] = {f"{r.predictor}|{r.gain}|{r.dataset}": dict(r=round(float(r.r), 3), lo=round(float(r.ci_lo), 3), hi=round(float(r.ci_hi), 3), dm=round(float(r.r_demeaned), 3)) for r in C.itertuples()}
surv = {ds: bool(C[(C.predictor == 'excess') & (C.gain == 'NC_adv') & (C.dataset == ds)].ci_hi.iloc[0] < 0) for ds in ('imagenet', 'cifar100', 'cifar10', 'dtd')}
M['a3_nc_excess_survives'] = surv; M['a3_fs_excess_survives'] = {ds: bool(C[(C.predictor == 'excess') & (C.gain == 'FS_adv') & (C.dataset == ds)].ci_hi.iloc[0] < 0) for ds in ('imagenet', 'cifar100', 'cifar10', 'dtd')}
M['a3_depth_any'] = bool(((C.predictor == 'depth_z') & (C.ci_hi < 0)).any())
L += ["## A3 — corollary correlations on the calibrated reading (`expR68_corollary_excess.csv`; Pearson r, Fisher-z 95% CI, n = 10 backbones; dm = family-demeaned)", "",
      "| gain | dataset | old (raw δ, Table B5) | raw δ̂₉₉.₉ | record excess | depth z |", "|---|---|---|---|---|---|"]
for g in ('NC_adv', 'FS_adv', 'FS_best_adv'):
    for ds in ('imagenet', 'cifar100', 'cifar10', 'dtd'):
        o = old[(old.task == g) & (old.dataset == ds)]; a = C[(C.predictor == 'raw') & (C.gain == g) & (C.dataset == ds)].iloc[0]; b = C[(C.predictor == 'excess') & (C.gain == g) & (C.dataset == ds)].iloc[0]; z = C[(C.predictor == 'depth_z') & (C.gain == g) & (C.dataset == ds)]
        L.append(f"| {g} | {ds} | {float(o.r.iloc[0]):+.2f} [{float(o.ci_lo.iloc[0]):+.2f}, {float(o.ci_hi.iloc[0]):+.2f}]" if len(o) else f"| {g} | {ds} | —")
        L[-1] += f" | {a.r:+.2f} [{a.ci_lo:+.2f}, {a.ci_hi:+.2f}] (dm {a.r_demeaned:+.2f}) | {b.r:+.2f} [{b.ci_lo:+.2f}, {b.ci_hi:+.2f}] (dm {b.r_demeaned:+.2f}) | " + (f"{z.iloc[0].r:+.2f} [{z.iloc[0].ci_lo:+.2f}, {z.iloc[0].ci_hi:+.2f}]" if len(z) else "n/a") + " |"
L += ["", f"- NC H−R on the record excess, CI excluding 0: " + ", ".join(f"{ds} {'yes' if v else 'no'}" for ds, v in surv.items()) + f"; FS H−R: " + ", ".join(f"{ds} {'yes' if v else 'no'}" for ds, v in M['a3_fs_excess_survives'].items()) + f"; any depth-z correlation with CI excluding 0: {M['a3_depth_any']}.", ""]
# ---- A4 ----
Rm = pd.read_csv(R + 'expR71_meru_radii.csv')
M['a4'] = dict(curv=float(Rm.curv.iloc[0]), radius_median=[float(Rm.radius_sqrtc_median.min()), float(Rm.radius_sqrtc_median.max())], radius_p95=[float(Rm.radius_sqrtc_p95.min()), float(Rm.radius_sqrtc_p95.max())],
               ratio_median=[float(Rm.lorentz_over_euclid_median.min()), float(Rm.lorentz_over_euclid_median.max())], ratio_centroids=[float(Rm.centroid_lorentz_over_euclid_median.min()), float(Rm.centroid_lorentz_over_euclid_median.max())])
L += ["## A4 — MERU radii (`expR71_meru_radii.csv`)", "", f"- Learned curvature c = {M['a4']['curv']:.3f} (all three sizes). Spatial norm × √c: median {M['a4']['radius_median'][0]:.3f}–{M['a4']['radius_median'][1]:.3f}, 95th percentile {M['a4']['radius_p95'][0]:.3f}–{M['a4']['radius_p95'][1]:.3f} across the six image clouds; Lorentz/Euclidean pairwise distance ratio at the median {M['a4']['ratio_median'][0]:.4f}–{M['a4']['ratio_median'][1]:.4f} (class centroids {M['a4']['ratio_centroids'][0]:.4f}–{M['a4']['ratio_centroids'][1]:.4f}).", ""]
# ---- A5 ----
if os.path.exists(R + 'expR72_budget_record_summary.csv'):
    B = pd.read_csv(R + 'expR72_budget_record_summary.csv')
    M['a5'] = {ds: dict(max_drift_ge1e5=float(B[B.dataset == ds].drift_ge1e5.max()), max_drift_over_sd=float(B[B.dataset == ds].drift_ge1e5_over_sd.max()), max_drift_all=float(B[B.dataset == ds].drift_all.max()), signs_stable=bool(B[B.dataset == ds].sign_stable.all())) for ds in ('imagenet', 'cifar100', 'dtd')}
    M['a5_budget_stable_imagenet'] = bool(M['a5']['imagenet']['max_drift_over_sd'] < 1.0)
    L += ["## A5 — quadruple-budget sweep under the record (`expR72_budget_record.csv`; {ViT-L, DINOv2-L, CLIP-B} × {ImageNet, CIFAR-100, DTD}; Haar × p99.9 × 200 at each budget)", "", "| cell | excess by budget | drift (≥ 10⁵) | drift / s.d. at 5×10⁵ | signs stable |", "|---|---|---|---|---|"]
    for r in B.itertuples(): L.append(f"| {NM[r.model]} / {r.dataset} | {r.excess_by_budget} | {r.drift_ge1e5:.4f} | {r.drift_ge1e5_over_sd:.2f} | {r.sign_stable} |")
    L += ["", "- " + "; ".join(f"{ds}: max drift {v['max_drift_ge1e5']:.4f} = {v['max_drift_over_sd']:.2f} s.d. (all budgets {v['max_drift_all']:.4f}), signs stable {v['signs_stable']}" for ds, v in M['a5'].items()) + f". 'Budget-stable' on ImageNet (drift below the excess's own s.d.): **{M['a5_budget_stable_imagenet']}**.", ""]
# ---- A6 ----
c52 = pd.read_csv(R + 'expR52_census_haar_p999_200.csv'); c52['frac'] = c52.excess / c52.null_mean; im = c52[c52.dataset == 'imagenet']
sl = pd.read_csv(R + 'expR62_samplelevel_record.csv'); sl['frac'] = sl.excess / sl.null_mean
g = c52[(c52.model == 'dinov2_g') & (c52.dataset == 'cifar100')].iloc[0]
M['a6'] = dict(imagenet_frac=[float(im.frac.min()), float(im.frac.max())], imagenet_frac_genuine=[float(im[im.genuine_bh].frac.min()), float(im[im.genuine_bh].frac.max())], sample_frac=[float(sl.frac.min()), float(sl.frac.max())],
               sample_frac_genuine=[float(sl[sl.genuine_bh].frac.min()), float(sl[sl.genuine_bh].frac.max())] if sl.genuine_bh.any() else None, dinov2g_c100_frac=float(g.frac), all_frac=[float(c52.frac.min()), float(c52.frac.max())])
L += ["## A6 — normalized effect size, excess / δ_null (fraction of the null reading)", "", f"- ImageNet centroids (12 cells): {100*M['a6']['imagenet_frac'][0]:+.0f}% to {100*M['a6']['imagenet_frac'][1]:+.0f}% (genuine cells {100*M['a6']['imagenet_frac_genuine'][0]:+.0f}% to {100*M['a6']['imagenet_frac_genuine'][1]:+.0f}%); all 72 cells {100*M['a6']['all_frac'][0]:+.0f}% to {100*M['a6']['all_frac'][1]:+.0f}%.",
      f"- Sample level (24 cells): {100*M['a6']['sample_frac'][0]:+.0f}% to {100*M['a6']['sample_frac'][1]:+.0f}%" + (f" (genuine cells {100*M['a6']['sample_frac_genuine'][0]:+.0f}% to {100*M['a6']['sample_frac_genuine'][1]:+.0f}%)" if M['a6']['sample_frac_genuine'] else "") + ".",
      f"- DINOv2-G / CIFAR-100: {100*M['a6']['dinov2g_c100_frac']:+.0f}% (excess {g.excess:+.4f} over a null of {g.null_mean:.4f}).", ""]
json.dump(M, open(R + 'final_pass_memo.json', 'w'), indent=1); open('ICLR2027/MEMO_final_pass.md', 'w').write("\n".join(L)); print("\n".join(L))
