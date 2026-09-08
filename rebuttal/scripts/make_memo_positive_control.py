#!/usr/bin/env python3
"""Memo of the positive-control pass (Phase A): numbers only, no recommendations. Reads expR64 (R9), expR66 (R11) and,
if present, expR65 (R10); writes ICLR2027/MEMO_positive_control.md and rebuttal/results/positive_control_memo.json
(the numbers sweep_freeze.py checks)."""
import json
import numpy as np, pandas as pd
R = 'rebuttal/results/'
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
M = {}
# ---- R9 ----
S = pd.read_csv(R + 'expR64_implanted_depth_summary.csv'); D = pd.read_csv(R + 'expR64_implanted_depth.csv')
dep = D[(D.kind == 'depth') & (D.partition == 'rand6') & (D.s != 'real')].copy(); dep['s'] = dep.s.astype(float)
M['r9_runs'] = int(len(dep)); M['r9_s0_any_hit'] = int((S.hits_s0 > 0).sum()); M['r9_s0_max_z'] = float(S.z_max_s0.max())
M['r9_s_star'] = {r.model: (None if pd.isna(r.s_star) else float(r.s_star)) for r in S.itertuples()}
M['r9_s1_all_certified'] = bool((S.hits_s1 >= 4).all()); M['r9_s1_min_hits'] = int(S.hits_s1.min()); M['r9_z_mean_s1_range'] = [float(S.z_mean_s1.min()), float(S.z_mean_s1.max())]
power = {s: float((dep[dep.s == s].z <= -2).mean()) for s in (0.0, 0.25, 0.5, 0.75, 1.0)}; M['r9_power_by_s'] = power
cert = ["i21k_s", "i21k_b", "i21k_l", "dinov2_l"]; M['r9_s_star_certified'] = {m: M['r9_s_star'][m] for m in cert}; M['r9_s_star_others'] = {m: M['r9_s_star'][m] for m in NM if m not in cert}
M['r9_census_spread_max'] = float(S.census_exc_spread.max()); M['r9_census_spread_over_nullsd_max'] = float((S.census_exc_spread / S.census_null_sd).max())
M['r9_census_r_min'] = int(S.census_r_min.min()); M['r9_real_z_match'] = float((S.real_z_here - S.real_z_B34).abs().max())
M['r9_ratio_real'] = {r.model: (None if pd.isna(r.ratio_real) else round(float(r.ratio_real), 2)) for r in S.itertuples()} if 'ratio_real' in S.columns else {}
M['r9_tight'] = {r.model: dict(z_s0=float(r.tight_z_s0), z_s05=float(r.tight_z_s05), z_s1=float(r.tight_z_s1), hits_s1=int(r.tight_hits_s1), hits_s0=int(r.tight_hits_s0)) for r in S.itertuples()} if 'tight_z_s1' in S.columns else {}
from sklearn.cluster import AgglomerativeClustering
_WN = np.load('/media/HDD_4TB_2/javi/Platonic/results/features/imagenet/imagenet1k_wordnet_dist.npy'); _sup = AgglomerativeClustering(n_clusters=30, metric='precomputed', linkage='average').fit_predict(_WN)
M['r9_frame_sizes'] = sorted(np.bincount(_sup).tolist())
wn = D[(D.kind == 'depth') & (D.partition == 'wn6')].copy(); wn['s'] = wn.s.astype(float); M['r9_wn6_power_s1'] = float((wn[wn.s == 1.0].z <= -2).mean()) if len(wn) else None; M['r9_wn6_power_s05'] = float((wn[wn.s == 0.5].z <= -2).mean()) if len(wn) else None
# ---- R11 ----
J = pd.read_csv(R + 'expR66_joint_sensitivity_summary.csv'); top = J.dataset.isin(['imagenet', 'cifar100'])
M['r11_cells'] = int(len(J)); M['r11_record'] = [int(J.genuine_bh_record.sum()), int((J.genuine_bh_record & top).sum())]
M['r11_joint'] = [int(J.joint_genuine.sum()), int((J.joint_genuine & top).sum())]; M['r11_bootbh'] = [int(J.boot_bh_genuine.sum()), int((J.boot_bh_genuine & top).sum())]
M['r11_joint_drop'] = [f"{NM[r.model]}/{r.dataset}" for r in J[J.genuine_bh_record & ~J.joint_genuine].itertuples()]
M['r11_bootbh_drop'] = [f"{NM[r.model]}/{r.dataset}" for r in J[J.genuine_bh_record & ~J.boot_bh_genuine].itertuples()]
M['r11_bootbh_new'] = [f"{NM[r.model]}/{r.dataset}" for r in J[~J.genuine_bh_record & J.boot_bh_genuine].itertuples()]
M['r11_joint_new'] = [f"{NM[r.model]}/{r.dataset}" for r in J[~J.genuine_bh_record & J.joint_genuine].itertuples()]
M['r11_sd_boot_max'] = float(J.sd_boot.max()); M['r11_sd_boot_max_in'] = float(J[J.dataset == 'imagenet'].sd_boot.max()); M['r11_n_boot_min'] = int(J.n_boot.min())
# ---- R10 (optional) ----
try:
    F = pd.read_csv(R + 'expR65_hier_finetune.csv').set_index('obj'); L = pd.read_csv(R + 'expR65_train_log.csv')
    M['r10'] = {o: dict(excess=float(F.loc[o, 'excess']), r=int(F.loc[o, 'r_above']), p=float(F.loc[o, 'p_left']), depth=float(F.loc[o, 'depth']), z=float(F.loc[o, 'z_depth'])) for o in F.index}
    M['r10_final_acc'] = {o: float(L[f'{o}_acc'].iloc[-1]) for o in ('ce', 'hier')}; M['r10_steps'] = int(L.step.iloc[-1])
    M['r10_positive'] = bool(M['r10']['hier']['z'] <= -2 and M['r10']['ce']['z'] > -2) or bool(M['r10']['hier']['z'] < M['r10']['ce']['z'] - 1)
except FileNotFoundError:
    M['r10'] = None
json.dump(M, open(R + 'positive_control_memo.json', 'w'), indent=1)
fmt = lambda v: "none" if v is None else f"{v:g}"
lines = ["# Memo — positive-control pass (Phase A). Numbers only.", "",
         f"## R9 — implanted depth on real ImageNet centroids (`expR64_implanted_depth.csv`, {M['r9_runs']} depth runs: 12 backbones × 5 strengths × 5 implant seeds; partition rand6 = 6 super-hubs of 5, RandomState(0); WordNet 6-cut is nested but 10/5/11/2/1/1, run as `wn6` for seed 0)", "",
         "| backbone | s* (first s with z ≤ −2 in ≥ 4/5 seeds) | hits at s = 0 / 0.25 / 0.5 / 0.75 / 1 (of 5) | mean z at s = 1 | census excess s = 0 → s = 1 (null s.d.) | real z here / B34 |", "|---|---|---|---|---|---|"]
for r in S.itertuples():
    lines.append(f"| {NM[r.model]} | {fmt(None if pd.isna(r.s_star) else r.s_star)} | {r.hits_s0}/{r.hits_s025}/{r.hits_s05}/{r.hits_s075}/{r.hits_s1} | {r.z_mean_s1:+.1f} | {r.census_exc_s0:+.4f} → {r.census_exc_s1:+.4f} ({r.census_null_sd:.4f}) | {r.real_z_here:+.2f} / {r.real_z_B34:+.2f} |")
lines += ["", f"- Backbones declared hierarchical at s = 0 (any seed): **{M['r9_s0_any_hit']}**; max z at s = 0: {M['r9_s0_max_z']:+.2f}.",
          f"- Power by s (fraction of runs with z ≤ −2, all backbones): " + ", ".join(f"s={s:g}: {p:.2f}" for s, p in power.items()) + ".",
          f"- s* of the four backbones certified in the real data: " + ", ".join(f"{NM[m]} {fmt(v)}" for m, v in M['r9_s_star_certified'].items()) + "; of the other eight: " + ", ".join(f"{NM[m]} {fmt(v)}" for m, v in M['r9_s_star_others'].items()) + ".",
          f"- Census excess across s (implant seed 0): max spread {M['r9_census_spread_max']:.4f}, i.e. at most {M['r9_census_spread_over_nullsd_max']:.2f} null s.d.; minimum r over all implanted clouds {M['r9_census_r_min']}/200.",
          f"- WordNet 6-cut partition (seed 0): power at s = 1 {fmt(M['r9_wn6_power_s1'])}, at s = 0.5 {fmt(M['r9_wn6_power_s05'])}.",
          f"- Untouched cloud through the same code vs Table B34: max |Δz| = {M['r9_real_z_match']:.2f}.", "",
          f"- Within/between spread of the real clouds at K = 30 (RMS of within-cluster offsets over point-weighted RMS of hub displacements; the synthetic power sweep covered 0.1–0.6): " + ", ".join(f"{NM[m]} {fmt(v)}" for m, v in M['r9_ratio_real'].items()) + ".",
          f"- Frame cluster sizes (classes per WordNet-30 cluster): {M['r9_frame_sizes']}.",
          ("- Tight variant (same implant after shrinking the real offsets to within/between = 0.6; s ∈ {0, 0.5, 1}, 2 seeds): " + "; ".join(f"{NM[m]} z {v['z_s0']:+.1f}/{v['z_s05']:+.1f}/{v['z_s1']:+.1f}, hits at s=1 {v['hits_s1']}/2, at s=0 {v['hits_s0']}/2" for m, v in M['r9_tight'].items()) + ".") if M['r9_tight'] else "- Tight variant: not run.", "",
          f"## R11 — joint sensitivity of the genuine count (`expR66_joint_sensitivity_summary.csv`; {M['r11_cells']} cells, {M['r11_n_boot_min']} resamples each, 50 Haar replicates per resample)", "",
          f"- Record (Haar × p99.9 × 200, BH): **{M['r11_record'][0]}/72**, {M['r11_record'][1]}/24 on ImageNet + CIFAR-100.",
          f"- z_joint = excess / sqrt(sd_null² + sd_boot² + sd_est²) ≤ −2: **{M['r11_joint'][0]}/72**, {M['r11_joint'][1]}/24. Record-genuine cells that drop: {', '.join(M['r11_joint_drop']) or 'none'}; non-record cells that pass: {', '.join(M['r11_joint_new']) or 'none'}.",
          f"- Bootstrap-BH (genuine under BH in ≥ 27 of 30 resamples): **{M['r11_bootbh'][0]}/72**, {M['r11_bootbh'][1]}/24. Record-genuine cells that drop: {', '.join(M['r11_bootbh_drop']) or 'none'}; newly genuine: {', '.join(M['r11_bootbh_new']) or 'none'}.",
          f"- Bootstrap s.d. of the excess: max {M['r11_sd_boot_max']:.4f} over all cells, {M['r11_sd_boot_max_in']:.4f} on ImageNet.", ""]
if M['r10']:
    lines += ["## R10 — fine-tuned ViT-B/16 with injected hierarchy (`expR65_hier_finetune.csv`; census subset of 100 images/class, 2 epochs, same batches for both objectives)", "",
              "| model | census excess | r/200 (p) | depth | z |", "|---|---|---|---|---|"]
    for o in ('frozen', 'ce', 'hier'):
        if o in M['r10']: v = M['r10'][o]; lines.append(f"| {o} | {v['excess']:+.4f} | {v['r']} ({v['p']:.3f}) | {v['depth']:+.4f} | {v['z']:+.2f} |")
    lines += ["", f"- Final training-batch accuracy: ce {M['r10_final_acc']['ce']:.2f}, hier {M['r10_final_acc']['hier']:.2f} ({M['r10_steps']} steps). Positive-control criterion (hier certified and ce not, or hier's z clearly below ce's): **{M['r10_positive']}**.", ""]
else:
    lines += ["## R10", "", "- Not available (expR65 not finished).", ""]
open('ICLR2027/MEMO_positive_control.md', 'w').write("\n".join(lines)); print("\n".join(lines))
