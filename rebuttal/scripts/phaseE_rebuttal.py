#!/usr/bin/env python3
"""Parallel track (brief of 2026-09-20): ICLR2027/iclr2027/main_iclr2027_rebuttal.tex = the frozen submission file plus one paragraph in
S5.3 for the trained positive control (expR77, table tab:r1-positive) and one paragraph in S5.1 for the replication of Khrulkov et al.
(expR78, table tab:r2-khrulkov). Nothing else changes; every frozen number stays. A paragraph is written only when its result file
exists; the replication paragraph makes no claim unless the reproduced raw numbers are within 0.03 of the published ones on every
dataset. Run from the repo root: python rebuttal/scripts/phaseE_rebuttal.py"""
import json, re, os
import pandas as pd
R = os.environ.get('PLATONIC_RESULTS', 'rebuttal/results').rstrip('/') + '/'; TEX = 'ICLR2027/iclr2027/'; FD = TEX + 'appendix_tables/final/'
SRC = TEX + 'main_iclr2027_final.tex'; DST = TEX + 'main_iclr2027_rebuttal.tex'
NM = {"frozen": "frozen checkpoint", "ce_seed0": "leaf CE", "hier_seed0": "leaf CE + hierarchical CE", "ce_seed1": "leaf CE (seed 1)", "hier_seed1": "leaf CE + hierarchical CE (seed 1)"}
T = open(SRC).read(); status = {}
IN78 = "\\paragraph{A published reading is reproduced and calibrated.}" in T; IN79 = ("detects a three-level hierarchy with ViT-L\'s spectrum and noise" in T) or ("given a three-level hierarchy with ViT-L\'s spectrum and noise" in T); IN80 = "Implanted alignment is detected in" in T or "(c) Implanted hub alignment on the real ImageNet clouds" in open(FD + "tab_q05_power_final.tex").read()   # already in the submission (brief of 2026-09-21; since 2026-09-23 only Table 10(c) carries the percentage, the (iii) sentence went with the page budget)
def insert_after_paragraph(text, lead, new_par):
    i = text.index("\\paragraph{" + lead + "}"); j = text.index("\n\n", i)
    return text[:j] + "\n\n" + new_par + text[j:]
tables = []
# ---- S5.3: the trained positive control
IN77 = "keeps firing once decoupled" in T
if os.path.exists(R + 'expR77_positive_control.csv') and os.path.exists(R + 'expR77_positive_control_verdict.json') and (not IN77 or ('hier_seed1' in pd.read_csv(R + 'expR77_positive_control.csv').model.values and 'over two seeds' not in T and 'over 2 seeds' not in T and 'in both seeds' not in T)):   # seed 0 in the submission -> the second seed goes to the rebuttal file only (rule of 2026-09-20)
    P = pd.read_csv(R + 'expR77_positive_control.csv').set_index('model'); V = json.load(open(R + 'expR77_positive_control_verdict.json'))['verdict']
    v0 = next((v for v in V if v['seed'] == 'seed0'), None); h = P.loc['hier_seed0']; c = P.loc['ce_seed0']; f = P.loc['frozen']
    cert = lambda r: (r.z_wn30 <= -2) and (r.z_wn30bal <= -2); dec = lambda r: (r.zdec_mean_wn30 <= -2) and (r.zdec_mean_wn30bal <= -2)
    s_h = ("is certified on both frames" if cert(h) else ("is certified on one frame only" if (h.z_wn30 <= -2) != (h.z_wn30bal <= -2) else "is not certified on either frame")) + (" and still fires once cluster orientations are randomized" if dec(h) else " and does not fire once cluster orientations are randomized")
    s_c = "neither the leaf model nor the frozen checkpoint is certified" if not (cert(c) or (c.z_wn30 <= -2) or (c.z_wn30bal <= -2) or (f.z_wn30 <= -2) or (f.z_wn30bal <= -2)) else "the leaf model or the frozen checkpoint is certified as well"
    met = bool(v0 and v0['criterion_met'])
    par = ("\\paragraph{A trained positive control.} ViT-B/16 from the census checkpoint was fine-tuned on the full ImageNet-1k training set, "
           "once with leaf cross-entropy alone and once with an added hierarchical cross-entropy at the WordNet 30, 6 and 2 cuts, weighted 1, 2 and 4, with identical batches and schedule. "
           f"Under the record the hierarchical model {s_h}, and {s_c}. "
           + ("The success criterion fixed before the run is met: the injected hierarchy lives in the hubs and the test sees it." if met else "The success criterion fixed before the run is not met, and the outcome is reported as it is.")
           + " Table~\\ref{tab:r1-positive} gives the readings. % expR77_positive_control.csv, expR77_positive_control_verdict.json")
    if not IN77:
        LEAD53 = next(l for l in ("No hierarchy above the superclasses is found in the supervised and contrastive backbones.", "No hierarchy above the superclasses is found.", "Whether the superclasses form a hierarchy is left open.") if ("\\paragraph{" + l + "}") in T)
        T = insert_after_paragraph(T, LEAD53, par)
    else:   # 2026-09-23: the second seed (expR76 --seed 1, expR77 shards) after S5.3's 'What the test can see.', numbers from the csv
        h1, c1 = P.loc['hier_seed1'], P.loc['ce_seed1']; v1 = next((v for v in V if v['seed'] == 'seed1'), None); met1 = bool(v1 and v1['criterion_met'])
        par = ("\\paragraph{A second seed of the trained control.} A second seed of both fine-tunings repeats the pattern of the first. "
               f"The hierarchical model is certified on both frames, $z$ ${h1.z_wn30:+.2f}$ and ${h1.z_wn30bal:+.2f}$, and keeps firing once decoupled, {int(round(h1.dec_frac_cert_wn30 * 10))} of 10 on the frame of record and {int(round(h1.dec_frac_cert_wn30bal * 10))} of 10 on the balanced frame. "
               f"The leaf-CE model is certified intact as well, $z$ ${c1.z_wn30:+.2f}$ and ${c1.z_wn30bal:+.2f}$, and this time also fires once decoupled on the frame of record, {int(round(c1.dec_frac_cert_wn30 * 10))} of 10 at a mean $z$ of ${c1.zdec_mean_wn30:+.2f}$ against ${h1.zdec_mean_wn30:+.2f}$ for the hierarchical model. "
               + ("The success criterion fixed before the run is met for this seed." if met1 else "The success criterion fixed before the run is not met for this seed either, and the outcome is reported as it is.")
               + " Table~\\ref{tab:r1-positive} gives both seeds. % expR77_positive_control.csv, expR77_positive_control_verdict.json")
        T = insert_after_paragraph(T, "What the test can see.", par)
    rows = [f"{NM.get(m, m)} & ${r.excess:+.4f}$ ({int(r.r_above)}, {r.p_left:.3f}) & ${r.z_wn30:+.2f}$ & ${r.z_wn30bal:+.2f}$ & ${r.zdec_mean_wn30:+.2f}$ ({r.dec_frac_cert_wn30:.1f}) & ${r.zdec_mean_wn30bal:+.2f}$ ({r.dec_frac_cert_wn30bal:.1f}) \\\\" for m, r in P.iterrows()]
    tables.append("% prov: expR77_positive_control.csv\n\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccccc}\n\\toprule\n"
                  "model & excess ($r$, $p$) & $z$, WordNet-30 & $z$, balanced & decoupled $z$, WN-30 (cert.) & decoupled $z$, balanced (cert.) \\\\\n\\midrule\n" + "\n".join(rows) +
                  "\n\\bottomrule\n\\end{tabular}\n\\caption{\\textbf{Trained positive control.} ViT-B/16 from the census checkpoint fine-tuned on the full ImageNet-1k training set (5 epochs, AdamW, lr $10^{-5}$ encoder / $10^{-3}$ heads, batch 256, seed 0) with leaf cross-entropy alone and with an added hierarchical cross-entropy at the WordNet 30/6/2 cuts (weights 1/2/4); "
                  "centroids of the census subset; census excess under the centered Haar null (200 replicates), depth test with the matched anisotropic star on the WordNet-30 frame of record and on the balanced frame (10 star seeds), and the decoupling control (real hubs kept, offsets Haar-rotated, 10 seeds; in parentheses the fraction of seeds certified). "
                  + ("Success criterion met." if met else "Success criterion not met.") + " % expR77_positive_control.csv\n}\n\\label{tab:r1-positive}\n\\end{table}\n")
    status['positive_control'] = dict(written=True, criterion_met=met, hier_z=[float(h.z_wn30), float(h.z_wn30bal)], hier_dec=[float(h.zdec_mean_wn30), float(h.zdec_mean_wn30bal)])
else: status['positive_control'] = dict(written=False, reason="in the submission" if IN77 else "expR77 results not available")
# ---- S5.3: priority 1b (expR79): synthetic deep hierarchy with ViT-L's spectrum, and the WordNet Poincare embeddings, through the depth test
if not IN79 and os.path.exists(R + 'expR79_synthetic_deep_poincare.csv') and pd.read_csv(R + 'expR79_synthetic_deep_poincare.csv').cloud.str.startswith('wordnet_poincare_d50').any():   # the last row of the run: partial results never enter
    E = pd.read_csv(R + 'expR79_synthetic_deep_poincare.csv'); E['cert'] = E.z <= -2; E['dec_cert'] = E.zdec_mean <= -2
    deep = E[E.cloud == 'synthetic_deep_vitl_spectrum']; flat = E[E.cloud == 'synthetic_flat_vitl_spectrum']; poi = E[E.cloud.str.startswith('wordnet_poincare')]
    def fires(df): return f"{int(df.cert.sum())} of {len(df)}"
    s_deep = f"the deep synthetic hierarchy fires in {fires(deep)} seeds" if len(deep) else "the deep synthetic hierarchy is not available"
    s_flat = f", the flat control in {fires(flat)}" if len(flat) else ""
    s_poi = (" The WordNet Poincar\\'e embeddings " + ("fire at every dimension tried" if len(poi) and poi.cert.all() else ("fire at no dimension tried" if len(poi) and not poi.cert.any() else "fire at some dimensions only")) + f" ({', '.join(f'$d{{=}}{int(r.dim)}$: $z{{=}}{r.z:+.2f}$' for _, r in poi.iterrows())}).") if len(poi) else ""
    par = ("\\paragraph{A deep hierarchy at the real noise level.} A synthetic cloud with ViT-L's real ImageNet spectrum and a three-level implanted hierarchy at ViT-L's real within/between ratio was read by the same test. "
           f"At $K{{=}}30$ {s_deep}{s_flat}, and the decoupling control fires in {int(deep.dec_cert.sum()) if len(deep) else 0} of {len(deep)} deep seeds.{s_poi} "
           "Table~\\ref{tab:r1b-deep} gives every reading. % expR79_synthetic_deep_poincare.csv")
    T = insert_after_paragraph(T, "Whether the superclasses form a hierarchy is left open.", par)
    rows = [f"{r.cloud.replace('_', ' ')} & {int(r.seed)} & {int(r.dim)} & ${r.excess:+.4f}$ ({int(r.r_above)}) & ${r.z:+.2f}$ & ${r.zdec_mean:+.2f}$ $\\pm$ {r.zdec_sd:.2f} \\\\" for _, r in E.iterrows()]
    tables.append("% prov: expR79_synthetic_deep_poincare.csv\n\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccccc}\n\\toprule\n"
                  "cloud & seed & $d$ & excess ($r$) & depth $z$ & decoupled $z$ \\\\\n\\midrule\n" + "\n".join(rows) +
                  "\n\\bottomrule\n\\end{tabular}\n\\caption{\\textbf{Priority 1b: a deep hierarchy at the real noise level, and the WordNet Poincar\\'e embeddings.} Synthetic clouds with ViT-L's real ImageNet spectrum and a three-level implanted hierarchy (nested 2/6/30 cuts of the frame) at ViT-L's real within/between ratio, and a flat two-level control, 5 seeds each; the WordNet Poincar\\'e embeddings of Nickel and Kiela (2017) trained on the transitive closure of the tree spanning the 1000 ImageNet leaves at $d{=}10$ and $50$. "
                  "Census excess under the centered Haar null (200 replicates, rank in parentheses), depth test with the matched anisotropic star at $K{=}30$ (10 star seeds), and the decoupling control (mean $\\pm$ s.d.\\ over 10 seeds). % expR79_synthetic_deep_poincare.csv\n}\n\\label{tab:r1b-deep}\n\\end{table}\n")
    status['deep_synthetic'] = dict(written=True, deep_fires=fires(deep) if len(deep) else None, flat_fires=fires(flat) if len(flat) else None, poincare=[dict(dim=int(r.dim), z=float(r.z)) for _, r in poi.iterrows()])
else: status['deep_synthetic'] = dict(written=False, reason="in the submission" if IN79 else "expR79 not finished (no Poincare d=50 row yet)")
# ---- S5.3: priority 1c (expR80): implanted alignment; enters the rebuttal file only when the decision rule is NOT met (otherwise it is in the submission already)
if not IN80 and os.path.exists(R + 'expR80_decision.csv') and os.path.exists(R + 'expR80_implanted_alignment.csv'):
    d80 = pd.read_csv(R + 'expR80_decision.csv').iloc[0]; A80 = pd.read_csv(R + 'expR80_implanted_alignment.csv'); A80['hit'] = A80.z_depth <= -2
    met80 = bool(d80.rule_power_ge_0_8_fa_le_0_05)
    if not met80:
        SS = sorted(A80.s.unique()); pm = A80.groupby(['model', 's']).hit.mean().unstack(); pooled = A80.groupby('s').hit.mean()
        n1, h1 = int((A80.s == 1.0).sum()), int(A80[A80.s == 1.0].hit.sum()); n0, h0 = int((A80.s == 0.0).sum()), int(A80[A80.s == 0.0].hit.sum())
        NM80 = {"i21k_t": "ViT-T", "i21k_s": "ViT-S", "i21k_b": "ViT-B", "i21k_l": "ViT-L", "dinov1_b": "DINO-B", "dinov2_s": "DINOv2-S", "dinov2_b": "DINOv2-B", "dinov2_l": "DINOv2-L", "dinov2_g": "DINOv2-g", "clip_b": "CLIP-B", "clip_l": "CLIP-L", "siglip_b": "SigLIP-B"}
        ORD = [m for m in NM80 if m in pm.index] + [m for m in pm.index if m not in NM80]; pm = pm.loc[ORD]   # the paper's backbone order
        full = [NM80.get(m, m) for m in pm.index if pm.loc[m, 1.0] == 1.0]; none_ = [NM80.get(m, m) for m in pm.index if pm.loc[m, 1.0] == 0.0]
        par = ("\\paragraph{Implanted alignment on the real clouds.} From the decoupled cloud of each backbone, each cluster's principal axis was rotated toward its hub direction by a fraction $s$ of the angle, hubs and within-cluster spectra unchanged, and the test was read at every $s$. "
               f"Implanted alignment is detected in {h1} of {n1} runs at full strength and in {h0} of {n0} at zero, a power of {float(d80.power_s1):.2f} against the pre-set bar of 0.8, so the power of the test for alignment is not established on the real clouds and the submission keeps the certification without a power statement. "
               + (f"Detection at full strength is complete in {', '.join(full)} and absent in {', '.join(none_)}. " if full or none_ else "")
               + "Table~\\ref{tab:r1c-implant} gives the detection rate per backbone. % expR80_implanted_alignment.csv, expR80_decision.csv")
        T = insert_after_paragraph(T, "Whether the superclasses form a hierarchy is left open.", par)
        rows = [NM80.get(m, m) + " & " + " & ".join(f"{pm.loc[m, s]:.1f}" for s in SS) + " \\\\" for m in pm.index] + ["\\midrule pooled & " + " & ".join(f"{pooled[s]:.2f}" for s in SS) + " \\\\"]
        tables.append("% prov: expR80_implanted_alignment.csv, expR80_decision.csv\n\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{6pt}\n\\begin{tabular}{l" + "c" * len(SS) + "}\n\\toprule\n"
                      "model & " + " & ".join(f"$s{{=}}{s:g}$" for s in SS) + " \\\\\n\\midrule\n" + "\n".join(rows) +
                      "\n\\bottomrule\n\\end{tabular}\n\\caption{\\textbf{Priority 1c: implanted alignment on the real ImageNet clouds.} From the decoupled cloud of expR74 (real hubs kept, offsets Haar-rotated), each cluster's principal axis rotated toward its hub direction by a fraction $s$ of the angle, hubs and within-cluster spectra unchanged; detection rate at $z\\le-2$ under the matched anisotropic star at $K{=}30$, "
                      f"{int(d80.n_seeds)} seeds per backbone and pooled. Decision rule of the brief: power $\\ge 0.8$ at $s{{=}}1$ with false alarms $\\le 0.05$ at $s{{=}}0$; measured power {float(d80.power_s1):.2f}, false alarms {float(d80.false_alarms_s0):.2f}: not met, so nothing entered the submission. % expR80_decision.csv\n}}\n\\label{{tab:r1c-implant}}\n\\end{{table}}\n")
        status['implanted_alignment'] = dict(written=True, rule_met=False, power_s1=float(d80.power_s1), false_alarms_s0=float(d80.false_alarms_s0), hits1=h1, runs1=n1, hits0=h0, runs0=n0)
    else: status['implanted_alignment'] = dict(written=False, rule_met=True, reason="rule met: Table 9c, the Figure 4b curve and the S5.3 sentence are in the submission file itself", power_s1=float(d80.power_s1), false_alarms_s0=float(d80.false_alarms_s0))
else: status['implanted_alignment'] = dict(written=False, reason="in the submission (limitation iii and Table 9c)" if IN80 else "expR80 not merged yet")
# ---- S5.1: the replication of Khrulkov et al. (2020)
if not IN78 and os.path.exists(R + 'expR78_khrulkov_replication_summary.csv'):
    S = pd.read_csv(R + 'expR78_khrulkov_replication_summary.csv').set_index('dataset'); order = [d for d in ("cifar10", "cifar100", "cub", "miniimagenet") if d in S.index]
    DN = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cub": "CUB-200", "miniimagenet": "MiniImageNet"}
    allin = len(order) == 4 and bool(S.loc[order].within_range.all())
    rows = [f"{DN[d]} & {S.loc[d].theirs:.2f} & {S.loc[d].ours_raw_mean:.3f} $\\pm$ {S.loc[d].ours_raw_sd:.3f} & ${S.loc[d].excess_mean:+.4f}$ $\\pm$ {S.loc[d].excess_sd:.4f} & {S.loc[d].r_above_mean:.0f}/200 & {S.loc[d].p_left_max:.3f} \\\\" for d in order]
    tables.append("% prov: expR78_khrulkov_replication.csv, expR78_khrulkov_replication_summary.csv\n\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccccc}\n\\toprule\n"
                  "dataset & their $\\delta_{\\text{rel}}$ & our $\\delta_{\\text{rel}}$ & excess & rank $r$ & largest $p$ \\\\\n\\midrule\n" + "\n".join(rows) +
                  "\n\\bottomrule\n\\end{tabular}\n\\caption{\\textbf{A published reading reproduced and calibrated.} The setting of Khrulkov et al. (2020), Table 1, ResNet-34 row: penultimate features of an ImageNet-pretrained ResNet-34 (torchvision, no substitution) on class-balanced batches of 1500 points, their estimator (exact $\\delta$ on the batch, $\\delta_{\\text{rel}}=2\\delta/\\text{diam}$; mean $\\pm$ s.d.\\ over 10 batches) next to their published value, and on the same clouds the record instrument: excess over the centered Haar null (200 replicates, 99.9th-percentile statistic), the mean rank of the reading among the replicates and the largest left-tail $p$ over the 10 batches. "
                  + ("Every reproduced raw value is within 0.03 of the published one." if allin else "Not every reproduced raw value is within 0.03 of the published one; no claim is drawn.") + " % expR78_khrulkov_replication_summary.csv\n}\n\\label{tab:r2-khrulkov}\n\\end{table}\n")
    if allin:
        gen = [d for d in order if S.loc[d].p_left_max <= 0.05]; neg = [d for d in order if S.loc[d].excess_mean < 0]
        DNl = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cub": "CUB", "miniimagenet": "MiniImageNet"}
        s_cal = (f"the excess is negative on {'every dataset' if len(neg) == 4 else f'{len(neg)} of the four'} and the largest left-tail $p$ over the ten batches is at or below 0.05 on "
                 + ('every dataset' if len(gen) == 4 else (f"{len(gen)} of the four, {' and '.join(DNl[d] for d in gen)}," if gen else 'none of them')))
        par = ("\\paragraph{A published reading, reproduced and calibrated.} On the setting of \\citet{Khrulkov_2020_CVPR}, ResNet-34 features on CIFAR-10, CIFAR-100, CUB and MiniImageNet, their estimator on our extraction reproduces their raw $\\delta_{\\text{rel}}$ within 0.03 on all four datasets. "
               f"Calibrated against the centered null on the same clouds, {s_cal}. The raw values they report are therefore real readings whose reference level, not their size, decides what they mean. Table~\\ref{{tab:r2-khrulkov}} gives the four rows. % expR78_khrulkov_replication_summary.csv")
    else:
        par = ("\\paragraph{A published reading, reproduced.} On the setting of \\citet{Khrulkov_2020_CVPR}, ResNet-34 features on CIFAR-10, CIFAR-100, CUB and MiniImageNet, their estimator on our extraction does not yet reproduce every published raw $\\delta_{\\text{rel}}$ within 0.03, so no calibrated claim is drawn. Table~\\ref{tab:r2-khrulkov} gives the reproduced and the published values. % expR78_khrulkov_replication_summary.csv")
    T = insert_after_paragraph(T, "Raw readings are not evidence, ours or published, and calibrated ones are weak and model-dependent." if "Raw readings are not evidence, ours or published" in T else "The raw reading is not evidence, and the calibrated reading is weak and model-dependent.", par)
    status['replication'] = dict(written=True, all_within_range=allin, datasets=order)
else: status['replication'] = dict(written=False, reason="in the submission" if IN78 else "expR78 summary not available")
# ---- appendix: the new tables before the provenance index; the rest is the frozen appendix
if tables:
    anchor = "\\input{appendix_tables/final/tab_z_provenance_final}"
    T = T.replace(anchor, "\\FloatBarrier\n\\subsection{Parallel track: the trained positive control, the deep-hierarchy and implanted-alignment controls, and the replicated reading}\n" + "".join(tables) + "\n" + anchor, 1)
T = T.replace("% final version: classic structure, plain prose", "% rebuttal version = the frozen submission file plus the parallel-track paragraphs (phaseE_rebuttal.py); the frozen text and numbers are unchanged\n% final version: classic structure, plain prose", 1)
open(DST, 'w').write(T); json.dump(status, open(R + 'rebuttal_build_status.json', 'w'), indent=1); print("rebuttal file written ->", DST, status)
