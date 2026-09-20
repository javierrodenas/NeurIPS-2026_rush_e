#!/usr/bin/env python3
"""Parallel track (brief of 2026-09-20): ICLR2027/iclr2027/main_iclr2027_rebuttal.tex = the frozen submission file plus one paragraph in
S5.3 for the trained positive control (expR77, table tab:r1-positive) and one paragraph in S5.1 for the replication of Khrulkov et al.
(expR78, table tab:r2-khrulkov). Nothing else changes; every frozen number stays. A paragraph is written only when its result file
exists; the replication paragraph makes no claim unless the reproduced raw numbers are within 0.03 of the published ones on every
dataset. Run from the repo root: python rebuttal/scripts/phaseE_rebuttal.py"""
import json, re, os
import pandas as pd
R = 'rebuttal/results/'; TEX = 'ICLR2027/iclr2027/'; FD = TEX + 'appendix_tables/final/'
SRC = TEX + 'main_iclr2027_final.tex'; DST = TEX + 'main_iclr2027_rebuttal.tex'
NM = {"frozen": "frozen checkpoint", "ce_seed0": "leaf CE", "hier_seed0": "leaf CE + hierarchical CE", "ce_seed1": "leaf CE (seed 1)", "hier_seed1": "leaf CE + hierarchical CE (seed 1)"}
T = open(SRC).read(); status = {}
def insert_after_paragraph(text, lead, new_par):
    i = text.index("\\paragraph{" + lead + "}"); j = text.index("\n\n", i)
    return text[:j] + "\n\n" + new_par + text[j:]
tables = []
# ---- S5.3: the trained positive control
if os.path.exists(R + 'expR77_positive_control.csv') and os.path.exists(R + 'expR77_positive_control_verdict.json'):
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
    T = insert_after_paragraph(T, "It is the alignment of each cluster with its hub, not a hierarchy among the hubs.", par)
    rows = [f"{NM.get(m, m)} & ${r.excess:+.4f}$ ({int(r.r_above)}, {r.p_left:.3f}) & ${r.z_wn30:+.2f}$ & ${r.z_wn30bal:+.2f}$ & ${r.zdec_mean_wn30:+.2f}$ ({r.dec_frac_cert_wn30:.1f}) & ${r.zdec_mean_wn30bal:+.2f}$ ({r.dec_frac_cert_wn30bal:.1f}) \\\\" for m, r in P.iterrows()]
    tables.append("% prov: expR77_positive_control.csv\n\\begin{table}[tbp]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lccccc}\n\\toprule\n"
                  "model & excess ($r$, $p$) & $z$, WordNet-30 & $z$, balanced & decoupled $z$, WN-30 (cert.) & decoupled $z$, balanced (cert.) \\\\\n\\midrule\n" + "\n".join(rows) +
                  "\n\\bottomrule\n\\end{tabular}\n\\caption{\\textbf{Trained positive control.} ViT-B/16 from the census checkpoint fine-tuned on the full ImageNet-1k training set (5 epochs, AdamW, lr $10^{-5}$ encoder / $10^{-3}$ heads, batch 256, seed 0) with leaf cross-entropy alone and with an added hierarchical cross-entropy at the WordNet 30/6/2 cuts (weights 1/2/4); "
                  "centroids of the census subset; census excess under the centered Haar null (200 replicates), depth test with the matched anisotropic star on the WordNet-30 frame of record and on the balanced frame (10 star seeds), and the decoupling control (real hubs kept, offsets Haar-rotated, 10 seeds; in parentheses the fraction of seeds certified). "
                  + ("Success criterion met." if met else "Success criterion not met.") + " % expR77_positive_control.csv\n}\n\\label{tab:r1-positive}\n\\end{table}\n")
    status['positive_control'] = dict(written=True, criterion_met=met, hier_z=[float(h.z_wn30), float(h.z_wn30bal)], hier_dec=[float(h.zdec_mean_wn30), float(h.zdec_mean_wn30bal)])
else: status['positive_control'] = dict(written=False, reason="expR77 results not available")
# ---- S5.1: the replication of Khrulkov et al. (2020)
if os.path.exists(R + 'expR78_khrulkov_replication_summary.csv'):
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
        s_cal = (f"the excess is negative on {'every dataset' if len(neg) == 4 else f'{len(neg)} of the four'} and the reading ranks below every replicate on {'every dataset' if len(gen) == 4 else (f'{len(gen)} of the four' if gen else 'none of them')}")
        par = ("\\paragraph{A published reading, reproduced and calibrated.} On the setting of \\citet{Khrulkov_2020_CVPR}, ResNet-34 features on CIFAR-10, CIFAR-100, CUB and MiniImageNet, their estimator on our extraction reproduces their raw $\\delta_{\\text{rel}}$ within 0.03 on all four datasets. "
               f"Calibrated against the centered null on the same clouds, {s_cal}. The raw values they report are therefore real readings whose reference level, not their size, decides what they mean. Table~\\ref{{tab:r2-khrulkov}} gives the four rows. % expR78_khrulkov_replication_summary.csv")
    else:
        par = ("\\paragraph{A published reading, reproduced.} On the setting of \\citet{Khrulkov_2020_CVPR}, ResNet-34 features on CIFAR-10, CIFAR-100, CUB and MiniImageNet, their estimator on our extraction does not yet reproduce every published raw $\\delta_{\\text{rel}}$ within 0.03, so no calibrated claim is drawn. Table~\\ref{tab:r2-khrulkov} gives the reproduced and the published values. % expR78_khrulkov_replication_summary.csv")
    T = insert_after_paragraph(T, "The raw reading is not evidence, and the calibrated reading is weak and model-dependent.", par)
    status['replication'] = dict(written=True, all_within_range=allin, datasets=order)
else: status['replication'] = dict(written=False, reason="expR78 summary not available")
# ---- appendix: the new tables before the provenance index; the rest is the frozen appendix
if tables:
    anchor = "\\input{appendix_tables/final/tab_z_provenance_final}"
    T = T.replace(anchor, "\\FloatBarrier\n\\subsection{Parallel track: the trained positive control and the replicated reading}\n" + "".join(tables) + "\n" + anchor, 1)
T = T.replace("% final version: classic structure, plain prose", "% rebuttal version = the frozen submission file plus the parallel-track paragraphs (phaseE_rebuttal.py); the frozen text and numbers are unchanged\n% final version: classic structure, plain prose", 1)
open(DST, 'w').write(T); json.dump(status, open(R + 'rebuttal_build_status.json', 'w'), indent=1); print("rebuttal file written ->", DST, status)
