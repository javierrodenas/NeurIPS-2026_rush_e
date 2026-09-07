#!/usr/bin/env python3
"""Restructuring pass: fills phaseC_main_body.tex.tmpl with numbers read from the result files (R7 = expR62, R8 = expR63,
exp2b gains, exp1 Gaussian band, expR52 ImageNet range), splices it into main_iclr2027.tex between \\begin{abstract} and the
Ethics Statement, moves the demoted paragraphs into their appendix sections, decides Figure 2(b) by the numbers
(phaseC_fig2b.json, read by make_fig_overview.py) and checks labels / forbidden words. Run from the repo root after
make_memo_R7R8.py printed GO."""
import json, re, sys
import pandas as pd
R = 'rebuttal/results/'; S = 'rebuttal/scripts/'
M = json.load(open(R + 'phaseC_memo.json'))
assert not M['stop_a_samplelevel_mostly_genuine'] and not M['stop_b_meru_depth_beyond_twin'], "memo says STOP; do not restructure"
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
DSN = {"imagenet":"ImageNet","cifar100":"CIFAR-100","cifar10":"CIFAR-10","dtd":"DTD","fashionmnist":"FashionMNIST","mnist":"MNIST"}
F = {}
# --- R7 sample level ---
F['SL_SUP_LO'], F['SL_SUP_HI'] = f"{M['sl_sup_lo']:.3f}", f"{M['sl_sup_hi']:.3f}"; F['SL_WITHIN'] = str(M['sl_within'])
sl0 = pd.read_csv(R + 'expR62_samplelevel_record.csv'); gen0 = sl0[sl0.genuine_bh]
def compress(models):
    """['dinov2_s','dinov2_b','siglip_b'] -> 'DINOv2-S/B and SigLIP-B' (sizes merged within a family, census order kept)."""
    order = list(NM); ms = sorted(models, key=order.index); out = []; i = 0
    while i < len(ms):
        fam, size = NM[ms[i]].rsplit('-', 1); sizes = [size]; j = i + 1
        while j < len(ms) and NM[ms[j]].rsplit('-', 1)[0] == fam: sizes.append(NM[ms[j]].rsplit('-', 1)[1]); j += 1
        out.append(f"{fam}-{'/'.join(sizes)}"); i = j
    return out[0] if len(out) == 1 else ", ".join(out[:-1]) + " and " + out[-1]
parts = [f"{compress(list(gen0[gen0.dataset == ds].model))} on {DSN[ds]}" for ds in ('cifar100', 'dtd') if (gen0.dataset == ds).any()]
F['SL_GEN_CLAUSE'] = (f" and genuine in {M['sl_genuine']} ({'; '.join(parts)})" if M['sl_genuine'] else " and genuine in none")
F['SL_GEN_EXC_HI'] = f"${gen0.excess.min():+.3f}$" if len(gen0) else "$0$"; F['SL_GEN_EXC_HI_ABS'] = f"${abs(gen0.excess.min()):.3f}$" if len(gen0) else "$0$"
# --- R8 MERU ---
F['MERU_EXC_LO'], F['MERU_EXC_HI'] = f"${M['meru_in_exc_range'][1]:+.3f}$", f"${M['meru_in_exc_range'][0]:+.3f}$"   # "lo to hi" reads least to most negative
F['CLIP_EXC_LO'], F['CLIP_EXC_HI'] = f"${M['clip_in_exc_range'][1]:+.3f}$", f"${M['clip_in_exc_range'][0]:+.3f}$"
F['MERU_R_CLAUSE'] = ", every replicate above the real value in all six" if M['meru_in_all_r200'] else ""
F['MERU_NATGAP'] = f"${M['meru_in_nat_vs_euc_maxgap']:.4f}$"
mz, cz = M['meru_in_z_range'], M['clip_in_z_range']
if not M['meru_any_depth_in']:
    F['MERU_DEPTH_SENT'] = f"neither MERU nor its twin is more hierarchical than a matched star at any size ($z$ from ${mz[0]:+.1f}$ to ${mz[1]:+.1f}$ for MERU, ${cz[0]:+.1f}$ to ${cz[1]:+.1f}$ for CLIP)"
else:
    F['MERU_DEPTH_SENT'] = f"MERU is never more hierarchical than its twin ($z$ from ${mz[0]:+.1f}$ to ${mz[1]:+.1f}$ for MERU, ${cz[0]:+.1f}$ to ${cz[1]:+.1f}$ for CLIP)"
# --- gains (exp2b, normalized stack): Poincare over cosine, few-shot, contrastive VLMs vs the rest on the hierarchical sets ---
b = pd.read_csv(R + 'exp2b_normalized_stack.csv'); H = b[b.dataset.isin(['imagenet', 'cifar100', 'cifar10', 'dtd'])]
g = H.FS_HN_COS_diff * 100; contr = H.paradigm.str.lower().str.startswith('contr')
gmax_c, gmax_o = float(g[contr].max()), float(g[~contr].max())
F['GAIN_MAX'] = f"$+{gmax_c:.1f}$"
assert gmax_o < 0.5, f"non-contrastive Poincare-over-cosine gain {gmax_o:.2f} pp is not 'nothing'; reword by hand"
# --- Khrulkov rule on the Gaussian band (exp1: variant gauss, deduplicated by d) ---
e1 = pd.read_csv(R + 'exp1_delta_controls.csv'); ga = e1[e1.variant == 'gauss'].drop_duplicates('d').sort_values('d')
d_lo, d_hi = int(ga.d.iloc[0]), int(ga.d.iloc[-1]); dl_lo, dl_hi = float(ga.delta_max.iloc[0]), float(ga.delta_max.iloc[-1])
c_lo, c_hi = (0.144 / (2 * dl_lo)) ** 2, (0.144 / (2 * dl_hi)) ** 2
F['C_LO'], F['C_HI'], F['D_LO'], F['D_HI'] = f"${c_lo:.2f}$", f"${c_hi:.1f}$", str(d_lo), str(d_hi)
assert abs(dl_lo - 0.104) < 0.001 and abs(dl_hi - 0.046) < 0.001, (dl_lo, dl_hi)
# --- ImageNet class-level excess range (record) ---
c52 = pd.read_csv(R + 'expR52_census_haar_p999_200.csv'); im = c52[c52.dataset == 'imagenet']
F['IN_EXC_HI'], F['IN_EXC_LO'] = f"${im.excess.max():+.3f}$", f"${im.excess.min():+.3f}$"
# --- Figure 2(b): decide by the numbers ---
sl = pd.read_csv(R + 'expR62_samplelevel_record.csv'); t53 = pd.read_csv(R + 'expR53_text_haar_p999_200.csv').set_index('model')
gen52 = c52[c52.genuine_bh]
bge_d = float(t53.loc['bge_base', 'delta']); bge_gap = float((c52.delta - bge_d).abs().min()); bge_cell = c52.iloc[int((c52.delta - bge_d).abs().argmin())]
best = None
for r in sl[~sl.genuine_bh].itertuples():
    j = int((gen52.delta - r.delta_999).abs().argmin()); q = gen52.iloc[j]; gap = abs(float(q.delta) - r.delta_999)
    if best is None or gap < best[0]: best = (gap, r.model, r.dataset, q.model, q.dataset, r.delta_999, r.excess, float(q.delta), float(q.excess))
use_sample = best is not None and best[0] <= bge_gap
dec = dict(mode="sample" if use_sample else "bge", sample_cell=[best[1], best[2]] if best else None, class_cell=[best[3], best[4]] if best else None,
           gap_sample=best[0] if best else None, gap_bge=bge_gap, bge_class_cell=[bge_cell.model, bge_cell.dataset],
           sample_raw=best[5] if best else None, sample_exc=best[6] if best else None, class_raw=best[7] if best else None, class_exc=best[8] if best else None)
json.dump(dec, open(R + 'phaseC_fig2b.json', 'w'), indent=1); print("fig2b decision:", dec)
if use_sample:
    F['FIG2B_CAPTION'] = (f"A sample-level cell ({NM[best[1]]} on {DSN[best[2]]} images) and a class-level cell ({NM[best[3]]} on {DSN[best[4]]} centroids) share the same raw value with opposite verdicts, within noise and genuine.")
    F['FIG2B_SOURCES'] = "expR62_samplelevel_record.csv"
else:
    F['FIG2B_CAPTION'] = f"A text embedder (BGE-base) and a vision cell ({NM[bge_cell.model]} on {DSN[bge_cell.dataset]}) share the same raw value with opposite null verdicts."
    F['FIG2B_SOURCES'] = "expR53_text_haar_p999_200.csv"
# --- fill the template ---
body = open(S + 'phaseC_main_body.tex.tmpl').read()
for k, v in F.items(): body = body.replace("{{" + k + "}}", v)
left = re.findall(r"\{\{[A-Z0-9_]+\}\}", body); assert not left, left
p = 'iclr2027/iclr2027/main_iclr2027.tex'; T = open(p).read()
i = T.index("\\begin{abstract}"); j = T.index("\\subsubsection*{Ethics Statement}")
old_main = T[i:j]; T = T[:i] + body + T[j:]
def rep(old, new):
    global T
    assert T.count(old) == 1, f"{T.count(old)} matches: {old[:80]!r}"; T = T.replace(old, new)
# --- demoted paragraphs -> appendix ---
rep("\\label{app:interventions}\n", "\\label{app:interventions}\n\\paragraph{The form tracks training.} Two interventions move it (Figure~\\ref{fig:causal}; raw $\\delta$, within architecture): fine-tuning on a task with no class hierarchy raises $\\delta$ by 19--91\\%, and $\\delta$ falls across transformer depth. Shuffling the embedding-to-label correspondence leaves it unchanged, as a null-invariant quantity should. Random-weight networks give higher raw $\\delta$ but near-Gaussian features, so they are not counted as evidence; excess-based versions of these interventions await stored point clouds. On matched architecture (ViT-B supervised/DINO/CLIP) all three objectives yield genuine form on most datasets, and the family separations emerge with scale rather than at fixed size. % analysis4_finetuning.csv, e1_delta_by_layer.csv, e5_random_control.csv, night/arch_matched_triplet.csv\n")
rep("\\subsection{Text census}\n", "\\subsection{Text census}\n\\label{app:textcensus}\n\\paragraph{Extraction, anisotropy and templates.} Extraction is part of the measurement: left-padded batching moves GPT-2's $\\hat\\delta$ by up to $0.02$, of the order of the fragile excesses; the padding-free protocol is the record (Table~\\ref{tab:b28-extraction}). Genuineness is not a proxy for anisotropy: GTE-base has the highest effective rank and is not genuine. The only LLM-backboned embedder tested (GTE-Qwen2-1.5B) is not genuine under the record (Table~\\ref{tab:b23-text200}). Causal-LM values are template-sensitive but the scale ordering is template-robust (Tables~\\ref{tab:a7} and \\ref{tab:b14-templates}). % expR53_text_haar_p999_200.csv, exp18b_text_anisotropy.csv, expR49_template_nulls_bs1.csv\n\n\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.8\\linewidth]{figures/fig_text_nulls.pdf}\n\\caption{\\textbf{In text, beyond-null structure depends on recipe and scale: robust at the larger GPT-2 sizes, fragile at the smaller.} Text census of record (padding-free extraction, Haar null, 99.9th-percentile statistic, 200 replicates; error bars combine null and estimator s.d.; hollow bars are not genuine, BH $p>0.05$; OLMo-7B, not re-extracted, omitted). Pythia and OLMo-1B are genuine throughout; classic sentence embedders are not on this probe despite raw values equal to genuinely structured vision models. % expR53_text_haar_p999_200.csv\n}\n\\label{fig:textnulls}\n\\end{figure}\n")
rep("\\subsection{Class-count vs hierarchy depth}\n", "\\subsection{Class-count vs hierarchy depth}\n\\label{app:csweep}\n\\paragraph{Hierarchy depth, not class count.} Because $C$ sets the quadruple population, we sweep ImageNet subsets $C\\in\\{10,\\dots,1000\\}$ in two compositions: random (spanning the full hierarchy) and WordNet-coherent (siblings, effectively \\emph{flat}). At fixed $C$, random subsets are more exploitable than coherent ones and carry more beyond-null structure at small $C$ (CLIP-L at every $C$, DINOv2-L up to $C{=}100$, DINOv2-G up to $C{=}20$; beyond that the two compositions converge; Table~\\ref{tab:b3-csweep}). Random subsets still score $-0.08$ to $-0.14$ at $C{=}10$, so MNIST's flat verdict is not a class-count artifact, and DINOv2-G stays below its null at every $C$ (excess $-0.008$ at $C{=}1000$, $r{=}200$), where the supremum had placed it at the null. % expR60_c_sweep_record.csv, exp19_c_sweep.csv\n")
rep("\\label{app:xi}\n", "\\label{app:xi}\n\\paragraph{$\\xi$ reads norm structure, not curvature.} A parallelogram estimator $\\xi$, intended as a curvature sign, is driven negative by heterogeneous norms alone, so it is reported only as a norm-structure descriptor. % exp12, expR43_xi_norm_references.csv\n")
rep("\\label{app:hierarcaps}\n", "\\label{app:hierarcaps}\nOn HierarCaps \\citep{alper2024hierarcaps} (1000 four-level caption chains), text encoders show the same reading with no class set: the projection radius orders levels within a chain, sibling triplets are resolved at $0.83$--$0.97$ (chance $0.5$), and the leaf excess repeats the text census (Table~\\ref{tab:a8}). % exp16_hierarcaps.csv\n")
open(p, 'w').write(T); open(R + 'phaseC_old_main_body.tex', 'w').write(old_main)
# --- checks ---
main = T[:T.index("\\appendix")]
for w in ["tool", "we believe", "nterestingly"]: assert w not in main, w
labels = set(re.findall(r"\\label\{([^}]*)\}", T))
import glob
for f in glob.glob('iclr2027/iclr2027/appendix_tables/*.tex') + ['iclr2027/iclr2027/tab_census.tex']: labels |= set(re.findall(r"\\label\{([^}]*)\}", open(f).read()))
refs = set(re.findall(r"\\ref\{([^}]*)\}", T)); missing = sorted(refs - labels); print("missing labels:", missing)
assert not missing
print("restructure applied; fills:", json.dumps({k: v for k, v in F.items() if not k.startswith('FIG2B')}))
