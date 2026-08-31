#!/usr/bin/env python3
"""Appendix tables for the family-identity control (mock-review W2) and, once the
expR results exist, the flat-hierarchy null (W1), template nulls (W3), and
estimator-robustness checks (W6/Q4). Reads only released result files."""
import csv, os, statistics as st
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
RES = Path(os.environ.get("PLATONIC_RESULTS", HERE.parents[1] / "rebuttal/results"))
OUT = HERE/"appendix_tables"
def load(f): return list(csv.DictReader(open(RES/f)))
def r(x,y): return float(np.corrcoef(x,y)[0,1])
fam=lambda m:'ViT' if m.startswith('i21k') else ('DINOv2' if m.startswith('dinov2') else 'CLIP')
def demean(vals,fams):
    mu={f:st.mean(v for v,g in zip(vals,fams) if g==f) for f in set(fams)}
    return [v-mu[g] for v,g in zip(vals,fams)]
d20={(a['model'],a['dataset']):a for a in load('exp20_null_ztable.csv')}
t1={(a['model'],a['dataset']):a for a in load('table1_regenerated.csv')}
e2={(a['model'],a['dataset']):a for a in load('exp2_metric_controls.csv')}
models=sorted({m for m,_ in t1}); H=['imagenet','cifar100','cifar10','dtd']
DSH={'imagenet':'ImageNet','cifar100':'CIFAR-100','cifar10':'CIFAR-10','dtd':'DTD'}
def bestadv(m,ds):
    a=e2[(m,ds)]; return 100*(max(float(a['FS_H']),float(a['FS_COS']))-float(a['FS_R']))
lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\begin{tabular}{llccccc}",r"\toprule",
 r"quantity & dataset & pooled $r$ & demeaned $r$ & ViT & DINOv2 & CLIP \\",r"\midrule"]
for name,col in [("NC H$-$R",lambda m,ds: float(t1[(m,ds)]['NC_adv'])),
                 ("FS H$-$R",lambda m,ds: float(t1[(m,ds)]['FS_adv'])),
                 ("FS best$-$R",bestadv)]:
    for ds in H:
        xs=[float(d20[(m,ds)]['delta']) for m in models]; ys=[col(m,ds) for m in models]; fs=[fam(m) for m in models]
        wf={f:r([x for x,g in zip(xs,fs) if g==f],[y for y,g in zip(ys,fs) if g==f]) for f in ['ViT','DINOv2','CLIP']}
        lines.append(f"{name if ds=='imagenet' else ''} & {DSH[ds]} & ${r(xs,ys):+.2f}$ & ${r(demean(xs,fs),demean(ys,fs)):+.2f}$ & ${wf['ViT']:+.2f}$ & ${wf['DINOv2']:+.2f}$ & ({wf['CLIP']:+.0f}) \\\\")
    lines.append(r"\midrule")
lines[-1]=r"\bottomrule"
lines+=[r"\end{tabular}",
 r"\caption{Family-identity control for the $\delta_{\text{norm}}$--gain correlations (within dataset, 10 backbones = 3 families with nested scales). Demeaned: both variables centered within family before correlating; ViT/DINOv2: within-family correlations (4 models each); CLIP has only two models, so its within-family value (parenthesized sign) is not informative. The within-dataset prediction survives the family control; pooled cross-dataset correlations of the best-metric advantage do not ($-0.31\to-0.18$ over all 60 cells, $-0.33\to-0.04$ on hierarchical cells) and are therefore family-driven.}",
 r"\label{tab:b12-familycorr}",r"\end{table}"]
(OUT/"tab_b12_familycorr.tex").write_text("\n".join(lines)+"\n"); print("b12 written")

# ---- B13: flat-hierarchy null (expR29) ----
f = RES/"expR29_flatnull.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    by={(a['model'],a['dataset']):a for a in rows}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\setlength{\tabcolsep}{4pt}",
      r"\begin{tabular}{lcccc|cccc}",r"\toprule",
      r"& \multicolumn{4}{c|}{ImageNet (30 WordNet groups)} & \multicolumn{4}{c}{CIFAR-100 (20 superclasses)} \\",
      r"model & $\hat\delta$ & exc.\ A & exc.\ B & $z_B$ & $\hat\delta$ & exc.\ A & exc.\ B & $z_B$ \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        cs=[]
        for ds in ("imagenet","cifar100"):
            a=by.get((m,ds))
            cs += ["--","--","--","--"] if a is None else \
                  [f"${float(a['delta']):.3f}$", f"${float(a['excessA']):+.3f}$", f"${float(a['excessB']):+.3f}$", f"${float(a['zB']):+.1f}$"]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Hierarchy-flattening null (null B): each centroid keeps its offset to its superclass hub, while the hub configuration is replaced by a spectrum-matched Gaussian sample of the hubs (5 replicates), so cluster structure is preserved and only the organization \emph{among} clusters is destroyed; null A is the spectrum null of \S\ref{sec:instrument}. Negative excess B ($z_B$ against combined null and estimator noise) is evidence of tree organization among superclass hubs beyond clustering. Three scope notes: (i) the hubs are defined by the human taxonomy, so this control certifies hierarchy \emph{aligned with WordNet / the CIFAR superclasses} and would not detect model-specific hierarchy unaligned with it; (ii) the ImageNet rows are computed on the precomputed centroid store, whose supervised-ViT centroids differ slightly from the census cache ($\hat\delta$ up to $+0.025$ vs Table~\ref{tab:b2-ztable}); every quantity in a row uses the same centroids, so the within-row verdicts are unaffected; (iii) at a $4\times$ quadruple budget ($2{\times}10^6$) the ImageNet verdicts persist (ViT-L $z_B{=}-5.1$, CLIP-B $-2.4$, DINOv2-L $-2.0$).}",
      r"\label{tab:b13-flatnull}",r"\end{table}"]
    (OUT/"tab_b13_flatnull.tex").write_text("\n".join(lines)+"\n"); print(f"b13 written ({len(rows)} cells)")

# ---- B14: template-conditioned nulls for GPT-2 (expR30) ----
f = RES/"expR30_template_nulls.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    T=["photo","name_only","image","closeup","this_is","wild","def_pair","gloss_only","concept","discussion"]
    M=["gpt2","gpt2_m","gpt2_l","gpt2_xl"]; MN={"gpt2":"GPT-2 S","gpt2_m":"GPT-2 M","gpt2_l":"GPT-2 L","gpt2_xl":"GPT-2 XL"}
    by={(a['model'],a['template']):a for a in rows}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\setlength{\tabcolsep}{4pt}",
      r"\begin{tabular}{l"+"c"*len(M)+"}",r"\toprule",
      r"template & "+" & ".join(MN[m] for m in M)+r" \\",r"\midrule"]
    for t in T:
        cs=[]
        for m in M:
            a=by.get((m,t))
            cs.append("--" if a is None else f"${float(a['excess']):+.3f}$"+(r"\rlap{$^*$}" if abs(float(a['z']))>=2 else ""))
        lines.append(t.replace("_",r"\_")+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Template-conditioned excess over the spectrum-matched null for the four GPT-2 sizes (10 templates; $^*$: $|z|\ge2$ against combined null and estimator noise). The scale trend of \S\ref{sec:form} is assessed on every template, not only on the paper's baseline.}",
      r"\label{tab:b14-templates}",r"\end{table}"]
    (OUT/"tab_b14_templates.tex").write_text("\n".join(lines)+"\n"); print(f"b14 written ({len(rows)} rows)")

# ---- B15: estimator robustness (expR31 + expR32) ----
f1, f2 = RES/"expR31_quad_sweep.csv", RES/"expR32_centroid_bootstrap.csv"
if f1.exists() and f2.exists():
    r31=list(csv.DictReader(open(f1))); r32=list(csv.DictReader(open(f2)))
    NAME={"i21k_l":"ViT-L","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B"}
    DS={"imagenet":"IN","cifar100":"C100","dtd":"DTD"}
    lines=[r"\begin{table}[H]",r"\centering",r"\scriptsize",r"\setlength{\tabcolsep}{3pt}",
      r"\begin{tabular}{llcccccc|c}",r"\toprule",
      r"model & data & \multicolumn{6}{c|}{excess vs quadruple budget ($10^4$..$2{\times}10^6$)} & p99.9 @$5{\times}10^5$ \\",r"\midrule"]
    for (m,ds) in [("i21k_l","imagenet"),("dinov2_l","imagenet"),("clip_b","imagenet"),
                   ("i21k_l","cifar100"),("dinov2_l","cifar100"),("clip_b","cifar100"),
                   ("i21k_l","dtd"),("dinov2_l","dtd"),("clip_b","dtd")]:
        mx=[a for a in r31 if a['model']==m and a['dataset']==ds and a['stat']=='max']
        p9=[a for a in r31 if a['model']==m and a['dataset']==ds and a['stat']=='p999' and a['n_quads']=='500000']
        if not mx: continue
        vals=" & ".join(f"${float(a['excess']):+.3f}$" for a in sorted(mx,key=lambda a:int(a['n_quads'])))
        pv = f"${float(p9[0]['excess']):+.3f}$" if p9 else "--"
        lines.append(f"{NAME[m]} & {DS[ds]} & {vals} & {pv} \\\\")
    lines+=[r"\midrule", r"\multicolumn{9}{l}{Centroid bootstrap (30 resamples of the per-class images): s.d.\ of the excess} \\"]
    seen=set()
    for a in r32:
        k=(a['model'],a['dataset'])
        if k in seen or int(a['b'])<0: continue
        sub=[float(x['excess']) for x in r32 if (x['model'],x['dataset'])==k and int(x['b'])>=0]
        if len(sub)>=10:
            seen.add(k)
            lines.append(f"{NAME[a['model']]} & {DS[a['dataset']]} & \\multicolumn{{6}}{{l}}{{excess ${st.mean(sub):+.4f}$, bootstrap s.d.\\ ${st.pstdev(sub):.4f}$ ({len(sub)} resamples, {a['n_per_class']} images/class)}} & \\\\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Estimator robustness. Top: on CIFAR-100 and DTD the excess is stable once the quadruple budget reaches $10^5$ (the paper uses $5{\times}10^5$); on ImageNet ($C{=}1000$) every sign holds at every budget but magnitudes drift by up to $0.02$ (DINOv2-L toward its null, ViT-L and CLIP-B away from it), so ImageNet excess magnitudes carry a budget caveat. The 99.9th-percentile variant of the four-point statistic, robust to the supremum, is negative wherever the supremum excess is. Bottom: resampling the per-class images that form each centroid (reported on CIFAR-100 and DTD) moves the excess by an order of magnitude less than its size.}",
      r"\label{tab:b15-robust}",r"\end{table}"]
    (OUT/"tab_b15_robust.tex").write_text("\n".join(lines)+"\n"); print("b15 written")

# ---- B16: data-driven (k-means) hub null (expR36) ----
f = RES/"expR36_kmeans_hubs.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    by={(a['model'],a['dataset']):a for a in rows}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\setlength{\tabcolsep}{5pt}",
      r"\begin{tabular}{lcc|cc}",r"\toprule",
      r"& \multicolumn{2}{c|}{ImageNet ($k{=}30$)} & \multicolumn{2}{c}{CIFAR-100 ($k{=}20$)} \\",
      r"model & exc.\ B$_{\text{km}}$ & $z$ & exc.\ B$_{\text{km}}$ & $z$ \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        cs=[]
        for ds in ("imagenet","cifar100"):
            a=by.get((m,ds))
            cs += ["--","--"] if a is None else [f"${float(a['excessB']):+.3f}$", f"${float(a['zB']):+.1f}$"]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Hierarchy-flattening null with \emph{data-driven} hubs: as Table~\ref{tab:b13-flatnull}, but the hubs are $k$-means clusters of each model's own centroids ($k{=}30$/$20$, restarts fixed), so no human taxonomy enters the construction. All 24 cells remain sign-negative (16/24 at $|z|\ge2$), and DINOv2-L/G on ImageNet, at their spectrum null and marginal under WordNet hubs, are strongly tree-organized among their own clusters ($z=-4.4$/$-4.6$): the scope note of Table~\ref{tab:b13-flatnull} is addressed empirically.}",
      r"\label{tab:b16-kmhubs}",r"\end{table}"]
    (OUT/"tab_b16_kmhubs.tex").write_text("\n".join(lines)+"\n"); print(f"b16 written ({len(rows)} cells)")

# ---- B17: sample-level census (expR37) ----
f = RES/"expR37_sample_level.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    by={(a['model'],a['dataset']):a for a in rows}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\setlength{\tabcolsep}{5pt}",
      r"\begin{tabular}{lccc|ccc}",r"\toprule",
      r"& \multicolumn{3}{c|}{CIFAR-100 (10/class)} & \multicolumn{3}{c}{DTD (22/class)} \\",
      r"model & raw $\hat\delta$ & excess & $z$ & raw $\hat\delta$ & excess & $z$ \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        cs=[]
        for ds in ("cifar100","dtd"):
            a=by.get((m,ds))
            cs += ["--","--","--"] if a is None else [f"${float(a['delta']):.3f}$", f"${float(a['excess']):+.4f}$", f"${float(a['z']):+.1f}$"]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{The instrument transplanted to sample-level features, the object of prior latent-hyperbolicity readings ($\approx$1000 stratified training images per cell, spectrum-matched nulls). Raw sample-level $\hat\delta$ sits in the same low band those works report, but the excess is mostly within null noise (6/24 cells at $|z|\ge2$; DINOv2 on DTD is sign-\emph{positive}): raw sample-level readings are confounded exactly as raw centroid readings are, and the genuine tree excess of the census is a statement about \emph{inter-class} geometry.}",
      r"\label{tab:b17-samplelevel}",r"\end{table}"]
    (OUT/"tab_b17_samplelevel.tex").write_text("\n".join(lines)+"\n"); print(f"b17 written ({len(rows)} cells)")

# ---- B18: p99.9 ImageNet column with 20 null replicates + percentile ranks (expR34) ----
f = RES/"expR34_p999_imagenet.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    by={a['model']:a for a in rows}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\begin{tabular}{lcccc}",r"\toprule",
      r"model & p99.9 $\hat\delta$ & excess & $z$ & null reps above real \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        a=by.get(m)
        if a is None: lines.append(NAME[m]+r" & -- & -- & -- & -- \\"); continue
        lines.append(f"{NAME[m]} & ${float(a['p999']):.3f}$ & ${float(a['excess']):+.4f}$ & ${float(a['z']):+.1f}$ & {round(20*float(a['frac_null_above']))}/20 \\\\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{The ImageNet census under the supremum-robust statistic: 99.9th percentile of the four-point defect, 20 spectrum-null replicates, with the percentile rank (how many replicates exceed the real value; 20/20 = below every replicate). The ordering shifts relative to the supremum: DINO/DINOv2 carry the most budget-robust excess, while ViT-T's supremum excess does not survive ($+0.007$, above 19/20 replicates), consistent with the curvature-sign ordering of Table~\ref{tab:a10}. Large $|z|$ should be read as ``below every replicate'', not as a Gaussian tail probability.}",
      r"\label{tab:b18-p999}",r"\end{table}"]
    (OUT/"tab_b18_p999.tex").write_text("\n".join(lines)+"\n"); print(f"b18 written ({len(rows)} models)")

# ---- B19: xi on the angular geometry (expR38) ----
f = RES/"expR38_xi_angular.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    by={a['model']:a for a in rows}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\begin{tabular}{lcccc}",r"\toprule",
      r"model & $\xi_{\text{geo}}$ & frac.\ neg.\ & excess & $z$ \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        a=by.get(m)
        if a is None: lines.append(NAME[m]+r" & -- & -- & -- & -- \\"); continue
        lines.append(f"{NAME[m]} & ${float(a['xi_geo']):+.3f}$ & ${float(a['frac_neg']):.2f}$ & ${float(a['excess']):+.4f}$ & ${float(a['z']):+.1f}$ \\\\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{$\xi$ on the angular geometry: centroids L2-normalized, distances the spherical geodesic, spectrum null built on the normalized cloud and re-normalized (20 replicates; ImageNet centroid store). Probes whether the deep negative $\xi$-excess of DINOv2 on ImageNet (Table~\ref{tab:a10}) is a norm-structure effect: if it persists here, the tree signal lives in the angular component as well.}",
      r"\label{tab:b19-xigeo}",r"\end{table}"]
    (OUT/"tab_b19_xigeo.tex").write_text("\n".join(lines)+"\n"); print(f"b19 written ({len(rows)} models)")
