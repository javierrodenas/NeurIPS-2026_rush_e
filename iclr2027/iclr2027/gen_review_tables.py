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
                  [f"${float(a['delta']):.3f}$", f"${float(a['excessA']):+.3f}$", f"${float(a['excessB']):+.3f}$", f"${float(a['zB']):+.2f}$"]
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
            cs += ["--","--"] if a is None else [f"${float(a['excessB']):+.3f}$", f"${float(a['zB']):+.2f}$"]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Hierarchy-flattening null with \emph{data-driven} hubs: as the WordNet-hub null, but the hubs are $k$-means clusters of each model's own centroids ($k{=}30$/$20$, restarts fixed), so no human taxonomy enters the construction. All 24 cells remain sign-negative (16/24 at $|z|\ge2$), and DINOv2-L/G on ImageNet, at their spectrum null and marginal under WordNet hubs, are strongly tree-organized among their own clusters ($z=-4.4$/$-4.6$): the taxonomy-dependence concern is addressed empirically.}",
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
      r"\caption{$\xi$ on the angular geometry: centroids L2-normalized, distances the spherical geodesic, spectrum null built on the normalized cloud and re-normalized (20 replicates; ImageNet centroid store). Probes whether the deep negative $\xi$-excess of DINOv2 on ImageNet (Table~\ref{tab:a10}) is a norm-structure effect. Outcome: it is for S/B/L, whose angular $\xi$-excess turns sphere-leaning positive, while DINOv2-G and CLIP-B/L remain negative on the sphere: the Euclidean $\xi$ signal of DINOv2 is largely carried by norm structure.}",
      r"\label{tab:b19-xigeo}",r"\end{table}"]
    (OUT/"tab_b19_xigeo.tex").write_text("\n".join(lines)+"\n"); print(f"b19 written ({len(rows)} models)")

# ---- B20: census re-run with 20 null replicates (expR39) vs the 5-replicate census ----
f = RES/"expR39_census20.csv"
if f.exists():
    r39=list(csv.DictReader(open(f)))
    r20={(a['model'],a['dataset']):a for a in load('exp20_null_ztable.csv')}
    n=len(r39); sign_agree=0; sig_agree=0; comp=0
    below_all=sum(float(a['frac_null_above'])==1.0 for a in r39)
    for a in r39:
        b=r20.get((a['model'],a['dataset']))
        if b is None or int(a.get('store_centroids',0))==1: continue
        comp+=1
        sign_agree += (float(a['excess'])<0)==(float(b['excess'])<0)
        sig_agree  += (float(a['z'])<=-2)==(float(b['z'])<=-2)
    DSH={'imagenet':'IN','cifar100':'C100','cifar10':'C10','dtd':'DTD','fashionmnist':'FMNIST','mnist':'MNIST'}
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    by={(a['model'],a['dataset']):a for a in r39}
    DS=['imagenet','cifar100','cifar10','dtd','fashionmnist','mnist']
    lines=[r"\begin{table}[H]",r"\centering",r"\scriptsize",r"\setlength{\tabcolsep}{2.5pt}",
      r"\begin{tabular}{l"+"cc"*len(DS)+"}",r"\toprule",
      " & "+" & ".join(f"\\multicolumn{{2}}{{c}}{{{DSH[d]}}}" for d in DS)+r" \\",
      r"model & "+" & ".join([r"exc.\ & r"]*len(DS))+r" \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        cs=[]
        for d in DS:
            a=by.get((m,d))
            if a is None: cs+=["--","--"]; continue
            star = r"\rlap{$^\dagger$}" if int(a.get('store_centroids',0))==1 else ""
            cs += [f"${float(a['excess']):+.3f}$".replace("0.", ".")+star, f"{round(20*float(a['frac_null_above']))}"]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{The full vision census re-run with 20 spectrum-null replicates: excess and percentile rank r (number of the 20 null replicates above the real value; 20 = below every replicate). "
      f"Agreement with the 5-replicate census on the {comp} directly comparable cells: sign {sign_agree}/{comp}, significance at $|z|\\ge2$ {sig_agree}/{comp}; {below_all}/{n} cells lie below every replicate. "
      r"$^\dagger$ImageNet rows for the supervised ViTs use the precomputed centroid store (precomputed centroid store, whose supervised-ViT centroids differ slightly from the census cache) and are not directly comparable to Table~\ref{tab:b2-ztable}.}",
      r"\label{tab:b20-census20}",r"\end{table}"]
    (OUT/"tab_b20_census20.tex").write_text("\n".join(lines)+"\n"); print(f"b20 written ({n} cells; agree sign {sign_agree}/{comp}, sig {sig_agree}/{comp})")

# ---- B20: p99.9 census with 20 null replicates (expR39) vs the 5-replicate census ----
f = RES/"expR40_p999census.csv"
if f.exists():
    r39=list(csv.DictReader(open(f)))
    r20={(a['model'],a['dataset']):a for a in load('exp20_null_ztable.csv')}
    n=len(r39); sign_agree=0; sig_agree=0; comp=0
    below_all=sum(float(a['frac_null_above'])==1.0 for a in r39)
    for a in r39:
        b=r20.get((a['model'],a['dataset']))
        if b is None or int(a.get('store_centroids',0))==1: continue
        comp+=1
        sign_agree += (float(a['excess'])<0)==(float(b['excess'])<0)
        sig_agree  += (float(a['z'])<=-2)==(float(b['z'])<=-2)
    DSH={'imagenet':'IN','cifar100':'C100','cifar10':'C10','dtd':'DTD','fashionmnist':'FMNIST','mnist':'MNIST'}
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    by={(a['model'],a['dataset']):a for a in r39}
    DS=['imagenet','cifar100','cifar10','dtd','fashionmnist','mnist']
    lines=[r"\begin{table}[H]",r"\centering",r"\scriptsize",r"\setlength{\tabcolsep}{2.5pt}",
      r"\begin{tabular}{l"+"cc"*len(DS)+"}",r"\toprule",
      " & "+" & ".join(f"\\multicolumn{{2}}{{c}}{{{DSH[d]}}}" for d in DS)+r" \\",
      r"model & "+" & ".join([r"exc.\ & r"]*len(DS))+r" \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        cs=[]
        for d in DS:
            a=by.get((m,d))
            if a is None: cs+=["--","--"]; continue
            star = r"\rlap{$^\dagger$}" if int(a.get('store_centroids',0))==1 else ""
            cs += [f"${float(a['excess']):+.3f}$".replace("0.", ".")+star, f"{round(20*float(a['frac_null_above']))}"]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{The full vision p99.9 census with 20 spectrum-null replicates: excess and percentile rank r (number of the 20 null replicates above the real value; 20 = below every replicate). "
      f"Agreement with the 5-replicate census on the {comp} directly comparable cells: sign {sign_agree}/{comp}, significance at $|z|\\ge2$ {sig_agree}/{comp}; {below_all}/{n} cells lie below every replicate. "
      r"$^\dagger$ImageNet rows for the supervised ViTs use the precomputed centroid store (precomputed centroid store, whose supervised-ViT centroids differ slightly from the census cache) and are not directly comparable to Table~\ref{tab:b2-ztable}. The distributional statistic is stricter on supervised ViTs and stronger on DINO/DINOv2 than the supremum (Table~\ref{tab:b18-p999}).}",
      r"\label{tab:b21-p999census}",r"\end{table}"]
    (OUT/"tab_b21_p999census.tex").write_text("\n".join(lines)+"\n"); print(f"b21 written ({n} cells; agree sign {sign_agree}/{comp}, sig {sig_agree}/{comp})")

# ---- B22: flattening null B under p99.9 (expR41) ----
f = RES/"expR41_flatnull_p999.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    by={(a['model'],a['dataset']):a for a in rows}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\setlength{\tabcolsep}{4pt}",
      r"\begin{tabular}{lcccc|cccc}",r"\toprule",
      r"& \multicolumn{4}{c|}{ImageNet (30 WordNet groups)} & \multicolumn{4}{c}{CIFAR-100 (20 superclasses)} \\",
      r"model & p99.9 & exc.\ A & exc.\ B & $z_B$ & p99.9 & exc.\ A & exc.\ B & $z_B$ \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        cs=[]
        for ds in ("imagenet","cifar100"):
            a=by.get((m,ds))
            cs += ["--","--","--","--"] if a is None else \
                  [f"${float(a['delta']):.3f}$", f"${float(a['excessA']):+.3f}$", f"${float(a['excessB']):+.3f}$", f"${float(a['zB']):+.2f}$"]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{The WordNet-hub null repeated under the p99.9 statistic (20 replicates of each null). Where a supervised model's cluster-preserving excess is significant under the supremum but not here, that excess is carried by a few extreme quadruples; where DINOv2's is significant here, it sits in the bulk of the distribution.}",
      r"\label{tab:b22-flatnull-p999}",r"\end{table}"]
    (OUT/"tab_b22_flatnull_p999.tex").write_text("\n".join(lines)+"\n"); print(f"b22 written ({len(rows)} cells)")

# ---- B23: text census with 20 null replicates (expR44) ----
f = RES/"expR44_text_census20.csv"
if f.exists():
    rows=list(csv.DictReader(open(f))); r18={a['model']:a for a in load('exp18_text_nulls.csv')}
    TN={"gpt2":"GPT-2 S","gpt2_m":"GPT-2 M","gpt2_l":"GPT-2 L","gpt2_xl":"GPT-2 XL","pythia_410m":"Pythia-410M","pythia_1b":"Pythia-1B","pythia_2b8":"Pythia-2.8B","olmo_1b":"OLMo-1B","olmo_7b":"OLMo-7B","bge_base":"BGE-base","bge_large":"BGE-large","gte_base":"GTE-base","gte_large":"GTE-large","gte_qwen2":"GTE-Qwen2-1.5B","e5_base":"E5-base","e5_large":"E5-large"}
    by={a['model']:a for a in rows}; agree=0; comp=0
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\begin{tabular}{lcccc|cc}",r"\toprule",
      r"& \multicolumn{4}{c|}{20 replicates} & \multicolumn{2}{c}{3 replicates (Table~\ref{tab:b1-textnulls})} \\",
      r"model & $\hat\delta$ & excess & $z$ & reps above & excess & $z$ \\",r"\midrule"]
    for m in TN:
        a=by.get(m); b=r18.get(m)
        if a is None and b is None: continue
        if a is None: lines.append(f"{TN[m]} & -- & -- & -- & -- & ${float(b['excess']):+.3f}$ & -- \\\\"); continue
        bz=(float(b['excess'])/max((float(b['null_sd'])**2+float(b['delta_sd'])**2)**0.5,1e-9)) if b else None
        if b: comp+=1; agree += ((float(a['excess'])<0)==(float(b['excess'])<0))
        lines.append(f"{TN[m]} & ${float(a['delta']):.3f}$ & ${float(a['excess']):+.3f}$ & ${float(a['z']):+.1f}$ & {round(20*float(a['frac_null_above']))}/20 & " + (f"${float(b['excess']):+.3f}$ & ${bz:+.1f}$" if b else "-- & --") + r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{The text census re-run with 20 spectrum-null replicates (paper template; OLMo-7B not re-run, it exceeds the local GPU). "
      f"Sign agreement with the 3-replicate census: {agree}/{comp}. "
      r"Large $|z|$ read as ``below every replicate''.}",
      r"\label{tab:b23-text20}",r"\end{table}"]
    (OUT/"tab_b23_text20.tex").write_text("\n".join(lines)+"\n"); print(f"b23 written ({len(rows)} models; sign agree {agree}/{comp})")

# ---- B24: extra backbones (ConvNet SSL, MAE, I-JEPA) on ImageNet (expR45) ----
f = RES/"expR45_convnet_rows.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    XN={"barlow_r50":"Barlow Twins (ResNet-50)","byol_r50":"BYOL (ResNet-50)","mae_b":"MAE-B (ViT)","ijepa_h":"I-JEPA-H (ViT)"}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\begin{tabular}{lcccc}",r"\toprule",
      r"model & $d$ & $\hat\delta$ & excess ($z$) & reps above \\",r"\midrule"]
    for a in rows:
        lines.append(f"{XN.get(a['model'],a['model'])} & {a['d']} & ${float(a['delta']):.3f}$ & ${float(a['excessA']):+.3f}$ (${float(a['zA']):+.1f}$) & {round(20*float(a['rankA']))}/20 \\\\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Backbones outside the ViT census, ImageNet centroid store: two ResNet-50 self-supervised models and two further ViT-SSL recipes, under the spectrum null (20 replicates, percentile rank).}",
      r"\label{tab:b24-extra}",r"\end{table}"]
    (OUT/"tab_b24_extra.tex").write_text("\n".join(lines)+"\n"); print(f"b24 written ({len(rows)} rows)")

# ---- B25: star vs hierarchy calibration of the nulls (expR42 v2: Gaussian and Haar constructions) ----
f = RES/"expR42_star_calibration.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    CN={"star30_tight":"30-cluster star, tight (0.1)","star30_mid":"30-cluster star, mid (0.3)","star30_loose":"30-cluster star, loose (0.6)","hier6x5_mid":"2-level hierarchy $6\\times5$ (0.3)"}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\setlength{\tabcolsep}{4pt}",r"\begin{tabular}{lc|cc|cc}",r"\toprule",
      r"& & \multicolumn{2}{c|}{spectrum null A} & \multicolumn{2}{c}{hub-randomizing null B} \\",
      r"synthetic cloud ($n{=}1000$, $d{=}768$) & $\hat\delta$ & Gaussian & Haar & Gaussian & Haar \\",r"\midrule"]
    for cfg in CN:
        sub=[a for a in rows if a['config']==cfg]
        if not sub: continue
        m=lambda k: st.mean(float(a[k]) for a in sub)
        lines.append(f"{CN[cfg]} & ${m('delta'):.3f}$ & ${m('excessA_gauss'):+.3f}$ & ${m('excessA_haar'):+.3f}$ & ${m('excessB_gauss'):+.3f}$ & ${m('excessB_haar'):+.3f}$ \\\\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Calibration of the nulls on synthetic clouds (excess, means over seeds; within/between-cluster noise ratio in parentheses). A pure star (Gaussian clusters around Gaussian centers, no hierarchy) sits far below the spectrum null A under either construction (Gaussian coefficients, the paper's; Haar-rotated coefficients, exact sample spectrum): null A certifies clustering, not depth. The hub-randomizing null B, which keeps every cluster intact and resamples only the hub configuration, also reports the star as ``hierarchical'': a near-regular simplex of hubs already minimizes $\delta$, so any hub randomization raises it. Null B is therefore not a valid depth test and is not used in the paper.}",
      r"\label{tab:b25-star}",r"\end{table}"]
    (OUT/"tab_b25_star.tex").write_text("\n".join(lines)+"\n"); print(f"b25 written ({len(rows)} rows)")

# ---- B26: xi on norm-heterogeneous references (expR43) ----
f = RES/"expR43_xi_norm_references.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    RN={"gaussian":"iid Gaussian","gauss_lognorm_r0.3":"Gaussian, lognormal radii ($\\sigma{=}0.3$)","gauss_lognorm_r0.6":"Gaussian, lognormal radii ($\\sigma{=}0.6$)","gauss_lognorm_r1.0":"Gaussian, lognormal radii ($\\sigma{=}1.0$)","core_shell":"core + shell mixture","sphere":"uniform sphere","sphere_radial_jitter":"sphere with radial jitter"}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\begin{tabular}{lcc}",r"\toprule",
      r"reference (no curvature unless stated) & $\xi$ & frac.\ negative \\",r"\midrule"]
    for k in RN:
        sub=[a for a in rows if a['reference']==k]
        if not sub: continue
        lines.append(f"{RN[k]} & ${st.mean(float(a['xi']) for a in sub):+.4f}$ & ${st.mean(float(a['frac_neg']) for a in sub):.2f}$ \\\\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{$\xi$ on references with heterogeneous norms but no curvature ($n{=}1000$, $d{=}768$, 5 seeds). Norm dispersion alone drives $\xi$ negative (lognormal radii, radial jitter on a sphere, core+shell): the estimator reads norm structure, not curvature, and the paper reports it only as such.}",
      r"\label{tab:b26-xinorm}",r"\end{table}"]
    (OUT/"tab_b26_xinorm.tex").write_text("\n".join(lines)+"\n"); print(f"b26 written ({len(rows)} rows)")

# ---- B27: spectrum-null construction sensitivity on real cells (expR46) ----
f = RES/"expR46_null_variants.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    DS=[d for d in ["imagenet","cifar100","cifar10"] if any(a['dataset']==d for a in rows)]; DSH={"imagenet":"IN","cifar100":"C100","cifar10":"C10"}
    by={(a['model'],a['dataset']):a for a in rows}
    lines=[r"\begin{table}[H]",r"\centering",r"\scriptsize",r"\setlength{\tabcolsep}{3pt}",r"\begin{tabular}{l"+"|ccc"*len(DS)+"}",r"\toprule",
      " & "+" & ".join(f"\\multicolumn{{3}}{{c}}{{{DSH[d]}}}" for d in DS)+r" \\",
      "model & "+" & ".join(["Gauss. & Haar & PC-perm."]*len(DS))+r" \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        cs=[]
        for d in DS:
            a=by.get((m,d))
            cs += ["--","--","--"] if a is None else [f"${float(a[k]):+.3f}$".replace("0.",".") for k in ("excess_gauss","excess_haar","excess_pcperm")]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Excess under three constructions of the spectrum-matched null (10 replicates each): Gaussian coefficients (the paper's), Haar-rotated coefficients (exact sample spectrum) and PC-permutation (exact per-component marginals). The Gaussian and PC-permutation constructions agree; the exact-sample-spectrum construction yields smaller excesses for low-effective-rank clouds on the small-$C$ sets (most visibly DINOv2 on CIFAR-100), so the size of those excesses is partly a property of the null hypothesis, while signs and the ImageNet readings are stable.}",
      r"\label{tab:b27-nullvariants}",r"\end{table}"]
    (OUT/"tab_b27_nullvariants.tex").write_text("\n".join(lines)+"\n"); print(f"b27 written ({len(rows)} cells)")
