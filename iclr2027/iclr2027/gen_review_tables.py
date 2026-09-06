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
 r"\caption{Family-identity control for the $\delta_{\text{norm}}$--gain correlations (within dataset, 10 backbones = 3 families with nested scales). Demeaned: both variables centered within family before correlating; ViT/DINOv2: within-family correlations (4 models each); CLIP has only two models, so its within-family value (parenthesized sign) is not informative. Without DINOv2 (six backbones) the NC correlation holds on ImageNet and CIFAR-100 ($-0.87$/$-0.65$) but not on CIFAR-10/DTD ($+0.23$/$-0.34$). The within-dataset prediction survives the family control; pooled cross-dataset correlations of the best-metric advantage do not ($-0.31\to-0.18$ over all 60 cells, $-0.33\to-0.04$ on hierarchical cells) and are therefore family-driven.}",
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

# ---- B14: template-conditioned nulls for GPT-2 (expR49 padding-free, else expR30 batched) ----
f = RES/"expR49_template_nulls_bs1.csv"; b14_pf = f.exists() and len(list(csv.DictReader(open(f))))>=40
if not b14_pf: f = RES/"expR30_template_nulls.csv"
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
      r"\caption{Template-conditioned excess over the spectrum-matched null for the four GPT-2 sizes (10 templates; $^*$: $|z|\ge2$ against combined null and estimator noise). The scale trend of \S\ref{sec:form} is assessed on every template, not only on the paper's baseline. "+("Padding-free extraction (one prompt at a time, the canonical protocol of Table~\\ref{tab:b23-text200}). Mean excess over templates: $-0.012$ (S), $-0.023$ (M), $-0.040$ (L), $-0.041$ (XL); cells at $|z|\\ge2$: 5, 8, 9 and 10 of 10. Only 3 null replicates per cell here, so the $z$ are coarser than the census's 20-replicate values: on the baseline template the 200-replicate census (Table~\\ref{tab:b23-text200}) finds S and M not genuine (excess $-0.013$/$-0.009$, $p=0.09$/$0.14$), so their starred cells are borderline, whereas the L/XL cells ($z$ from $-4$ to $-9$) are not in doubt. Template sensitivity of raw $\\hat\\delta$: range up to $0.065$ for causal LMs and $\\le0.012$ for embedders (prompt table in Appendix~\\ref{tab:a7})." if b14_pf else "Batched (batch 16, left-padded) extraction: padding raises GPT-2's $\\hat\\delta$ (Table~\\ref{tab:b28-extraction}), so the L/XL verdicts are conservative while S's sign-positive cells are not robust.")+"}",
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
      r"\caption{The ImageNet census under the supremum-robust statistic: 99.9th percentile of the four-point defect, 20 spectrum-null replicates, with the percentile rank (how many replicates exceed the real value; 20/20 = below every replicate). The ordering shifts relative to the supremum: DINO/DINOv2 carry the most budget-robust excess, while ViT-T's supremum excess does not survive ($+0.007$, above 19/20 replicates), consistent with the ordering of the norm-structure descriptor $\xi$ (Table~\ref{tab:a10}). Large $|z|$ should be read as ``below every replicate'', not as a Gaussian tail probability.}",
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

# ---- B20: census of record (R6: 200 null replicates; falls back to the 20-replicate expR39b) ----
f = RES/"expR52_census_haar_p999_200.csv"                 # census of record (Phase B): Haar x p99.9, BH
if not f.exists(): f = RES/"expR39c_census200_cache.csv"
if f.exists():
    r39=list(csv.DictReader(open(f)))
    N20 = 200; RMIN20 = 191
    rk = lambda a: int(a["r_above"]) if "r_above" in a else round(20*float(a["frac_null_above"]))
    pl = lambda a: float(a["p_left"]) if "p_left" in a else (1 + 20 - rk(a)) / 21
    r20={(a['model'],a['dataset']):a for a in load('exp20_null_ztable.csv')}
    n=len(r39); sign_agree=0; comp=0
    for a in r39:
        b=r20.get((a['model'],a['dataset']))
        if b is None: continue
        comp+=1; sign_agree += (float(a['excess'])<0)==(float(b['excess'])<0)
    HIERSET={'imagenet','cifar100','cifar10','dtd'}
    gbh = lambda a: str(a.get("genuine_bh", str(pl(a)<=0.05)))=="True"
    gen=sum(gbh(a) for a in r39); genh=sum(gbh(a) for a in r39 if a['dataset'] in {'imagenet','cifar100'}); nh=sum(1 for a in r39 if a['dataset'] in {'imagenet','cifar100'})
    nz2=sum(abs(float(a['z']))>=2 for a in r39); nz3h=sum(float(a['z'])<=-3 for a in r39 if a['dataset'] in HIERSET)
    DSH={'imagenet':'IN','cifar100':'C100','cifar10':'C10','dtd':'DTD','fashionmnist':'FMNIST','mnist':'MNIST'}
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    by={(a['model'],a['dataset']):a for a in r39}
    DS=['imagenet','cifar100','cifar10','dtd','fashionmnist','mnist']
    lines=[r"\begin{table}[H]",r"\centering",r"\scriptsize",r"\setlength{\tabcolsep}{1.4pt}",
      r"\begin{tabular}{l"+"ccc"*len(DS)+"}",r"\toprule",
      " & "+" & ".join(f"\\multicolumn{{3}}{{c}}{{{DSH[d]}}}" for d in DS)+r" \\",
      r"model & "+" & ".join([r"exc.\ & $r$ & $p$"]*len(DS))+r" \\",r"\midrule"]
    def pfmt(p): return f"{p:.3f}".lstrip("0") if p < 1 else "1"
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        cs=[]
        for d in DS:
            a=by.get((m,d))
            if a is None: cs+=["--","--","--"]; continue
            cs += [f"${float(a['excess']):+.3f}" + ("" if gbh(a) else r"^{\circ}") + "$", f"{rk(a)}", pfmt(pl(a))]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{\textbf{The census of record per cell: Haar-rotated spectrum-matched null, 99.9th-percentile statistic, 200 replicates.} Excess over the null mean; "
      f"$r$ = replicates above the real value; $p=(1+\\#\\{{\\text{{null}}\\le\\text{{real}}\\}})/201$; genuine = Benjamini--Hochberg-corrected $p\\le0.05$ over the 72 cells ($^{{\\circ}}$: not genuine). "
      f"Genuine cells: {gen}/{n} overall, {genh}/{nh} on ImageNet+CIFAR-100. Sign agreement with the original 5-replicate Gaussian$\\times$supremum census on the {comp} comparable cells: {sign_agree}/{comp}. "
      r"Every cell on the census cache (one centroid source). The four null$\times$statistic verdicts per cell are in Table~\ref{tab:b21-2x2}.}",
      r"\label{tab:b20-census200}",r"\end{table}"]
    (OUT/"tab_b20_census20.tex").write_text("\n".join(lines)+"\n"); print(f"b20 written ({n} cells, N={N20}; genuine {gen}/{n})")

# ---- B21: the 2x2 {null} x {statistic} verdicts per cell (Phase B) ----
_four = [("Hp", "expR52_census_haar_p999_200.csv"), ("Hs", "expR54_census_haar_sup_200.csv"), ("Gp", "expR40b_p999census200.csv"), ("Gs", "expR39c_census200_cache.csv")]
if all((RES/f).exists() for _, f in _four):
    import numpy as _np
    def _bh(p):
        p = _np.asarray(p, dtype=float); n = len(p); order = _np.argsort(p); ranked = p[order]*n/_np.arange(1, n+1)
        adj = _np.minimum.accumulate(ranked[::-1])[::-1]; out = _np.empty(n); out[order] = _np.minimum(adj, 1.0); return out
    V = {}
    for tag, f in _four:
        rows_ = list(csv.DictReader(open(RES/f))); pb = _bh([float(a["p_left"]) for a in rows_])
        V[tag] = {(a["model"], a["dataset"]): (float(a["excess"]), pb[i] <= 0.05) for i, a in enumerate(rows_)}
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B",
          "dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G",
          "clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    DSH={'imagenet':'IN','cifar100':'C100','cifar10':'C10','dtd':'DTD','fashionmnist':'FMNIST','mnist':'MNIST'}
    DS=['imagenet','cifar100','cifar10','dtd','fashionmnist','mnist']
    counts = {tag: sum(v[1] for v in V[tag].values()) for tag, _ in _four}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\setlength{\tabcolsep}{5pt}",
      r"\begin{tabular}{l"+"c"*len(DS)+"}",r"\toprule",
      r"model & "+" & ".join(DSH[d] for d in DS)+r" \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        lines.append(NAME[m]+" & "+" & ".join("".join(("$\\bullet$" if V[tag][(m,d)][1] else "$\\circ$") for tag, _ in _four) for d in DS)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{\textbf{Verdicts under all four null$\times$statistic constructions, per cell.} Each cell shows four symbols in the order "
      r"Haar$\times$p99.9 (the record), Haar$\times$supremum, Gaussian$\times$p99.9, Gaussian$\times$supremum; $\bullet$ = genuine (BH-corrected $p\le0.05$ over the 72 cells of that construction), $\circ$ = not. "
      f"Genuine counts: {counts['Hp']}/72, {counts['Hs']}/72, {counts['Gp']}/72, {counts['Gs']}/72. "
      r"The Haar construction reproduces the sample spectrum exactly; the Gaussian one does not at small $n$ and inflates excesses for low-rank clouds (Table~\ref{tab:b27-nullvariants}). "
      r"The supremum is decided by a few extreme quadruples and disagrees with the 99.9th percentile in both directions on DINOv2-S/B/G ImageNet (genuine under both p99.9 constructions, above every replicate under both supremum constructions); the 99.9th percentile is decided by the bulk. % expR52, expR54, expR40b, expR39c" + "\n}",
      r"\label{tab:b21-2x2}",r"\end{table}"]
    (OUT/"tab_b21_2x2.tex").write_text("\n".join(lines)+"\n"); print("b21 (2x2 verdicts) written", counts)

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

# ---- B23: text census of record (R6: padding-free, 200 replicates; falls back to expR48 at 20) ----
f = RES/"expR53_text_haar_p999_200.csv"                     # text census of record (Phase B)
if not f.exists(): f = RES/"expR48b_text_census200_bs1.csv"
import pandas as _pd
rows = list(csv.DictReader(open(f))) if f.exists() and len(_pd.read_csv(f)) >= 15 else None
if rows is not None:
    r18={a['model']:a for a in load('exp18_text_nulls.csv')}
    N23 = 200; RMIN23 = 191
    _tfour = [("Hp","expR53_text_haar_p999_200.csv"),("Hs","expR53_text_haar_sup_200.csv"),("Gp","expR53_text_gauss_p999_200.csv"),("Gs","expR48b_text_census200_bs1.csv")]
    import numpy as _np2
    def _bh2(p):
        p = _np2.asarray(p, dtype=float); n = len(p); order = _np2.argsort(p); ranked = p[order]*n/_np2.arange(1, n+1)
        adj = _np2.minimum.accumulate(ranked[::-1])[::-1]; out = _np2.empty(n); out[order] = _np2.minimum(adj, 1.0); return out
    _TV = {}
    for _tag, _tf in _tfour:
        if (RES/_tf).exists():
            _tr = list(csv.DictReader(open(RES/_tf))); _pb = _bh2([float(a["p_left"]) for a in _tr])
            _TV[_tag] = {a["model"]: _pb[i] <= 0.05 for i, a in enumerate(_tr)}
    def tcode(m): return "".join(("$\\bullet$" if _TV[t].get(m, False) else "$\\circ$") for t, _ in _tfour if t in _TV)
    rk = lambda a: int(a["r_above"]) if "r_above" in a else round(20*float(a["frac_null_above"]))
    pl = lambda a: float(a["p_left"]) if "p_left" in a else (1 + 20 - rk(a)) / 21
    def pfmt(p): return f"{p:.3f}".lstrip("0") if p < 1 else "1"
    TN={"gpt2":"GPT-2 S","gpt2_m":"GPT-2 M","gpt2_l":"GPT-2 L","gpt2_xl":"GPT-2 XL","pythia_410m":"Pythia-410M","pythia_1b":"Pythia-1B","pythia_2b8":"Pythia-2.8B","olmo_1b":"OLMo-1B","olmo_7b":"OLMo-7B","bge_base":"BGE-base","bge_large":"BGE-large","gte_base":"GTE-base","gte_large":"GTE-large","gte_qwen2":"GTE-Qwen2-1.5B","e5_base":"E5-base","e5_large":"E5-large"}
    by={a['model']:a for a in rows}; agree=0; comp=0
    lines=[r"\begin{table}[H]",r"\centering",r"\scriptsize",r"\setlength{\tabcolsep}{2.5pt}",r"\begin{tabular}{lccccc|cc}",r"\toprule",
      r"& \multicolumn{5}{c|}{padding-free extraction, record (Haar$\times$p99.9, 200 replicates)} & \multicolumn{2}{c}{original extraction (Table~\ref{tab:b1-textnulls})} \\",
      r"model & $\hat\delta_{99.9}$ & excess & $r$/200 & $p$ & 2$\times$2 & $\hat\delta$ & excess \\",r"\midrule"]
    for m in TN:
        a=by.get(m); b=r18.get(m)
        if a is None and b is None: continue
        if a is None: lines.append(f"{TN[m]} & -- & -- & -- & -- & -- & ${float(b['delta']):.3f}$ & ${float(b['excess']):+.3f}$ \\\\"); continue
        if b: comp+=1; agree += ((float(a['excess'])<0)==(float(b['excess'])<0))
        lines.append(f"{TN[m]} & ${float(a['delta']):.3f}$ & ${float(a['excess']):+.3f}$ & {rk(a)} & {pfmt(pl(a))} & {tcode(m)} & " + (f"${float(b['delta']):.3f}$ & ${float(b['excess']):+.3f}$" if b else "-- & --") + r" \\")
    q=by.get("gte_qwen2")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{\textbf{The text census of record: padding-free extraction, Haar null, 99.9th-percentile statistic, 200 replicates.} "
      f"$r$ = replicates above the real value, $p$ the left-tail add-one $p$-value; genuine = BH-corrected $p\\le0.05$ over the 15 models. The 2$\\times$2 column gives the verdicts under Haar$\\times$p99.9, Haar$\\times$supremum, Gaussian$\\times$p99.9, Gaussian$\\times$supremum ($\\bullet$ genuine). OLMo-7B not re-run (exceeds the local GPU). "
      r"The original extraction (left-padded batches, supremum statistic) is kept for comparison and is not directly comparable. "
      f"Sign agreement with the 3-replicate original: {agree}/{comp}. "
      r"Three verdicts change between extractions, each with a known cause: GPT-2 S (above null in the left-padded original; Table~\ref{tab:b28-extraction}), OLMo-1B (the original used a different checkpoint revision), and GTE-Qwen2 (genuine in the original with 3 replicates, $-0.019$; "
      + (f"here excess ${float(q['excess']):+.3f}$, $r={rk(q)}$, $p={pfmt(pl(q))}$" if q else "at null here") + r"; not used as evidence).}",
      r"\label{tab:b23-text200}",r"\end{table}"]
    (OUT/"tab_b23_text20.tex").write_text("\n".join(lines)+"\n"); print(f"b23 written ({len(rows)} models, N={N23}; sign agree {agree}/{comp})")

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
      r"\caption{Backbones outside the ViT census, ImageNet centroid store: two ResNet-50 self-supervised models, under the spectrum null (20 replicates, percentile rank).}",
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
      r"\caption{Calibration of the nulls on synthetic clouds (excess, means over seeds; within/between-cluster noise ratio in parentheses). A pure star (Gaussian clusters around Gaussian centers, no hierarchy) sits far below the spectrum null A under either construction (Gaussian coefficients, the paper's; Haar-rotated coefficients, exact sample spectrum): null A certifies clustering, not depth. The hub-randomizing null B, which keeps every cluster intact and resamples only the hub configuration, also reports the star as ``hierarchical'': a near-regular simplex of hubs already minimizes $\delta$, so any hub randomization raises it. Null B is therefore not a valid depth test as constructed and is not used in the paper; a star-calibrated version (excess B relative to a matched star with the same cluster count and tightness) may still serve as a depth test, since under the Haar construction the two-level hierarchy sits $2$--$3\times$ below the star. That version is run on the real backbones in Table~\ref{tab:b34-depthvariants}. Under the plain Gaussian null the mid-radius star scores $-0.104$ and the $6{\times}5$ hierarchy $-0.14$: a star and a two-level tree differ by a fraction of either's excess, which is why depth cannot be read from the spectrum excess alone.}",
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
            cs += ["--","--","--"] if a is None else [f"${float(a[k]):+.3f}$" for k in ("excess_gauss","excess_haar","excess_pcperm")]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Excess under three constructions of the spectrum-matched null (10 replicates each): Gaussian coefficients (the paper's), Haar-rotated coefficients (exact sample spectrum) and PC-permutation (exact per-component marginals). The Gaussian and PC-permutation constructions agree; the exact-sample-spectrum construction yields smaller excesses for low-effective-rank clouds on the small-$C$ sets (most visibly DINOv2 on CIFAR-100), so the size of those excesses is partly a property of the null hypothesis, while signs and the ImageNet readings are stable.}",
      r"\label{tab:b27-nullvariants}",r"\end{table}"]
    (OUT/"tab_b27_nullvariants.tex").write_text("\n".join(lines)+"\n"); print(f"b27 written ({len(rows)} cells)")

# ---- B28: extraction variance attribution (expR47) ----
f = RES/"expR47_extraction_variance.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\begin{tabular}{lcc|cc}",r"\toprule",
      r"& \multicolumn{2}{c|}{GPT-2 M} & \multicolumn{2}{c}{OLMo-1B} \\", r"batch size & fp32 $\hat\delta$ & fp16 $\hat\delta$ & fp32 $\hat\delta$ & fp16 $\hat\delta$ \\",r"\midrule"]
    for bs in (1,16,32):
        cs=[]
        for m in ("gpt2_m","olmo_1b"):
            for fp in (0,1):
                a=[r for r in rows if r['model']==m and int(r['fp16'])==fp and int(r['batch'])==bs]
                cs.append(f"${float(a[0]['delta']):.4f}$" if a else "--")
        lines.append(f"{bs}{' (no padding)' if bs==1 else ''} & "+" & ".join(cs)+r" \\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{Where extraction-to-extraction variability comes from. Same 1000 prompts, same model; only batch size and precision vary. Precision is irrelevant (fp16 equals fp32 to four decimals). For GPT-2, left-padded batching changes the hidden states (mean cosine to the padding-free extraction $0.98$) and moves $\hat\delta$ by up to $0.018$; OLMo handles left padding correctly and is invariant. Padding-free (batch-size-1) extraction is therefore the protocol of Table~\ref{tab:b23-text200}.}",
      r"\label{tab:b28-extraction}",r"\end{table}"]
    (OUT/"tab_b28_extraction.tex").write_text("\n".join(lines)+"\n"); print(f"b28 written ({len(rows)} rows)")

# ---- B29: star-calibrated depth test (expR50) ----
f = RES/"expR50_depth_test.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    if len(rows)>=24:
        by={(a['model'],a['dataset']):a for a in rows}
        M=["i21k_t","i21k_s","i21k_b","i21k_l","dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g","clip_b","clip_l","siglip_b"]
        MN={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
        def cell(a):
            return f"${float(a['excessB_real']):+.3f}$ & ${float(a['excessB_star']):+.3f}$ & ${float(a['depth_excess']):+.3f}$ ({float(a['z_depth']):+.1f})"
        lines=[r"\begin{table}[H]",r"\centering",r"\footnotesize",r"\setlength{\tabcolsep}{3.5pt}",
          r"\begin{tabular}{lccc@{\hspace{9pt}}ccc}",r"\toprule",
          r" & \multicolumn{3}{c}{CIFAR-100 ($K{=}20$ coarse)} & \multicolumn{3}{c}{ImageNet ($K{=}30$ WordNet)} \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
          r"model & real & star & depth ($z$) & real & star & depth ($z$) \\",r"\midrule"]
        for m in M:
            lines.append(MN[m]+" & "+cell(by[(m,'cifar100')])+" & "+cell(by[(m,'imagenet')])+r" \\")
        lines+=[r"\bottomrule",r"\end{tabular}",
          r"\caption{Star-calibrated depth test (excess B, Haar construction). For each backbone the class centroids are compared with a \emph{matched star}: $K$ Gaussian hubs with the real hubs' RMS radius and, around each hub, an isotropic Gaussian cloud with that superclass's real within-cluster RMS spread (same $n$, $d$, $K$ and cluster sizes; 3 star seeds). ``real'' and ``star'' are excesses of $\hat\delta$ over the hub-randomizing null (each cluster kept intact, hubs replaced by a Haar-rotated sample with the hubs' exact spectrum, 10 replicates), which has a small star bias of its own; ``depth'' is their difference (real minus star; negative = more tree-like than a matched star), with $z$ against the combined star, null and estimator spread. On ImageNet only DINOv2-L exceeds its matched star beyond two spreads ($z{=}-2.5$; DINOv2-G $-1.7$, all others $|z|\le1.6$); on CIFAR-100 every backbone is \emph{less} tree-like than its matched star. The excess of \S\ref{sec:form} is therefore accounted for by clustered, star-like organization; residual depth is at most marginal.}",
          r"\label{tab:b29-depth}",r"\end{table}"]
        (OUT/"tab_b29_depth.tex").write_text("\n".join(lines)+"\n"); print(f"b29 written ({len(rows)} rows)")

# ---- B30: hyperbolic-backbone control (expR51, MERU vs CLIP twins) ----
f = RES/"expR51_meru_control.csv"
if f.exists():
    rows=list(csv.DictReader(open(f)))
    by={(a['model'],a['dataset'],a['modality']):a for a in rows}
    M=[m for m in ["meru_s","clip_s","meru_b","clip_b","meru_l","clip_l"] if any(a['model']==m for a in rows)]
    MN={"meru_s":"MERU ViT-S","clip_s":"CLIP ViT-S","meru_b":"MERU ViT-B","clip_b":"CLIP ViT-B","meru_l":"MERU ViT-L","clip_l":"CLIP ViT-L"}
    if len(rows)>=15:
        def cell(a):
            if a is None: return "-- & -- & --"
            return f"${float(a['excessA']):+.3f}$ ({float(a['zA']):+.1f}) & ${float(a['depth']):+.3f}$ ({float(a['z_depth']):+.1f}) & ${float(a['delta_native']):.3f}$/${float(a['delta']):.3f}$"
        lines=[r"\begin{table}[H]",r"\centering",r"\scriptsize",r"\setlength{\tabcolsep}{2.6pt}",
          r"\begin{tabular}{lccc@{\hspace{8pt}}ccc}",r"\toprule",
          r" & \multicolumn{3}{c}{CIFAR-100 images ($K{=}20$)} & \multicolumn{3}{c}{ImageNet images ($K{=}30$)} \\",
          r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
          r"model & exc.\ ($z$) & depth ($z$) & $\hat\delta$ nat./Eucl. & exc.\ ($z$) & depth ($z$) & $\hat\delta$ nat./Eucl. \\",r"\midrule"]
        for m in M:
            lines.append(MN[m]+" & "+cell(by.get((m,'cifar100','image')))+" & "+cell(by.get((m,'imagenet','image')))+r" \\")
        lines+=[r"\bottomrule",r"\end{tabular}",
          r"\caption{Hyperbolic-backbone control. MERU \citep{desai2023meru} embeds images on the Lorentz hyperboloid with an entailment objective; its released Euclidean twin (a CLIP baseline: same ViT, same RedCaps data and recipe, minus the hyperbolic lift and entailment loss) differs only in geometry. Both pass through the paper's instrument unchanged (class centroids of the projected embeddings; ``exc.'' is the excess over the spectrum-matched null, ``depth'' the star-calibrated test of Table~\ref{tab:b34-depthvariants}, ``nat.'' the Gromov $\hat\delta$ computed with the model's own metric: Lorentz distance between tangent-space-mean centroids for MERU, angular for CLIP). Training in hyperbolic space changes nothing at the class level: MERU shows the same clustering excess as its twin and its Lorentz-metric $\hat\delta$ is indistinguishable from the Euclidean one. The depth columns use the isotropic matched star of expR50 at the superclass frame, a construction whose power for hierarchy below its frame is zero (Table~\ref{tab:b35-power}); they are reported for completeness, not as evidence about depth. MERU's hierarchy is a generic$\to$specific (text$\supset$image) partial order, not a class taxonomy; the class-tree depth that no standard backbone shows is not present in a hyperbolic one either. The synthetic two-level tree of Table~\ref{tab:b25-star} remains the existence proof that the depth test fires when depth is present.}",
          r"\label{tab:b30-meru}",r"\end{table}"]
        (OUT/"tab_b30_meru.tex").write_text("\n".join(lines)+"\n"); print(f"b30 written ({len(rows)} rows, {len(M)} models)")


# ---- B31: Groger-style permutation calibration, K=200 (exp21b) ----
f = RES/"exp21b_local_global_K200.csv"
if f.exists():
    import statistics as _st
    rows=list(csv.DictReader(open(f)))
    if len(rows)==66:
        TAGS=[("knn_R","mutual-kNN ($k{=}10$), Euclidean"),("knn_H","mutual-kNN, Poincar\\'e"),("cka_R","linear CKA, Euclidean"),("cka_H","linear CKA, Poincar\\'e")]
        lines=[r"\begin{table}[H]",r"\centering",r"\footnotesize",r"\setlength{\tabcolsep}{3pt}",
          r"\begin{tabular}{lcccccc}",r"\toprule",
          r"measure & raw & null mean & $\tau_{0.05}$ & calibrated & null-centered & pairs $p<0.05$ \\",r"\midrule"]
        for tag,name in TAGS:
            g=lambda k: _st.mean(float(a[f"{tag}_{k}"]) for a in rows)
            fr=sum(float(a[f"{tag}_p"])<0.05 for a in rows)
            lines.append(f"{name} & ${g('raw'):.3f}$ & ${g('null_mean'):.3f}$ & ${g('tau95'):.3f}$ & ${g('cal'):.3f}$ & ${g('nullcentered'):.3f}$ & {fr}/66 \\\\")
        lines+=[r"\bottomrule",r"\end{tabular}",
          r"\caption{\textbf{Cross-model agreement survives permutation calibration.} Means over the 66 model pairs of the 12 vision backbones on the 1000 ImageNet class centroids (census cache). Calibration of \citet{groger2026aristotelian} with $K{=}200$ permutations of the class correspondence and $\alpha{=}0.05$, scalar (no layer search): $\tau_{0.05}$ is the $\lceil0.95(K{+}1)\rceil$-th order statistic of the observed score and its nulls (eq.\ 9), $p=(1+\#\{\text{null}\ge\text{obs}\})/(K{+}1)$ (eq.\ 10), calibrated $=\max\{(\text{obs}-\tau_{0.05})/(1-\tau_{0.05}),0\}$ (eq.\ 12); ``null-centered'' is the earlier raw-minus-null-mean variant, kept for continuity. The mKNN null mean equals the analytic chance level $k/(n{-}1)=10/999$. % exp21b_local_global_K200.csv",
          r"}", r"\label{tab:b31-groger}",r"\end{table}"]
        (OUT/"tab_b31_groger.tex").write_text("\n".join(lines)+"\n"); print("b31 written (66 pairs)")


# ---- B32: cosine census (expR57; robustness, never the record) ----
f = RES/"expR57_census_cosine_haar_p999_200.csv"; ft = RES/"expR57_text_cosine_haar_p999_200.csv"; frec = RES/"expR52_census_haar_p999_200.csv"
if f.exists() and frec.exists():
    cv=list(csv.DictReader(open(f))); rec={(a['model'],a['dataset']): a for a in csv.DictReader(open(frec))}
    by={(a['model'],a['dataset']):a for a in cv}
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    DSH={'imagenet':'IN','cifar100':'C100','cifar10':'C10','dtd':'DTD','fashionmnist':'FMNIST','mnist':'MNIST'}; DS=list(DSH)
    gb=lambda a: str(a['genuine_bh'])=="True"
    agree=sum(gb(by[k])==gb(rec[k]) for k in by if k in rec); n=len(cv); gen=sum(gb(a) for a in cv)
    top=sum(gb(a) for a in cv if a['dataset'] in ('imagenet','cifar100'))
    lines=[r"\begin{table}[H]",r"\centering",r"\scriptsize",r"\setlength{\tabcolsep}{2.4pt}",r"\begin{tabular}{l"+"cc"*len(DS)+"}",r"\toprule",
      " & "+" & ".join(f"\\multicolumn{{2}}{{c}}{{{DSH[d]}}}" for d in DS)+r" \\", r"model & "+" & ".join([r"exc.\ & $r$"]*len(DS))+r" \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        cs=[]
        for d in DS:
            a=by[(m,d)]; cs += [f"${float(a['excess']):+.3f}" + ("" if gb(a) else r"^{\circ}") + "$", f"{int(a['r_above'])}"]
        lines.append(NAME[m]+" & "+" & ".join(cs)+r" \\")
    txt=""
    if ft.exists():
        ct=list(csv.DictReader(open(ft))); txt=f" Text (same protocol on the padding-free embeddings): genuine {sum(gb(a) for a in ct)}/15: " + ", ".join(a['model'].replace('_','-') for a in ct if gb(a)) + "."
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{\textbf{Cosine census (robustness reading, not the record).} Centroids L2-normalized, spherical geodesic distances, Haar null built on the normalized cloud and re-normalized, 99.9th-percentile statistic, 200 replicates, BH over 72 cells ($^{\circ}$: not genuine). "
      f"Sign-negative {sum(float(a['excess'])<0 for a in cv)}/{n}; genuine {gen}/{n}, {top}/24 on ImageNet+CIFAR-100; verdict agreement with the Euclidean record {agree}/{n} cells." + txt + r" % expR57_census_cosine_haar_p999_200.csv, expR57_text_cosine_haar_p999_200.csv" + "\n}",
      r"\label{tab:b32-cosine}",r"\end{table}"]
    (OUT/"tab_b32_cosine.tex").write_text("\n".join(lines)+"\n"); print(f"b32 written (cosine: genuine {gen}/{n}, agree {agree})")

# ---- B33: tree map with and without a cut (expR58) ----
f = RES/"expR58_treemap_cutfree_summary.csv"
if f.exists():
    s=list(csv.DictReader(open(f)))
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\setlength{\tabcolsep}{4pt}",r"\begin{tabular}{llccc}",r"\toprule",
      r"dataset & configuration & ARI at the cut & cophenetic corr. & triplet agreement \\",r"\midrule"]
    for a in s:
        lines.append(f"{'ImageNet' if a['dataset']=='imagenet' else 'CIFAR-100'} & {a['metric']}-{a['linkage']} & {float(a['ari_cut_big_vs_block']):.2f} / {float(a['ari_cut_within_block']):.2f} & {float(a['coph_corr_big_vs_block']):.2f} / {float(a['coph_corr_within_block']):.2f} & {float(a['triplet_agree_big_vs_block']):.2f} / {float(a['triplet_agree_within_block']):.2f} \\\\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{\textbf{The island is a property of the cut and of the Euclidean configurations.} Each entry: DINOv2-B/L/G vs the supervised+contrastive block / within the block. ARI between cuts at 30 (ImageNet) or 20 (CIFAR-100) clusters; Pearson correlation between the two trees' cophenetic distance vectors; agreement on which pair merges first over $10^4$ random class triplets. Dendrograms rebuilt from the census cache with the six configurations of Table~\ref{tab:b7-treemapcontrols}. % expR58_treemap_cutfree_summary.csv" + "\n}",
      r"\label{tab:b33-cutfree}",r"\end{table}"]
    (OUT/"tab_b33_cutfree.tex").write_text("\n".join(lines)+"\n"); print("b33 written")

# ---- B34: matched-star depth test, isotropic vs anisotropic star, K sweep (expR56) ----
f = RES/"expR56_depth_variants.csv"
if f.exists():
    dv=list(csv.DictReader(open(f))); g2={(a['model'],a['dataset'],int(a['K']),a['variant']):a for a in dv}
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    def cell(a): return "--" if a is None else f"${float(a['depth_excess']):+.3f}$ ({float(a['z_depth']):+.1f})"
    lines=[r"\begin{table}[H]",r"\centering",r"\footnotesize",r"\setlength{\tabcolsep}{3pt}",r"\begin{tabular}{lcc@{\hspace{8pt}}cc}",r"\toprule",
      r" & \multicolumn{2}{c}{CIFAR-100 ($K{=}20$)} & \multicolumn{2}{c}{ImageNet ($K{=}30$)} \\",r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
      r"model & isotropic star & anisotropic star & isotropic star & anisotropic star \\",r"\midrule"]
    for i,m in enumerate(NAME):
        if i in (4,9): lines.append(r"\midrule")
        lines.append(NAME[m]+" & "+" & ".join(cell(g2.get(k)) for k in [(m,'cifar100',20,'iso'),(m,'cifar100',20,'aniso'),(m,'imagenet',30,'iso'),(m,'imagenet',30,'aniso')])+r" \\")
    def summ(ds,K,v):
        rows_=[a for a in dv if a['dataset']==ds and int(a['K'])==K and a['variant']==v]
        return f"{sum(float(a['z_depth'])<=-2 for a in rows_)}/{len(rows_)} at $z\\le-2$, {sum(float(a['z_depth'])>=2 for a in rows_)}/{len(rows_)} at $z\\ge+2$"
    ks=" ".join(f"CIFAR-100 $K{{=}}{K}$: iso {summ('cifar100',K,'iso')}; aniso {summ('cifar100',K,'aniso')}." for K in (5,10,20)) + " " + " ".join(f"ImageNet $K{{=}}{K}$: iso {summ('imagenet',K,'iso')}; aniso {summ('imagenet',K,'aniso')}." for K in (10,30,60))
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{\textbf{The matched-star depth test on the real backbones: the star's shape decides the verdict.} Depth = excess B of the real centroids minus that of a matched star (negative = more hierarchical above the frame than the star), with $z$ against the combined spread; 10 star seeds; hub-randomizing Haar null. Isotropic star: Gaussian clouds with each superclass's RMS spread (the construction of expR50). Anisotropic star: within each superclass a Haar sample with the cloud's own covariance. Frames: CIFAR-100 coarse labels; ImageNet WordNet cut. K sweep: " + ks + r" % expR56_depth_variants.csv" + "\n}",
      r"\label{tab:b34-depthvariants}",r"\end{table}"]
    (OUT/"tab_b34_depthvariants.tex").write_text("\n".join(lines)+"\n"); print("b34 written")

# ---- B35: power of the depth test (expR55 top frame; expR55b leaf frame when available) ----
f1 = RES/"expR55_depth_power.csv"; f2 = RES/"expR55b_depth_power_leafframe.csv"
if f1.exists():
    import pandas as _pd
    def _summ(f):
        d=_pd.read_csv(f); h=d[d.level!="star"]; st=d[d.level=="star"]; out=[]
        for n in (100,1000):
            for K in (6,12,20,30):
                hh=h[(h.n==n)&(h.K==K)]; ss=st[(st.n==n)&(st.K==K)]
                if len(hh)==0: continue
                out.append((n,K,[(hh[hh.ratio==r].z<=-2).mean() if (hh.ratio==r).any() else float('nan') for r in (0.1,0.3,0.6)], (ss.z<=-2).mean() if len(ss) else float('nan'), (ss.z>=2).mean() if len(ss) else float('nan')))
        return out
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\setlength{\tabcolsep}{4pt}",r"\begin{tabular}{llcccccc}",r"\toprule",
      r"frame & $n$ & $K$ & power @ ratio 0.1 & 0.3 & 0.6 & star $z\le-2$ & star $z\ge+2$ \\",r"\midrule"]
    for label, f in (("top-level, isotropic star", f1), ("leaf clusters, anisotropic star", f2)):
        if not f.exists(): continue
        for n,K,pw,fa1,fa2 in _summ(f):
            lines.append(f"{label} & {n} & {K} & " + " & ".join("--" if p!=p else f"{p:.2f}" for p in pw) + f" & {fa1:.2f} & {fa2:.2f} \\\\")
        lines.append(r"\midrule")
    lines=lines[:-1]
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{\textbf{Power of the matched-star depth test on synthetic hierarchies.} Two- and three-level hierarchies (pooled) and pure stars, $d{=}768$, within/between noise ratios $0.1/0.3/0.6$, isotropic and anisotropic clusters pooled, 5 seeds. Power = fraction of hierarchy runs with $z\le-2$; star columns = false-alarm rates in each direction. Top-level frame: the test receives the $K$ super-cluster labels, as the isotropic test of expR50 did on the real backbones; its hub null keeps every frame cluster intact and only rearranges the $K$ hubs, so hierarchy below the frame is invisible by construction. Leaf frame: the test receives the finest cluster labels with the anisotropic matched star of Table~\ref{tab:b34-depthvariants} (configurations with more leaves than points are infeasible and omitted). % expR55_depth_power.csv, expR55b_depth_power_leafframe.csv" + "\n}",
      r"\label{tab:b35-power}",r"\end{table}"]
    (OUT/"tab_b35_power.tex").write_text("\n".join(lines)+"\n"); print("b35 written", "(leaf frame included)" if f2.exists() else "(top frame only so far)")

# ---- B36: ImageNet centroid bootstrap under the record (expR59) ----
f = RES/"expR59_imagenet_bootstrap_summary.csv"
if f.exists():
    b=list(csv.DictReader(open(f)))
    NAME={"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
    lines=[r"\begin{table}[H]",r"\centering",r"\small",r"\begin{tabular}{lcccc}",r"\toprule",
      r"model & excess (reference) & bootstrap mean & bootstrap s.d. & fraction negative \\",r"\midrule"]
    for a in b: lines.append(f"{NAME.get(a['model'],a['model'])} & ${float(a['excess_ref']):+.4f}$ & ${float(a['excess_boot_mean']):+.4f}$ & ${float(a['excess_boot_sd']):.4f}$ & {float(a['frac_boot_negative']):.2f} \\\\")
    lines+=[r"\bottomrule",r"\end{tabular}",
      r"\caption{\textbf{The ImageNet excess is stable under resampling of the 100 training images per class.} 30 bootstrap resamples per backbone under the record protocol (Haar null, 99.9th-percentile statistic; 20 null replicates per resample). The bootstrap s.d. of the excess is at most " + f"{max(float(a['excess_boot_sd']) for a in b):.4f}" + r"; every resample of every backbone is sign-negative. % expR59_imagenet_bootstrap_summary.csv" + "\n}",
      r"\label{tab:b36-bootstrap}",r"\end{table}"]
    (OUT/"tab_b36_bootstrap.tex").write_text("\n".join(lines)+"\n"); print("b36 written")
