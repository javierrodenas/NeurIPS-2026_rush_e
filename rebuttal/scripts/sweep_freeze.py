#!/usr/bin/env python3
"""
FREEZE SWEEP — verifies the numbers in main_iclr2027.tex prose against their
source CSVs, plus a LIVE synthetic re-verification of the calibration table.
Extends the night sweep with exp18b/21/22/23/24 and the rewritten Secs. 4-6.
"""
import csv, itertools, sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
# numpy<2 compat: this npz was pickled under numpy>=2 (numpy._core module path)
import sys as _sys, numpy.core as _nc
_sys.modules.setdefault("numpy._core", _nc)
for _s in ("multiarray", "umath", "numeric", "_multiarray_umath"):
    try:
        _sys.modules.setdefault("numpy._core." + _s, __import__("numpy.core." + _s, fromlist=["_"]))
    except Exception:
        pass
from statistics import mean
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

import os
R = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
checks = []
def chk(name, cond, detail=""):
    checks.append((name, bool(cond), detail))
def load(f, root=R): return list(csv.DictReader(open(root/f)))

def delta_from_D(D, n_quads=200_000, n_seeds=5):
    diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s)
        i,j,k,l = (rng.randint(0,n,n_quads) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l],D[i,k]+D[j,l],D[i,l]+D[j,k]],1),1)
        out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out))

# ---------- 1. LIVE calibration-table verification ----------
# binary tree depth 10 (1023 nodes), tree metric
n_nodes = 2**10 - 1
depth = np.array([int(np.floor(np.log2(i+1))) for i in range(n_nodes)])
def lca_depth(u, v):
    a, b = u+1, v+1
    while a != b:
        if a > b: a //= 2
        else: b //= 2
    return int(np.floor(np.log2(a)))
sub = np.random.RandomState(0).choice(n_nodes, 400, replace=False)
Dt = np.zeros((400,400))
for x in range(400):
    for y in range(x+1,400):
        u,v = sub[x], sub[y]
        d = depth[u]+depth[v]-2*lca_depth(u,v)
        Dt[x,y]=Dt[y,x]=d
chk("calib: tree delta=0.000", delta_from_D(Dt) < 0.005, f"{delta_from_D(Dt):.4f}")

# H2 region radius 4
rng = np.random.RandomState(1)
Rr = 4.0
u = rng.rand(1000)
r = np.arccosh(1 + u*(np.cosh(Rr)-1))
th = rng.rand(1000)*2*np.pi
c1 = np.cosh(r)[:,None]*np.cosh(r)[None,:]
s1 = np.sinh(r)[:,None]*np.sinh(r)[None,:]
Dh = np.arccosh(np.clip(c1 - s1*np.cos(th[:,None]-th[None,:]), 1, None))
np.fill_diagonal(Dh, 0)
dh_norm = delta_from_D(Dh)
# absolute delta: max defect (unnormalized) approx via norm*diam
chk("calib: H2 R=4 delta_norm ~0.087", abs(dh_norm-0.087) < 0.012, f"{dh_norm:.3f}")
chk("calib: H2 R=4 absolute ~0.69 (ln2)", abs(dh_norm*Dh.max()-0.69) < 0.06, f"{dh_norm*Dh.max():.3f}")

# S99 chord + geodesic
X = rng.randn(1000,100); X /= np.linalg.norm(X,axis=1,keepdims=True)
Dc = squareform(pdist(X)); Dg = np.arccos(np.clip(1 - Dc**2/2, -1, 1))
chk("calib: S99 chord ~0.143", abs(delta_from_D(Dc)-0.143) < 0.012, f"{delta_from_D(Dc):.3f}")
chk("calib: S99 geodesic ~0.179", abs(delta_from_D(Dg)-0.179) < 0.015, f"{delta_from_D(Dg):.3f}")

# iid gaussian d=768
G = rng.randn(1000,768)
chk("calib: gauss d=768 ~0.061", abs(delta_from_D(squareform(pdist(G)))-0.061) < 0.008,
    f"{delta_from_D(squareform(pdist(G))):.3f}")

# ---------- 2. Vision census of record (expR39c: 200 spectrum-null replicates, census cache) ----------
e20 = load("exp20_null_ztable.csv")          # kept: raw deltas for the corollary checks below
e39 = load("expR39c_census200_cache.csv")
HIER = {"imagenet","cifar100","cifar10","dtd"}
exc = {(r["model"],r["dataset"]):float(r["excess"]) for r in e39}
zz  = {(r["model"],r["dataset"]):float(r["z"]) for r in e39}
rk  = {(r["model"],r["dataset"]):int(r["r_above"]) for r in e39}
pp  = {(r["model"],r["dataset"]):float(r["p_left"]) for r in e39}
chk("census: 72 cells, 200 replicates, none from the store", len(e39)==72 and all(int(r["store_centroids"])==0 for r in e39)
    and all(0<=v<=200 for v in rk.values()))
chk("68/72 sign-neg", sum(1 for v in exc.values() if v<0)==68)
pos = [(m,d) for (m,d),v in exc.items() if v>0]
chk("4 sign-positive: Dv2 S/B/G IN (exc<=+0.011, p>=0.99) + ViT-T FMNIST (~0, p~0.5)",
    set(pos)=={("dinov2_s","imagenet"),("dinov2_b","imagenet"),("dinov2_g","imagenet"),("i21k_t","fashionmnist")}
    and all(exc[p]<=0.0115 and pp[p]>=0.99 for p in pos if p[1]=="imagenet") and exc[("i21k_t","fashionmnist")]<0.002,
    str([(p,round(exc[p],4),round(pp[p],3)) for p in pos]))
chk("genuine p<=0.05: 55/72, hier 43/48, flat 12/24", sum(1 for v in pp.values() if v<=0.05)==55
    and sum(1 for (m,d),v in pp.items() if d in HIER and v<=0.05)==43 and sum(1 for (m,d),v in pp.items() if d not in HIER and v<=0.05)==12)
chk("44/72 below all 200 replicates; flat 23/24 sign-neg", sum(1 for v in rk.values() if v==200)==44
    and sum(1 for (m,d),v in exc.items() if d not in HIER and v<0)==23)
chk("genuine <=> r>=191", all((v<=0.05)==(rk[k]>=191) for k,v in pp.items()))
chk("B20 caption: |z|>=2 49/72, hier z<=-3 32/48", sum(1 for v in zz.values() if abs(v)>=2)==49
    and sum(1 for (m,d),v in zz.items() if d in HIER and v<=-3)==32)
vit = [exc[(m,"imagenet")] for m in ["i21k_t","i21k_s","i21k_b","i21k_l"]]; cs = [exc[(m,"imagenet")] for m in ["clip_b","clip_l","siglip_b"]]
chk("IN ranges: ViT -0.020..-0.030, CLIP/SigLIP -0.016..-0.025", abs(max(vit)+0.020)<0.0015 and abs(min(vit)+0.030)<0.0015
    and abs(max(cs)+0.016)<0.0015 and abs(min(cs)+0.025)<0.0015, f"{max(vit):.4f}..{min(vit):.4f} | {max(cs):.4f}..{min(cs):.4f}")
chk("Dv2-L IN -0.008 genuine r=200; S->G C10 -0.095->-0.155", abs(exc[("dinov2_l","imagenet")]+0.008)<0.001 and rk[("dinov2_l","imagenet")]==200
    and abs(exc[("dinov2_s","cifar10")]+0.095)<0.002 and abs(exc[("dinov2_g","cifar10")]+0.155)<0.002)
chk("R1 fidelity: ImageNet raw delta == exp20 cache (4 dp)",
    all(abs(float(r["delta"])-float(next(a for a in e20 if a["model"]==r["model"] and a["dataset"]=="imagenet")["delta"]))<5e-5
        for r in e39 if r["dataset"]=="imagenet"))

# sphere controls (exp1)
e1 = {}
for r in load("exp1_delta_controls.csv"):
    e1.setdefault(r["model"],{})[r["variant"]] = float(r["delta_max"])
chk("sphere 12/12 geo, 10/12 chord",
    sum(1 for m,d in e1.items() if d.get("l2_geodesic",9)>d["real"])==12
    and sum(1 for m,d in e1.items() if d.get("l2_chord",9)>d["real"])==10)

# curvature sign (exp12)
e12 = {r["model"]:r for r in load("exp12_curvature_sign.csv")}
chk("xi Dv2 -0.020->-0.070 66->89%", abs(float(e12["dinov2_s"]["xi_mean"])+0.020)<0.002
    and abs(float(e12["dinov2_g"]["xi_mean"])+0.070)<0.002
    and abs(float(e12["dinov2_s"]["frac_neg"])-0.66)<0.02 and abs(float(e12["dinov2_g"]["frac_neg"])-0.89)<0.02)
others_xi = [float(e12[m]["xi_mean"]) for m in ["i21k_t","i21k_s","i21k_b","i21k_l","clip_b","clip_l","siglip_b"]]
chk("sup/con xi +0.02..+0.04, < gauss 0.09", 0.015<min(others_xi) and max(others_xi)<0.041
    and max(others_xi)<float(e12["ref_gauss768"]["xi_mean"]), f"{min(others_xi):.3f}..{max(others_xi):.3f}")

# ---------- 3. Text census of record (expR48b: padding-free, 200 replicates) + anisotropy (exp18b) ----------
e48 = {r["model"]:r for r in load("expR48b_text_census200_bs1.csv")}
g = lambda m,k: float(e48[m][k])
chk("text: 15 models, 200 replicates", len(e48)==15 and all(0<=float(r["r_above"])<=200 for r in e48.values()))
chk("GPT-2 S/M not genuine (-0.013/-0.009; p .090/.139), L/XL genuine (-0.039/-0.047)",
    abs(g("gpt2","excess")+0.0132)<0.001 and abs(g("gpt2_m","excess")+0.0090)<0.001 and g("gpt2","p_left")>0.05 and g("gpt2_m","p_left")>0.05
    and abs(g("gpt2_l","excess")+0.0389)<0.001 and abs(g("gpt2_xl","excess")+0.0472)<0.001 and g("gpt2_l","p_left")<=0.05 and g("gpt2_xl","p_left")<=0.05,
    f"S {g('gpt2','excess'):+.4f} p{g('gpt2','p_left'):.3f} M {g('gpt2_m','excess'):+.4f} p{g('gpt2_m','p_left'):.3f}")
chk("Pythia x3 + OLMo-1B genuine", all(g(m,"p_left")<=0.05 for m in ["pythia_410m","pythia_1b","pythia_2b8","olmo_1b"]))
emb = ["bge_base","bge_large","gte_base","gte_large","e5_base","e5_large","gte_qwen2"]
chk("7 embedders not genuine, excess +0.003..+0.012", all(g(m,"p_left")>0.05 and 0.002<g(m,"excess")<0.0125 for m in emb),
    str([round(g(m,"excess"),4) for m in emb]))
chk("genuine 6/15", sum(1 for r in e48.values() if float(r["p_left"])<=0.05)==6)
e18b = {r["model"]:r for r in load("exp18b_text_anisotropy.csv")}
chk("erank GPT2 1.3->56; GTE 98", abs(float(e18b["gpt2"]["erank"])-1.3)<0.2
    and abs(float(e18b["gpt2_xl"]["erank"])-56)<2 and abs(float(e18b["gte_base"]["erank"])-98)<2)

# prompts (exp4+17)
allp = {}
for r in load("exp4_prompt_variation.csv"): allp.setdefault(r["model"],[]).append(float(r["delta_max"]))
for r in load("exp17_wordnet_prompts.csv"): allp.setdefault(r["model"],[]).append(float(r["delta"]))
chk("templates: emb <=.012, LM hasta .065",
    max(max(v)-min(v) for m,v in allp.items() if "gpt" not in m)<=0.0125
    and abs(max(max(v)-min(v) for m,v in allp.items() if "gpt" in m)-0.065)<0.004)

# DBpedia (exp14)
db = load("exp14_dbpedia.csv")
chk("DBpedia exc -0.020..-0.028, delta .122-.132, NC -0.4..-0.8, FS +0.04..+0.08",
    all(-0.0285<float(r["excess"])<-0.0195 for r in db)
    and all(0.121<float(r["delta"])<0.1325 for r in db)
    and all(-0.85<(float(r["NC_H"])-float(r["NC_R"]))*100<-0.35 for r in db)
    and all(0.035<float(r["FS_HR_pp"])<0.085 for r in db))

# C-sweep (exp19)
e19 = load("exp19_c_sweep.csv")
def agg19(m,C,mode,f):
    v=[float(r[f]) for r in e19 if r["model"]==m and int(r["C"])==C and r["mode"]==mode]
    return mean(v) if v else None
chk("C50 Dv2-L (exp19, supremum) exc -0.124/-0.093 NC +1.58/+0.53 [historical; Table B3 now reads expR60]",
    abs(agg19("dinov2_l",50,"random","excess")+0.124)<0.003 and abs(agg19("dinov2_l",50,"coherent","excess")+0.093)<0.003
    and abs(agg19("dinov2_l",50,"random","nc_adv_pp")-1.58)<0.05 and abs(agg19("dinov2_l",50,"coherent","nc_adv_pp")-0.53)<0.05)
if (R/"expR60_c_sweep_record.csv").exists() and (R/"phaseB_final_numbers.json").exists():
    import json as _j
    e60 = load("expR60_c_sweep_record.csv"); FN = _j.load(open(R/"phaseB_final_numbers.json"))
    def agg60(m,C,mode,f):
        v=[float(r[f]) for r in e60 if r["model"]==m and int(r["C"])==C and r["mode"]==mode]; return mean(v) if v else None
    def allbelow(m,C,mode): return all(float(r["p_left"])<=0.05 for r in e60 if r["model"]==m and int(r["C"])==C and r["mode"]==mode)
    chk("C-sweep record: text numbers (Dv2-L C=50 random/coherent excess) match expR60; every seed has 200 replicates",
        abs(agg60("dinov2_l",50,"random","excess")-FN["dv2l_c50_random"])<1e-6 and abs(agg60("dinov2_l",50,"coherent","excess")-FN["dv2l_c50_coherent"])<1e-6
        and len(e60)==165 and all(int(r["r_above"])<=200 for r in e60))
    def thr60(m):
        t=0
        for C in (10,20,50,100,200,500):
            if agg60(m,C,"random","excess")<agg60(m,C,"coherent","excess"): t=C
            else: break
        return t
    chk("C-sweep record: random-more-than-coherent thresholds per model, NC random>coherent everywhere, C=10 random excess, Dv2-G below null at every C, as stated",
        all(thr60(m)==FN["random_more_thresholds"][m] for m in ("dinov2_l","dinov2_g","clip_l"))
        and FN["nc_random_more_everywhere"]==all(agg60(m,C,"random","nc_adv_pp")>agg60(m,C,"coherent","nc_adv_pp") for m in ("dinov2_l","dinov2_g","clip_l") for C in (10,20,50,100,200,500))
        and all(abs(agg60(m,10,"random","excess")-FN["c10_random_excess"][m])<1e-6 for m in ("dinov2_l","dinov2_g","clip_l"))
        and FN["dv2g_all_below"]==all(allbelow("dinov2_g",C,"random") for C in (10,20,50,100,200,500,1000))
        and abs(agg60("dinov2_g",1000,"random","excess")-FN["dv2g_c1000_excess"])<1e-6)
    d61 = load("expR61_dbpedia_record.csv")
    chk("DBpedia record: three embedders, excess range and min r as stated in the text",
        len(d61)==3 and abs(max(float(r["excess"]) for r in d61)-FN["dbpedia_excess_lo"])<1e-6 and abs(min(float(r["excess"]) for r in d61)-FN["dbpedia_excess_hi"])<1e-6
        and min(int(r["r_above"]) for r in d61)==FN["dbpedia_r_min"] and all(str(r["genuine_bh"])=="True" for r in d61)==FN["dbpedia_bh_all"])

# ORC bridges
br = load("night/orc_bridges.csv")
def brf(pref, ds, col):
    return mean(float(r[col]) for r in br if r["dataset"]==ds and r["model"].startswith(pref))
chk("bridges IN ViT 7.6/3.2 CLIPSIG 8.0/4.4",
    abs(100*brf("i21k","imagenet","fneg_across")-7.6)<0.3 and abs(100*brf("i21k","imagenet","fneg_within")-3.2)<0.3
    and abs(100*mean([brf("clip","imagenet","fneg_across"),brf("siglip","imagenet","fneg_across")]*1)-8.0)<0.8)

# ---------- 4. Tree maps (exp22/23) ----------
z22 = np.load(R/"exp22_tree_similarity_imagenet.npz", allow_pickle=True)
ari = z22["ari"]; names = list(z22["models"])
ix = {m:i for i,m in enumerate(names)}
SUP = ["i21k_t","i21k_s","i21k_b","i21k_l","clip_b","clip_l","siglip_b"]
D2 = ["dinov2_s","dinov2_b","dinov2_l","dinov2_g"]
chk("naive IN: ViT-T~CLIP-B 0.82", abs(ari[ix["i21k_t"],ix["clip_b"]]-0.82)<0.01)
wnv = [ari[ix[m],ix["wordnet"]] for m in SUP]
chk("naive block WordNet 0.38-0.51", abs(min(wnv)-0.38)<0.01 and abs(max(wnv)-0.51)<0.01, f"{min(wnv):.2f}-{max(wnv):.2f}")
isl = [ari[ix[a],ix[b]] for a in D2 for b in SUP]
chk("naive island <=0.06 vs supervised", max(isl)<=0.065, f"max {max(isl):.3f}")
z23i = np.load(R/"exp23_treemap_controls.npz", allow_pickle=True)["summary_in"].item()
z23c = np.load(R/"exp23_treemap_controls.npz", allow_pickle=True)["summary_c1"].item()
cw = z23c["('cosine', 'ward')"]
chk("C100 cosine-ward: 0.56 vs 0.61, coph 0.74/0.74",
    abs(cw["big_vs_sup"]-0.558)<0.01 and abs(cw["sup_vs_sup"]-0.609)<0.01
    and abs(cw["cop_big_sup"]-0.740)<0.01 and abs(cw["cop_sup_sup"]-0.735)<0.01)
ca = z23i["('cosine', 'average')"]
chk("IN cosine-avg 0.38 vs 0.48", abs(ca["big_vs_sup"]-0.380)<0.01 and abs(ca["sup_vs_sup"]-0.481)<0.01)

# ---------- 5. Cross-model calibrated (exp21b: Groger calibration, K=200, alpha=0.05) ----------
e21 = load("exp21b_local_global_K200.csv")
m21 = {c: mean(float(r[c]) for r in e21) for c in ["knn_R_raw","knn_R_cal","knn_R_null_mean","knn_H_raw","cka_R_raw","cka_R_cal","cka_R_null_mean","cka_H_raw"]}
chk("exp21b: 66 pairs", len(e21)==66)
chk("exp21b kNN raw .472 cal .464 null .010 (= 10/999)", abs(m21["knn_R_raw"]-0.472)<0.002 and abs(m21["knn_R_cal"]-0.464)<0.002
    and abs(m21["knn_R_null_mean"]-10/999)<0.001, f"{m21['knn_R_raw']:.4f}/{m21['knn_R_cal']:.4f}/{m21['knn_R_null_mean']:.4f}")
chk("exp21b CKA raw .633 cal .577 null .108", abs(m21["cka_R_raw"]-0.633)<0.002 and abs(m21["cka_R_cal"]-0.577)<0.002
    and abs(m21["cka_R_null_mean"]-0.108)<0.002, f"{m21['cka_R_raw']:.4f}/{m21['cka_R_cal']:.4f}/{m21['cka_R_null_mean']:.4f}")
chk("exp21b Poincare: kNN .425, CKA .640", abs(m21["knn_H_raw"]-0.425)<0.002 and abs(m21["cka_H_raw"]-0.640)<0.002)
chk("exp21b every pair p=1/201 (<0.05) for kNN and CKA", all(float(r["knn_R_p"])<0.05 and float(r["cka_R_p"])<0.05 for r in e21)
    and all(abs(float(r["knn_R_p"])-1/201)<1e-6 for r in e21))

# ---------- 6. Corollary (table1_regenerated + exp2 + night + exp13 + exp24) ----------
t1 = load("table1_regenerated.csv")
d20 = {(r["model"],r["dataset"]):float(r["delta"]) for r in e20}
def pear(x,y):
    x,y=np.asarray(x),np.asarray(y); return float(np.corrcoef(x,y)[0,1])
cells = [(r["model"],r["dataset"]) for r in t1]
x = [d20[c] for c in cells]
chk("NC pooled -0.45", abs(pear(x,[float(r["NC_adv"]) for r in t1])+0.452)<0.005)
for ds, target in [("imagenet",-0.83),("cifar100",-0.87)]:
    sub=[r for r in t1 if r["dataset"]==ds]
    chk(f"NC {ds} {target}", abs(pear([d20[(r['model'],ds)] for r in sub],[float(r["NC_adv"]) for r in sub])-target)<0.01)
chk("FS 56/60 estricto", sum(1 for r in t1 if float(r["FS_adv"])>0)==56)
hl = next(r for r in t1 if r["model"]=="dinov2_l" and r["dataset"]=="cifar100")
chk("+2.05 headline", abs(float(hl["NC_adv"])-2.05)<0.01)
mc = next(r for r in load("exp13_mcnemar.csv") if r["model"]=="dinov2_l" and r["dataset"]=="cifar100")
chk("McNemar p~4e-38", 1e-39<float(mc["p_H_vs_R"])<1e-37)
e2 = load("exp2_metric_controls.csv")
fs_rt = [abs(float(r["FS_RT"])-float(r["FS_R"]))*100 for r in e2]
nc_rt = [abs(float(r["NC_RT"])-float(r["NC_R"]))*100 for r in e2]
chk("RT means 0.20/0.12", abs(mean(fs_rt)-0.20)<0.02 and abs(mean(nc_rt)-0.12)<0.02)
pts = [(d20[(r["model"],r["dataset"])], (max(float(r["FS_H"]),float(r["FS_COS"]))-float(r["FS_R"]))*100, r["dataset"]) for r in e2 if (r["model"],r["dataset"]) in d20]
chk("best-metric FS -0.31/-0.33", abs(pear([p[0] for p in pts],[p[1] for p in pts])+0.306)<0.006
    and abs(pear([p[0] for p in pts if p[2] in HIER],[p[1] for p in pts if p[2] in HIER])+0.330)<0.006)
e2b = load("exp2b_normalized_stack.csv")
clhn = [float(r["FS_HN_COS_diff"])*100 for r in e2b if r["model"] in ("clip_b","clip_l") and r["dataset"] in ("cifar100","cifar10","dtd")]
chk("CLIP HN-COS +0.9..+1.3 transfer", 0.85<min(clhn) and max(clhn)<1.35)
e24 = load("exp24_val_metric_selection.csv")
h24 = [r for r in e24 if r["dataset"] in HIER]
chk("exp24 val +0.63 oracle +0.64 (hier +0.78)", abs(mean(float(r["adv_val"]) for r in e24)-0.629)<0.01
    and abs(mean(float(r["adv_oracle"]) for r in e24)-0.638)<0.01
    and abs(mean(float(r["adv_val"]) for r in h24)-0.776)<0.01)
chk("exp24 rule +0.41 vs cos +0.28 hier", abs(mean(float(r["adv_rule"]) for r in h24)-0.414)<0.01
    and abs(mean(float(r["adv_cos"]) for r in h24)-0.282)<0.01)
ci = {(r["task"],r["dataset"]):(float(r["r"]),float(r["ci_lo"]),float(r["ci_hi"])) for r in load("night/correlation_cis.csv")}
chk("CIs NC pooled [-0.60,-0.36], FS pooled inc. 0",
    abs(ci[("NC_adv","POOLED(cluster-bs)")][1]+0.60)<0.01 and ci[("FS_adv","POOLED(cluster-bs)")][2]>0)
ts = load("night/t_sweep.csv")
chk("t-sweep NC cruza cero, FS>0 siempre", min(float(r["NC_adv_mean"]) for r in ts)<0<max(float(r["NC_adv_mean"]) for r in ts)
    and all(float(r["FS_adv_mean"])>0 for r in ts))

# alignment ladder + controls (exp3, exp8, exp1, exp10)
e3 = {r["model"]:r for r in load("exp3_alignment.csv")}
chk("rho ladder CLIP .57-.59 ViT .49-.53 DINOB .36 Dv2 .18-.22",
    all(0.565<float(e3[m]["spearman_wn"])<0.595 for m in ["clip_b","clip_l","siglip_b"] if m in e3)
    and all(0.485<float(e3[m]["spearman_wn"])<0.535 for m in ["i21k_t","i21k_s","i21k_b","i21k_l"])
    and abs(float(e3["dinov1_b"]["spearman_wn"])-0.36)<0.01
    and all(0.175<float(e3[m]["spearman_wn"])<0.225 for m in ["dinov2_s","dinov2_b","dinov2_l","dinov2_g"]))
chk("ARI 0.61 (exp8_p1)", abs(max(float(r["ari"]) for r in load("exp8_p1_recovery.csv"))-0.61)<0.01)
chk("groupings 11/12 clipB .126/.174", sum(1 for m,d in e1.items() if d.get("grp_wordnet",9)<d.get("grp_random",0))==11
    and abs(e1["clip_b"]["grp_wordnet"]-0.126)<0.002 and abs(e1["clip_b"]["grp_random"]-0.174)<0.002)
chk("no-pooling +0.26", abs(max(float(r["rho_mean"]) for r in load("exp8_p6_pooling.csv") if r["m_imgs"]=="1")-0.264)<0.01)
e10 = {r["model"]:r for r in load("exp10_local_vs_global.csv")}
chk("triplets 0.91/0.85 vs 0.71/0.63", abs(float(e10["dinov2_l"]["c100_sibtrip_c"])-0.91)<0.01
    and abs(float(e10["dinov2_g"]["c100_sibtrip_c"])-0.85)<0.01
    and abs(float(e10["dinov2_l"]["c100_sibtrip_e"])-0.71)<0.01
    and abs(float(e10["dinov2_g"]["c100_sibtrip_e"])-0.63)<0.01)
# arch-matched triplet honesty (Sec 4 claim "all three objectives yield genuine form")
tri = {(r["dataset"]): (float(r["i21k_b"]), float(r["dinov1_b"]), float(r["clip_b"])) for r in load("night/arch_matched_triplet.csv")}
weak = [(ds,v) for ds,vals in tri.items() for v in [vals[1]] if v>-0.01]
chk("triplet: 'all genuine' necesita scoping (DINO-B IN -0.004)", len(weak)>0, str(weak))

n_fail = sum(1 for _,ok,_ in checks if not ok)
for name, ok, det in checks:
    print(("PASS" if ok else "FAIL"), name, ("| "+det if det and not ok else ""))
print(f"\n{len(checks)-n_fail}/{len(checks)} checks passed")

# ---------- ROUND-3 additions (appended 22-ago) ----------
def round3():
    e26 = {r["model"]: r for r in load("exp26_xi_nulls.csv")}
    chk("xi-null v2: 9/12 exceso negativo, ViT-T y CLIP-L at-null, SigLIP +z",
        sum(1 for r in e26.values() if float(r["excess"])<0 and abs(float(r["z"]))>2)==9
        and abs(float(e26["i21k_t"]["z"]))<2 and abs(float(e26["clip_l"]["z"]))<2
        and float(e26["siglip_b"]["z"])>2)
    chk("xi-exceso Dv2 -0.081->-0.152 monotono, CI<=0.008", abs(float(e26["dinov2_s"]["excess"])+0.081)<0.003
        and abs(float(e26["dinov2_g"]["excess"])+0.152)<0.003
        and float(e26["dinov2_s"]["excess"])>float(e26["dinov2_b"]["excess"])>float(e26["dinov2_l"]["excess"])
        and max(float(r["ci95"]) for r in e26.values())<=0.0082)
    e2p = {(r["model"],r["dataset"]): r for r in load("exp2_metric_controls.csv")}
    d20p = {(r["model"],r["dataset"]): float(r["delta"]) for r in load("exp20_null_ztable.csv")}
    FLATp = {"fashionmnist","mnist"}
    famp = lambda m: "ssl" if m.startswith("dinov") else "sup" if m.startswith("i21k") else "con"
    advH, advC, advR_flat, advH_flat, advH_hier = [], [], [], [], []
    for (m,ds), r in e2p.items():
        if (m,ds) not in d20p: continue
        R_,H_,C_ = float(r["NC_R"]), float(r["NC_H"]), float(r["NC_COS"])
        advH.append((H_-R_)*100); advC.append((C_-R_)*100)
        if ds in FLATp: advH_flat.append((H_-R_)*100)
        else: advH_hier.append((H_-R_)*100)
    chk("NC medias: H -0.24 all / -0.00 hier / -0.70 flat; COS +0.48",
        abs(mean(advH)+0.24)<0.02 and abs(mean(advH_hier)-0.0)<0.02
        and abs(mean(advH_flat)+0.70)<0.03 and abs(mean(advC)-0.48)<0.02)
    from scipy.stats import ttest_rel, wilcoxon
    e24r = load("exp24_val_metric_selection.csv")
    h24r = [r for r in e24r if r["dataset"] in HIER]
    _, pt = ttest_rel([float(r["adv_rule"]) for r in h24r], [float(r["adv_cos"]) for r in h24r])
    chk("FS regla vs coseno pareado p=0.011", abs(pt-0.011)<0.003, f"p={pt:.4f}")
    z23 = np.load(R/"exp23_treemap_controls.npz", allow_pickle=True)
    zi = z23["summary_in"].item(); zc = z23["summary_c1"].item()
    chk("configs seleccionadas: IN cos-avg 0.38/0.48, C100 cos-comp 0.39/0.64",
        abs(zi["('cosine', 'average')"]["big_vs_sup"]-0.380)<0.01
        and abs(zc["('cosine', 'complete')"]["big_vs_sup"]-0.389)<0.01
        and abs(zc["('cosine', 'complete')"]["sup_vs_sup"]-0.635)<0.01)
round3()
n_fail = sum(1 for _,ok,_ in checks if not ok)
print(f"[round3 re-total] {len(checks)-n_fail}/{len(checks)}")

# ---------- ROUND-4 additions ----------
def round4():
    import json
    d27 = json.load(open(R/"exp27_dbpedia_treemap.json"))
    chk("DBpedia map: sin degeneracion, seleccion euclid-average CPCC .705",
        all(v["maxfrac"]<=0.105 for v in d27.values())
        and abs(d27["euclid-average"]["cpcc"]-0.705)<0.005
        and d27["euclid-average"]["cpcc"]==max(v["cpcc"] for v in d27.values()))
    l2_all = [a for v in d27.values() for a in v["ari_l2"].values()]
    chk("DBpedia ARI L2 .21-.29 estable, cross .65-.82",
        0.205<min(l2_all) and max(l2_all)<0.295
        and 0.64<min(v["cross_model"] for v in d27.values())
        and max(v["cross_model"] for v in d27.values())<0.83)
    import csv as _csv
    diag = json.load(open(R/"exp23_config_diagnostics.json"))
    adm_in = {k for k,v in diag.items() if k.startswith("imagenet") and v["maxfrac"]<=0.5}
    adm_c1 = {k for k,v in diag.items() if k.startswith("cifar100") and v["maxfrac"]<=0.5}
    chk("seleccion reproducible: IN cos-avg, C100 cos-comp",
        max(adm_in, key=lambda k: diag[k]["cpcc"])=="imagenet|cosine|average"
        and max(adm_c1, key=lambda k: diag[k]["cpcc"])=="cifar100|cosine|complete")
round4()
n_fail = sum(1 for _,ok,_ in checks if not ok)
print(f"[round4 re-total] {len(checks)-n_fail}/{len(checks)}")


# ---------- TOOL: calibrated_delta.py reproduces Table 1 (record) / B34 on two cells ----------
def tool_check():
    import json, subprocess
    jf = R/"tool_check.json"
    if not jf.exists():   # ~15 CPU-min: run once, then compare the saved output
        subprocess.run([sys.executable, str(Path(__file__).resolve().parents[2]/"ICLR2027/tool/run_checks.py")], check=True)
    J = json.load(open(jf))
    e52 = {(r["model"],r["dataset"]): r for r in load("expR52_census_haar_p999_200.csv")}
    e56 = {(r["model"],r["dataset"],int(r["K"]),r["variant"]): r for r in load("expR56_depth_variants.csv")}
    for cell, K in [("i21k_l/cifar100", 20), ("dinov2_l/imagenet", 30)]:
        m, d = cell.split("/"); c = J[cell]["census"]; ref = e52[(m,d)]
        chk(f"tool == Table 1 record ({cell}): excess 3dp, r, p", round(c["excess"],3)==round(float(ref["excess"]),3)
            and c["r_above"]==int(ref["r_above"]) and abs(c["p_left"]-float(ref["p_left"]))<1e-9,
            f"tool {c['excess']:+.4f} r{c['r_above']} p{c['p_left']:.4f} | table {float(ref['excess']):+.4f} r{ref['r_above']} p{float(ref['p_left']):.4f}")
        dd = J[cell]["depth"]; rf = e56[(m,d,K,"aniso")]
        chk(f"tool == B34 aniso ({cell}): depth 3dp, z 1dp", round(dd["depth_excess"],3)==round(float(rf["depth_excess"]),3) and round(dd["z_depth"],1)==round(float(rf["z_depth"]),1),
            f"tool {dd['depth_excess']:+.4f} z{dd['z_depth']:+.2f} | table {float(rf['depth_excess']):+.4f} z{float(rf['z_depth']):+.2f}")
tool_check()
n_fail = sum(1 for _,ok,_ in checks if not ok)
for name, ok, det in checks[-4:]: print(("PASS" if ok else "FAIL"), name, ("| "+det if det else ""))
print(f"[tool re-total] {len(checks)-n_fail}/{len(checks)}")

# ---------- p99.9 robustness census at 200 replicates (expR40b, census cache) ----------
e40 = load("expR40b_p999census200.csv")
p40 = {(r["model"],r["dataset"]): float(r["p_left"]) for r in e40}; x40 = {(r["model"],r["dataset"]): float(r["excess"]) for r in e40}
chk("p99.9@200: 70/72 sign-neg, 56/72 genuine, 44/48 hier genuine", sum(1 for v in x40.values() if v<0)==70
    and sum(1 for v in p40.values() if v<=0.05)==56 and sum(1 for (m,d),v in p40.items() if d in HIER and v<=0.05)==44)
chk("p99.9@200 ImageNet: DINO/DINOv2 genuine, ViT-T not", all(p40[(m,"imagenet")]<=0.05 for m in ["dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g"])
    and p40[("i21k_t","imagenet")]>0.05)
n_fail = sum(1 for _,ok,_ in checks if not ok)
for name, ok, det in checks[-2:]: print(("PASS" if ok else "FAIL"), name, ("| "+det if det else ""))
print(f"[p999 re-total] {len(checks)-n_fail}/{len(checks)}")


# ---------- Review-response Phase A files: existence + internal consistency (values checked in Phase B) ----------
def phaseA_checks():
    import numpy as np
    def bh_(p):
        p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p); ranked = p[order]*n/np.arange(1, n+1)
        adj = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty(n); out[order] = np.minimum(adj, 1.0); return out
    for f, nrows, key in [("expR52_census_haar_p999_200.csv", 72, "vision record"), ("expR54_census_haar_sup_200.csv", 72, "vision haar x sup"),
                          ("expR53_text_haar_p999_200.csv", 15, "text record"), ("expR53_text_haar_sup_200.csv", 15, "text haar x sup"),
                          ("expR53_text_gauss_p999_200.csv", 15, "text gauss x p999"), ("expR57_census_cosine_haar_p999_200.csv", 72, "cosine vision"),
                          ("expR57_text_cosine_haar_p999_200.csv", 15, "cosine text")]:
        if not (R/f).exists(): chk(f"phaseA {key}: {f} present", False, "missing"); continue
        rows = load(f); p = [float(r["p_left"]) for r in rows]; ra = [int(r["r_above"]) for r in rows]
        chk(f"phaseA {key}: {nrows} rows, r/p consistent, BH monotone, genuine flags", len(rows)==nrows
            and all(abs(pp-(1+200-r)/201)<1e-9 for pp, r in zip(p, ra)) and all(abs(float(r["p_bh"])-b)<1e-9 for r, b in zip(rows, bh_(p)))
            and all((float(r["p_left"])<=0.05)==(str(r["genuine"])=="True") and (float(r["p_bh"])<=0.05)==(str(r["genuine_bh"])=="True") for r in rows))
    for f, key in [("expR55_depth_power.csv","depth power"), ("expR56_depth_variants.csv","depth variants"), ("expR58_treemap_cutfree_summary.csv","cut-free tree map"), ("expR59_imagenet_bootstrap_summary.csv","ImageNet bootstrap")]:
        chk(f"phaseA {key}: {f} present", (R/f).exists(), "missing")
phaseA_checks()
n_fail = sum(1 for _,ok,_ in checks if not ok)
for name, ok, det in checks[-11:]: print(("PASS" if ok else "FAIL"), name, ("| "+det if det and not ok else ""))
print(f"[phaseA re-total] {len(checks)-n_fail}/{len(checks)}")


# ---------- Phase B: the census of record and the numbers stated in the main text ----------
def phaseB_checks():
    import numpy as np
    def bh_(p):
        p = np.asarray(p, dtype=float); n = len(p); order = np.argsort(p); ranked = p[order]*n/np.arange(1, n+1)
        adj = np.minimum.accumulate(ranked[::-1])[::-1]; out = np.empty(n); out[order] = np.minimum(adj, 1.0); return out
    rec = load("expR52_census_haar_p999_200.csv"); G = {(r["model"],r["dataset"]): str(r["genuine_bh"])=="True" for r in rec}
    X = {(r["model"],r["dataset"]): float(r["excess"]) for r in rec}
    TOP = {"imagenet","cifar100"}
    chk("record: 70/72 sign-neg, 49/72 genuine, 18/24 IN+C100", sum(v<0 for v in X.values())==70 and sum(G.values())==49 and sum(v for (m,d),v in G.items() if d in TOP)==18)
    chk("record: every family genuine somewhere; ViT-T genuine only on DTD", all(any(G[(m,d)] for d in ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"]) for m in set(m for m,_ in G))
        and [d for d in ["imagenet","cifar100","cifar10","dtd","fashionmnist","mnist"] if G[("i21k_t",d)]]==["dtd"])
    chk("record: non-genuine IN/C100 = ViT-T (both), CLIP-B IN, SigLIP-B IN, ViT-S/B C100",
        {k for k,v in G.items() if k[1] in TOP and not v}=={("i21k_t","imagenet"),("i21k_t","cifar100"),("clip_b","imagenet"),("siglip_b","imagenet"),("i21k_s","cifar100"),("i21k_b","cifar100")})
    vit=[X[(m,"imagenet")] for m in ["i21k_s","i21k_b","i21k_l"]]; dn=[X[(m,"imagenet")] for m in ["dinov1_b","dinov2_s","dinov2_b","dinov2_l","dinov2_g"]]
    chk("record IN ranges: ViT-S/B/L -0.012..-0.016, DINO/DINOv2 -0.008..-0.020, CLIP-L -0.009", abs(max(vit)+0.012)<0.0015 and abs(min(vit)+0.016)<0.0015
        and abs(max(dn)+0.008)<0.0015 and abs(min(dn)+0.020)<0.0015 and abs(X[("clip_l","imagenet")]+0.009)<0.0015)
    chk("record: DINOv2 S->G CIFAR-10 -0.071->-0.130, CIFAR-100 -0.031->-0.040, both monotone",
        abs(X[("dinov2_s","cifar10")]+0.071)<0.0015 and abs(X[("dinov2_g","cifar10")]+0.130)<0.0015 and abs(X[("dinov2_s","cifar100")]+0.031)<0.0015 and abs(X[("dinov2_g","cifar100")]+0.040)<0.0015
        and all(X[(b,ds)]<X[(a,ds)] for ds in ("cifar10","cifar100") for a,b in [("dinov2_s","dinov2_b"),("dinov2_b","dinov2_l"),("dinov2_l","dinov2_g")]))
    four = {"Hp":"expR52_census_haar_p999_200.csv","Hs":"expR54_census_haar_sup_200.csv","Gp":"expR40b_p999census200.csv","Gs":"expR39c_census200_cache.csv"}
    V = {}
    for tag,f in four.items():
        rows=load(f); pb=bh_([float(r["p_left"]) for r in rows]); V[tag]={(r["model"],r["dataset"]):(pb[i]<=0.05, int(r["r_above"])) for i,r in enumerate(rows)}
    chk("DINOv2-S/B/G ImageNet: genuine under both p99.9, r<=1 under both supremum", all(V[t][(m,"imagenet")][0] for t in ("Hp","Gp") for m in ["dinov2_s","dinov2_b","dinov2_g"])
        and all(V[t][(m,"imagenet")][1]<=1 for t in ("Hs","Gs") for m in ["dinov2_s","dinov2_b","dinov2_g"]))
    chk("2x2 genuine counts 49/46/52/52", [sum(v[0] for v in V[t].values()) for t in ("Hp","Hs","Gp","Gs")]==[49,46,52,52])
    bs = load("expR59_imagenet_bootstrap_summary.csv"); chk("bootstrap: excess s.d. <= 0.001 (3dp), all resamples negative", max(float(r["excess_boot_sd"]) for r in bs)<0.0015 and all(float(r["frac_boot_negative"])==1.0 for r in bs))
    cv = load("expR57_census_cosine_haar_p999_200.csv"); GC={(r["model"],r["dataset"]): str(r["genuine_bh"])=="True" for r in cv}
    chk("cosine: 56/72 genuine, 22/24 IN+C100, agreement 65/72, DINOv2 IN genuine", sum(GC.values())==56 and sum(v for k,v in GC.items() if k[1] in TOP)==22
        and sum(GC[k]==G[k] for k in GC)==65 and all(GC[(m,"imagenet")] for m in ["dinov2_s","dinov2_b","dinov2_l","dinov2_g"]))
    tx = {r["model"]: r for r in load("expR53_text_haar_p999_200.csv")}
    chk("text record: 7/15 genuine; GPT-2 S p~0.02 genuine, M p~0.15 not; L/XL, Pythia x3, OLMo-1B genuine; embedders +0.002..+0.004",
        sum(str(r["genuine_bh"])=="True" for r in tx.values())==7 and abs(float(tx["gpt2"]["p_left"])-0.020)<0.002 and abs(float(tx["gpt2_m"]["p_left"])-0.154)<0.002
        and all(str(tx[m]["genuine_bh"])=="True" for m in ["gpt2","gpt2_l","gpt2_xl","pythia_410m","pythia_1b","pythia_2b8","olmo_1b"])
        and all(0.0015<float(tx[m]["excess"])<0.0045 for m in ["bge_base","bge_large","gte_base","gte_large","gte_qwen2","e5_base","e5_large"]))
    tfour = ["expR53_text_haar_p999_200.csv","expR53_text_haar_sup_200.csv","expR53_text_gauss_p999_200.csv","expR48b_text_census200_bs1.csv","expR57_text_cosine_haar_p999_200.csv"]
    ok=True
    for f in tfour:
        rows=load(f); pb=bh_([float(r["p_left"]) for r in rows]); gg={r["model"]: pb[i]<=0.05 for i,r in enumerate(rows)}
        ok = ok and all(gg[m] for m in ["gpt2_l","gpt2_xl","pythia_410m","pythia_1b","pythia_2b8","olmo_1b"])
    chk("text: GPT-2 L/XL, Pythia x3, OLMo-1B genuine under all four constructions and cosine", ok)
    tm = {(r["dataset"],r["metric"],r["linkage"]): r for r in load("expR58_treemap_cutfree_summary.csv")}
    na=tm[("imagenet","euclid","average")]; ca=tm[("imagenet","cosine","average")]
    chk("tree map: naive coph 0.36/0.80, triplets 0.47/0.74; cosine-average triplets 0.77/0.76, coph 0.48/0.78, ARI 0.38/0.48",
        abs(float(na["coph_corr_big_vs_block"])-0.36)<0.005 and abs(float(na["coph_corr_within_block"])-0.80)<0.005 and abs(float(na["triplet_agree_big_vs_block"])-0.47)<0.005 and abs(float(na["triplet_agree_within_block"])-0.74)<0.005
        and abs(float(ca["triplet_agree_big_vs_block"])-0.77)<0.005 and abs(float(ca["triplet_agree_within_block"])-0.76)<0.005 and abs(float(ca["coph_corr_big_vs_block"])-0.48)<0.005 and abs(float(ca["coph_corr_within_block"])-0.78)<0.005
        and abs(float(ca["ari_cut_big_vs_block"])-0.38)<0.01 and abs(float(ca["ari_cut_within_block"])-0.48)<0.01)
    # depth test (B3): leaf-frame power sweep against the pre-set bar, and the anisotropic real-data readings
    import json as _json
    D = _json.load(open(R/"phaseB_depth_decision.json"))
    lf = load("expR55b_depth_power_leafframe.csv"); hz=[r for r in lf if r["level"]!="star"]; sz=[r for r in lf if r["level"]=="star"]
    pw = {(n,lv): np.mean([float(r["z"])<=-2 for r in hz if int(r["n"])==n and r["level"]==lv and float(r["ratio"])<=0.3]) for n in (100,1000) for lv in ("hier2","hier3")}
    fa_neg = np.mean([float(r["z"])<=-2 for r in sz]); fa_pos = np.mean([float(r["z"])>=2 for r in sz])
    fa100 = np.mean([float(r["z"])<=-2 for r in sz if int(r["n"])==100 and int(r["K"])>=12]); fa1000 = np.mean([float(r["z"])<=-2 for r in sz if int(r["n"])==1000])
    chk("depth power (leaf frame): 600 runs; power 1.00 at ratio<=0.3 for hier2/hier3, both n", len(lf)==600 and all(v==1.0 for v in pw.values()))
    chk("depth false alarms: z>=+2 never; z<=-2 6.2% pooled, 17% at n=100 K>=12, 0% at n=1000 -> pooled bar failed (decision JSON)",
        fa_pos==0 and abs(fa_neg-0.0625)<0.001 and abs(fa100-1/6)<0.001 and fa1000==0 and D["validated"] is False and abs(D["fa_neg"]-fa_neg)<1e-9)
    # regime-scoped certification (final pass): n=1000 meets the bar; n=100 K=6 clean, K>=12 not; ImageNet hits = ViT-S/B/L + DINOv2-L
    pw1000 = {r: np.mean([float(x["z"])<=-2 for x in hz if int(x["n"])==1000 and float(x["ratio"])==r]) for r in (0.1,0.3,0.6)}
    fa100_6 = np.mean([float(r["z"])<=-2 for r in sz if int(r["n"])==100 and int(r["K"])==6]); fa1000_pos = np.mean([float(r["z"])>=2 for r in sz if int(r["n"])==1000])
    G = _json.load(open(R/"phaseB_depth_regime.json"))
    zin = {r["model"]: float(r["z_depth"]) for r in load("expR56_depth_variants.csv") if r["variant"]=="aniso" and r["dataset"]=="imagenet" and int(r["K"])==30}; hits = sorted(m for m,v in zin.items() if v<=-2)
    chk("depth regime: n=1000 power 1.00/1.00/0.90 at ratio .1/.3/.6, fa 0/0; n=100 K=6 fa 0%; ImageNet z<=-2 = ViT-S/B/L + DINOv2-L, z -2.4..-4.0",
        pw1000[0.1]==1.0 and pw1000[0.3]==1.0 and abs(pw1000[0.6]-0.90)<0.011 and fa1000==0 and fa1000_pos==0 and fa100_6==0
        and hits==["dinov2_l","i21k_b","i21k_l","i21k_s"] and abs(max(zin[m] for m in hits)+2.4)<0.05 and abs(min(zin[m] for m in hits)+4.0)<0.05
        and G["names"]==["ViT-S","ViT-B","ViT-L","DINOv2-L"] and abs(G["pw06"]-0.90)<0.011)
    dv = load("expR56_depth_variants.csv"); an={(r["model"],r["dataset"],int(r["K"])): float(r["z_depth"]) for r in dv if r["variant"]=="aniso"}; iso={(r["model"],r["dataset"],int(r["K"])): float(r["z_depth"]) for r in dv if r["variant"]=="iso"}
    chk("depth real (aniso): 4/12 IN K=30 and 3/12 C100 K=20 with z<=-2, none z>=+2; iso C100 K=20: 8/12 z>=+2, max +6.8",
        sum(v<=-2 for (m,d,K),v in an.items() if d=="imagenet" and K==30)==4 and sum(v<=-2 for (m,d,K),v in an.items() if d=="cifar100" and K==20)==3
        and not any(v>=2 for (m,d,K),v in an.items() if (d,K) in {("imagenet",30),("cifar100",20)})
        and sum(v>=2 for (m,d,K),v in iso.items() if d=="cifar100" and K==20)==8 and abs(max(v for (m,d,K),v in iso.items() if d=="cifar100" and K==20)-6.8)<0.05)
    tp = load("expR55_depth_power.csv"); chk("depth top frame (iso star): power 0.00, star false alarms 16%", all(float(r["z"])>-2 for r in tp if r["level"]!="star") and abs(np.mean([abs(float(r["z"]))>=2 for r in tp if r["level"]=="star"])-0.1625)<0.001)
phaseB_checks()
# ---------- Restructuring reruns R7 (expR62 sample-level) and R8 (expR63 MERU) against the memo ----------
def r7r8_checks():
    import json as _j, numpy as np
    if not (R/"phaseC_memo.json").exists(): return
    M = _j.load(open(R/"phaseC_memo.json")); s = load("expR62_samplelevel_record.csv"); m = load("expR63_meru_record.csv")
    gen = [r for r in s if str(r["genuine_bh"])=="True"]
    chk("R7 sample-level: 24 cells, 200 replicates each; genuine count and names match the memo; raw supremum band matches",
        len(s)==24 and all(int(r["r_above"])<=200 for r in s) and len(gen)==M["sl_genuine"] and abs(min(float(r["delta_sup"]) for r in s)-M["sl_sup_lo"])<1e-9 and abs(max(float(r["delta_sup"]) for r in s)-M["sl_sup_hi"])<1e-9)
    chk("R7 sample-level: the reading stays within null noise in most cells (genuine BH <= 12 of 24; memo stop condition (a) false)",
        M["sl_genuine"]<=12 and M["stop_a_samplelevel_mostly_genuine"] is False and M["sl_within"]==24-M["sl_genuine"])
    im = {(r["model"]): r for r in m if r["dataset"]=="imagenet" and r["modality"]=="image"}
    chk("R8 MERU: 24 cells; ImageNet-image excess ranges, depth-z ranges and native-vs-Euclidean gap match the memo",
        len(m)==24 and abs(min(float(im[k]["excess"]) for k in im if k.startswith("meru"))-M["meru_in_exc_range"][0])<1e-9 and abs(max(float(im[k]["excess"]) for k in im if k.startswith("clip"))-M["clip_in_exc_range"][1])<1e-9
        and abs(min(float(im[k]["z_depth"]) for k in im if k.startswith("meru"))-M["meru_in_z_range"][0])<1e-9 and abs(max(float(im[k]["z_depth"]) for k in im if k.startswith("clip"))-M["clip_in_z_range"][1])<1e-9
        and abs(max(abs(float(im[k]["delta_999_native"])-float(im[k]["delta_999"])) for k in im if k.startswith("meru"))-M["meru_in_nat_vs_euc_maxgap"])<1e-9)
    chk("R8 MERU: no MERU model is more hierarchical than its twin on ImageNet (memo stop condition (b) false)",
        M["stop_b_meru_depth_beyond_twin"] is False and all(not (float(im[f"meru_{sz}"]["z_depth"])<=-2 and float(im[f"clip_{sz}"]["z_depth"])>-2) for sz in ("s","b","l") if f"meru_{sz}" in im))
r7r8_checks()
# ---------- Restructured main text: the numbers it states vs the files ----------
def phaseC_text_checks():
    import json as _j, numpy as np
    if not (R/"phaseC_memo.json").exists(): return
    TEX = Path(__file__).resolve().parents[2]/"ICLR2027"/"iclr2027"
    T = open(TEX/"main_iclr2027.tex").read(); main = T[:T.index("\\appendix")]
    b17 = open(TEX/"appendix_tables"/"tab_q03_sample.tex").read(); b30 = open(TEX/"appendix_tables"/"tab_q04_depth.tex").read(); q09 = open(TEX/"appendix_tables"/"tab_q09_corollary.tex").read()
    M = _j.load(open(R/"phaseC_memo.json"))
    chk("text: sample-level verdict in the prose ('not genuine in most cells', 24-cell count in the Table 1 caption); names, band and max excess in the sample-level table caption", "it is not genuine in most cells" in main and M['sl_within'] > 12
        and f"is genuine in {M['sl_genuine']} of 24 cells" in open(TEX/"tab_census.tex").read() and f"not genuine in {M['sl_within']} of 24 cells and genuine in {M['sl_genuine']}" in b17 and "DINOv2-S, DINOv2-B, DINOv2-L, DINOv2-G, SigLIP-B on CIFAR-100; ViT-S, ViT-B, ViT-L, DINOv2-L, DINOv2-G on DTD" in b17
        and f"{M['sl_sup_lo']:.3f}--{M['sl_sup_hi']:.3f}" in b17 and f"${M['sl_exc_lo']:+.3f}$" in b17)
    chk("text: MERU ranges (ImageNet-image excess, native gap, depth z) live in the B30 caption and match the memo",
        f"from ${M['meru_in_exc_range'][1]:+.3f}$ to ${M['meru_in_exc_range'][0]:+.3f}$ for MERU" in b30 and f"by at most ${M['meru_in_nat_vs_euc_maxgap']:.4f}$" in b30
        and f"from ${M['meru_in_z_range'][0]:+.1f}$ to ${M['meru_in_z_range'][1]:+.1f}$ for MERU" in b30 and f"from ${M['clip_in_z_range'][0]:+.1f}$ to ${M['clip_in_z_range'][1]:+.1f}$ for CLIP" in b30)
    b = load("exp2b_normalized_stack.csv"); H = [r for r in b if r["dataset"] in ("imagenet","cifar100","cifar10","dtd")]; Ht = [r for r in b if r["dataset"] in ("cifar100","cifar10","dtd")]
    gc = [100*float(r["FS_HN_COS_diff"]) for r in Ht if r["paradigm"].lower().startswith("contr")]; go = [100*float(r["FS_HN_COS_diff"]) for r in H if not r["paradigm"].lower().startswith("contr")]
    chk("text: Poincare-over-cosine gain of the contrastive VLMs on the transfer sets as stated; the other families' range (inconsistent in sign) lives in the corollary table", f"adds from ${min(gc):+.1f}$ to ${max(gc):+.1f}$ pp over cosine for the contrastive VLMs on the transfer sets" in main
        and min(go) < 0 < max(go) and f"{min(go):+.2f}" in q09 and f"{max(go):+.2f}" in q09)
    e1 = load("exp1_delta_controls.csv"); ga = sorted({int(r["d"]): float(r["delta_max"]) for r in e1 if r["variant"]=="gauss"}.items())
    c_lo, c_hi = (0.144/(2*ga[0][1]))**2, (0.144/(2*ga[-1][1]))**2
    chk("text: Khrulkov curvature on the Gaussian band (c=(0.144/delta_rel)^2, delta_rel=2 delta_norm; exp1 gauss d=192/1536)", f"${c_lo:.2f}$ at $d{{=}}{ga[0][0]}$ to ${c_hi:.1f}$ at $d{{=}}{ga[-1][0]}$" in main)
    imr = [r for r in load("expR52_census_haar_p999_200.csv") if r["dataset"]=="imagenet"]; fim = [abs(float(r["excess"])/float(r["null_mean"])) for r in imr]
    chk("text: ImageNet class-level excess as a fraction of the null reading as stated (S6), every ImageNet cell sign-negative; the absolute range lives in Table 1", f"it removes {100*min(fim):.0f} to {100*max(fim):.0f} per cent of the null reading" in main and all(float(r["excess"]) < 0 for r in imr)
        and f"${max(float(r['excess']) for r in imr):+.3f}" in open(TEX/"tab_census.tex").read())
    D = _j.load(open(R/"phaseC_fig2b.json")); chk("fig2b: decision recorded and consistent with the caption (ViT-T image features vs SigLIP-B centroids)", (D["mode"]=="sample") == ("on ViT-T image features the null sits at the reading, on SigLIP-B centroids far above it" in main)
        and D["sample_cell"]==["i21k_t","dtd"] and D["class_cell"]==["siglip_b","cifar10"] and (D["gap_sample"] <= D["gap_bge"] if D["mode"]=="sample" else True))
    body_ = T[T.index("\\begin{abstract}"):T.index("\\subsubsection*{Ethics Statement}")]   # the ICLR AI-use statement says 'generative AI tools' in its required form
    for w in ("tool", "we believe", "nterestingly"): chk(f"text: no '{w}' in the main text", w not in body_)
phaseC_text_checks()

# ---------- Prose pass (style of Groger et al.): metrics per paragraph, fixed vocabulary, thesis x5, numbers preserved ----------
def prose_checks():
    import re as _re
    TEX = Path(__file__).resolve().parents[2]/"ICLR2027"/"iclr2027"
    T = open(TEX/"main_iclr2027.tex").read(); body = T[T.index("\\begin{abstract}"):T.index("\\subsubsection*{Ethics Statement}")]
    THESIS = "Read correctly, foundation models organize classes into clustered structure that is occasionally hierarchical and moderately shared; they do not converge to one common tree, and their raw tree-likeness is not evidence for hyperbolic geometry."
    SHORT = "Clustered, occasionally hierarchical, moderately shared: not one common tree, and no license for curvature."
    chk("prose: the thesis appears verbatim exactly twice (abstract's last sentence, first paragraph of S7), no short form anywhere, no page-1 box", body.count(THESIS) == 2 and SHORT not in body and "tcolorbox" not in T
        and _re.sub(r"(?<!\\)%.*$", "", body[:body.index("\\end{abstract}")].rstrip().split("\n")[-1]).rstrip().endswith(THESIS) and THESIS in body[body.index("\\paragraph{What the paper establishes.}"):body.index("\\paragraph{Open questions.}")])
    # strip floats, comments, the enumerate; keep section markers
    src = _re.sub(r"(?m)(?<!\\)%.*$", "", body)
    src = _re.sub(r"\\begin\{(figure|table|tcolorbox)\}.*?\\end\{\1\}", "", src, flags=_re.S); src = _re.sub(r"\\input\{[^}]*\}", "", src)
    secs = _re.split(r"\\section\{([^}]*)\}", src); secs = [("Abstract", secs[0])] + [(secs[i], secs[i+1]) for i in range(1, len(secs), 2)]
    def clean(par):
        p = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", par); p = _re.sub(r"\\(S)?\\?ref\{[^}]*\}", "REF", p); p = _re.sub(r"\\label\{[^}]*\}", "", p)
        p = _re.sub(r"\$([A-Za-z])\{=\}(\d+)\$", r"\1=\2", p)    # $d{=}192$ is a parameter value, not a formula
        p = _re.sub(r"\$([^$]*)\$", lambda m: " FORMULA " if "=" in m.group(1) else m.group(0), p)   # inline equations (definitions) are not result numbers
        return p
    GROUP = r"(?<![A-Za-z\-^_{\d.])[-+]?\d+(?:\.\d+)?(?![A-Za-z\-\d])"   # digits glued to names (CIFAR-100, GPT-2, DINOv2, L2, 5-way) are not numbers
    RANGE = _re.compile(rf"(?:from\s+)?\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?(?:\s*(?:of the|of|to|against|vs|and|--)\s+\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?){{0,2}}(?:\\%)?")   # up to two connectors: '42 and 47 of the 72' is one group
    def groups(s):
        s = clean(s).replace("{=}", "="); s = _re.sub(r"\\begin\{enumerate\}.*?\\end\{enumerate\}", " ", s, flags=_re.S); s = _re.sub(r"\\[a-zA-Z]+", " ", s)
        return [m.group(0) for m in RANGE.finditer(s) if not _re.fullmatch(r"\s*", m.group(0))]
    BANNED = ["tool", "beyond-null", "tree-like structure", "hierarchical structure", "the form ", "reading of record", "our approach", "the method", "essentially", "largely", "substantially", "somewhat", "nterestingly", "notably", "importantly", "we believe", "we note"]
    ALLOW17 = {"49 of 72", "18 of 24", "4 of 12", "49 of the 72"}
    bad = []
    for name, text in secs:
        sec_no = {"Introduction": 1, "Related Work and Background": 2, "The Instrument": 3, "Findings I: Latent Hyperbolicity, Calibrated": 4, "Findings II: Whose Tree": 5, "Consequences for Imposing Curvature": 6, "Discussion and Limitations": 7}.get(name, 0)
        for par in [q.strip() for q in _re.split(r"\n\s*\n", text) if q.strip() and not q.strip().startswith(("\\begin{enumerate}", "\\end{enumerate}", "\\item", "\\end{abstract}"))]:
            if par.startswith("\\label") or par.startswith("\\item"): continue
            head = par[:60].replace("\n", " ")
            gs = groups(par)
            if sec_no in (4, 5, 6):
                if len(gs) > 2: bad.append(f"S{sec_no} >2 numbers {gs}: {head}")
                for sent in _re.split(r"(?<=[.!?])\s+", clean(par)):
                    sg = groups(sent)
                    if len(sg) > 1: bad.append(f"S{sec_no} >1 number per sentence {sg}: {sent[:80]}")
                if ";" in par.replace(THESIS, "") and "\\paragraph{Limitations" not in par: bad.append(f"S{sec_no} semicolon: {head}")
            if sec_no in (1, 7):
                extra = [g for g in gs if g.strip() not in ALLOW17]
                if extra: bad.append(f"S{sec_no} numbers beyond the allowed three {extra}: {head}")
            n_par = len(_re.findall(r"\((?!(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|[a-c])\))", clean(par)))   # (i)...(viii) and (a)-(c) are enumeration marks
            if sec_no in (1, 3, 4, 5, 6, 7) and n_par > 1: bad.append(f"S{sec_no} >1 parenthetical: {head}")
            low = clean(par).lower()
            for w in BANNED:
                if w in low: bad.append(f"S{sec_no} banned '{w}': {head}")
    for line in bad: print("   PROSE:", line[:200])
    chk("prose: <=2 number groups per paragraph and <=1 per sentence in S4-S6; only 49/72, 18/24, 4/12 in S1 and S7; <=1 parenthetical; no semicolons in S4-S6; no banned words", not bad)
    # every number that left the prose still lives in the paper (main text, appendix prose or generated tables/captions)
    old = open(R/"phaseD_old_main_body.tex").read(); old = _re.sub(r"(?m)(?<!\\)%.*$", "", old); old = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", old); old = _re.sub(r"\\(S)?\\?ref\{[^}]*\}", "", old); old = _re.sub(r"\\label\{[^}]*\}", "", old)
    new_all = T + "".join(open(f).read() for f in list((TEX/"appendix_tables").glob("*.tex")) + [TEX/"tab_census.tex"])
    new_nums = [float(x) for x in _re.findall(r"(?<![\w.])[-+]?\d+\.\d+|(?<![\w.])\d+(?![\w.])", new_all.replace("{=}", "="))]
    missing = []
    for tok in sorted(set(_re.findall(r"[-+]?\d+\.\d+", old))):
        v = float(tok); prec = len(tok.split(".")[1]); tol = 0.5 * 10 ** (-prec) + 1e-12
        if tok in new_all: continue
        if any(abs(abs(v) - abs(u)) <= tol or (abs(v) > 0 and abs(round(u, prec) - abs(v)) <= tol) for u in new_nums): continue
        if tok in ("0.11",) and "-0.103" in new_all: continue     # the mid star's Haar excess, stated approx in the old prose, is -0.103 in Table B25
        missing.append(tok)
    for tok in missing: print("   NUMBER LOST:", tok)
    chk("prose: no number that left the main text disappeared from the paper (tables, captions or appendix prose keep it)", not missing)
prose_checks()

# ---------- Positive-control pass (Phase A): R9 = expR64 implanted depth, R11 = expR66 joint sensitivity, R10 = expR65 (optional) ----------
def positive_control_checks():
    import json as _j, numpy as np
    if not (R/"positive_control_memo.json").exists(): return
    M = _j.load(open(R/"positive_control_memo.json"))
    D = load("expR64_implanted_depth.csv"); dep = [r for r in D if r["kind"]=="depth" and r["partition"]=="rand6" and r["s"]!="real"]
    chk("R9: 12 backbones x 5 strengths x 5 seeds depth runs, every run n=1000 with the real hub RMS preserved (hub_rms equal across s per backbone)",
        len(dep)==300 and all(int(r["n"])==1000 for r in dep)
        and all(max(float(r["hub_rms"]) for r in dep if r["model"]==m)-min(float(r["hub_rms"]) for r in dep if r["model"]==m) < 1e-2*float(next(r["hub_rms"] for r in dep if r["model"]==m)) for m in {r["model"] for r in dep}))
    s0 = [float(r["z"]) for r in dep if float(r["s"])==0.0]; s1 = [float(r["z"]) for r in dep if float(r["s"])==1.0]
    chk("R9: no backbone declared hierarchical at s=0 (the memo's count), and the memo's s=1 verdict matches the file",
        sum(z<=-2 for z in s0)==0 and M["r9_s0_any_hit"]==0 and (all(z<=-2 for z in s1) == M["r9_s1_all_certified"]) and abs(max(s0)-M["r9_s0_max_z"])<1e-9)
    S = {r["model"]: r for r in load("expR64_implanted_depth_summary.csv")}
    chk("R9: s* per backbone in the memo equals the summary file; the untouched cloud reproduces Table B34 (|dz| <= 0.05)",
        all((S[m]["s_star"]=="" and v is None) or (S[m]["s_star"]!="" and abs(float(S[m]["s_star"])-v)<1e-9) for m, v in M["r9_s_star"].items()) and M["r9_real_z_match"] <= 0.05)
    cen = [r for r in D if r["kind"]=="census"]
    chk("R9: census excess on the implanted clouds (5 strengths x 12 backbones, 200 replicates): every cloud below its null mean, minimum rank equals the memo's",
        len(cen)==60 and all(float(r["excess"])<0 for r in cen) and min(int(float(r["r_above"])) for r in cen)==M["r9_census_r_min"])
    tg = [r for r in D if r["kind"]=="depth" and r["partition"]=="rand6_t06"]
    if tg: chk("R9 tight variant: 12 backbones x 3 strengths x 2 seeds; memo's per-backbone z and hit counts match the file",
        len(tg)==72 and all(abs(np.mean([float(r["z"]) for r in tg if r["model"]==m and float(r["s"])==1.0])-v["z_s1"])<1e-9 and sum(float(r["z"])<=-2 for r in tg if r["model"]==m and float(r["s"])==0.0)==v["hits_s0"] for m, v in M["r9_tight"].items()))
    J = load("expR66_joint_sensitivity_summary.csv"); top = lambda r: r["dataset"] in ("imagenet","cifar100")
    chk("R11: 72 cells x 30 resamples; record 49/72 & 18/24 reproduced from the summary; joint and bootstrap-BH counts equal the memo",
        len(J)==72 and all(int(r["n_boot"])==30 for r in J) and sum(r["genuine_bh_record"]=="True" for r in J)==49 and sum(r["genuine_bh_record"]=="True" and top(r) for r in J)==18
        and [sum(r["joint_genuine"]=="True" for r in J), sum(r["joint_genuine"]=="True" and top(r) for r in J)]==M["r11_joint"]
        and [sum(r["boot_bh_genuine"]=="True" for r in J), sum(r["boot_bh_genuine"]=="True" and top(r) for r in J)]==M["r11_bootbh"])
    zj = [float(r["excess"])/np.sqrt(float(r["sd_null"])**2+float(r["sd_boot"])**2+float(r["sd_est"])**2) for r in J]
    chk("R11: z_joint recomputed from the stored s.d.s matches the file", all(abs(z-float(r["z_joint"]))<1e-6 for z, r in zip(zj, J)))
    if M.get("r10"):
        F = {r["obj"]: r for r in load("expR65_hier_finetune.csv")}
        chk("R10: frozen/ce/hier rows present; memo values equal the file; frozen census excess equals the record cell to 3 dp",
            all(o in F for o in ("frozen","ce","hier")) and all(abs(float(F[o]["z_depth"])-M["r10"][o]["z"])<1e-9 and abs(float(F[o]["excess"])-M["r10"][o]["excess"])<1e-9 for o in F)
            and abs(float(F["frozen"]["excess"])-float(next(r["excess"] for r in load("expR52_census_haar_p999_200.csv") if r["model"]=="i21k_b" and r["dataset"]=="imagenet")))<0.0015)
positive_control_checks()

def positive_control_prose_checks():
    import json as _j
    TEX = Path(__file__).resolve().parents[2]/"ICLR2027"/"iclr2027"; T = open(TEX/"main_iclr2027.tex").read(); main = T[:T.index("\\appendix")]
    if not (R/"expR64b_wn30_summary.csv").exists() or "{{" in main: return
    S9 = load("expR64b_wn30_summary.csv"); dep = [r for r in load("expR64b_wn30.csv") if r["kind"]=="depth" and r["partition"]=="rand6" and r["s"]!="real"]
    ndet = sum(int(r["hits_s1"])>=4 for r in S9); fa0 = sum(float(r["z"])<=-2 for r in dep if float(r["s"])==0.0); n0 = sum(1 for r in dep if float(r["s"])==0.0)
    lo, hi = min(float(r["ratio_real"]) for r in S9), max(float(r["ratio_real"]) for r in S9)
    q05 = open(TEX/"appendix_tables"/"tab_q05_power.tex").read()
    chk("R9b prose: 'detected in x of 12', 'between one and four times' (range in the power-table caption) and 'none of the sixty' match expR64b (0 false alarms in 60)",
        (f"detected in {ndet} of 12 backbones" if ndet else "detected in none of the twelve backbones") in main and "is between one and four times their between-hub spread" in main and 1.0 <= lo and hi <= 4.0
        and f"({lo:.1f} to {hi:.1f} on the real clouds" in q05 and fa0==0 and n0==60 and "none of the sixty zero-strength runs" in main)
    D9 = load("expR64b_wn30.csv"); cert4 = ("i21k_s","i21k_b","i21k_l","dinov2_l")
    z0 = [float(r["z"]) for r in D9 if r["kind"]=="depth" and r["partition"]=="rand6" and r["s"]!="real" and r["model"] in cert4 and float(r["s"])==0.0]
    tg0 = {r["model"] for r in D9 if r["kind"]=="depth" and r["partition"]=="rand6_t06" and float(r["s"])==0.0 and float(r["z"])<=-2}
    chk("S4.4 sentence (1): the certified four raise no alarm at zero strength (0 of 20 in expR64b; the count is subsumed by the 0 of 60 stated for the Haar-hub star), and their real z is deeper than the implanted tree's",
        "raise no alarm in any zero-strength run, so the verdict comes from their real hub arrangement" in main and sum(z<=-2 for z in z0)==0 and len(z0)==20
        and all(float(r["real_z"]) < float(r["z_mean_s1"]) for r in S9 if r["model"] in cert4) and "the certified set shifts with the choice of frame" in main.lower())
    chk("S4.4 sentence (2): two contrastive backbones fire at zero strength in the shrunk variant, as stated", "two contrastive backbones also fire at zero strength, so the shrunk variant is a diagnostic of power, not a substitute test" in main
        and len(tg0)==2 and tg0 <= {"clip_b","clip_l","siglip_b"})
    J = load("expR66_joint_sensitivity_summary.csv"); top = lambda r: r["dataset"] in ("imagenet","cifar100")
    a, b = sum(r["joint_genuine"]=="True" for r in J), sum(r["boot_bh_genuine"]=="True" for r in J); at, bt = sum(r["joint_genuine"]=="True" and top(r) for r in J), sum(r["boot_bh_genuine"]=="True" and top(r) for r in J)
    chk("R11 prose: S4.3 says 'leaves most of the count in place' (true: joint and bootstrap counts > 36 of 72 and > 12 of 24) and the Table 1 caption carries the exact range", "leaves most of the count in place" in main and min(a,b) > 36 and min(at,bt) > 12 and f"{min(a,b)}--{max(a,b)} of 72" in open(TEX/"tab_census.tex").read())
    im = [r for r in load("expR52_census_haar_p999_200.csv") if r["dataset"]=="imagenet"]; u = [abs(float(r["excess"])/float(r["null_sd"])) for r in im]
    chk("B3 prose: the ImageNet excess is 'many times the null's own spread' (min ratio > 1) and stated as a fraction of the null in S6", min(u) > 1 and "many times the null's own spread" in main)
    chk("B5 prose: the plain-language gloss of the two nulls appears in S3 and in the nulls-table caption", T.count("the spectrum null is a cloud with the same shape as the real one and no structure inside it") == 2)
    chk("wording: 'no additional depth' replaced by 'no detected depth' everywhere in the main text", "additional depth" not in main and main.count("no detected depth") >= 3)
    if (R/"expR65_hier_finetune.csv").exists(): chk("R10 in the appendix as an inconclusive control with the file's z values", all(f"${float(r['z_depth']):+.2f}$" in T[T.index("\\appendix"):] for r in load("expR65_hier_finetune.csv")))
positive_control_prose_checks()

# ---------- Final pass (Phase A): A1 Haar-hub star, A2 IN-1k supervised ViTs, A3 corollary on the excess, A4 MERU radii, A5 budget under the record, A6 normalized effect size ----------
def final_pass_checks():
    import json as _j, numpy as np
    if not (R/"final_pass_memo.json").exists(): return
    M = _j.load(open(R/"final_pass_memo.json"))
    if "a1" in M:
        S = load("expR69_depth_haarhubs_summary.csv"); D = load("expR69_depth_haarhubs.csv")
        chk("A1: 12 backbones x (2 real stars + 10 implanted) depth runs; certified sets and implant counts equal the memo; star rank in 0..10",
            len(D)==12*12 and [r["model"] for r in S if r["cert_gauss"]=="True"]==M["a1"]["cert_gauss"] and [r["model"] for r in S if r["cert_haar"]=="True"]==M["a1"]["cert_haar"]
            and sum(int(r["fa_s0_haar"]) for r in S)==M["a1"]["fa_s0"] and sum(int(r["hits_s1_haar"]) for r in S)==M["a1"]["hits_s1"] and all(0<=int(r["r_star"])<=10 for r in D))
        b34 = {r["model"]: float(r["z_depth"]) for r in load("expR56_depth_variants.csv") if r["dataset"]=="imagenet" and int(r["K"])==30 and r["variant"]=="aniso"}
        chk("A1: the Gaussian-star column reproduces Table B34 (|dz| <= 0.05)", all(abs(float(r["z_gauss"])-b34[r["model"]])<=0.05 for r in S))
    if "a2" in M:
        T = {r["model"]: r for r in load("expR70_inet1k_supervised.csv")}
        chk("A2: the IN-1k supervised rows exist with 200-replicate censuses; memo values equal the file; the i21k_b side-by-side reproduces the record excess to 3 dp",
            all(m in T for m in ("deit_b","i21k_b")) and all(abs(float(T[m]["excess_in"])-M["a2"][m]["excess_in"])<1e-4 and abs(float(T[m]["z_gauss"])-M["a2"][m]["z_gauss"])<0.01 for m in M["a2"])
            and abs(float(T["i21k_b"]["excess_in"])-float(next(r["excess"] for r in load("expR52_census_haar_p999_200.csv") if r["model"]=="i21k_b" and r["dataset"]=="imagenet")))<0.0015)
    C = load("expR68_corollary_excess.csv")
    raw_in = float(next(r["r"] for r in C if r["predictor"]=="raw" and r["gain"]=="NC_adv" and r["dataset"]=="imagenet"))
    old_in = float(next(r["r"] for r in load("night/correlation_cis.csv") if r["task"]=="NC_adv" and r["dataset"]=="imagenet"))
    chk("A3: correlations on raw/excess/depth for 3 gains x 4 datasets (+pooled); the raw ImageNet NC correlation reproduces Table B5 (|dr| <= 0.02); memo survival flags equal the file",
        len(C) >= 39 and all(r["predictor"] in ("raw","excess","depth_z") for r in C) and abs(raw_in-old_in) <= 0.02
        and all(M["a3_nc_excess_survives"][ds]==(float(next(r["ci_hi"] for r in C if r["predictor"]=="excess" and r["gain"]=="NC_adv" and r["dataset"]==ds))<0) for ds in ("imagenet","cifar100","cifar10","dtd")))
    Rm = load("expR71_meru_radii.csv")
    chk("A4: MERU radii: six clouds, one curvature, memo ranges equal the file, Lorentz/Euclidean ratio within 1% of one",
        len(Rm)==6 and len({r["curv"] for r in Rm})==1 and abs(max(float(r["radius_sqrtc_p95"]) for r in Rm)-M["a4"]["radius_p95"][1])<1e-9 and all(0.99<float(r["lorentz_over_euclid_median"])<=1.0 for r in Rm))
    if "a5" in M:
        B = load("expR72_budget_record_summary.csv"); Bd = load("expR72_budget_record.csv")
        chk("A5: 9 cells x 6 budgets under the record; ImageNet drift over s.d. equals the memo and decides 'budget-stable'",
            len(Bd)==54 and len(B)==9 and abs(max(float(r["drift_ge1e5_over_sd"]) for r in B if r["dataset"]=="imagenet")-M["a5"]["imagenet"]["max_drift_over_sd"])<1e-9 and M["a5_budget_stable_imagenet"]==(M["a5"]["imagenet"]["max_drift_over_sd"]<1.0))
    c52 = load("expR52_census_haar_p999_200.csv"); fr = {(r["model"],r["dataset"]): float(r["excess"])/float(r["null_mean"]) for r in c52}
    chk("A6: normalized effect sizes in the memo equal the file (ImageNet range, DINOv2-G/CIFAR-100)", abs(min(v for k,v in fr.items() if k[1]=="imagenet")-M["a6"]["imagenet_frac"][0])<1e-9 and abs(fr[("dinov2_g","cifar100")]-M["a6"]["dinov2g_c100_frac"])<1e-9)
final_pass_checks()
# ---------- Final pass (Phase B): abstract, thesis, bridges, numbers in prose, captions, figures, the consolidated appendix ----------
def final_pass_prose_checks():
    import re as _re, json as _j, subprocess, importlib.util
    TEX = Path(__file__).resolve().parents[2]/"ICLR2027"/"iclr2027"; ROOT = Path(__file__).resolve().parents[2]
    T = open(TEX/"main_iclr2027.tex").read(); main = T[:T.index("\\appendix")]; body = T[T.index("\\begin{abstract}"):T.index("\\subsubsection*{Ethics Statement}")]; app = T[T.index("\\appendix"):]
    nocom = lambda s: _re.sub(r"(?m)(?<!\\)%.*$", "", s)
    ABS = ("Hyperbolic methods for representation learning rest on a premise we call latent hyperbolicity: standard models are already tree-like, because their class geometry scores a low Gromov $\\delta$. We show that this raw reading is confounded by dimension, by covariance spectrum and by the supremum statistic itself, and we build an instrument that reads every score as an excess over a random cloud with the same dimension and spectrum, ranks it against 200 matched replicates, and adds a depth test whose false-alarm rate and power are measured on real clouds. Applied to 12 vision backbones, 6 datasets and 16 text models, the instrument gives a nuanced answer. On image features, where the premise is read, the excess sits within null noise in most cells and reaches at most 40\\% of the null where it survives. On class centroids, clustered structure is genuine in 49 of 72 cells, but a star already produces it. Hierarchy above the superclasses is certified in 4 of 12 ImageNet backbones, the same four under a star with the real hubs' spectrum, by a test that never fires on real clouds with randomized hubs; a backbone trained in hyperbolic space lives in the near-flat regime and shows the same clustering and no detected depth. The trees are moderately shared: a naive comparison isolates the DINOv2 family as an island, an artifact of the clustering cut; once the cut is controlled a gap of about a third remains, every training recipe recovers the human taxonomy partially, more so with supervision, and the self-supervised tree lives in angles rather than distances. Read correctly, foundation models organize classes into clustered structure that is occasionally hierarchical and moderately shared; they do not converge to one common tree, and their raw tree-likeness is not evidence for hyperbolic geometry.")
    abstract = nocom(body[body.index("\\begin{abstract}")+len("\\begin{abstract}"):body.index("\\end{abstract}")])
    chk("final: abstract verbatim as approved in the brief", " ".join(abstract.split()) == " ".join(ABS.split()))
    rec = load("expR52_census_haar_p999_200.csv"); G = {(r["model"],r["dataset"]): str(r["genuine_bh"])=="True" for r in rec}
    sl = load("expR62_samplelevel_record.csv"); fr = [abs(float(r["excess"])/float(r["null_mean"])) for r in sl if str(r["genuine_bh"])=="True"]
    H9 = load("expR69_depth_haarhubs_summary.csv"); cg = sorted(r["model"] for r in H9 if r["cert_gauss"]=="True"); ch = sorted(r["model"] for r in H9 if r["cert_haar"]=="True")
    dv = [r for r in load("expR56_depth_variants.csv") if r["dataset"]=="imagenet" and int(r["K"])==30 and r["variant"]=="aniso"]
    chk("final: abstract numbers traced (12 backbones, 6 datasets, 16 text models, 200 replicates, sample-level max fraction in (35%, 40%], 49 of 72, 4 of 12 = the same four under both stars, 0 false alarms with randomized hubs)",
        len({r["model"] for r in rec})==12 and len({r["dataset"] for r in rec})==6 and len(load("exp18_text_nulls.csv"))==16 and all(int(r["r_above"])<=200 for r in rec) and 0.35 < max(fr) <= 0.40
        and sum(G.values())==49 and sum(float(r["z_depth"])<=-2 for r in dv)==4 and cg==ch==sorted(r["model"] for r in dv if float(r["z_depth"])<=-2) and sum(int(r["fa_s0_haar"]) for r in H9)==0
        and sum(float(r["z"])<=-2 for r in load("expR64b_wn30.csv") if r["kind"]=="depth" and r["partition"]=="rand6" and r["s"]!="real" and float(r["s"])==0.0)==0, f"max sample fraction {max(fr):.3f}")
    gap = _j.load(open(R/"final_pass_island_gap.json"))
    chk("final: 'a gap of about a third' = median fraction of within-block agreement missing over the admissible ImageNet configurations and the three measures, in [0.25, 0.42]", 0.25 <= gap["median_missing_admissible_imagenet"] <= 0.42 and gap["n"] >= 9, f"{gap['median_missing_admissible_imagenet']:.2f}")
    Rm = load("expR71_meru_radii.csv"); chk("final: MERU near-flat ratio in S4 equals the file (min Lorentz/Euclidean median, 3 dp)", f"distance ratio of ${min(float(r['lorentz_over_euclid_median']) for r in Rm):.3f}$" in main)
    # ---- paragraph structure: bridges, numbers, sentences, parentheses
    src = nocom(body); src = _re.sub(r"\\begin\{(figure|table)\}.*?\\end\{\1\}", "", src, flags=_re.S); src = _re.sub(r"\\input\{[^}]*\}", "", src)
    secs = _re.split(r"\\section\{([^}]*)\}", src); secs = [(secs[i], secs[i+1]) for i in range(1, len(secs), 2)]
    SEC = {"Introduction": 1, "Related Work and Background": 2, "The Instrument": 3, "Findings I: Latent Hyperbolicity, Calibrated": 4, "Findings II: Whose Tree": 5, "Consequences for Imposing Curvature": 6, "Discussion and Limitations": 7}
    def clean(par):
        p = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", par); p = _re.sub(r"\\(S)?\\?ref\{[^}]*\}", "REF", p); p = _re.sub(r"\\label\{[^}]*\}", "", p)
        p = _re.sub(r"\$([A-Za-z])\{=\}(\d+)\$", r"\1=\2", p); p = _re.sub(r"\$([^$]*)\$", lambda m: " FORMULA " if ("=" in m.group(1) or "\\" in m.group(1)) else m.group(0), p)
        return p
    GROUP = r"(?<![A-Za-z\-^_{\d.])[-+]?\d+(?:\.\d+)?(?![A-Za-z\-\d])"
    RANGE = _re.compile(rf"(?:from\s+)?\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?(?:\s*(?:of the|of|to|against|vs|and|--)\s+\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?){{0,2}}(?:\\%)?")
    def groups(s):
        s = clean(s).replace("{=}", "="); s = _re.sub(r"\\begin\{enumerate\}.*?\\end\{enumerate\}", " ", s, flags=_re.S); s = _re.sub(r"\\[a-zA-Z]+", " ", s)
        return [m.group(0) for m in RANGE.finditer(s) if not _re.fullmatch(r"\s*", m.group(0))]
    MARK = ["next paragraph", "next question", "subject of the next", "turn to next", "take up last", "which we review", "builds that comparison", "the concern of Section", "First we ask", "last question of this section", "states the three acts", "With the instrument in place", "raises the question of", "which text makes explicit", "has to be re-read", "points to where", "is the first candidate", "What to read instead", "we turn to them next"]
    bridges = []; bad = []; total = 0; paras_by_sec = {}
    for name, text in secs:
        s = SEC.get(name, 0); pars = [q.strip() for q in _re.split(r"\n\s*\n", text) if q.strip() and not q.strip().startswith(("\\begin{enumerate}", "\\end{enumerate}", "\\item", "\\label"))]
        paras_by_sec[s] = pars
        for k, par in enumerate(pars):
            head = par[:50].replace("\n", " "); cp = clean(par)
            for mk in MARK:
                if mk.lower() in cp.lower(): bridges.append((s, k == len(pars)-1, mk, head))
            gs = groups(par); total += len(gs)
            if len(gs) > 2: bad.append(f"S{s} >2 numbers {gs}: {head}")
            for inner in _re.findall(r"\(([^()]*)\)", cp):
                if groups(inner): bad.append(f"S{s} number in parentheses ({inner[:40]}): {head}")
            if s in (3, 4, 5, 6):
                ns = len([x for x in _re.split(r"(?<=[.!?])\s+(?=[A-Z\\$(])", cp) if x.strip()])
                if ns > 9: bad.append(f"S{s} {ns} sentences: {head}")
    for line in bad: print("   FINAL:", line[:200])
    chk("final: <=25 number groups in the prose of S1-S7, <=2 per paragraph, none inside parentheses, <=9 sentences per paragraph in S3-S6", not bad and total <= 25, f"{total} number groups")
    want = {(3, "With the instrument in place"), (4, "turn to next"), (5, "subject of the next")}
    got = {(s, mk) for s, last, mk, h in bridges}
    for s, last, mk, h in bridges: print("   BRIDGE:", s, "last-paragraph" if last else "NOT last", mk, "|", h)
    chk("final: exactly three bridges (end of S3, S4, S5), in the last paragraph of their section, no two with the same wording", got == want and len(bridges) == 3 and all(last for _, last, _, _ in bridges))
    # ---- openings, wording, statements
    chk("final: S4 and S6 open with the two-sentence tension", "It finds one within null noise in most cells, and what survives lives on class centroids, where a star already passes the census." in main
        and "If the raw reading were evidence, a practitioner could set the curvature from it. It is not, so this section states" in main)
    nc = nocom(T)
    chk("final: no 'vanishes', 'closes entirely', 'self-supervised family as an island', 'pre-registered'/'pre-registration', 'Appendix B'-style table refs; the island is the DINOv2 family",
        all(w not in nc for w in ("vanishes", "closes entirely", "self-supervised family as an island", "pre-registered", "pre-registration", "the island vanishes")) and "isolates the DINOv2 family as an island" in nc)
    chk("final: GPT-2 sentences agree with the text census of record (S genuine at the margin under the record, M not, L/XL under every construction)", "GPT-2 S is genuine at the margin under the record" in main and "GPT-2 M is not genuine under the record" in main and "GPT-2 M sits at its matched null" in main
        and "GPT-2 L and XL are genuine under every construction of the null and statistic and under cosine" in main)
    caps = _re.findall(r"\\caption\{(.*?)\n?\}\n\\label", nocom(main), flags=_re.S)
    chk("final: every main-text caption is a bold takeaway followed by what is plotted", len(caps) >= 5 and all(c.lstrip().startswith("\\textbf{") for c in caps) and all(len(_re.sub(r"^\s*\\textbf\{[^}]*\}", "", c).strip()) > 40 for c in caps))
    chk("final: Table 1 in \\small without the sample-level columns; statements in the required form (AI use with the responsibility sentence; reproducibility with TODO(author))",
        "\\small" in open(TEX/"tab_census.tex").read() and "sample level (images)" not in open(TEX/"tab_census.tex").read()
        and "\\subsubsection*{AI Use Statement}" in main and "We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI." in main and "TODO(author)" in main)
    def height(f):
        out = subprocess.run(["pdfinfo", str(TEX/"figures"/f)], capture_output=True, text=True).stdout; m = _re.search(r"Page size:\s+([\d.]+) x ([\d.]+) pts", out); return float(m.group(2))/72 if m else 0.0
    chk("final: figure heights (PDF): Figure 2 >= 1.3 in, Figure 3 >= 1.4 in, Figure 4 >= 1.5 in; Figure 1 slot 1.4 in", height("fig_overview.pdf") >= 1.3 and height("fig_excess_panel.pdf") >= 1.4 and height("fig_depth_main.pdf") >= 1.5 and "[1.4in]" in main,
        f"{height('fig_overview.pdf'):.2f}/{height('fig_excess_panel.pdf'):.2f}/{height('fig_depth_main.pdf'):.2f}")
    # ---- the consolidated appendix
    tabfiles = sorted((TEX/"appendix_tables").glob("tab_*.tex")); qfiles = [f for f in tabfiles if f.name.startswith("tab_q")]
    inputs = _re.findall(r"\\input\{appendix_tables/(tab_[^}]*)\}", app)
    chk("final: fourteen consolidated tables plus the provenance index, no other table file, every file input once", len(qfiles)==14 and {f.stem for f in tabfiles} == set(inputs) and len(inputs)==len(set(inputs))==15 and "tab_z_provenance" in inputs)
    labels = {}
    for f in qfiles:
        m = _re.search(r"\\label\{(tab:q[^}]*)\}", f.read_text()); labels[f.stem] = m.group(1)
    cited_order = []
    for m in _re.finditer(r"\\ref\{(tab:q[^}]*)\}", main):
        if m.group(1) not in cited_order: cited_order.append(m.group(1))
    input_order = [labels[s] for s in inputs if s in labels]
    chk("final: the consolidated tables are numbered in the order the main text first cites them, and every table label is cited from the main text or another table", input_order[:len(cited_order)] == cited_order
        and all(("\\ref{"+lab+"}" in main) or any(("\\ref{"+lab+"}") in g.read_text() for g in qfiles if g.stem != s) for s, lab in labels.items()), f"cited {cited_order} | input {input_order}")
    alltex = nocom(T) + "".join(nocom(f.read_text()) for f in tabfiles) + nocom(open(TEX/"tab_census.tex").read())
    refs = set(_re.findall(r"\\(?:eq)?ref\{([^}]*)\}", alltex)); defs = set(_re.findall(r"\\label\{([^}]*)\}", alltex))
    chk("final: every cross-reference resolves to a label", refs <= defs, str(sorted(refs - defs)))
    chk("final: Table 13 (the 3-replicate original text extraction) removed and the record table carries no 'original extraction' columns", "tab_b1" not in T and "original extraction" not in open(TEX/"appendix_tables"/"tab_q02_text.tex").read())
    spec = importlib.util.spec_from_file_location("gaf", TEX/"gen_appendix_final.py"); gaf = importlib.util.module_from_spec(spec); spec.loader.exec_module(gaf)
    missing = gaf.conservation_check(verbose=False)
    for tok, srcs in missing: print("   NUMBER LOST (appendix):", tok, srcs)
    chk("final: every decimal token of the old appendix (44 tables, Table 13 excepted) survives in the consolidated tables or the paper", not missing)
    cl = open(ROOT/"ICLR2027"/"CHANGELOG_final.md").read().split("\n")[:6]
    chk("final: freeze note at the top of the CHANGELOG and the QA rasterizations present", any("Frozen 17 Sept 2026" in l for l in cl) and len(list((ROOT/"ICLR2027"/"qa_pages").glob("*.png"))) >= 9)
final_pass_prose_checks()
# ---------- Parallel version v2 (main_iclr2027_v2.tex): same numbers, same files, the v1 prose rules, the v2 skeleton ----------
def v2_checks():
    import re as _re
    TEX = Path(__file__).resolve().parents[2]/"ICLR2027"/"iclr2027"; P2 = TEX/"main_iclr2027_v2.tex"
    if not P2.exists(): chk("v2: main_iclr2027_v2.tex present", False, "missing"); return
    T2 = open(P2).read(); T1 = open(TEX/"main_iclr2027.tex").read()
    nocom = lambda s: _re.sub(r"(?m)(?<!\\)%.*$", "", s)
    b2 = T2[T2.index("\\begin{abstract}"):T2.index("\\subsubsection*{Ethics Statement}")]; b1 = T1[T1.index("\\begin{abstract}"):T1.index("\\subsubsection*{Ethics Statement}")]
    a2 = nocom(b2[:b2.index("\\end{abstract}")]); a1 = nocom(b1[:b1.index("\\end{abstract}")])
    chk("v2: abstract identical to v1; statements, bibliography and appendix identical to v1", " ".join(a2.split()) == " ".join(a1.split()) and T2[T2.index("\\subsubsection*{Ethics Statement}"):] == T1[T1.index("\\subsubsection*{Ethics Statement}"):])
    tabfiles = sorted((TEX/"appendix_tables").glob("tab_*.tex")); tabs = "".join(f.read_text() for f in tabfiles) + open(TEX/"tab_census.tex").read()
    toks = lambda s: set(_re.findall(r"(?<![\w.])[-+]?\d+\.\d+(?![\w.])|(?<![\w.])\d{2,}(?![\w.])", s.replace("{=}", "=")))   # a LaTeX length such as 10.6cm is not a number
    extra = toks(nocom(b2)) - toks(nocom(T1) + nocom(tabs))
    chk("v2: no number appears in v2 that is not in v1 (prose, boxes, equations, captions)", not extra, str(sorted(extra)))
    c2 = set(m.strip() for m in _re.findall(r"(?m)(?<!\\)%\s*(.*)$", b2)); c1 = set(m.strip() for m in _re.findall(r"(?m)(?<!\\)%\s*(.*)$", b1))
    chk("v2: every provenance comment of v2 exists in v1, so every number traces to the same result file", c2 <= c1, str(sorted(c2 - c1)[:4]))
    chk("v2: same figures and tables as v1 (identical \\includegraphics and \\input sets)", set(_re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", b2)) == set(_re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", b1))
        and set(_re.findall(r"\\input\{([^}]*)\}", b2)) == set(_re.findall(r"\\input\{([^}]*)\}", b1)))
    THESIS = "Read correctly, foundation models organize classes into clustered structure that is occasionally hierarchical and moderately shared; they do not converge to one common tree, and their raw tree-likeness is not evidence for hyperbolic geometry."
    SHORT = "Clustered, occasionally hierarchical, moderately shared: not one common tree, and no license for curvature."
    chk("v2: thesis verbatim exactly twice (abstract, S8), no short form", b2.count(THESIS) == 2 and SHORT not in b2 and THESIS in b2[b2.index("\\paragraph{What the paper establishes.}"):])
    secs_ = _re.findall(r"\\section\{([^}]*)\}", b2); subs_ = _re.findall(r"\\subsection\{([^}]*)\}", b2)
    chk("v2: skeleton of eight numbered sections and twelve subsections, seven definitions, seven numbered equations, the hypothesis box, four finding boxes and the definitions-at-a-glance box",
        len(secs_) == 8 and secs_[0] == "Introduction" and secs_[-1] == "Discussion and Limitations" and len(subs_) == 12 and b2.count("\\begin{definition}[") == 7
        and all(("\\label{eq:%s}" % e) in b2 for e in ("defect", "record", "excess", "rank", "bh", "depth", "curvature")) and b2.count("\\begin{equation}") == 7
        and "Hypothesis under test" in b2 and all(("\\textbf{Finding %d.}" % k) in b2 for k in (1, 2, 3, 4)) and "Definitions at a glance" in b2)
    # prose rules of v1 mapped onto the v2 skeleton
    src = nocom(b2); src = _re.sub(r"\\begin\{(figure|table|equation\*?|tabular|center)\}.*?\\end\{\1\}", "", src, flags=_re.S); src = _re.sub(r"\\input\{[^}]*\}", "", src)
    def clean(par):
        p = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", par); p = _re.sub(r"\\(S)?\\?ref\{[^}]*\}", "REF", p); p = _re.sub(r"\\label\{[^}]*\}", "", p)
        p = _re.sub(r"\$([A-Za-z])\{=\}(\d+)\$", r"\1=\2", p); p = _re.sub(r"\$([^$]*)\$", lambda m: " FORMULA " if ("=" in m.group(1) or "\\" in m.group(1)) else m.group(0), p)
        return p
    GROUP = r"(?<![A-Za-z\-^_{\d.])[-+]?\d+(?:\.\d+)?(?![A-Za-z\-\d])"
    RANGE = _re.compile(rf"(?:from\s+)?\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?(?:\s*(?:of the|of|to|against|vs|and|--)\s+\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?){{0,2}}(?:\\%)?")
    def groups(s):
        s = clean(s).replace("{=}", "="); s = _re.sub(r"\\begin\{enumerate\}.*?\\end\{enumerate\}", " ", s, flags=_re.S); s = _re.sub(r"\\begin\{itemize\}.*?\\end\{itemize\}", " ", s, flags=_re.S); s = _re.sub(r"\\[a-zA-Z]+", " ", s)
        return [m.group(0) for m in RANGE.finditer(s) if not _re.fullmatch(r"\s*", m.group(0))]
    KIND = {"Introduction": "intro", "Related Work": "related", "Background and Problem Setup": "other", "The Instrument": "other", "Experimental Setup": "other", "Results": "results", "Implications for Hyperbolic Representation Learning": "results", "Discussion and Limitations": "intro"}
    BANNED = ["tool", "beyond-null", "tree-like structure", "hierarchical structure", "the form ", "reading of record", "our approach", "the method", "essentially", "largely", "substantially", "somewhat", "nterestingly", "notably", "importantly", "we believe", "we note"]
    BANNED = [w for w in BANNED if w != "reading of record"]   # v2 names Definition 2 'reading of record' on the author's brief
    MARK = ["next paragraph", "next question", "subject of the next", "turn to next", "take up last", "which we review", "builds that comparison", "the concern of Section", "First we ask", "last question of this section", "states the three acts", "With the instrument in place", "raises the question of", "which text makes explicit", "has to be re-read", "points to where", "is the first candidate", "What to read instead", "we turn to them next"]
    ALLOW = {"49 of 72", "18 of 24", "4 of 12", "49 of the 72"}
    bad = []; total = 0; bridges = []
    blocks = _re.split(r"\\section\{([^}]*)\}", src); blocks = [(blocks[i], blocks[i+1]) for i in range(1, len(blocks), 2)]
    for name, text in blocks:
        kind = KIND.get(name, "other")
        units = _re.split(r"\\subsection\{[^}]*\}", text)
        for unit in units:
            pars = [q.strip() for q in _re.split(r"\n\s*\n", unit) if q.strip() and not q.strip().startswith(("\\begin{enumerate}", "\\end{enumerate}", "\\item", "\\label", "\\begin{shaded}", "\\end{shaded}"))]
            for k, par in enumerate(pars):
                head = par[:50].replace("\n", " "); cp = clean(par); gs = groups(par); total += len(gs)
                for mk in MARK:
                    if mk.lower() in cp.lower(): bridges.append((name, k == len(pars)-1, mk, head))
                if len(gs) > 2: bad.append(f"{name}: >2 numbers {gs}: {head}")
                for inner in _re.findall(r"\(([^()]*)\)", cp):
                    if groups(inner): bad.append(f"{name}: number in parentheses: {head}")
                if kind == "results":
                    for sent in _re.split(r"(?<=[.!?])\s+", cp):
                        if len(groups(sent)) > 1: bad.append(f"{name}: >1 number per sentence: {sent[:70]}")
                    if ";" in par.replace(THESIS, "") and "\\paragraph{Limitations" not in par and "\\begin{shaded}" not in par: bad.append(f"{name}: semicolon: {head}")
                    ns = len([x for x in _re.split(r"(?<=[.!?])\s+(?=[A-Z\\$(])", cp) if x.strip()])
                    if ns > 9: bad.append(f"{name}: {ns} sentences: {head}")
                if kind == "intro" and [g for g in gs if g.strip() not in ALLOW]: bad.append(f"{name}: numbers beyond the allowed three {gs}: {head}")
                if kind != "related" and len(_re.findall(r"\((?!(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|[a-c])\))", cp)) > 1: bad.append(f"{name}: >1 parenthetical: {head}")
                low = cp.lower()
                for w in BANNED:
                    if w in low: bad.append(f"{name}: banned '{w}': {head}")
    for line in bad: print("   V2:", line[:200])
    chk("v2: the v1 prose rules hold on the v2 skeleton (<=25 number groups in S1-S8, <=2 per paragraph, none in parentheses, results rules in S6-S7, only 49/72, 18/24, 4/12 in S1 and S8, <=1 parenthetical, no banned words)", not bad and total <= 25, f"{total} number groups")
    for s, last, mk, h in bridges: print("   V2 BRIDGE:", s, "last" if last else "NOT last", mk, "|", h)
    chk("v2: exactly three bridges (end of S5, end of the depth subsection, S6.5), each in the last paragraph of its unit, distinct wording", len(bridges) == 3 and {mk for _, _, mk, _ in bridges} == {"With the instrument in place", "turn to next", "subject of the next"} and all(last for _, last, _, _ in bridges))
    alltex = nocom(T2) + nocom(tabs)
    refs = set(_re.findall(r"\\(?:eq)?ref\{([^}]*)\}", alltex)); defs = set(_re.findall(r"\\label\{([^}]*)\}", alltex))
    chk("v2: every cross-reference resolves", refs <= defs, str(sorted(refs - defs)))
    labels = {}
    for f in tabfiles:
        m = _re.search(r"\\label\{(tab:q[^}]*)\}", f.read_text())
        if m: labels[f.stem] = m.group(1)
    main2 = T2[:T2.index("\\appendix")]; cited = []
    for m in _re.finditer(r"\\ref\{(tab:q[^}]*)\}", main2):
        if m.group(1) not in cited: cited.append(m.group(1))
    inputs = _re.findall(r"\\input\{appendix_tables/(tab_[^}]*)\}", T2[T2.index("\\appendix"):]); order = [labels[s] for s in inputs if s in labels]
    chk("v2: the shared appendix tables are still numbered in the order v2 first cites them", order[:len(cited)] == cited, f"cited {cited}")
v2_checks()
# ---------- Version v3 (main_iclr2027_v3.tex): classic structure written from the claim lists; verbatim parts from main_local.tex ----------
def v3_checks():
    import re as _re, json as _j
    TEX = Path(__file__).resolve().parents[2]/"ICLR2027"/"iclr2027"; P3 = TEX/"main_iclr2027_v3.tex"
    if not P3.exists(): chk("v3: main_iclr2027_v3.tex present", False, "missing"); return
    T3 = open(P3).read(); T1 = open(TEX/"main_iclr2027.tex").read(); LOC = open(TEX/"main_local.tex").read()
    nocom = lambda s: _re.sub(r"(?m)(?<!\\)%.*$", "", s); norm = lambda s: " ".join(nocom(s).split())
    b3 = T3[T3.index("\\begin{abstract}"):T3.index("\\subsubsection*{Reproducibility Statement}")]
    def seg(s, a, b): i = s.index(a); j = s.index(b, i); return s[i:j]
    chk("v3: abstract, S1 (with Figure 1) and S2 verbatim from main_local.tex", norm(seg(b3, "\\begin{abstract}", "\\end{abstract}")) == norm(seg(LOC, "\\begin{abstract}", "\\end{abstract}"))
        and norm(seg(b3, "\\section{Introduction}", "\\section{Related Work}")) == norm(seg(LOC, "\\section{Introduction}", "\\section{Related Work}"))
        and norm(seg(b3, "\\section{Related Work}", "\\section{Methodology}")) == norm(seg(LOC, "\\section{Related Work}", "\\section{The Instrument}")))
    grom = norm(seg(LOC, "\\paragraph{Gromov $\\delta$.} ", "\\paragraph{Estimation and normalization.}").split("} ", 1)[1]); est = norm(seg(LOC, "\\paragraph{Estimation and normalization.} ", "\\begin{figure}").split("} ", 1)[1])   # the run-in titles stay out: the subsections carry them
    chk("v3: the 'Gromov delta' and 'Estimation and normalization' paragraphs verbatim from main_local.tex", grom in norm(b3) and est in norm(b3))
    fills = _j.load(open(R/"phaseE_v3_fills.json")) if (R/"phaseE_v3_fills.json").exists() else {}
    tabfiles = sorted(f for f in (TEX/"appendix_tables").glob("tab_*.tex") if "_final" not in f.stem); tabs = "".join(f.read_text() for f in tabfiles) + open(TEX/"tab_census.tex").read()   # the final version's copies are checked by final_checks
    can = lambda x: _re.sub(r"^[-+]", "", x).lstrip("0") if "." in x else _re.sub(r"^[-+]", "", x)
    toks = lambda s: {can(x) for x in _re.findall(r"(?<![\w.])[-+]?\d*\.\d+(?![\w.])|(?<![\w.])\d{2,}(?![\w.])", s.replace("{=}", "="))}
    allowed = toks(nocom(T1) + nocom(tabs)) | {can(x) for v in fills.values() for x in _re.findall(r"[-+]?\d*\.\d+|\d+", str(v))}
    extra = toks(_re.sub(r"\\includegraphics\[[^\]]*\]", "", nocom(b3))) - allowed     # graphics options (trim, width) are not numbers of the paper
    chk("v3: no number appears in v3 that is not in v1 or in a fill traced to a result file", not extra, str(sorted(extra)))
    files = set(_re.findall(r"([\w/]+\.(?:csv|npz|json))", "".join(_re.findall(r"(?m)(?<!\\)%(.*)$", b3))))
    missing = sorted(f for f in files if not (R/f).exists() and not Path("/media/HDD_4TB_2/javi/Platonic/results", f).exists())
    chk("v3: every result file named in a provenance comment exists", not missing, str(missing))
    chk("v3: no boxes, no colored text, definitions inline (amsthm)", "tcolorbox" not in T3 and "\\textcolor" not in b3 and "\\colorbox" not in b3 and "\\newtheorem{definition}" in T3)
    THESIS = "Read correctly, foundation models organize classes into clustered structure that is occasionally hierarchical and moderately shared; they do not converge to one common tree, and their raw tree-likeness is not evidence for hyperbolic geometry."
    chk("v3: thesis verbatim exactly twice (abstract's last sentence, S7 conclusion), no short form", b3.count(THESIS) == 2 and "no license for curvature" not in b3 and THESIS in b3[b3.index("\\paragraph{Conclusion.}"):])
    secs_ = _re.findall(r"\\section\{([^}]*)\}", b3); subs_ = _re.findall(r"\\subsection\{([^}]*)\}", b3)
    chk("v3: skeleton of seven numbered sections and eleven subsections, seven unframed definitions, one two-part proposition with a remark, cited where the dimension confound is discussed, seven numbered equations, its proof in the appendix",
        len(secs_) == 7 and secs_[2] == "Methodology" and secs_[-1] == "Conclusion and Limitations" and len(subs_) == 11 and b3.count("\\begin{definition}[") == 7 and b3.count("\\begin{proposition}") == 1 and "(a)" in b3[b3.index("\\begin{proposition}"):b3.index("\\end{proposition}")] and "(b)" in b3[b3.index("\\begin{proposition}"):b3.index("\\end{proposition}")]
        and "\\begin{lemma}" not in b3 and "\\begin{corollary}" not in b3 and b3.count("\\begin{remark}") == 1 and b3.count("\\ref{prop:bound}") >= 2
        and b3.count("\\begin{equation}") == 7 and all(("\\label{eq:%s}" % e) in b3 for e in ("pairings", "deltanorm", "excess", "rank", "bh", "depth", "curvature")) and T3.count("\\begin{proof}") == 1 and "\\label{app:proofs}" in T3)
    # prose rules: S5 <=2 number groups per paragraph and <=1 per sentence; S1/S7 only the allowed numbers; <=1 parenthetical per non-verbatim paragraph of S3-S7; bridges only in the last paragraph of a section; banned words
    src = nocom(b3); src = _re.sub(r"\\begin\{(figure|table|equation\*?|tabular|center)\}.*?\\end\{\1\}", "", src, flags=_re.S); src = _re.sub(r"\\input\{[^}]*\}", "", src)
    def clean(par):
        p = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", par); p = _re.sub(r"\\(S)?\\?ref\{[^}]*\}", "REF", p); p = _re.sub(r"\\label\{[^}]*\}", "", p)
        p = _re.sub(r"\$([A-Za-z])\{=\}(\d+)\$", r"\1=\2", p); p = _re.sub(r"\$([^$]*)\$", lambda m: " FORMULA " if ("=" in m.group(1) or "\\" in m.group(1)) else m.group(0), p)
        return p
    GROUP = r"(?<![A-Za-z\-^_{\d.])[-+]?\d+(?:\.\d+)?(?![A-Za-z\-\d])"
    RANGE = _re.compile(rf"(?:from\s+)?\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?(?:\s*(?:of the|of|to|against|vs|and|--)\s+\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?){{0,2}}(?:\\%)?")
    def groups(s):
        s = clean(s).replace("{=}", "="); s = _re.sub(r"\\begin\{enumerate\}.*?\\end\{enumerate\}", " ", s, flags=_re.S); s = _re.sub(r"\\[a-zA-Z]+", " ", s)
        return [m.group(0) for m in RANGE.finditer(s) if not _re.fullmatch(r"\s*", m.group(0))]
    KIND = {"Introduction": "intro", "Related Work": "related", "Methodology": "other", "Experimental Setup": "other", "Results": "results", "Implications for Hyperbolic Representation Learning": "other", "Conclusion and Limitations": "intro"}
    BANNED = ["tool", "beyond-null", "tree-like structure", "hierarchical structure", "the form ", "our approach", "the method", "essentially", "largely", "substantially", "somewhat", "nterestingly", "notably", "importantly", "we believe", "we note"]
    MARK = ["next paragraph", "next question", "subject of the next", "turn to next", "take up last", "which we review", "builds that comparison", "the concern of Section", "First we ask", "last question of this section", "states the three acts", "With the setup fixed", "next section fixes", "raises the question of", "which text makes explicit", "has to be re-read", "points to where", "is the first candidate", "What to read instead", "we turn to them next"]
    ALLOW = {"49 of 72", "18 of 24", "4 of 12", "49 of the 72"}
    verbatim = [grom, est, norm(seg(LOC, "\\section{Introduction}", "\\section{Related Work}")), norm(seg(LOC, "\\section{Related Work}", "\\section{The Instrument}"))]
    isverb = lambda par: any(norm(par) in v for v in verbatim)
    bad = []; bridges = []; total = 0
    blocks = _re.split(r"\\section\{([^}]*)\}", src); blocks = [(blocks[i], blocks[i+1]) for i in range(1, len(blocks), 2)]
    for name, text in blocks:
        kind = KIND.get(name, "other")
        pars = [q.strip() for q in _re.split(r"\n\s*\n", text) if q.strip() and not q.strip().startswith(("\\begin{enumerate}", "\\end{enumerate}", "\\item", "\\label", "\\subsection", "\\begin{definition}", "\\begin{lemma}", "\\begin{corollary}", "\\begin{remark}"))]
        for k, par in enumerate(pars):
            head = par[:50].replace("\n", " "); cp = clean(par); gs = groups(par); total += len(gs); vb = isverb(par)
            for mk in MARK:
                if mk.lower() in cp.lower() and not vb: bridges.append((name, k == len(pars)-1, mk, head))
            if kind == "results":
                if len(gs) > 2: bad.append(f"{name}: >2 numbers {gs}: {head}")
                for sent in _re.split(r"(?<=[.!?])\s+", cp):
                    if len(groups(sent)) > 1: bad.append(f"{name}: >1 number per sentence: {sent[:70]}")
            if kind == "intro" and not vb and [g for g in gs if g.strip() not in ALLOW]: bad.append(f"{name}: numbers beyond the allowed three {gs}: {head}")
            if kind != "related" and not vb and len(_re.findall(r"\((?!(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|[a-c])\))", cp)) > 1: bad.append(f"{name}: >1 parenthetical: {head}")
            low = cp.lower()
            for w in BANNED:
                if w in low and not vb: bad.append(f"{name}: banned '{w}': {head}")
    for line in bad: print("   V3:", line[:200])
    chk("v3: the prose rules hold (S5: <=2 number groups per paragraph and <=1 per sentence; S1/S7: only 49/72, 18/24, 4/12 outside the verbatim text; <=1 parenthetical per non-verbatim paragraph; no banned words)", not bad, f"{total} number groups")
    for s, last, mk, h in bridges: print("   V3 BRIDGE:", s, "last" if last else "NOT last", mk, "|", h)
    chk("v3: bridges only at section ends (every bridge in the last paragraph of its section, at most three, verbatim paragraphs excepted)", len(bridges) <= 3 and all(last for _, last, _, _ in bridges))
    alltex = nocom(T3) + nocom(tabs); refs = set(_re.findall(r"\\(?:eq)?ref\{([^}]*)\}", alltex)); defs = set(_re.findall(r"\\label\{([^}]*)\}", alltex))
    chk("v3: every cross-reference resolves", refs <= defs, str(sorted(refs - defs)))
    b1 = T1[T1.index("\\begin{abstract}"):T1.index("\\subsubsection*{Ethics Statement}")]
    f3 = set(_re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", b3)); f1 = set(_re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", b1))
    chk("v3: same figures as v1 except Figure 4 (fig_depth_test.pdf, the v3 redraw of fig_depth_main.pdf), the same main-text table, Figure 1 slot 1.4 in", f3 - f1 == {"figures/fig_depth_test.pdf"} and f1 - f3 <= {"figures/fig_depth_main.pdf", "figures/fig4_schematic.pdf"} and "\\input{tab_census}" in b3 and "[1.4in]" in b3)
    labels = {}
    for f in tabfiles:
        m = _re.search(r"\\label\{(tab:q[^}]*)\}", f.read_text())
        if m: labels[f.stem] = m.group(1)
    main3 = T3[:T3.index("\\appendix")]; cited = []
    for m in _re.finditer(r"\\ref\{(tab:q[^}]*)\}", main3):
        if m.group(1) not in cited: cited.append(m.group(1))
    app3 = T3[T3.index("\\appendix"):]; inputs = _re.findall(r"\\input\{appendix_tables/(tab_[^}]*)\}", app3); order = [labels[s] for s in inputs if s in labels]
    chk("v3: the fourteen appendix tables are all input, numbered in the order v3 first cites them, and each is cited from the main text or another table",
        len(inputs) == 15 and set(inputs) == {f.stem for f in tabfiles} and order[:len(cited)] == cited
        and all(("\\ref{"+lab+"}" in main3) or any(("\\ref{"+lab+"}") in g.read_text() for g in tabfiles if g.stem != s) for s, lab in labels.items()), f"cited {cited}")
    # key claim fills recomputed from the files
    if fills:
        e21 = load("exp21b_local_global_K200.csv"); tx = {r["model"]: r for r in load("expR53_text_haar_p999_200.csv")}
        chk("v3: claim fills recomputed (mKNN/CKA calibrated means, text genuine count and GPT-2 S p, joint-sensitivity range, power at s=1, star depth)",
            fills["KNN_CAL"] == f"{mean(float(r['knn_R_cal']) for r in e21):.3f}" and fills["CKA_CAL"] == f"{mean(float(r['cka_R_cal']) for r in e21):.3f}" and fills["TEXT_GEN"] == str(sum(str(r["genuine_bh"])=="True" for r in tx.values()))
            and fills["GPT2S_P"] == f"{float(tx['gpt2']['p_left']):.3f}".lstrip("0") and fills["R11_LO"] + "--" + fills["R11_HI"] in open(TEX/"tab_census.tex").read() and fills["POWER_S1"] in open(TEX/"appendix_tables"/"tab_q05_power.tex").read()
            and fills["STAR_DEEP"] in open(TEX/"appendix_tables"/"tab_q10_calibration.tex").read())
v3_checks()

def final_checks():
    """Final version (author's brief 'Final version — plain, short, nine pages', 2026-09-18): main_iclr2027_final.tex."""
    import re as _re, json as _j
    TEX = Path(__file__).resolve().parents[2]/"ICLR2027"/"iclr2027"; PF = TEX/"main_iclr2027_final.tex"
    if not PF.exists(): chk("final: main_iclr2027_final.tex present", False, "missing"); return
    TF = open(PF).read(); T1 = open(TEX/"main_iclr2027.tex").read(); LOC = open(TEX/"main_local.tex").read()
    nocom = lambda s: _re.sub(r"(?m)(?<!\\)%.*$", "", s); norm = lambda s: " ".join(nocom(s).split())
    bf = TF[TF.index("\\begin{abstract}"):TF.index("\\subsubsection*{Reproducibility Statement}")]; stm = TF[TF.index("\\subsubsection*{Reproducibility Statement}"):TF.index("\\bibliographystyle")]
    def seg(s, a, b): i = s.index(a); j = s.index(b, i); return s[i:j]
    ED = _j.load(open(R/"final_verbatim_edits.json"))["edits"] if (R/"final_verbatim_edits.json").exists() else []
    def edit(s):
        for a, b in ED: s = s.replace(a, b)
        return s
    # ---- verbatim parts, modulo the edits the brief names (recorded in final_verbatim_edits.json)
    chk("final: abstract verbatim from main_local.tex modulo the two recorded fourth-review edits (15 text models, the S5.1 wording)", norm(seg(bf, "\\begin{abstract}", "\\end{abstract}")) == norm(edit(seg(LOC, "\\begin{abstract}", "\\end{abstract}"))) and "15 text models" in bf and "16 text models" not in bf and "sixteen" not in bf and "two OLMo" not in bf)
    chk("final: S1 with Figure 1 verbatim from main_local.tex modulo the three recorded metaphor edits", norm(seg(bf, "\\section{Introduction}", "\\section{Related Work}")) == norm(edit(seg(LOC, "\\section{Introduction}", "\\section{Related Work}"))))
    rel_loc = norm(seg(LOC, "\\section{Related Work}", "\\section{The Instrument}")); rel_f = norm(seg(bf, "\\section{Related Work}", "\\section{Methodology}"))
    rel_loc = norm(edit(seg(LOC, "\\section{Related Work}", "\\section{The Instrument}")))
    chk("final: S2 verbatim from main_local.tex modulo the recorded citation sentence (Sala 2018, Gu 2019), or its first six sentences (cut step 4 of the page budget)", rel_f == rel_loc or (rel_f in rel_loc and rel_loc.startswith(rel_f)))
    grom = norm(edit(seg(LOC, "\\paragraph{Gromov $\\delta$.} ", "\\paragraph{Estimation and normalization.}").split("} ", 1)[1])); est = norm(edit(seg(LOC, "\\paragraph{Estimation and normalization.} ", "\\begin{figure}").split("} ", 1)[1]))
    chk("final: 'Gromov delta' and 'Estimation and normalization' verbatim modulo the recorded edits (supremum phrase, bridge sentence)", grom in norm(bf) and est in norm(bf))
    chk("final: the recorded edits are exactly the briefs' (shadow x2, geometric face, intent of the supremum, the bridge; 4th/5th reviews: abstract, S1 confounds and counts, S2 citations) and none of the old phrases survives",
        len(ED) == 15 and all((a not in bf) or (a in b) for a, b in ED) and any("hub-aligned structure in a few" in b for _, b in ED) and any("once cluster orientations are randomized" in b for _, b in ED) and "49 of 72" not in bf and all((b in bf) for _, b in ED if b) and sum(1 for a, _ in ED if "shadow" in a) == 2 and any("geometric face" in a for a, _ in ED) and any("intent of the supremum" in a for a, _ in ED) and any("next question" in a for a, _ in ED)
        and any("16 text models" in a for a, _ in ED) and any("three artifacts push it down" in a for a, _ in ED) and any("two of which survive every choice of frame" in b for _, b in ED) and any("sala2018representation" in b and "gu2019learning" in b for _, b in ED)
        and bf.count("weak and model-dependent") >= 3 and bf.count("30 of 36") >= 3 and "cannot be called low on its own" in bf)
    # ---- metaphors, bridges and banned phrases anywhere in the body (Figure 1 and its caption excepted), thesis twice
    body_nofig = _re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", nocom(bf), flags=_re.S); body_nocite = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", body_nofig)
    META = ["shadow", "star caveat", "aristotelian", "geometric face", "intent of the supremum"]
    hits = [w for w in META if w in body_nocite.lower()]
    chk("final: no metaphor outside Figure 1 and its caption (shadow, star caveat, Aristotelian, geometric face, the intent of the supremum)", not hits, str(hits))
    MARK = ["next paragraph", "next question", "subject of the next", "turn to next", "take up last", "which we review", "builds that comparison", "the concern of Section", "First we ask", "last question of this section", "states the three acts", "With the setup fixed", "next section fixes", "raises the question of", "which text makes explicit", "has to be re-read", "points to where", "is the first candidate", "What to read instead", "we turn to them next", "is the next question", "we now", "below we", "in what follows"]
    bh = [m for m in MARK if m.lower() in body_nocite.lower()]
    chk("final: no bridge sentences anywhere in the body", not bh, str(bh))
    BANNED = ["we note", "interestingly", "importantly", "in plain terms", "notably", "essentially", "largely", "substantially", "somewhat", "we believe", "our approach", "the method", "tool"]
    THESIS = "Read correctly, foundation models organize classes into clustered, hub-aligned structure that is moderately shared; no hierarchy above the superclasses is certified, they do not converge to one common tree, and their raw tree-likeness is not evidence for hyperbolic geometry."   # author's decision, 2026-09-20
    chk("final: thesis verbatim exactly twice (abstract's last sentence, S7 conclusion), no short form", bf.count(THESIS) == 2 and "no license for curvature" not in bf and THESIS in bf[bf.index("\\paragraph{Conclusion.}"):] and "occasionally hierarchical" not in bf)
    # ---- structure: classic skeleton, seven inline unframed definitions, seven equations, Proposition 1 (a)(b) proved in Appendix A, no boxes
    secs_ = _re.findall(r"\\section\{([^}]*)\}", bf); subs_ = _re.findall(r"\\subsection\{([^}]*)\}", bf); prop = bf[bf.index("\\begin{proposition}"):bf.index("\\end{proposition}")]
    chk("final: skeleton unchanged from v3 (seven sections, eleven subsections), seven definitions inline and unframed, seven numbered equations, Proposition 1 (a)(b) with its one proof in Appendix A, no boxes and no colored text",
        len(secs_) == 7 and secs_[2] == "Methodology" and secs_[-1] == "Conclusion and Limitations" and len(subs_) == 11 and bf.count("\\begin{definition}[") == 7 and bf.count("\\begin{proposition}") == 1 and "(a)" in prop and "(b)" in prop
        and bf.count("\\begin{equation}") == 7 and all(("\\label{eq:%s}" % e) in bf for e in ("pairings", "deltanorm", "excess", "rank", "bh", "depth", "curvature")) and TF.count("\\begin{proof}") == 1 and "\\label{app:proofs}" in TF and TF.index("\\section{Proofs}") > TF.index("\\appendix")
        and "tcolorbox" not in TF and "\\textcolor" not in bf and "\\colorbox" not in bf and "\\begin{lemma}" not in bf and "\\begin{corollary}" not in bf and "\\begin{remark" not in bf and "\\newtheorem{definition}" in TF)
    # ---- prose rules on the non-verbatim prose of S3-S7 (definitions, equations, captions and tables excluded)
    src = nocom(bf); src = _re.sub(r"\\begin\{(figure|table|tabular|center)\}.*?\\end\{\1\}", "", src, flags=_re.S); src = _re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " EQUATION. ", src, flags=_re.S); src = _re.sub(r"\\input\{[^}]*\}", "", src)
    def clean(par):
        p = _re.sub(r"\\paragraph\{[^}]*\}\s*", "", par); p = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "CITE", p); p = _re.sub(r"\\(S)?\\?ref\{[^}]*\}", "REF", p); p = _re.sub(r"\\label\{[^}]*\}", "", p)
        p = _re.sub(r"\$([A-Za-z])\{=\}(\d+)\$", r"\1=\2", p); p = _re.sub(r"\$([^$]*)\$", lambda m: " FORMULA " if ("=" in m.group(1) or "\\" in m.group(1)) else m.group(0), p)
        p = p.replace("``", "").replace("''", "").replace("~", " "); p = _re.sub(r"\\[a-zA-Z]+\*?", " ", p); p = _re.sub(r"[{}]", "", p); return " ".join(p.split())
    GROUP = r"(?<![A-Za-z\-^_{\d.])[-+]?\d+(?:\.\d+)?(?![A-Za-z\-\d])"
    RANGE = _re.compile(rf"(?:from\s+)?\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?(?:\s*(?:of the|of|to|against|vs|and|--)\s+\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?){{0,2}}(?:\\%)?")
    def groups(s):
        s = _re.sub(r"\\paragraph\{[^}]*\}\s*", "", s); s = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", s); s = _re.sub(r"\\(S)?\\?ref\{[^}]*\}", "REF", s); s = _re.sub(r"\\label\{[^}]*\}", "", s)
        s = _re.sub(r"\$([^$]*)\$", lambda m: " FORMULA " if ("=" in m.group(1) or "\\" in m.group(1)) else m.group(0), s).replace("{=}", "="); s = _re.sub(r"\\[a-zA-Z]+", " ", s)
        return [m.group(0).replace("$", "").replace("from ", "").strip() for m in RANGE.finditer(s) if not _re.fullmatch(r"\s*", m.group(0))]
    HEADLINE = {"44 of 72", "18 of 24", "4 of 12", "0 of 60", "7 of 15", "0.48 to 2.5", "+0.9 to +1.3", "30 of 36", "47", "5 of 12", "210", "+0.41 against +0.28"}   # 2026-09-20: the record is the centered Haar null (44 of 72; FMNIST 5 of 12)
    EXEMPT = {"Most cells show clustered structure.": (4, 3)}   # fifth review: the author's count sentence carries three numbers (paragraph max, sentence max)
    noeq = lambda s: _re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " ", s, flags=_re.S)
    verbatim = [norm(noeq(v)) for v in (seg(LOC, "\\paragraph{Gromov $\\delta$.} ", "\\paragraph{Estimation and normalization.}").split("} ", 1)[1], edit(seg(LOC, "\\paragraph{Estimation and normalization.} ", "\\begin{figure}").split("} ", 1)[1]), edit(seg(LOC, "\\section{Introduction}", "\\section{Related Work}")), seg(LOC, "\\section{Related Work}", "\\section{The Instrument}"))]
    isverb = lambda par: any(norm(par.replace(" EQUATION. ", " ")) in v for v in verbatim)
    sent_split = lambda s: [x.strip() for x in _re.split(r"(?<=[.!?])\s+(?=[A-Z(\\])", s) if len(x.split()) > 1]
    bad = []; stats = {}; numpar = []; nverb_sents = []
    blocks = _re.split(r"\\section\{([^}]*)\}", src); blocks = [(blocks[i], blocks[i+1]) for i in range(1, len(blocks), 2)]
    for name, text in blocks:
        results = name in ("Results", "Implications for Hyperbolic Representation Learning"); prose = name not in ("Introduction", "Related Work")
        pars = [q.strip() for q in _re.split(r"\n\s*\n", text) if q.strip() and not q.strip().startswith(("\\begin{definition}", "\\begin{proposition}", "\\label", "\\subsection", "\\begin{equation}"))]
        ws = []
        for par in pars:
            vb = isverb(par) or not prose; cp = clean(par); sents = sent_split(cp); w = [len(s.split()) for s in sents]; ws += w
            if not par.startswith("\\paragraph") and prose and not vb: bad.append(f"{name}: prose paragraph without a bold lead-in: {par[:60]!r}"); continue
            lead = _re.match(r"\\paragraph\{([^}]*)\}", par); gs = groups(par)
            if results: numpar.append((name, lead.group(1) if lead else par[:40], gs))
            if vb: continue
            nverb_sents += w
            w_ = [len(s.split()) for s in sents if not s.startswith("Read correctly, foundation models")]   # the author's thesis sentence is verbatim and exempt from the 35-word rule
            if max(w_, default=0) > 35: bad.append(f"{name}: sentence over 35 words ({max(w_)}): {sents[w.index(max(w_))][:80]}")
            if not (3 <= len(sents) <= 6): bad.append(f"{name}: paragraph with {len(sents)} sentences: {par[:60]!r}")
            if lead and (len(lead.group(1).split()) > 16 or not lead.group(1).endswith(".")): bad.append(f"{name}: lead-in not plain/short: {lead.group(1)!r}")
            if results:
                pmax, smax = EXEMPT.get(lead.group(1) if lead else "", (2, 1))
                if len(gs) > pmax: bad.append(f"{name}: >{pmax} numbers in a paragraph {gs}: {par[:60]!r}")
                for s_ in sents:
                    if len(groups(s_)) > smax: bad.append(f"{name}: >{smax} number in a sentence: {s_[:80]}")
                ptr = [i for i, s_ in enumerate(sents) if ("Table REF" in s_ or "Figure REF" in s_ or "Tables REF" in s_)]
                if ptr and ptr != [len(sents) - 1]: bad.append(f"{name}: table/figure pointer not confined to the last sentence: {par[:60]!r}")
            off = [g for g in gs if g not in HEADLINE]
            if off: bad.append(f"{name}: number outside the headline set {off}: {par[:60]!r}")
            for m in _re.finditer(r"\(([^()]*)\)", cp):
                if not _re.fullmatch(r"(i|ii|iii|iv|v|vi|vii|viii|ix|x|[a-c])", m.group(1)) and len(m.group(1).split()) > 3: bad.append(f"{name}: parenthetical over three words: ({m.group(1)[:50]})")
            for s_ in sents:
                if s_.count(";") >= 2: bad.append(f"{name}: semicolon chain: {s_[:80]}")
            low = cp.lower()
            for wd in BANNED:
                if _re.search(r"\b" + _re.escape(wd) + r"\b", low): bad.append(f"{name}: banned '{wd}': {par[:60]!r}")
            if results and not _re.search(r"(?m)%\s*[\w/]+\.(csv|npz|json)", par + "\n") and not _re.search(r"%.*\.(csv|npz|json)", bf[bf.index(par[:80]):bf.index(par[:80]) + len(par) + 400]): bad.append(f"{name}: results paragraph without a provenance comment: {par[:60]!r}")
        stats[name] = {"sentences": len(ws), "avg_words": round(mean(ws), 1) if ws else 0, "max_words": max(ws, default=0)}
    avg_nv = round(mean(nverb_sents), 1) if nverb_sents else 0
    if avg_nv > 22: bad.append(f"average sentence length of the non-verbatim prose {avg_nv} > 22")
    for line in bad: print("   FINAL:", line[:220])
    chk("final: plain-prose rules on the non-verbatim prose of S3-S7 (avg <= 22 words, none > 35, paragraphs of 3-6 sentences with a plain bold lead-in, S5-S6 <= 1 number per sentence and <= 2 per paragraph with one pointer in the last sentence, only headline numbers, no parenthetical over three words, no semicolon chains, no banned phrases, provenance comment on every results paragraph)",
        not bad, f"avg non-verbatim sentence {avg_nv} words over {len(nverb_sents)} sentences")
    _j.dump({"sections": stats, "avg_nonverbatim": avg_nv, "n_nonverbatim_sentences": len(nverb_sents), "numbers_per_paragraph_S5_S6": [{"section": a, "lead": b, "numbers": c} for a, b, c in numpar]}, open(R/"final_prose_stats.json", "w"), indent=1)
    # ---- vocabulary defined once in S3 (or in the verbatim S1 for 'premise'), before its first use in S4-S7
    voc = {"excess": "\\begin{definition}[excess]", "genuine": "\\begin{definition}[genuine]", "reading of record": "\\begin{definition}[reading of record]", "matched star": "\\begin{definition}[matched star]", "hub null": "\\begin{definition}[hub null", "Haar null": "\\begin{definition}[Haar null]", "four-point defect": "\\begin{definition}[four-point defect]"}
    s3 = seg(bf, "\\section{Methodology}", "\\section{Experimental Setup}")
    chk("final: the seven defined terms are each defined once in S3 (one definition environment each) and 'premise' is fixed in S1", all(s3.count(v) == 1 for v in voc.values()) and "premise" in seg(bf, "\\section{Introduction}", "\\section{Related Work}"))
    # ---- numbers: nothing new (every number in the final is in v1, in a table or in a fill traced to a result file); every result file named exists
    fills = {}
    for fn in ("phaseE_fills.json", "phaseE_v3_fills.json", "final_fig5_values.json", "final_fills.json"):
        if (R/fn).exists(): fills.update(_j.load(open(R/fn)))
    tabfiles = sorted((TEX/"appendix_tables").glob("tab_*.tex")); tabs = "".join(f.read_text() for f in tabfiles) + open(TEX/"tab_census.tex").read() + open(TEX/"tab_census_final.tex").read()
    can = lambda x: _re.sub(r"^[-+]", "", x).lstrip("0") if "." in x else _re.sub(r"^[-+]", "", x)
    toks = lambda s: {can(x) for x in _re.findall(r"(?<![\w.])[-+]?\d*\.\d+(?![\w.])|(?<![\w.])\d{2,}(?![\w.])", s.replace("{=}", "="))}
    allowed = toks(nocom(T1) + nocom(tabs)) | {can(x) for v in fills.values() for x in _re.findall(r"[-+]?\d*\.\d+|\d+", str(v))}
    extra = toks(_re.sub(r"\\includegraphics\[[^\]]*\]", "", nocom(bf))) - allowed
    chk("final: no number appears in the final that is not in v1, in a table or in a fill traced to a result file", not extra, str(sorted(extra)))
    files = set(_re.findall(r"([\w/]+\.(?:csv|npz|json))", "".join(_re.findall(r"(?m)(?<!\\)%(.*)$", bf))))
    missing = sorted(f for f in files if not (R/f).exists() and not Path("/media/HDD_4TB_2/javi/Platonic/results", f).exists())
    chk("final: every result file named in a provenance comment of the main text exists", not missing, str(missing))
    # ---- figures: the four bar-language figures and Figure 1, the 1.4 in slot kept, captions with a bold takeaway
    figs = _re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", bf)
    chk("final: Figure 1 is the author's figure command verbatim from main_local.tex and Figures 2-5 are the bar-language files fig_overview_final, fig_excess_final, fig_depth_final, fig_treemap_final, all present",
        figs == ["figures/fig1_concept.pdf", "figures/fig_overview_final.pdf", "figures/fig_excess_final.pdf", "figures/fig_depth_final.pdf", "figures/fig_treemap_final.pdf"] and all((TEX/f).exists() for f in figs[1:]) and _re.search(r"\\includegraphics\[[^\]]*\]\{figures/fig1_concept\.pdf\}", LOC).group(0) in bf and "\\IfFileExists{figures/fig1_concept.pdf}" in bf)   # Figure 1 is the author's IfFileExists slot (1.4 in box when the file is absent)
    caps = _re.findall(r"\\caption\{(.*?)\n\}", bf, flags=_re.S)
    chk("final: every main-text figure caption opens with a bold takeaway and names its source files", len(caps) >= 4 and all(c.lstrip().startswith("\\textbf{") and _re.search(r"%.*\.(csv|npz|json)", c) for c in caps[:4]))
    v5 = _j.load(open(R/"final_fig5_values.json")) if (R/"final_fig5_values.json").exists() else {}
    chk("final: the Figure 5 caption states the naive and the selected DINOv2-vs-block agreement read from the figure's data", bool(v5) and f"{v5['naive_dinov2_vs_block']:.2f}" in caps[3] and f"{v5['selected_dinov2_vs_block']:.2f}" in caps[3])
    # ---- appendix: the cited tables only, in citation order, figures gone, xi/ORC/null-variant panel gone, provenance comments kept, cross-references resolve
    main_f = TF[:TF.index("\\appendix")]; app_f = TF[TF.index("\\appendix"):]; FD = TEX/"appendix_tables"/"final"
    inputs = _re.findall(r"\\input\{appendix_tables/final/(tab_[^}]*)\}", app_f)
    labels = {}
    for f in sorted(FD.glob("tab_q*.tex")):   # the final copies (v1 tables split by panel, plus the tables regenerated for the final: robustness, depth, wordnet)
        m = _re.search(r"\\label\{(tab:q[^}]*)\}", f.read_text())
        if m: labels[f.stem] = m.group(1)
    cited = []
    for m in _re.finditer(r"\\ref\{(tab:q[^}]*)\}", main_f):
        if m.group(1) not in cited: cited.append(m.group(1))
    order = [labels[s] for s in inputs if s in labels]
    kept = [s for s in inputs if s.startswith("tab_q")]
    chk("final: the appendix inputs exactly the tables the main text cites (twelve question tables, robustness in its final form) plus the provenance index, in first-citation order, from appendix_tables/final/",
        set(order) == set(cited) and order == cited and "tab_q08_robust_final" in inputs and "tab_q08_robust" not in inputs and inputs[-1] == "tab_z_provenance_final" and len(kept) == 12 and "\\input{appendix_tables/tab_" not in app_f, f"inputs {inputs}")
    gone = ["tab_q11_interventions", "tab_q12_xi"]; gone_lab = ["tab:q11-interventions", "tab:q12-xi", "fig:depth-c100", "fig:depthpower", "fig:causal", "fig:treemapc100", "fig:textnulls", "fig:bestmetric"]
    rob = (FD/"tab_q08_robust_final.tex").read_text()
    chk("final: xi, ORC/interventions table, appendix figures, null-variant panel and the class-count sweep are gone from the final and nothing refers to them",
        all(g not in inputs for g in gone) and all(("\\ref{" + l + "}") not in TF for l in gone_lab) and "\\includegraphics" not in app_f and "Ollivier" not in app_f
        and "null variants" not in rob.lower() and rob.count("DINOv2-L & 100 &") == 1 and rob.count("non-hierarchical fine-tuning") == 1 and "supremum, Gaussian" not in rob
        and "(b) Correlation between the raw supremum" not in (FD/"tab_q09_corollary.tex").read_text())
    chk("final: every kept appendix table keeps its provenance comments (% prov: lines and % source comments) and the provenance index lists all thirteen", all(("% prov:" in (FD/(s + ".tex")).read_text()) for s in kept) and (FD/"tab_z_provenance_final.tex").read_text().count("\\texttt{tab\\_") == 12)
    # the final copies are the v1 tables split into one floating table per panel: same numbers, same captions, [tbp] instead of [H]
    same = []
    for s in kept:
        if s.endswith("_final"): continue   # regenerated for the final by gen_appendix_final.py (robustness, depth, wordnet): checked below
        strip_ = lambda s_: _re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", nocom(s_))   # the column separation is repeated per panel in the copies
        a = strip_((TEX/"appendix_tables"/(s + ".tex")).read_text()); b = strip_((FD/(s + ".tex")).read_text())
        na, nb = _re.findall(r"\d+\.\d+|\d+", a), _re.findall(r"\d+\.\d+|\d+", b)
        if s in ("tab_q01_census", "tab_q09_corollary"): ok = set(nb) <= set(na) and len(nb) < len(na)      # one panel dropped by the brief (null-variant panel, uncited supremum table)
        elif s in ("tab_q02_text", "tab_q14_panel"): ok = set(nb) - set(na) <= {"8"} and "OLMo-7B" in a and "OLMo-7B" not in b   # fourth review: the OLMo-7B row (not extracted) is gone; fifth: "8 causal LMs" in the panel caption
        else: ok = sorted(na) == sorted(nb) and _re.findall(r"\\caption\{(?!\(continued\)\})", a) == _re.findall(r"\\caption\{(?!\(continued\)\})", b)
        same.append(ok and "[H]" not in b and b.count("\\begin{table}") >= a.count("\\begin{table}") - 1)
    chk("final: each final copy in appendix_tables/final/ carries the numbers and captions of its v1 table (minus the dropped panel in the census and corollary tables), floating and split by panel", all(same) and len(same) == 7, str([s for s, ok in zip([k for k in kept if not k.endswith("_final")], same) if not ok]))
    # fourth review: the regenerated tables of the final
    dep = (FD/"tab_q04_depth_final.tex").read_text(); wn = (FD/"tab_q07_wordnet_final.tex").read_text(); txt_ = (FD/"tab_q02_text.tex").read_text(); pan_ = (FD/"tab_q14_panel.tex").read_text()
    pw = (FD/"tab_q05_power_final.tex").read_text(); cor = (FD/"tab_q09_corollary.tex").read_text()
    chk("final (5th review): confounds stated as a reference level in S1 and S3.2; Proposition 1(b) for fixed n; Khrulkov's rule calibrated with the supremum; S4 210 quadruples; S5.2 flat datasets and 30 of 36; S5.3 WordNet levels and the decoupling control; S6 depth verdict predicts no hyperbolic gain and the calibration paragraph; Figure 2(b) same dataset; no 'OLMo-7B was not extracted'; power table with two-decimal z; Table 13 and S B.12 'Both readings'; Table 5 with 8 causal LMs; AI statement in three items",
        "reference level of the reading depends on dimension, spectrum and statistic" in bf and "with $n$ fixed" in bf and "calibrated with the supremum" in bf and "210 quadruples" in bf and "FMNIST is genuine in" in bf and "levels" in bf and "decoupling control" in bf
        and "The depth verdict predicts no hyperbolic gain." in bf and "\\paragraph{The calibration certifies structure and does not choose the readout.}" in bf and "Same dataset, same reading, opposite verdict" in bf and "OLMo-7B was not extracted" not in bf
        and not _re.search(r"\$[+-]\d\.\d\$", pw) and "Both readings predict the zero-cost gain" in cor and "The calibrated reading predicts the gain" not in TF and "8 causal LMs" in pan_ and "9 causal LMs" not in pan_ and "research ideation or execution" in stm)
    d74 = load("expR74_decoupling_summary.csv") if (R/"expR74_decoupling_summary.csv").exists() else []
    cen_ = (FD/"tab_q01_census_final.tex").read_text()
    chk("final (author's decisions, 2026-09-20): the record is the centered Haar null (44 of 72; five constructions in the census table with the uncentered census as a column; Table 1, Figure 3 and the joint sensitivity from it); the decoupling result rewrites the depth claim with the author's wording (abstract, S1, contribution 2, thesis, S5.3, MERU, Figure 4 caption); 'certified' only for the decoupling-tested structure",
        bool(d74) and all(float(a["frac_certified"]) == 0 for a in d74 if a["real_certified"] == "True") and "Five symbols per cell" in cen_ and "expR75_census_centered_haar.csv" in cen_ and "expR75_census_centered_haar.csv" in open(TEX/"tab_census_final.tex").read()
        and bf.count("once cluster orientations are randomized, none of the four fires, and no hierarchy above the superclasses is certified in any backbone") >= 2 and "Once cluster orientations are randomized, none of the four fires, and no hierarchy above the superclasses is certified in any backbone" in bf and "hub-aligned structure in a few and no certified hierarchy above the superclasses" in bf
        and "\\textbf{Structure above the superclasses is certified in 4 of 12 ImageNet backbones; it is the alignment of each cluster with its hub, not a hierarchy among the hubs.}" in bf and "no such structure either" in bf and "certified against its matched star" in bf
        and "hub--offset" not in bf and bf.count("certified hierarchy") == bf.count("no certified hierarchy") and "centering term" not in bf and "reproduces the centered spectrum exactly" in bf and "decoupled $z$" in dep and "expR66c_joint_sensitivity_summary.csv" in rob)
    chk("final (4th review): depth table with two-decimal z everywhere and the K = 10/30/60 sweep with the balanced frame; DBpedia supremum columns under the Haar null; no OLMo-7B row; no expR32 bootstrap row",
        not _re.search(r"\(([+-]\d\.\d)\)", dep) and not _re.search(r"\$[+-]\d\.\d\$", dep) and "$K{=}10$" in dep and "$K{=}60$" in dep and "supremum, Haar" in wn and "supremum, Gaussian" not in wn
        and "OLMo-7B" not in txt_ and "OLMo-7B" not in pan_ and "centroid bootstrap: excess" not in rob and "expR32" not in rob and "expR73" in rob and "shrinks with the number of classes" in rob and rob.count("DINOv2-L & ") >= 5)
    alltex = nocom(TF) + "".join(nocom((FD/(s + ".tex")).read_text()) for s in inputs) + nocom(open(TEX/"tab_census_final.tex").read())
    refs = set(_re.findall(r"\\(?:eq)?ref\{([^}]*)\}", alltex)); defs = set(_re.findall(r"\\label\{([^}]*)\}", alltex))
    chk("final: every cross-reference of the final resolves", refs <= defs, str(sorted(refs - defs)))
    tcf = open(TEX/"tab_census_final.tex").read()
    chk("final: Table 1 is the census on the centered Haar record with the short caption (tab_census_final from expR75, twelve rows, same layout as v1's tab_census)", "\\input{tab_census_final}" in bf and "expR75_census_centered_haar.csv" in tcf and tcf.count("\\\\") >= 13 and tcf.split("\\midrule")[0] == open(TEX/"tab_census.tex").read().split("\\midrule")[0])
    # ---- statements
    AI = "In this work, we used generative AI tools for writing assistance; for the retrieval of references; and for research ideation or execution, namely the implementation and execution of experiments under the authors' direction and LLM-simulated reviews used as methodological feedback. We have not used generative AI tools for other tasks with required disclosure. We have reviewed all AI-assisted work, and we take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI."
    chk("final: AI Use Statement with exactly the three declared items and the responsibility sentence; Reproducibility with the anonymized-repository placeholder; Ethics present", AI in stm and "anonymized repository" in stm and "TODO(author)" in stm and "Ethics Statement" in stm)
    chk("final: preamble of the frozen v1 plus amsthm only (same class, same packages)", TF[:TF.index("\\usepackage{amsthm}")] == T1[:T1.index("\\usepackage{array}\n") + len("\\usepackage{array}\n")] and TF.count("\\usepackage") == T1.count("\\usepackage") + 1)
final_checks()
n_fail = sum(1 for _,ok,_ in checks if not ok)
for name, ok, det in checks[-30:]: print(("PASS" if ok else "FAIL"), name, ("| "+det if det and not ok else ""))
print(f"[phaseB re-total] {len(checks)-n_fail}/{len(checks)}")
for name, ok, det in checks:
    if not ok: print("FAIL(all):", name, ("| "+det if det else ""))
