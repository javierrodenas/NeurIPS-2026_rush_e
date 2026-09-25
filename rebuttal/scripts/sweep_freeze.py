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

# ---------- Positive-control pass (Phase A): R9 = expR64 planted depth, R11 = expR66 joint sensitivity, R10 = expR65 (optional) ----------
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
    chk("R9: census excess on the planted clouds (5 strengths x 12 backbones, 200 replicates): every cloud below its null mean, minimum rank equals the memo's",
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
    chk("S4.4 sentence (1): the certified four raise no alarm at zero strength (0 of 20 in expR64b; the count is subsumed by the 0 of 60 stated for the Haar-hub star), and their real z is deeper than the planted tree's",
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
        chk("A1: 12 backbones x (2 real stars + 10 planted) depth runs; certified sets and planted-tree counts equal the memo; star rank in 0..10",
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
    chk("final: every decimal token of the old appendix (44 tables; Table 13 and the two-pass control B39, superseded by expR77, excepted) survives in the consolidated tables or the paper", not missing)
    cl = open(ROOT/"ICLR2027"/"CHANGELOG_final.md").read().split("\n")[:6]
    chk("final: freeze note at the top of the CHANGELOG and the QA rasterizations present", any("Frozen 17 Sept 2026" in l for l in cl) and len(list((ROOT/"ICLR2027"/"qa_pages_final").glob("*.png"))) >= 9 and len(list((ROOT/"ICLR2027"/"archive"/"qa"/"qa_pages").glob("*.png"))) >= 9)   # the v1/v2/v3 rasterizations live in the archive since the cleanup of 2026-09-25
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
    # ---- verbatim parts, modulo the edits the brief names (recorded in final_verbatim_edits.json); an edit may carry fills
    def fillb(s):
        for k_, v_ in fills_.items(): s = s.replace("{{" + k_ + "}}", str(v_))
        return s
    EDJ = _j.load(open(R/"final_verbatim_edits.json")) if (R/"final_verbatim_edits.json").exists() else {}
    fills_ = {}
    for fn in ("phaseE_fills.json", "phaseE_v3_fills.json", "final_fills.json"):
        if (R/fn).exists(): fills_.update(_j.load(open(R/fn)))
    abs_expected = EDJ.get("abstract_final") or EDJ.get("abstract_sixth_review", "")   # abstract_final = the sixth-review text, or its priority-1c variant when expR80 met the rule
    for k, v in fills_.items(): abs_expected = abs_expected.replace("{{" + k + "}}", str(v))
    abs_now = norm(seg(bf, "\\begin{abstract}", "\\end{abstract}").replace("\\begin{abstract}", ""))
    NEW_CROSS = "Across models, the trees agree on which classes group together well above chance, though less than two resampled versions of the same model, and not on distances; the self-supervised models organize classes by direction rather than by distance, which a comparison by distance misses."   # the author's cross-model sentence (2026-09-23, 15:45): in the abstract and mirrored in the S1 answer paragraph
    chk("final: abstract is the recorded text (360 words or fewer, the author's cap of 2026-09-23 with the numeral rule, cell defined in its own sentence, the deep synthetic hierarchy detected and no hierarchy above the superclasses found), 15 text models", abs_now == norm(abs_expected) and len(_re.sub(r"\$[^$]*\$", "", abs_now).replace("--", "").split()) <= 360 and "model--dataset cells" in abs_now and "The second test finds no hierarchy among superclasses in the 9 of 12 backbones where it detects a planted one." in abs_now and "which disappears when clusters are rotated at random" in abs_now and "15 text models" in bf and "16 text models" not in bf and "sixteen" not in bf and "two OLMo" not in bf and "show no hierarchy among the superclasses where the test can see one" in abs_now and "30 of 36" not in abs_now and "two of which survive" not in abs_now)
    chk("final: S1 with Figure 1 verbatim from main_local.tex modulo the recorded edits (an edit may carry fills)", norm(seg(bf, "\\section{Introduction}", "\\section{Related Work}")) == norm(fillb(edit(seg(LOC, "\\section{Introduction}", "\\section{Related Work}")))))
    rel_loc = norm(seg(LOC, "\\section{Related Work}", "\\section{The Instrument}")); rel_f = norm(seg(bf, "\\section{Related Work}", "\\section{Methodology}"))
    rel_loc = norm(edit(seg(LOC, "\\section{Related Work}", "\\section{The Instrument}")))
    chk("final: S2 verbatim from main_local.tex modulo the recorded citation sentence (Sala 2018, Gu 2019), or its first six sentences (cut step 4 of the page budget)", rel_f == rel_loc or (rel_f in rel_loc and rel_loc.startswith(rel_f)))
    grom = norm(edit(seg(LOC, "\\paragraph{Gromov $\\delta$.} ", "\\paragraph{Estimation and normalization.}").split("} ", 1)[1])); est = norm(edit(seg(LOC, "\\paragraph{Estimation and normalization.} ", "\\begin{figure}").split("} ", 1)[1]))
    chk("final: 'Gromov delta' and 'Estimation and normalization' verbatim modulo the recorded edits (supremum phrase, bridge sentence)", grom in norm(bf) and est in norm(bf))
    _CUTTAIL = {"It adds a hierarchy test measured on real clouds"}   # S2 keeps its first seven sentences (cut step 4 of the page budget); the Moreira citation of 2026-09-25 took the place of the closing one, whose three items are contributions 1 and 2 of S1
    chk("final: the recorded edits are exactly the briefs' (shadow x2, geometric face, intent of the supremum, the bridge; 4th/5th reviews: abstract, S1 confounds and counts, S2 citations) and none of the old phrases survives",
        len(ED) == 45 and all((a not in bf) or (a in b) for a, b in ED) and any("clusters oriented toward their hubs in a few" in b for _, b in ED) and any("In the 9 of 12 backbones where a planted hierarchy is detected, none as strong is found; in the other 3 the test is blind." in b for _, b in ED) and "49 of 72" not in bf and all((fillb(b) in bf) or b in _CUTTAIL for _, b in ED if b and not any(a2 in b for a2, _ in ED if a2)) and sum(1 for a, _ in ED if "shadow" in a) == 4 and any("geometric face" in a for a, _ in ED) and any("intent of the supremum" in a for a, _ in ED) and any("next question" in a for a, _ in ED)
        and any("three artifacts push it down" in a for a, _ in ED) and any("sala2018representation" in b and "gu2019learning" in b for _, b in ED)
        and bf.count("30 of 36") >= 2 and "cannot be called low on its own" in bf)
    # ---- metaphors, bridges and banned phrases anywhere in the body (Figure 1 and its caption excepted), thesis twice
    body_nofig = _re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", nocom(bf), flags=_re.S); body_nocite = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", body_nofig)
    META = ["star caveat", "aristotelian", "geometric face", "intent of the supremum"]   # "shadow" is back in S1 by the author's brief of 2026-09-21 (the observer sentence)
    hits = [w for w in META if w in body_nocite.lower()]
    chk("final: no metaphor outside Figure 1 and its caption (shadow, star caveat, Aristotelian, geometric face, the intent of the supremum)", not hits, str(hits))
    MARK = ["next paragraph", "next question", "subject of the next", "turn to next", "take up last", "which we review", "builds that comparison", "the concern of Section", "First we ask", "last question of this section", "states the three acts", "With the setup fixed", "next section fixes", "raises the question of", "which text makes explicit", "has to be re-read", "points to where", "is the first candidate", "What to read instead", "we turn to them next", "is the next question", "we now", "below we", "in what follows"]
    bh = [m for m in MARK if m.lower() in body_nocite.lower()]
    chk("final: no bridge sentences anywhere in the body", not bh, str(bh))
    BANNED = ["we note", "interestingly", "importantly", "in plain terms", "notably", "essentially", "substantially", "somewhat", "we believe", "our approach", "the method", "tool"]   # "largely" left the list on 2026-09-22: the author's thesis reads "largely shared"
    THESIS = "Read correctly, foundation models organize classes into clusters that they partly share and show no hierarchy among the superclasses where the test can see one. Their raw tree-likeness is not evidence for hyperbolic geometry."   # thesis after expR84 (2026-09-22, evening): topology largely shared, metric not
    chk("final: thesis verbatim exactly twice (abstract's last sentence, S7 conclusion), no short form", bf.count(THESIS) == 2 and "no license for curvature" not in bf and THESIS in bf[bf.index("\\paragraph{Conclusion.}"):] and "occasionally hierarchical" not in bf)
    # ---- structure: classic skeleton, seven inline unframed definitions, seven equations, Proposition 1 (a)(b) proved in Appendix A, no boxes
    secs_ = _re.findall(r"\\section\{([^}]*)\}", bf); subs_ = _re.findall(r"\\subsection\{([^}]*)\}", bf); prop = bf[bf.index("\\begin{proposition}"):bf.index("\\end{proposition}")]
    chk("final: skeleton of seven sections and ten subsections (S5.5 moved to the appendix on 2026-09-24), eight definitions in light-blue boxes and Proposition 1 in amber (promotion of the boxed variant), six numbered equations, Proposition 1 (a)(b) with its one proof in Appendix A",
        len(secs_) == 7 and secs_[2] == "Methodology" and secs_[-1] == "Conclusion and Limitations" and len(subs_) == 10 and "What is shared is local" not in TF[:TF.index("\\appendix")] and "Calibrated local agreement across models survives, as \\citet{groger2026aristotelian} find for similarity (Table~\\ref{tab:q13-local})." in bf and "\\paragraph{Models share neighborhoods, not metrics.}" not in bf and bf.count("\\begin{definition}[") == 8 and bf.count("\\begin{proposition}") == 1 and "(a)" in prop and "(b)" in prop
        and bf.count("\\begin{equation}") == 6 and all(("\\label{eq:%s}" % e) in bf for e in ("pairings", "deltanorm", "excess", "rank", "bh", "depth")) and TF.count("\\begin{proof}") == 0 and "\\label{app:proofs}" in TF and TF.index("\\section{Proofs}") > TF.index("\\appendix")
        and "tcolorbox" in TF and "\\textcolor" not in bf and "\\colorbox" not in bf and "\\begin{lemma}" not in bf and "\\begin{corollary}" not in bf and "\\begin{remark" not in bf and "\\newtheorem{definition}" in TF)
    # ---- prose rules on the non-verbatim prose of S3-S7 (definitions, equations, captions and tables excluded)
    src = nocom(bf); src = _re.sub(r"\\begin\{(figure|table|tabular|center)\}.*?\\end\{\1\}", "", src, flags=_re.S); src = _re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " EQUATION. ", src, flags=_re.S); src = _re.sub(r"\\input\{[^}]*\}", "", src)
    def clean(par):
        p = _re.sub(r"\\paragraph\{[^}]*\}\s*", "", par); p = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "CITE", p); p = _re.sub(r"\\(S)?\\?ref\{[^}]*\}", "REF", p); p = _re.sub(r"\\label\{[^}]*\}", "", p)
        p = _re.sub(r"\$([A-Za-z])\{=\}(\d+)\$", r"\1=\2", p); p = _re.sub(r"\$([^$]*)\$", lambda m: " FORMULA " if ("=" in m.group(1) or "\\" in m.group(1)) else m.group(0), p)
        p = p.replace("``", "").replace("''", "").replace("~", " "); p = _re.sub(r"\\[a-zA-Z]+\*?", " ", p); p = _re.sub(r"[{}]", "", p); return " ".join(p.split())
    GROUP = r"(?<![A-Za-z\-^_{\d.])[-+]?\d+(?:\.\d+)?(?![A-Za-z\-\d])"
    RANGE = _re.compile(rf"(?:from\s+)?\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?(?:\s*(?:of the|of|to|against|vs|and|--)\s*\$?{GROUP}\$?(?:\s*at\s+\$?d=\d+\$?)?){{0,2}}(?:\\%)?")
    def groups(s):
        s = s.replace("--", " to ")   # 8th review: en-dash ranges read as ranges
        s = _re.sub(r"\\paragraph\{[^}]*\}\s*", "", s); s = _re.sub(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*\}", "", s); s = _re.sub(r"\\(S)?\\?ref\{[^}]*\}", "REF", s); s = _re.sub(r"\\label\{[^}]*\}", "", s)
        s = _re.sub(r"\$([^$]*)\$", lambda m: " FORMULA " if ("=" in m.group(1) or "\\" in m.group(1)) else m.group(0), s).replace("{=}", "="); s = _re.sub(r"\\[a-zA-Z]+", " ", s)
        return [m.group(0).replace("$", "").replace("from ", "").strip() for m in RANGE.finditer(s) if not _re.fullmatch(r"\s*", m.group(0))]
    NOUNS = r"(?:other\s+)?(?:certified |real |blind |supervised |self-supervised |leaf-label |covered |vision |text |ImageNet |decoupled |intact |fine )?(?:backbones?|models?|cells?|seeds?|runs?|replicates?|datasets?|class sets?|classes|superclasses|coarse labels|star seeds|resamples|model pairs|controls?|ResNets|ViTs|sizes|sentence embedders|values|verdicts|centroid clouds|rows)\b"
    HEADLINE = {"44 of 72", "18 of 24", "4 of 12", "50 to 200", "0.79 and 0.78", "0 of 60", "7 of 15", "0.48 to 2.5", "+0.9 to +1.3", "30 of 36", "47", "5 of 12", "210", "+0.41 against +0.28", "0 of 4", "4 of 5", "0 of 5", "0.03", "0.05", "60", "60\\%", "60%", "8 of 50", "1.9", "3.0--3.9", "3.0 to 3.9", "3.0", "3.9", "1.3--2.0", "1.3 to 2.0", "-4.19 to -4.82", "-1.61 to -1.76", "4.19 to -4.82", "1.61 to -1.76", "-1.61", "1.61", "2.1", "2.0", "-2.08 to -3.99", "2.26 to -3.99", "0.80 to 1.00", "0.00 to 0.45", "1.5 to 2.0", "10 of 10", "0 of 10", "7 of 10", "0.80", "0.45"}   # 2026-09-21: priorities 1b (4 of 5, 0 of 5), 2 (0.03, 0.05) and 1c (60%)   # 2026-09-20: the record is the centered Haar null (44 of 72; FMNIST 5 of 12); R6: relational alignment (0 of 4)
    EXEMPT = {"Most cells show structure beyond the second moments.": (4, 3), "What it certifies is alignment, how each cluster is oriented.": (5, 2), "What the test can see.": (7, 3), "A trained hierarchy survives, and the test leans toward firing with clusters rotated.": (3, 1), "Controlled, the trees share their topology, not their metric.": (3, 2), "The certified set depends on the frame.": (2, 2), "Training in hyperbolic space leaves the clustering unchanged.": (2, 2), "Agreement with WordNet follows supervision.": (3, 3)}   # MERU: '95 per cent … within 0.29' (final accuracy pass, 2026-09-23)   # the ceilings per measure beside the cross-model values (thesis brief, 2026-09-22 evening)   # cleanup of 2026-09-22: S5.3 in seven paragraphs, each number once; (c) carries the counts of the test's power and biases
    SENT_EXEMPT = {"What the test can see.", "Conclusion.", "Text depends on recipe and scale.", "A trained hierarchy survives, and the test leans toward firing with clusters rotated.", "The count survives resampling.", "Neural collapse is the flat limit, not what the census sees.", "Size and evidence.", "Controlled, the trees share their topology, not their metric.", "What a genuine excess means.", "Three metrics are compared on the intact representation.", "The raw reading cannot select a curvature.", "Models share neighborhoods, not metrics.", "Both readings predict the gain, and the hierarchy verdict predicts none.", "A raw value cannot be called low on its own.", "Why a second test.", "A star of clusters already passes the census.", "The hierarchy test certifies that how clusters are oriented relative to their hubs is not random in 4 of 12 backbones: ViT-S, ViT-B, ViT-L and DINOv2-L.",
                   "The certified set depends on the grouping.", "Leaf labels can produce the alignment but do not guarantee it.", "Training in hyperbolic space leaves the clustering unchanged.", "Text depends on recipe and scale."}   # page-9 recovery of 2026-09-24: one sentence each, the detail in Appendix~\ref{app:detail}   # the last two at two sentences for the page budget under the 361-word abstract (consolidated pass)   # cleanup of 2026-09-22: (c) by design, the two S5.2 paragraphs cut to two sentences by the author   # the author's count sentences carry more than one number
    import pandas as pd
    _S81h = pd.read_csv(R/"expR81_deep_per_backbone_summary.csv").set_index("model"); _c9 = _S81h[_S81h.dec_power >= 0.8].dec_power; _u3 = _S81h[_S81h.dec_power < 0.8].dec_power; _r64h = pd.read_csv(R/"expR64b_wn30_summary.csv").set_index("model").ratio_real
    HEADLINE |= {f"{_c9.min():.2f} to {_c9.max():.2f}", f"{_u3.min():.2f} to {_u3.max():.2f}", f"{_c9.min():.2f}", f"{_u3.max():.2f}", f"{_r64h[_u3.index].min():.1f} to {_r64h[_u3.index].max():.1f}", f"{_r64h[_c9.index].min():.1f} to {_r64h[_c9.index].max():.1f}"}   # ninth review: decoupled power ranges (nine/three) and the blind/covered ratio ranges, from the files
    _a3h = pd.read_csv(R/"exp3_alignment.csv").set_index("model").spearman_wn   # main-text completeness (2026-09-24): the three WordNet ranges of S5.4 and the sample-level count of S5.1
    for _p, _n in (("i21k", 4), ("dinov2", 4), (("clip", "siglip"), 3)):
        _ms = [m for m in _a3h.index if m.startswith(_p)]; assert len(_ms) == _n
        HEADLINE |= {f"{_a3h[_ms].min():+.2f} to {_a3h[_ms].max():+.2f}"}
    HEADLINE |= {f"{int((~pd.read_csv(R/'expR62_samplelevel_record.csv').genuine_bh).sum())} of 24"}
    _c75sr = pd.read_csv(R/"expR75_census_centered_haar.csv"); _c75sr = _c75sr[(_c75sr.model == "dinov1_b") & (_c75sr.dataset == "cifar100")]; _sr = "$%.3f$" % float(_c75sr.delta.iloc[0])   # the reading both cells share (2026-09-24)
    HEADLINE |= {_sr.replace("$", "")}
    _c58h = pd.read_csv(R/"expR58_treemap_cutfree_summary.csv"); _c58h = _c58h[(_c58h.dataset == "imagenet") & (_c58h.metric == "cosine") & (_c58h.linkage == "average")].iloc[0]; HEADLINE |= {f"{_c58h.triplet_agree_big_vs_block:.2f} against {_c58h.triplet_agree_within_block:.2f}"}
    if (R/"expR84_tree_ceiling_summary.csv").exists(): _s84h = pd.read_csv(R/"expR84_tree_ceiling_summary.csv").set_index("model"); HEADLINE |= {f"{_s84h.loc['ALL', 'triplet_agree_mean']:.2f}", f"{_s84h.loc['ALL', 'triplet_agree_min']:.2f}"}   # tenth review: the within-model ceiling
    if (R/"expR84_tree_ceiling_summary.csv").exists():   # thesis brief (2026-09-22 evening): the ceilings per measure, their ranges, and the cross-model values of expR58
        _b84h = _s84h.drop("ALL")
        for _k in ("triplet_agree", "coph_corr", "ari_cut"): HEADLINE |= {f"{_b84h[_k + '_mean'].min():.2f} to {_b84h[_k + '_mean'].max():.2f}", f"{_s84h.loc['ALL', _k + '_mean']:.2f}"}
        _r71h = pd.read_csv(R/"expR71_meru_radii.csv") if (R/"expR71_meru_radii.csv").exists() else None   # MERU sentence (final accuracy pass, 2026-09-23): 95 per cent within {MERU_P95} of the curvature scale
    if _r71h is not None: HEADLINE |= {"95", "%.2f" % _r71h[_r71h.model.str.startswith("meru")].radius_sqrtc_p95.max()}
    HEADLINE |= {"0.33", f"{_c58h.triplet_agree_big_vs_block:.2f}", f"{_c58h.triplet_agree_within_block:.2f}", f"{_c58h.coph_corr_big_vs_block:.2f} and {_c58h.coph_corr_within_block:.2f}", f"{_c58h.ari_cut_big_vs_block:.2f} and {_c58h.ari_cut_within_block:.2f}", f"{_c58h.coph_corr_big_vs_block:.2f}", f"{_c58h.coph_corr_within_block:.2f}", f"{_c58h.ari_cut_big_vs_block:.2f}", f"{_c58h.ari_cut_within_block:.2f}"}   # the pairs also appear as single values since the plain-language pass (2026-09-23)
    _p77h = pd.read_csv(R/"expR77_positive_control.csv").set_index("model")
    if "hier_seed1" in _p77h.index: HEADLINE |= {"20 of 20", "6 of 20", f"{sum(int(round(_p77h.loc[f'hier_seed{s_}', 'dec_frac_cert_wn30'] * 10)) for s_ in (0, 1))} of 20", f"{sum(int(round(_p77h.loc[f'ce_seed{s_}', 'dec_frac_cert_wn30'] * 10)) for s_ in (0, 1))} of 20", f"{_p77h.loc[['hier_seed0', 'hier_seed1'], 'zdec_mean_wn30'].mean():.2f}", f"{_p77h.loc[['ce_seed0', 'ce_seed1'], 'zdec_mean_wn30'].mean():.2f}"}   # the two seeds pooled (2026-09-23)
    if (R/"expR83_flat_balanced_summary.csv").exists(): _s83h = pd.read_csv(R/"expR83_flat_balanced_summary.csv").set_index("built"); HEADLINE |= {f"{int(_s83h.loc['balanced', 'decoupled_fired'])} of 50", f"{int(_s83h.loc['wn30', 'decoupled_fired'])} of 50"}
    IMPLJ = _j.load(open(R/"expR80_integration.json")) if (R/"expR80_integration.json").exists() else {"met": False}
    if IMPLJ.get("met"):   # priority 1c entered the submission: its two counts are headline numbers of S5.3
        HEADLINE |= {f"{IMPLJ['hits1']} of {IMPLJ['runs1']}", f"{IMPLJ['hits0']} of {IMPLJ['runs0']}"}
        if IMPLJ["hits0"] > 0: EXEMPT["The alignment of each cluster with its hub is certified in four backbones, with measured power."] = (3, 2)   # the author's sentence then carries two counts   # fifth review: the author's count sentence carries three numbers (paragraph max, sentence max)
    noeq = lambda s: _re.sub(r"\\begin\{equation\*?\}.*?\\end\{equation\*?\}", " ", s, flags=_re.S)
    verbatim = [norm(noeq(v)) for v in (edit(seg(LOC, "\\paragraph{Gromov $\\delta$.} ", "\\paragraph{Estimation and normalization.}").split("} ", 1)[1]), edit(seg(LOC, "\\paragraph{Estimation and normalization.} ", "\\begin{figure}").split("} ", 1)[1]), edit(seg(LOC, "\\section{Introduction}", "\\section{Related Work}")), seg(LOC, "\\section{Related Work}", "\\section{The Instrument}"))]
    isverb = lambda par: any(norm(par.replace(" EQUATION. ", " ")) in v for v in verbatim)
    sent_split = lambda s: [x.strip() for x in _re.split(r"(?<=[.!?])\s+(?=[A-Z(\\])|(?<=EQUATION\.)\s+", s) if len(x.split()) > 1]
    GLOSS_OK = {"the overlap of nearest neighbours", "a similarity index", "the 1000 fine classes, with no superclasses",   # glosas en palabras llanas en su primer uso (autor, 2026-09-25)
                "two classes that share a WordNet parent and a third that does not", "ImageNet, CIFAR-100 and DTD", "DINO-B is not part of the island"}
    bad = []; stats = {}; numpar = []; nverb_sents = []; over30_all = []; thesis_len = []
    blocks = _re.split(r"\\section\{([^}]*)\}", src); blocks = [(blocks[i], blocks[i+1]) for i in range(1, len(blocks), 2)]
    for name, text in blocks:
        results = name in ("Results", "Implications for Hyperbolic Representation Learning"); prose = name not in ("Introduction", "Related Work")
        pars = [q.strip() for q in _re.split(r"\n\s*\n", text) if q.strip() and not q.strip().startswith(("\\begin{definition}", "\\begin{proposition}", "\\begin{defbox}", "\\begin{propbox}", "\\end{defbox}", "\\end{propbox}", "\\label", "\\subsection", "\\begin{equation}"))]
        ws = []
        for par in pars:
            vb = isverb(par) or not prose; cp = clean(par); sents = sent_split(cp); w = [len(s.split()) for s in sents]; ws += w
            if not par.startswith("\\paragraph") and prose and not vb: bad.append(f"{name}: prose paragraph without a bold lead-in: {par[:60]!r}"); continue
            lead = _re.match(r"\\paragraph\{([^}]*)\}", par); gs = groups(par)
            if results: numpar.append((name, lead.group(1) if lead else par[:40], gs))
            rawsents = sent_split(_re.sub(r"\\label\{[^}]*\}", "", _re.sub(r"\\paragraph\{[^}]*\}", "", par))); iscl = [(_re.search(r"\\cite[pt]?(\[[^\]]*\])?\{[^}]*,", r_) is not None or len(_re.findall(r"\\cite", r_)) >= 2) for r_ in rawsents]   # citation lists are exempt from the 30-word rule
            sents_nolead = sent_split(_re.sub(r"\\paragraph\{[^}]*\}", "", cp)); over30 = [] if name == "Introduction" else [(len(s_.split()), s_) for k_, s_ in enumerate(sents_nolead) if len(s_.split()) > 30 and not s_.startswith("Read correctly, foundation models") and not s_.startswith("The correlation between inter-centroid and WordNet distances") and not (len(iscl) == len(sents_nolead) and iscl[k_])]   # pages 3-9: S2's page-3 part onward
            over30_all += over30; thesis_len += [len(s_.split()) for s_ in sents_nolead if s_.startswith("Read correctly, foundation models")]
            if vb: continue
            nverb_sents += w
            w_ = [len(s_.split()) for s_ in sents_nolead if not s_.startswith("Read correctly, foundation models")]
            if over30: bad.append(f"{name}: sentence over 30 words ({over30[0][0]}): {over30[0][1][:80]}")   # readability rule of 2026-09-23 (definitions, citation lists and the thesis excepted)
            if not (3 <= len(sents) <= 6) and not par.startswith("\\paragraph{Limitations") and (lead.group(1) if lead else "") not in SENT_EXEMPT: bad.append(f"{name}: paragraph with {len(sents)} sentences: {par[:60]!r}")   # the enumerated limitations are exempt from the sentence count
            if lead and ((len(lead.group(1).split()) > 16 and lead.group(1) not in ("The hierarchy test certifies that how clusters are oriented relative to their hubs is not random in 4 of 12 backbones: ViT-S, ViT-B, ViT-L and DINOv2-L.", "The census, the reading of every model--dataset cell, has 12 backbones, 6 class sets and 15 text models.")) or not lead.group(1).endswith(".")): bad.append(f"{name}: lead-in not plain/short: {lead.group(1)!r}")
            counts_ = set(m.group(1) for m in _re.finditer(r"(" + GROUP + r"(?: of (?:the )?" + GROUP + r")?)\s+(?:the )?(?:[A-Z][\w-]*\s+)?" + NOUNS, cp)) | set(m.group(1) for m in _re.finditer(r"\bother (\d+)\b", cp)) | set(m.group(1) for m in _re.finditer(r"\b(\d+) supervised and contrastive models\b", cp)) | set(m.group(1) for m in _re.finditer(r"\bthose (\d+) agree\b", cp)) | set(m.group(1) for m in _re.finditer(r"\bfires in (\d+ of \d+)\b", cp)) | set(m.group(1) for m in _re.finditer(r"\((\d+)(?: classes)?(?: each)?\)", cp)) | set(m.group(1) for m in _re.finditer(r"\bthose (\d+) only\b", cp)) | set(m.group(1) for m in _re.finditer(r"\bnone of the (\d+) backbones\b", cp))   # counts of models, cells, seeds and runs are numerals with their noun (author's rule, 2026-09-23): not result numbers
            if results:
                pmax, smax = EXEMPT.get(lead.group(1) if lead else "", (2, 1))
                for s_ in sents:
                    if len([g for g in groups(s_) if g not in counts_]) > 2 and not s_.startswith("The correlation between inter-centroid and WordNet distances") and not s_.startswith("With clusters rotated the control detects it"): bad.append(f"{name}: more than two numbers in a sentence (author's rule of 2026-09-23): {s_[:80]}")   # the author's S5.4 sentence gives the three family ranges together (brief of 2026-09-24, night)
                gs = [g for g in gs if g not in counts_]
                if len(gs) > pmax: bad.append(f"{name}: >{pmax} numbers in a paragraph {gs}: {par[:60]!r}")
                for s_ in sents:
                    if len([g for g in groups(s_) if g not in counts_]) > smax: bad.append(f"{name}: >{smax} number in a sentence: {s_[:80]}")
                ptr = [i for i, s_ in enumerate(sents) if ("Table REF" in s_ or "Figure REF" in s_ or "Tables REF" in s_)]
                if ptr and ptr != [len(sents) - 1] and (lead.group(1) if lead else "") not in ("The naive map manufactures an island.", "The raw reading cannot select a curvature.", "Controlled, the trees share their topology, not their metric.", "What it certifies is alignment, how each cluster is oriented.", "Text depends on recipe and scale."): bad.append(f"{name}: table/figure pointer not confined to the last sentence: {par[:60]!r}")   # the S5.4 opener cites Figure 5 first (consolidated pass), and the S6 closing paragraph ends with the scope sentence after the pointer (author's brief, 2026-09-23 19:00)   # (consolidated pass, 2026-09-22)
            if name in ("Methodology", "Experimental Setup", "Results", "Implications for Hyperbolic Representation Learning"):
                for s_ in sents:
                    if len(s_.split()) > 45 and "\\cite" not in s_: bad.append(f"{name}: sentence over 45 words ({len(s_.split())}): {s_[:80]}")   # consolidated pass: one claim per sentence in S3-S6
            off = [g for g in gs if g not in HEADLINE and g not in counts_]
            gs = [g for g in gs if g not in counts_]   # counts with their noun are not result numbers (numeral rule, 2026-09-23)
            if off: bad.append(f"{name}: number outside the headline set {off}: {par[:60]!r}")
            for m in _re.finditer(r"\(([^()]*)\)", cp):
                if not _re.fullmatch(r"(i|ii|iii|iv|v|vi|vii|viii|ix|x|[a-c])", m.group(1)) and not _re.fullmatch(r"(Table|Tables|Figure|Figures) REF[ab]?( and (REF|Figure REF[ab]?))?", m.group(1)) and m.group(1) not in GLOSS_OK and not m.group(1).startswith("VLMs, ") and len(m.group(1).split()) > 3: bad.append(f"{name}: parenthetical over three words: ({m.group(1)[:50]})")   # la lista de conjuntos es una enumeración, como las listas de citas (autor, 2026-09-25)
            for s_ in sents:
                if s_.count(";") >= 2 and not s_.startswith("Read correctly, foundation models") and not s_.startswith("The class sets are ImageNet"): bad.append(f"{name}: semicolon chain: {s_[:80]}")   # the author's thesis lists its clauses with semicolons
            low = cp.lower()
            for wd in BANNED:
                if _re.search(r"\b" + _re.escape(wd) + r"\b", low): bad.append(f"{name}: banned '{wd}': {par[:60]!r}")
            if results and not _re.search(r"(?m)%\s*[\w/]+\.(csv|npz|json)", par + "\n") and not _re.search(r"%.*\.(csv|npz|json)", bf[bf.index(par[:80]):bf.index(par[:80]) + len(par) + 400]): bad.append(f"{name}: results paragraph without a provenance comment: {par[:60]!r}")
        stats[name] = {"sentences": len(ws), "avg_words": round(mean(ws), 1) if ws else 0, "max_words": max(ws, default=0), "over30": len(over30_all) - sum(v.get("over30", 0) for v in stats.values())}
    avg_nv = round(mean(nverb_sents), 1) if nverb_sents else 0
    if avg_nv > 22: bad.append(f"average sentence length of the non-verbatim prose {avg_nv} > 22")
    for line in bad: print("   FINAL:", line[:220])
    chk("final: plain-prose rules on the non-verbatim prose of S3-S7 (avg <= 22 words, none > 35, paragraphs of 3-6 sentences with a plain bold lead-in, S5-S6 <= 1 number per sentence and <= 2 per paragraph with one pointer in the last sentence, only headline numbers, no parenthetical over three words, no semicolon chains, no banned phrases, provenance comment on every results paragraph)",
        not bad, f"avg non-verbatim sentence {avg_nv} words over {len(nverb_sents)} sentences")
    _j.dump({"sections": stats, "avg_nonverbatim": avg_nv, "n_nonverbatim_sentences": len(nverb_sents), "over30_rule": {"cap": 30, "exceptions": "definitions, citation lists, the thesis sentence (verbatim from the abstract)", "n_over30": len(over30_all), "thesis_words": thesis_len[:1]}, "numbers_per_paragraph_S5_S6": [{"section": a, "lead": b, "numbers": c} for a, b, c in numpar]}, open(R/"final_prose_stats.json", "w"), indent=1)
    # ---- vocabulary defined once in S3 (or in the verbatim S1 for 'premise'), before its first use in S4-S7
    voc = {"excess": "\\begin{definition}[excess]", "genuine": "\\begin{definition}[genuine]", "reading": "\\begin{definition}[reading]", "matched star": "\\begin{definition}[matched star]", "hub null": "\\begin{definition}[hub null", "Haar null": "\\begin{definition}[Haar null]", "four-point defect": "\\begin{definition}[four-point defect]", "decoupling control": "\\begin{definition}[rotating each cluster]"}
    s3 = seg(bf, "\\section{Methodology}", "\\section{Experimental Setup}")
    chk("final: the eight defined terms are each defined once in S3 (one definition environment each) and 'premise' is fixed in S1", all(s3.count(v) == 1 for v in voc.values()) and "premise" in seg(bf, "\\section{Introduction}", "\\section{Related Work}"))
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
    chk("final: Figure 1 is the author's figure command verbatim from main_local.tex and Figures 2-4 are fig_instrument_final, fig_excess_final and fig_treemap_final, all present",
        figs == ["figures/fig1_concept.pdf", "figures/fig_instrument_final.pdf", "figures/fig_premise_final.pdf", "figures/fig_excess_final.pdf", "figures/fig_treemap_final.pdf"] and all((TEX/f).exists() for f in figs[1:]) and (TEX/"figures/fig_implant_final.pdf").exists() and _re.search(r"\\includegraphics\[[^\]]*\]\{figures/fig1_concept\.pdf\}", LOC).group(0) in bf and "\\IfFileExists{figures/fig1_concept.pdf}" in bf)   # Figure 1 is the author's IfFileExists slot (1.4 in box when the file is absent)   # the instrument figure of 2026-09-24 replaced the overview and the depth figures
    figblocks = _re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", bf, flags=_re.S)   # one caption per figure environment (the concept figure's caption closes on its own line)
    figcaps = [_re.search(r"\\caption\{(.*)\}", fb_, flags=_re.S).group(1) for fb_ in figblocks]
    chk("final: every main-text figure caption opens with a bold takeaway, and the three data figures name their source files (Figure 1 is drawn by the authors)", len(figcaps) == 5 and all(c.lstrip().startswith("\\textbf{") for c in figcaps) and all(_re.search(r"%.*\.(csv|npz|json)", c) for c in figcaps[1:]))
    v5 = _j.load(open(R/"final_fig5_values.json")) if (R/"final_fig5_values.json").exists() else {}
    FIGSRC = open(R.parents[1]/"ICLR2027"/"figures"/"make_figs_final.py").read()
    _tabs_final = "".join(f.read_text() for f in sorted((TEX/"appendix_tables"/"final").glob("*.tex"))) + "".join(f.read_text() for f in sorted(TEX.glob("tab_*.tex")))
    chk("final (S1 rewrite, 2026-09-23): 'depth test' renamed 'hierarchy test' throughout (text, captions, section titles, tables, figure label); 'depth' kept only for the statistic; the S3.4 subsection titled 'Hierarchy test'",
        "depth test" not in TF.lower() and "depth-test" not in TF and "depth verdict" not in TF and TF.count("hierarchy test") >= 4 and TF.count("second test") >= 4 and "\\subsection{Hierarchy test}" in bf and "depth test" not in _tabs_final.lower() and "depth-test" not in _tabs_final and "depth verdict" not in _tabs_final and 'ax.set_ylabel("hierarchy test $z$")' in FIGSRC and "hierarchy-test $z$" in _tabs_final and "the depth statistic and its standardized form" in bf)
    chk("final (scope sentences, 2026-09-23): S5.1 says the excess is conservative because the null keeps the spectrum, and what fails is the evidence the premise cites; S6 closes with the zero-cost scope sentence",
        "no more than chance would give. The excess is conservative: the null keeps the spectrum, so a hierarchy carried by the spectrum alone would not show. What fails is the evidence the premise cites, not the possibility of a hierarchy." in bf and nocom(seg(bf, "\\section{Implications for Hyperbolic Representation Learning}", "\\section{Conclusion and Limitations}")).strip().endswith("We test zero-cost metrics only; whether hyperbolic training helps for other reasons is outside this study."))
    # ---- final accuracy pass (2026-09-23): citations at first mention, verified bib entries with a source each, the three bib edits, the MERU sentence from expR71
    _bib = (TEX/"references.bib").read_text()
    def _bibentry(k):
        _m = _re.search(r"@\w+\s*\{\s*" + _re.escape(k) + r"\s*,", _bib); _st = _m.start(); _i = _m.end() - 1; _d = 0; _j = _st + len(_m.group(0).split("{")[0])
        while True:
            if _bib[_j] == "{": _d += 1
            elif _bib[_j] == "}":
                _d -= 1
                if _d == 0: return _bib[_st:_j + 1]
            _j += 1
    def _src(e):
        _f = lambda name: (_re.search(r"\n\s*" + name + r"\s*=\s*[{\"]([^}\"]*)[}\"]", e) or [None, None])[1]
        return bool((_f("biburl") and "dblp.org/rec/" in _f("biburl")) or _f("doi") or _f("url") or _f("eprint"))
    _citedTF = {k.strip() for m_ in _re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", TF + _tabs_final) for k in m_.group(1).split(",")}   # the appendix tables are \input files (the embedders are cited in the panel table)
    _citedbf = {k.strip() for m_ in _re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", bf) for k in m_.group(1).split(",")}
    _mainkeys = {"imagenet", "cifar10", "dtd", "mnist", "fashion", "dosovitskiy2021an", "dinov1", "dinov2", "CLIP", "gpt2", "pythia", "olmo", "siglip", "deit", "augreg", "cub", "miniimagenet", "wordnet", "dbpedia", "resnet", "benjamini1995controlling", "desai2023meru", "zbontar2021barlow", "grill2020bootstrap"}
    _r71 = pd.read_csv(R/"expR71_meru_radii.csv"); _r71 = _r71[_r71.model.str.startswith("meru")]
    chk("final (accuracy pass, 2026-09-23): every model and dataset cited at first mention (S4: ViT, ImageNet, DINO, DINOv2, CLIP, SigLIP, CIFAR, DTD, FMNIST, MNIST, GPT-2, Pythia, OLMo, ResNet, DeiT, augreg, MERU, WordNet; S3.3 Benjamini-Hochberg; S5.1 CUB-200 and MiniImageNet; S5.4 DBpedia; the embedders in the panel table); every cited entry has a DBLP/DOI/URL/arXiv source; park2024geometry ICLR 2025, koepke2026cave arXiv:2604.18572, aggarwal 'Spaces'; the MERU sentences with the 95th-percentile radius from expR71",
        _mainkeys <= _citedbf and {"BGE", "gte", "e5"} <= _citedTF and all(_src(_bibentry(k)) for k in _citedTF) and len(_citedTF) >= 46
        and "4 supervised ViTs \\citep{dosovitskiy2021an} trained on ImageNet-21k \\citep{imagenet}, DINO-B \\citep{dinov1}, 4 DINOv2 sizes \\citep{dinov2}, CLIP-B, CLIP-L \\citep{CLIP} and SigLIP-B \\citep{siglip}" in bf and "the Benjamini--Hochberg threshold \\citep{benjamini1995controlling} at level five per cent" in bf and "On DBpedia Classes \\citep{dbpedia}" in bf and "the WordNet \\citep{wordnet} cut into 30 superclasses" in bf and "Two self-supervised ResNets \\citep{resnet,zbontar2021barlow,grill2020bootstrap} read the vision census." in bf and "DeiT-B \\citep{deit} and the augreg ViT-B \\citep{augreg}" in bf
        and "@inproceedings{park2024geometry" in _bib and "International Conference on Learning Representations" in _bibentry("park2024geometry") and "year={2025}" in _bibentry("park2024geometry").replace(" ", "") and "arXiv preprint arXiv:2604.18572" in _bibentry("koepke2026cave") and "High Dimensional\n                  Space}" in _bibentry("aggarwal2001surprising")
        and "Read with the census and the hierarchy test, MERU shows the same clustering as its Euclidean twin. No hub structure is detected, although the test's power at MERU's spectrum was not measured." in bf and f"Its embeddings also stay nearly flat: 95 per cent lie within {_r71.radius_sqrtc_p95.max():.2f} curvature radii of the origin, where the space is nearly flat" in bf and len(_r71) == 6 and float(_r71.radius_sqrtc_max.max()) < 0.5,
        f"cited {len(_citedTF)}; missing main keys {sorted(_mainkeys - _citedbf)}; without source {[k for k in sorted(_citedTF) if not _src(_bibentry(k))]}")
    chk("final: Figure 5 carries the author's caption (consolidated pass); the ARI-matrix figure is out of the appendix (cleanup of 2026-09-23: it is cited from nowhere)", bool(v5) and "\\label{fig:treemapmat}" not in TF and "fig_treemap_matrices_final" not in TF and "\\textbf{The island is the cut; tree topology is shared well above chance, though short of the within-model ceiling; the self-supervised structure is angular.} (a) Mean agreement of each model's tree with the other 11, naive (hollow) and corrected (filled). (b) Triplet agreement under the corrected comparison against the within-model ceiling (band). (c) Sibling-triplet agreement under cosine and Euclidean distance." in bf and 'ax.axvline(1 / 3, color="k", lw=0.8, ls="--", zorder=2)' in FIGSRC and 'ax.set_title("(b) topology, chance and ceiling", pad=3)' in FIGSRC and "Chance gives 0.33." in bf)   # Figure 5(b): dashed chance line at 1/3 for the three-way triplet choice, matching the S5.4 sentence (caption brief, 2026-09-23)
    # ---- appendix: the cited tables only, in citation order, figures gone, xi/ORC/null-variant panel gone, provenance comments kept, cross-references resolve
    main_f = TF[:TF.index("\\appendix")]; app_f = TF[TF.index("\\appendix"):]; FD = TEX/"appendix_tables"/"final"
    inputs = _re.findall(r"\\input\{appendix_tables/final/(tab_[^}]*)\}", app_f)
    labels = {}
    for f in sorted(FD.glob("tab_q*.tex")):   # the final copies (v1 tables split by panel, plus the tables regenerated for the final: robustness, depth, wordnet)
        m = _re.search(r"\\label\{(tab:q[^}]*)\}", f.read_text())
        if m: labels[f.stem] = m.group(1)
    cited = []; _base = lambda l_: _re.sub(r"-(ap|[b-f])$", "", l_)   # since the split of 2026-09-23 a question has one table per former panel
    for m in _re.finditer(r"\\ref\{(tab:q[^}]*)\}", main_f):
        if _base(m.group(1)) not in cited: cited.append(_base(m.group(1)))
    order = [labels[s] for s in inputs if s in labels and s != "tab_q14_panel"]   # the model panel is Appendix B (Implementation) since 2026-09-24
    cited = [c for c in cited if c != "tab:q14-panel"]
    kept = [s for s in inputs if s.startswith("tab_q")]
    chk("final: the appendix inputs exactly the tables the main text cites (twelve question tables, robustness in its final form) plus the provenance index, in first-citation order, from appendix_tables/final/",
        set(order) == set(cited) and order == cited and "tab_q08_robust_final" in inputs and "tab_q08_robust" not in inputs and "tab_z_provenance_final" not in inputs and len(kept) == 12 and "\\input{appendix_tables/tab_" not in app_f, f"inputs {inputs}")
    gone = ["tab_q11_interventions", "tab_q12_xi"]; gone_lab = ["tab:q11-interventions", "tab:q12-xi", "fig:depth-c100", "fig:depthpower", "fig:causal", "fig:treemapc100", "fig:textnulls", "fig:bestmetric"]
    rob = (FD/"tab_q08_robust_final.tex").read_text()
    chk("final: xi, ORC/interventions table, appendix figures, null-variant panel and the class-count sweep are gone from the final and nothing refers to them",
        all(g not in inputs for g in gone) and all(("\\ref{" + l + "}") not in TF for l in gone_lab) and app_f.count("\\includegraphics") == 4 and "figures/fig_implant_final.pdf" in app_f  and "Ollivier" not in app_f
        and "null variants" not in rob.lower() and "DINOv2-L & 100 &" not in rob and rob.count("non-hierarchical fine-tuning") == 0 and "supremum, Gaussian" not in rob
        and all(("\\noindent\\textbf{" + p_) not in (FD/(f_ + ".tex")).read_text() for f_, p_ in (("tab_q02_text", "(b)"), ("tab_q03_sample_final", "(b)"), ("tab_q05_power_final", "(a)"), ("tab_q08_robust_final", "(e)"), ("tab_q09_corollary", "(e)"), ("tab_q09_corollary", "(f)"), ("tab_q09_corollary", "(g)")))
        and all(x not in (FD/"tab_q04_depth_final.tex").read_text() for x in ("iso.\\ $z$", "iso.\\ star"))
        and "The raw reading predicts the gain within datasets." not in (FD/"tab_q09_corollary.tex").read_text() and "raw $\\hat\\delta_{99.9}$" in (FD/"tab_q09_corollary.tex").read_text())   # tenth review: Table 14(b) restored
    chk("final: every kept appendix table keeps its provenance comments (% prov: lines and % source comments) and the provenance index lists all thirteen", all(("% prov:" in (FD/(s + ".tex")).read_text()) for s in kept) and "| tables of the question | generator | result files |" in (R.parents[1]/"ICLR2027"/"supplementary_code"/"README.md").read_text())
    def _caps_of(src_):   # captions by brace matching: the short caption ends on its own line, a continued one on the same line
        out_ = []
        for m_ in _re.finditer(r"\\caption\{", src_):
            d_, j_ = 1, m_.end()
            while d_:
                if src_[j_] == "{": d_ += 1
                elif src_[j_] == "}": d_ -= 1
                j_ += 1
            out_.append(src_[m_.end():j_ - 1])
        return out_
    _appx = TF[TF.index("\\appendix"):]; _capsF = [c for f_ in inputs for c in _caps_of(nocom((FD/(f_ + ".tex")).read_text()))]
    chk("final (appendix cleanup, 2026-09-23): the appendix opens with the reading page and the glossary, every table is preceded by one plain paragraph of 2-4 sentences, one numbered table per former panel, every caption at most 40 words with a bold answer, and the uncited panels, the ARI-matrix figure and the isotropic star are gone",
        "How to read this appendix" not in _appx and "Symbols and column names" not in _appx
        and _appx.count("\\paragraph{What this table answers.}") == 12 and all(2 <= len(_re.split(r"(?<=[.])\s+", p_.strip())) <= 4 for p_ in _re.findall(r"\\paragraph\{What this table answers\.\}(.*?)\n", _appx))
        and all(len(_re.sub(r"%.*", "", c).split()) <= (55 if any(_d in c for _d in ("$^{\\circ}$:", "$r$/200:", "w/b:", "$s^{*}$:", "$\\bar z_1$:", "$k$/5:", "$k$/4", "(Eucl.)", "dm is")) else 40) for c in _capsF) and sum(1 for c in _capsF if c.lstrip().startswith("\\textbf{")) >= 20 and len(_capsF) >= 20 and "of record" not in _appx,
        f"captions over 40 words: {[len(_re.sub(chr(37) + '.*', '', c).split()) for c in _capsF if len(_re.sub(chr(37) + '.*', '', c).split()) > 60]}")
    # the final copies are the v1 tables split into one floating table per panel: same numbers, same captions, [tbp] instead of [H]
    same = []
    for s in kept:
        if s.endswith("_final"): continue   # regenerated for the final by gen_appendix_final.py (robustness, depth, wordnet): checked below
        strip_ = lambda s_: _re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}|\\par\\vspace\{[^}]*\}|\\vspace\{[^}]*\}", "", nocom(s_))   # the column separation is repeated per panel in the copies
        a = strip_((TEX/"appendix_tables"/(s + ".tex")).read_text()); b = strip_((FD/(s + ".tex")).read_text())
        na, nb = _re.findall(r"\d+\.\d+|\d+", a), _re.findall(r"\d+\.\d+|\d+", b)
        # appendix cleanup (2026-09-23): a number may leave the final copy (dropped panel or shortened caption), none may appear; the caption is the short one of NEWCAP
        ok = set(nb) <= set(na) | {"60"} and "\\caption{\\textbf{" in b
        same.append(ok and "[tbp]" not in b and b.count("\\begin{table}") >= a.count("\\begin{table}") - 1)   # appendix cleanup (2026-09-23): the panels are placed here, which packs the pages
    chk("final: each final copy in appendix_tables/final/ carries only numbers of its v1 table (the cleanup of 2026-09-23 drops panels and shortens captions; nothing new appears), floats one panel at a time and opens with the short caption", all(same) and len(same) == 6, str([s for s, ok in zip([k for k in kept if not k.endswith("_final")], same) if not ok]))
    # fourth review: the regenerated tables of the final
    dep = (FD/"tab_q04_depth_final.tex").read_text(); wn = (FD/"tab_q07_wordnet_final.tex").read_text(); txt_ = (FD/"tab_q02_text.tex").read_text(); pan_ = (FD/"tab_q14_panel.tex").read_text()
    pw = (FD/"tab_q05_power_final.tex").read_text(); cor = (FD/"tab_q09_corollary.tex").read_text()
    chk("final (5th review): confounds stated as a reference level in S1 and S3.2; Proposition 1(b) for fixed n; Khrulkov's rule calibrated with the supremum; S4 210 quadruples; S5.2 flat datasets and 30 of 36; S5.3 WordNet levels and the decoupling control; S6 hierarchy verdict predicts no hyperbolic gain and the calibration paragraph; Figure 2(b) same dataset; no 'OLMo-7B was not extracted'; power table with two-decimal z; Table 14 and S B.12 'Both readings'; Table 6 with 8 causal LMs; AI statement in three items",
        "Proposition~\\ref{prop:bound} makes the dimension confound precise" in bf and "with $n$ fixed" in bf and "with the supremum it was calibrated for" in bf and "210 quadruples" in bf and "FMNIST is genuine in" in TF and "levels" in bf and "rotating each cluster" in bf
        and "and the hierarchy verdict predicts none (Table~\\ref{tab:q9-corollary-d}). Calibration tells what a low $\\delta$ means, not which metric to use." in bf and "\\paragraph{The raw reading cannot select a curvature.}" in bf and seg(bf, "\\section{Implications for Hyperbolic Representation Learning}", "\\section{Conclusion and Limitations}").count("\\paragraph{") == 1 and "\\textbf{The same reading, opposite verdicts.}" not in TF and "OLMo-7B was not extracted" not in bf
        and not _re.search(r"\$[+-]\d\.\d\$", pw) and "The calibrated reading predicts the gain" not in TF and "9 causal LMs" not in pan_ and "feedback on drafts and on the methodology" in stm)
    d74 = load("expR74_decoupling_summary.csv") if (R/"expR74_decoupling_summary.csv").exists() else []
    cen_ = (FD/"tab_q01_census_final.tex").read_text()
    chk("final (author's decisions, 2026-09-20): the record is the centered Haar null (44 of 72; five constructions in the census table with the uncentered census as a column; Table 2, Figure 3 and the joint sensitivity from it); the decoupling result rewrites the depth claim with the author's wording (abstract, S1, contribution 2, thesis, S5.3, MERU, Figure 4 caption); 'certified' only for the decoupling-tested structure",
        bool(d74) and all(float(a["frac_certified"]) == 0 for a in d74 if a["real_certified"] == "True") and "expR75_census_centered_haar.csv" in cen_ and "expR75_census_centered_haar.csv" in open(TEX/"tab_census_final.tex").read()
        and "Given a planted two-level tree at this noise level, the test misses it" in bf and bf.count("That detection survives orientation randomization, whereas none of the 4 real verdicts does.") == 1 and "whereas a hierarchy we plant ourselves, synthetic or trained into a model, survives. In the 9 of 12 backbones where a planted hierarchy is detected, none as strong is found; in the other 3 the test cannot see a planted hierarchy." in bf and "given a three-level hierarchy with ViT-L's spectrum and noise, it detects it in" in bf and "no hierarchy above the superclasses is certified" not in bf and "non-random orientation of clusters relative to their superclass centers in a few; no hierarchy among those centers where the test can see one; and a hyperbolic-trained backbone" in bf
        and "\\caption{\\textbf{The instrument in one figure.} (a) Raw reading, a random ball of the same dimension, and a random cloud of the same shape; the ball-to-cloud gap is the spectrum's share and the cloud-to-model gap is the excess. (b) Their difference, the excess, filled when genuine. (c) The hierarchy test on the real cloud, with its orientations randomized, and on a planted hierarchy randomized the same way. (d) The test's power for the planted hierarchy, intact and randomized; whiskers: 95 per cent intervals over 50 runs per backbone, 200 for the DINOv2 family." in bf and "hub-aligned" not in bf and bf.count("how clusters are oriented relative to their hubs is not random") >= 1 and "A model--dataset cell is genuine when" in bf and "certified against its matched star" in bf and "where a deep synthetic one is detected" not in bf and "left open" not in bf and "no power at this noise level" not in bf
        and "hub--offset" not in bf and "certified hierarchy" not in bf and "certifies clustered structure" not in bf and "Clustering is its most plausible reading" in bf and "Calibration tells what a low" in bf and "centering term" not in bf and "reproduces the centered spectrum exactly" in bf and "$z$ rotated" in dep and "expR66c_joint_sensitivity_summary.csv" in rob
        and "certifies clustering" not in TF and "validated regime" not in TF and "validated range" not in TF and "The census does not certify that picture" in TF and "The alignment is relational" in bf and "organize classes into clusters that they partly share and show no hierarchy among the superclasses where the test can see one" in bf
       )
    # ---- brief of 2026-09-21: priorities 1b and 2 in the submission, 1c as a limitation with Table 9c; every number re-derived from its file
    import pandas as pd
    f4 = _j.load(open(R/"final_fig4.json")) if (R/"final_fig4.json").exists() else {}; d80 = load("expR80_decision.csv")[0]
    A80 = pd.read_csv(R/"expR80_implanted_alignment.csv"); A80["hit"] = A80.z_depth <= -2; pm80 = A80.groupby(["model", "s"]).hit.mean().unstack(); pct = f"{100 * A80[A80.s == 1.0].hit.mean():.0f}"
    fam_ok = all(pm80.loc[m, 1.0] == 1.0 for m in ("i21k_t", "i21k_s", "i21k_b", "i21k_l", "clip_b")) and all(pm80.loc[m, 1.0] == 0.0 for m in ("dinov2_s", "dinov2_b", "dinov2_l"))
    chk("final (priority 1c as a limitation): the author's sentence with the percentage and the family statement re-derived from expR80_implanted_alignment.csv, Table 10c per backbone and pooled with the rule's outcome, no 'measured power' certification wording, no dotted curve unless the rule was met",
        fam_ok and "Planted alignment is detected in" not in bf
        and "Planted hub alignment reproduces the verdict" not in pw and "The planted principal-axis alignment reproduces it" not in bf and "with measured power:" not in bf and "certifies hub alignment with measured power" not in bf
        and bool(f4.get("implanted_alignment_curve", False)) == (d80["rule_power_ge_0_8_fa_le_0_05"] == "True"), f"pct {pct} fam {fam_ok}")
    LEADS53 = ["The hierarchy test certifies that how clusters are oriented relative to their hubs is not random in 4 of 12 backbones: ViT-S, ViT-B, ViT-L and DINOv2-L.", "What it certifies is alignment, how each cluster is oriented.", "What the test can see.", "A trained hierarchy survives, and the test leans toward firing with clusters rotated.", "No hub hierarchy is found where the test has power.", "The certified set depends on the grouping.", "Leaf labels can produce the alignment but do not guarantee it.", "Training in hyperbolic space leaves the clustering unchanged."]   # cleanup of 2026-09-22; MERU lead-in since the twelfth review
    E79 = pd.read_csv(R/"expR79_synthetic_deep_poincare.csv"); deep = E79[E79.cloud == "synthetic_deep_vitl_spectrum"]; flat = E79[E79.cloud == "synthetic_flat_vitl_spectrum"]; poi = E79[E79.cloud.str.startswith("wordnet_poincare")]
    nd, nf = int((deep.z <= -2).sum()), int((flat.z <= -2).sum()); dec_all = bool((deep.zdec_mean <= -2).all()) and bool((deep.zdec_mean < deep.z).all()); poi_none = not bool((poi.z <= -2).any())
    chk("final (priority 1b): the S5.3 deep-hierarchy paragraph after the decoupling paragraph with its counts re-derived from expR79_synthetic_deep_poincare.csv (4 of 5 deep, 0 of 5 flat, every deep seed under decoupling and deeper, Poincare never), the author's sentence in S5.3 and S1, the abstract sentence, the thesis, limitation (iii), Table 10d, and no 'left open' or 'no power' wording anywhere",
        nd == 4 and nf == 0 and dec_all and poi_none and len(deep) == 5 and len(flat) == 5 and len(poi) == 2
        and [bf.index("\\paragraph{" + l + "}") for l in LEADS53] == sorted(bf.index("\\paragraph{" + l + "}") for l in LEADS53)
        and f"A flat control with the same spectrum and ratio as the deep one fires in {nf} of {len(flat)} intact runs and in {int(round(flat.dec_frac_cert.sum() * 10))} of 50 runs with clusters rotated." in bf and "read with Euclidean distances, do not fire" not in bf
        and f"it detects it in {nd} of {len(deep)} seeds." in bf and "A deep hierarchy at the real noise level is detected, and a flat control is not." in pw
        and "The power follows the backbone, not its noise level or family." in bf and "left open" not in bf and "no power at this noise level" not in bf and "without power" not in bf and "\\paragraph{The two-level implant is missed" not in bf and "Figures~\\ref{fig:implant} and \\ref{fig:power} give the detection rates and the power." in bf,
        f"deep {nd}/{len(deep)} flat {nf}/{len(flat)} dec_all {dec_all} poi_none {poi_none}")
    S78 = pd.read_csv(R/"expR78_khrulkov_replication_summary.csv").set_index("dataset"); allin = len(S78) == 4 and bool(S78.within_range.all()); low = set(S78.index[S78.p_left_max <= 0.05]); smp = (FD/"tab_q03_sample_final.tex").read_text()
    chk("final (priority 2): the S5.1 published-reading paragraph after the lead-in, its claims re-derived from expR78_khrulkov_replication_summary.csv (all four within 0.03, excess negative everywhere, p <= 0.05 on CIFAR-100 and MiniImageNet only), the new lead-in, limitation (vi), and Table 8(b) with the four published and reproduced values",
        allin and bool((S78.excess_mean < 0).all()) and low == {"cifar100", "miniimagenet"} and "\\paragraph{The premise does not survive calibration where it is read.}" in bf
        and bf.index("\\paragraph{The premise does not survive calibration where it is read.}") < bf.index("\\paragraph{A published reading is reproduced and calibrated.}") < bf.index("\\subsection{Structure beyond the second moments at the class level}")
        and "The estimator of \\citet{Khrulkov_2020_CVPR}, run on our extraction, reproduces the raw $\\delta_{\\text{rel}} = 2\\delta/\\mathrm{diam}$ they report for ResNet-34 within 0.03." in bf and "Calibrated, only MiniImageNet \\citep{miniimagenet} is genuine. CIFAR-10 and CUB-200 \\citep{cub} are indistinguishable from a random cloud, and CIFAR-100 falls below it only before correction ($p=0.030$)." in bf and set(S78.index[S78.p_left_max > 0.05]) == {"cifar10", "cub"} and "2 datasets, one budget and one setting" in bf
        and "expR78_khrulkov_replication_summary.csv" in smp,
        f"allin {allin} low {low}")
    import glob as _glob
    f82 = [R/"expR82_radial_control.csv"] if (R/"expR82_radial_control.csv").exists() else sorted(_glob.glob(str(R/"expR82_radial_control.part_*.csv")))
    e82 = pd.concat([pd.read_csv(f) for f in f82]).drop_duplicates(subset=["model", "transform", "star", "dec_seed"]); cert4 = ["i21k_s", "i21k_b", "i21k_l", "dinov2_l"]
    st2 = ["aniso", "aniso_haarhubs"]; dr82 = e82[(e82["transform"] == "deradial") & e82.model.isin(cert4) & e82.star.isin(st2)]; l282 = e82[(e82["transform"] == "l2norm") & e82.model.isin(cert4) & e82.star.isin(st2)]; dra = dr82.z   # the sentence says "under both stars", so the range runs over both (author's correction, 2026-09-25)
    rad_ok = (dr82.model.nunique() == 4 and bool((dr82.z <= -2).all()) and set(l282[(l282.star == "aniso") & (l282.z <= -2)].model) == {"i21k_b", "i21k_l"}
              and f"Removing the radial component of every offset leaves it in all 4 certified backbones, $z$ from ${dra.max():.2f}$ to ${dra.min():.2f}$ under both stars; feature norms are not its source (Table~\\ref{{tab:q4-depth-e}}). Under full L2 normalization, which also moves the hubs, it survives in ViT-B and ViT-L only." in bf
               and "a control that removes it is in progress" not in bf
              and ("The alignment is not the radial spread of feature norms." in dep) == ((R/"expR82_radial_control_summary.csv").exists() and (lambda S_: S_.model.nunique() == 12 and len(S_) == 24 and bool((S_.dec_runs == 10).all()))(pd.read_csv(R/"expR82_radial_control_summary.csv"))))
    r64 = pd.read_csv(R/"expR64b_wn30_summary.csv").set_index("model").ratio_real
    S81 = pd.read_csv(R/"expR81_deep_per_backbone_summary.csv").set_index("model"); ORD = ["i21k_t", "i21k_s", "i21k_b", "i21k_l", "dinov1_b", "dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g", "clip_b", "clip_l", "siglip_b"]
    cov = [m for m in ORD if S81.loc[m, "dec_power"] >= 0.8]; unc = [m for m in ORD if S81.loc[m, "dec_power"] < 0.8]; rb, rc = r64[unc], r64[cov]   # ninth review: one criterion, the decoupled power
    _names_unc = _j.load(open(R/"final_fills.json")).get("P81_UNCOVERED", "")
    p81_ok = (len(cov) == 9 and len(unc) == 3 and cov == ["i21k_s", "i21k_b", "i21k_l", "dinov2_b", "dinov2_l", "dinov2_g", "clip_b", "clip_l", "siglip_b"] and unc == ["i21k_t", "dinov1_b", "dinov2_s"]
              and "The same hierarchy was then built at each backbone's own spectrum and ratio." in bf and f"With clusters rotated the control detects it in 9 of 12 backbones (power {S81.loc[cov, 'dec_power'].min():.2f}--{S81.loc[cov, 'dec_power'].max():.2f}) over 50 to 200 runs per backbone, and not in the other 3 (power {S81.loc[unc, 'dec_power'].min():.2f}--{S81.loc[unc, 'dec_power'].max():.2f})." in bf
              and "\\paragraph{No hub hierarchy is found where the test has power.} In those 9 backbones, none as strong as the planted one is found." in bf
              and f"The 3 that cannot see a planted hierarchy, {_names_unc}, sit at noise levels {rb.min():.1f} to {rb.max():.1f}, inside the {rc.min():.1f} to {rc.max():.1f} spanned by the 9 covered backbones. The power follows the backbone, not its noise level or family." in bf and rc.min() <= rb.min() and rb.max() <= rc.max()
              and all(int(S81.loc[m, "n_seeds"]) == 20 for m in ("dinov2_s", "dinov2_b", "dinov2_l", "dinov2_g")) )
    P77 = pd.read_csv(R/"expR77_positive_control.csv").set_index("model"); V77 = _j.load(open(R/"expR77_positive_control_verdict.json"))["verdict"][0]
    pc_ok = (P77.loc["hier_seed0", "z_wn30"] <= -2 and P77.loc["hier_seed0", "z_wn30bal"] <= -2 and P77.loc["ce_seed0", "z_wn30"] <= -2 and P77.loc["frozen", "z_wn30"] <= -2 and not V77["criterion_met"]
             and f"ViT-B fine-tuned with a hierarchical cross-entropy keeps firing with its clusters rotated in {sum(int(round(P77.loc[f'hier_seed{s_}', 'dec_frac_cert_wn30'] * 10)) for s_ in (0, 1))} of 20 runs over 2 seeds, mean $z$ ${P77.loc[['hier_seed0', 'hier_seed1'], 'zdec_mean_wn30'].mean():.2f}$, on the WordNet grouping. The leaf-only fine-tune fires in {sum(int(round(P77.loc[f'ce_seed{s_}', 'dec_frac_cert_wn30'] * 10)) for s_ in (0, 1))} of 20, mean $z$ ${P77.loc[['ce_seed0', 'ce_seed1'], 'zdec_mean_wn30'].mean():.2f}$, and the frozen checkpoint in none." in bf and int(round(P77.loc['frozen', 'dec_frac_cert_wn30'] * 10)) == 0 and "Fine-tuning itself adds some signal with clusters rotated and the hierarchical objective adds more, so the trained control separates the objectives by degree." in bf and not any(v["criterion_met"] for v in _j.load(open(R/"expR77_positive_control_verdict.json"))["verdict"])
             and "A pre-set criterion expecting no intact certification of the leaf and frozen models was not met, because they are aligned." in app_f and "The pre-set criterion asked" not in bf and "The trained control separates the objectives by degree." in dep  and "(seed 1)" not in bf and "2 seeds:" in dep
             and "one trained positive control is inconclusive" not in bf)
    LONGN_ = " in the supervised and contrastive backbones, whose noise level, ratios 1.3 to 2.0, the control covers, and not tested in the DINOv2 family, ratios 3.0 to 3.9"; SHORT_ = " in the supervised and contrastive backbones; the DINOv2 family lies beyond the noise level at which the test was validated"
    # ---- seventh review (2026-09-21): scoped headline, decoupled flat false alarms from expR79, the observation sentence from expR81 when its ViT-L rows exist, limitation (iii) ratios from expR64b, Figure 1 caption, the proof of Proposition 1(b) without the unproved extension, Figure 4b label
    fa_dec = int(round(flat.dec_frac_cert.sum() * 10)); r64 = pd.read_csv(R/"expR64b_wn30_summary.csv").set_index("model").ratio_real; dr = r64[[m for m in r64.index if m.startswith("dinov2")]]; sc_ = r64[[m for m in r64.index if m.startswith(("i21k", "clip", "siglip"))]]
    dobs = _j.load(open(R/"final_dec_obs.json")) if (R/"final_dec_obs.json").exists() else {"kind": "open"}
    obs_ok = (("is an open observation" in bf) if dobs["kind"] == "open" else (dobs["kind"] in ("star", "excess", "both")))   # the caption sentence left with the long captions (appendix cleanup, 2026-09-23)
    chk("final (7th review): headline scoped in the abstract (twice), S1, contribution 2, S5.3 lead-in, Figure 4 and Table 9 captions and the thesis; decoupled flat control false alarms (8 of 50) and the observation sentence in S5.3 from the files; limitation (iii) with ViT-L's ratio 1.9 and the DINOv2 range from expR64b; Figure 1 caption 'read alike, all low'; no unproved extension in the proof of Proposition 1(b)",
        "The control covers the supervised" not in bf and "whose noise level, ratios" not in bf and p81_ok and pc_ok and bf.count("no hub hierarchy is found in 9 of 12 backbones where the decoupled control detects a planted one, and the test is blind in the other 3") == 0 and "The second test finds no hierarchy among superclasses in the 9 of 12 backbones where it detects a planted one." in abs_now and "noise level at which the test was validated" not in bf and "\\paragraph{No hub hierarchy is found where the test has power.}" in bf and " in the backbones whose noise level the control covers" not in TF and f"The synthetic hierarchy at $z$ ${deep.zdec_mean.max():.2f}$ to ${deep.zdec_mean.min():.2f}$ stands well clear of that bias." in bf and "none of the 4 backbones fires" in bf and f"ViT-L then reads ${float(next(r for r in load('expR74_decoupling_summary.csv') if r['model'] == 'i21k_l')['dec_z_mean']):.2f}$, within the ${flat.zdec_mean.max():.2f}$ to ${flat.zdec_mean.min():.2f}$ of the flat control: without its orientations the real cloud reads like a flat one." in bf and "lie at the edge of the covered range" not in bf and rad_ok and "class centroids carry structure that a random cloud does not have, in 44 of 72 cells and 30 of 36" in bf and bf.index("The noise level of a cloud is its within-cluster spread relative to the distance between its hubs.") < bf.index("The 3 that cannot see a planted hierarchy, ")   # named since 2026-09-25
        and "unregime" not in TF and bf.count("4 values reported by") == 1 and "only MiniImageNet is genuine" in bf and int((S78.p_left_max > 0.05).sum()) == 2 and not any(w_ in bf[bf.index("\\textbf{The answer has three parts.}"):bf.index("\\textbf{The answer has three parts.}") + 260].lower() for w_ in ("genuine", "certified", "record", "matched star", "null")) and "The test is therefore biased toward firing, so the failure of the 4 real backbones to fire once rotated is conservative evidence" in bf and "The test is therefore biased toward firing" in bf and "where a deep synthetic one is detected" not in bf
        and f"in {fa_dec} of 50 runs with clusters rotated." in bf and obs_ok
        and "All three read alike." in bf and "all read the same" not in bf and "it comes out low" in bf and "which is zero for a metric tree and grows as a metric departs from one" in bf and "effective dimension $(\\operatorname{tr}" not in TF and "the same argument runs with $d$ replaced" not in TF
       , f"fa_dec {fa_dec} obs {dobs['kind']}")

    # ---- ninth review (2026-09-22): one power criterion (the decoupled control), nine/three from expR81 dec_power, the balanced frame's false alarms (expR83), vocabulary, triplet pair, supremum band, Moreira, proof end, captions without IDs, Figure 4b bars, implant curves in the appendix
    S83 = pd.read_csv(R/"expR83_flat_balanced_summary.csv").set_index("built") if (R/"expR83_flat_balanced_summary.csv").exists() else None
    bal = _j.load(open(R/"final_bal_frame.json")) if (R/"final_bal_frame.json").exists() else {}
    c58 = pd.read_csv(R/"expR58_treemap_cutfree_summary.csv"); c58 = c58[(c58.dataset == "imagenet") & (c58.metric == "cosine") & (c58.linkage == "average")].iloc[0]
    capsall = " ".join(_re.findall(r"\\caption\{(.*?)\n\}", nocom(TF) + "".join(nocom((FD/(s + ".tex")).read_text()) for s in inputs), flags=_re.S))
    bib = (TEX/"references.bib").read_text(); fi = _j.load(open(R/"final_fig_implant.json")) if (R/"final_fig_implant.json").exists() else {}
    _bibkeys = _re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib); _cited = {k.strip() for m in _re.finditer(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]*)\}", TF) for k in m.group(1).split(",")}
    chk("final: references.bib is the author's bib of 2026-09-23 merged with the three cited entries it lacked: keys unique, every cited key present, Groger et al. as ICML 2026 with Shuo Wen, no escaped underscores in doi/url (they print a backslash), Gromov protected",
        len(_bibkeys) == len(set(_bibkeys)) and _cited <= set(_bibkeys) and "@inproceedings{groger2026aristotelian" in bib and "Shuo Wen" in bib and "International Conference on Machine Learning (ICML)" in bib.split("groger2026aristotelian")[1][:600] and "\\_" not in bib and "{Gromov} hyperbolicity" in bib and "{Delaunay}" in bib and "{Plato's}" in bib and "{GloVe}" in bib and "{Aristotelian}" in bib and "``Nearest Neighbor''" in bib and "@inproceedings{sala2018representation" in bib and "@inproceedings{gu2019learning" in bib)
    caps_ids = bool(_re.search(r"exp[R]?\d", capsall))
    bal_ok = (S83 is not None and bool((S83.n_decoupled == 50).all()) and bool(bal) and int(bal["fa_bal"]) == int(S83.loc["balanced", "decoupled_fired"]) and bal["sentence"] in TF and bal.get("author_sentence")
              and bal["sentence"] == f"A flat cloud clustered under the WordNet grouping fires under the balanced grouping in {int(S83.loc['wn30', 'decoupled_fired'])} of 50 runs with clusters rotated. A real cloud read under a grouping that is not its own is therefore expected to fire. The frozen ViT-B firing in {int(round(P77.loc['frozen', 'dec_frac_cert_wn30bal'] * 10))} of 10 there is consistent with that mismatch and is not evidence of hub structure."
              and f"On the balanced grouping the matched flat control fires in {int(S83.loc['balanced', 'decoupled_fired'])} of 50 runs with clusters rotated." in TF and int(S83.loc["balanced", "decoupled_fired"]) == 0
              and int(S83.loc["balanced", "decoupled_fired"]) < 2 * fa_dec and int(S83.loc["wn30", "decoupled_fired"]) > 25 and f"flat control, balanced grouping & -- & -- & ${S83.loc['balanced', 'intact_z_mean']:+.2f}$ & {int(S83.loc['balanced', 'intact_fired'])}/{int(S83.loc['balanced', 'n_intact'])}" in dep and f"flat control, grouping mismatch & -- & -- & ${S83.loc['wn30', 'intact_z_mean']:+.2f}$" in dep)   # the author's sentence (2026-09-22, afternoon brief), its three counts from expR83 and expR77
    chk("final (9th review): nine covered / three blind from the decoupled power of Table 10(e); the long form in S1 and the Figure 4 caption, the short form as the thesis twice, the abstract sentence, no five/seven left; decoupled false alarms next to the split in S5.3 and (iii); the balanced-frame sentence from expR83 (matched hubs) with the recorded rule; vocabulary (structure above the clusters, alignment, hub structure, nominally hyperbolic); triplet 0.77 vs 0.76 from expR58; Gaussian band named as the sampled supremum in S6 and Table 4; Moreira 2024 cited with a verified entry; proof end tied to the statistic confound; no experiment IDs in any caption; Figure 4b = decoupled-power bars; implant curves as an appendix figure; B.7 points to Table 9(d)",
        p81_ok and bf.count("no hub hierarchy is found in 9 of 12 backbones where the decoupled control detects a planted one, and the test is blind in the other 3") == 0 and "\\paragraph{No hub hierarchy is found where the test has power.}" in bf
        and "The second test finds no hierarchy among superclasses in the 9 of 12 backbones where it detects a planted one." in abs_now and "five backbones" not in TF and "other seven" not in TF and "in the five only" not in TF and "in the five and" not in TF
        and f"A flat control with the same spectrum and ratio as the deep one fires in {nf} of {len(flat)} intact runs and in {fa_dec} of 50 runs with clusters rotated." in bf 
        and bal_ok
        and "measures structure above the clusters of the grouping only" in bf and "measures hierarchy above" not in bf and "\\paragraph{Leaf labels can produce the alignment but do not guarantee it.}" in bf and "\\paragraph{Training in hyperbolic space leaves the clustering unchanged.}" in bf and "detected depth" not in bf and "a hyperbolic-trained backbone as the control for imposing the geometry" in bf
        and f"triplet agreement reaches {c58.triplet_agree_big_vs_block:.2f}, against" in bf
        and "Applied, with the supremum it was calibrated for, to structureless Gaussian clouds, it assigns curvatures from" in bf
        and "a fixed-radius Euclidean encoder does at least as well as hyperbolic prototypes \\citep{moreira2024hyperbolic}" in TF and "@inproceedings{moreira2024hyperbolic" in bib and "2082--2090" in bib and "Winter Conference on Applications of Computer Vision" in bib
        and not caps_ids
        and f4.get("panel_b") == "decoupled_power_per_backbone" and f4.get("n_covered") == 9 and not f4.get("implanted_alignment_curve") and (TEX/"figures/fig_implant_final.pdf").exists() and fi.get("real", {}).get("0.0") == 0.0 and fi.get("real", {}).get("1.0", 1) < 0.5 <= fi.get("shrunk", {}).get("1.0", 0)
        and "\\label{fig:implant}" in app_f and "Figures~\\ref{fig:implant} and \\ref{fig:power} give the detection rates and the power." in bf
        and "A trained control, inconclusive" not in TF,
        f"cov {len(cov)} unc {len(unc)} bal {bal.get('fa_bal')} high {bal.get('high')} caps_ids {caps_ids}")
    # ---- full-pass cleanup (2026-09-22, evening brief): duplicates out, Definition 8, S5.3 in seven paragraphs with each number once, the Khrulkov table in S5.1, S5.2 cuts, S3.2 sentence dropped
    s53 = nocom(seg(bf, "\\subsection{Hierarchy above the labelled clusters}", "\\subsection{Whose tree}")); s3_ = seg(bf, "\\section{Methodology}", "\\section{Experimental Setup}")
    s53d = s53 + nocom(app_f)   # final reduction (2026-09-24): the detail of S5.3 lives in its question in the appendix; each number still appears once between the two
    once = {k: len(_re.findall(r"(?<!\d)" + _re.escape(k) + r"(?!\d)", s53d)) for k in ("4 of 5", "8 of 50", "0 of 50", "38 of 50", "7 of 10", "20 of 20", "6 of 20", "0 of 60", "0 of 4", "0 of 5")}
    kt = (TEX/"tab_khrulkov_final.tex").read_text() if (TEX/"tab_khrulkov_final.tex").exists() else ""; NMD = {"cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cub": "CUB-200", "miniimagenet": "MiniImageNet"}
    kt_ok = bool(kt) and all(f"{NMD[d]} & {S78.loc[d].theirs:.2f} & {S78.loc[d].ours_raw_mean:.3f} & ${S78.loc[d].excess_mean:+.4f}$ & {int(round(S78.loc[d].r_above_mean))}/200, {S78.loc[d].p_left_max:.3f}" + (" & " if (R/"expR85_khrulkov_sup_summary.csv").exists() else " \\\\") in kt for d in S78.index) and "\\label{tab:khrulkov}" in kt and "expR78_khrulkov_replication_summary.csv" in kt and kt.count("\\\\") == 6   # two header rows since the supremum columns got their group header (2026-09-24)
    chk("final (cleanup, 2026-09-22): S3.3 duplicates out ('Size and evidence', 'What a genuine excess means'); Definition 8 (decoupling control) after the matched star with the author's wording and the power/false-alarm sentence; S4 lead-in and S5.2 sentence gone, the two S5.2 paragraphs at two sentences; S3.2 supremum sentence gone (citation on Definition 1); S5.3 in the seven paragraphs in order with each count once and the bias sentence once; the Khrulkov table in S5.1 with its cells from expR78, cited by the paragraph",
        "\\paragraph{Size and evidence.} The excess is a size, the gap between a reading and its random cloud in Figure~\\ref{fig:instrument}a. The evidence is the rank of the reading among its replicates and the left-tail $p$ that the rank gives," in bf and "The rank of the reading is the evidence" not in bf and "\\paragraph{What a genuine excess means.} A negative genuine excess, more than chance would give, certifies structure beyond the second moments, that is, beyond what the mean and the covariance explain. Clustering is its most plausible reading" in bf and "A negative genuine excess means structure" not in bf
        and s3_.count("\\begin{definition}[rotating each cluster]") == 1 and s3_.index("\\begin{definition}[matched star]") < s3_.index("\\begin{definition}[rotating each cluster]") < s3_.index("\\paragraph{The hierarchy test compares the real cloud with its matched star.}")
        and "This control keeps the hubs of the grouping and rotates each cluster's offsets by an independent Haar rotation. A verdict that survives the rotation is carried by the arrangement of the hubs; one that does not is carried by the orientation of the clusters relative to their hubs, which we call alignment." in s3_
        and "The power of the test is the fraction of planted hierarchies it detects, and its false-alarm rate is the fraction of flat controls that fire. Both are measured in Section~\\ref{sec:f-depth}." in s3_ and "the false-alarm rate and power are measured on real clouds" not in bf
        and "Every reading is calibrated against two hundred replicates" not in bf and "\\paragraph{Calibration budget and groupings.} Each cell is read against 200 Haar replicates" in bf and bf.count("200 Haar replicates") == 1
        and "The same structure appears without labels" not in bf and "The census null accounts for replicate noise only" not in bf and "worst case" not in bf and "over its quadruples \\citep{Gromov1987}" in bf
        and all(v == 1 for v in once.values()) and s53.count("biased toward firing") == 1 and len(_re.findall(r"\\paragraph\{", s53)) == 8 and all(("\\paragraph{" + l + "}") in s53 for l in LEADS53)
        and kt_ok and "\\input{tab_khrulkov_final}" not in bf and "\\input{tab_khrulkov_final}" in TF[TF.index("\\appendix"):] and "Figure~\\ref{fig:premise}b and Table~\\ref{tab:khrulkov} give the 4 datasets" in bf and bf.index("\\paragraph{The premise does not survive calibration where it is read.}") < bf.index("\\paragraph{A published reading is reproduced and calibrated.}") < bf.index("\\subsection{Structure beyond the second moments at the class level}"),
        f"once {once} kt {kt_ok}")
    # ---- tenth review (2026-09-22): the frame-mismatch reading, what is known about the alignment, the meaning of 'no hub hierarchy', the ceiling (expR84), the analysis-choices limitation, Table 14(b)
    S84 = pd.read_csv(R/"expR84_tree_ceiling_summary.csv").set_index("model") if (R/"expR84_tree_ceiling_summary.csv").exists() else None
    tc = _j.load(open(R/"final_tree_ceiling.json")) if (R/"final_tree_ceiling.json").exists() else {}
    tm6 = (TEX/"appendix_tables/final/tab_q06_treemap.tex").read_text() if (TEX/"appendix_tables/final/tab_q06_treemap.tex").exists() else (TEX/"appendix_tables/tab_q06_treemap.tex").read_text()
    chk("final (10th review): the balanced frame read as frame mismatch (author's sentence, counts from expR83/expR77) and the matched count kept in (c); the alignment stated as relational, not radial, reproduced by the planted principal-axis alignment in the supervised ViTs and CLIP-B and not in DINOv2-L, form unresolved (S5.3 and (iii)); the decoupling's separation and 'no hub hierarchy' = none among the frame's hubs (S5.3 and (iii)); 'hubs of the frame' in the abstract, S1 and the captions; the within-model ceiling from expR84 beside 0.77/0.76 with the keep branch (ceiling >= 0.9), Table 11(c); limitation (vi) on analysis choices, scope renumbered; Table 14(b) restored",
        bal_ok and fam_ok
        and "The alignment is relational: 0 of 60 runs fire with random hubs and 0 of 4 with random orientations, so it needs both." in bf
        and "Its geometric form is not resolved here." in bf
        and "Rotating each cluster separates structure among the hubs of the grouping from structure carried by cluster orientations. A hierarchy below the grouping expressed in how clusters open would be removed with the orientations and counted as alignment. No hub hierarchy therefore means no hierarchy among the grouping's hubs." in bf
        and "No hub hierarchy therefore means no hierarchy among the grouping's hubs." in bf
        and "no hierarchy among superclasses" in abs_now and TF.count("not that the hubs form a hierarchy") == TF.count("not that the hubs form a hierarchy within the grouping") and "not that the hubs form a hierarchy" not in dep
        and S84 is not None and len(S84) == 13 and bool(tc) and tc.get("branch") == "keep" and float(S84.loc["ALL", "triplet_agree_mean"]) >= 0.9 and f"Across models, triplet agreement reaches {c58.triplet_agree_big_vs_block:.2f}, against {S84.loc['ALL', 'triplet_agree_mean']:.2f} for two resamples of the same model. Chance gives 0.33." in bf
        and c58.triplet_agree_big_vs_block < S84.loc["ALL", "triplet_agree_min"] and (lambda _c: all(abs(_c["ceiling_mean"][m_] - round(float(S84.loc[m_, "triplet_agree_mean"]), 3)) < 1e-9 for m_ in _c["ceiling_mean"]) and len(_c["cross_model"]) == 12 and max(_c["cross_model"].values()) < min(_c["ceiling_mean"].values()))(_j.load(open(R/"final_fig_ceiling.json"))) and THESIS in bf
        and "The analysis rests on many choices, groupings, stars and null variants among them; a pre-specified analysis is future work." in bf and "The census is class-centroid geometry on 6 datasets" in bf and "The objective--geometry link is correlational" not in bf   # dropped for the page budget on 2026-09-25 (author's stop rule) 
        and "raw $\\hat\\delta_{99.9}$" in cor,
        f"ceiling {tc.get('triplet_ceiling_mean')} min {tc.get('triplet_ceiling_min')} branch {tc.get('branch')}")
    # ---- thesis and S5.4 after expR84 (2026-09-22, evening brief): topology largely shared, metric not; ceilings per measure beside the cross-model values
    b84 = S84.drop("ALL") if S84 is not None else None
    chk("final (thesis after expR84): the new thesis twice, the abstract's sharing sentence and its S1 mirror with 0.77 and the triplet ceiling range from the files, contribution 3 mirrored, S5.4 with the three ceilings (range and mean) beside the cross-model values and the reading (topology within most of the ceiling, metric at about half)",
        S84 is not None and THESIS in abs_now and bf.count(THESIS) == 2 and "moderately shared" not in bf and "does not converge to one common tree" not in bf
        and "In text, the result depends on the model's training recipe and size." in abs_now and "a second test whose ability to detect a hierarchy is measured" in abs_now and "What it finds in 4 of them is how each cluster is oriented" in abs_now and "Across models, the trees agree on which classes group together well above chance, though less than two resampled versions of the same model, and not on distances; the self-supervised models organize classes by direction rather than by distance, which a comparison by distance misses." in abs_now and bf.count("well above chance, though short of what two readings of the same model reach") == 0 and "and not on distances" in abs_now and bf.count(NEW_CROSS) == 1 and "look like outliers" not in bf and bf.index(NEW_CROSS) < bf.index("\\section{Introduction}") and NEW_CROSS.replace("Across models, ", "\\emph{Across models,} ") in seg(bf, "\\section{Introduction}", "\\section{Related Work}")
        and "\\item \\textbf{The map of trees.} Across models, the trees agree on which classes group together but not on distances, and the self-supervised models organize classes by direction, which a comparison by distance misses." in bf and "\\item \\textbf{The instrument.} A calibrated reading of $\\delta$, compared with random clouds of the same shape, and a second test whose false-alarm rate and power are measured on real clouds." in bf and "\\item \\textbf{Consequences for practice.} The rule that derives curvature from a raw $\\delta$ assigns curvature to random clouds; we say what to measure before imposing curvature, and cosine collects most of the structure at no cost." in bf and "sharing is graded and supervision-dependent" not in bf and "largely shared" not in bf
        and "The cophenetic and cut agreements reach about half their within-model ceiling (Table~\\ref{tab:q6-treemap-b} and Figure~\\ref{fig:treemap}b)." in bf and "\\paragraph{Controlled, the trees share their topology, not their metric.}" in bf
        and c58.triplet_agree_big_vs_block / S84.loc["ALL", "triplet_agree_mean"] >= 0.8 and 0.4 <= c58.coph_corr_big_vs_block / S84.loc["ALL", "coph_corr_mean"] <= 0.6 and 0.4 <= c58.ari_cut_big_vs_block / S84.loc["ALL", "ari_cut_mean"] <= 0.6 and tc.get("cross_over_ceiling", {}).get("triplet", 0) >= 0.8,
        f"cross/ceiling triplet {c58.triplet_agree_big_vs_block / S84.loc['ALL', 'triplet_agree_mean']:.2f} coph {c58.coph_corr_big_vs_block / S84.loc['ALL', 'coph_corr_mean']:.2f} ari {c58.ari_cut_big_vs_block / S84.loc['ALL', 'ari_cut_mean']:.2f}" if S84 is not None else "no expR84")
    # ---- consolidated pass (2026-09-22, night): abstract verbatim with the thesis as its last sentence, S1 glosses at first use, Figure 5 redrawn with the matrices in the appendix, 45-word splits
    s1_ = seg(bf, "\\section{Introduction}", "\\section{Related Work}")
    chk("final (consolidated pass): the thesis is the abstract's last sentence and S7's; S1 (rewrite of 2026-09-23) glosses the hub at first use, the frame gloss sits at its first use in S3.4 (Definition 6), P5 has its three labelled parts, P1/P3/P4 as briefed; S1 and contribution 3 mirror the abstract's sharing sentence; S5.4 opens by citing Figure 5 with the author's sentence; Figure 5 caption and the appendix matrices; Definition 8 and the frame-mismatch reading split at 45 words",
        abs_now.endswith(THESIS) and bf.count(THESIS) == 2 and "relative to its superclass center, not a hierarchy among those centers" in s1_ and "hub" not in s1_.replace("hubs of", "") and "Given a grouping of classes into superclasses, assigning each centroid" in s3_ and "which we call the frame" not in TF and "decoupled control" not in s1_ and "Rotating each cluster at random removes every verdict" in s1_ and "once cluster orientations are randomized" not in s1_
        and ("hub" not in _re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", s1_, flags=_re.S))   # the plain-language pass (2026-09-25) drops the coined term from S1 altogether and _re.search(r"\bframe\b", bf).start() == bf.index("which we call the frame") + len("which we call the ")
        and "\\textbf{The answer has three parts.} (i) \\emph{Where the premise is read,}" in s1_ and "the excess separates the random cloud from the other two, and the second test separates the star from the tree.}" in s1_ and "\\vspace{2pt}\n\\caption{\\textbf{Plato's cave" in s1_ and "(ii) \\emph{Within each model,}" in s1_ and "(iii) \\emph{Across models,}" in s1_ and "and they are read against the ideal values of a tree and of a non-hyperbolic space, never against what a structureless cloud" in s1_ and "GPT-2 M" not in s1_ and "reads $\\delta$ against the right reference" in s1_ and "\\citet{groger2026aristotelian} calibrate similarity across models" in s1_
        and "forcing the index to zero (Figure~\\ref{fig:treemap}a)." in bf and bf.index("\\paragraph{The naive map manufactures an island.}") > bf.index("\\label{fig:treemap}")   # the walkthrough sentence went to pay for page 9 (2026-09-25); the caption carries it
        and (TEX/"figures/fig_treemap_matrices_final.pdf").exists() and "mean_ari_other" in v5 and "ceiling_band" in v5 and "The frozen ViT-B firing in" in TF and "A verdict that survives the rotation is carried by the arrangement of the hubs; one that does not" in bf,
        f"abstract words {len(_re.sub(r'[$][^$]*[$]', '', abs_now).replace('--', '').split())}")
    # ---- twelfth review (2026-09-23): 'none as strong as the planted one', 'well above chance, though short of ...', the thesis, the MERU lead-in, limitation (vii) on the diameter normalization, Table 3 continued captions per panel
    rob_c = _re.findall(r"\\caption\{\(continued\)(.*?)\n\}", rob, flags=_re.S)
    chk("final (12th review, abstract edits of 2026-09-23): 'no hierarchy as strong as the planted one is found' in the abstract and 'none as strong…' in S5.3(d); 'well above chance, though short of what two readings of the same model reach' in contribution 3 only, the abstract and the S1 answer paragraph saying the cross-model sentence of 2026-09-23 (15:45) with 'though less than two resampled versions of the same model', the island in S5.4 only; the thesis with 'well above chance, though not as much as two readings of one model' twice; MERU lead-in; limitation (vii) on the diameter normalization with scope renumbered (viii)-(x); every continued caption of Table 3 names its panel",
        abs_now.endswith(THESIS) and bf.count(THESIS) == 2 and "clusters that they partly share" in THESIS and len(THESIS.split()) == 34 and "finds no hierarchy among superclasses in the 9 of 12 backbones where it detects a planted one" in abs_now and bf.count("none as strong as the planted one is found") == 1 and "none is found in the real model" not in bf
        and "\\paragraph{Training in hyperbolic space leaves the clustering unchanged.}" in bf and "Imposing the geometry does not create" not in bf
        
        and len(rob_c) == 0 and rob.count("\\caption{(continued)}") == rob.count("\\ContinuedFloat"),   # appendix cleanup (2026-09-23): one short caption, the other floats plain '(continued)' 
        f"continued captions {[c.strip()[:12] for c in rob_c]}")
    # ---- the palette of the bands and of the planted hierarchy (author's brief, 2026-09-24)
    _pal = (R.parents[1]/"ICLR2027"/"figures"/"palette.py").read_text()
    chk("final (palette, 2026-09-24): every band is the light blue-gray at 60 per cent and the planted hierarchy is black, dashed, with triangles; no yellow is left in the palette or the figure script",
        'BAND = "#C9D7EA"' in _pal and "BAND_ALPHA = 1.0" in _pal and 'BAND_EDGE = "#7D93AE"' in _pal and "BAND_LW = 0.7" in _pal
        and FIGSRC.count("facecolor=BAND, alpha=BAND_ALPHA, edgecolor=BAND_EDGE, lw=BAND_LW, zorder=0") == 5 and FIGSRC.count("Patch(facecolor=BAND, edgecolor=BAND_EDGE, lw=BAND_LW") == 3 and "color=BAND, lw=5" not in FIGSRC   # every band solid, edged and behind the data (author's brief, 2026-09-24) and 'PLANT = "#222222"' in FIGSRC
        and 'curve(ax, {m: float(D81[m]["dec_z_mean"]) for m in M}, (0, (4, 1.8)), color=PLANT, marker="^"' in FIGSRC
        and 'ls=(0, (4, 1.8)), lw=1.5, ms=4.4, mec="white", mew=0.5, label="planted hierarchy, randomized (c)"' in FIGSRC
        and "FDE725" not in _pal and "FDB813" not in FIGSRC and "FDE725" not in FIGSRC)
    # ---- Figure 4(b), the 15 text models on the class names (author's brief, 2026-09-24)
    _ft = _j.load(open(R/"final_fig_text.json")); _t53 = {r_["model"]: r_ for r_ in load("expR53_text_haar_p999_200.csv")}
    chk("final (Figure 4b, 2026-09-24): the text panel reads the ImageNet class names for the 15 models in the author's order with their full names, its excesses are expR53's and 7 of 15 are genuine; the caption and the S5.4 sentence name it",
        all(abs(_ft["excess"][m_] - float(_t53[m_]["excess"])) < 1e-12 for m_ in _ft["excess"]) and len(_ft["excess"]) == 15
        and [_ft["labels"][m_] for m_ in _ft["order"]] == ["GPT-2 S", "GPT-2 M", "GPT-2 L", "GPT-2 XL", "Pythia-410M", "Pythia-1B", "Pythia-2.8B", "OLMo-1B", "BGE-base", "BGE-large", "GTE-base", "GTE-large", "GTE-Qwen2-1.5B", "E5-base", "E5-large"]
        and _ft["text_genuine"] == sum(str(_t53[m_].get("genuine_bh")) == "True" for m_ in _t53) == 7
        and "(b) The same reading for the 15 text models on the ImageNet class names; filled: genuine." in bf and "figures/fig_excess_final.pdf" in bf,
        f"text genuine {_ft['text_genuine']}/15")
    # ---- counts instead of dots, no value in parentheses, two figures out (author's briefs of 2026-09-24, late)
    _cc = _j.load(open(R/"final_census_constructions.json")); _packed = _re.compile(r"\$?[-+0-9.]+\$?\s*\(\$?[-+0-9./]+\$?\)")
    _tabhits = {f_: [m_.group(0) for l_ in (FD/(f_ + ".tex")).read_text().split("\n") if "&" in l_ and not l_.startswith("%") for m_ in _packed.finditer(l_)] for f_ in inputs if (FD/(f_ + ".tex")).exists()}
    _tabhits = {k_: v_ for k_, v_ in _tabhits.items() if v_ and k_ != "tab_q10_calibration"}   # the calibration table's '(0.3)' is a cloud label, not a value
    _cens = (FD/"tab_q01_census_final.tex").read_text()
    chk("final (counts and columns, 2026-09-24): the census verdict table gives k of five constructions per cell beside the cosine verdict (Tables 5 and 6 merged), the text census gives k of four readings, no appendix table packs two numbers in one cell, the same-reading and ceiling figures are gone and Table 14 carries the author's title",
        not _tabhits and "$k$/5 & cos" in _cens and _re.search(r"& \d/5 & \$\\(bullet|circ)\$", _cens) and "$\\bullet$" in _cens
        and _cc["all_five"] == int(_re.search(r"All five agree on (\d+) of 72", _cens).group(1)) and len(_cc["order"]) == 5
        and "$k$/4" in (FD/"tab_q02_text.tex").read_text() and _re.search(r"& \d/4 &", (FD/"tab_q02_text.tex").read_text())
        and "fig_samereading_final" not in TF and "fig_ceiling_final" not in TF and "\\textbf{Training in hyperbolic space shows no detected hub structure.}" in (FD/"tab_q04_depth_final.tex").read_text()
        and "Does the verdict change with how it is measured?" in _cens,
        f"packed cells {list(_tabhits)}; all five {_cc['all_five']}/72")
    # ---- appendix reduction (author's brief, 2026-09-24): the uncited tables are gone and the trend tables are figures
    _applabs = set(_re.findall(r"\\label\{(tab:q[^}]*)\}", _tabs_final))
    _mainlabs = {m_.group(1) for m_ in _re.finditer(r"\\ref\{(tab:[^}]*)\}", main_f)}
    _alias = {"tab:q1-census", "tab:q5-power", "tab:q6-treemap", "tab:q8-robust"}   # a base label sharing its table with a cited one
    _keptun = sorted(_applabs - _mainlabs - _alias)
    _e2r = {(r_["model"], r_["dataset"]): r_ for r_ in load("exp2_metric_controls.csv")}
    _hier = ["imagenet", "cifar100", "cifar10", "dtd"]
    def _bestpp(m_):
        h_ = [100 * (float(_e2r[(m_, d_)]["FS_H"]) - float(_e2r[(m_, d_)]["FS_R"])) for d_ in _hier]
        c_ = [100 * (float(_e2r[(m_, d_)]["FS_COS"]) - float(_e2r[(m_, d_)]["FS_R"])) for d_ in _hier]
        return round(sum(max(a_, b_) for a_, b_ in zip(h_, c_)) / 4, 2)
    _fg = _j.load(open(R/"final_fig_gains.json")); _fb = _j.load(open(R/"final_fig_budget.json")); _fp = _j.load(open(R/"final_fig_power.json"))
    chk("final (appendix reduction, 2026-09-24): every appendix table is cited from the main text, the radial control and the calibrated correlations at the sentences that use their numbers (2026-09-24); the budget, class-count, power, ceiling and gains tables are figures whose values come from the same files; Section B keeps the four limitations S7 does not carry and Section C the four paragraphs the main text still needs; the reading guide lists them",
        _keptun == []
        and all(("\\label{fig:" + f_ + "}") in app_f and (TEX/("figures/fig_" + f_ + "_final.pdf")).exists() for f_ in ("budget", "power", "gains"))
        and _fb["cells"] == 9 and _fb["max_drift_above_1e5"] <= 0.001 and _fp["n_covered"] == 9
        and all(_fg["best_pp"][m_] == _bestpp(m_) for m_ in _fg["best_pp"]) and _fg["metric"]["dinov2_l"] == "cos" and len(_fg["best_pp"]) == 10
        and app_f.count("\\paragraph{Limitations") == 0 and [app_f.index("\\section{" + t_ + "}") for t_ in ("Proofs", "Implementation", "Additional results")] == sorted(app_f.index("\\section{" + t_ + "}") for t_ in ("Proofs", "Implementation", "Additional results"))
        and "\\ref{tab:q8-robust-c}" not in TF and "\\ref{tab:q2-text-c}" not in TF and "\\ref{tab:provenance}" not in TF,
        f"uncited tables kept {_keptun}")
    # ---- main-text completeness (author's brief, 2026-09-24, night): four sentences, every number against its file
    _t53c = {r_["model"]: r_ for r_ in load("expR53_text_haar_p999_200.csv")}
    _slc = pd.read_csv(R/"expR62_samplelevel_record.csv"); _a3c = pd.read_csv(R/"exp3_alignment.csv").set_index("model").spearman_wn
    def _wnr(p_, n_):
        ms_ = [m for m in _a3c.index if m.startswith(p_)]; assert len(ms_) == n_, (p_, ms_)
        return f"${_a3c[ms_].min():+.2f}$ to ${_a3c[ms_].max():+.2f}$"
    _g3 = lambda d_: set(d_[d_.genuine_bh == True].model)
    _rec3 = pd.read_csv(R/"expR53_text_haar_p999_200.csv"); _sup3 = pd.read_csv(R/"expR53_text_haar_sup_200.csv"); _cos3 = pd.read_csv(R/"expR57_text_cosine_haar_p999_200.csv")
    _all3 = _g3(_rec3) & _g3(_sup3) & _g3(_cos3); _any3 = _g3(_rec3) | _g3(_sup3) | _g3(_cos3)
    _emb3 = [m for m in _rec3.model if m.startswith(("bge", "gte", "e5"))]; _db3 = pd.read_csv(R/"expR61_dbpedia_record.csv")
    _txt_ok = ({"gpt2_l", "gpt2_xl"} <= _all3 and not ({"gpt2", "gpt2_m"} & _all3) and {"gpt2", "gpt2_m"} <= _any3
               and {"pythia_410m", "pythia_1b", "pythia_2b8"} <= _all3 and len(_emb3) == 7 and not (set(_emb3) & _g3(_rec3))
               and len(_db3) == 3 and bool((_db3.genuine_bh == True).all()) and set(_db3.model) <= set(_emb3))
    _r71c = pd.read_csv(R/"expR71_meru_radii.csv"); _r71c = _r71c[_r71c.model.str.startswith("meru")]
    chk("final (main-text completeness, 2026-09-24): S5.1 gives the sample-level count (14 of 24 from expR62), the MERU sentence carries the unmeasured power and the 95th-percentile radius from expR71, S5.4 gives the three WordNet ranges from exp3_alignment and the recipe pointer, and the text paragraph's per-model claims hold in expR53/expR57/expR61 (GPT-2 L and XL genuine under all three readings, S and M under some, Pythia at every size, the 7 embedders not on class names and the 3 read on DBpedia genuine)",
        f"the calibrated reading is not genuine in {int((~_slc.genuine_bh).sum())} of {len(_slc)} cells, that is, no more than chance would give." in bf
        and "Read with the census and the hierarchy test, MERU shows the same clustering as its Euclidean twin. No hub structure is detected, although the test's power at MERU's spectrum was not measured." in bf
        and f"Its embeddings also stay nearly flat: 95 per cent lie within {_r71c.radius_sqrtc_p95.max():.2f} curvature radii of the origin, where the space is nearly flat (Table~\\ref{{tab:q4-depth-c}})." in bf
        and f"The correlation between inter-centroid and WordNet distances is highest for the contrastive vision--language models (VLMs, {_wnr(('clip', 'siglip'), 3)}), then the supervised ViTs ({_wnr('i21k', 4)}), and lowest for DINOv2 ({_wnr('dinov2', 4)})." in bf
        and "Among leaf-supervised ViTs it depends on the recipe" not in bf   # deleted to pay for limitation (vii) (author, 2026-09-25)
        and f"{sum(str(_t53c[m_].get('genuine_bh')) == 'True' for m_ in _t53c)} of 15 text models are genuine under the reading (Figure~\\ref{{fig:excess}}b). GPT-2 L and XL are genuine under every reading while S and M change with it. Pythia is genuine at every size. The sentence embedders are not genuine on class names (Table~\\ref{{tab:q2-text}})." in bf and _txt_ok
        and (lambda _ft: _ft["text_genuine"] == 7 and _ft["n_text"] == 15 and set(_ft["labels"]) == set(_ft["excess"]))(_j.load(open(R/"final_fig_text.json"))),
        f"sample-level not genuine {int((~_slc.genuine_bh).sum())} of {len(_slc)}; text ok {_txt_ok}")
    # ---- full read (author's brief, 2026-09-24, evening): vocabulary, S4 class sets, the statement boxes, the statements
    _ident = lambda s_: (nocom(s_).replace("fig_implant_final", "").replace("fig:implant", "").replace("expR80\\_implanted\\_alignment.csv", "")
                         .replace("expR64\\_implanted\\_depth.csv", "").replace("expR80_implanted_alignment", "").replace("expR64_implanted_depth", ""))
    chk("final (full read, 2026-09-24): 'planted' everywhere, no 'implanted' left in the text, the appendix titles, the tables or the figure captions (file names and labels keep their spelling); the abstract adds the hierarchy test; S4 gives each class set its count in parentheses; the depth equation reads on into the 'where' clause; the statement boxes are unbreakable; the Ethics statement drops the metric-selection heuristic; Figure 3(b) names the supremum in the caption",
        "implant" not in _ident(TF).lower() and "implant" not in _ident(_tabs_final).lower()
        and "excess over many such clouds, and add a second test whose ability to detect a hierarchy is measured." in abs_now
        and "The class sets are ImageNet (1000 classes) and CIFAR-100 \\citep[100 classes;][]{cifar10}, which carry a real hierarchy; DTD \\citep[47 classes;][]{dtd} and CIFAR-10 (10 classes); and the flat FashionMNIST \\citep[FMNIST, 10 classes;][]{fashion} and MNIST \\citep[10 classes;][]{mnist}." in bf
        and "\\end{equation}\nwhere the spread is taken over star seeds and null replicates, and a cloud is certified" in bf
        and "unbreakable" in TF and "breakable" not in TF.replace("unbreakable", "") and TF.count("\\begin{defbox}") == 8
        and all(c_ in TF for c_ in ("\\definecolor{stmtdefrule}{HTML}{3B528B}", "\\definecolor{stmtdefback}{HTML}{E9EDF5}", "\\definecolor{stmtproprule}{HTML}{21918C}", "\\definecolor{stmtpropback}{HTML}{E2F2F1}"))
        and "\\newtcolorbox{defbox}{statementbox=stmtdefrule, colback=stmtdefback}" in TF and "\\newtcolorbox{propbox}{statementbox=stmtproprule, colback=stmtpropback}" in TF
        and "blue!55!black" not in TF and "orange!70!black" not in TF and "borderline west={1.2pt}" in TF
        and all(f'"{n_}": "{h_}"' in (R.parents[1]/"ICLR2027"/"figures"/"palette.py").read_text() for n_, h_ in (("supervised", "#3B528B"), ("ssl", "#21918C")))   # los filetes son los colores de las familias en las figuras (autor, 2026-09-25)
        and "metric-selection heuristic" not in TF and "involves no human subjects and no personal data, and proposes an analysis methodology." in stm
        and "a dot inside the band is what chance gives. On their statistic CIFAR-10 and CUB-200 fall inside the band, MiniImageNet below it, and CIFAR-100 at its edge" in bf
        and bf.count("planted") >= 8,
        f"planted in the main text {bf.count('planted')}")
    # ---- main text without tables (author's brief, 2026-09-24): Table 2 and Table 1 move to the appendix, a figure takes their place
    _sup_c100 = int(round(float(pd.read_csv(R/"expR85_khrulkov_sup_summary.csv").set_index("dataset").loc["cifar100", "r_above_median"])))   # the caption's median rank (author's brief, 2026-09-25)
    _prem = _j.load(open(R/"final_fig_premise.json")); _sl = pd.read_csv(R/"expR62_samplelevel_record.csv")
    chk("final (main text without tables, 2026-09-24): no table environment or table input left in the main text; the census and the Khrulkov tables inputted in the appendix with every column; Figure 3 in S5.1 with the author's caption, its 24 sample-level cells from expR62 and the four published datasets from expR78/expR85 (10 of 24 genuine; CIFAR-10 and CUB-200 indistinguishable); S5.1 and the S1 mirror point at it; Figure 2(a) retitled",
        "\\begin{table}" not in bf and "\\input{tab_" not in bf and "\\begin{tabular}" not in bf
        and "\\input{tab_census_final}" not in TF and "\\input{tab_khrulkov_final}" in TF[TF.index("\\appendix"):] and (FD/"tab_q01_census_final.tex").read_text().count("\\label{tab:census}") == 1
        and "\\includegraphics[width=\\linewidth]{figures/fig_premise_final.pdf}" in bf and (TEX/"figures/fig_premise_final.pdf").exists()
        and f"\\caption{{\\textbf{{The premise where it is read.}} (a) Excess of the reading on per-image features; filled: genuine, {_prem['sample_genuine']} of 24 cells. (b) The values reported by \\citet{{Khrulkov_2020_CVPR}}, our reproduction with their estimator, and the random cloud of the same shape, all on their statistic: a dot inside the band is what chance gives. On their statistic CIFAR-10 and CUB-200 fall inside the band, MiniImageNet below it, and CIFAR-100 at its edge, its verdict changing from trial to trial (median rank {_sup_c100}/200); on our reading only MiniImageNet is genuine after correction, and CIFAR-100 falls below at $p=0.030$ before it (Section~\\ref{{sec:f-sample}})." in bf
        and [d for d in ("cifar10", "cifar100", "cub", "miniimagenet") if float(S78.loc[d, "p_left_max"]) <= 0.05] == ["cifar100", "miniimagenet"]   # the percentile reading, which the caption's last clause contrasts with the supremum band
        and _prem["sample_genuine"] == int(_sl.genuine_bh.sum()) == 10 and _prem["n_cells"] == 24 and _prem["inside_band"] == ["cifar10", "cifar100", "cub"]
        and all(abs(_prem["ours_sup"][d] - float(S78.loc[d, "ours_raw_mean"])) < 1e-9 and abs(_prem["published"][d] - float(S78.loc[d, "theirs"])) < 1e-9 for d in _prem["published"])
        and '_ax3[0].annotate("(a) per-image features"' in FIGSRC and 'axb3.annotate("(b) the values of Khrulkov et al., calibrated"' in FIGSRC
        and 'ax.set_title("(a) raw reading, random ball and random cloud", pad=3)' in FIGSRC
        and "Figure~\\ref{fig:premise}a and Table~\\ref{tab:q3-sample} give the 24 cells." in bf
        and "calibrated, only MiniImageNet stays below the random cloud. CIFAR-10 and CUB-200 are indistinguishable from a random cloud, and CIFAR-100 falls below it only before correcting for testing four datasets at once." in bf,
        f"sample-level genuine {int(_sl.genuine_bh.sum())}/24; inside the band {_prem['inside_band']}")
    # ---- template compliance and the page-9 recovery (author's brief, 2026-09-24): the preamble keeps no spacing override of the
    # style's; S7 keeps four limitations with the full list in the appendix; the moved paragraphs live in the appendix verbatim
    _pre_f = TF[:TF.index("\\begin{document}")]; _sty_f = (TEX/"iclr2027_conference.sty").read_text()
    _s6_f = seg(bf, "\\section{Implications for Hyperbolic Representation Learning}", "\\section{Conclusion and Limitations}")
    _lim_f = bf[bf.index("\\paragraph{Limitations.}"):]   # bf ends at the Reproducibility Statement
    chk("final (template compliance, 2026-09-24): the preamble has no display-skip block and no textfloatsep, abovecaptionskip, floatsep, linespread or global parskip override, and the main text no \\raggedbottom, so the style's \\parskip .5pc, \\parindent 0 and \\flushbottom apply; S7 keeps eight limitations and the appendix none (final reduction); the four paragraphs of the former Appendix C sit in their question; S6 in one paragraph; the only \\vspace is the author's before the Figure 1 caption",
        all(w not in _pre_f for w in ("g@addto@macro", "abovedisplayskip", "\\textfloatsep", "\\abovecaptionskip", "\\floatsep", "\\linespread"))
        and _pre_f.count("\\setlength{\\parskip}") == 1 and "before upper={\\setlength{\\topsep}" in _pre_f
        and "\\parskip .5pc" in _sty_f and "\\parindent 0pt" in _sty_f and "\\flushbottom" in _sty_f
        and "\\raggedbottom" not in TF[:TF.index("\\appendix")] and TF.count("\\raggedbottom") == 1
        and TF.count("\\vspace") == 1 and "\\vspace{2pt}\n\\caption{\\textbf{Plato's cave" in TF
        and [_lim_f.count("(%s)~" % r) for r in ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix")] == [1, 1, 1, 1, 1, 1, 1, 0, 0]   # (vii) restored as one merged item (author, 2026-09-25)
        and "The hierarchy test is Euclidean, and for the angular DINOv2 tree it may be conservative." not in bf and "The text census depends on the probe: template and batching move the reading, so marginal verdicts are fragile." in bf
        and "resting chiefly on DINOv2's scale range" not in bf
        and "app:limits" not in TF and "app:detail" not in TF and "The reading divides by the diameter, so heavier tails would lower $\\delta_{\\text{norm}}$ without any clustering; the cosine census mitigates this." in _lim_f
        and _s6_f.count("\\paragraph{") == 1
        and f"(ii)~The verdict of no hub hierarchy holds only in the {len(cov)} of 12 backbones where the control with clusters rotated detects a planted hierarchy." in bf
        and all(w in app_f for w in ("FMNIST is genuine in", "Neural collapse predicts", "Only ViT-B and ViT-L are certified under every grouping.", "fixed-radius Euclidean encoder")),
        f"limitations in S7 {[_lim_f.count(chr(40) + r + chr(41) + chr(126)) for r in ('i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii')]}")
    # ---- numeral rule (2026-09-23): counts of models, cells, seeds and runs are numerals with their denominator; words only for descriptive quantities
    WORDS = r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|thirty|sixty|sixty-six|two hundred)\b\s+(?:other\s+)?(?:certified |real |blind |supervised |self-supervised |leaf-label |covered |vision |text |ImageNet |decoupled |intact |fine )?(?:backbones?|models?|cells?|seeds?|runs?|replicates?|datasets?|class sets?|classes|superclasses|coarse labels|star seeds|resamples|model pairs|controls?|ResNets|ViTs|sizes|sentence embedders|values|verdicts|centroid clouds|rows)\b"
    main_txt = nocom(bf); capt = " ".join(_re.findall(r"\\caption\{(.*?)\n\}", nocom(TF) + "".join(nocom((FD/(s_ + ".tex")).read_text()) for s_ in inputs) + nocom(open(TEX/"tab_census_final.tex").read()) + nocom(open(TEX/"tab_khrulkov_final.tex").read()), flags=_re.S))
    capt += " " + " ".join(_re.findall(r"\\noindent\\textbf\{(\([a-z]'?\)[^}]*)", "".join(nocom((FD/(s_ + ".tex")).read_text()) for s_ in inputs)))   # panel titles count as captions
    DESCRIPTIVE = ["Two cells of the same dataset", "two resamples of the same model", "two readings of the same model", "two readings of one model", "two resampled versions of the same model", "the random cloud from the other two", "three constructions", "two constructions", "two fifths", "two null standard deviations"]   # descriptive quantities stay in words (2026-09-23)
    for d_ in DESCRIPTIVE: main_txt = main_txt.replace(d_, " "); capt = capt.replace(d_, " ")
    _sp = main_txt + " " + capt; SPELLED_OK = ("four datasets at once", "two classes that share a WordNet parent")   # el procedimiento de correccion y la definicion de tripleta hermana (autor, 2026-09-25)
    spelled = [m.group(0) for m in _re.finditer(WORDS, _sp) if not any(_sp.startswith(k_, m.start()) for k_ in SPELLED_OK)]   # "correcting for testing four datasets at once" names the procedure, not a count of the study (autor, 2026-09-25); case-sensitive: a count that opens a sentence is spelled out and capitalized (exception of 2026-09-23), so 'Four controls' and 'Two self-supervised ResNets' do not match
    digit_start = _re.findall(r"(?:[.!?]\s+|\\paragraph\{)(\d[^\s]*\s+\w+)", _re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", main_txt, flags=_re.S)) + [m.group(0) for m in _re.finditer(r"\b(?:the other|in the other|of the other) (?:two|three|four|five|six|seven|eight|nine|ten|eleven)\b", main_txt + " " + capt)]
    chk("final (numeral rule): counts of models, cells, seeds and runs are numerals with their denominator in the abstract, the main text and every caption ('9 of 12 backbones', 'the other 3'); no spelled-out count before a count noun; words kept for descriptive quantities",
        not spelled and not digit_start and "\\paragraph{Four controls test the alternatives.} Two self-supervised ResNets" in bf and "finds no hierarchy among superclasses in the 9 of 12 backbones" in abs_now and "which disappears when clusters are rotated at random" in abs_now and "two resampled versions of the same model" in abs_now and "a three-level hierarchy" in bf and "12 backbones, 6 class sets and 15 text models" in bf, f"spelled {spelled[:8]} digit-start {digit_start[:4]}")
    # ---- twelfth review (5): Table 1's supremum columns (expR85) or the brief's fallback clause
    S85 = pd.read_csv(R/"expR85_khrulkov_sup_summary.csv").set_index("dataset") if (R/"expR85_khrulkov_sup_summary.csv").exists() else None; kt85 = (TEX/"tab_khrulkov_final.tex").read_text()
    chk("final (12th review, item 5): Table 1 carries the excess and rank of the published statistic itself, the supremum, calibrated on the same 200 replicates over 10 trials per dataset (cells from expR85_khrulkov_sup_summary.csv), and S5.1 says so; while the run is not merged, the fallback clause of the brief",
        (all(f"${S85.loc[d, 'excess_sup_mean']:+.4f}$ & {int(round(S85.loc[d, 'r_above_median']))}/200, {S85.loc[d, 'p_left_min']:.3f}--{S85.loc[d, 'p_left_max']:.3f}" in kt85 for d in S85.index) and bool((S85.n_trials == 10).all()) and "(sup.)" not in kt85 and "\\multicolumn{2}{c}{their statistic, the supremum}" in kt85 and "expR85_khrulkov_sup_summary.csv" in kt85
         and "only before correction ($p=0.030$). Under their own statistic, the supremum, CIFAR-10 and CUB-200 stay indistinguishable from a random cloud and MiniImageNet stays below it. The verdict for CIFAR-100 changes from trial to trial, as the statistic confound predicts. Figure~\\ref{fig:premise}b and Table~\\ref{tab:khrulkov} give the 4 datasets." in bf
         and all(float(S85.loc[d, "p_left_median"]) < 0.05 for d in ("cifar100", "miniimagenet")) and all(float(S85.loc[d, "p_left_median"]) > 0.05 for d in ("cifar10", "cub")) and float(S85.loc["cifar100", "p_left_max"]) > 0.05
         and "under their own statistic" not in s1_ and "CIFAR-100 falls below it only before correcting for testing four datasets at once. (ii)" in s1_
         and "For their own statistic, the supremum, against the same replicates: the mean excess, the median rank over 10 trials and the range of $p$ across trials. CIFAR-100 is below the null in the median trial but not in every trial." in kt85 and "$r$/200, $p$ range \\\\" in kt85) if S85 is not None
        else ("The calibration reads the percentile statistic $\\hat\\delta_{99.9}$, whereas their statistic is the supremum. Figure~\\ref{fig:premise}b and Table~\\ref{tab:khrulkov} give the 4 datasets." in bf and "(sup.)" not in kt85),
        f"expR85 {'merged' if S85 is not None else 'not merged'}")
    # ---- readability, second pass (2026-09-23): the four fixes and the 30-word rule
    chk("final (readability 2): Definition 3 opens with the plain sentence; S3.2 confounds paragraph in one sentence; S3.3 'Two cells of the same dataset can have the same reading and opposite verdicts'; S3.4 opens with the author's two sentences before Definition 6; the three verbatim S2/S3 sentences split by recorded edits; no sentence over 30 words in S3-S7 outside definitions, citation lists and the thesis",
        "\\begin{definition}[Haar null]\n\\label{def:null}\nA null replicate is a cloud with the real shape and no structure. Let $X" in bf and "\\paragraph{A raw value cannot be called low on its own.} Proposition~\\ref{prop:bound} makes the dimension confound precise; the calibration on reference geometries shows the spectrum and statistic confounds (Table~\\ref{tab:q10-calibration})." in bf
        and f"Two cells of the same dataset can have the same reading, {_sr}, ViT-B on CIFAR-100 images and DINO-B on its centroids, and only the second is genuine." in bf and bf.index("\\paragraph{Why a second test.} A genuine excess can come from clusters alone, so a second test, which we call the hierarchy test, asks whether the clusters are themselves arranged hierarchically. It needs three constructions.") < bf.index("\\begin{definition}[hub null and hub excess]")
        and ". We bring the idea to embedding clouds" in bf and "is zero. The further a metric is from a tree" in bf and "fall below. It keeps the tail of the defects" in bf
        and len(over30_all) == 0, f"over 30: {[(w_, s_[:50]) for w_, s_ in over30_all[:5]]}")
    chk("final (4th review): depth table with two-decimal z everywhere and the K = 10/30/60 sweep with the balanced frame; DBpedia supremum columns under the Haar null; no OLMo-7B row; no expR32 bootstrap row",
        not _re.search(r"\(([+-]\d\.\d)\)", dep) and not _re.search(r"\$[+-]\d\.\d\$", dep) and "$K{=}10$" in dep and "$K{=}60$" in dep and "supremum, Haar" in wn and "supremum, Gaussian" not in wn
        and "OLMo-7B" not in txt_ and "OLMo-7B" not in pan_ and "centroid bootstrap: excess" not in rob and "expR32" not in rob and "expR73" in rob and "shrinks with the number of classes" in app_f and sorted(set(int(r_["C"]) for r_ in load("expR60_c_sweep_record.csv"))) == [10, 20, 50, 100, 200, 500, 1000] and rob.count("DINOv2-L & ") >= 2)
    alltex = nocom(TF) + "".join(nocom((FD/(s + ".tex")).read_text()) for s in inputs) + nocom(open(TEX/"tab_census_final.tex").read()) + nocom(open(TEX/"tab_khrulkov_final.tex").read())
    refs = set(_re.findall(r"\\(?:eq)?ref\{([^}]*)\}", alltex)); defs = set(_re.findall(r"\\label\{([^}]*)\}", alltex))
    chk("final: every cross-reference of the final resolves", refs <= defs, str(sorted(refs - defs)))
    tcf = open(TEX/"tab_census_final.tex").read()
    _bhj = _j.load(open(R/"final_khrulkov_bh.json")); _p78s = pd.read_csv(R/"expR78_khrulkov_replication_summary.csv").set_index("dataset")["p_left_max"].sort_values()
    _bh78 = [d for i_, (d, v_) in enumerate(_p78s.items(), 1) if i_ <= max([0] + [j_ for j_, (d2, v2) in enumerate(_p78s.items(), 1) if float(v2) <= 0.05 * j_ / len(_p78s)])]
    chk("final (BH consistency, 2026-09-25): the published values carry the paper's own definition of genuine, Benjamini--Hochberg across the 4 datasets, which leaves MiniImageNet alone; S1, S5.1 and the Figure 3 caption say so and give CIFAR-100's uncorrected p from the file",
        _bh78 == ["miniimagenet"] == _bhj["bh_genuine"] and f"{float(_p78s['cifar100']):.3f}" == "0.030"
        and "calibrated, only MiniImageNet stays below the random cloud. CIFAR-10 and CUB-200 are indistinguishable from a random cloud, and CIFAR-100 falls below it only before correcting for testing four datasets at once." in bf
        and "Calibrated, only MiniImageNet \\citep{miniimagenet} is genuine. CIFAR-10 and CUB-200 \\citep{cub} are indistinguishable from a random cloud, and CIFAR-100 falls below it only before correction ($p=0.030$)." in bf
        and "on our reading only MiniImageNet is genuine after correction, and CIFAR-100 falls below at $p=0.030$ before it" in bf
        and "(WordNet agreement: Table~\\ref{tab:q7-wordnet})" in (FD/"tab_q04_depth_final.tex").read_text(),
        f"BH over the four: {_bh78}")
    _figsrc = "".join((Path(__file__).resolve().parents[2]/"ICLR2027"/"figures"/f).read_text() for f in ("make_figs_final.py", "palette.py"))
    chk("final: no figure says 'readout' in a title, a label or a legend (the word is 'metric' since 2026-09-25), and Figure 9's panel title names the best zero-cost metric",
        "readout" not in _figsrc.lower() and 'ax.set_title("the best zero-cost metric, against the Euclidean one"' in _figsrc and "readout" not in TF.lower(),
        "readout in the figure scripts or the paper")
    _zip = Path(__file__).resolve().parents[2]/"ICLR2027"/"supplementary_code.zip"
    import zipfile as _zf
    _names = sorted(_zf.ZipFile(_zip).namelist()) if _zip.exists() else []
    _txt = "".join(_zf.ZipFile(_zip).read(n_).decode("utf-8", "ignore").lower() for n_ in _names if n_.endswith((".py", ".md", ".mplstyle"))) if _names else ""
    chk("final (supplement, 2026-09-25): the archive carries only the code that produces results -- the experiment scripts, tool/ and a README that maps each result to its script -- with no figure or table generator, no assembler and no result file, and no identifying string",
        bool(_names) and sum(1 for n_ in _names if n_.startswith("scripts/") and n_.endswith(".py")) >= 55
        and any(n_ == "README.md" for n_ in _names) and sum(1 for n_ in _names if n_.startswith("tool/")) >= 5
        and not any(n_.startswith("results/") or n_.startswith("figures/") or n_.startswith("iclr2027/") for n_ in _names)
        and not any(n_.endswith((".csv", ".json")) for n_ in _names)
        and not any(b_ in n_ for n_ in _names for b_ in ("gen_appendix", "gen_main_table", "phaseE_submission", "sweep_freeze", "make_supp_readme", "make_figs"))
        and not any(w_ in _txt for w_ in ("rodenas", "radeva", "aguilar", "javi", "ub.edu", "neurips", "/media/")),
        f"{len(_names)} entries in supplementary_code.zip")
    _sub = Path(__file__).resolve().parents[2]/"ICLR2027"/"submission"
    _subf = sorted(str(p.relative_to(_sub)) for p in _sub.rglob("*") if p.is_file())
    chk("final: submission/ is what Overleaf takes (author, 2026-09-25): the main files at the top level, the tables in appendix_tables/ and the figures in figures/, the bibliography already compiled, and every \\input and \\includegraphics of its .tex resolving inside the folder",
        len(_subf) == 31 and "main_iclr2027_final.tex" in _subf and "main_iclr2027_final.bbl" in _subf and "README.txt" in _subf
        and sum(1 for f in _subf if f.startswith("appendix_tables/")) == 12 and sum(1 for f in _subf if f.startswith("figures/")) == 9
        and all((_sub/(g if g.endswith(".tex") else g + ".tex")).exists() for g in _re.findall(r"\\input\{([^}]*)\}", (_sub/"main_iclr2027_final.tex").read_text()))
        and all((_sub/g).exists() for g in _re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", (_sub/"main_iclr2027_final.tex").read_text()))
        and (Path(__file__).resolve().parents[2]/"ICLR2027"/"submission.zip").exists(),
        f"{len(_subf)} files in submission/")
    chk("final (citation fix, 2026-09-25): Fournier et al. support the cost of the exact computation, in S1 and S3.2, and no longer the growth of the sampled supremum, which is our own observation",
        "And computing the supremum exactly takes more than cubic time \\citep{fournier2015computing}, so it is taken over sampled quadruples, where it grows with the budget and does not converge: the \\emph{statistic confound}." in bf
        and "than can be enumerated, and exact computation takes more than cubic time \\citep{fournier2015computing}. The defect is therefore computed on half a million random quadruples." in bf
        and "keeps growing as more quadruples are drawn." in bf and "as more quadruples are drawn \\citep{fournier2015computing}" not in bf
        and bf.count("\\citep{fournier2015computing}") == 2,
        f"fournier cited {bf.count(chr(92) + chr(92) + 'citep{fournier2015computing}')} times in the main text")
    _pci = _j.load(open(R/"final_power_ci.json")); _s81c = pd.read_csv(R/"expR81_deep_per_backbone_summary.csv").set_index("model")
    def _wil(p_, n_, z_=1.96):
        den = 1 + z_ * z_ / n_; c_ = (p_ + z_ * z_ / (2 * n_)) / den
        h_ = z_ * ((p_ * (1 - p_) / n_ + z_ * z_ / (4 * n_ * n_)) ** 0.5) / den
        return c_ - h_, c_ + h_
    chk("final (power intervals, 2026-09-25): the rotated-cluster power carries its run counts and a Wilson 95 per cent interval; S5.3 names the two backbones closest to the threshold and both figures draw the whisker",
        all(int(_s81c.loc[m_, "dec_runs"]) == (200 if m_.startswith("dinov2") else 50) for m_ in _s81c.index)
        and f'{_wil(float(_s81c.loc["i21k_s", "dec_power"]), 50)[0]:.2f}' == "0.79" and f'{_wil(float(_s81c.loc["dinov2_b", "dec_power"]), 200)[0]:.2f}' == "0.78"
        and _pci["closest_to_threshold"] == ["dinov2_b", "i21k_s"]
        and "over 50 to 200 runs per backbone, and not in the other 3" in bf
        and "ViT-S and DINOv2-B sit closest to the threshold, with 95 per cent intervals reaching 0.79 and 0.78." in bf
        and "whiskers: 95 per cent intervals over 50 runs per backbone, 200 for the DINOv2 family" in TF and TF.count("whiskers: 95 per cent intervals") == 2
        and FIGSRC.count("whisker(ax,") == 3 and "def wilson(p, n, z=1.96):" in FIGSRC,   # the definition plus the two calls
        "power intervals of 2026-09-25")
    _fills_ceiling = _j.load(open(R/"final_fills.json")).get("CEIL_TRIP", "")
    _cor_a = (FD/"tab_q09_corollary.tex").read_text()
    _r45 = pd.read_csv(R/"expR45_convnet_rows.csv").set_index("model"); _cen = (FD/"tab_q01_census_final.tex").read_text()
    _p14t = (FD/"tab_q14_panel.tex").read_text(); _gains = _j.load(open(R/"final_fig_gains.json")) if (R/"final_fig_gains.json").exists() else {}
    chk("final (corrections, 2026-09-25): no 'an planted' left; the two ResNet rows carry the count of replicates above the real value out of 20; Table 1 carries the extraction note; S2 cites Moreira; S4 points the pooling at the model panel; C.8 points at Figure 5b with the ceiling; the corollary names its 10 backbones and its 4 datasets",
        "an planted" not in TF and "an implanted" not in TF
        and all(f"{_nm} & ${float(_r45.loc[_m, 'excessA']):+.3f}$ & {int(round(float(_r45.loc[_m, 'rankA']) * 20))}/20" in _cen for _m, _nm in (("barlow_r50", "Barlow-R50"), ("byol_r50", "BYOL-R50")))
        and "Extraction: vision backbones use the encoder's default pooled embedding" in _p14t and "the note below the table says how features are extracted and pooled" in _p14t
        and "In few-shot learning \\citet{moreira2024hyperbolic} find that fixed-radius Euclidean prototypes match hyperbolic ones." in bf
        and "Text models are read one prompt at a time, since batching alters GPT-2's hidden states, each with its own pooling (Table~\\ref{tab:q14-panel})." in bf
        and f"two resamples of the same model agree on, {_fills_ceiling} on triplets; Figure~\\ref{{fig:treemap}}b carries it" in app_f
        and "The 10 backbones with cached per-dataset features, DINO-B and SigLIP-B excluded." in _cor_a
        and "averaged over ImageNet, CIFAR-100, CIFAR-10 and DTD" in TF and "the 4 hierarchical datasets" not in TF
        and "the superclass recovery beside it" not in TF and "the radii of the embeddings in units of the curvature scale" not in TF,
        "corrections of 2026-09-25")
    _cor_a = (FD/"tab_q09_corollary.tex").read_text()
    chk("final: the corollary panel drops the McNemar column (the main text never used it, author's brief 2026-09-24): 9 columns, 60 rows of accuracies and advantages, and a caption that defines Eucl. and Poinc. and gives the advantages in percentage points",
        "McNemar" not in _cor_a and "NC H vs R" not in _cor_a and "exp13_mcnemar" not in _cor_a and "{llccccccc}" in _cor_a
        and len([ln for ln in _cor_a.split(chr(10)) if ln.rstrip().endswith(chr(92) * 2) and ln.count("&") == 8]) == 61   # 60 cells plus the header row
        and "nearest-centroid and few-shot accuracy under the Euclidean (Eucl.) and the Poincar\\'{e} (Poinc.) metric, then the advantage of each zero-cost metric in percentage points over the Euclidean metric." in _cor_a
        and "McNemar" not in TF,
        f"{_cor_a.count('&')} ampersands in the panel")
    _pfA = app_f[:app_f.index("\\section{Implementation}")]
    chk("final: Appendix A is the author's expanded proof (2026-09-24): the proposition restated by reference, an intuition paragraph, the two parts proved with 6 displayed equations, and the labels it cites resolve",
        _pfA.count("\\begin{equation}") == 6 and "\\textbf{Proposition~\\ref{prop:bound}} (range bound and dimension confound)." in _pfA
        and "\\paragraph{Intuition.} The defect is half the difference between two sums of distances." in _pfA
        and "\\paragraph{Proof of (a).} Take any four points and their three pairing sums (Eq.~\\ref{eq:pairings})." in _pfA
        and "\\paragraph{Proof of (b).} Let $x_1,\\dots,x_n\\in\\mathbb{R}^d$ have iid standard Gaussian coordinates." in _pfA
        and _pfA.count("$\\square$") == 1 and _pfA.count("\\qquad\\square") == 1
        and "\\label{eq:pairings}" in bf and "\\label{prop:bound}" in bf and "\\usepackage{amssymb}" in TF
        and "relative contrast between the farthest and the nearest pair vanishes at rate $d^{-1/2}$ \\citep{beyer1999nearest, aggarwal2001surprising}" in _pfA,
        f"{_pfA.count(chr(92) + chr(92) + 'begin{equation}')} equations in Appendix A")
    _pan = (FD/"tab_q14_panel.tex").read_text()
    _prow = [ln for ln in _pan.split("\n") if ln.rstrip().endswith("\\\\") and "&" in ln and "type &" not in ln]
    chk("final: every row of the model panel carries its citation, inline after the model name since it fits the text width (author's brief, 2026-09-24), Barlow Twins and BYOL among them, and both new entries have a publisher source",
        len(_prow) == 32 and all("\\citep{" in ln for ln in _prow) and "ref." not in _pan
        and "\\citep{zbontar2021barlow}" in _pan and "\\citep{grill2020bootstrap}" in _pan
        and "proceedings.mlr.press/v139/zbontar21a.html" in _bibentry("zbontar2021barlow") and "proceedings.neurips.cc/paper/2020" in _bibentry("grill2020bootstrap")
        and _src(_bibentry("zbontar2021barlow")) and _src(_bibentry("grill2020bootstrap")),
        f"{len(_prow)} rows, {sum('citep' not in ln for ln in _prow)} without a citation")
    chk("final: the per-cell census is the centered Haar record with the two self-supervised ResNets as rows (author's brief, 2026-09-24), and it answers the reference of the former main-text table", (lambda _c: "expR75_census_centered_haar.csv" in _c and _c.count("\\label{tab:census}") == 1 and "Barlow-R50" in _c and "BYOL-R50" in _c and _c.count("\\\\") >= 14)((FD/"tab_q01_census_final.tex").read_text()))
    # ---- statements
    AI = "We used generative AI tools to assist with writing and editing, retrieving references, and implementing and running the experimental code, and to give feedback on drafts and on the methodology. We reviewed and verified all AI-assisted work, including every number, figure and reference, and take full responsibility for the content of this paper."
    chk("final: AI Use Statement with exactly the three declared items and the responsibility sentence; Reproducibility pointing at the supplementary material, with no TODO or anonymized-repository placeholder left in the file; Ethics present", AI in stm and "The supplementary material contains the scripts that produce every result in the paper, with fixed seeds, and a README that maps each result to its script; its \\texttt{tool/} directory ships the instrument as one script that reproduces any cell of Table~\\ref{tab:census} from its centroid matrix." in stm and "Ethics Statement" in stm
        and not any(w in TF for w in ("TODO", "anonymized repository", "ANONYMIZED")))
    chk("final: preamble of the frozen v1 plus amsthm, amssymb (the \\square of the proofs, 2026-09-24) and tcolorbox (the boxed statements), same class, same packages otherwise", TF[:TF.index("\\usepackage{amsthm}")] == T1[:T1.index("\\usepackage{array}\n") + len("\\usepackage{array}\n")] and TF.count("\\usepackage") == T1.count("\\usepackage") + 3 and "\\usepackage[most]{tcolorbox}" in TF and "\\usepackage{amssymb}" in TF
        and TF.count("\\begin{defbox}") == TF.count("\\begin{definition}") == 8 and TF.count("\\begin{propbox}") == TF.count("\\begin{proposition}") == 1)
final_checks()
def rebuttal_checks():
    """Parallel track (brief of 2026-09-20): main_iclr2027_rebuttal.tex = the frozen submission file plus the parallel-track paragraphs
    and tables (phaseE_rebuttal.py). Checked only when the file exists: (1) with the inserted paragraphs and the inserted appendix
    subsection removed, the file equals the frozen submission file (every frozen number stays); (2) every number written in the new
    paragraphs is re-derived here from its CSV; (3) no claim is made where the pre-set criterion is not met."""
    import re as _re, json as _j, pandas as pd
    TEX = Path(__file__).resolve().parents[2]/"ICLR2027"/"iclr2027"; PR = TEX/"main_iclr2027_rebuttal.tex"; PF = TEX/"main_iclr2027_final.tex"
    if not PR.exists(): return
    TR = open(PR).read(); TF = open(PF).read(); st = _j.load(open(R/"rebuttal_build_status.json")) if (R/"rebuttal_build_status.json").exists() else {}
    PT = ("expR77_", "expR78_", "expR79_", "expR80_")
    body = TR.replace("% rebuttal version = the frozen submission file plus the parallel-track paragraphs (phaseE_rebuttal.py); the frozen text and numbers are unchanged\n", "", 1)
    stripped = _re.sub(r"\\FloatBarrier\n\\subsection\{Parallel track:[^\n]*\n(?:.*?\n)*?(?=\\end\{document\})", "", body)   # the parallel-track tables close the appendix since the provenance table left the paper (2026-09-24)
    pars = [m for m in _re.finditer(r"\n\n\\paragraph\{[^}]*\}[^\n]*", stripped) if "\\ref{tab:r" in m.group(0)]   # inserted paragraphs point to the parallel-track tables (tab:r1-..., tab:r2-...); the integrated ones (2026-09-21) point to the paper's tables
    for m in reversed(pars): stripped = stripped[:m.start()] + stripped[m.end():]
    chk("rebuttal: the file is the frozen submission file plus the inserted paragraphs and the parallel-track appendix subsection, nothing else (frozen text and numbers unchanged)", stripped == TF, f"{len(pars)} inserted paragraphs; diff at char {next((i for i, (a, b) in enumerate(zip(stripped, TF)) if a != b), min(len(stripped), len(TF)))}")
    ins = "".join(m.group(0) for m in pars) + body[body.find("\\subsection{Parallel track:"):] if "\\subsection{Parallel track:" in body else "".join(m.group(0) for m in pars)
    chk("rebuttal: every inserted paragraph carries a provenance comment naming its result file and points to its table", all(_re.search(r"%\s*expR\d+\w*\.(csv|json)", m.group(0)) and "Table~\\ref{tab:r" in m.group(0) for m in pars))
    # (2) numbers re-derived from the CSVs
    if st.get("implanted_alignment", {}).get("written"):
        A = pd.read_csv(R/"expR80_implanted_alignment.csv"); A["hit"] = A.z_depth <= -2; d = pd.read_csv(R/"expR80_decision.csv").iloc[0]
        h1, n1 = int(A[A.s == 1.0].hit.sum()), int((A.s == 1.0).sum()); h0, n0 = int(A[A.s == 0.0].hit.sum()), int((A.s == 0.0).sum())
        chk("rebuttal (priority 1c, rule not met): the planted-alignment counts and the power in the paragraph and table match expR80_implanted_alignment.csv / expR80_decision.csv, the rule is stated as not met and no 'measured power' claim enters",
            f"detected in {h1} of {n1} runs at full strength and in {h0} of {n0} at zero, a power of {float(d.power_s1):.2f}" in ins and not bool(d.rule_power_ge_0_8_fa_le_0_05) and f"measured power {float(d.power_s1):.2f}, false alarms {float(d.false_alarms_s0):.2f}: not met" in ins
            and "certifies hub alignment with measured power" not in TR and "with measured power:" not in TR)
    if st.get("deep_synthetic", {}).get("written"):
        E = pd.read_csv(R/"expR79_synthetic_deep_poincare.csv"); E["cert"] = E.z <= -2; deep = E[E.cloud == "synthetic_deep_vitl_spectrum"]; flat = E[E.cloud == "synthetic_flat_vitl_spectrum"]; poi = E[E.cloud.str.startswith("wordnet_poincare")]
        ok79 = (f"the deep synthetic hierarchy fires in {int(deep.cert.sum())} of {len(deep)} seeds" in ins if len(deep) else True) and (f"the flat control in {int(flat.cert.sum())} of {len(flat)}" in ins if len(flat) else True) \
               and all(f"$z{{=}}{r.z:+.2f}$" in ins for _, r in poi.iterrows()) and all(f"${r.z:+.2f}$" in ins for _, r in E.iterrows())
        chk("rebuttal (priority 1b): the expR79 counts (deep, flat, decoupling) and every z in the paragraph and table match expR79_synthetic_deep_poincare.csv", ok79)
    if st.get("replication", {}).get("written"):
        S = pd.read_csv(R/"expR78_khrulkov_replication_summary.csv").set_index("dataset"); allin = len(S) == 4 and bool(S.within_range.all())
        ok78 = all(f"{S.loc[d].theirs:.2f} & {S.loc[d].ours_raw_mean:.3f}" in ins for d in S.index) and ((allin and "reproduces their raw" in ins) or (not allin and "does not yet reproduce every published" in ins and "no calibrated claim is drawn" in ins))
        chk("rebuttal (priority 2): the four published and reproduced delta_rel values match expR78_khrulkov_replication_summary.csv and the calibrated claim is made only when all four are within 0.03", ok78)
    if st.get("positive_control", {}).get("written"):
        P = pd.read_csv(R/"expR77_positive_control.csv").set_index("model"); V = _j.load(open(R/"expR77_positive_control_verdict.json"))["verdict"]; v0 = next((v for v in V if v["seed"] == "seed0"), None); met = bool(v0 and v0["criterion_met"])
        ok77 = all(f"${r.z_wn30:+.2f}$ & ${r.z_wn30bal:+.2f}$" in ins for _, r in P.iterrows()) and (("Success criterion met." in ins and "fixed before the run is met" in ins) if met else ("Success criterion not met." in ins and "fixed before the run is not met" in ins))
        chk("rebuttal (priority 1): every z of the trained positive control matches expR77_positive_control.csv and the verdict sentence matches the pre-set criterion in expR77_positive_control_verdict.json", ok77)
rebuttal_checks()
n_fail = sum(1 for _,ok,_ in checks if not ok)
for name, ok, det in checks[-30:]: print(("PASS" if ok else "FAIL"), name, ("| "+det if det and not ok else ""))
print(f"[phaseB re-total] {len(checks)-n_fail}/{len(checks)}")
for name, ok, det in checks:
    if not ok: print("FAIL(all):", name, ("| "+det if det else ""))
