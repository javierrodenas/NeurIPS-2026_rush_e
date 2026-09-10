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
    b17 = open(TEX/"appendix_tables"/"tab_b17_samplelevel.tex").read(); b30 = open(TEX/"appendix_tables"/"tab_b30_meru.tex").read()
    M = _j.load(open(R/"phaseC_memo.json"))
    chk("text: sample-level count in the prose; names, band and max excess in the B17 caption", f"not genuine in {M['sl_within']} of 24 cells" in main
        and f"not genuine in {M['sl_within']} of 24 cells and genuine in {M['sl_genuine']}" in b17 and "DINOv2-S, DINOv2-B, DINOv2-L, DINOv2-G, SigLIP-B on CIFAR-100; ViT-S, ViT-B, ViT-L, DINOv2-L, DINOv2-G on DTD" in b17
        and f"{M['sl_sup_lo']:.3f}--{M['sl_sup_hi']:.3f}" in b17 and f"${M['sl_exc_lo']:+.3f}$" in b17)
    chk("text: MERU ranges (ImageNet-image excess, native gap, depth z) live in the B30 caption and match the memo",
        f"from ${M['meru_in_exc_range'][1]:+.3f}$ to ${M['meru_in_exc_range'][0]:+.3f}$ for MERU" in b30 and f"by at most ${M['meru_in_nat_vs_euc_maxgap']:.4f}$" in b30
        and f"from ${M['meru_in_z_range'][0]:+.1f}$ to ${M['meru_in_z_range'][1]:+.1f}$ for MERU" in b30 and f"from ${M['clip_in_z_range'][0]:+.1f}$ to ${M['clip_in_z_range'][1]:+.1f}$ for CLIP" in b30)
    b = load("exp2b_normalized_stack.csv"); H = [r for r in b if r["dataset"] in ("imagenet","cifar100","cifar10","dtd")]
    gc = [100*float(r["FS_HN_COS_diff"]) for r in H if r["paradigm"].lower().startswith("contr")]; go = [100*float(r["FS_HN_COS_diff"]) for r in H if not r["paradigm"].lower().startswith("contr")]
    chk("text: Poincare-over-cosine gains (exp2b, few-shot, hierarchical sets) as stated", f"adds from ${min(gc):+.1f}$ to ${max(gc):+.1f}$ pp" in main and f"from ${min(go):+.1f}$ to ${max(go):+.1f}$ pp" in main)
    e1 = load("exp1_delta_controls.csv"); ga = sorted({int(r["d"]): float(r["delta_max"]) for r in e1 if r["variant"]=="gauss"}.items())
    c_lo, c_hi = (0.144/(2*ga[0][1]))**2, (0.144/(2*ga[-1][1]))**2
    chk("text: Khrulkov curvature on the Gaussian band (c=(0.144/delta_rel)^2, delta_rel=2 delta_norm; exp1 gauss d=192/1536)", f"${c_lo:.2f}$ at $d{{=}}{ga[0][0]}$ to ${c_hi:.1f}$ at $d{{=}}{ga[-1][0]}$" in main)
    im = [float(r["excess"]) for r in load("expR52_census_haar_p999_200.csv") if r["dataset"]=="imagenet"]
    chk("text: ImageNet class-level excess range as stated", f"On ImageNet centroids it runs from ${max(im):+.3f}$ to ${min(im):+.3f}$" in main)
    D = _j.load(open(R/"phaseC_fig2b.json")); chk("fig2b: decision recorded and consistent with the caption", (D["mode"]=="sample") == ("images) and a class-level cell" in main) and D["gap_sample"] <= D["gap_bge"] if D["mode"]=="sample" else True)
    for w in ("tool", "we believe", "nterestingly"): chk(f"text: no '{w}' in the main text", w not in main)
phaseC_text_checks()

# ---------- Prose pass (style of Groger et al.): metrics per paragraph, fixed vocabulary, thesis x5, numbers preserved ----------
def prose_checks():
    import re as _re
    TEX = Path(__file__).resolve().parents[2]/"ICLR2027"/"iclr2027"
    T = open(TEX/"main_iclr2027.tex").read(); body = T[T.index("\\begin{abstract}"):T.index("\\subsubsection*{Ethics Statement}")]
    THESIS = "Read correctly, foundation models organize classes into clustered structure that is occasionally hierarchical and moderately shared; they do not converge to one common tree, and their raw tree-likeness is not evidence for hyperbolic geometry."
    SHORT = "Clustered, occasionally hierarchical, moderately shared: not one common tree, and no license for curvature."
    chk("prose: the thesis appears verbatim three times (abstract, box, S7) and in its short form twice (end of S1, end of S5)", body.count(THESIS) == 3 and body.count(SHORT) == 2)
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
    chk("R9b prose: 'detected in x of 12', the within/between range and 'none of the sixty' match expR64b (0 false alarms in 60)",
        (f"detected in {ndet} of 12 backbones" if ndet else "detected in none of the twelve backbones") in main and f"is {lo:.1f} to {hi:.1f} times their between-hub spread" in main and fa0==0 and n0==60 and "none of the sixty zero-strength runs" in main)
    D9 = load("expR64b_wn30.csv"); cert4 = ("i21k_s","i21k_b","i21k_l","dinov2_l")
    z0 = [float(r["z"]) for r in D9 if r["kind"]=="depth" and r["partition"]=="rand6" and r["s"]!="real" and r["model"] in cert4 and float(r["s"])==0.0]
    tg0 = {r["model"] for r in D9 if r["kind"]=="depth" and r["partition"]=="rand6_t06" and float(r["s"])==0.0 and float(r["z"])<=-2}
    chk("S4.4 sentence (1): the certified four raise no alarm at zero strength, exact fraction from expR64b, and their real z is deeper than the implanted tree's",
        f"raise no alarm in any zero-strength run, {sum(z<=-2 for z in z0)} of {len(z0)}, so the verdict comes from their real hub arrangement" in main and sum(z<=-2 for z in z0)==0 and len(z0)==20
        and all(float(r["real_z"]) < float(r["z_mean_s1"]) for r in S9 if r["model"] in cert4) and "the certified set shifts with the choice of frame" in main.lower())
    chk("S4.4 sentence (2): two contrastive backbones fire at zero strength in the shrunk variant, as stated", "two contrastive backbones also fire at zero strength, so the shrunk variant is a diagnostic of power, not a substitute test" in main
        and len(tg0)==2 and tg0 <= {"clip_b","clip_l","siglip_b"})
    J = load("expR66_joint_sensitivity_summary.csv"); top = lambda r: r["dataset"] in ("imagenet","cifar100")
    a, b = sum(r["joint_genuine"]=="True" for r in J), sum(r["boot_bh_genuine"]=="True" for r in J); at, bt = sum(r["joint_genuine"]=="True" and top(r) for r in J), sum(r["boot_bh_genuine"]=="True" and top(r) for r in J)
    chk("R11 prose: the joint-sensitivity ranges in S4.2 and in the Table 1 caption equal the file", f"between {min(a,b)} and {max(a,b)} of the 72 cells" in main and f"between {min(at,bt)} and {max(at,bt)} of the 24" in main and f"{min(a,b)}--{max(a,b)} of 72" in open(TEX/"tab_census.tex").read())
    im = [r for r in load("expR52_census_haar_p999_200.csv") if r["dataset"]=="imagenet"]; u = [abs(float(r["excess"])/float(r["null_sd"])) for r in im]
    chk("B3 prose: the ImageNet excess in units of the null s.d. as stated", f"is {min(u):.0f} to {max(u):.0f} times the null's standard deviation" in main)
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
    chk("A3: correlations on raw/excess/depth for 3 gains x 4 datasets (+pooled); the raw ImageNet NC correlation reproduces Table B5 (|dr| <= 0.02); memo survival flags equal the file",
        len(C)==3*(3*5)-3*1 or len(C)>=39) and all(r["predictor"] in ("raw","excess","depth_z") for r in C)
        and abs(float(next(r["r"] for r in C if r["predictor"]=="raw" and r["gain"]=="NC_adv" and r["dataset"]=="imagenet"))-float(next(r["r"] for r in load("night/correlation_cis.csv") if r["task"]=="NC_adv" and r["dataset"]=="imagenet")))<=0.02
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
n_fail = sum(1 for _,ok,_ in checks if not ok)
for name, ok, det in checks[-12:]: print(("PASS" if ok else "FAIL"), name, ("| "+det if det and not ok else ""))
print(f"[phaseB re-total] {len(checks)-n_fail}/{len(checks)}")
for name, ok, det in checks:
    if not ok: print("FAIL(all):", name, ("| "+det if det else ""))
