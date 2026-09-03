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

# ---------- 2. Vision census (expR39b: homogeneous 20-replicate census on the census cache) ----------
e20 = load("exp20_null_ztable.csv")          # kept: raw deltas for the corollary checks below
e39 = load("expR39b_census20_cache.csv")
HIER = {"imagenet","cifar100","cifar10","dtd"}
exc = {(r["model"],r["dataset"]):float(r["excess"]) for r in e39}
zz  = {(r["model"],r["dataset"]):float(r["z"]) for r in e39}
rk  = {(r["model"],r["dataset"]):round(20*float(r["frac_null_above"])) for r in e39}
chk("census: 72 cells, none from the store", len(e39)==72 and all(int(r["store_centroids"])==0 for r in e39))
chk("69/72 sign-neg", sum(1 for v in exc.values() if v<0)==69)
pos = [(m,d) for (m,d),v in exc.items() if v>0]
chk("3 excepciones Dv2 S/B/G IN: rank 0/20, z<=+1.1", set(pos)=={("dinov2_s","imagenet"),("dinov2_b","imagenet"),("dinov2_g","imagenet")}
    and all(rk[p]==0 and zz[p]<=1.15 for p in pos), str([(p,rk[p],round(zz[p],1)) for p in pos]))
chk("57/72 below every replicate", sum(1 for v in rk.values() if v==20)==57)
chk("hier 43/48 below every replicate", sum(1 for (m,d),v in rk.items() if d in HIER and v==20)==43)
chk("flat 24/24 sign-neg, 14/24 below all", sum(1 for (m,d),v in exc.items() if d not in HIER and v<0)==24
    and sum(1 for (m,d),v in rk.items() if d not in HIER and v==20)==14)
chk("B20 caption: |z|>=2 50/72, hier z<=-3 31/48", sum(1 for v in zz.values() if abs(v)>=2)==50
    and sum(1 for (m,d),v in zz.items() if d in HIER and v<=-3)==31)
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

# ---------- 3. Text census (exp18 + 18b) ----------
e18 = {r["model"]:r for r in load("exp18_text_nulls.csv")}
g = lambda m,k: float(e18[m][k])
chk("GPT2 +0.048/-0.013/-0.034/-0.048", all(abs(g(m,"excess")-v)<0.002 for m,v in
    [("gpt2",0.0477),("gpt2_m",-0.0126),("gpt2_l",-0.0339),("gpt2_xl",-0.0477)]))
chk("OLMo -0.001 / -0.015", abs(g("olmo_1b","excess")+0.0009)<0.002 and abs(g("olmo_7b","excess")+0.0152)<0.002)
chk("Pythia -0.035..-0.041", all(-0.042<g(m,"excess")<-0.034 for m in ["pythia_410m","pythia_1b","pythia_2b8"]))
emb = ["bge_base","bge_large","gte_base","gte_large","e5_base","e5_large"]
chk("embedders +0.001..+0.011", all(0.0005<g(m,"excess")<0.0115 for m in emb),
    str([round(g(m,"excess"),4) for m in emb]))
chk("qwen -0.019", abs(g("gte_qwen2","excess")+0.0188)<0.002)
chk("embedder raw .114-.124", abs(min(g(m,"delta") for m in emb)-0.114)<0.002 and abs(max(g(m,"delta") for m in emb)-0.124)<0.002)
e18b = {r["model"]:r for r in load("exp18b_text_anisotropy.csv")}
chk("erank GPT2 1.3->56; GTE 98", abs(float(e18b["gpt2"]["erank"])-1.3)<0.2
    and abs(float(e18b["gpt2_xl"]["erank"])-56)<2 and abs(float(e18b["gte_base"]["erank"])-98)<2)
chk("pcperm z S/M/L/XL -1.4/-3.7/-7.8/-6.4", all(abs(float(e18b[m]["z_pcperm"])-v)<0.3 for m,v in
    [("gpt2",-1.4),("gpt2_m",-3.7),("gpt2_l",-7.8),("gpt2_xl",-6.4)]))

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
chk("C50 Dv2-L exc -0.124/-0.093 NC +1.58/+0.53",
    abs(agg19("dinov2_l",50,"random","excess")+0.124)<0.003 and abs(agg19("dinov2_l",50,"coherent","excess")+0.093)<0.003
    and abs(agg19("dinov2_l",50,"random","nc_adv_pp")-1.58)<0.05 and abs(agg19("dinov2_l",50,"coherent","nc_adv_pp")-0.53)<0.05)
chk("Dv2-G at-null C>=500", abs(agg19("dinov2_g",500,"random","excess"))<0.01 and abs(agg19("dinov2_g",1000,"random","excess"))<0.01)

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
