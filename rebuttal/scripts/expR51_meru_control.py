#!/usr/bin/env python3
"""expR51: positive control with a published hyperbolic model.

MERU (Desai et al., 2023; Lorentz hyperboloid + entailment cones) vs its Euclidean twin,
the CLIP baseline released in the same repo (same ViT, same RedCaps data, same recipe
minus the hyperbolic lift and entailment loss). Both go through the paper's instrument
unchanged: class centroids of the projected embeddings (MERU: space-like hyperboloid
coordinates; CLIP: unit vectors), Euclidean Gromov delta, excess over the spectrum-matched
Gaussian null (null A, 20 replicates), and the star-calibrated depth test of expR50
(excess B under the Haar hub null minus the same excess on a matched star).
Secondary readings: delta in each model's native metric (Lorentz distance between
tangent-mean centroids for MERU; angular distance for CLIP), delta on the pre-projection
features, and superclass recovery (ARI of agglomerative clustering vs the frame).

Cells: {meru,clip} x {S,B,(L)} x {CIFAR-100 images (coarse-20 frame), ImageNet 50 img/class
(WordNet-30 frame), CIFAR-100 class prompts, ImageNet class prompts}.
Output: rebuttal/results/expR51_meru_control.csv ; embeddings cached under
/media/HDD_4TB_2/javi/Platonic/results/meru_cache/.
"""
import sys, os, time, csv, argparse
import numpy as np, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score

MERU_DIR = Path("/media/HDD_4TB_2/javi/meru"); sys.path.insert(0, str(MERU_DIR))
ROOT = Path("/media/HDD_4TB_2/javi/Platonic"); CK = Path("/media/HDD_4TB_2/javi/hf_cache/meru")
CACHE = ROOT/"results/meru_cache"; CACHE.mkdir(parents=True, exist_ok=True)
OUT = Path(__file__).resolve().parents[1]/"results"; CSV = OUT/"expR51_meru_control.csv"
MODELS = {"meru_s": ("train_meru_vit_s.py", "meru_vit_s.pth"), "clip_s": ("train_clip_vit_s.py", "clip_vit_s.pth"),
          "meru_b": ("train_meru_vit_b.py", "meru_vit_b.pth"), "clip_b": ("train_clip_vit_b.py", "clip_vit_b.pth"),
          "meru_l": ("train_meru_vit_l.py", "meru_vit_l.pth"), "clip_l": ("train_clip_vit_l.py", "clip_vit_l.pth")}
COARSE = {0:[4,30,55,72,95],1:[1,32,67,73,91],2:[54,62,70,82,92],3:[9,10,16,28,61],4:[0,51,53,57,83],5:[22,39,40,86,87],6:[5,20,25,84,94],7:[6,7,14,18,24],8:[3,42,43,88,97],9:[12,17,37,68,76],10:[23,33,49,60,71],11:[15,19,21,31,38],12:[34,63,64,66,75],13:[26,45,77,79,99],14:[2,11,35,46,98],15:[27,29,44,78,93],16:[36,50,65,74,80],17:[47,52,56,59,96],18:[8,13,48,58,90],19:[41,69,81,85,89]}
SUP100 = np.zeros(100, dtype=int)
for s, cls in COARSE.items():
    for c in cls: SUP100[c] = s
_WN30 = None
def wn30():
    global _WN30
    if _WN30 is None:
        WN = np.load(ROOT/"results/features/imagenet/imagenet1k_wordnet_dist.npy")
        _WN30 = AgglomerativeClustering(n_clusters=30, metric="precomputed", linkage="average").fit_predict(WN)
    return _WN30
FRAMES = {"cifar100": lambda: SUP100, "imagenet": wn30}

# ---------------- stage 1: embeddings ----------------
def load_model(key, dev):
    from meru.config import LazyConfig
    from hydra.utils import instantiate
    from meru.utils.checkpointing import CheckpointManager
    cfg = LazyConfig.load(str(MERU_DIR/"configs"/MODELS[key][0]))
    m = instantiate(cfg.model).to(dev).eval()
    CheckpointManager(model=m).load(str(CK/MODELS[key][1]))
    return m

def imagenet_class_names():
    from timm.data import ImageNetInfo
    info = ImageNetInfo()
    return [info.index_to_description(i).split(",")[0].strip() for i in range(1000)]

@torch.no_grad()
def project(m, raw):
    from meru import lorentz as L
    from meru.models import MERU
    if isinstance(m, MERU):
        with torch.autocast(m.device.type, dtype=torch.float32):
            return L.exp_map0(raw.float() * m.visual_alpha.exp(), m.curv.exp())
    return torch.nn.functional.normalize(raw.float(), dim=-1)

@torch.no_grad()
def extract_images(m, key, ds, dev, bs=256):
    import torchvision.transforms as T
    from torchvision.datasets import CIFAR100, ImageFolder
    from meru.models import CLIPBaseline
    f = CACHE/f"{key}_{ds}_img.npz"
    if f.exists(): return
    TF = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()])
    data = CIFAR100(ROOT/"data", train=True, transform=TF) if ds == "cifar100" else ImageFolder(ROOT/"data/imagenet_train_50", transform=TF)
    dl = torch.utils.data.DataLoader(data, batch_size=bs, shuffle=False, num_workers=8, pin_memory=False)
    raws, projs, labs = [], [], []; t0 = time.time()
    for x, y in dl:
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            raw = CLIPBaseline.encode_image(m, x, project=False)   # visual_proj output, before lift/normalisation
        raws.append(raw.float().cpu().numpy()); projs.append(project(m, raw).cpu().numpy()); labs.append(y.numpy())
    np.savez(f, raw=np.concatenate(raws), proj=np.concatenate(projs), labels=np.concatenate(labs),
             curv=float(m.curv.exp().item()) if hasattr(m, "curv") else 0.0)
    print(f"  extracted {key} {ds}: {sum(len(l) for l in labs)} images ({time.time()-t0:.0f}s)", flush=True)

@torch.no_grad()
def extract_text(m, key, ds, dev):
    from meru.tokenizer import Tokenizer
    f = CACHE/f"{key}_{ds}_txt.npz"
    if f.exists(): return
    if ds == "cifar100":
        from torchvision.datasets import CIFAR100
        names = [c.replace("_", " ") for c in CIFAR100(ROOT/"data", train=True).classes]
    else:
        names = imagenet_class_names()
    tok = Tokenizer(); prompts = [f"a photo of a {n}" for n in names]
    raws, projs = [], []
    for i in range(0, len(prompts), 256):
        toks = tok(prompts[i:i+256])
        raw = m.encode_text(toks, project=False)
        raws.append(raw.float().cpu().numpy())
        if hasattr(m, "curv"):
            from meru import lorentz as L
            projs.append(L.exp_map0(raw.float() * m.textual_alpha.exp(), m.curv.exp()).cpu().numpy())
        else:
            projs.append(torch.nn.functional.normalize(raw.float(), dim=-1).cpu().numpy())
    np.savez(f, raw=np.concatenate(raws), proj=np.concatenate(projs), labels=np.arange(len(prompts)),
             curv=float(m.curv.exp().item()) if hasattr(m, "curv") else 0.0)
    print(f"  extracted {key} {ds} text: {len(prompts)} prompts", flush=True)

# ---------------- stage 2: instrument ----------------
def delta_from_D(D, n_seeds=10, n_quads=500_000):
    diam = D.max(); n = len(D); out = []
    for s in range(n_seeds):
        rng = np.random.RandomState(s); i, j, k, l = (rng.randint(0, n, n_quads) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l); i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]], 1), 1); out.append(((S[:,2]-S[:,1])/2).max()/diam)
    return float(np.mean(out)), float(np.std(out))
def delta_norm(X, n_seeds=10): return delta_from_D(squareform(pdist(X, "euclidean")), n_seeds)
def specnull(C, rep):                      # null A: Gaussian coefficients on the real singular spectrum
    mu = C.mean(0); Cc = C - mu; U, S, Vt = np.linalg.svd(Cc, full_matrices=False)
    rng = np.random.RandomState(300+rep); G = rng.randn(len(C), len(S)).astype(np.float32)
    G /= G.std(0, keepdims=True) * np.sqrt(len(C)); return (G*S)@Vt + mu
def excessA(C, n_rep=20):
    dr, dr_sd = delta_norm(C, 10); nA = [delta_norm(specnull(C, r), 3)[0] for r in range(n_rep)]
    ex = float(dr - np.mean(nA)); sd = float(np.sqrt(np.std(nA, ddof=1)**2 + dr_sd**2))
    return dr, dr_sd, ex, ex/max(sd, 1e-9), float(np.mean(np.array(nA) <= dr))
def haar_sample(M, rep, seed0=700):
    mu = M.mean(0); Mc = M - mu; U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    rng = np.random.RandomState(seed0+rep); Z = rng.randn(len(M), len(M)); Q, R = np.linalg.qr(Z); Q = Q*np.sign(np.diag(R))
    return ((Q[:, :len(S)]*S)@Vt + mu).astype(np.float32)
def flatnull(C, sup, rep):
    hubs = np.stack([C[sup==s].mean(0) for s in range(sup.max()+1)]); return C - hubs[sup] + haar_sample(hubs, rep)[sup]
def excessB(C, sup, n_real=10, n_rep=10):
    dr, dr_sd = delta_norm(C, n_real); nB = [delta_norm(flatnull(C, sup, r), 3)[0] for r in range(n_rep)]
    return dr, dr_sd, float(dr-np.mean(nB)), float(np.std(nB, ddof=1))
def matched_star(C, sup, seed):
    rng = np.random.RandomState(seed); K = sup.max()+1; n, d = C.shape
    hubs = np.stack([C[sup==s].mean(0) for s in range(K)]); hub_rms = np.sqrt(((hubs-hubs.mean(0))**2).sum(1).mean())
    off = C - hubs[sup]; wr = np.array([np.sqrt((off[sup==s]**2).sum(1).mean()) for s in range(K)])
    H = rng.randn(K, d); H *= hub_rms/np.sqrt((H**2).sum(1).mean())
    Z = rng.randn(n, d); Z *= (wr[sup]/np.sqrt(d))[:, None]
    return (H[sup] + Z).astype(np.float32)
def depth_test(C, sup):
    dr, dr_sd, exB, nBsd = excessB(C, sup)
    stars = [excessB(matched_star(C, sup, s), sup, n_real=5, n_rep=5) for s in range(3)]
    exS = float(np.mean([x[2] for x in stars])); exS_sd = float(np.std([x[2] for x in stars], ddof=1))
    depth = exB - exS; z = depth/max(np.sqrt(exS_sd**2 + nBsd**2 + dr_sd**2), 1e-9)
    return exB, exS, depth, z
def native_delta(C_members, labels, n_cls, curv):
    """MERU: tangent-space mean centroid per class, Lorentz distances. CLIP (curv=0): renormalised mean, angular distances."""
    from meru import lorentz as L
    X = torch.tensor(C_members)
    if curv > 0:
        V = L.log_map0(X, curv); Cm = torch.stack([V[labels==c].mean(0) for c in range(n_cls)]); Cc = L.exp_map0(Cm, curv)
        D = L.pairwise_dist(Cc, Cc, curv).numpy(); D = 0.5*(D+D.T); np.fill_diagonal(D, 0)
    else:
        Cm = torch.stack([X[labels==c].mean(0) for c in range(n_cls)]); Cm = torch.nn.functional.normalize(Cm, dim=-1)
        D = np.arccos(np.clip((Cm@Cm.T).numpy(), -1, 1)); np.fill_diagonal(D, 0)
    return delta_from_D(D, 10)[0]
def recovery(C, sup):
    K = sup.max()+1; out = {}
    for link in ("ward", "average"):
        lab = AgglomerativeClustering(n_clusters=K, linkage=link).fit_predict(C); out[link] = adjusted_rand_score(sup, lab)
    return out

def analyse(key, ds, mod):
    f = CACHE/f"{key}_{ds}_{'img' if mod=='image' else 'txt'}.npz"
    if not f.exists(): return None
    d = np.load(f); proj, raw, lab, curv = d["proj"].astype(np.float32), d["raw"].astype(np.float32), d["labels"], float(d["curv"])
    n_cls = int(lab.max())+1; sup = FRAMES[ds]()
    C = np.stack([proj[lab==c].mean(0) for c in range(n_cls)]); Craw = np.stack([raw[lab==c].mean(0) for c in range(n_cls)])
    t0 = time.time()
    dr, dr_sd, exA, zA, frac = excessA(C)
    exB, exS, depth, zd = depth_test(C, sup)
    dnat = native_delta(proj, lab, n_cls, curv); draw = delta_norm(Craw, 5)[0]; rec = recovery(C, sup)
    row = dict(model=key, family=key.split("_")[0], size=key.split("_")[1], dataset=ds, modality=mod, K=int(sup.max()+1), n=n_cls, d=C.shape[1], curv=curv,
               delta=dr, delta_sd=dr_sd, excessA=exA, zA=zA, frac_null_below=frac, excessB_real=exB, excessB_star=exS, depth=depth, z_depth=zd,
               delta_native=dnat, delta_raw=draw, ari_ward=rec["ward"], ari_avg=rec["average"], time_s=time.time()-t0)
    print(f"{key:7s} {ds:9s} {mod:5s} K={row['K']:2d} delta {dr:.4f} excA {exA:+.4f} (z {zA:+.1f}) | excB {exB:+.4f} star {exS:+.4f} depth {depth:+.4f} (z {zd:+.1f}) | native {dnat:.4f} raw {draw:.4f} | ARI ward {rec['ward']:.2f} avg {rec['average']:.2f} ({row['time_s']:.0f}s)", flush=True)
    return row

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--stage", default="all", choices=["extract", "analyse", "all"]); A = ap.parse_args()
    if A.stage in ("extract", "all"):
        for key in A.models:
            if not (CK/MODELS[key][1]).exists(): print(f"missing checkpoint for {key}, skipping"); continue
            t0 = time.time(); m = load_model(key, A.device); print(f"loaded {key} ({time.time()-t0:.0f}s)", flush=True)
            for ds in ("cifar100", "imagenet"):
                extract_images(m, key, ds, A.device); extract_text(m, key, ds, A.device)
            del m; torch.cuda.empty_cache()
    if A.stage in ("analyse", "all"):
        done = set()
        if CSV.exists():
            done = {(r["model"], r["dataset"], r["modality"]) for r in csv.DictReader(open(CSV))}
        for key in A.models:
            for ds in ("cifar100", "imagenet"):
                for mod in ("image", "text"):
                    if (key, ds, mod) in done: continue
                    row = analyse(key, ds, mod)
                    if row is None: continue
                    new = not CSV.exists()
                    with open(CSV, "a", newline="") as fh:
                        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                        if new: w.writeheader()
                        w.writerow(row)
    print("done", flush=True)
