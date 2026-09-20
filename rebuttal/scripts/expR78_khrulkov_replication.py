#!/usr/bin/env python3
"""expR78 (parallel track, priority 2): replicate one published latent-hyperbolicity reading and calibrate it.

Setting: Khrulkov et al. (2020), Table 1: delta_rel of penultimate features of an ImageNet-pretrained CNN on CIFAR10, CIFAR100, CUB and
MiniImageNet; their ResNet34 row reads 0.26 / 0.25 / 0.25 / 0.21. Their protocol (Sec. 3 and the released code): Euclidean distances
between penultimate features, delta computed exactly on a sampled batch by the min-max matrix product (Gromov product with a base
point), delta_rel = 2 delta / diam, batches of N = 1500 points, averaged over trials (mean +- s.d.).
Here: torchvision resnet34 (IMAGENET1K_V1) avgpool features (512-d) at 224 px with ImageNet normalization (CIFAR images upsampled from
32 px, CUB and MiniImageNet resized to 256 and center-cropped); class-balanced batches of 1500 points (1500 / C per class), 10 trials
(seeds 0..9); per trial their delta_rel (exact supremum) and, on the SAME cloud, the record instrument: delta_norm with the 99.9th-
percentile statistic and the centered Haar spectrum-matched null (200 replicates), reported as excess and rank. Datasets: CIFAR train
sets (Platonic/data), CUB-200-2011 (all 11,788 images), MiniImageNet (the 100 classes of the Ravi split, 600 images each).
Usage: python expR78_khrulkov_replication.py --extract [--datasets ...] ; python expR78_khrulkov_replication.py --delta"""
import os, sys, time, argparse, json, csv
sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("TORCH_HOME", "/media/HDD_4TB_2/javi/torch_cache")
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic")); OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))   # HERE = rebuttal/scripts -> parents[1] = the repo root
FEAT = ROOT / "results/khrulkov_features"; FEAT.mkdir(parents=True, exist_ok=True)
THEIRS = {"cifar10": 0.26, "cifar100": 0.25, "cub": 0.25, "miniimagenet": 0.21}   # Khrulkov et al. 2020, Table 1, ResNet34 row
N_BATCH, N_TRIALS, N_REP = 1500, 10, 200

# ---------------- datasets
def load_cifar(name):
    import pickle
    if name == "cifar10":
        d = ROOT / "data/cifar-10-batches-py"; X, y = [], []
        for i in range(1, 6):
            b = pickle.load(open(d / f"data_batch_{i}", "rb"), encoding="bytes"); X.append(b[b"data"]); y += list(b[b"labels"])
        X = np.concatenate(X).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    else:
        b = pickle.load(open(ROOT / "data/cifar-100-python/train", "rb"), encoding="bytes"); X = b[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1); y = list(b[b"fine_labels"])
    return [("array", X[i]) for i in range(len(X))], np.asarray(y)
def load_cub():
    d = ROOT / "data/CUB/CUB_200_2011"; imgs = {int(a): b for a, b in (l.split() for l in open(d / "images.txt"))}; lab = {int(a): int(b) - 1 for a, b in (l.split() for l in open(d / "image_class_labels.txt"))}
    ids = sorted(imgs); return [("file", str(d / "images" / imgs[i])) for i in ids], np.asarray([lab[i] for i in ids])
def load_mini():
    """The 100 classes of the Ravi & Larochelle split (train+val+test CSVs of mini-imagenet-tools); the first 600 images of each class in
    sorted order from the local raw copy (the tool's own 600-image selection is not reproduced; class-balanced sampling makes this immaterial)."""
    d = Path("/media/HDD_4TB_2/javi/mini-imagenet-tools"); cls = set()
    for split in ("train", "val", "test"): cls |= {r["label"] for r in csv.DictReader(open(d / "csv_files" / f"{split}.csv"))}
    cls = sorted(cls); assert len(cls) == 100; listing = {}
    for f in os.listdir(d / "mini_imagenet"):
        if f.endswith(".JPEG"): listing.setdefault(f.split("_")[0], []).append(f)
    files, labels = [], []
    for i, c in enumerate(cls):
        fs = sorted(listing[c])[:600]; files += [("file", str(d / "mini_imagenet" / f)) for f in fs]; labels += [i] * len(fs)
    return files, np.asarray(labels)
LOADERS = {"cifar10": lambda: load_cifar("cifar10"), "cifar100": lambda: load_cifar("cifar100"), "cub": load_cub, "miniimagenet": load_mini}

# ---------------- features: torchvision resnet34, avgpool output
def extract(ds, dev="cuda:0", bs=128):
    import torch, torchvision; from torchvision import transforms as T; from PIL import Image
    items, y = LOADERS[ds](); n = len(items); print(f"{ds}: {n} images, {y.max()+1} classes")
    m = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1).eval().to(dev); m.fc = torch.nn.Identity()
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tf_small = T.Compose([T.ToPILImage(), T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), norm])
    tf_file = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
    class DS(torch.utils.data.Dataset):
        def __len__(self): return n
        def __getitem__(self, i):
            kind, v = items[i]
            return tf_small(v) if kind == "array" else tf_file(Image.open(v).convert("RGB"))
    dl = torch.utils.data.DataLoader(DS(), batch_size=bs, num_workers=4, shuffle=False); feats = []; t0 = time.time()
    with torch.no_grad():
        for i, x in enumerate(dl):
            feats.append(m(x.to(dev)).float().cpu().numpy())
            if i % 50 == 0: print(f"  {i * bs}/{n} ({time.time() - t0:.0f}s)")
    F = np.concatenate(feats).astype(np.float32); np.savez(FEAT / f"resnet34_{ds}.npz", features=F, labels=y); print("wrote", FEAT / f"resnet34_{ds}.npz", F.shape)

# ---------------- their estimator: exact delta on a batch via the min-max product (Gromov product with base point 0)
def delta_hyp(D):
    p = 0; row = D[p, :][None, :]; col = D[:, p][:, None]; XY = 0.5 * (row + col - D)     # Gromov products (x|y)_p
    n = len(D); maxmin = np.empty_like(XY)
    for i in range(0, n, 64):
        maxmin[i:i + 64] = np.max(np.minimum(XY[i:i + 64, :, None], XY[None, :, :]), axis=1)   # (XY (x) XY)_{ik} = max_j min(XY_ij, XY_jk)
    return float(np.max(maxmin - XY))
def delta_rel_theirs(X):
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X)); return 2 * delta_hyp(D) / D.max()

def run_delta():
    import calibrated_delta as cd, expR75_census_centered_haar as R75; cd.haarnull = R75.null_haar_centered
    rows = []
    for ds in THEIRS:
        f = FEAT / f"resnet34_{ds}.npz"
        if not f.exists(): print(ds, "features missing, skipped"); continue
        d = np.load(f); X = d["features"].astype(np.float64); y = d["labels"]; C = int(y.max()) + 1; per = N_BATCH // C
        for t in range(N_TRIALS):
            rng = np.random.RandomState(t); idx = np.concatenate([rng.choice(np.where(y == c)[0], min(per, int((y == c).sum())), replace=False) for c in range(C)])
            if len(idx) < N_BATCH: idx = np.concatenate([idx, rng.choice(np.setdiff1d(np.arange(len(y)), idx), N_BATCH - len(idx), replace=False)])
            Xb = X[idx]; t0 = time.time(); dr = delta_rel_theirs(Xb)
            cen = cd.census(Xb.astype(np.float32), N_REP, "haar", "p999")
            rows.append(dict(dataset=ds, trial=t, n=len(idx), classes=C, theirs_table1=THEIRS[ds], delta_rel_ours_sup=dr, delta_norm_p999=cen["delta"], null_mean=cen["null_mean"], null_sd=cen["null_sd"], excess=cen["excess"], r_above=cen["r_above"], p_left=cen["p_left"], time_s=time.time() - t0))
            print(f"{ds:12s} trial {t} delta_rel {dr:.3f} (theirs {THEIRS[ds]}) | record delta_999 {cen['delta']:.4f} excess {cen['excess']:+.4f} r {cen['r_above']}/200 ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(OUT / "expR78_khrulkov_replication.csv", index=False)
    df = pd.DataFrame(rows); S = df.groupby("dataset").agg(theirs=("theirs_table1", "first"), ours_raw_mean=("delta_rel_ours_sup", "mean"), ours_raw_sd=("delta_rel_ours_sup", "std"), excess_mean=("excess", "mean"), excess_sd=("excess", "std"), r_above_mean=("r_above", "mean"), p_left_max=("p_left", "max"), n=("n", "first")).reset_index()
    S["within_range"] = (S.ours_raw_mean - S.theirs).abs() <= 0.03
    S.to_csv(OUT / "expR78_khrulkov_replication_summary.csv", index=False); print(S.round(4).to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--extract", action="store_true"); ap.add_argument("--delta", action="store_true"); ap.add_argument("--datasets", nargs="+", default=list(THEIRS)); ap.add_argument("--gpu", default="cuda:0"); A = ap.parse_args()
    if A.extract:
        for ds in A.datasets: extract(ds, A.gpu)
    if A.delta: run_delta()
