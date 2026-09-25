#!/usr/bin/env python3
"""Regenerate the ImageNet feature caches locally (final-version pass).

The census cache (`{m}_imagenet_train.npz`, 100 img/class) and the exp5 subsample of the
full-train cache were produced on the remote 4090 machine by `codigo/scripts/
extract_imagenet_features.py` / `extract_imagenet_full.py` and never copied here. Raw
ImageNet train exists at /media/HDD_4TB_1/javi/ILSVRC2012_img_train, and both splits are
deterministic functions of the sorted file listing, so the caches are regenerated with the
SAME model definitions, transforms, fp16 rules and file splits as the original scripts:

  train    : per class, sorted files[:100]                    -> {m}_imagenet_train.npz
  exp5sub  : the exact 100/class subsample that exp5_crossmodel_alignment.load_centroids
             draws from the fulltrain file (RandomState(0), sequential per-class
             choice(replace=False)); reproduced WITHOUT extracting the 1.28M-image
             fulltrain, because choice(replace=False) consumes randomness as a function of
             the population size only, and the fulltrain listing (files[:100]+files[150:]
             per class, classes in sorted order) is deterministic
                                                              -> {m}_imagenet_exp5sub100.npz

Only these two per-image sets are extracted (~200k images/model). Deviation from the
original pipeline: none in preprocessing or ordering; only the machine and the subset of
images extracted. GPU use: one model at a time on --device.
"""
import os, sys, gc, time, argparse, traceback, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path

ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[1]/"results")))
CACHE = ROOT/"results/practical_tasks_cache"
IMAGENET_DIR = Path(os.environ.get("IMAGENET_DIR", "/media/HDD_4TB_1/javi/ILSVRC2012_img_train"))
os.environ.setdefault("HF_HOME", "/media/HDD_4TB_2/javi/hf_cache")
os.environ.setdefault("TORCH_HOME", "/media/HDD_4TB_2/javi/hf_cache/torch")

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

TRAIN_PER_CLASS = 100
TEST_START, TEST_END = 100, 150     # fulltrain skips files[100:150] (extract_imagenet_full.py)
PER_CLASS_EXP5 = 100                # exp5 PER_CLASS

MODEL_DEFS = {
    "i21k_t":   ("timm",  "vit_tiny_patch16_224.augreg_in21k",        128),
    "i21k_s":   ("timm",  "vit_small_patch16_224.augreg_in21k",       128),
    "i21k_b":   ("timm",  "vit_base_patch16_224.augreg_in21k",        128),
    "i21k_l":   ("timm",  "vit_large_patch16_224.augreg_in21k",        64),
    "dinov1_b": ("timm",  "vit_base_patch16_224.dino",                128),
    "dinov2_s": ("timm",  "vit_small_patch14_dinov2.lvd142m",         128),
    "dinov2_b": ("timm",  "vit_base_patch14_dinov2.lvd142m",          128),
    "dinov2_l": ("timm",  "vit_large_patch14_dinov2.lvd142m",          64),
    "dinov2_g": ("timm",  "vit_giant_patch14_dinov2.lvd142m",          32),
    "clip_b":   ("clip",  ("ViT-B-32", "openai"),                     128),
    "clip_l":   ("clip",  ("ViT-L-14", "openai"),                      64),
    "siglip_b": ("siglip",("ViT-B-16-SigLIP", "webli"),               128),
}

def build_indices():
    classes = sorted([d for d in os.listdir(IMAGENET_DIR) if os.path.isdir(IMAGENET_DIR/d)])
    assert len(classes) == 1000, len(classes)
    train_files, full_lengths, full_paths = [], [], []
    for ci, c in enumerate(classes):
        files = sorted(os.listdir(IMAGENET_DIR/c))
        assert len(files) >= TEST_END, (c, len(files))
        train_files.extend((IMAGENET_DIR/c/f, ci) for f in files[:TRAIN_PER_CLASS])
        fp = files[:TEST_START] + files[TEST_END:]
        full_paths.append([IMAGENET_DIR/c/f for f in fp]); full_lengths.append(len(fp))
    # exp5.load_centroids reproduction: y_full is grouped by class in order, so
    # np.where(y_full==c)[0] = arange(start_c, start_c+L_c); rng.choice picks global indices.
    rng = np.random.RandomState(0)
    starts = np.concatenate([[0], np.cumsum(full_lengths)[:-1]])
    exp5_files = []
    for ci in range(1000):
        ic = np.arange(starts[ci], starts[ci]+full_lengths[ci])
        sel = rng.choice(ic, min(PER_CLASS_EXP5, len(ic)), replace=False)
        exp5_files.extend((full_paths[ci][int(g - starts[ci])], ci) for g in sel)
    assert len(train_files) == 100_000 and len(exp5_files) == 100_000
    return train_files, exp5_files

class FileDataset(Dataset):
    def __init__(self, files, transform): self.files, self.transform = files, transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path, label = self.files[idx]
        try: img = Image.open(path).convert("RGB")
        except Exception: img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.transform(img), label

def load_model(model_key, device):
    paradigm_t, args, batch = MODEL_DEFS[model_key]
    if paradigm_t == "timm":
        import timm
        model = timm.create_model(args, pretrained=True, num_classes=0).eval().to(device)
        cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**cfg)
        return model, transform, batch
    import open_clip
    arch, pretrained = args
    full, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    model = full.visual.eval().to(device); del full
    return model, preprocess, batch

def extract(model_key, files, out_path, device):
    if out_path.exists():
        print(f"  CACHED {out_path.name}", flush=True); return
    print(f">>> {model_key} -> {out_path.name}  n={len(files)}", flush=True)
    model, transform, batch = load_model(model_key, device)
    use_fp16 = model_key in ("dinov2_l", "dinov2_g", "i21k_l", "clip_l")
    loader = DataLoader(FileDataset(files, transform), batch_size=batch, shuffle=False,
                        num_workers=8, pin_memory=False)
    feats, labels = [], []; t0 = time.time()
    with torch.no_grad():
        for bi, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(device)
            if use_fp16:
                with torch.cuda.amp.autocast(): f = model(imgs)
            else: f = model(imgs)
            if f.dim() == 3: f = f[:, 0]
            feats.append(f.float().cpu().numpy()); labels.append(lbls.numpy())
            if (bi+1) % 100 == 0:
                el = time.time()-t0; print(f"  batch {bi+1}/{len(loader)} ({el:.0f}s, {(bi+1)*batch/el:.0f} img/s)", flush=True)
    np.savez_compressed(out_path, features=np.concatenate(feats), labels=np.concatenate(labels))
    print(f"  saved {out_path.name} ({time.time()-t0:.0f}s)", flush=True)
    del model; torch.cuda.empty_cache(); gc.collect()

def extract_union(model_key, union_files, device, sets=("train", "exp5sub"), shard=None):
    """Extract each unique file once (disk order), assemble the requested npz sets by index.
    shard=(k, n): process only the k-th of n contiguous chunks and write a partial file
    {m}_imagenet_train.shard{k}of{n}.npz (train set only); merge_shards() assembles the full set."""
    outs = {s: CACHE/f"{model_key}_imagenet_{'train' if s=='train' else 'exp5sub100'}.npz" for s in sets}
    if shard is None and all(o.exists() for o in outs.values()):
        print(f"  CACHED {model_key}", flush=True); return
    files = union_files
    if shard is not None:
        k, n = shard; b = len(files)*k//n; e = len(files)*(k+1)//n; files = files[b:e]
        part = CACHE/f"{model_key}_imagenet_train.shard{k}of{n}.npz"
        if part.exists() or outs.get("train", Path("/nonexistent")).exists():
            print(f"  CACHED {part.name}", flush=True); return
    print(f">>> {model_key} n={len(files)}" + (f" shard {shard[0]}/{shard[1]}" if shard else ""), flush=True)
    model, transform, batch = load_model(model_key, device)
    use_fp16 = model_key in ("dinov2_l", "dinov2_g", "i21k_l", "clip_l")
    loader = DataLoader(FileDataset([(f, 0) for f in files], transform),
                        batch_size=batch, shuffle=False, num_workers=6, pin_memory=False)
    feats = []; t0 = time.time()
    with torch.no_grad():
        for bi, (imgs, _) in enumerate(loader):
            imgs = imgs.to(device)
            if use_fp16:
                with torch.cuda.amp.autocast(): f = model(imgs)
            else: f = model(imgs)
            if f.dim() == 3: f = f[:, 0]
            feats.append(f.float().cpu().numpy())
            if (bi+1) % 100 == 0:
                el = time.time()-t0; print(f"  batch {bi+1}/{len(loader)} ({el:.0f}s, {(bi+1)*batch/el:.0f} img/s)", flush=True)
    U = np.concatenate(feats)
    if shard is not None:
        np.savez_compressed(part, features=U, labels=np.array([TRAIN_LABEL[f] for f in files], dtype=np.int64))
        print(f"  saved {part.name} ({time.time()-t0:.0f}s)", flush=True)
    else:
        rowof = {f: i for i, f in enumerate(files)}
        for name, flist in (("train", TRAIN_FILES), ("exp5sub", EXP5_FILES)):
            if name not in sets: continue
            rows = np.array([rowof[f] for f, _ in flist])
            np.savez_compressed(outs[name], features=U[rows],
                                labels=np.array([l for _, l in flist], dtype=np.int64))
            print(f"  saved {outs[name].name} ({time.time()-t0:.0f}s)", flush=True)
    del model, U, feats; torch.cuda.empty_cache(); gc.collect()

def merge_shards(model_key, n):
    out = CACHE/f"{model_key}_imagenet_train.npz"
    parts = [CACHE/f"{model_key}_imagenet_train.shard{k}of{n}.npz" for k in range(n)]
    if out.exists() or not all(p.exists() for p in parts): return False
    F = np.concatenate([np.load(p)["features"] for p in parts]); L = np.concatenate([np.load(p)["labels"] for p in parts])
    assert len(F) == len(TRAIN_FILES) and (L == np.array([l for _, l in TRAIN_FILES])).all()
    np.savez_compressed(out, features=F, labels=L); print(f"  merged {out.name} from {n} shards", flush=True)
    return True

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--sets", default="train,exp5sub")
    ap.add_argument("--shard", default=None, help="k/n: contiguous chunk k of n (train set only)")
    ap.add_argument("--merge", default=None, help="n: merge the n shards of --models into the full train file")
    ap.add_argument("--warm-only", action="store_true")
    A = ap.parse_args()
    TRAIN_FILES, EXP5_FILES = build_indices()
    TRAIN_LABEL = {f: l for f, l in TRAIN_FILES}
    SETS = tuple(A.sets.split(","))
    pool = set(p for p, _ in TRAIN_FILES) | (set(p for p, _ in EXP5_FILES) if "exp5sub" in SETS else set())
    UNION = sorted(pool, key=str)
    print(f"index ready: {len(TRAIN_FILES)} train, {len(EXP5_FILES)} exp5sub, {len(UNION)} to extract ({A.sets})", flush=True)
    if A.merge:
        for m in A.models: merge_shards(m, int(A.merge))
        sys.exit(0)
    if A.warm_only:
        t0 = time.time(); nb = 0
        for i, f in enumerate(UNION):
            with open(f, "rb") as fh: nb += len(fh.read())
            if (i+1) % 20000 == 0: print(f"  warmed {i+1}/{len(UNION)} ({nb/1e9:.1f} GB, {time.time()-t0:.0f}s)", flush=True)
        print(f"warm done: {nb/1e9:.1f} GB in {time.time()-t0:.0f}s", flush=True); sys.exit(0)
    torch.cuda.set_device(A.device)
    shard = tuple(int(x) for x in A.shard.split("/")) if A.shard else None
    for m in A.models:
        try:
            extract_union(m, UNION, A.device, sets=SETS, shard=shard)
        except Exception as e:
            print(f"ERROR {m}: {e}", flush=True); traceback.print_exc()
    print("done", flush=True)
