#!/usr/bin/env python3
"""
Extract ImageNet features for 12 models on REMOTE (4090).

Uses ~/datasets/ILSVRC2012_img_train (140 GB) and splits each class
into train (100 images) + test (50 images) deterministically by filename.

Saves: ~/Platonic/results/practical_tasks_cache/<model>_imagenet_<split>.npz
"""
import os, sys, gc, time, traceback, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

os.environ.setdefault("TORCH_HOME", "/home/javi/Platonic/.torch_cache")
os.environ.setdefault("HF_HOME",    "/home/javi/Platonic/.hf_cache")
os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"],    exist_ok=True)

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results/practical_tasks_cache"
CACHE.mkdir(parents=True, exist_ok=True)
IMAGENET_DIR = Path("/home/javi/datasets/ILSVRC2012_img_train")

TRAIN_PER_CLASS = 100
TEST_PER_CLASS = 50

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

# ─── Discover classes + per-class file split ──────────────────────────────
def build_index():
    classes = sorted([d for d in os.listdir(IMAGENET_DIR) if os.path.isdir(IMAGENET_DIR/d)])
    print(f"Found {len(classes)} class folders", flush=True)
    train_files = []
    test_files  = []
    for ci, c in enumerate(classes):
        files = sorted(os.listdir(IMAGENET_DIR/c))
        n = len(files)
        if n < TRAIN_PER_CLASS + TEST_PER_CLASS:
            sub = files
            train_n = max(1, int(0.7 * len(sub)))
            train_part = [(IMAGENET_DIR/c/f, ci) for f in sub[:train_n]]
            test_part  = [(IMAGENET_DIR/c/f, ci) for f in sub[train_n:]]
        else:
            train_part = [(IMAGENET_DIR/c/f, ci) for f in files[:TRAIN_PER_CLASS]]
            test_part  = [(IMAGENET_DIR/c/f, ci) for f in files[TRAIN_PER_CLASS:TRAIN_PER_CLASS+TEST_PER_CLASS]]
        train_files.extend(train_part)
        test_files.extend(test_part)
    return classes, train_files, test_files


class FileDataset(Dataset):
    def __init__(self, files, transform):
        self.files = files
        self.transform = transform
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path, label = self.files[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            # corrupt image: random gray
            img = Image.new("RGB", (224, 224), (128,128,128))
        return self.transform(img), label


def load_model(model_key):
    paradigm_t, args, batch = MODEL_DEFS[model_key]
    if paradigm_t == "timm":
        import timm
        model = timm.create_model(args, pretrained=True, num_classes=0).eval().to(DEVICE)
        cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**cfg)
        return model, transform, batch
    elif paradigm_t in ("clip", "siglip"):
        import open_clip
        arch, pretrained = args
        full, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model = full.visual.eval().to(DEVICE)
        del full
        return model, preprocess, batch
    raise ValueError(paradigm_t)


def extract(model_key, files, split, classes_dim_marker=None):
    out_path = CACHE / f"{model_key}_imagenet_{split}.npz"
    if out_path.exists():
        d = np.load(out_path)
        print(f"  CACHED {out_path.name}: {d['features'].shape}", flush=True)
        return
    print(f"\n>>> {model_key}/{split}  n={len(files)}", flush=True)
    model, transform, batch = load_model(model_key)
    use_fp16 = model_key in ("dinov2_l","dinov2_g","i21k_l","clip_l")
    ds = FileDataset(files, transform)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4, pin_memory=True)

    feats = []
    labels = []
    t0 = time.time()
    with torch.no_grad():
        for bi, (imgs, lbls) in enumerate(loader):
            imgs = imgs.to(DEVICE, non_blocking=True)
            if use_fp16:
                with torch.cuda.amp.autocast():
                    f = model(imgs)
            else:
                f = model(imgs)
            if f.dim() == 3:
                f = f[:, 0]
            feats.append(f.float().cpu().numpy())
            labels.append(lbls.numpy())
            if (bi + 1) % 50 == 0 or bi == len(loader) - 1:
                elapsed = time.time() - t0
                print(f"  batch {bi+1}/{len(loader)}  ({elapsed:.0f}s, {(bi+1)*batch/elapsed:.0f} img/s)", flush=True)
    feats = np.concatenate(feats, 0)
    labels = np.concatenate(labels, 0)
    np.savez_compressed(out_path, features=feats, labels=labels)
    print(f"  Saved {out_path.name}: {feats.shape} ({time.time()-t0:.0f}s)", flush=True)
    del model
    torch.cuda.empty_cache()
    gc.collect()


def main():
    print("Indexing ImageNet train folder...", flush=True)
    classes, train_files, test_files = build_index()
    print(f"Train: {len(train_files)} files | Test: {len(test_files)} files", flush=True)

    for model_key in MODEL_DEFS:
        try:
            extract(model_key, train_files, "train")
            extract(model_key, test_files, "test")
        except Exception as e:
            print(f"ERROR {model_key}: {e}", flush=True)
            traceback.print_exc()


if __name__ == "__main__":
    main()
