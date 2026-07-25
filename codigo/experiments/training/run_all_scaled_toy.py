#!/usr/bin/env python3
"""
Full Pipeline: Feature Extraction + Training + Checkpoints + Inference Images + Correlations
============================================================================================

12 models x 5 datasets x 3 heads x 3 seeds = 540 training runs
Plus feature extraction for all missing combos.

Run with:
    TMPDIR=/home/javi/Platonic/.tmp TORCH_HOME=/home/javi/Platonic/.torch_cache \
    HF_HOME=/home/javi/Platonic/.hf_cache \
    CUDA_VISIBLE_DEVICES=1 python3 -u experiments/training/run_all_scaled_toy.py

Phases:
  1. Extract train+test features for all model x dataset combos (save .npz)
  2. Train MLP + head (Linear, Hyperbolic, Cosine) for each combo
  3. Save .pth checkpoints
  4. Generate inference comparison images
  5. Compute correlations with delta/ORC
"""

import sys, os, gc, time, warnings, traceback
warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Environment setup
os.makedirs("/home/javi/Platonic/.tmp", exist_ok=True)
os.makedirs("/home/javi/Platonic/.torch_cache", exist_ok=True)
os.makedirs("/home/javi/Platonic/.hf_cache", exist_ok=True)
os.environ.setdefault("TMPDIR", "/home/javi/Platonic/.tmp")
os.environ.setdefault("TORCH_HOME", "/home/javi/Platonic/.torch_cache")
os.environ.setdefault("HF_HOME", "/home/javi/Platonic/.hf_cache")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
#  CONSTANTS AND PATHS
# ═══════════════════════════════════════════════════════════════════════════

ROOT = Path("/home/javi/Platonic")
CACHE = ROOT / "results" / "practical_tasks_cache"
CHECKPOINT_DIR = ROOT / "results" / "training" / "checkpoints"
INFERENCE_DIR = ROOT / "results" / "training" / "inference_comparisons"
TRAINING_DIR = ROOT / "results" / "training"
DATA_DIR = ROOT / "data"

CACHE.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

# ═══════════════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════
# key -> (paradigm, loader_type, loader_args, batch_size_for_extraction)

MODEL_DEFS = {
    # --- DINOv2 family ---
    "dinov2_s": ("SSL",         "timm", "vit_small_patch14_dinov2.lvd142m",        128),
    "dinov2_b": ("SSL",         "timm", "vit_base_patch14_dinov2.lvd142m",         128),
    "dinov2_l": ("SSL",         "timm", "vit_large_patch14_dinov2.lvd142m",        64),
    "dinov2_g": ("SSL",         "timm", "vit_giant_patch14_dinov2.lvd142m",        32),
    # --- DINOv1 ---
    "dinov1_b": ("SSL",         "timm", "vit_base_patch16_224.dino",               128),
    # --- ImageNet-21k supervised ---
    "i21k_t":   ("supervised",  "timm", "vit_tiny_patch16_224.augreg_in21k",       128),
    "i21k_s":   ("supervised",  "timm", "vit_small_patch16_224.augreg_in21k",      128),
    "i21k_b":   ("supervised",  "timm", "vit_base_patch16_224.augreg_in21k",       128),
    "i21k_l":   ("supervised",  "timm", "vit_large_patch16_224.augreg_in21k",      64),
    # --- CLIP family ---
    "clip_b":   ("contrastive", "clip", ("ViT-B-32", "openai"),                    128),
    "clip_l":   ("contrastive", "clip", ("ViT-L-14", "openai"),                    64),
    # --- SigLIP ---
    "siglip_b": ("contrastive", "siglip", ("ViT-B-16-SigLIP", "webli"),           128),
}

ALL_MODELS = [
    # REMOTE NODE: only non-ViT (i21k_*) models -- ViTs are running on local
    'clip_b', 'clip_l', 'siglip_b',
    'dinov1_b',
    'dinov2_s', 'dinov2_b', 'dinov2_l', 'dinov2_g',
]

# Datasets (excluding ImageNet -- we focus on the 5 smaller ones)
ALL_DATASETS = ["cifar100", "cifar10", "dtd", "fashionmnist", "mnist"]

# Short name -> torchvision dataset info
DATASET_INFO = {
    "cifar100":     {"n_classes": 100, "name": "CIFAR-100"},
    "cifar10":      {"n_classes": 10,  "name": "CIFAR-10"},
    "dtd":          {"n_classes": 47,  "name": "DTD"},
    "fashionmnist": {"n_classes": 10,  "name": "FashionMNIST"},
    "mnist":        {"n_classes": 10,  "name": "MNIST"},
}


# ═══════════════════════════════════════════════════════════════════════════
#  SUPERCLASS MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════

# CIFAR-100: 100 fine -> 20 coarse
COARSE_LABELS = {
    0:  [4, 30, 55, 72, 95],   1:  [1, 32, 67, 73, 91],   2:  [54, 62, 70, 82, 92],
    3:  [9, 10, 16, 28, 61],   4:  [0, 51, 53, 57, 83],   5:  [22, 39, 40, 86, 87],
    6:  [5, 20, 25, 84, 94],   7:  [6, 7, 14, 18, 24],    8:  [3, 42, 43, 88, 97],
    9:  [12, 17, 37, 68, 76],  10: [23, 33, 49, 60, 71],  11: [15, 19, 21, 31, 38],
    12: [34, 63, 64, 66, 75],  13: [26, 45, 77, 79, 99],  14: [2, 11, 35, 46, 98],
    15: [27, 29, 44, 78, 93],  16: [36, 50, 65, 74, 80],  17: [47, 52, 56, 59, 96],
    18: [8, 13, 48, 58, 90],   19: [41, 69, 81, 85, 89],
}
FINE_TO_COARSE = {}
for coarse, fines in COARSE_LABELS.items():
    for f in fines:
        FINE_TO_COARSE[f] = coarse

# CIFAR-10: vehicle vs animal
CIFAR10_SUPER = {0: 1, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1}

# FashionMNIST: upper(0), lower(1), dress(3), foot(2), acc(3)
FMNIST_SUPER = {0: 0, 1: 1, 2: 0, 3: 3, 4: 0, 5: 2, 6: 0, 7: 2, 8: 3, 9: 2}

# MNIST: round(0), straight(1), mixed(2)
MNIST_SUPER = {0: 0, 1: 1, 2: 2, 3: 2, 4: 1, 5: 2, 6: 0, 7: 1, 8: 0, 9: 0}


def get_n_superclasses(dataset_key):
    """Return number of superclasses for a dataset."""
    if dataset_key == "cifar100":
        return 20
    elif dataset_key == "cifar10":
        return 2
    elif dataset_key == "fashionmnist":
        return 4
    elif dataset_key == "mnist":
        return 3
    elif dataset_key == "dtd":
        return 8
    return None


def map_to_superclass(labels, dataset_key):
    """Map fine labels -> superclass labels."""
    if dataset_key == "cifar100":
        return np.array([FINE_TO_COARSE[y] for y in labels])
    elif dataset_key == "cifar10":
        return np.array([CIFAR10_SUPER[y] for y in labels])
    elif dataset_key == "fashionmnist":
        return np.array([FMNIST_SUPER[y] for y in labels])
    elif dataset_key == "mnist":
        return np.array([MNIST_SUPER[y] for y in labels])
    elif dataset_key == "dtd":
        # Will be computed from feature centroids when needed
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  MATPLOTLIB STYLE
# ═══════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "savefig.dpi":       200,
    "axes.grid":         False,
})


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_path(model_key, dataset_key, split):
    """Get path to .npz feature file."""
    return CACHE / f"{model_key}_{dataset_key}_{split}.npz"


def features_exist(model_key, dataset_key):
    """Check if both train and test features exist."""
    return (get_feature_path(model_key, dataset_key, "train").exists() and
            get_feature_path(model_key, dataset_key, "test").exists())


def load_torchvision_dataset(dataset_key, split="train"):
    """Load a torchvision dataset. Returns (dataset, n_classes)."""
    from torchvision import datasets as tv_datasets

    is_train = (split == "train")
    n_classes = DATASET_INFO[dataset_key]["n_classes"]

    if dataset_key == "cifar100":
        ds = tv_datasets.CIFAR100(root=str(DATA_DIR), train=is_train, download=True)
    elif dataset_key == "cifar10":
        ds = tv_datasets.CIFAR10(root=str(DATA_DIR), train=is_train, download=True)
    elif dataset_key == "dtd":
        if is_train:
            ds_train = tv_datasets.DTD(root=str(DATA_DIR), split="train", download=True)
            ds_val = tv_datasets.DTD(root=str(DATA_DIR), split="val", download=True)
            ds = torch.utils.data.ConcatDataset([ds_train, ds_val])
        else:
            ds = tv_datasets.DTD(root=str(DATA_DIR), split="test", download=True)
    elif dataset_key == "fashionmnist":
        ds = tv_datasets.FashionMNIST(root=str(DATA_DIR), train=is_train, download=True)
    elif dataset_key == "mnist":
        ds = tv_datasets.MNIST(root=str(DATA_DIR), train=is_train, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    return ds, n_classes


class WrappedDataset(torch.utils.data.Dataset):
    """Wraps a torchvision dataset with a custom transform, ensuring RGB conversion."""
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if isinstance(label, torch.Tensor):
            label = label.item()
        # Ensure PIL and RGB
        if isinstance(img, torch.Tensor):
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(img)
        if hasattr(img, 'convert'):
            img = img.convert("RGB")
        img = self.transform(img)
        return img, label


def load_model_and_transform(model_key):
    """Load model + preprocessing. Returns (model, transform)."""
    mdef = MODEL_DEFS[model_key]
    paradigm, loader_type, loader_args, batch_size = mdef

    if loader_type == "timm":
        import timm
        model_name = loader_args
        model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(DEVICE)
        data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        transform = timm.data.create_transform(**data_cfg)
        return model, transform

    elif loader_type in ("clip", "siglip"):
        import open_clip
        arch, pretrained = loader_args
        full_model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        model = full_model.visual.eval().to(DEVICE)
        del full_model
        return model, preprocess

    else:
        raise ValueError(f"Unknown loader_type: {loader_type}")


def extract_features_for_model(model_key):
    """Extract features for one model across all datasets."""
    # Check which datasets need extraction
    needed = []
    for ds_key in ALL_DATASETS:
        if not features_exist(model_key, ds_key):
            needed.append(ds_key)

    if not needed:
        print(f"  [{model_key}] All features already cached. Skipping.", flush=True)
        return

    print(f"\n{'='*70}", flush=True)
    print(f"  Extracting features for: {model_key}", flush=True)
    print(f"  Datasets needed: {needed}", flush=True)
    print(f"{'='*70}", flush=True)

    # Load model once
    model, transform = load_model_and_transform(model_key)
    batch_size = MODEL_DEFS[model_key][3]

    # Use fp16 for large models
    use_fp16 = model_key in ("dinov2_l", "dinov2_g", "i21k_l", "clip_l")

    for ds_key in needed:
        for split in ["train", "test"]:
            out_path = get_feature_path(model_key, ds_key, split)
            if out_path.exists():
                print(f"    {model_key}_{ds_key}_{split}: already exists, skip", flush=True)
                continue

            print(f"    Extracting {model_key} / {ds_key} / {split}...", flush=True)
            t0 = time.time()

            try:
                ds, n_classes = load_torchvision_dataset(ds_key, split)
                wrapped = WrappedDataset(ds, transform)
                loader = DataLoader(wrapped, batch_size=batch_size, shuffle=False,
                                    num_workers=2, pin_memory=True, drop_last=False,
                                    persistent_workers=False)

                all_feats = []
                all_labels = []
                model.eval()
                n_batches = len(loader)
                with torch.no_grad():
                    for bi, (imgs, labels) in enumerate(loader):
                        imgs = imgs.to(DEVICE)
                        if use_fp16:
                            with torch.cuda.amp.autocast():
                                feats = model(imgs)
                        else:
                            feats = model(imgs)
                        # Handle ViT output shapes
                        if feats.dim() == 3:
                            feats = feats[:, 0]  # CLS token
                        all_feats.append(feats.float().cpu().numpy())
                        all_labels.append(labels.numpy() if isinstance(labels, torch.Tensor) else np.array(labels))
                        if (bi + 1) % 50 == 0 or bi == n_batches - 1:
                            print(f"        batch {bi+1}/{n_batches}", flush=True)

                features = np.concatenate(all_feats, axis=0)
                labels = np.concatenate(all_labels, axis=0)
                np.savez_compressed(str(out_path), features=features, labels=labels)
                elapsed = time.time() - t0
                print(f"      Saved {features.shape} -> {out_path.name} ({elapsed:.1f}s)", flush=True)

            except Exception as e:
                print(f"      ERROR extracting {model_key}/{ds_key}/{split}: {e}", flush=True)
                traceback.print_exc()
                continue

    # Cleanup model
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  [{model_key}] Feature extraction complete.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════

class MLPEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class HyperbolicHead(nn.Module):
    def __init__(self, embed_dim, n_classes, c_init=1.0):
        super().__init__()
        self.c = nn.Parameter(torch.tensor(float(c_init)))
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.prototypes = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.01)

    def forward(self, x):
        c = torch.abs(self.c).clamp(min=1e-5)
        sqrt_c = torch.sqrt(c)
        x_scaled = x * torch.abs(self.scale)
        x_norm = x_scaled.norm(dim=-1, keepdim=True).clamp(min=1e-5)
        x_hyp = torch.tanh(sqrt_c * x_norm / 2) / (sqrt_c * x_norm) * x_scaled

        max_norm = (1.0 / sqrt_c) - 1e-3
        x_hyp = x_hyp * torch.clamp(max_norm / x_hyp.norm(dim=-1, keepdim=True).clamp(min=1e-7), max=1.0)

        p = self.prototypes
        p = p * torch.clamp(max_norm / p.norm(dim=-1, keepdim=True).clamp(min=1e-7), max=1.0)

        x_e = x_hyp.unsqueeze(1)
        p_e = p.unsqueeze(0)
        diff_sq = ((x_e - p_e)**2).sum(-1)
        x_sq = (x_e**2).sum(-1).clamp(max=1.0 / c.item() - 1e-3)
        p_sq = (p_e**2).sum(-1).clamp(max=1.0 / c.item() - 1e-3)
        arg = 1 + 2 * c * diff_sq / ((1 - c * x_sq) * (1 - c * p_sq)).clamp(min=1e-5)
        dists = (2.0 / sqrt_c) * torch.acosh(arg.clamp(min=1 + 1e-5))
        return -dists


class CosineHead(nn.Module):
    def __init__(self, embed_dim, n_classes, temp=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_classes, embed_dim) * 0.01)
        self.temp = nn.Parameter(torch.tensor(temp))

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weight, dim=-1)
        return x_norm @ w_norm.T / torch.abs(self.temp).clamp(min=0.01)


class LinearHead(nn.Module):
    def __init__(self, embed_dim, n_classes):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


HEAD_CLASSES = {
    "linear":     LinearHead,
    "hyperbolic": HyperbolicHead,
    "cosine":     CosineHead,
}


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2+3: TRAINING + CHECKPOINT SAVING
# ═══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(features_train, labels_train, features_test, labels_test,
                       head_type, n_classes, in_dim, embed_dim=128,
                       epochs=100, seed=0, device=DEVICE):
    """Train MLP encoder + head. Returns (encoder, head, best_acc, final_acc, preds, embeddings)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    encoder = MLPEncoder(in_dim, 256, embed_dim).to(device)
    head_cls = HEAD_CLASSES[head_type]
    if head_type == "linear":
        head = head_cls(embed_dim, n_classes).to(device)
    elif head_type == "hyperbolic":
        head = head_cls(embed_dim, n_classes).to(device)
    elif head_type == "cosine":
        head = head_cls(embed_dim, n_classes).to(device)

    params = list(encoder.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        torch.tensor(features_train, dtype=torch.float32),
        torch.tensor(labels_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True,
                              num_workers=0, pin_memory=True)

    best_acc = 0
    best_encoder_state = None
    best_head_state = None

    for epoch in range(epochs):
        encoder.train()
        head.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            emb = encoder(batch_x)
            logits = head(emb)
            loss = criterion(logits, batch_y)
            if torch.isnan(loss):
                print(f"      NaN at epoch {epoch} -- aborting", flush=True)
                return None, None, 0.0, 0.0, None, None
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        scheduler.step()

        # Eval every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            encoder.eval()
            head.eval()
            with torch.no_grad():
                test_x = torch.tensor(features_test, dtype=torch.float32).to(device)
                test_y = torch.tensor(labels_test, dtype=torch.long).to(device)
                # Process in chunks to avoid OOM
                all_logits = []
                chunk_size = 2048
                for i in range(0, len(test_x), chunk_size):
                    chunk = test_x[i:i + chunk_size]
                    emb_chunk = encoder(chunk)
                    logits_chunk = head(emb_chunk)
                    all_logits.append(logits_chunk)
                logits = torch.cat(all_logits, dim=0)
                acc = (logits.argmax(1) == test_y).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
                    best_head_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    # Restore best and do final eval
    if best_encoder_state is not None:
        encoder.load_state_dict(best_encoder_state)
        head.load_state_dict(best_head_state)
    encoder = encoder.to(device).eval()
    head = head.to(device).eval()

    with torch.no_grad():
        test_x = torch.tensor(features_test, dtype=torch.float32).to(device)
        all_embs = []
        all_logits = []
        chunk_size = 2048
        for i in range(0, len(test_x), chunk_size):
            chunk = test_x[i:i + chunk_size]
            emb_chunk = encoder(chunk)
            logits_chunk = head(emb_chunk)
            all_embs.append(emb_chunk.cpu())
            all_logits.append(logits_chunk.cpu())
        logits = torch.cat(all_logits, dim=0)
        embeddings = torch.cat(all_embs, dim=0).numpy()
        preds = logits.argmax(1).numpy()
        final_acc = (preds == labels_test).mean()

    return encoder, head, best_acc, final_acc, preds, embeddings


def save_checkpoint(encoder, head, model_key, dataset_key, head_type, seed,
                    in_dim, embed_dim, n_classes, best_acc):
    """Save a checkpoint."""
    save_dir = CHECKPOINT_DIR / f"{model_key}_{dataset_key}_{head_type}_seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        'encoder': encoder.state_dict(),
        'head': head.state_dict(),
        'config': {
            'in_dim': in_dim,
            'embed_dim': embed_dim,
            'n_classes': n_classes,
            'head_type': head_type,
            'hidden_dim': 256,
        },
        'best_acc': best_acc,
        'seed': seed,
        'model_key': model_key,
        'dataset_key': dataset_key,
    }, save_dir / "model.pth")


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: INFERENCE COMPARISON IMAGES
# ═══════════════════════════════════════════════════════════════════════════

def get_class_names(dataset_key):
    """Get human-readable class names."""
    if dataset_key == "cifar100":
        from torchvision.datasets import CIFAR100
        ds = CIFAR100(root=str(DATA_DIR), train=False, download=True)
        # Access the classes attribute
        if hasattr(ds, 'classes'):
            return ds.classes
        return [str(i) for i in range(100)]
    elif dataset_key == "cifar10":
        return ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_key == "fashionmnist":
        return ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']
    elif dataset_key == "mnist":
        return [str(i) for i in range(10)]
    elif dataset_key == "dtd":
        from torchvision.datasets import DTD
        ds = DTD(root=str(DATA_DIR), split="test", download=True)
        if hasattr(ds, 'classes'):
            return ds.classes
        return [str(i) for i in range(47)]
    return None


def generate_inference_comparison(model_key, dataset_key, preds_linear, preds_hyp,
                                  labels_test, n_show=20):
    """Generate comparison image showing Linear vs Hyperbolic predictions."""
    class_names = get_class_names(dataset_key)
    if class_names is None:
        return

    save_path = INFERENCE_DIR / f"{model_key}_{dataset_key}_inference.png"

    # Load test images
    try:
        ds, _ = load_torchvision_dataset(dataset_key, "test")
    except Exception as e:
        print(f"    Could not load test images for {dataset_key}: {e}", flush=True)
        return

    # Find interesting indices: where H and L disagree
    n_test = len(labels_test)
    h_right_l_wrong = np.where((preds_hyp == labels_test) & (preds_linear != labels_test))[0]
    l_right_h_wrong = np.where((preds_linear == labels_test) & (preds_hyp != labels_test))[0]
    both_right = np.where((preds_linear == labels_test) & (preds_hyp == labels_test))[0]
    both_wrong = np.where((preds_linear != labels_test) & (preds_hyp != labels_test))[0]

    # Pick a mix
    rng = np.random.RandomState(42)
    selected = []
    # Up to 5 H-right-L-wrong
    if len(h_right_l_wrong) > 0:
        selected.extend(rng.choice(h_right_l_wrong, min(5, len(h_right_l_wrong)), replace=False).tolist())
    # Up to 5 L-right-H-wrong
    if len(l_right_h_wrong) > 0:
        selected.extend(rng.choice(l_right_h_wrong, min(5, len(l_right_h_wrong)), replace=False).tolist())
    # Fill rest with both_right
    remaining = n_show - len(selected)
    if remaining > 0 and len(both_right) > 0:
        selected.extend(rng.choice(both_right, min(remaining, len(both_right)), replace=False).tolist())
    # If still short, add both_wrong
    remaining = n_show - len(selected)
    if remaining > 0 and len(both_wrong) > 0:
        selected.extend(rng.choice(both_wrong, min(remaining, len(both_wrong)), replace=False).tolist())

    if len(selected) == 0:
        print(f"    No test samples for {model_key}/{dataset_key}", flush=True)
        return

    selected = selected[:n_show]

    n_cols = 5
    n_rows = (len(selected) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"{model_key} / {DATASET_INFO[dataset_key]['name']}\nLinear vs Hyperbolic predictions",
                 fontsize=14, y=1.02)

    for idx_in_grid, test_idx in enumerate(selected):
        row = idx_in_grid // n_cols
        col = idx_in_grid % n_cols
        ax = axes[row, col]

        # Get image
        img, _ = ds[test_idx]
        if isinstance(img, torch.Tensor):
            from torchvision.transforms.functional import to_pil_image
            img = to_pil_image(img)
        if hasattr(img, 'convert'):
            img = img.convert("RGB")
        img_arr = np.array(img)
        ax.imshow(img_arr)

        true_label = labels_test[test_idx]
        pred_l = preds_linear[test_idx]
        pred_h = preds_hyp[test_idx]

        true_name = class_names[true_label] if true_label < len(class_names) else str(true_label)
        pred_l_name = class_names[pred_l] if pred_l < len(class_names) else str(pred_l)
        pred_h_name = class_names[pred_h] if pred_h < len(class_names) else str(pred_h)

        # Title with colors
        l_correct = pred_l == true_label
        h_correct = pred_h == true_label

        title_parts = [f"True: {true_name}"]
        l_color = "green" if l_correct else "red"
        h_color = "green" if h_correct else "red"

        ax.set_title(f"True: {true_name}\nL:{pred_l_name} | H:{pred_h_name}",
                     fontsize=7, pad=3)

        # Border color based on category
        if h_correct and not l_correct:
            border_color = 'blue'  # H wins
        elif l_correct and not h_correct:
            border_color = 'orange'  # L wins
        elif l_correct and h_correct:
            border_color = 'green'  # both right
        else:
            border_color = 'red'  # both wrong

        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty axes
    for idx_in_grid in range(len(selected), n_rows * n_cols):
        row = idx_in_grid // n_cols
        col = idx_in_grid % n_cols
        axes[row, col].set_visible(False)

    fig.text(0.5, -0.02,
             "Border: Blue=H wins, Orange=L wins, Green=both right, Red=both wrong",
             ha='center', fontsize=10)
    plt.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Inference image saved: {save_path.name}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: COMPUTE CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════

def compute_delta_embeddings(embeddings, n_sample=300, n_quads=3000, seed=42):
    """Gromov delta on embeddings."""
    from scipy.spatial.distance import pdist, squareform
    rng = np.random.RandomState(seed)
    n = min(n_sample, len(embeddings))
    idx = rng.choice(len(embeddings), n, replace=False)
    X = embeddings[idx]
    D = squareform(pdist(X, metric='euclidean'))

    quads = np.array([rng.choice(n, 4, replace=False) for _ in range(n_quads)])
    i, j, k, l = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    s1 = D[i, j] + D[k, l]
    s2 = D[i, k] + D[j, l]
    s3 = D[i, l] + D[j, k]
    sums = np.stack([s1, s2, s3], axis=1)
    sums.sort(axis=1)
    deltas = (sums[:, 2] - sums[:, 1]) / 2.0
    return deltas.mean()


def compute_correlations(results_df):
    """Correlate hyperbolic advantage with geometric measures."""
    # Load geometric measures
    geom_csv = ROOT / "results" / "dinov2_family_across_datasets.csv"
    if not geom_csv.exists():
        print("  WARNING: dinov2_family_across_datasets.csv not found, skipping correlations", flush=True)
        return None

    geom_df = pd.read_csv(geom_csv)

    # Compute H improvement per model x dataset
    summary_rows = []
    for (model_key, ds_key), grp in results_df.groupby(["model", "dataset"]):
        linear_acc = grp[grp["head"] == "linear"]["final_acc"].values
        hyp_acc = grp[grp["head"] == "hyperbolic"]["final_acc"].values
        cosine_acc = grp[grp["head"] == "cosine"]["final_acc"].values

        if len(linear_acc) == 0 or len(hyp_acc) == 0:
            continue

        h_improvement = hyp_acc.mean() - linear_acc.mean()
        c_improvement = cosine_acc.mean() - linear_acc.mean() if len(cosine_acc) > 0 else np.nan

        # Map dataset_key to the name used in geom_csv
        ds_name_map = {
            "cifar100": "CIFAR-100", "cifar10": "CIFAR-10", "dtd": "DTD",
            "fashionmnist": "FashionMNIST", "mnist": "MNIST",
        }
        ds_name = ds_name_map.get(ds_key, ds_key)

        # Look up geometric measures
        match = geom_df[(geom_df["model"] == model_key) & (geom_df["dataset"] == ds_name)]
        if len(match) == 0:
            delta_mean = np.nan
            delta_ratio = np.nan
            orc_mean = np.nan
        else:
            delta_mean = match["delta_mean"].values[0]
            delta_ratio = match["delta_ratio"].values[0]
            orc_mean = match["orc_mean"].values[0]

        summary_rows.append({
            "model": model_key,
            "dataset": ds_key,
            "acc_linear": linear_acc.mean(),
            "acc_hyperbolic": hyp_acc.mean(),
            "acc_cosine": cosine_acc.mean() if len(cosine_acc) > 0 else np.nan,
            "h_improvement": h_improvement,
            "c_improvement": c_improvement,
            "delta_mean": delta_mean,
            "delta_ratio": delta_ratio,
            "orc_mean": orc_mean,
        })

    if not summary_rows:
        return None

    summary_df = pd.DataFrame(summary_rows)

    # Compute correlations
    from scipy.stats import pearsonr, spearmanr
    corr_rows = []
    for geom_col in ["delta_mean", "delta_ratio", "orc_mean"]:
        valid = summary_df.dropna(subset=[geom_col, "h_improvement"])
        if len(valid) >= 4:
            r_p, p_p = pearsonr(valid[geom_col], valid["h_improvement"])
            r_s, p_s = spearmanr(valid[geom_col], valid["h_improvement"])
            corr_rows.append({
                "geometry_metric": geom_col,
                "n": len(valid),
                "pearson_r": r_p,
                "pearson_p": p_p,
                "spearman_r": r_s,
                "spearman_p": p_s,
            })

    corr_df = pd.DataFrame(corr_rows)
    return summary_df, corr_df


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def main():
    overall_t0 = time.time()

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 1: Feature extraction
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 1: Feature Extraction", flush=True)
    print("=" * 80, flush=True)

    for model_key in ALL_MODELS:
        extract_features_for_model(model_key)

    # Report what we have
    print("\nFeature availability matrix:", flush=True)
    for model_key in ALL_MODELS:
        statuses = []
        for ds_key in ALL_DATASETS:
            if features_exist(model_key, ds_key):
                statuses.append(f"{ds_key}:OK")
            else:
                statuses.append(f"{ds_key}:MISSING")
        print(f"  {model_key:12s}  {', '.join(statuses)}", flush=True)

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 2+3: Training + Checkpoints
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 2+3: Training + Checkpoints", flush=True)
    print("=" * 80, flush=True)

    HEAD_TYPES = ["linear", "hyperbolic", "cosine"]
    N_SEEDS = 3
    EPOCHS = 100
    EMBED_DIM = 128

    all_results = []
    # Store predictions for inference comparison (Phase 4)
    # Key: (model_key, dataset_key, head_type) -> list of (seed, preds)
    all_preds = {}

    total_combos = len(ALL_MODELS) * len(ALL_DATASETS) * len(HEAD_TYPES) * N_SEEDS
    done = 0

    for model_key in ALL_MODELS:
        for ds_key in ALL_DATASETS:
            if not features_exist(model_key, ds_key):
                print(f"\n  SKIP {model_key}/{ds_key}: features missing", flush=True)
                done += len(HEAD_TYPES) * N_SEEDS
                continue

            # Load features
            train_data = np.load(str(get_feature_path(model_key, ds_key, "train")))
            test_data = np.load(str(get_feature_path(model_key, ds_key, "test")))
            X_train = train_data["features"].astype(np.float32)
            y_train = train_data["labels"].astype(np.int64)
            X_test = test_data["features"].astype(np.float32)
            y_test = test_data["labels"].astype(np.int64)

            in_dim = X_train.shape[1]
            n_classes = DATASET_INFO[ds_key]["n_classes"]

            print(f"\n{'─'*60}", flush=True)
            print(f"  {model_key} / {ds_key}  (in_dim={in_dim}, n_classes={n_classes})", flush=True)
            print(f"  train={X_train.shape}, test={X_test.shape}", flush=True)
            print(f"{'─'*60}", flush=True)

            for head_type in HEAD_TYPES:
                seed_results = []
                for seed in range(N_SEEDS):
                    done += 1

                    # Check if checkpoint already exists
                    ckpt_dir = CHECKPOINT_DIR / f"{model_key}_{ds_key}_{head_type}_seed{seed}"
                    ckpt_path = ckpt_dir / "model.pth"
                    if ckpt_path.exists():
                        # Load existing checkpoint to get accuracy
                        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
                        best_acc = ckpt.get("best_acc", 0.0)
                        print(f"  [{done}/{total_combos}] {model_key}/{ds_key}/{head_type}/s{seed}: "
                              f"CACHED acc={best_acc:.4f}", flush=True)

                        # We still need predictions for inference comparison
                        # Reload and evaluate
                        encoder = MLPEncoder(in_dim, 256, EMBED_DIM).to(DEVICE)
                        head_cls = HEAD_CLASSES[head_type]
                        if head_type == "linear":
                            head_obj = head_cls(EMBED_DIM, n_classes).to(DEVICE)
                        else:
                            head_obj = head_cls(EMBED_DIM, n_classes).to(DEVICE)
                        encoder.load_state_dict(ckpt["encoder"])
                        head_obj.load_state_dict(ckpt["head"])
                        encoder.eval()
                        head_obj.eval()

                        with torch.no_grad():
                            test_x = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
                            all_logits = []
                            all_embs = []
                            for i in range(0, len(test_x), 2048):
                                chunk = test_x[i:i+2048]
                                emb_chunk = encoder(chunk)
                                logits_chunk = head_obj(emb_chunk)
                                all_logits.append(logits_chunk.cpu())
                                all_embs.append(emb_chunk.cpu())
                            logits = torch.cat(all_logits, dim=0)
                            embeddings = torch.cat(all_embs, dim=0).numpy()
                            preds = logits.argmax(1).numpy()
                            final_acc = (preds == y_test).mean()

                        del encoder, head_obj
                        torch.cuda.empty_cache()

                        row = {
                            "model": model_key, "dataset": ds_key,
                            "head": head_type, "seed": seed,
                            "best_acc": best_acc, "final_acc": final_acc,
                            "in_dim": in_dim, "embed_dim": EMBED_DIM,
                            "n_classes": n_classes,
                        }
                        all_results.append(row)

                        # Store preds for seed 0
                        if seed == 0:
                            key = (model_key, ds_key, head_type)
                            all_preds[key] = preds

                        continue

                    print(f"  [{done}/{total_combos}] {model_key}/{ds_key}/{head_type}/s{seed}...",
                          end="", flush=True)
                    t0 = time.time()

                    try:
                        encoder, head_obj, best_acc, final_acc, preds, embeddings = \
                            train_and_evaluate(
                                X_train, y_train, X_test, y_test,
                                head_type, n_classes, in_dim,
                                embed_dim=EMBED_DIM, epochs=EPOCHS, seed=seed,
                                device=DEVICE)

                        if encoder is None:
                            print(f" FAILED (NaN)", flush=True)
                            row = {
                                "model": model_key, "dataset": ds_key,
                                "head": head_type, "seed": seed,
                                "best_acc": np.nan, "final_acc": np.nan,
                                "in_dim": in_dim, "embed_dim": EMBED_DIM,
                                "n_classes": n_classes,
                            }
                            all_results.append(row)
                            continue

                        elapsed = time.time() - t0
                        print(f" best={best_acc:.4f} final={final_acc:.4f} ({elapsed:.1f}s)", flush=True)

                        # Save checkpoint
                        save_checkpoint(encoder, head_obj, model_key, ds_key, head_type,
                                        seed, in_dim, EMBED_DIM, n_classes, best_acc)

                        row = {
                            "model": model_key, "dataset": ds_key,
                            "head": head_type, "seed": seed,
                            "best_acc": best_acc, "final_acc": final_acc,
                            "in_dim": in_dim, "embed_dim": EMBED_DIM,
                            "n_classes": n_classes,
                        }
                        all_results.append(row)

                        # Store preds for seed 0 (for inference comparison)
                        if seed == 0:
                            key = (model_key, ds_key, head_type)
                            all_preds[key] = preds

                        # Cleanup
                        del encoder, head_obj, embeddings
                        torch.cuda.empty_cache()

                    except Exception as e:
                        elapsed = time.time() - t0
                        print(f" ERROR: {e} ({elapsed:.1f}s)", flush=True)
                        traceback.print_exc()
                        row = {
                            "model": model_key, "dataset": ds_key,
                            "head": head_type, "seed": seed,
                            "best_acc": np.nan, "final_acc": np.nan,
                            "in_dim": in_dim, "embed_dim": EMBED_DIM,
                            "n_classes": n_classes,
                        }
                        all_results.append(row)

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_csv = TRAINING_DIR / "scaled_toy_all_models.csv"
    results_df.to_csv(str(results_csv), index=False)
    print(f"\nResults saved: {results_csv}", flush=True)

    # Print summary table
    print("\n" + "=" * 80, flush=True)
    print("  RESULTS SUMMARY (mean accuracy across seeds)", flush=True)
    print("=" * 80, flush=True)

    pivot = results_df.groupby(["model", "dataset", "head"])["final_acc"].mean().reset_index()
    for ds_key in ALL_DATASETS:
        ds_data = pivot[pivot["dataset"] == ds_key]
        if len(ds_data) == 0:
            continue
        print(f"\n  {ds_key}:", flush=True)
        pt = ds_data.pivot(index="model", columns="head", values="final_acc")
        print(pt.to_string(float_format=lambda x: f"{x:.4f}"), flush=True)

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 4: Inference comparison images
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 4: Inference Comparison Images", flush=True)
    print("=" * 80, flush=True)

    for model_key in ALL_MODELS:
        for ds_key in ALL_DATASETS:
            linear_key = (model_key, ds_key, "linear")
            hyp_key = (model_key, ds_key, "hyperbolic")
            if linear_key not in all_preds or hyp_key not in all_preds:
                continue

            # Load test labels
            if not features_exist(model_key, ds_key):
                continue
            test_data = np.load(str(get_feature_path(model_key, ds_key, "test")))
            y_test = test_data["labels"].astype(np.int64)

            try:
                generate_inference_comparison(
                    model_key, ds_key,
                    all_preds[linear_key],
                    all_preds[hyp_key],
                    y_test, n_show=20)
            except Exception as e:
                print(f"    ERROR generating inference image for {model_key}/{ds_key}: {e}", flush=True)

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 5: Correlations with geometric measures
    # ──────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("  PHASE 5: Correlations with Geometry", flush=True)
    print("=" * 80, flush=True)

    try:
        corr_result = compute_correlations(results_df)
        if corr_result is not None:
            summary_df, corr_df = corr_result
            summary_csv = TRAINING_DIR / "correlation_summary.csv"
            summary_df.to_csv(str(summary_csv), index=False)
            print(f"\n  Correlation summary saved: {summary_csv}", flush=True)
            print(f"\n  Summary (H improvement vs geometry):", flush=True)
            print(summary_df.to_string(index=False), flush=True)

            corr_csv = TRAINING_DIR / "correlation_coefficients.csv"
            corr_df.to_csv(str(corr_csv), index=False)
            print(f"\n  Correlation coefficients saved: {corr_csv}", flush=True)
            print(corr_df.to_string(index=False), flush=True)
        else:
            print("  No correlations computed.", flush=True)
    except Exception as e:
        print(f"  ERROR computing correlations: {e}", flush=True)
        traceback.print_exc()

    # ──────────────────────────────────────────────────────────────────────
    # DONE
    # ──────────────────────────────────────────────────────────────────────
    total_time = time.time() - overall_t0
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    print(f"\n{'='*80}", flush=True)
    print(f"  ALL DONE in {hours}h {minutes}m ({total_time:.0f}s)", flush=True)
    print(f"{'='*80}", flush=True)


if __name__ == "__main__":
    main()
