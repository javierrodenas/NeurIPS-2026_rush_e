#!/usr/bin/env python3
"""expR76 step 0 (parallel track, positive control): decode the FULL ImageNet-1k training set once, with the eval transform of the
census backbone (timm vit_base_patch16_224.augreg_in21k: resize the short side to 248 bicubic, center crop 224), into a uint8 NHWC
memmap in directory order (classes sorted, files sorted within each class; the first 100 files of each class are the census subset).
The memmap is read sequentially in chunks by the training script, which is what a seek-bound HDD can deliver.
Output: $PLATONIC_ROOT/results/imagenet_full_224_uint8.mm (N x 224 x 224 x 3) and imagenet_full_224_meta.json (N, class starts).
Usage: python expR76_prep_imagenet_memmap.py [--workers 6] [--limit N]   (idempotent: resumes from the last written block)."""
import os, sys, time, json, argparse
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from PIL import Image
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
IMAGENET_DIR = Path(os.environ.get("IMAGENET_TRAIN", "/media/HDD_4TB_1/javi/ILSVRC2012_img_train"))
MM = ROOT / "results/imagenet_full_224_uint8.mm"; META = ROOT / "results/imagenet_full_224_meta.json"; PROG = ROOT / "results/imagenet_full_224_progress.json"
SIZE, SCALE = 224, 248   # timm: input 224, crop_pct 0.9 -> scale size int(224/0.9) = 248, bicubic

def file_list():
    classes = sorted(d for d in os.listdir(IMAGENET_DIR) if (IMAGENET_DIR / d).is_dir()); assert len(classes) == 1000
    files, labels, starts = [], [], []
    for ci, c in enumerate(classes):
        fs = sorted(os.listdir(IMAGENET_DIR / c)); starts.append(len(files)); files += [str(IMAGENET_DIR / c / f) for f in fs]; labels += [ci] * len(fs)
    return classes, files, np.asarray(labels, dtype=np.int16), starts

_TF = None
def decode_one(path):
    """Exactly the census extraction's transform (timm create_transform for the model: torchvision Resize(248, bicubic) + CenterCrop(224)), before ToTensor/Normalize."""
    global _TF
    if _TF is None:
        from torchvision import transforms as T
        _TF = T.Compose([T.Resize(SCALE, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(SIZE)])
    try:
        return np.asarray(_TF(Image.open(path).convert("RGB")), dtype=np.uint8)
    except Exception as e:
        print("DECODE ERROR", path, e); return np.zeros((SIZE, SIZE, 3), np.uint8)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--limit", type=int, default=None); A = ap.parse_args()
    classes, files, labels, starts = file_list(); N = len(files) if A.limit is None else A.limit
    meta = dict(N=len(files), classes=classes, class_start=starts, labels_file=str(ROOT / "results/imagenet_full_224_labels.npy"), size=SIZE, scale=SCALE, order="classes sorted, files sorted; census subset = first 100 files of each class")
    if not META.exists():
        json.dump(meta, open(META, "w")); np.save(meta["labels_file"], labels)
    mm = np.memmap(MM, dtype=np.uint8, mode="r+" if MM.exists() else "w+", shape=(len(files), SIZE, SIZE, 3))
    done = json.load(open(PROG))["done"] if PROG.exists() else 0
    print(f"{len(files)} files, {len(classes)} classes; decoding from {done} to {N} with {A.workers} workers -> {MM}")
    t0 = time.time(); BLOCK = 2048
    with Pool(A.workers) as pool:
        for b0 in range(done, N, BLOCK):
            b1 = min(b0 + BLOCK, N); arrs = pool.map(decode_one, files[b0:b1], chunksize=32)
            mm[b0:b1] = np.stack(arrs); mm.flush(); json.dump({"done": b1}, open(PROG, "w"))
            if (b0 // BLOCK) % 25 == 0:
                el = time.time() - t0; rate = (b1 - done) / max(el, 1e-9); print(f"{b1}/{N} ({rate:.0f} img/s, {(N - b1) / max(rate, 1e-9) / 60:.0f} min left)")
    print("DONE", N, "images", f"{(time.time() - t0) / 60:.1f} min")
