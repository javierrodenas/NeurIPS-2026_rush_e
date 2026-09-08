#!/usr/bin/env python3
"""expR65 (positive-control pass, R10): a real fine-tuned model with injected hierarchy, at ImageNet's class count.

ViT-B/16 (the census backbone, timm vit_base_patch16_224.augreg_in21k, patch embedding frozen) is fine-tuned on the
census's ImageNet training subset (the first 100 files per class in sorted order, 100k images: the images behind the
census centroids; the full 1.28M-image train set is not readable at training speed from the seek-bound HDD, so the
schedule is EPOCHS passes over this subset) with two objectives, in ONE process so both models see the same batches in
the same order (seed 0):
  (a) 'ce'   : cross-entropy over the 1000 classes;
  (b) 'hier' : the same plus a hierarchical cross-entropy over the WordNet 30-cut (a second linear head on the pooled
               embedding), i.e. the hierarchical-CE fallback of the brief (Sinha et al. 2024's regularizer is not available).
Images are decoded once with the model's eval transform (resize, center crop 224, the transform of the census extraction)
into a uint8 array in RAM; training augmentation is a random horizontal flip. After training, the pooled embedding of the
same 100 images per class gives the centroids; each set goes through the census of record (Haar x p99.9 x 200) and the
validated depth test (anisotropic star, K = 30, 10 star seeds), next to the frozen backbone read the same way.
Output: expR65_hier_finetune.csv (rows: frozen, ce, hier), expR65_train_log.csv, centroids in
$PLATONIC_ROOT/results/centroids/imagenet_ft/{obj}.npy.
"""
import os, sys, time, argparse, json
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "ICLR2027" / "tool"))
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(HERE.parents[1] / "rebuttal/results")))
IMAGENET_DIR = Path(os.environ.get("IMAGENET_TRAIN", "/media/HDD_4TB_1/javi/ILSVRC2012_img_train"))
CENT = ROOT / "results/centroids/imagenet_ft"; CENT.mkdir(parents=True, exist_ok=True)
DECODED = ROOT / "results/imagenet_train100_224_uint8.npy"
MODEL_ID = "vit_base_patch16_224.augreg_in21k"; N_PER_CLASS = 100; K = 30

def file_list():
    classes = sorted([d for d in os.listdir(IMAGENET_DIR) if (IMAGENET_DIR / d).is_dir()]); assert len(classes) == 1000
    files, labels = [], []
    for ci, c in enumerate(classes):
        fs = sorted(os.listdir(IMAGENET_DIR / c))[:N_PER_CLASS]; files += [IMAGENET_DIR / c / f for f in fs]; labels += [ci] * len(fs)
    return files, np.array(labels, dtype=np.int64)

def decode(files, n_workers=3):
    """Eval transform of the census extraction (timm data config: resize, center crop 224), stored as uint8 NHWC."""
    import timm, torch
    from PIL import Image
    from concurrent.futures import ProcessPoolExecutor
    cfg = timm.data.resolve_data_config(timm.create_model(MODEL_ID, pretrained=False).pretrained_cfg)
    size = cfg["input_size"][-1]; crop_pct = cfg.get("crop_pct", 0.9); resize = int(round(size / crop_pct))
    global _DEC
    def dec(p):
        im = Image.open(p).convert("RGB"); w, h = im.size; r = resize / min(w, h)
        im = im.resize((max(size, int(round(w * r))), max(size, int(round(h * r)))), Image.BICUBIC); w, h = im.size
        l, t = (w - size) // 2, (h - size) // 2; return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8)
    arr = np.lib.format.open_memmap(DECODED, mode="w+", dtype=np.uint8, shape=(len(files), size, size, 3)); t0 = time.time()
    with ProcessPoolExecutor(n_workers) as ex:
        for i, a in enumerate(ex.map(_decode_one, [(str(p), resize, size) for p in files], chunksize=64)):
            arr[i] = a
            if i % 10000 == 0: print(f"decoded {i}/{len(files)} ({time.time()-t0:.0f}s)")
    arr.flush(); del arr; print(f"decoded {len(files)} images -> {DECODED} ({time.time()-t0:.0f}s); mean/std {cfg['mean']} {cfg['std']}")
    return cfg

def _decode_one(args):
    from PIL import Image
    p, resize, size = args
    im = Image.open(p).convert("RGB"); w, h = im.size; r = resize / min(w, h)
    im = im.resize((max(size, int(round(w * r))), max(size, int(round(h * r)))), Image.BICUBIC); w, h = im.size
    l, t = (w - size) // 2, (h - size) // 2; return np.asarray(im.crop((l, t, l + size, t + size)), dtype=np.uint8)

def wn30():
    from sklearn.cluster import AgglomerativeClustering
    WN = np.load(ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy")
    return AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average").fit_predict(WN)

def main(A):
    import torch, timm, torch.nn as nn, torch.nn.functional as F
    torch.manual_seed(0); np.random.seed(0)
    files, labels = file_list()
    cfg = timm.data.resolve_data_config(timm.create_model(MODEL_ID, pretrained=False).pretrained_cfg)
    if not DECODED.exists(): decode(files, A.workers)
    X = np.load(DECODED, mmap_mode="r"); assert X.shape[0] == len(labels); X = np.ascontiguousarray(X)   # ~15 GB in RAM
    sup = wn30(); sup_t = torch.tensor(sup); N = len(labels); print(f"{N} images in RAM, {X.shape[1:]} uint8; K={K}")
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1); std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    devs = {"ce": torch.device("cuda:0"), "hier": torch.device("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")}
    models, heads, opts, scheds, scalers = {}, {}, {}, {}, {}
    steps_per_epoch = N // A.bs; total = steps_per_epoch * A.epochs; warm = 100
    for obj, dev in devs.items():
        torch.manual_seed(0); m = timm.create_model(MODEL_ID, pretrained=True, num_classes=0).to(dev)
        for p in m.patch_embed.parameters(): p.requires_grad = False                     # frozen patch embedding
        h = nn.ModuleDict({"cls": nn.Linear(m.num_features, 1000), "sup": nn.Linear(m.num_features, K)}).to(dev)
        torch.manual_seed(0); nn.init.trunc_normal_(h["cls"].weight, std=0.02); nn.init.zeros_(h["cls"].bias); nn.init.trunc_normal_(h["sup"].weight, std=0.02); nn.init.zeros_(h["sup"].bias)
        opt = torch.optim.AdamW([{"params": [p for p in m.parameters() if p.requires_grad], "lr": A.lr}, {"params": h.parameters(), "lr": A.lr_head}], weight_decay=0.05)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
        models[obj], heads[obj], opts[obj], scheds[obj], scalers[obj] = m, h, opt, sched, torch.cuda.amp.GradScaler()
    g = torch.Generator().manual_seed(0); log = []; t0 = time.time(); step = 0
    for ep in range(A.epochs):
        perm = torch.randperm(N, generator=g)
        for bi in range(steps_per_epoch):
            idx = perm[bi * A.bs:(bi + 1) * A.bs].numpy(); idx.sort()
            xb = torch.from_numpy(X[idx]).permute(0, 3, 1, 2).float().div_(255); xb = (xb - mean) / std
            flip = torch.rand(len(idx), generator=g) < 0.5; xb[flip] = xb[flip].flip(-1)
            yb = torch.from_numpy(labels[idx]); sb = sup_t[yb]
            rec = dict(step=step, epoch=ep)
            for obj, dev in devs.items():
                m, h, opt, sc = models[obj], heads[obj], opts[obj], scalers[obj]; m.train()
                x, y, s = xb.to(dev, non_blocking=True), yb.to(dev), sb.to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    z = m(x); lc = F.cross_entropy(h["cls"](z), y); ls = F.cross_entropy(h["sup"](z), s)
                    loss = lc + (ls if obj == "hier" else 0.0)
                opt.zero_grad(set_to_none=True); sc.scale(loss).backward(); sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); sc.step(opt); sc.update(); scheds[obj].step()
                rec[f"{obj}_loss_cls"] = float(lc); rec[f"{obj}_loss_sup"] = float(ls); rec[f"{obj}_acc"] = float((h["cls"](z.float()).argmax(1) == y).float().mean())
            step += 1
            if step % 50 == 0 or step == 1:
                log.append(rec); pd.DataFrame(log).to_csv(OUT / "expR65_train_log.csv", index=False)
                print(f"ep {ep} step {step}/{total} | ce: cls {rec['ce_loss_cls']:.3f} acc {rec['ce_acc']:.2f} | hier: cls {rec['hier_loss_cls']:.3f} sup {rec['hier_loss_sup']:.3f} acc {rec['hier_acc']:.2f} | {time.time()-t0:.0f}s")
    # centroids of the same 100 images per class, eval transform (no flip), pooled embedding
    import calibrated_delta as cd
    rows = []
    def read(C, obj):
        c = cd.census(C, 200, "haar", "p999"); d = cd.depth_test(C, sup, "aniso", 10)
        rows.append(dict(obj=obj, n=c["n"], d=c["d"], delta_999=c["delta"], delta_sd=c["delta_sd"], null_mean=c["null_mean"], null_sd=c["null_sd"], excess=c["excess"], r_above=c["r_above"], p_left=c["p_left"],
                         excessB_real=d["excessB_real"], excessB_star=d["excessB_star"], depth=d["depth_excess"], z_depth=d["z_depth"]))
        pd.DataFrame(rows).to_csv(OUT / "expR65_hier_finetune.csv", index=False)
        print(f"{obj:6s}: census exc {c['excess']:+.4f} r={c['r_above']} p={c['p_left']:.3f} | depth {d['depth_excess']:+.4f} z {d['z_depth']:+.2f}")
    for obj, dev in devs.items():
        m = models[obj].eval(); feats = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, N, 256):
                xb = torch.from_numpy(X[i:i + 256]).permute(0, 3, 1, 2).float().div_(255); xb = ((xb - mean) / std).to(dev); feats.append(m(xb).float().cpu().numpy())
        Fz = np.concatenate(feats); C = np.stack([Fz[labels == c].mean(0) for c in range(1000)]).astype(np.float32); np.save(CENT / f"{obj}.npy", C); read(C, obj)
        torch.save({"model": m.state_dict(), "heads": heads[obj].state_dict()}, CENT / f"{obj}_vitb16.pt")
    dz = np.load(ROOT / "results/practical_tasks_cache/i21k_b_imagenet_train.npz"); Xf = dz["features"].astype(np.float32); yf = dz["labels"]
    read(np.stack([Xf[yf == c].mean(0) for c in range(1000)]), "frozen")
    print("DONE expR65")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--lr-head", dest="lr_head", type=float, default=1e-3); ap.add_argument("--workers", type=int, default=3); main(ap.parse_args())
