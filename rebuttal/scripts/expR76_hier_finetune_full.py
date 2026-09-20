#!/usr/bin/env python3
"""expR76 (parallel track, positive control, priority 1): ViT-B/16 (census checkpoint, timm vit_base_patch16_224.augreg_in21k)
fine-tuned on the FULL ImageNet-1k training set (the memmap of expR76_prep_imagenet_memmap.py) with two objectives that share
everything else (seed, data order, schedule):
  (a) --obj ce   : cross-entropy on the 1000 leaf labels;
  (b) --obj hier : the same plus hierarchical cross-entropy at the WordNet 30-cut, the 6-cut and the 2-cut above it (nested
                   average-linkage cuts of the WordNet distance matrix, the 30-cut being the frame of record), weighted 1 / 2 / 4
                   (30 / 6 / 2: the higher the level, the higher the weight, so the hierarchy above the 30 hubs is rewarded), leaf CE
                   weight 1.
Schedule: 5 epochs, AdamW (lr 1e-5 encoder with weight decay 0.05, lr 1e-3 heads), effective batch 256 (micro-batch x accumulation),
mixed precision, 500 warm-up steps then constant, random horizontal flip, one seed. The sampler is a pure function of the seed: the
memmap is read in random chunk order (sequential 2048-image chunks, a pool of 16 chunks in RAM, batches drawn from the pool), so runs
(a) and (b) see identical batches. One epoch = N/256 steps. Checkpoints per epoch; at the end the CLS embedding of the census subset
(first 100 files per class, eval transform, no flip) is written in the cache format as vitb_ft_{obj}_seed{seed}_imagenet_train.npz.
Usage: python expR76_hier_finetune_full.py --obj hier --gpu 1 [--seed 0] [--epochs 5] [--micro 64] [--resume]"""
import os, sys, time, json, argparse, threading, queue
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(os.environ.get("PLATONIC_ROOT", "/media/HDD_4TB_2/javi/Platonic"))
OUT = Path(os.environ.get("PLATONIC_RESULTS", str(Path(__file__).resolve().parents[2] / "rebuttal/results")))
CACHE = ROOT / "results/practical_tasks_cache"; MM = ROOT / "results/imagenet_full_224_uint8.mm"; META = ROOT / "results/imagenet_full_224_meta.json"
MODEL_ID = "vit_base_patch16_224.augreg_in21k"; CHUNK, POOL, BATCH = 2048, 16, 256

def frames():
    """Nested average-linkage cuts of the WordNet distance matrix at 30 (the frame of record), 6 and 2 clusters."""
    from sklearn.cluster import AgglomerativeClustering
    f = OUT / "expR76_frames.json"
    if f.exists(): d = json.load(open(f)); return {int(k): np.asarray(v, dtype=np.int64) for k, v in d.items()}
    WN = np.load(ROOT / "results/features/imagenet/imagenet1k_wordnet_dist.npy")
    cuts = {K: AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average").fit_predict(WN).astype(np.int64) for K in (30, 6, 2)}
    for K in (6, 2):   # nested: every 30-cluster lies in one coarser cluster
        assert all(len(set(cuts[K][cuts[30] == c])) == 1 for c in range(30)), K
    json.dump({str(k): v.tolist() for k, v in cuts.items()}, open(f, "w")); return cuts

class ChunkSampler:
    """Pure function of (seed, epoch, step): chunk order per epoch, a pool of POOL chunks, BATCH indices per step drawn from the pool."""
    def __init__(self, N, seed):
        self.N, self.seed = N, seed; self.n_chunks = (N + CHUNK - 1) // CHUNK
    def epoch_plan(self, epoch):
        rng = np.random.RandomState(self.seed * 1000 + epoch)   # resumable: the plan of an epoch depends on (seed, epoch) only
        order = rng.permutation(self.n_chunks); steps = self.N // BATCH
        return order, steps, rng

def loader_thread(mm, order, q, stop):
    for c in order:
        if stop.is_set(): break
        lo, hi = c * CHUNK, min((c + 1) * CHUNK, mm.shape[0]); q.put((lo, np.asarray(mm[lo:hi])))
    q.put(None)

def gather(pool, gidx):
    """pool: list of (lo, array); gidx: global indices, each inside one pool chunk."""
    arr = {lo: a for lo, a in pool}   # the pool holds chunks in the epoch's random order, so locate each index by its chunk start, not by a sorted search
    return np.stack([arr[(g // CHUNK) * CHUNK][g % CHUNK] for g in gidx])

def main(A):
    import torch, torch.nn as nn, torch.nn.functional as F, timm
    torch.backends.cudnn.benchmark = True; dev = torch.device(f"cuda:{A.gpu}")
    meta = json.load(open(META)); N = meta["N"]; labels = np.load(meta["labels_file"]).astype(np.int64); starts = meta["class_start"]
    prog = json.load(open(ROOT / "results/imagenet_full_224_progress.json"))["done"]; assert prog >= N, f"memmap incomplete: {prog}/{N}"
    mm = np.memmap(MM, dtype=np.uint8, mode="r", shape=(N, 224, 224, 3))
    cuts = frames(); sup = {K: torch.tensor(cuts[K][labels]) for K in (30, 6, 2)}   # per-image superclass labels
    torch.manual_seed(A.seed); np.random.seed(A.seed)
    enc = timm.create_model(MODEL_ID, pretrained=True, num_classes=0).to(dev); D = enc.embed_dim
    heads = nn.ModuleDict({"leaf": nn.Linear(D, 1000)}); W = {"leaf": 1.0}
    if A.obj == "hier":
        for K, w in ((30, 1.0), (6, 2.0), (2, 4.0)): heads[f"k{K}"] = nn.Linear(D, K); W[f"k{K}"] = w
    heads.to(dev)
    cfg = timm.data.resolve_data_config(enc.pretrained_cfg); mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": A.lr_enc, "weight_decay": 0.05}, {"params": heads.parameters(), "lr": A.lr_head, "weight_decay": 0.0}])
    scaler = torch.cuda.amp.GradScaler(); tag = f"{A.obj}_seed{A.seed}"; ckdir = ROOT / "results/ft_full" / tag; ckdir.mkdir(parents=True, exist_ok=True)
    logf = OUT / f"expR76_train_{tag}.csv"; log = []; start_epoch = 0; gstep = 0
    if A.resume and (ckdir / "last.pt").exists():
        ck = torch.load(ckdir / "last.pt", map_location=dev); enc.load_state_dict(ck["enc"]); heads.load_state_dict(ck["heads"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, gstep = ck["epoch"] + 1, ck["gstep"]; log = pd.read_csv(logf).to_dict("records") if logf.exists() else []; print(f"resumed after epoch {ck['epoch']} (gstep {gstep})")
    sampler = ChunkSampler(N, A.seed); accum = BATCH // A.micro; base = {"leaf": A.lr_enc}
    def lr_scale(step): return min(1.0, (step + 1) / 500.0)
    def to_input(x_uint8, flip):
        x = torch.from_numpy(x_uint8).to(dev, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0)
        if flip is not None: x[flip] = x[flip].flip(-1)
        return (x - mean) / std
    for epoch in range(start_epoch, A.epochs):
        order, steps, rng = sampler.epoch_plan(epoch)
        q = queue.Queue(maxsize=4); stop = threading.Event(); th = threading.Thread(target=loader_thread, args=(mm, order, q, stop), daemon=True); th.start()
        pool = [q.get() for _ in range(POOL)]; nxt = q.get(); enc.train(); heads.train(); t0 = time.time(); seen = 0   # the pool is filled before the first step
        for step in range(steps):
            if step > 0 and step % (CHUNK // BATCH) == 0 and nxt is not None:   # bring in the next chunk, drop the oldest
                pool.append(nxt); pool.pop(0); nxt = q.get()
            P = np.concatenate([np.arange(lo, lo + len(a)) for lo, a in pool]); gidx = rng.choice(P, BATCH, replace=False); flips = rng.rand(BATCH) < 0.5
            xs = gather(pool, gidx)
            for g in opt.param_groups: g["lr"] = (A.lr_enc if g["weight_decay"] > 0 else A.lr_head) * lr_scale(gstep)
            opt.zero_grad(set_to_none=True); tot = {k: 0.0 for k in heads}; acc = 0
            for mb in range(accum):
                sl = slice(mb * A.micro, (mb + 1) * A.micro); x = to_input(xs[sl], torch.from_numpy(flips[sl]).to(dev)); yi = gidx[sl]
                with torch.autocast("cuda", dtype=torch.float16):
                    z = enc(x); loss = 0.0
                    for k, h in heads.items():
                        y = torch.from_numpy(labels[yi]).to(dev) if k == "leaf" else sup[int(k[1:])][yi].to(dev)
                        logits = h(z.float()); lk = F.cross_entropy(logits, y); loss = loss + W[k] * lk; tot[k] += lk.item() / accum
                        if k == "leaf": acc += (logits.argmax(1) == y).float().sum().item()
                scaler.scale(loss / accum).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(heads.parameters()), 1.0); scaler.step(opt); scaler.update(); gstep += 1; seen += BATCH
            if step % 50 == 0 or step == steps - 1:
                rate = seen / (time.time() - t0); row = dict(epoch=epoch, step=step, gstep=gstep, img_per_s=round(rate, 1), acc_leaf=acc / BATCH, **{f"loss_{k}": round(v, 4) for k, v in tot.items()}, lr_enc=opt.param_groups[0]["lr"], t=time.time())
                log.append(row); pd.DataFrame(log).to_csv(logf, index=False)
                print(f"{tag} ep {epoch} step {step}/{steps} {rate:.0f} img/s acc {acc / BATCH:.3f} " + " ".join(f"{k} {v:.3f}" for k, v in tot.items()) + f" ({(steps - step) * BATCH / max(rate, 1e-9) / 60:.0f} min left)")
        stop.set()
        torch.save(dict(enc=enc.state_dict(), heads=heads.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=epoch, gstep=gstep), ckdir / "last.pt")
        torch.save(dict(enc=enc.state_dict(), heads=heads.state_dict(), epoch=epoch), ckdir / f"epoch{epoch}.pt"); print(f"checkpoint epoch {epoch} saved")
    # ---- extraction of the census subset: first 100 files per class, eval transform, CLS pooled embedding
    enc.eval(); feats = []; idx = np.concatenate([np.arange(s, s + 100) for s in starts]); ylab = labels[idx]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(idx), 256):
            ii = idx[i:i + 256]; x = to_input(np.asarray(mm[ii]), None); feats.append(enc(x).float().cpu().numpy())
    feats = np.concatenate(feats).astype(np.float32); out = CACHE / f"vitb_ft_{tag}_imagenet_train.npz"
    np.savez(out, features=feats, labels=ylab.astype(np.int64)); print("cache written", out, feats.shape)
    print("DONE", tag)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--obj", choices=["ce", "hier"], required=True); ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5); ap.add_argument("--micro", type=int, default=64); ap.add_argument("--lr_enc", type=float, default=1e-5); ap.add_argument("--lr_head", type=float, default=1e-3); ap.add_argument("--resume", action="store_true")
    main(ap.parse_args())
