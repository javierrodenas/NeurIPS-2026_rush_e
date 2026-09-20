# PARALLEL_STATUS — the parallel track (brief of 2026-09-20). The submission file `main_iclr2027_final.tex` is not touched.

Updated at every checkpoint. Newest entry first.

## 2026-09-20 19:20 — priority 1c added (sixth-review follow-up)

- **§1c (CPU, reuses expR74)**: `expR80_implanted_alignment.py` running in four shards at nice 10: from each backbone's decoupled cloud,
  each cluster's principal axis is rotated toward its hub direction by a fraction s of the angle (s = 0, 0.25, 0.5, 0.75, 1; hubs and
  within-cluster spectra unchanged, unit-checked), 5 seeds, depth test at every s. Decision rule (power >= 0.8 at s = 1 with false alarms
  <= 0.05 at s = 0) is computed by `--merge` into `expR80_decision.csv`; if met, the submission gains Table 9c, a third curve in
  Figure 4b and the sentence "certifies hub alignment with measured power"; otherwise the result stays in the parallel file.
- CPU is shared by expR66c (12 parts, nice 15), the ImageNet decode (6 workers), expR78 (delta step), expR79 and now expR80; the
  fine-tuning starts when the decode finishes.

## 2026-09-20 19:05 — priority 1b added (sixth review); replication features done

- **§1b (CPU)**: `expR79_synthetic_deep_poincare.py` running at nice 10: (a) a synthetic cloud with ViT-L's real ImageNet spectrum and
  a three-level implanted hierarchy (nested 2/6/30 cuts of the frame) at ViT-L's real within/between ratio, 5 seeds, plus a flat
  two-level control; (b) the WordNet Poincaré embeddings of Nickel & Kiela trained with gensim on the transitive closure of the tree
  spanning the 1000 ImageNet leaves (d = 10, 50); each through the census, the depth test (K = 30) and the decoupling control.
- **§2**: ResNet-34 features extracted for CIFAR-10/100 and MiniImageNet; the delta step (their estimator + the record instrument, 10
  batches of 1500) is running; CUB-200 unpacking, then extracted and added.
- **§1**: decode at 258k / 1.28M (19:00).

## 2026-09-20 18:35 — decode and replication extraction running

- **§1**: `expR76_prep_imagenet_memmap.py` running (6 workers, since 18:27; the decode reproduces the census cache features to cosine
  0.99995 on the first 100 images of class 0; ETA ~2 h for 1.28M images). The two fine-tuning runs start when it finishes (GPU 0: leaf
  CE; GPU 1: leaf CE + hierarchical CE), ~9–10 h each, concurrently. `expR77_positive_control_tests.py` written (census under the
  centered null, depth test on both frames, decoupling control; prints the verdict of the success criterion).
- **§2**: `expR78_khrulkov_replication.py` written. Their protocol (paper text, arXiv v2): ImageNet-pretrained CNN features (Table 1
  rows: Inception v3, ResNet34, VGG19; ResNet34 row = 0.26 / 0.25 / 0.25 / 0.21 for CIFAR10 / CIFAR100 / CUB / MiniImageNet),
  Euclidean distances, exact delta on a sampled batch by the min-max product, delta_rel = 2 delta / diam, averaged over trials. Ours:
  torchvision resnet34 (IMAGENET1K_V1, no substitution), 512-d avgpool features at 224 px, class-balanced batches of 1500 points,
  10 trials; on each cloud their delta_rel and the record instrument (centered Haar null x 200, p99.9): excess and rank. Feature
  extraction running on GPU 0 for CIFAR10/100 (train sets, local) and MiniImageNet (local raw copy of the 100 classes, 600 per class);
  CUB-200-2011 downloading from the deepai mirror (Caltech's server refuses; 1.19 GB), then extracted.
- **Submission**: `expR66c` 12 parts at nice 15; the closing rebuild follows when it finishes (ETA moved to ~03:00 by the contention).

## 2026-09-20 18:30 — track opened

- **Submission**: frozen at `1c97bda` (interim) pending the closing rebuild with `expR66c` (joint sensitivity under the centered
  record, 12 CPU parts running since 17:45, reniced to 15 so the parallel track's data pipeline gets CPU; ETA ~02:00). That
  closing was ordered by the brief of 2026-09-20 (author's decisions) and is the only pending edit to the submission.
- **§1 positive control (priority 1)**: setup fixed as in the brief. Pipeline: `expR76_prep_imagenet_memmap.py` decodes the full
  ImageNet-1k train set (1.28M images, eval transform of the census backbone, directory order) into a 193 GB uint8 memmap on
  HDD_4TB_2 (515 GB free); `expR76_hier_finetune_full.py` fine-tunes ViT-B/16 (timm `vit_base_patch16_224.augreg_in21k`, the
  census checkpoint) with (a) leaf CE and (b) leaf CE + hierarchical CE at the WordNet 30/6/2 cuts weighted 1/2/4 (30-cut 1,
  6-cut 2, 2-cut 4; leaf 1), AdamW, lr 1e-5 encoder / 1e-3 heads, effective batch 256 (micro-batch 64 x 4 on an 11 GB 2080 Ti),
  mixed precision, seed 0, 5 epochs; identical batches for (a) and (b) (the sampler is a pure function of the seed); run (a) on
  GPU 0 and (b) on GPU 1 concurrently. Extraction of the census subset (first 100 files per class, CLS pooled embedding) into the
  cache format; tests by `expR77_positive_control_tests.py` (census excess under the centered Haar null x 200, depth test with the
  matched star at K = 30 and on the balanced frame, decoupling control of expR74). Success criterion as in the brief.
- **§2 replication of Khrulkov et al. (2020) (priority 2)**: not started; datasets to be located or fetched (CUB-200, MiniImageNet).
- **§3**: not started, as ordered.
- **Integration**: `main_iclr2027_rebuttal.tex` to be built from the frozen file plus one paragraph in §5.3 and one in §5.1 when
  results exist; sweep checks to be added for the new numbers.
