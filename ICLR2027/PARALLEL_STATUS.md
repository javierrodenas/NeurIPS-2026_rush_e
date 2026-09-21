# PARALLEL_STATUS — the parallel track (brief of 2026-09-20). The submission file `main_iclr2027_final.tex` is not touched.

Updated at every checkpoint. Newest entry first.

## 2026-09-21 02:40 — submission closed with expR66c; parallel track unchanged

- **Submission**: expR66c merged (joint z <= -2 in 39 of 72, genuine in >= 27 of 30 resamples in 42 of 72; record 44; ImageNet and
  CIFAR-100 18 of 24 under every reading) and the closing rebuild done: sweep 214/214, main text ends on page 9, 31 pages, 0 warnings.
  The rebuttal file was rebuilt on top of it (frozen text identical plus the expR80 paragraph and table).
- **§1**: epoch 1 of 5 running on both GPUs (epoch 0 took 4.25 h; ends ~18:00). **§2**: MiniImageNet batches 6 of 10 (summary
  after the 40th batch, ~03:40). **§1b**: flat control at seed 4 of 5, then the Poincaré embeddings.

## 2026-09-21 00:05 — priority 1c: decision rule NOT met; nothing enters the submission

- **§1c result** (`expR80_implanted_alignment.csv`, 12 backbones x 5 seeds x 5 strengths, merged at 00:02): detection rate at
  z <= -2 by strength s = 0 / 0.25 / 0.5 / 0.75 / 1: 0.017 / 0.233 / 0.500 / 0.550 / 0.600. False alarms at s = 0: 1 of 60 (0.017,
  within the 0.05 bar). Power at s = 1: 36 of 60 (0.60, below the 0.8 bar). Rule not met, so the submission file is untouched: no
  Table 9c, no third curve in Figure 4b, no "measured power" wording; the §5.3 certification stays as in `92f3c81`.
- Per backbone at full strength: detected in every seed for ViT-T, ViT-S, ViT-B, ViT-L and CLIP-B (ViT-B/L already at s = 0.25–0.5);
  DINO-B 4 of 5, SigLIP-B 3 of 5, CLIP-L and DINOv2-g 2 of 5, DINOv2-S/B/L 0 of 5. The implant that the supervised ViTs and CLIP-B
  reveal at half strength is invisible to the DINOv2 family even at full strength: the power of the test for alignment is
  backbone-dependent, and the rebuttal paragraph says so with the numbers.
- Written into `main_iclr2027_rebuttal.tex` (§5.3 paragraph "Implanted alignment on the real clouds." after the "left open"
  paragraph, plus the per-backbone table in the parallel-track appendix subsection) by `phaseE_rebuttal.py`; compiled by
  `rebuttal_rebuild.sh` to `ICLR2027/main_iclr2027_rebuttal.pdf`; sweep `rebuttal_checks` re-derives the counts from the CSVs.
- **§1**: both fine-tuning runs at 82 img/s, leaf accuracy 0.91 at step 2000 of 5004 of epoch 0 (hierarchical losses near zero);
  ~5 h per epoch, seed 0 of both runs expected around Tuesday 22 morning.
- **§2**: CIFAR-10 mean δ_rel 0.275 (published 0.26) and CIFAR-100 0.261 (published 0.25), both within 0.03; CUB running.
- **§1b**: four deep synthetic seeds done (z = −1.80 at seed 0, not firing at K = 30; decoupled z −4.8); flat control and Poincaré next.

## 2026-09-20 20:55 — decode done, fine-tuning launched; rule-met branch tested end to end

- **§1**: the decode finished at 20:42 (1,281,167 images, 134.8 min; memmap size equals the expected 192,851,506,176 bytes). Both
  fine-tuning runs launched at 20:50 (`--obj ce --gpu 0`, `--obj hier --gpu 1`, seed 0, nohup, logs `expR76_ft_{ce,hier}_seed0.log`).
  A first launch at 20:46 crashed at step 0 in the chunk-pool sampler: `gather` located each index by a sorted search over the pool's
  chunk starts, but the pool holds chunks in the epoch's random order; fixed to a direct lookup by chunk start (unit-tested), relaunched.
  Identical batches for (a) and (b) are preserved (the plan is a function of seed and epoch only). Throughput to be read at step 100.
- **§1c**: the rule-met code path was exercised end to end on an isolated copy of the results directory in the scratchpad
  (`PLATONIC_RESULTS=.../results_test` with synthetic expR80 fixtures that never touched `rebuttal/results`): Table 9c per backbone and
  pooled, the dotted "implanted alignment" curve in Figure 4b, the §5.3 sentence, the caption, limitation (iii) and the abstract
  (249 words) all rendered; the sweep passed every check except the known expR66c one; main text still ends on page 9 (the three
  statements move to page 10, with the references). The real build was then restored from the real results directory.
- **§1b / §2**: unchanged since 19:50 (expR79 on its first synthetic seed; expR78 on its second batch).

## 2026-09-20 19:50 — priority 1c integration wired (brief of the evening); §2 delta step confirmed running

- **§1c**: the conditional integration is coded and will run unattended when `expR80 --merge` writes `expR80_decision.csv`.
  Rule met (power >= 0.8 at s = 1, false alarms <= 0.05 at s = 0): `gen_appendix_final.py` adds Table 9c (detection rate against s, per
  backbone and pooled), `make_figs_final.py` adds the dotted "implanted alignment" curve to Figure 4b, `phaseE_submission.py` rewrites
  the §5.3 sentence with the author's wording ("certifies the alignment of each cluster with its hub, with measured power: implanted
  alignment is detected in x of y runs at full strength and in none at zero"), the Figure 4 caption, limitation (iii) ("measured power
  for alignment and none for hierarchy at this noise level") and the abstract phrase ("plus a depth test whose power is measured for
  the structure it certifies"; the same sentence loses five words to keep the 250-word cap, listed in the CHANGELOG for the author);
  the sweep gains a check that re-verifies the rule from the decision file and every wording. Rule not met: nothing enters the
  submission; `phaseE_rebuttal.py` writes the paragraph and table with the numbers into `main_iclr2027_rebuttal.tex`.
  Progress at 19:45: 18 of 300 tests (4 shards, 160–280 s per test on the loaded machine), ETA ~00:30. Early readings: ViT-L fires at
  s = 0.5 (z = −2.9), CLIP-B at s = 1 (z = −4.2), DINOv2-B not at s = 0.75 (z = −1.0).
- **§1b**: `phaseE_rebuttal.py` gains the expR79 paragraph and table (fires or not, per cloud, with the decoupling control); expR79
  still on its first synthetic seed (each seed = census + depth test + 10 decoupling tests).
- **§2**: the delta step is running since 19:26 (one process; a duplicate launched at 19:42 was stopped). First trial, CIFAR-10: their
  estimator on our extraction gives 0.297 against the published 0.26; the record instrument gives excess −0.002 (rank 155 of 200,
  p = 0.23) on the same 1500-point batch. One trial takes ~15 min on the loaded machine, so the 40 trials end around 05:30.
- **§1**: decode at 668k / 1.28M (19:40), ~66 min left; the two fine-tuning runs start when it finishes.
- Submission: expR66c still running (12 parts); closing rebuild afterwards.

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
