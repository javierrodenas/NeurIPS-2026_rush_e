# PARALLEL_STATUS — the parallel track (brief of 2026-09-20). The submission file `main_iclr2027_final.tex` is not touched.

Updated at every checkpoint. Newest entry first.

## 2026-09-22 01:10 — expR81 complete, DINOv2 at 20 seeds; every parallel-track run finished

- Final power for the deep three-level hierarchy at each backbone's own spectrum and ratio (K = 30, z <= -2): ViT-T 0.00, ViT-S 0.00,
  ViT-B 0.00, ViT-L 0.80 (5 seeds each); DINO-B 0.00 (5); DINOv2-S 0.00, DINOv2-B 0.45, DINOv2-L 0.95, DINOv2-G 1.00 (20 seeds
  each); CLIP-B 0.20, CLIP-L 1.00, SigLIP-B 1.00 (5). By family: supervised ViTs 0.20, DINO/DINOv2 0.57, contrastive 0.73; no family
  reaches 0.8. Files: `expR81_deep_per_backbone.csv`, `_summary.csv`, `_families.csv`.
- For the 23 September decision: the control has power (>= 0.8) in ViT-L, DINOv2-L, DINOv2-G, CLIP-L and SigLIP-B, and none or
  little in ViT-T/S/B, DINO-B, DINOv2-S/B and CLIP-B. The scoping by family in the submission does not match; the author chooses the
  replacement wording (a per-backbone list, or "where the control has power" with the list in §5.3 and the limitation).
- Nothing else is running. The rebuttal file holds the positive control; the submission is at `26bc28b` plus this status.

## 2026-09-21 20:35 — §1 positive control: criterion NOT met (the controls are certified too); the hierarchical signature is there

- `expR77_positive_control.csv` (centroids of the census subset, centered Haar null x 200, matched anisotropic star, 10 star seeds,
  decoupling 10 seeds), seed 0:
  frozen ViT-B/16: excess −0.016 (r 200, p 0.005); z −3.62 (WordNet-30) / −2.82 (balanced); decoupled −1.40 (0 of 10) / −2.10 (7 of 10).
  leaf CE: excess −0.024; z −2.45 / −2.09; decoupled −0.72 (0 of 10) / −1.38 (0 of 10).
  leaf CE + hierarchical CE: excess −0.029; z −2.97 / −3.57; decoupled −2.38 (10 of 10) / −3.51 (10 of 10).
- Verdict against the pre-set criterion ("the hierarchical model is certified under both frames and still fires with the offsets
  decoupled, while the leaf-CE model and the frozen checkpoint do not"): NOT MET. The first half holds in full (certified on both
  frames, fires in every decoupled seed); the second half fails because the leaf-CE model and the frozen checkpoint are certified
  (they are ViT-B, one of the four certified backbones of the paper). What the run does show: after decoupling, only the hierarchical
  model keeps firing on the WordNet frame (10 of 10, against 0 of 10 for leaf CE and for the frozen checkpoint), i.e. the injected
  hierarchy lives in the hubs and survives orientation randomization exactly as the synthetic deep hierarchy does; on the balanced
  frame the frozen checkpoint also fires in 7 of 10 decoupled seeds, so the separation is clean on the frame of record only.
- Per the brief: the paragraph and table go to `main_iclr2027_rebuttal.tex` only (built, sweep checks pass); no edit to the submission
  until the author decides. A second seed of both runs (another ~22 h of GPU) is worth it only if the author wants to pursue it.

## 2026-09-21 19:00 — expR81 first pass complete (12 backbones x 5 seeds; DINOv2 with 7-11 seeds so far, extension running)

- Power for the deep three-level hierarchy at each backbone's own spectrum and within/between ratio (K = 30, z <= -2):
  ViT-T 0.0, ViT-S 0.0, ViT-B 0.0, ViT-L 0.8 | DINO-B 0.0 | DINOv2-S 0.0 (11 seeds), DINOv2-B 0.44 (9), DINOv2-L 0.89 (9),
  DINOv2-G 1.0 (7) | CLIP-B 0.2, CLIP-L 1.0, SigLIP-B 1.0. Pooled by family: supervised ViTs 0.20, DINO/DINOv2 0.46, contrastive
  0.73; no family reaches the 0.8 of the seventh-review rule. Mean intact z: ViT-T −0.48, ViT-S −1.04, ViT-B −1.74, ViT-L −2.18,
  DINO-B −1.21, DINOv2-S −0.99, DINOv2-B −1.91, DINOv2-L −2.70, DINOv2-G −3.04, CLIP-B −1.69, CLIP-L −2.83, SigLIP-B −2.49.
  Decoupled clouds (10 seeds per cloud) fire far more often than the intact ones everywhere except DINOv2-S (0.01) and DINO-B (0.28):
  the deeper-once-decoupled effect of expR79 is general.
- Reading: the control's power does not follow the within/between ratio or the family. It is not established at ratios 1.3–2.0 for
  ViT-T/S/B and CLIP-B, and it is high for DINOv2-L/G at 3.9. The wording adopted on 21 September ("covers the supervised and
  contrastive backbones … DINOv2 beyond the noise level at which the test was validated") is contradicted; the author decides the
  replacement on 23 September with the 20-seed DINOv2 extension (8 shards at nice 19, ~half done).

## 2026-09-21 18:40 — fine-tuning done (both runs, 5 epochs); expR77 tests launched

- **§1**: both runs finished at 18:35 (5 epochs each, identical batches; ~22 h wall-clock under the shared CPU); the census-subset
  centroids of both models are extracted to `practical_tasks_cache/vitb_ft_{ce,hier}_seed0_imagenet_train.npz` (308 MB each).
  `expR77_positive_control_tests.py` running since 18:38 on frozen / leaf-CE / hierarchical-CE: census under the centered Haar null
  (200 replicates), depth test on the WordNet-30 frame and the balanced frame (10 star seeds), decoupling control (10 seeds) on both;
  ~1 h per model under the current load. The verdict against the pre-set criterion goes to the author first, before any edit.

## 2026-09-21 17:00 (b) — expR81 interim power per backbone (deep three-level hierarchy at each backbone's own spectrum and ratio)

- With 5 seeds unless noted: ViT-T 0.0, ViT-S 0.0, ViT-B 0.0, ViT-L 0.8, DINO-B 0.0, DINOv2-S 0.0 (7), DINOv2-B 0.43 (7),
  DINOv2-L 1.0 (7), DINOv2-G 1.0 (2), CLIP-B 0.2, CLIP-L 1.0, SigLIP-B 1.0 (1). The control's power is not a matter of family: three
  of the four supervised ViTs miss the deep hierarchy at their own spectra and ratios (1.5–2.0) while DINOv2-L and DINOv2-G detect it
  at 3.9. The current scoping ("covers the supervised and contrastive backbones, DINOv2 beyond the noise level") will not stand; the
  author decides on 23 September with the full 20-seed DINOv2 extension (running at low priority).

## 2026-09-21 17:00 — expR82 complete (12 backbones); Table 8 panel (e) in the submission

- Radial component removed (both stars): certified four keep z <= -2 (ViT-S −2.26/−2.08, ViT-B −3.60/−3.57, ViT-L −3.99/−3.47,
  DINOv2-L −2.59/−2.28); among the others only ViT-T crosses (−2.06/−2.07); DINO-B −1.61/−1.89, DINOv2-B −1.18, DINOv2-G −1.75, CLIP
  and SigLIP −0.9 to −1.5, DINOv2-S +0.6. L2-normalized: ViT-B −2.45/−2.52 and ViT-L −2.00/−2.23 keep it; no other backbone reaches −2.
  Decoupled transformed clouds: certified fraction 0.0 everywhere except ViT-T (0.3 under deradial) and ViT-L (0.1 under L2).
- The paper keeps "alignment" with the author's two sentences (§5.3, limitation iii) and Table 8(e). expR81 first pass and the
  DINOv2 extension keep running; fine-tuning in epoch 4 of 5.

## 2026-09-21 16:30 — expR82: the alignment is not the spread of feature norms (rule met by the radial-component control)

- **§1e**, four certified backbones read (the other eight still running): with the radial component of every offset removed, all
  four keep z <= -2 under both stars (aniso: ViT-S −2.26, ViT-B −3.60, ViT-L −3.99, DINOv2-L −2.59; Haar-hub star: −2.08, −3.57,
  −3.47, −2.28); under full L2 normalization only ViT-B (−2.45/−2.52) and ViT-L (−2.00/−2.23) keep it. Decoupled transformed clouds
  do not fire where read (ViT-B −0.57 and −1.40, ViT-S −1.16 and −1.25, ViT-L −1.41 under L2, DINOv2-L +0.89 under L2). The brief's
  rule holds through (b): "alignment" stays; the author was told first and gave the wording, now in §5.3 (new paragraph "The
  alignment is not the spread of feature norms.") and limitation (iii). Table 8 gains panel (e) when the twelve backbones are merged.
- **Monitors**: the git pull with autostash at 16:15 detached the visible logs of the two fine-tunings and of expR81's CLIP/SigLIP
  shard (the processes keep writing to their CSVs); completion is now watched on the output files (fine-tuned caches in
  `practical_tasks_cache`, shard CSV rows). Fine-tuning at epoch 4 of 5, step ~2500 of 5004 at 16:15 (87 img/s).

## 2026-09-21 11:45 — eighth review: expR82 (radial control) and the DINOv2 extension of expR81 launched

- **§1e** `expR82_radial_control.py` (six shards, nice 10, since 11:44): the depth test of record on the 12 ImageNet clouds after
  (a) L2-normalizing every centroid and (b) removing the radial component of each offset (its projection onto the hub direction, hub
  kept); both matched stars (anisotropic Gaussian-hub star and Haar-resampled-hub star), 10 star seeds, and the decoupling control (10
  seeds) on each transformed cloud. Decision rule fixed by the brief and coded in `--merge` (`expR82_decision.csv`): if ViT-S, ViT-B,
  ViT-L and DINOv2-L keep z <= -2 under (a) or (b), "alignment of each cluster with its hub" stays and the sentence "the alignment
  survives L2 normalization and the removal of the radial component, so it is not the spread of feature norms" is added; otherwise
  "alignment" becomes "the radial spread of feature norms within each superclass" everywhere, the abstract drops the alignment claim
  and limitation (iii) says it. The author is told first, then the edit. ~80 min of tests per backbone; ETA tonight (22 processes on
  8 cores: expR81's remaining shards, its DINOv2 extension, expR82 and the two fine-tunings).
- **§1d extension**: expR81 seeds 5–19 for DINOv2-S/B/L/G (eight shards since 11:36), so the DINOv2 family gets 20 seeds; the other
  eight backbones stay at 5. The first-pass shards are still running (ViT-L done).
- **Text (eighth review)**: thesis reordered; the full scoped sentence once in §5.3, short form elsewhere; noise level defined in §5.3;
  ViT-L's decoupled reading (−1.61 from expR74) inside the flat control's range (−1.61 to −1.76 from expR79; z now two decimals
  throughout §5.3); DINO-B (2.1) and ViT-B (2.0) at the edge of the covered range in limitation (iii); the Khrulkov sentence in the
  abstract and §1 (two of four indistinguishable from a random cloud, from expR78's largest p > 0.05); "unregime" fixed (replacement
  order); the AI-use statement kept with the three form items (the form's exact wording is not available here; see TODO).

## 2026-09-21 10:15 — expR81: ViT-L seeds done (they reproduce expR79); first rows of the other backbones

- **ViT-L (5 seeds, = expR79's deep clouds)**: intact z −1.80, −2.45, −2.38, −2.08, −2.20 (power 0.8); decoupled z mean −4.44 over
  50 runs (all certified). Decomposition: the star spread shrinks to 0.69 of its intact value (0.0032 → 0.0022) and the excess below
  the star nearly doubles (−0.019 → −0.036; the cloud's own excess B −0.020 → −0.040, the star's −0.0015 → −0.0044). Both sides move,
  so the builder writes "The deeper reading comes from both sides: decoupling shrinks the star spread and deepens the excess below
  the star." into §5.3 (recorded in `final_dec_obs.json`); why the excess itself grows under decoupling is not explained by the runs.
- **First rows elsewhere** (one seed each unless stated): ViT-B z −1.91 (not firing; decoupled −5.1), ViT-T −0.58 and −0.56 (two
  seeds), CLIP-B −1.41 (decoupled −5.2), DINOv2-S −0.49, DINOv2-L −2.10 (fires; decoupled −2.9). If this holds, the power for a
  deep hierarchy at each backbone's own spectrum and ratio is not a matter of family: ViT-T/B and CLIP-B miss it at ratios 1.3–2.0
  while DINOv2-L detects it at 3.9. The scoping adopted this morning ("supervised and contrastive covered, DINOv2 not tested") will
  have to be revisited on 23 September as the brief foresees. All ten shards keep running (ETA for the full 60 clouds: evening).

## 2026-09-21 09:15 — seventh review: priority 1d launched (deadline 23 September)

- **§1d** `expR81_deep_per_backbone.py`: the deep synthetic hierarchy of expR79 (three levels over the 30 hubs) with each of the 12
  backbones' own ImageNet spectrum and within/between ratio (expR64b ratio_real: ViT-T 1.55, ViT-S 1.93, ViT-B 1.96, ViT-L 1.88,
  DINO-B 2.12, DINOv2-S 3.00, DINOv2-B 3.71, DINOv2-L 3.88, DINOv2-G 3.92, CLIP-B 1.33, CLIP-L 1.49, SigLIP-B 1.54), 5 seeds each,
  depth test at K = 30 (10 star seeds) and the decoupling control (10 seeds) on each; every row keeps the excess B of the cloud and of
  its star and the star spread, so the deeper-once-decoupled reading can be decomposed. Ten shards at nice 10 since 09:12: the five
  ViT-L seeds as single-seed shards (they reproduce expR79's deep clouds and feed the §5.3 observation sentence first), then five
  two/three-backbone shards. ~11 depth tests per cloud; ETA for ViT-L ~1.5 h, for the rest ~8 h (the two fine-tunings share the CPU).
  `--merge` writes the per-backbone summary (power over the 5 seeds, decoupled power over 50 runs, B and star spread before/after) and
  the per-family power with the 0.8 rule of the brief.
- **Submission (text now)**: headline scoped ("no hierarchy above the superclasses is found in the backbones whose noise level the
  control covers") in the abstract (twice), §1, contribution 2, §5.3 lead-in, Figure 4 and Table 8 captions and the thesis; limitation
  (iii) names the ratio the power was measured at (1.9) and the DINOv2 range (3.0–3.9, from expR64b: DINOv2-S 3.0 is included, the
  brief said 3.7–3.9 for B/L/G); §5.3: the decoupled flat control fires in 8 of 50 runs (expR79), and the deeper-once-decoupled reading
  is stated as an open observation until the ViT-L rows of expR81 decompose it (the builder then writes the supported explanation:
  star spread, excess, or both). Minors: Figure 4b label on the real-spread curve, the unproved covariance extension removed from the
  proof of Proposition 1(b), Figure 1 caption "read alike, all low", the statistic of Table 2(e) stated.
- **§1**: fine-tuning in epoch 2 of 5 (ETA ~19:00 for both runs).

## 2026-09-21 (morning) — priorities 1b and 2 integrated into the submission, 1c as a limitation (author's brief)

- The submission file now carries the deep-hierarchy paragraph and Table 9(d) (1b), the published-reading paragraph and Table 3(b)
  (2), the implanted-alignment sentence in limitation (iii) and Table 9(c) (1c), with the author's wording in the abstract, §1,
  contribution 2, §5.3, thesis and limitations; details in CHANGELOG §37. `main_iclr2027_rebuttal.tex` is again the frozen file plus
  nothing, until the trained positive control (expR77) exists.
- **§1**: both fine-tuning runs in epoch 2 of 5 (76 img/s); extraction and expR77 tests after epoch 4; verdict to the author first.

## 2026-09-21 03:15 — priority 1b closed: the deep synthetic hierarchy fires at K = 30 and survives decoupling; Poincaré does not fire

- **§1b result** (`expR79_synthetic_deep_poincare.csv`): synthetic clouds with ViT-L's real ImageNet spectrum (d = 1024) and a
  three-level implanted hierarchy (2/6/30) at ViT-L's real within/between ratio (1.88), 5 seeds: depth z = −1.80, −2.45, −2.38,
  −2.08, −2.20, so the test fires in 4 of 5 seeds at K = 30 (the frame of record); the flat two-level control (30 iid hubs, same
  spectrum and ratio) fires in 0 of 5 (z from −0.08 to −0.69). The decoupling control on the deep clouds (real hubs kept, offsets
  Haar-rotated, 10 seeds) fires in 5 of 5 with z ≈ −4.2 to −4.8, deeper than the intact cloud: a hierarchy carried by the hubs
  survives the orientation randomization, which is exactly what the real backbones do not do (none of the four certified backbones
  fires once decoupled). Census excess −0.008 to −0.010 (deep) and −0.020 (flat), all genuine.
- WordNet Poincaré embeddings (Nickel & Kiela, gensim, closure of the tree over the 1000 ImageNet leaves), read with Euclidean
  distances on the ball coordinates: census excess +0.083 (d = 10) and +0.035 (d = 50), above every replicate; depth z = +2.36 and
  +1.53 (the wrong direction), decoupled +0.32 and −2.22. They do not fire. The reading is Euclidean (limitation iv); the hyperbolic
  distances of the ball were not used, which is the follow-up if the author wants it.
- **Bearing on the submission (author's decision, per the brief)**: the submission says "the test has no power at this noise level"
  (§5.3, abstract, §7, limitation iii). expR79 shows that at ViT-L's noise level and spectrum the test does detect a three-level
  hierarchy above the 30 hubs in 4 of 5 seeds, and that this detection survives decoupling. That strengthens the reading of the real
  result (alignment, not hierarchy) but weakens the "no power" wording: the power of the test for a deep hierarchy at this noise level
  is 0.8 in this synthetic setting, not zero. Nothing was changed in the submission; the paragraph and table are in the rebuttal file.
- Written into `main_iclr2027_rebuttal.tex` (§5.3 paragraph "A deep hierarchy at the real noise level." and table `tab:r1b-deep`),
  compiled (33 pages, 0 warnings); sweep 216/216 with the priority-1b check passing.

## 2026-09-21 03:00 — priority 2 (Khrulkov replication) closed: all four raw values reproduced; calibrated, near the null on three

- **§2 result** (`expR78_khrulkov_replication_summary.csv`, ResNet-34 torchvision, 10 class-balanced batches of 1500):
  their estimator on our extraction, mean ± s.d. over batches, against their Table 1 row:
  CIFAR-10 0.275 ± 0.017 (0.26); CIFAR-100 0.261 ± 0.013 (0.25); CUB 0.270 ± 0.012 (0.25); MiniImageNet 0.203 ± 0.013 (0.21).
  Every value within 0.03 of the published one, so the pre-set condition for a claim holds.
- On the same clouds, the record instrument (centered Haar null, 200 replicates, 99.9th-percentile statistic): excess
  −0.004 / −0.009 / −0.004 / −0.023; mean rank among the replicates 173 / 198 / 177 / 200 of 200; largest left-tail p over the ten
  batches 0.33 / 0.03 / 0.25 / 0.005. The published readings are real, but their reference level is what decides: CIFAR-10 and CUB
  are indistinguishable from a spectrum-matched random cloud, CIFAR-100 is borderline, and only MiniImageNet is clearly below the null.
- Written into `main_iclr2027_rebuttal.tex` (§5.1 paragraph "A published reading, reproduced and calibrated." plus the four-row
  table) and compiled; sweep 215/215 with the priority-2 check passing.

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
