#!/usr/bin/env python3
"""REVIEWER_CHECKLIST_third.md: the questions of the third review answered from the main text alone, with the ICLR margin line
numbers of the sentences that answer them, read from the compiled PDF (pdftotext -layout keeps the margin numbers).
Usage: python make_reviewer_checklist_third.py <path/to/main_iclr2027.pdf>   (run from the repo root)"""
import re, subprocess, sys, json
pdf = sys.argv[1]
txt = subprocess.run(["pdftotext", "-layout", "-l", "9", pdf, "-"], capture_output=True, text=True).stdout
lines = txt.split("\n"); numbered = []; last = None
for l in lines:
    m = re.match(r"\s*(\d{3})\s+(.*)", l)
    if m: last = int(m.group(1)); numbered.append((last, m.group(2)))
    elif l.strip():
        if re.fullmatch(r"\s*\d{3}\s*", l): last = int(l.strip()); continue
        numbered.append((last, l.strip()))
def find(phrase):
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace("ﬁ", "fi").replace("ﬂ", "fl"))
    target = norm(phrase)
    for i in range(len(numbered)):
        if target in norm("".join(t for _, t in numbered[i:i + 3])): return numbered[i][0]
    return None
M = json.load(open("rebuttal/results/final_pass_memo.json")); F = json.load(open("rebuttal/results/phaseE_fills.json")); G = json.load(open("rebuttal/results/final_pass_island_gap.json"))
a1 = M["a1"]; a2 = M["a2"]; a4 = M["a4"]; a5 = M["a5"]; a6 = M["a6"]
Q = [("Q1. The matched star's hubs are Gaussian; a star with the real hubs' spectrum could pass the depth test too.",
      [("Abstract", "not that the hubs form a hierarchy", "The abstract states that the four certified backbones are the same under a star whose hubs carry the real hubs' spectrum."),
       ("Section 4, 'A spectrum-matched star confirms the four'", "under both matched stars", f"Hubs drawn as a Haar resample of the real hubs (exact hub spectrum, ten star seeds): the certified set is unchanged, z from {F['HAAR_Z_HI']} to {F['HAAR_Z_LO']}; on the implanted clouds the new star raises {F['FA_HAAR']} false alarms at s = 0 and detects {F['POWER_HAAR']} at s = 1; read as a rank over the star seeds (resolution 1/11) the verdict does not separate certified from uncertified backbones while z does. Table 6 has both stars per backbone.")]),
     ("Q2. The certified ViTs were trained on the hierarchical IN-21k label set; the depth may be the label hierarchy.",
      [("Section 4, same paragraph", "Supervision on leaf labels alone can produce this depth and need not", f"Two ViT-B/16 supervised on ImageNet-1k leaf labels only: the augreg recipe is certified under both stars (z {a2['vit_b_in1k']['z_gauss']:+.2f}/{a2['vit_b_in1k']['z_haar']:+.2f}) and DeiT-B is not (z {a2['deit_b']['z_gauss']:+.2f}/{a2['deit_b']['z_haar']:+.2f}), so the IN-21k label hierarchy is not what makes the census ViTs deep."),
       ("Section 5, 'WordNet alignment follows supervision and recipe'", "Leaf-label supervision does not guarantee the alignment", f"The augreg IN-1k ViT-B is aligned with WordNet like the IN-21k ViTs (rho {a2['vit_b_in1k']['rho_wn']:+.2f}) while DeiT-B is nearly unaligned ({a2['deit_b']['rho_wn']:+.2f}) yet recovers the CIFAR-100 superclasses at ARI {a2['deit_b']['ari_max']:.2f}: alignment is recipe-dependent even among leaf-supervised ViTs (Table 10)."),
       ("Limitations (iv)", "two leaf-label ViTs show that supervision can produce depth", "The limitation records that the objective-geometry link is correlational and that leaf-label supervision can produce depth and alignment without guaranteeing either.")]),
     ("Q3. The corollary correlates the raw delta with the gains; the paper's own point is that the raw reading is confounded.",
      [("Section 6, 'The calibrated reading predicts the zero-cost gain'", "The calibrated reading predicts the zero-cost gain", "Recomputed on the record excess with Fisher-z CI95 and the family control: the nearest-centroid prediction survives on all four hierarchical datasets, the few-shot prediction on three (not ImageNet), and the depth verdict predicts no gain anywhere. Table 13 gives raw and calibrated correlations side by side; Figure 11 is drawn on the excess.")]),
     ("Q4. MERU's embeddings may sit where the hyperboloid is essentially flat, so the control may not test curvature.",
      [("Section 4, 'Imposing the geometry does not create detected depth'", "near-flat regime of the hyperboloid", f"Stated in the text: MERU's embeddings sit in the near-flat regime (learned c = {a4['curv']:.2f}, radius x sqrt(c) at the median {a4['radius_median'][0]:.3f}-{a4['radius_median'][1]:.3f}, Lorentz-to-Euclidean distance ratio {F['MERU_RATIO']}), so the control tests the training objective, not a curved geometry; the lead-in now says 'does not create detected depth'."),
       ("Abstract", "lives in the near-flat regime", "The abstract says so in one clause.")]),
     ("Q5. 'Budget-stable' rested on a supremum-statistic sweep whose ImageNet magnitudes drifted by 0.02.",
      [("Section 3, 'Two choices changed after pre-specification'", "The excess is budget-stable under the record", f"The sweep was rerun under the record (Haar null, p99.9, 200 replicates) at every budget: the ImageNet excess drifts by at most {a5['imagenet']['max_drift_over_sd']:.2f} of its own spread from 10^5 upward and never changes sign (Table 4, with the superseded supremum rows next to it).")]),
     ("Q6. Effect sizes: the excesses are small in absolute terms; 'small' is not a size.",
      [("Section 4, 'The premise does not survive calibration'", "removes between", f"'Small' is replaced by the fraction of the null reading: at the sample level the genuine cells remove {F['SL_FRAC_LO']} to {F['SL_FRAC_HI']} per cent of the null, the larger figure for DINOv2-G on CIFAR-100 images."),
       ("Section 6, 'The raw reading cannot select a curvature'", "per cent of the null reading, many times", f"On ImageNet centroids the excess removes {F['IN_FRAC_LO']} to {F['IN_FRAC_HI']} per cent of the null reading; the normalized column is in Table 3(b), Table 5 and Table 11."),
       ("Abstract", "reaches at most 40", f"The abstract bounds the surviving sample-level excess at 40% of the null (file: {100*abs(a6['sample_frac'][0]):.0f}%).")]),
     ("Q7. 'The island vanishes' / 'closes entirely' overstates: a gap remains under most configurations.",
      [("Section 5, 'The categorical island is an artifact, and a moderate gap remains'", "The categorical island is an artifact, and a moderate gap remains", f"Under the selected configuration the DINOv2 family agrees with the block at {F['ARI_BIG']} against {F['ARI_BLOCK']}; under the other admissible configurations about a third of the within-block agreement is missing (median over configurations and measures {G['median_missing_admissible_imagenet']:.2f}, Table 9); the triplet gap closes under cosine-average only. 'Vanishes' and 'closes entirely' are gone, Figure 5's caption included."),
       ("Abstract", "a gap of about a third remains", "The abstract says a gap of about a third remains once the cut is controlled.")]),
     ("Q8. The text census: Table 13 and Table 14 disagreed on GPT-2 S/M, and the prose followed the older table.",
      [("Section 5, 'In text, clustered structure depends on recipe, scale and probe'", "GPT-2 S is genuine at the margin under the record", "One census, the record (Table 11): GPT-2 S genuine at p = 0.020 and flipping under the supremum and under cosine, M not genuine under the record, L and XL genuine under every construction, Pythia at every scale. The 3-replicate original extraction table is removed and the template table's caption is rewritten from the record."),
       ("Section 1, 'The raw reading is confounded'", "GPT-2 M sits at its matched null", "The introduction's example now names GPT-2 M, the size that sits at its null under the record.")]),
     ("Q9. Minor: 'pre-registered', the ordering trees < hyperbolic < spherical, the repeated sentence in A.3, the old illustration, bold in Table 1, the bridges and the number density.",
      [("Section 3", "Two choices changed after pre-specification", "'Pre-registered' is 'pre-specified' throughout (no dated public record exists)."),
       ("Section 2, 'Background'", "for hyperbolic regions of radius four and above", "The ordering is stated for hyperbolic regions of radius >= 4, where it holds in Table 2."),
       ("Table 1 caption", "bold: less clustered than the null", "Bold is restored on the sign-positive cells."),
       ("Sections 3-5, last paragraphs", "With the instrument in place, we read the premise where it is read", "Three bridges remain (end of Sections 3, 4 and 5, distinct wording); every other paragraph ends on its claim. The repeated sentence of A.3 and the old illustration (former Figure 12) are gone; the appendix is fourteen tables, one per question, numbered in citation order.")])]
out = ["# Reviewer checklist — third review (final pass)", "", "Each question of the third review, answered from the main text alone; line numbers are the ICLR margin numbers of the compiled PDF (`ICLR2027/main_iclr2027_final.pdf`).", ""]
missing = []
for q, items in Q:
    out.append(f"## {q}"); out.append("")
    for where, phrase, answer in items:
        ln = find(phrase)
        if ln is None: missing.append(phrase)
        out.append(f"- **{where}** (line {ln if ln else '?'}): {answer}")
    out.append("")
FOURTH = """
## Fourth review (2026-09-18): the four rebuttal items, answered in `main_iclr2027_final.pdf`

- **R4.1 Bootstrap of the transfer sets under the record.** The centroid bootstrap of CIFAR-100 and DTD was rerun under the census of record (Haar null, 99.9th-percentile statistic, 200 replicates, 30 resamples at the census image budget; `expR73_transfer_bootstrap_record`), and the bootstrap panel of the robustness table (Table 2b) now reads ImageNet, CIFAR-100 and DTD from expR59 and expR73; the old expR32 rows are gone.
- **R4.2 Class count.** The class-count control is stated for what it shows (Section 5.2 and Table 2d): the excess shrinks with the number of classes for coherent and random subsets alike, so magnitudes are not comparable across class counts; the verdict is what carries across datasets.
- **R4.3 The depth test's frame.** Section 5.3 describes the balanced frame and the K sweep (Table 8a, two-decimal z): under the balanced frame ViT-T joins and ViT-S and DINOv2-L leave, cutting at ten or sixty superclasses changes the set again, and only ViT-B and ViT-L are certified under every frame. The CIFAR-100 readings are outside the validated regime (regime paragraph). Two new limitations: the spectrum null conditions on the second moments, to which a hierarchy contributes, so the excess is conservative (also stated in Section 3.3); and the depth test is Euclidean, so for the angular DINOv2 tree it may be conservative.
- **R4.4 The recommendation.** Section 6 says what Tables 13d–e support: the raw reading predicts the readout gain as well as the calibrated one, the best simple policy is cosine everywhere, and the instrument's role is to certify whether there is structure to justify a non-Euclidean readout at all. No sentence claims that the excess predicts gains better than the raw reading.
- Editorial: 15 text models everywhere (OLMo-7B was not extracted, said once in Section 4; "OLMo-1B" replaces "two OLMo sizes"); Table 11c's supremum columns are read against the Haar null; the abstract and Section 5.1 say the premise does not survive calibration on the two datasets tested; a `\FloatBarrier` closes every appendix subsection.
"""
out.append(FOURTH)   # fourth review (2026-09-18): appended verbatim, no line numbers (they refer to the final version)
FIFTH = """
## Fifth review (2026-09-18): corrections made, and the rebuttal list (no action now)

- **Corrections in `main_iclr2027_final.pdf`.** Confounds stated as a reference level (S1, S3.2: high dimension lowers the reading, anisotropy raises it, a sampled supremum grows with the budget); Khrulkov's rule recomputed on the supremum Gaussian band of Table 3 and said to be calibrated with the supremum; S5.2 counts (49 of 72, 30 of 36 on the datasets with 47 classes or more; FMNIST genuine in 7 of 12) and the flat-dataset sentence; S4: with 10 classes there are 210 quadruples, so the 99.9th percentile is the supremum there; S6: the objective rule beats cosine (+0.41 against +0.28 pp) but was chosen on the same cells, cosine is best among non-circular policies, the depth verdict predicts no hyperbolic gain, and a paragraph on what the calibration is for; abstract, S1 and S5.1 with the new wording (4 of 12 certified, two of which survive every choice of frame; the raw reading is not evidence and the calibrated reading is weak and model-dependent); Figure 2b on one dataset (ViT-B images vs DINO-B centroids, CIFAR-100); Proposition 1(b) for fixed n; S5.3 on the implant's two levels against the WordNet hierarchy above the hubs and the decoupling control (expR74); Sala et al. (2018) and Gu et al. (2019) in S2; Alper & Averbuch-Elor cited in S5.4; AI Use Statement in the three items of the form.
- **Runs.** `expR74_decoupling.py` (real hubs kept, each cluster's offsets rotated by an independent Haar rotation, 10 seeds; z next to the real z in Table 8) and `expR75_census_centered_haar.py` (Haar null with Q orthogonal to 1; the vision census rerun under the record; cells whose verdict changes reported in Table 2).
- **Rebuttal list, no action now:** (i) multiplicity correction for z across backbones and frames; (ii) the n-dependence of Proposition 1(b); (iii) what the text census measures when the probe decides the verdict; (iv) the prompt used for causal LMs; (v) the diameter as a single-pair maximum, with a 99th-percentile-distance normalization to be reported as a check.
"""
out.append(FIFTH)
SIXTH = """
## Sixth review (2026-09-20): corrections made, and the rebuttal list (no action now)

- **Corrections in `main_iclr2027_final.pdf`.** Depth stated as inconclusive wherever it was a finding (abstract, S1, contribution 2, thesis, S5.3 lead-ins, limitation iii): the test has no power at ImageNet's noise level, so whether the superclasses are arranged hierarchically is left open; what it certifies in 4 of 12 backbones is the alignment of each cluster with its hub. "Certifies clustered structure" softened to "certifies structure beyond the second moments; clustering is its most plausible reading, supported by the superclass recovery of S5.4". S6: calibration buys interpretation (a low delta is not hierarchy), not prediction. Abstract at 250 words or fewer with the cell defined in its own sentence.
- **Rebuttal list, no action now:** (vi) the p floor at 1/201 and its interaction with BH; (vii) z from 10 star seeds; (viii) normalization by a high distance percentile instead of the diameter; (ix) multiplicity over frames.
- **Parallel track, priority 1b (CPU):** a synthetic cloud with ViT-L's real ImageNet spectrum and a deep implanted hierarchy at the real within/between ratio, and the WordNet Poincare embeddings of Nickel & Kiela, both through the depth test (`expR79_synthetic_deep_poincare.py`).
"""
out.append(SIXTH)
SEVENTH = """
## Seventh and eighth reviews, closing pass (2026-09-21/22): power per backbone, the radial control and the trained control

- **Power of the depth test per backbone (expR81, Table 9(e)).** The three-level synthetic hierarchy rebuilt with each backbone's real ImageNet spectrum and within/between ratio, 5 seeds (DINOv2: 20): power 0.80 or more in ViT-L, DINOv2-L, DINOv2-G, CLIP-L and SigLIP-B, and 0.45 or less in ViT-T/S/B, DINO-B, DINOv2-S/B and CLIP-B. The power follows the backbone, not its ratio or family (three supervised ViTs at ratios 1.5-2.0 miss the implant; DINOv2-L and G detect it at 3.9). The headline is scoped accordingly in the seven places (abstract, S1, contribution 2, S5.3, Figure 4 and Table 8 captions, thesis).
- **The alignment is not the spread of feature norms (expR82, Table 8(e)).** Removing the radial component of every offset keeps z <= -2 in the four certified backbones under both stars (-2.26 to -3.99 under the anisotropic star); full L2 normalization keeps it in ViT-B and ViT-L only.
- **The trained positive control (expR76/expR77, S5.3 and Table 8(d)).** ViT-B/16 fine-tuned on ImageNet-1k with leaf CE and with leaf CE + hierarchical CE (WordNet 30/6/2), one seed, identical batches: the hierarchical model is certified on both frames and keeps firing once decoupled (10 of 10), the leaf-CE and frozen models do not (0 of 10) on the frame of record; on the balanced frame the frozen checkpoint fires in 7 of 10. The pre-set criterion (leaf-CE and frozen not certified intact) is not met because ViT-B is certified by alignment; the discriminating comparison is the decoupled one. A second seed of both runs is in progress for the rebuttal file.
"""
out.append(SEVENTH)
open("ICLR2027/REVIEWER_CHECKLIST_third.md", "w").write("\n".join(out)); print("\n".join(out)); print("missing anchors:", missing)
