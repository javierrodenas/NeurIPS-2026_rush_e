#!/usr/bin/env python3
"""Final pass (author brief after the full read), items 1-4 and the §5 sentence of item 5: regime-scoped depth test
(numbers from expR56/expR55b), Figure 5 = fig_depth_main.pdf, interventions figure to the appendix, §3 null/aggregation
wording, duplicated sentence removed. Run from the repo root; idempotent guards via exact anchors."""
import re, json
import pandas as pd
R = 'rebuttal/results/'
dv = pd.read_csv(R + 'expR56_depth_variants.csv'); d = pd.read_csv(R + 'expR55b_depth_power_leafframe.csv')
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
an = dv[(dv.dataset == 'imagenet') & (dv.K == 30) & (dv.variant == 'aniso')].set_index('model').z_depth
hits = an[an <= -2]; names = [NM[m] for m in hits.index]; assert set(names) == {"ViT-S", "ViT-B", "ViT-L", "DINOv2-L"}, names
zmin, zmax = hits.max(), hits.min(); assert (an >= 2).sum() == 0
c = dv[(dv.dataset == 'cifar100') & (dv.K == 20) & (dv.variant == 'aniso')]; n_c = int((c.z_depth <= -2).sum())
h = d[d.level != 'star']; st = d[d.level == 'star']
pw03 = (h[(h.n == 1000) & (h.ratio <= 0.3)].z <= -2).mean(); pw06 = (h[(h.n == 1000) & (h.ratio == 0.6)].z <= -2).mean()
fa1000 = ((st[st.n == 1000].z <= -2).mean(), (st[st.n == 1000].z >= 2).mean()); fa100 = (st[(st.n == 100) & (st.K >= 12)].z <= -2).mean(); fa100_6 = (st[(st.n == 100) & (st.K == 6)].z <= -2).mean()
assert pw03 == 1.0 and abs(pw06 - 0.90) < 0.011 and fa1000 == (0.0, 0.0) and abs(fa100 - 1/6) < 0.01 and fa100_6 == 0.0
iso = dv[(dv.dataset == 'cifar100') & (dv.K == 20) & (dv.variant == 'iso')]; iso_max = iso.z_depth.max(); iso_pos = int((iso.z_depth >= 2).sum())
top = pd.read_csv(R + 'expR55_depth_power.csv'); top_fa = (top[top.level == 'star'].z.abs() >= 2).mean()
print(f"ImageNet K=30 aniso z<=-2: {names} z {zmin:+.1f}..{zmax:+.1f}; C100 {n_c}/12; power n=1000: {pw03:.2f}/{pw06:.2f}; fa n=1000 {fa1000}; fa n=100 K>=12 {fa100:.2f}, K=6 {fa100_6:.2f}")
json.dump(dict(names=names, z_min=float(zmin), z_max=float(zmax), n_c=n_c, pw03=float(pw03), pw06=float(pw06), fa1000=list(map(float, fa1000)), fa100_kge12=float(fa100), fa100_k6=float(fa100_6)),
          open(R + 'phaseB_depth_regime.json', 'w'), indent=1)

p = 'ICLR2027/iclr2027/main_iclr2027.tex'; T = open(p).read()
def rep(old, new, count=1):
    global T
    assert T.count(old) == count, f"{T.count(old)} matches: {old[:90]!r}"; T = T.replace(old, new)

# (1) abstract, intro, contributions
rep("A depth test against matched stars does not certify hierarchy above the labelled clusters.",
    f"A matched-star depth test, validated for class sets of ImageNet's size, finds hierarchy above the WordNet superclasses in {len(names)} of 12 ImageNet backbones; on smaller class sets it is not validated.")
rep(" A matched-star test does not certify hierarchy above the labelled clusters.",
    f" A matched-star test, validated at ImageNet's class count, finds hierarchy above the WordNet superclasses in {len(names)} of 12 backbones (ViT-S/B/L and DINOv2-L) and is not validated on the smaller class sets.")
rep("A star calibration shows that the excess certifies clustering, not depth.",
    f"A star calibration shows that the excess certifies clustering, not depth; a matched-star test, validated at $n{{=}}1000$, certifies depth above the superclasses in {len(names)} of 12 ImageNet backbones.")
# (2)-(4) and the stale p99.9-variant clause in §3
rep("on ImageNet every sign holds at every budget but magnitudes drift by up to $0.02$; a 99.9th-percentile variant preserves every sign (Appendix~\\ref{app:robust}).",
    "on ImageNet every sign holds at every budget but magnitudes drift by up to $0.02$ (Appendix~\\ref{app:robust}).")
rep("the dimension confound. A low raw value is not evidence.", "the dimension confound.")
rep("(Gaussian coefficients recombined with the real singular values; alternative constructions in Appendix~\\ref{app:robust})",
    "(the Haar construction: random orthogonal coefficients recombined with the real singular values, so the sample spectrum is exact; the Gaussian-coefficient construction is the alternative reported in Appendix~\\ref{app:robust})")
rep("the same supremum over the same quadruple budget ($5{\\times}10^5$ per seed) is evaluated on every null replicate",
    "the same statistic, the 99.9th percentile of the same $5{\\times}10^5$ sampled defects per seed, is evaluated on every null replicate")
# (1) depth paragraph + Figure 5
i = T.find("\\paragraph{Depth above the labelled clusters is not certified.}"); j = T.find("\n\n", i); assert i > 0
depth = (r"\paragraph{Depth above the labelled clusters: certified on ImageNet, not on CIFAR-100.} A matched star tests hierarchy \emph{above} the labelled superclass clusters: Gaussian hubs with the real hub radius and, within each cluster, a Haar sample with the cluster's own covariance (10 star seeds), compared with the real centroids under the hub-randomizing null (Table~\ref{tab:b34-depthvariants}). The test certifies hierarchy above its frame only, since the hub null keeps every frame cluster intact (Appendix~\ref{app:star}). With the frame at the finest clusters, our pre-set bar (power $\ge0.8$ for two- and three-level hierarchies at noise ratios $\le0.3$ for both $n$, false alarms $\le5$\% each way) fails at $n{=}100$: pure stars with $K\ge12$ leaf clusters, fewer than about ten points per cluster, are declared more hierarchical ($z\le-2$) in "
         + f"{100*fa100:.0f}\\% of runs. At $n{{=}}1000$ the test meets the bar: power {pw03:.2f} at ratios $\\le0.3$ and {pw06:.2f} at $0.6$, no false alarm in either direction (Figure~\\ref{{fig:depth}}b; Table~\\ref{{tab:b35-power}}); we certify it within that regime only, a scoping adopted after the sweep. On ImageNet ($n{{=}}1000$, WordNet $K{{=}}30$) the real cloud is more hierarchical than its matched star beyond the noise in {len(names)} of 12 backbones, ViT-S/B/L and DINOv2-L ($z$ from ${zmin:+.1f}$ to ${zmax:+.1f}$), and not in the other eight; none is less hierarchical than its star (Figure~\\ref{{fig:depth}}a). CIFAR-100 ($n{{=}}100$, $K{{=}}20$) lies in the unvalidated regime and its readings stay unvalidated (Appendix~\\ref{{app:star}}). An earlier isotropic star, blind to each cluster's covariance, gave $z$ up to ${iso_max:+.1f}$ on CIFAR-100 and {100*top_fa:.0f}\\% false alarms on synthetic stars; those readings are withdrawn. The hierarchically fine-tuned checkpoints of Appendix~\\ref{{app:interventions}} were not stored, so that positive control could not be run. % expR56_depth_variants.csv, expR55b_depth_power_leafframe.csv, expR55_depth_power.csv")
fig5 = ("\n\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/fig_depth_main.pdf}\n\\caption{\\textbf{Depth above the labelled clusters: a validated matched-star test certifies it in "
        + f"{len(names)} of 12 ImageNet backbones.}} (a)~Excess B under the hub-randomizing null for the real ImageNet centroids (filled, family colors) and for their matched stars (hollow circle: isotropic, withdrawn; square: anisotropic), WordNet $K{{=}}30$; the number under each backbone is the $z$ of real minus anisotropic star (red: $z\\le-2$, more hierarchical above the frame). (b)~Power ($z\\le-2$) on synthetic hierarchies at $n{{=}}1000$ with the frame at the leaf clusters (solid, one line per $K$) or at the top level (dotted, zero by construction); dashed: false alarms on pure stars at $n{{=}}1000$ (gray) and $n{{=}}100$ (black), the unvalidated regime. % expR56_depth_variants.csv, expR55b_depth_power_leafframe.csv, expR55_depth_power.csv\n}}\n\\label{{fig:depth}}\n\\end{{figure}}")
T = T[:i] + depth + fig5 + T[j:]
# MERU / NC sentences; bridges paragraph folded into NC
rep("A backbone \\emph{trained} in hyperbolic space is no exception: MERU \\citep{desai2023meru} and its Euclidean CLIP twin show the same clustering excess, and MERU's Lorentz-metric $\\hat\\delta$ matches its Euclidean value (Appendix~\\ref{app:star}).",
    "A backbone \\emph{trained} in hyperbolic space is no exception: MERU \\citep{desai2023meru} and its Euclidean CLIP twin show the same clustering excess and MERU's Lorentz-metric $\\hat\\delta$ matches its Euclidean value; its depth reading used the withdrawn isotropic star and is not part of the certified result (Appendix~\\ref{app:star}).")
rep("and negative curvature concentrates on the bridges between clusters (Appendix~\\ref{app:orc}). And the same clustered arrangement",
    "and negative curvature concentrates on the bridges between clusters, which is why a high mean ORC coexists with beyond-null structure (Appendix~\\ref{app:orc}). And the same clustered arrangement")
rep("the depth test asks what neural collapse does not: whether there is hierarchy above the hubs.",
    "the depth test asks what neural collapse does not, whether there is hierarchy above the hubs, and answers yes for four ImageNet backbones, three of them supervised ViTs. % night/orc_bridges.csv")
rep("\\paragraph{Negative curvature concentrates on bridges.} High mean ORC coexists with beyond-null structure because negative curvature concentrates on the bridges between superclass clusters, not within them (Appendix~\\ref{app:orc}). % night/orc_bridges.csv\n\n", "")
# interventions figure -> appendix, one-line pointer
m = re.search(r"\\begin\{figure\}\[t\]\n  \\centering\n  \\includegraphics\[width=\\linewidth\]\{figures/fig3_causal\.pdf\}.*?\\end\{figure\}\n", T, re.S); assert m
figenv = m.group(0); T = T[:m.start()] + T[m.end():]
rep("Two interventions move it (Figure~\\ref{fig:causal}; raw $\\delta$, within architecture):", "Two interventions move it (Figure~\\ref{fig:causal}, Appendix~\\ref{app:interventions}; raw $\\delta$, within architecture):")
app_int = "\\FloatBarrier\n\\subsection{Training interventions}\n\\label{app:interventions}\n" + figenv.replace("\\begin{figure}[t]", "\\begin{figure}[H]") + "\n"
k = T.find("\\subsection{Text census"); assert k > 0
kb = T.rfind("\\FloatBarrier\n", 0, k); kb = kb if k - kb < 20 else k
T = T[:kb] + app_int + T[kb:]
# Limitation (i), §7, §5
rep("(i)~The spectrum excess certifies structure beyond second moments, star-like clustering included; the matched-star test certifies hierarchy above its frame only, and hierarchy within the labelled clusters is not tested (Appendix~\\ref{app:star}), so the hierarchy's reach rests on the class-subsampling control and its content on \\S\\ref{sec:content}.",
    "(i)~The spectrum excess certifies structure beyond second moments, star-like clustering included; the matched-star test certifies hierarchy above its frame only and is validated for $n\\approx1000$, not below about ten points per cluster, so depth is certified on ImageNet alone and hierarchy within the labelled clusters is not tested (Appendix~\\ref{app:star}); the hierarchy's reach rests on the class-subsampling control and its content on \\S\\ref{sec:content}.")
rep("recipe- and scale-dependent in text, always relative to a concept set.",
    f"recipe- and scale-dependent in text, always relative to a concept set; depth above the WordNet superclasses is certified in {len(names)} of 12 ImageNet backbones, where the depth test is validated.")
rep("The census exception is consistent with this: whatever structure DINOv2 has on ImageNet (genuine under the record, above the null under the supremum), it is the least aligned with WordNet's top level: the form-vs-content split.",
    "This is consistent with the census: DINOv2's ImageNet structure is genuine under the record and the least aligned with WordNet's top level, the form-vs-content split.")
# appendix: CIFAR-100 figure + readings by regime
i = T.find("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/fig_depth_test.pdf}"); j = T.find("\\end{figure}\n", i) + len("\\end{figure}\n"); assert i > 0
T = T[:i] + ("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.55\\linewidth]{figures/fig_depth_cifar100.pdf}\n\\caption{\\textbf{The depth test on CIFAR-100 ($K{=}20$, five points per cluster): the unvalidated regime.} Excess B under the hub-randomizing null for the real centroids (filled) and their matched stars (hollow circle: isotropic, withdrawn; square: anisotropic); the number under each backbone is the $z$ of real minus anisotropic star. In this regime pure synthetic stars are declared more hierarchical in "
                + f"{100*fa100:.0f}\\% of runs (Table~\\ref{{tab:b35-power}}), so these readings are reported without certification. % expR56_depth_variants.csv, expR55b_depth_power_leafframe.csv\n}}\n\\label{{fig:depth-c100}}\n\\end{{figure}}\n") + T[j:]
i = T.find("\\paragraph{Real-data readings of the anisotropic test, not validated.}"); j = T.find("\n\n", i); assert i > 0
T = T[:i] + (f"\\paragraph{{Real-data readings by regime.}} ImageNet ($n{{=}}1000$, WordNet $K{{=}}30$) lies in the validated regime: the real cloud is more hierarchical than its anisotropic matched star beyond the noise ($z\\le-2$) in {len(names)} of 12 backbones ({', '.join(names)}), never less (Table~\\ref{{tab:b34-depthvariants}}, Figure~\\ref{{fig:depth}}a). CIFAR-100 ($n{{=}}100$, $K{{=}}20$) lies in the unvalidated regime, where pure stars are declared more hierarchical in {100*fa100:.0f}\\% of runs; its readings ({n_c} of 12 with $z\\le-2$, none with $z\\ge+2$) are reported without certification (Figure~\\ref{{fig:depth-c100}}). Validity domain of the test as run: $n\\approx1000$; at $n{{=}}100$ the false-alarm rate is {100*fa100_6:.0f}\\% with $K{{=}}6$ (17 points per cluster) and {100*fa100:.0f}\\% with $K\\ge12$, so the test is not validated below about ten points per cluster. The bar was fixed before the sweep; the restriction to the validated regime was adopted after it. % expR56_depth_variants.csv, expR55b_depth_power_leafframe.csv") + T[j:]
open(p, 'w').write(T); print("t4 applied")
