#!/usr/bin/env python3
"""Phase B, B3: decide the depth-test branch from expR55b (rule fixed a priori) and write the paragraph,
figure and abstract/intro sentences accordingly. Run from the repo root."""
import re, sys, json
import pandas as pd
R = 'rebuttal/results/'
d = pd.read_csv(R + 'expR55b_depth_power_leafframe.csv'); h = d[d.level != 'star']; st = d[d.level == 'star']
dv = pd.read_csv(R + 'expR56_depth_variants.csv')
# decision rule: power >= 0.8 for 2- and 3-level hierarchies at ratio <= 0.3 for both n; star false alarms <= 5% each direction
pw = {(n, lv): (h[(h.n == n) & (h.level == lv) & (h.ratio <= 0.3)].z <= -2).mean() for n in (100, 1000) for lv in ('hier2', 'hier3')}
fa_neg = (st.z <= -2).mean(); fa_pos = (st.z >= 2).mean()
validated = all(v >= 0.8 for v in pw.values()) and fa_neg <= 0.05 and fa_pos <= 0.05
inf = {(n, lv): int(((d.n == n) & (d.level == lv)).sum()) for n in (100, 1000) for lv in ('hier2', 'hier3')}
an_in = dv[(dv.dataset == 'imagenet') & (dv.K == 30) & (dv.variant == 'aniso')]; an_c = dv[(dv.dataset == 'cifar100') & (dv.K == 20) & (dv.variant == 'aniso')]
iso_c = dv[(dv.dataset == 'cifar100') & (dv.K == 20) & (dv.variant == 'iso')]
n_in, n_c = int((an_in.z_depth <= -2).sum()), int((an_c.z_depth <= -2).sum()); pos_any = int((an_in.z_depth >= 2).sum() + (an_c.z_depth >= 2).sum())
iso_max = iso_c.z_depth.max(); iso_pos = int((iso_c.z_depth >= 2).sum())
top = pd.read_csv(R + 'expR55_depth_power.csv'); top_fa = (top[top.level == 'star'].z.abs() >= 2).mean(); top_pw = (top[top.level != 'star'].z <= -2).mean()
pwtxt = ", ".join(f"{pw[(n,lv)]:.2f} ({lv.replace('hier','')}-level, $n{{=}}{n}$)" for n in (100, 1000) for lv in ('hier2', 'hier3'))
print("DECISION:", "MAIN (validated)" if validated else "APPENDIX (not validated)", pw, f"fa- {fa_neg:.3f} fa+ {fa_pos:.3f}", "real:", n_in, n_c, pos_any, "iso max z", round(iso_max, 1), iso_pos, "top-frame power", round(top_pw, 2), "top fa", round(top_fa, 2))
json.dump(dict(validated=bool(validated), power=dict((f"{n}_{lv}", float(v)) for (n, lv), v in pw.items()), fa_neg=float(fa_neg), fa_pos=float(fa_pos), n_in=n_in, n_c=n_c, pos_any=pos_any, iso_max=float(iso_max), iso_pos=iso_pos, top_pw=float(top_pw), top_fa=float(top_fa)), open('rebuttal/results/phaseB_depth_decision.json', 'w'), indent=1)
if '--dry' in sys.argv: sys.exit(0)

p = 'iclr2027/iclr2027/main_iclr2027.tex'; T = open(p).read()
def rep(old, new):
    global T
    assert T.count(old) == 1, f"{T.count(old)} matches: {old[:80]!r}"; T = T.replace(old, new)
common = (r"We test hierarchy \emph{above} the labelled superclass clusters with a matched star: Gaussian hubs with the real hub radius and, within each cluster, a Haar sample with the cluster's own covariance (10 star seeds), compared with the real centroids under the hub-randomizing null (Table~\ref{tab:b34-depthvariants}). The test certifies hierarchy above its frame only: the hub null keeps every frame cluster intact, so hierarchy below the frame is invisible by construction, and with the frame at the top level its power on synthetic hierarchies is zero (Appendix~\ref{app:star}). ")
withdraw = (f"An earlier isotropic star, whose Gaussian clouds ignore each cluster's covariance, produced the opposite verdict on CIFAR-100 ($z$ up to ${iso_max:+.1f}$, {iso_pos} of 12 backbones less hierarchical than their star) and false alarms on {100*top_fa:.0f}\\% of pure synthetic stars; those readings are withdrawn. The hierarchically fine-tuned checkpoints of Figure~\\ref{{fig:causal}} were not stored, so that positive control could not be run. % expR56_depth_variants.csv, expR55b_depth_power_leafframe.csv, expR55_depth_power.csv")
if validated:
    lead = r"\paragraph{Depth above the labelled clusters is detected in a minority of backbones.} "
    body = common + (f"With the frame at the finest clusters the test meets our pre-set bar on synthetic hierarchies: power {pwtxt} at noise ratios $\\le0.3$, with false alarms of {100*fa_neg:.1f}\\% ($z\\le-2$) and {100*fa_pos:.1f}\\% ($z\\ge+2$) on pure stars (Figure~\\ref{{fig:depth}}c; Table~\\ref{{tab:b35-power}}). On the real backbones it detects hierarchy above the superclass hubs in {n_in} of 12 on ImageNet (WordNet-30 frame) and {n_c} of 12 on CIFAR-100 (Figure~\\ref{{fig:depth}}a,b), and in no backbone is the real cloud less hierarchical than its star. ") + withdraw
    abstract_sent = f" A depth test against matched stars, validated on synthetic hierarchies, detects hierarchy above the labelled clusters in a minority of backbones ({n_in} of 12 on ImageNet) and never less than a star."
    intro_sent = f" A validated matched-star test detects hierarchy above the labelled clusters in {n_in} of 12 backbones on ImageNet and {n_c} of 12 on CIFAR-100."
    figcap = r"\textbf{Depth above the labelled clusters: detected in a minority of backbones by a validated matched-star test.}"
    fig_main = True
else:
    fa100 = (st[(st.n == 100) & (st.K >= 12)].z <= -2).mean(); fa1000 = (st[st.n == 1000].z <= -2).mean(); fa100_6 = (st[(st.n == 100) & (st.K == 6)].z <= -2).mean()
    pw_min03 = min(pw.values()); pw1000_min = (h[h.n == 1000].groupby(['level', 'ratio']).z.apply(lambda z: (z <= -2).mean())).min()
    lead = r"\paragraph{Depth above the labelled clusters is not certified.} "
    body = common + (f"With the frame at the finest clusters the test fails our pre-set validation bar (power $\\ge0.8$ for two- and three-level hierarchies at noise ratios $\\le0.3$ for both $n$, and false alarms $\\le5$\\% in each direction): power is {pw_min03:.2f} in every such configuration, and no pure star is ever declared less hierarchical than its match, but with $n{{=}}100$ centroids and $K\\ge12$ leaf clusters pure stars are declared more hierarchical ($z\\le-2$) in {100*fa100:.0f}\\% of runs ({100*fa_neg:.1f}\\% pooled over all stars; {100*fa1000:.0f}\\% at $n{{=}}1000$; Table~\\ref{{tab:b35-power}}). Depth above the labelled clusters is therefore not certified in this paper; the anisotropic real-data readings are reported in Appendix~\\ref{{app:star}} as not validated (Figure~\\ref{{fig:depth}}). ") + withdraw
    app_par = (f"\n\\paragraph{{Real-data readings of the anisotropic test, not validated.}} With the anisotropic star the real cloud is more hierarchical than its star beyond the noise ($z\\le-2$) in {n_in} of 12 backbones on ImageNet ($K{{=}}30$) and {n_c} of 12 on CIFAR-100 ($K{{=}}20$), and never less ({pos_any} of 24 with $z\\ge+2$; Table~\\ref{{tab:b34-depthvariants}}, Figure~\\ref{{fig:depth}}a,b). The CIFAR-100 readings fall in the regime ($n{{=}}100$, $K\\ge12$) where the synthetic false-alarm rate is {100*fa100:.0f}\\%; the ImageNet readings fall in the regime ($n{{=}}1000$) where it is {100*fa1000:.0f}\\% and power is at least {pw1000_min:.2f} at every noise ratio (Figure~\\ref{{fig:depth}}c). We report both without certification, under the bar fixed before the sweep. % expR56_depth_variants.csv, expR55b_depth_power_leafframe.csv\n")
    json.dump(dict(json.load(open('rebuttal/results/phaseB_depth_decision.json')), fa_n100_kge12=float(fa100), fa_n1000=float(fa1000), fa_n100_k6=float(fa100_6), pw_min_r03=float(pw_min03), pw1000_min=float(pw1000_min)), open('rebuttal/results/phaseB_depth_decision.json', 'w'), indent=1)
    abstract_sent = " A depth test against matched stars does not certify hierarchy above the labelled clusters."
    intro_sent = " A matched-star test does not certify hierarchy above the labelled clusters."
    figcap = r"\textbf{The matched-star depth test, reported as not validated.}"
    fig_main = False
figenv = ("\n\\begin{figure}[t]\n\\centering\n\\includegraphics[width=\\linewidth]{figures/fig_depth_test.pdf}\n\\caption{" + figcap +
          r" (a,b)~Excess B under the hub-randomizing null for the real centroids (filled, family colors) and for matched stars (hollow circle: isotropic; square: anisotropic), CIFAR-100 ($K{=}20$) and ImageNet ($K{=}30$); the number under each backbone is the $z$ of real minus anisotropic star (negative = more hierarchical above the frame). (c)~Power ($z\le-2$) on synthetic hierarchies with $n{=}1000$ and the frame at the leaf clusters (solid, one line per $K$) or at the top level (dotted, zero by construction); dashed: false alarms ($z\le-2$) on pure stars at $n{=}1000$ (gray) and $n{=}100$ (black). % expR56_depth_variants.csv, expR55b_depth_power_leafframe.csv, expR55_depth_power.csv" + "\n}\n\\label{fig:depth}\n\\end{figure}\n")
anchor = "% expR52_census_haar_p999_200.csv, expR59_imagenet_bootstrap_summary.csv, expR45_convnet_rows.csv"
rep(anchor, anchor + "\n\n" + lead + body + ("\n" + figenv if fig_main else ""))
if not fig_main:
    rep("\\subsection{What the excess certifies: synthetic calibration}\n\\label{app:star}\n", "\\subsection{What the excess certifies: synthetic calibration}\n\\label{app:star}\n" + figenv.replace("\\begin{figure}[t]", "\\begin{figure}[H]") + app_par + "\n")
# abstract + intro sentences
rep("present throughout Pythia and OLMo, and absent for sentence embedders on class names yet present for the same embedders on a real text hierarchy.",
    "present throughout Pythia and OLMo, and absent for sentence embedders on class names yet present for the same embedders on a real text hierarchy." + abstract_sent)
rep("and a backbone trained in hyperbolic space shows the same clustered arrangement as its Euclidean twin. % expR52_census_haar_p999_200.csv, expR53_text_haar_p999_200.csv, expR51_meru_control.csv",
    "and a backbone trained in hyperbolic space shows the same clustered arrangement as its Euclidean twin." + intro_sent + " % expR52_census_haar_p999_200.csv, expR53_text_haar_p999_200.csv, expR51_meru_control.csv, expR56_depth_variants.csv")
open(p, 'w').write(T); print("T3 applied:", "MAIN" if validated else "APPENDIX")
