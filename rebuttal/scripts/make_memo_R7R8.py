#!/usr/bin/env python3
"""Memo of the restructuring reruns (R7 = expR62 sample-level under the record; R8 = expR63 MERU under the record):
prints the numbers, writes rebuttal/results/phaseC_memo.json (what sweep_freeze.py checks) and appends the memo to
iclr2027/CHANGELOG_final.md. Also evaluates the two stop conditions of the brief:
  (a) sample-level excess no longer within noise (most cells genuine)  -> stop;
  (b) MERU showing depth beyond its twin on ImageNet (validated regime) -> stop."""
import json, sys
import pandas as pd
R = 'rebuttal/results/'
s = pd.read_csv(R + 'expR62_samplelevel_record.csv'); m = pd.read_csv(R + 'expR63_meru_record.csv')
NM = {"i21k_t":"ViT-T","i21k_s":"ViT-S","i21k_b":"ViT-B","i21k_l":"ViT-L","dinov1_b":"DINO-B","dinov2_s":"DINOv2-S","dinov2_b":"DINOv2-B","dinov2_l":"DINOv2-L","dinov2_g":"DINOv2-G","clip_b":"CLIP-B","clip_l":"CLIP-L","siglip_b":"SigLIP-B"}
DS = {"cifar100": "CIFAR-100", "dtd": "DTD"}
gen = s[s.genuine_bh]; names = [f"{NM[r.model]}/{DS[r.dataset]}" for r in gen.itertuples()]
memo = dict(sl_n=len(s), sl_genuine=int(s.genuine_bh.sum()), sl_genuine_names=names, sl_within=int((~s.genuine_bh).sum()), sl_absz_lt2=int((s.z.abs() < 2).sum()),
            sl_sign_neg=int((s.excess < 0).sum()), sl_sup_lo=float(s.delta_sup.min()), sl_sup_hi=float(s.delta_sup.max()), sl_d999_lo=float(s.delta_999.min()), sl_d999_hi=float(s.delta_999.max()),
            sl_exc_lo=float(s.excess.min()), sl_exc_hi=float(s.excess.max()), sl_sup37_lo=float(s.delta_sup_expR37.min()), sl_sup37_hi=float(s.delta_sup_expR37.max()))
img = m[m.modality == 'image']
for ds in ('imagenet', 'cifar100'):
    for sz in ('s', 'b', 'l'):
        a = img[(img.dataset == ds) & (img.model == f'meru_{sz}')].iloc[0]; b = img[(img.dataset == ds) & (img.model == f'clip_{sz}')].iloc[0]
        memo[f'meru_{ds}_{sz}'] = dict(meru_exc=float(a.excess), meru_r=int(a.r_above), meru_gen=bool(a.genuine_bh), clip_exc=float(b.excess), clip_r=int(b.r_above), clip_gen=bool(b.genuine_bh),
                                      meru_nat=float(a.delta_999_native), meru_euc=float(a.delta_999), clip_nat=float(b.delta_999_native), clip_euc=float(b.delta_999), meru_z=float(a.z_depth), clip_z=float(b.z_depth))
im = img[img.dataset == 'imagenet']
memo['meru_in_exc_range'] = [float(im[im.family == 'meru'].excess.min()), float(im[im.family == 'meru'].excess.max())]
memo['clip_in_exc_range'] = [float(im[im.family == 'clip'].excess.min()), float(im[im.family == 'clip'].excess.max())]
memo['meru_in_z_range'] = [float(im[im.family == 'meru'].z_depth.min()), float(im[im.family == 'meru'].z_depth.max())]
memo['clip_in_z_range'] = [float(im[im.family == 'clip'].z_depth.min()), float(im[im.family == 'clip'].z_depth.max())]
memo['meru_in_nat_vs_euc_maxgap'] = float((im[im.family == 'meru'].delta_999_native - im[im.family == 'meru'].delta_999).abs().max())
memo['meru_in_all_r200'] = bool((im.r_above == 200).all()); memo['meru_genuine_total'] = int(m.genuine_bh.sum()); memo['meru_rows'] = len(m)
# depth beyond twin on ImageNet: MERU z <= -2 while CLIP twin z > -2, for any size
memo['meru_depth_beyond_twin'] = [sz for sz in ('s', 'b', 'l') if memo[f'meru_imagenet_{sz}']['meru_z'] <= -2 and memo[f'meru_imagenet_{sz}']['clip_z'] > -2]
memo['meru_any_depth_in'] = [f"{fam}_{sz}" for fam in ('meru', 'clip') for sz in ('s', 'b', 'l') if memo[f'meru_imagenet_{sz}'][f'{fam}_z'] <= -2]
stop_a = memo['sl_genuine'] > memo['sl_n'] / 2; stop_b = len(memo['meru_depth_beyond_twin']) > 0
memo['stop_a_samplelevel_mostly_genuine'] = bool(stop_a); memo['stop_b_meru_depth_beyond_twin'] = bool(stop_b)
json.dump(memo, open(R + 'phaseC_memo.json', 'w'), indent=1)
print(json.dumps(memo, indent=1))
print("STOP" if (stop_a or stop_b) else "GO")
if '--changelog' in sys.argv:
    cl = f"""
## 18a. Reestructuración — memo de los reruns R7 y R8 (antes de tocar el texto)

**R7 — lectura a nivel de muestra bajo el registro** (`expR62_samplelevel_record.py` → `expR62_samplelevel_record.csv`; mismos subconjuntos que
expR37: CIFAR-100 10 img/clase, DTD 22 img/clase, semilla 0; Haar × p99.9 × 200, BH sobre 24 celdas; supremo crudo de la misma nube al lado).
Supremo crudo {memo['sl_sup_lo']:.3f}–{memo['sl_sup_hi']:.3f} (expR37: {memo['sl_sup37_lo']:.3f}–{memo['sl_sup37_hi']:.3f}); δ̂₉₉.₉ {memo['sl_d999_lo']:.3f}–{memo['sl_d999_hi']:.3f};
exceso {memo['sl_exc_lo']:+.4f}..{memo['sl_exc_hi']:+.4f}, signo negativo en {memo['sl_sign_neg']}/24; |z| < 2 en {memo['sl_absz_lt2']}/24; **genuinas BH {memo['sl_genuine']}/24**
({', '.join(names) if names else 'ninguna'}); dentro del ruido (no genuinas) {memo['sl_within']}/24.

**R8 — control MERU bajo el registro** (`expR63_meru_record.py` → `expR63_meru_record.csv`; 6 modelos × 2 datasets × 2 modalidades; censo Haar ×
p99.9 × 200 con BH sobre 24 celdas; δ̂₉₉.₉ nativo (Lorentz/angular) vs euclídeo; test de profundidad anisotrópico, 10 semillas: ImageNet K = 30
validado, CIFAR-100 K = 20 no validado). Imágenes de ImageNet: exceso MERU {memo['meru_in_exc_range'][0]:+.4f}..{memo['meru_in_exc_range'][1]:+.4f} vs CLIP
{memo['clip_in_exc_range'][0]:+.4f}..{memo['clip_in_exc_range'][1]:+.4f} (r = 200 en todas: {memo['meru_in_all_r200']}); |nativo − euclídeo| ≤ {memo['meru_in_nat_vs_euc_maxgap']:.4f};
profundidad z MERU {memo['meru_in_z_range'][0]:+.2f}..{memo['meru_in_z_range'][1]:+.2f} vs CLIP {memo['clip_in_z_range'][0]:+.2f}..{memo['clip_in_z_range'][1]:+.2f};
celdas con z ≤ −2 en ImageNet: {memo['meru_any_depth_in'] or 'ninguna'}; MERU con profundidad más allá de su gemelo: {memo['meru_depth_beyond_twin'] or 'ninguno'}.
Genuinas BH en total {memo['meru_genuine_total']}/{memo['meru_rows']}.

**Condiciones de parada del brief.** (a) exceso a nivel de muestra ya no dentro del ruido: {'SÍ — STOP' if stop_a else 'no'}; (b) MERU con profundidad
más allá del gemelo: {'SÍ — STOP' if stop_b else 'no'}. → {'STOP: se informa antes de la fase de texto.' if (stop_a or stop_b) else 'GO: la lectura cualitativa se mantiene; sigue la fase de texto.'}
Checks en `sweep_freeze.py` (sección R7/R8) contra `phaseC_memo.json`.
"""
    open('iclr2027/CHANGELOG_final.md', 'a').write(cl); print("memo appended to CHANGELOG")
