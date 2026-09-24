> **Frozen 17 Sept 2026; only typographical changes after this date.**

# CHANGELOG — pasada de versión final (2026-09-03)

Brief: "Revision brief — Is There a Platonic Tree? — final-version pass". Todo número del paper sale de
un fichero de `rebuttal/results/`; ningún número se ha teclado a mano salvo las filas árbol/H²/esfera de
la tabla de calibración (sin CSV; ver `TODO_author.md`). Los seis protocolos vigentes de `CODE_MAP.md`
no se han tocado.


> **Numeración actual de tablas (desde la limpieza del 22-09, con la tabla de Khrulkov en §5.1):** 1 Khrulkov (§5.1), 2 censo (§5.2);
> apéndice por orden de primera cita: 3 robustez, 4 calibración, 5 censo completo, 6 panel de modelos, 7 texto, 8 muestra
> (Khrulkov en el panel b), 9 profundidad, 10 potencia, 11 mapa de árboles, 12 WordNet, 13 local, 14 corolario, 15 procedencia.
> Las entradas anteriores a §49 usan la numeración de su momento (profundidad = 8, potencia = 9, muestra = 3, corolario = 13).

## 0. Desviación importante respecto al brief (y por qué)

El brief asumía que las cachés por imagen de ImageNet existían en esta máquina ("all CPU, all cheap").
No existían: `*_imagenet_train.npz` y `*_imagenet_fulltrain.npz` se extrajeron en la máquina remota
(4090) y nunca se copiaron. El ImageNet crudo sí está en `/media/HDD_4TB_1/javi/ILSVRC2012_img_train`,
y ambos splits son funciones deterministas del listado ordenado de ficheros, así que:

- `rebuttal/scripts/extract_imagenet_cache_local.py` regenera `{m}_imagenet_train.npz` (100 img/clase,
  `files[:100]` por clase) con el pipeline original **idéntico** (mismos `MODEL_DEFS`, transforms de
  timm/open_clip por modelo, mismas reglas fp16, mismo orden). GPU: 2× RTX 2080 Ti, ~3.5 h de pared
  (DINOv2-G a 518 px en dos shards, uno por GPU, fusionados después).
- **Evidencia de fidelidad**: la δ_norm cruda de ImageNet de los 12 modelos coincide con la de
  `exp20_null_ztable.csv` (caché original) a 4 decimales en los 12 casos
  (p. ej. ViT-T 0.1227/0.1227, DINOv2-S 0.1185/0.1185, DINOv2-G 0.0808/0.0808). Check permanente en
  `sweep_freeze.py` ("R1 fidelity").
- Para **R2** no se regeneró el submuestreo de exp5 (100/clase con `RandomState(0)` sobre el fulltrain,
  otras 100k imágenes por modelo): `exp21b` usa los centroides de la caché del censo, que es el
  protocolo que §3 declara ("the mean of each class's cached training embeddings, 100 on ImageNet") y
  trata a los 12 modelos igual. Los raws cambian ligeramente respecto a exp21 (ver §5 abajo).
  **Pendiente de veto del autor** (`TODO_author.md`).
- El centroid store (`results/centroids/imagenet_train/*.npy`) difiere de la caché en los **12** modelos
  (0.001–0.004 en DINOv2/CLIP/SigLIP, 0.017–0.025 en ViT supervisados), no solo en los marcados con †:
  por eso R1 recomputó las 12 celdas de ImageNet y no solo las 4 supervisadas.

## 1. Entorno (§1 del brief)

- `gen_appendix.py`, `gen_appendix2.py`, `sweep_freeze.py`: rutas `/home/javi/...` sustituidas por
  `PLATONIC_RESULTS` (default: `<repo>/rebuttal/results`). Scripts nuevos usan `PLATONIC_ROOT` /
  `PLATONIC_RESULTS`. `HF_HOME`/`TORCH_HOME` → `/media/HDD_4TB_2/javi/hf_cache`.
- Shim `numpy._core → numpy.core` en `gen_appendix2.py`, `make_treemap_fig.py`, `make_fig_overview.py`,
  `sweep_freeze.py`: `exp23_treemap_controls.npz` fue picklado con numpy ≥ 2 y este entorno tiene 1.x.
- Compilación: tectonic en un entorno conda de scratch (no hay TeX local); `tcolorbox` añadido al preámbulo.

## 2. Reruns

### R1 — `expR39b_census20_cache.py` → `expR39b_census20_cache.csv` (72 celdas)
Mismo estimador, misma null y mismas semillas que `expR39` (real 10 semillas; 20 réplicas espectrales,
semillas 300+rep, 5 semillas cada una). Las 12 celdas de ImageNet desde la caché regenerada; las 60 filas
de transfer copiadas **verbatim** de `expR39_census20.csv` (ya eran cache-based) — dicho en el log del
script. Ninguna fila usa el store (`store_centroids = 0` en las 72).

Recuentos recalculados (no copiados del brief): **69/72** sign-negativas · **57/72** por debajo de las 20
réplicas · **43/48** en los cuatro datasets jerárquicos · planos **24/24** en signo, **14/24** bajo todas ·
excepciones sign-positivas: DINOv2-S/B/G en ImageNet, rango 0/20 y z ≤ +1.1 · resumen z secundario para
la leyenda de B20: |z| ≥ 2 en 50/72, z ≤ −3 en 31/48 jerárquicas.
(El brief esperaba 69/72, 56/72 y 42/48 a partir de las filas de transfer de expR39; con las 12 celdas
ImageNet desde caché: 57/72 y 43/48.)

### R2 — `exp21b_local_global_K200.py` → `exp21b_local_global_K200.csv` (66 pares)
K = 200 permutaciones (`RandomState(0)`), α = 0.05, calibración escalar (sin búsqueda de capas). Por par
y métrica (mKNN k=10 y CKA lineal, en R y en H): raw, media y s.d. de la null, τ95 (eq. 9: estadístico de
orden ⌈(1−α)(K+1)⌉ de {raw} ∪ nulls), p = (1+#{null ≥ raw})/(K+1) (eq. 10), score calibrado
max((raw−τ95)/(1−τ95), 0) (eq. 12), y el antiguo raw − media (continuidad). Números en §5 abajo.

### R3 — criterio del tree map ejecutable
`gen_appendix2.py`: `SELECTED` ya no es un dict tecleado; se calcula de `exp23_config_diagnostics.json`
(entre configuraciones con `maxfrac ≤ 0.5`, mayor `cpcc`, por dataset) y se **asserta** igual a
{ImageNet: cosine-average, CIFAR-100: cosine-complete}; falla ruidosamente si cambia. Misma función en
`make_treemap_fig.py` y `make_fig_overview.py`.

### R4 — tabla de calibración generada
`gen_appendix.py` escribe `appendix_tables/tab_calibration.tex` (el .tex tecleado se sustituyó por
`\input`). Fila gaussiana desde `exp1_delta_controls.csv` (variante `gauss`, deduplicada por d):
d = 192/384/512/768/1024/1536 → δ_norm 0.104/0.081/0.074/0.061/0.055/0.046 (**antes** solo 192/768/1536:
0.104/0.061/0.046). Filas árbol/H²/esfera: sin fichero de resultados (exp6 imprimió a stdout), tecleadas
dentro del generador y re-verificadas en vivo por `sweep_freeze.py`.

### R5 — regeneración y verificación
`gen_main_table.py`, `gen_appendix.py`, `gen_appendix2.py`, `gen_review_tables.py`, `gen_provenance.py`,
figuras, y `sweep_freeze.py` con la sección 2 (censo) leyendo `expR39b` y la sección 5 (§5) leyendo
`exp21b`. Resultado del sweep al cierre: ver "Verificación" al final.

## 3. Cambios estructurales

- **Caja de página 1** (`tcolorbox`) antes del primer párrafo de §1, texto del brief.
- **Figura 1** (conceptual): `\IfFileExists{figures/fig1_concept.pdf}` → imagen; si no, placeholder del
  mismo tamaño (1.25 in de alto, ancho de texto) para que el layout sea final. Leyenda del brief.
  `fig_intro_concept.pdf` (cueva NeurIPS) borrado; ya no se referenciaba.
- **Figura 2 nueva** (`figures/make_fig_overview.py`): (a) δ_norm cruda vs d con la curva gaussiana
  (exp1) y los 12 modelos (expR39b); (b) "same raw, opposite verdict": BGE-base (expR48) frente a la celda
  de visión con raw más cercano, elegida por datos = SigLIP-B/DTD (raw 0.121 ambos; exceso +0.00x vs
  −0.062); (c) ARI DINOv2-B/L/G vs bloque, naive vs seleccionada, ImageNet 0.029→0.380 y CIFAR-100
  0.005→0.389 (exp23).
- **Tabla 1**: 10 → 7 columnas (modelo · familia · δ_norm IN · exceso IN (r/20) · exc. C100 · exc. C10 ·
  ρ_WN) desde `expR39b` + `exp3_alignment.csv`; negrita en celdas sign-positivas. Las columnas p99.9,
  DTD, FMNIST/MNIST, best−R y métrica → `appendix_tables/tab_census_extra.tex` (nueva, generada en
  `gen_main_table.py` desde `expR34`, `exp2`, `expR39b`).
- **Tree map**: la figura principal pasa de 4 paneles a 2 (solo ImageNet, naive vs seleccionada; etiquetas
  ≥ 7 pt); la pareja CIFAR-100 es una figura de apéndice (`fig_treemap_c100.pdf`, mismo script con
  `--appendix`).
- **Figura best-metric** (scatter) movida de §6 al apéndice (subsección Statistics), referenciada desde
  §6. Motivo: §6 se comprimió a media página y el corolario queda en una línea de contribuciones; la
  figura a ancho completo allí desbordaba el texto principal a la p10.
- **§4** retitulado "Findings I: Structure beyond the Null"; el párrafo "The excess certifies clustering,
  not depth" sigue inmediatamente al censo (ya lo hacía).
- **§5**: nuevo párrafo final "Form and content anti-align" (dos frases, movidas desde §7).
- **§6**: comprimido a tres párrafos ("δ_norm predicts the gain within datasets" / "The training objective
  picks the metric" / "A diagnostic, not a policy"), un número cada uno; punteros y comparación de
  políticas al apéndice.
- **Apéndice**: nueva etiqueta `app:census` en "Vision census: excess and significance per cell";
  `tab_census_extra` insertada antes de B2; figura CIFAR-100 del tree map en `app:treemap`.
- Todos los lead-ins de §3–§6 son afirmaciones (p. ej. "A low raw value is not evidence.", "The reading is
  an excess over a matched null.", "ξ reads norm structure, not curvature.", "The island is an artifact;
  sharing is graded.", "WordNet alignment follows supervision, not tree-likeness.", "The recovery is not
  WordNet circularity.", "Local agreement survives calibration.", "Negative curvature concentrates on
  bridges.").
- Leyendas: primera frase en negrita con el takeaway, en todas las figuras y en Tabla 1, B20 y calibración.

## 4. Párrafos reescritos (texto)

- **Abstract**: sustituido por el borrador del brief íntegro (los números que dependían de R1 no cambiaron:
  "69 of 72"). Antes: "(1)~Tree-like class geometry… (2)~The trees are moderately shared… (3)~…";
  ahora prosa continua con la frase de la estrella "…whether or not they were trained in hyperbolic space".
- **§1**: párrafos 1–2 intactos; "Our first contribution is therefore an instrument…" y "The answer has two
  halves…" → tres párrafos: instrumento (2 frases), "What survives is clustered form." y "Sharing is graded,
  and its naive measurement misleads." Solo dos números en toda la intro: "69 of 72" y "0.03 → 0.38"
  (exp23: `('euclid','average')` big_vs_sup 0.029 → `('cosine','average')` 0.380). Contribuciones: cuatro
  ítems ≤ 2 líneas, sin cadenas de puntos y coma; el 4.º es una línea.
- **§3 "The reading is an excess over a matched null."**: "5 null replicates for δ, 20 for ξ" y la frase
  del z-score → 20 réplicas para ambos; evidencia primaria = rango de percentil (20/20 = bajo todas);
  justificación rank-based citando Gröger §5.1; z relegado al apéndice; frase de null "aggregation-aware"
  (Gröger §5.2): mismo supremo con el mismo presupuesto (5×10⁵ cuádruplas por semilla) sobre cada
  réplica; real 10 semillas / réplica 5 afecta a la varianza, no al sesgo. Eliminado el paréntesis largo
  sobre construcciones alternativas de la null (→ puntero a A.robust) y el aparte del whitening.
- **§4 censo**: "Vision: 69/72 cells, stated at two levels." → "Vision: structure beyond the null in 69 of
  72 cells." en lenguaje de rangos (arriba). Eliminados del texto principal: "51/72 |z|≥2", "41/48",
  "34/48 z<−3" (exp20) → sustituidos en la leyenda de B20 por 50/72 y 31/48 (expR39b); "−0.04..−0.08 →
  −0.007..−0.03" (→ A.robust); "−0.039/−0.020, z −4.8/−2.4" de Barlow/BYOL (→ B24).
- **§4 texto**: frases cortas; eliminadas las magnitudes en cadena (S/M −0.015/−0.010, L/XL
  −0.041/−0.049, Pythia −0.032..−0.036, OLMo −0.039/−0.015, erank 1.3→56/98, +0.002..+0.011,
  0.114–0.124, −0.019) → viven en B23/B28/Fig. 4; se conserva una magnitud (0.02 del padding).
- **§4 profundidad-vs-C, training, ORC**: una magnitud por párrafo (se conservan −0.124 vs −0.093 y
  19–91 %; eliminados +1.58/+0.53 pp, −61 %/−46 %, p>0.12, +65–144 %, 7.6/3.2 %, 8.0/4.4 %, 0.3/0.5 % →
  B3, Fig. 3, B6).
- **§5**: "The island is an artifact; sharing is graded." sin cadena (eliminados 0.2–0.5 vs 0.5–0.6, gaps
  +0.03..+0.36, −0.02, 0.13–0.24 vs 0.46–0.57 → B7/B11); párrafo del criterio compactado ("highest CPCC
  among non-degenerate configurations"); "Under the selected configurations…" conserva 0.38 vs 0.48
  (eliminados 0.39 vs 0.63, 0.003–0.12, 0.48–0.52 vs 0.54–0.60, 0.56 vs 0.61, 0.74/0.74 → B7);
  HierarCaps conserva 0.83–0.97 (eliminados +0.32..+0.87, 10–54 % vs 4.2 % → A8); semántica angular
  conserva 0.91 vs 0.71 (eliminados 0.85/0.63); ARI 0.61 conserva; definición de ARI y CPCC en una cláusula.
- **§6**: ver estructural; conserva r = −0.83 (ImageNet NC) y +1.3 pp (CLIP); eliminados del texto
  principal −0.87, −0.67..−0.78, −0.62/−0.79, −0.31..−0.86, −0.87/−0.65, +0.23/−0.34, −0.20..−0.53,
  +2.05 pp, p≈4×10⁻³⁸, ≤0.2 pp, +0.9, −0.60..−0.83, −0.55..−0.82, +0.63/+0.64, +0.41/+0.28, p=0.011,
  −0.24 pp, +0.5..+2 pp, −2.6 pp, 0/60 → B12, A2/A3, B8, A6, B9.
- **§7**: primer párrafo abre con el espejo de la caja; conserva la frase del "corrected picture" (0.13–0.38);
  la frase de anti-alineación se movió a §5; Limitations (i)–(vii) intactas.
- **§2**: frase del nivel-muestra acortada (puntero a A.sample).

## 5. Tabla 1 — celda a celda (exp20, 5 réplicas → expR39b, 20 réplicas; fuente de cada valor en la cabecera)

| modelo | exc IN (z) exp20 → exc IN (r/20) expR39b | exc C100 exp20 → expR39b | exc C10 exp20 → expR39b |
|---|---|---|---|
| i21k_t | −0.021 (−2.7) → −0.019 (20/20) | −0.043 → −0.039 | −0.059 → −0.051 |
| i21k_s | −0.020 (−2.3) → −0.021 (20/20) | −0.033 → −0.031 | −0.096 → −0.090 |
| i21k_b | −0.028 (−3.0) → −0.027 (20/20) | −0.017 → −0.015 | −0.090 → −0.083 |
| i21k_l | −0.030 (−4.4) → −0.031 (20/20) | −0.048 → −0.043 | −0.098 → −0.096 |
| dinov1_b | −0.005 (−0.6) → −0.005 (16/20) | −0.068 → −0.063 | −0.097 → −0.094 |
| dinov2_s | +0.012 (+1.1) → +0.012 (0/20) | −0.053 → −0.051 | −0.095 → −0.095 |
| dinov2_b | +0.006 (+0.6) → +0.007 (0/20) | −0.043 → −0.042 | −0.135 → −0.134 |
| dinov2_l | −0.008 (−1.1) → −0.008 (20/20) | −0.051 → −0.051 | −0.155 → −0.154 |
| dinov2_g | +0.004 (+0.5) → +0.004 (0/20) | −0.079 → −0.079 | −0.157 → −0.155 |
| clip_b | −0.025 (−3.1) → −0.027 (20/20) | −0.047 → −0.044 | −0.077 → −0.071 |
| clip_l | −0.021 (−2.3) → −0.020 (20/20) | −0.043 → −0.038 | −0.083 → −0.080 |
| siglip_b | −0.019 (−1.7) → −0.017 (20/20) | −0.042 → −0.037 | −0.091 → −0.088 |

Cambio máximo en las filas de transfer (nulls 5 → 20 réplicas): 0.012. ρ_WN sin cambios (exp3).
Ceros a la izquierda restaurados en todas las tablas (−0.021, no −.021): eliminados los `.replace("0.", ".")`
de `gen_main_table.py` y `gen_review_tables.py`.

## 6. Figuras

- `figures/style.mplstyle` (STIX, 8/7 pt, sin spines superior/derecho, fonttype 42) y `figures/palette.py`
  (`FAMILY_COLORS`: supervisado #4C72B0, SSL #DD8452, contrastivo #55A868, LM causal #8172B2, embedder
  #937860, null #7F7F7F discontinuo; datasets solo por marcador) usados por todos los scripts.
- `fig_text_nulls.py`: **corregido** el uso de colores (GPT-2 iba en naranja SSL y Pythia en azul
  supervisado): todos los LMs causales en morado, embedders en marrón; barras huecas = at null (no bajo
  todas las réplicas), en lugar de los tres callouts; barras de error conservadas; fuente `expR48`.
- `make_figs.py`: fuente `expR39b` (antes exp20); título (a) calculado ("69/72"); leyenda única fuera
  (abajo); scatter best-metric sin r en el título → r a la leyenda (FS −0.60..−0.83; NC −0.61..−0.86;
  pooled −0.31/−0.22, salida del script).
- `make_fig3_causal.py`: barras en el color de familia (no rojo); líneas de profundidad DINOv2-B naranja,
  CLIP-B verde.
- `make_treemap_fig.py`: 2 paneles + `--appendix`; viridis; etiquetas 7 pt.
- Alturas: Fig. 1 1.25 in, overview 1.55, excesos 1.85, texto 1.75, causal 1.45, tree map 2.05,
  best-metric 1.5 (apéndice).

## 7. Verificación / estado de compilación

- Compila con tectonic: 0 warnings (undefined/overfull), 0 `??`; texto principal + Ethics Statement en
  p9; Reproducibility abre p10; 30 páginas en total.
- `sweep_freeze.py`: se completa en el bloque "Cierre" al final de este fichero.

## 8. §5 — calibración Gröger (exp21 → exp21b)

Fuente nueva: `exp21b_local_global_K200.csv` (66 pares; K = 200; α = 0.05; centroides = caché del censo,
ver §0). Tabla de apéndice nueva `tab_b31_groger.tex` (B31) con raw / null mean / τ₀.₀₅ / calibrado /
null-centered / pares p<0.05 para mKNN y CKA en R y H.

| medida | exp21 (3 perms, raw → raw−null) | exp21b raw | null mean | τ₀.₀₅ | calibrado (eq. 12) | null-centered | pares p<0.05 |
|---|---|---|---|---|---|---|---|
| mKNN R | 0.474 → 0.464 | 0.472 | 0.0100 (= 10/999) | 0.015 | **0.464** | 0.462 | 66/66 (p = 1/201 todos) |
| mKNN H | 0.427 | 0.425 | 0.0100 | 0.019 | 0.414 | 0.415 | 66/66 |
| CKA R | 0.636 → 0.529 | 0.633 | 0.108 | 0.111 | **0.577** | 0.524 | 66/66 |
| CKA H | 0.644 | 0.640 | 0.108 | 0.111 | 0.586 | 0.532 | 66/66 |

Texto de §5 "Local agreement survives calibration.": antes "mutual-kNN 0.474→0.464 calibrated…; null 0.010",
"CKA 0.636→0.529", "kNN 0.474→0.427; CKA 0.636→0.644"; ahora K/α/eq. 12 explícitos, "0.472 raw, 0.464
calibrated, every pair at p=1/201, null mean 0.010 = k/(n−1)", "CKA 0.633 raw, 0.577 calibrated, null 0.108",
Poincaré "0.472→0.425; 0.633→0.640". La frase de Poincaré se conserva.

## 9. Cierre — verificación

- `sweep_freeze.py` (sección 2 → expR39b, sección 5 → exp21b, check de fidelidad R1 añadido): **67/67 PASS**
  (`rebuttal/results/sweep_freeze_final.log`).
- Compilación tectonic: 30 páginas, 0 warnings (undefined/overfull), 0 `??`; Ethics Statement en p9,
  p10 abre con Reproducibility Statement. PDF: `ICLR2027/main_iclr2027_final.pdf`.
- Generadores ejecutados en orden: gen_main_table, gen_appendix, gen_appendix2, gen_review_tables,
  gen_provenance (`tab_z_provenance.tex` en sincronía; no se incluye en el PDF por decisión previa del autor).

## 10. Follow-up (cinco arreglos)

1. **Figura 1** `[t]` → `[H]`: queda bajo la caja de página 1 (placeholder `\IfFileExists` intacto).
2. **make_fig_overview.py**: panel (a) con leyenda de familias (sup./SSL/contr.) + curva gaussiana, colocada
   bajo los ejes; leyenda raw/exceso del panel (b) bajo los ejes (etiqueta "SigLIP-B (DTD)" en una línea).
3. **fig_text_nulls.py**: leyenda dentro de los ejes, abajo a la derecha (zona vacía), ya no cubre GPT-2 S/M.
4. **Línea ~137 (§4)**: "DINOv2-L is sign-negative but marginal there (z=−1.1)" → "DINOv2-L is below every
   replicate on ImageNet, with a small excess (−0.008)" (expR39b: −0.0084, rango 20/20). En la misma frase, los
   rangos por familia pasan de exp20 a expR39b: ViT −0.020..−0.030 → **−0.019..−0.031**; CLIP/SigLIP
   −0.019..−0.025 → **−0.017..−0.027**; S→G CIFAR-10 −0.095→−0.157 → **−0.095→−0.155**. La frase p99.9
   "40/48 hierarchical cells beyond 2σ" → "**42/48** hierarchical cells below every replicate" (expR40, rango);
   70/72 sin cambio. En el párrafo del censo, "(z ≤ +1.1)" → "(excess ≤ +0.012)" (expR39b, máx. sign-positivo
   +0.0118) y "per-cell ranks and z" → "per-cell ranks". Comentario de fuente de la línea actualizado
   (exp26/exp12/exp11 → expR39b, expR40).
   **Grep de z en líneas 1–210 tras el cambio**: no queda ningún valor de z-score. Quedan dos menciones del
   concepto en §3 (línea 115: "a z-score on a handful of replicates is not rank-based, so z is kept only as a
   secondary summary in the appendix"), que es la frase que el brief §3.4 pide explícitamente.
5. **Párrafo "Text: …"**: tres magnitudes — 0.02 (padding), S/M vs L/XL (−0.015/−0.010 vs −0.041/−0.049,
   expR48/B23) y el rango DBpedia (−0.020..−0.028, exp14), este último absorbido desde la frase "Second, …".
   Eliminadas del cuerpo las cifras de plantillas ("Two scopes": .065/.012, medias −0.012/−0.023/−0.040/−0.041,
   9/10, 10/10) → leyenda de B14 (ya llevaba medias y recuentos; añadido el rango de sensibilidad .065/.012 con
   puntero a la Tabla A7); el detalle GTE-Qwen2 (−0.019 con 3 réplicas; +0.002, rango 8/20 con 20) → leyenda
   de B23. El bloque queda en tres párrafos cortos con una frase-puntero a A.templates.
- Verificación: `sweep_freeze.py` **67/67 PASS**; compilación 30 páginas, warnings = 0, `??` = 0; Ethics en
  p9 (1); p10 abre con "R EPRODUCIBILITY S TATEMENT". PDF actualizado en `ICLR2027/main_iclr2027_final.pdf`.

## 11. R6 — censo a 200 réplicas (resolución Gröger): censo de registro

Ficheros nuevos: `expR39c_census200_cache.csv` (72 celdas; copia de expR39b con `range(200)`, semillas 300+rep, 5 semillas/réplica) y
`expR48b_text_census200_bs1.csv` (15 modelos; copia de expR48 con `range(200)`, 3 semillas/réplica como expR48). Ambos guardan `frac_null_above`,
`r_above` y `p_left = (1+#{null ≤ real})/201`. Cómputo: 3 procesos de visión (ImageNet en dos mitades + transfer) y 2 de texto (una GPU cada uno),
BLAS monohilo, ~1.5 h de pared en máquina compartida. Los CSV de 20 réplicas (`expR39b`, `expR48`) se conservan como histórico; sus tablas B20/B23
se sustituyen por las de 200 réplicas (labels `tab:b20-census200`, `tab:b23-text200`). **"Genuine" se define una sola vez en §3: p ≤ 0.05, r ≥ 191/200**;
todas las frases "below every replicate" del texto principal se han sustituido.

### Recuentos (texto principal)

| enunciado | antes (20 réplicas, expR39b) | ahora (200 réplicas, expR39c) |
|---|---|---|
| sign-negativas (abstract, intro, §4) | 69/72 | **68/72** (ViT-T FashionMNIST pasa de -0.012 (11/20) a +0.001 (95/200, p=0.527)) |
| celdas genuinas (antes: bajo las 20 réplicas) | 57/72 | **55/72** (p ≤ 0.05); bajo las 200 réplicas: 44/72 |
| jerárquicas genuinas | 43/48 | **43/48** |
| planas: signo / genuinas | 24/24 / 14/24 | **23/24 / 12/24** |
| excepciones sign-positivas | DINOv2-S/B/G IN (excess ≤ +0.012, rango 0/20) | DINOv2-S/B/G IN (excess ≤ +0.011, r = 0/0/1 de 200, p ≥ 0.99) + ViT-T FMNIST (+0.001, p 0.53) |
| leyenda B20: \|z\| ≥ 2 / z ≤ −3 jerárquicas | 50/72 / 31/48 | 49/72 / 32/48 |
| §4 l.137 rangos ImageNet: ViT / CLIP-SigLIP | −0.019..−0.031 / −0.017..−0.027 | −0.020..−0.030 / −0.016..−0.025 |
| §4 l.137 DINOv2-L ImageNet | "below every replicate, excess −0.008" | "genuine, excess −0.008, r = 200/200" |
| §4 l.137 p99.9 (expR40, 20 réplicas) | "42/48 below every replicate" | "42/48 genuine at 20 replicates" (mismo recuento, lenguaje de p) |
| Fig. 3 título (a) | 69/72 | 68/72 (calculado) · huecos = p > 0.05 |
| Fig. 2 (b) exceso SigLIP-B/DTD | −0.062 | −0.066 |

### Tabla 1, celda a celda (exceso (r) 20 réplicas → exceso (r/200, p) 200 réplicas)

| modelo | ImageNet | CIFAR-100 | CIFAR-10 |
|---|---|---|---|
| i21k_t | -0.019 (20/20) → -0.020 (199/200, p=0.010) | -0.039 (20/20) → -0.038 (200/200, p=0.005) | -0.051 (20/20) → -0.042 (191/200, p=0.050) |
| i21k_s | -0.021 (20/20) → -0.022 (200/200, p=0.005) | -0.031 (20/20) → -0.031 (199/200, p=0.010) | -0.090 (20/20) → -0.081 (200/200, p=0.005) |
| i21k_b | -0.027 (20/20) → -0.025 (200/200, p=0.005) | -0.015 (18/20) → -0.016 (187/200, p=0.070) | -0.083 (20/20) → -0.073 (200/200, p=0.005) |
| i21k_l | -0.031 (20/20) → -0.030 (200/200, p=0.005) | -0.043 (20/20) → -0.044 (200/200, p=0.005) | -0.096 (20/20) → -0.088 (200/200, p=0.005) |
| dinov1_b | -0.005 (16/20) → -0.003 (162/200, p=0.194) | -0.063 (20/20) → -0.064 (200/200, p=0.005) | -0.094 (20/20) → -0.085 (200/200, p=0.005) |
| dinov2_s | +0.012 (0/20) → +0.011 (0/200, p=1.000) | -0.051 (20/20) → -0.052 (200/200, p=0.005) | -0.095 (20/20) → -0.095 (200/200, p=0.005) |
| dinov2_b | +0.007 (0/20) → +0.007 (0/200, p=1.000) | -0.042 (20/20) → -0.042 (200/200, p=0.005) | -0.134 (20/20) → -0.135 (200/200, p=0.005) |
| dinov2_l | -0.008 (20/20) → -0.008 (200/200, p=0.005) | -0.051 (20/20) → -0.051 (200/200, p=0.005) | -0.154 (20/20) → -0.153 (200/200, p=0.005) |
| dinov2_g | +0.004 (0/20) → +0.005 (1/200, p=0.995) | -0.079 (20/20) → -0.079 (200/200, p=0.005) | -0.155 (20/20) → -0.155 (200/200, p=0.005) |
| clip_b | -0.027 (20/20) → -0.025 (200/200, p=0.005) | -0.044 (20/20) → -0.044 (200/200, p=0.005) | -0.071 (20/20) → -0.062 (200/200, p=0.005) |
| clip_l | -0.020 (20/20) → -0.018 (200/200, p=0.005) | -0.038 (20/20) → -0.039 (200/200, p=0.005) | -0.080 (20/20) → -0.072 (200/200, p=0.005) |
| siglip_b | -0.017 (20/20) → -0.016 (199/200, p=0.010) | -0.037 (20/20) → -0.038 (200/200, p=0.005) | -0.088 (20/20) → -0.080 (200/200, p=0.005) |

Cambio máximo de exceso entre 20 y 200 réplicas: 0.0131. Celdas que cambian de veredicto (bajo-todas a 20 vs p ≤ 0.05 a 200): i21k_s/mnist (-0.044, r=186, p=0.075; antes 20/20); i21k_b/mnist (-0.030, r=181, p=0.100; antes 20/20).
Marca ∘ en Tabla 1 (transfer no genuino): i21k_b/cifar100 (p=0.070).

### Censo de texto (expR48, 20 réplicas → expR48b, 200 réplicas)

| modelo | exceso 20 (rango/20) | exceso 200 (r/200, p) | veredicto 200 |
|---|---|---|---|
| gpt2 | -0.0150 (20/20) | -0.0132 (183/200, p=0.090) | no genuino |
| gpt2_m | -0.0098 (18/20) | -0.0090 (173/200, p=0.139) | no genuino |
| gpt2_l | -0.0406 (20/20) | -0.0389 (200/200, p=0.005) | genuino |
| gpt2_xl | -0.0485 (20/20) | -0.0472 (200/200, p=0.005) | genuino |
| pythia_410m | -0.0361 (20/20) | -0.0352 (200/200, p=0.005) | genuino |
| pythia_1b | -0.0319 (20/20) | -0.0310 (200/200, p=0.005) | genuino |
| pythia_2b8 | -0.0361 (20/20) | -0.0346 (200/200, p=0.005) | genuino |
| olmo_1b | -0.0393 (20/20) | -0.0381 (200/200, p=0.005) | genuino |
| bge_base | +0.0037 (5/20) | +0.0053 (23/200, p=0.886) | no genuino |
| bge_large | +0.0029 (5/20) | +0.0034 (53/200, p=0.736) | no genuino |
| gte_base | +0.0113 (0/20) | +0.0121 (2/200, p=0.990) | no genuino |
| gte_large | +0.0069 (1/20) | +0.0071 (11/200, p=0.945) | no genuino |
| gte_qwen2 | +0.0019 (8/20) | +0.0033 (63/200, p=0.687) | no genuino |
| e5_base | +0.0024 (9/20) | +0.0040 (39/200, p=0.806) | no genuino |
| e5_large | +0.0047 (3/20) | +0.0049 (28/200, p=0.861) | no genuino |

Genuinos: 6/15. GPT-2 S (p = 0.090) y M (p = 0.139) no son genuinos y L/XL sí → se mantiene "emerges with scale" (regla del brief).
Frase de §4: "S and M sit at their null (−0.015/−0.010) while L and XL lie below every replicate (−0.041/−0.049)" → "S and M are not genuine (−0.013/−0.009) while L and XL are (−0.039/−0.047)"; "OLMo at both scales tested" → "OLMo-1B" (OLMo-7B no re-extraído);
embedders "sit at their nulls" → "are not genuine"; GTE-Qwen2 "not robust across replicate counts" → "not genuine either". Leyenda B14: S/M (z −1.6/−1.2, −0.015/−0.010) → (−0.013/−0.009, p 0.09/0.14).
§3: "over 20 spectrum-null replicates, for δ and ξ alike" → "over 200 (20 for ξ)"; definición de genuine; "each replicate 5" → "5 (3 in the text census)". Intro: "ranked against 20" → "200". Leyendas Fig. 3/4: huecos = p > 0.05; Fig. 4 "20 null replicates" → "200".

### Verificación
- `sweep_freeze.py` (sección 2 → expR39c; sección 3 → expR48b): **67/67 PASS**.
- Compilación: 30 páginas, warnings = 0 (Overfull de B20 corregido con tabcolsep 1.8 pt), `??` = 0; Ethics en p9; p10 abre con Reproducibility Statement. PDF: `ICLR2027/main_iclr2027_final.pdf`.

## 12. Pasada de contribución (cuatro ediciones de texto + el instrumento como herramienta)

1. **Abstract**: "nearly universal in vision (68 of 72 cells)" → "nearly universal on datasets with a class hierarchy (43 of 48 cells;
   55 of 72 overall)" (expR39c, genuine = p ≤ 0.05). §4 (párrafo del censo): "p ≥ 1.00" → "p ≥ 0.99" (DINOv2-G ImageNet, r = 1/200 → p = 0.995).
2. **§4, nuevo párrafo "Relation to neural collapse."** (8 líneas, solo `papyan2020prevalence`), inmediatamente después de "The excess certifies
   clustering, not depth.": el ETF de neural collapse como caso extremo de la estrella (δ = 0 sin jerarquía); lo que separa nuestra lectura:
   (i) la organización certificada no es plana — hubs de superclase recuperados en §5 y curvatura negativa en los puentes; (ii) aparece en
   modelos sin etiquetas, en datasets de transfer y en LMs causales, fuera del ámbito de neural collapse, y el test de estrella emparejada
   pregunta lo que neural collapse no pregunta. §2: "…without testing tree-likeness" + "; §4 relates our excess to the collapsed simplex".
3. **Intro**: al final del párrafo del instrumento, dos frases de posicionamiento: Gröger et al. calibran similitud entre modelos; nosotros
   geometría dentro de un modelo, y comparar los árboles resultantes entre modelos necesita su propia calibración (§5).
4. **El instrumento como herramienta**: `ICLR2027/tool/calibrated_delta.py` (entrada: matriz de centroides n×d `.npy` + etiquetas de superclase
   opcionales; salida: δ_norm cruda, exceso espectral con rango r/200 y p de cola izquierda, y el test de profundidad con estrella emparejada de
   B29; funciones y semillas copiadas verbatim de `expR39c_census200_cache.py` y `expR50_depth_test.py`). `ICLR2027/tool/run_checks.py`
   lo ejecuta en ViT-L/CIFAR-100 y DINOv2-L/ImageNet y escribe `rebuttal/results/tool_check.json`; `sweep_freeze.py` (sección TOOL) compara con
   la Tabla 1 (exceso 3 dp, r, p) y con B29 (profundidad 3 dp, z 1 dp; en ImageNet con los centroides del store, como expR50).
   **Resultado: reproducción exacta** — ViT-L/C100 exceso −0.0441, r 200, p 0.005; DINOv2-L/IN −0.0079, r 200, p 0.005; profundidad
   +0.0396 (z +4.06) y −0.0289 (z −2.46), idénticos a las tablas. Frase nueva en §3 ("The instrument is a single script: …") y en el
   Reproducibility Statement con el placeholder `\url{ANONYMIZED-REPO}` (ver `TODO_author.md`).
5. **Ajuste de páginas**: el párrafo nuevo y las frases añadidas empujaban ~9 líneas a p10 → §6 comprimido de nuevo (sin la pregunta
   introductoria; tres párrafos de 3–4 líneas; se conservan r = −0.83 y +1.3 pp), cláusula de la null PC-permutación eliminada en §4
   ("a PC-permutation null confirms every verdict": los veredictos son ahora por rango), cláusula de Poincaré acortada en §5, frase de las
   ResNet-50 acortada en el censo, Figura 1 a 0.95 in, overview 1.65 in, excesos 1.7 in, tree map 1.95 in.
- Verificación: `sweep_freeze.py` **71/71 PASS** (67 + 4 checks del tool); compila con 30 páginas, 0 warnings, 0 `??`; el texto principal
  (Limitations incluidas) termina en p9 y el Ethics Statement abre p10. PDF: `ICLR2027/main_iclr2027_final.pdf`.

## 13. Pasada final de consistencia

1. **Vocabulario signo/genuino alineado con §3**: intro "What survives is clustered form." → "Structure beyond the matched null is nearly
   universal on datasets with a class hierarchy (43 of 48 cells; 55 of 72 overall; negative excess in 68 of 72)"; lead-in de §4 →
   "Vision: negative excess in 68 of 72 cells, genuine in 55."; contribución 2 → "Genuine tree-like form in nearly every vision cell with
   a class hierarchy".
2. **§3**: "(exact enumeration when C ≤ 30)" → "(for C = 10 the 5×10⁵ draws cover every quadruple many times over, so the supremum is exact)".
   Verificado: `delta_norm` de `expR39c_census200_cache.py` (y de todos los censos) muestrea 5×10⁵ cuádruplas por semilla en todos los casos;
   con C = 10 hay 210 cuádruplas distintas, cubiertas miles de veces.
3. **MERU**: "whether or not they were trained in hyperbolic space" (abstract) y "whether or not the model was trained in hyperbolic space"
   (intro) → "including a backbone trained in hyperbolic space" (un solo modelo, MERU). La caja de p1 no contenía la frase.
4. **ORC**: el párrafo "Negative curvature concentrates on bridges" pasa íntegro al apéndice A.14 ("ORC edge types", nuevo label `app:orc`);
   en §4 queda un puntero de una línea; en el párrafo de neural collapse "(below)" → "(Appendix A.14)". Hueco de la Figura 1 de vuelta a 1.2 in.
5. **p99.9 a 200 réplicas** (`expR40b_p999census200.py` → `expR40b_p999census200.csv`; copia de expR40 con `range(200)` y la caché del censo
   como fuente de centroides en todos los datasets — expR40 leía ImageNet del store —, mismo estadístico p99.9 y mismas semillas; 3 procesos,
   ~70 min). Recuentos: sign-negativas **70/72 (sin cambio)**; jerárquicas genuinas **44/48** (antes "42/48 below every replicate" a 20 réplicas
   sobre el store); genuinas 56/72; ImageNet: DINO/DINOv2 genuinos (p 0.005), ViT-T no (p 0.114), CLIP-B p 0.0498. Frase de §4 reescrita
   ("…44/48 hierarchical cells genuine under the same protocol (200 replicates)…"); B21 regenerada desde expR40b (exc/r/p por celda, recuentos
   en la leyenda); la columna p99.9 de `tab_census_extra` pasa de expR34 (20 réplicas) a expR40b (200). Check nuevo en el sweep.
   Sign-positivas bajo p99.9: ViT-B/CIFAR-100 (+0.001, p 0.56) y ViT-T/FMNIST (+0.001, p 0.53).
6. **`ICLR2027/tool/demo.py`** (n = 60, d = 32, 50 réplicas, semillas fijas): nube aleatoria → exceso −0.014, r 47/50, p 0.078 → no genuino;
   estrella de 6 clusters → exceso −0.054, p 0.020 → genuino, profundidad +0.039 (z +0.9) → sin profundidad; jerarquía 6×5 → exceso −0.115,
   p 0.020 → genuino, profundidad −0.025 (z −0.6) → más arbóreo que su estrella. A n = 60 con K = 6 hubs el test de profundidad está
   infra-potenciado (probé cuatro parametrizaciones; |z| ≤ 0.6 en todas), así que el demo y el README leen el signo de la profundidad y remiten
   a B25/B29 (n = 100–1000) para la calibración. `tool/README.md`: uso, check de reproducción de dos celdas, demo.
- Verificación: `sweep_freeze.py` **73/73 PASS**; compila con 30 páginas, 0 warnings, 0 `??`; el texto principal (Limitations incluidas)
  termina en p9 y el Ethics Statement abre p10. PDF: `ICLR2027/main_iclr2027_final.pdf`.

## 14. Respuesta a la revisión — FASE A (experimentos y memo; sin ediciones de prosa) — CHECKPOINT

Todo lo que sigue está fijado antes de las ejecuciones; ningún ajuste se eligió mirando los números. Memo con solo números:
`ICLR2027/MEMO_phaseA.md` (generado por `rebuttal/scripts/make_memo_phaseA.py`).

**A1 — 2×2 {null} × {estadístico}, 200 réplicas, caché del censo, BH (FDR 0.05) por 72 celdas / 15 modelos.**
Scripts: `expR52_census_haar_p999_200.py` (motor de visión; `--null gauss|haar --stat sup|p999`), `expR54_census_haar_sup_200.py`
(wrapper Haar × sup), `expR53_text_haar_p999_200.py` (motor de texto; embeddings extraídos una vez y cacheados en
`Platonic/results/text_cache`). Ficheros: `expR52_census_haar_p999_200.csv` (registro propuesto), `expR54_census_haar_sup_200.csv`,
`expR53_text_{haar_p999,haar_sup,gauss_p999}_200.csv`; los cuadrantes Gauss × sup/p99.9 de visión son expR39c/expR40b.
Semillas: null `300+rep` en ambas construcciones (expR46 usaba `700+rep` para Haar); real 10 semillas de cuádruplas; réplica 5 (visión) /
3 (texto). Bootstrap ImageNet bajo el registro: `expR59_imagenet_bootstrap.py` (30 remuestreos × 12 modelos; **20** réplicas Haar por
remuestreo en vez de 200 — basta para la s.d. del exceso; documentado).

**A2 — test de profundidad.** `expR55_depth_power.py` (barrido sintético: {estrella, 2 niveles, 3 niveles} × K {6,12,20,30} ×
ratio {0.1,0.3,0.6} × {iso, aniso} × n {100,1000}, d = 768, 5 semillas; protocolo exacto de B29 con estrella **isotrópica**; 720 runs)
→ `expR55_depth_power.csv` y `figures/fig_depth_power.pdf`. `expR56_depth_variants.py` (backbones reales: estrella isotrópica (a) vs
anisotrópica (b) = muestra Haar con el espectro real de cada superclase; **10** semillas de estrella; K: CIFAR-100 {20,10,5} — los marcos
10/5 son un clustering aglomerativo consenso de los 20 hubs, `expR56_frames_cifar100.csv` —, ImageNet cortes WordNet {30,10,60})
→ `expR56_depth_variants.csv`. Control con checkpoints fine-tuned: **no disponible** (sin pesos ni features guardados; coste en
`TODO_author.md`). Hueco anotado: la potencia sintética se barrió solo con la estrella isotrópica (la de B29), no con la anisotrópica.

**A3 — censo coseno.** `expR57_census_cosine.py`: centroides L2-normalizados, distancia geodésica esférica (arccos), null Haar
construida sobre la nube normalizada y renormalizada (como expR38), p99.9, 200 réplicas, BH → `expR57_census_cosine_haar_p999_200.csv`,
`expR57_text_cosine_haar_p999_200.csv`.

**A4 — tree map sin corte.** `expR58_treemap_cutfree.py`: dendrogramas reconstruidos desde la caché del censo (exp22/23 solo guardan
resúmenes), 6 configuraciones, correlación cofenética entre modelos y acuerdo en 10⁴ tripletes (`RandomState(0)`), más ARI en el corte
→ `expR58_treemap_cutfree{,_summary}.csv`.

**A5.** Semillas de estrella 3 → 10 en expR56. Bib: `he2026helm` y `yang2025hyperbolic` son NeurIPS **2025** (39.ª edición; verificado
en dblp) → `year={2025}`. Añadidos `narayan2011curvature` (Phys. Rev. E 84, 066108, 2011; arXiv 0907.1478), `kennedy2013hyperbolicity`
(arXiv:1307.0031, 2013) y `adcock2013treelike` (ICDM 2013, pp. 1–10; Crossref DOI 10.1109/ICDM.2013.77). Sin usar aún en el texto.

**Números clave (detalle en el memo).** Registro (Haar × p99.9, BH): visión sign-negativas 70/72, genuinas 49/72 (51 sin BH),
ImageNet+CIFAR-100 18/24, otros cuatro 31/48; texto 7/15 (GPT-2 S, L, XL, Pythia ×3, OLMo-1B; GPT-2 M p = 0.15). Otras construcciones:
Haar × sup 66/72, 46/72, 15/24, 31/48; Gauss × p99.9 70, 52, 19, 33; Gauss × sup 68, 52, 19, 33. DINOv2-S/B/L/G en ImageNet: genuinos bajo
las dos construcciones p99.9, por encima de todas las réplicas (r = 0) bajo las dos supremo. Bootstrap ImageNet: s.d. del exceso
≤ 0.0011, 30/30 remuestreos negativos en los 12 modelos. Potencia del test de profundidad (estrella isotrópica, sintético): **0.00**
en las 360 jerarquías (2 y 3 niveles, todo K, ratio, anisotropía, n); falsas alarmas en estrellas puras: z ≥ +2 en 16 %, z ≤ −2 nunca.
Backbones reales: la estrella isotrópica da z ≥ +2 en 8/12 (CIFAR-100, K = 20; máx. +6.8); la anisotrópica lo elimina (0/12) y da
z ≤ −2 en 3/12 (CIFAR-100) y 4/12 (ImageNet, K = 30). Coseno: 70/72, BH 56/72, IN+C100 22/24; acuerdo de veredictos con el registro
euclídeo 65/72; DINOv2 ImageNet genuino; texto 8/15 (incl. GPT-2 M y GTE-Qwen2; no GPT-2 S). Tree map: la isla naive aparece también sin
corte (ImageNet euclid-average: cofenética 0.36 vs 0.80 dentro del bloque, tripletes 0.47 vs 0.74); bajo cosine-average tripletes 0.77 vs
0.76 y cofenética 0.48 vs 0.78.

**Verificación.** `sweep_freeze.py` con la sección de Fase A (existencia, consistencia r↔p, BH monótono, flags): **84/84 PASS**.
No se ha tocado el .tex ni el PDF en esta fase.

## 15. Respuesta a la revisión — FASE B (decisiones del autor aplicadas al texto)

Decisiones fijadas por el autor antes de tocar el .tex: censo de registro = Haar × p99.9 × BH euclídeo (el coseno solo como
robustez); titular 49/72, 18/24, 70/72; excepción DINOv2–ImageNet resuelta como artefacto del supremo; escala solo DINOv2 en
CIFAR-10/CIFAR-100; texto "robust at the larger scales, fragile at the smaller"; test de profundidad reejecutado con el marco en las
hojas y estrella anisotrópica bajo una regla de decisión fijada de antemano; B29 retirada del texto principal. Commits: WIP `552fbf5`,
cierre en el commit de esta sección.

**B1 — censo de registro en Tabla 1, Figs. 2(a,b), 3 y 4.** `gen_main_table.py` lee `expR52_census_haar_p999_200.csv` (columnas: modelo,
familia, δ̂₉₉.₉ IN, exceso IN con ∘ si no es genuino BH, r/200 (p), exceso C100, exceso C10, ρ_WN; `tab_census_extra` añade la columna
"sup. IN (r)" de expR54). `make_figs.py` (Fig. 2a: hueco = no genuino BH; título "70/72 sign-negative, 49 genuine"),
`make_fig_overview.py` (medias del null como marcadores huecos; celda más cercana a BGE-base = ViT-L/DTD), `fig_text_nulls.py` (expR53),
`make_fig3_causal.py` (1.3 in). Apéndice: B20 desde expR52; B21 sustituida por la tabla de veredictos 2×2 (`tab_b21_2x2.tex`, códigos
Hp/Hs/Gp/Gs, 49/46/52/52); B23 desde expR53 con columna 2×2; nuevas B32 (coseno), B33 (tree map sin corte), B34 (estrella iso/aniso ×
K), B35 (potencia, marcos superior y de hojas), B36 (bootstrap ImageNet). Eliminadas del apéndice: B2 (tabla z), B18, B21 antigua,
B29. §3: dos frases de disclosure ("Two choices changed after pre-registration…": estadístico p99.9 y null Haar, con el motivo de cada una
y el puntero a la 2×2) — en el cuerpo, no en nota. §4: explicación mecanicista del artefacto del supremo en DINOv2–ImageNet (el supremo
sobre 5·10⁵ cuádruplas de 1000 centroides es un extremo de una sola cuádrupla; en DINOv2 unas pocas clases muy separadas lo elevan por
encima de todas las réplicas mientras el p99.9 queda por debajo).

**B2 — nombres y recuentos.** "tree-like" → "beyond-null (clustered) structure" en abstract, intro, contribuciones, §4, §7 y pies;
"genuine" := p BH ≤ 0.05 (frase de definición en §3). Recuentos, viejo → nuevo: signo negativo 68/72 → **70/72**; genuinas 55/72
(Gauss × sup, sin BH) → **49/72** (Haar × p99.9, BH); "43 of 48 …" → **18/24 en ImageNet + CIFAR-100** (y 31/48 en los otros cuatro
solo en el apéndice); "every family" se mantiene con la salvedad explícita de que ViT-T es genuino solo en DTD; celdas no genuinas en
ImageNet/CIFAR-100 listadas (ViT-T ambas, CLIP-B y SigLIP-B en ImageNet, ViT-S/B en CIFAR-100; el resto en B20). DTD y CIFAR-10 nunca "hierarchical".
Texto: 7/15 → **7/15** (mismo número, distinto conjunto: entra GPT-2 S p = 0.02, sale GPT-2 M p = 0.15); GPT-2 L/XL genuinos bajo las
cuatro construcciones y bajo coseno; Pythia ×3 y OLMo-1B genuinos en todo; embedders +0.002..+0.004 en nombres de clase, DBpedia
−0.020..−0.028 (bajo el supremo original, dicho en el texto).

**B3 — test de profundidad.** El barrido con el marco en las hojas (`expR55b_depth_power_leafframe.py`: {estrella, 2 niveles, 3 niveles} × K {6,12,20,30} × ratio {0.1,0.3,0.6} × {iso, aniso} × n {100,1000}, d = 768, 5 semillas; estrella anisotrópica con 10 semillas; configuraciones con más hojas que puntos omitidas; 600 runs, 2 h 08 min en 4 procesos) → `expR55b_depth_power_leafframe.csv`, `fig_depth_power.pdf` (el de marco superior queda como `fig_depth_power_topframe.pdf`). Regla fijada de antemano: potencia ≥ 0.8 para 2 y 3 niveles a ratio ≤ 0.3 para ambos n, y falsas alarmas ≤ 5 % en cada dirección. Resultado: potencia **1.00** en todas esas configuraciones (≥ 0.85 a ratio 0.6 con n = 1000; 0.70 en K = 30, ratio 0.6); falsas alarmas z ≥ +2: **0 %**; z ≤ −2: **6.2 %** agregadas — 0 % con n = 1000 y con K = 6, **17 %** con n = 100 y K ≥ 12 (3–8 puntos por cluster). La regla **falla** por la dirección z ≤ −2 → rama de apéndice: el texto principal dice que la profundidad por encima de los clusters etiquetados **no está certificada**, con la regla, las cifras y el motivo (párrafo nuevo tras el censo en §4); `fig_depth_test.pdf` ((a) CIFAR-100 K = 20, (b) ImageNet K = 30, real vs estrella iso/aniso con z; (c) potencia n = 1000 y falsas alarmas para n = 100 y 1000) va al apéndice A.5 con el párrafo "Real-data readings of the anisotropic test, not validated" (aniso: 4/12 ImageNet, 3/12 CIFAR-100 con z ≤ −2, 0/24 con z ≥ +2; las lecturas de CIFAR-100 caen en el régimen n = 100, K ≥ 12 con 17 % de falsas alarmas; las de ImageNet en el régimen n = 1000 con 0 % y potencia ≥ 0.85). B29 (estrella isotrópica) retirada del texto principal: el párrafo dice que dio el veredicto contrario en CIFAR-100 (z hasta +6.8, 8/12) y 16 % de falsas alarmas en estrellas sintéticas. Control positivo fine-tuned: no se pudo correr (sin checkpoints), dicho en el texto. Abstract e intro (ii): una frase "does not certify hierarchy above the labelled clusters". Frases MERU/NC rebajadas (B30, §4). Decisión: `rebuttal/results/phaseB_depth_decision.json`; script de aplicación con las dos ramas: `rebuttal/scripts/phaseB_t3_apply.py`. Limpieza: los pies de B30/B34/B35 ya no nombran scripts internos; B34 dice que sus lecturas no están certificadas.

**B4 — afirmaciones recortadas.** Escala: solo "DINOv2 deepens with scale on CIFAR-10 (−0.071 → −0.130) and, weakly, on CIFAR-100
(−0.031 → −0.040)"; sin afirmación general. Anti-alineación: una frase sin lead-in dentro del párrafo WordNet de §5. "Most tree-like
family" eliminado. Tree map con tres medidas (ARI en el corte, cofenética entre modelos, acuerdo en 10⁴ tripletes): la isla naive es
propiedad del corte y de la configuración euclídea (cofenética 0.36 vs 0.80, tripletes 0.47 vs 0.74; bajo cosine-average tripletes
0.77 vs 0.76). DBpedia abre el párrafo de texto. Párrafo coseno en §4 ("The angular reading agrees": 56/72, 22/24, acuerdo 65/72; DINOv2
ImageNet genuino en ambas lecturas).

**B5 — estructura.** §6 (corolario ORC) íntegro al apéndice (`app:corollary`) con un puntero de tres líneas en §7; ξ a una frase en §3;
párrafo de proyección/tareas de §3 al apéndice; related work: párrafo "δ against a random null, in network science" (Narayan & Saniee
2011; Kennedy et al. 2013; Adcock et al. 2013) que dice qué es nuevo (null con espectro exacto, corrección BH en censo, test de
profundidad con estrella emparejada, backbone hiperbólico como control); intro "What is new, in order": test de profundidad, MERU,
receta/escala. Limitación (i): la frase "against matched stars the residual depth is marginal" afirmaba un resultado retirado (B29)
y se sustituye por el alcance del test (certifica jerarquía por encima de su marco; dentro de los clusters no se testea) — única
edición en Limitations, forzada por la retirada.

**B6 — verificación.** `sweep_freeze.py` con sección de Fase B (recuentos de registro, ViT-T solo DTD, conjunto no genuino, rangos,
tendencias DINOv2, 2×2, bootstrap, coseno, texto, tree map, tool bajo el protocolo de registro, potencia/falsas alarmas del test de profundidad contra la regla, lecturas reales aniso/iso, marco superior): **100/100 PASS**.
Tool: `calibrated_delta.py` reproduce Tabla 1 (exceso, r, p exactos) y B34 (profundidad 3 dp, z 1 dp) en ViT-L/CIFAR-100 y
DINOv2-L/ImageNet. Compilación: 33 páginas, 0 warnings, texto principal termina en la p. 9. `main_iclr2027_final.pdf` exportado.
Checklist de preguntas de revisor: `ICLR2027/REVIEWER_CHECKLIST_phaseB.md`.

## 16. Pasada final tras la lectura completa (seis puntos del autor)

**(1) Test de profundidad acotado por régimen.** Se mantiene la regla prefijada y su fallo en n = 100 (K ≥ 12: 17 % de falsas alarmas
z ≤ −2; K = 6: 0 %) y se añade la lectura por régimen: en n = 1000 el test cumple la barra (potencia 1.00 a ratio ≤ 0.3,
0.90 a 0.6; 0 % de falsas alarmas en ambas direcciones), así que las lecturas de ImageNet (K = 30) quedan **certificadas dentro
del régimen validado**: profundidad por encima de las superclases WordNet en 4/12 backbones (ViT-S, ViT-B, ViT-L, DINOv2-L;
z de -2.4 a -4.0), no en los otros ocho, ninguno menos jerárquico que su estrella; CIFAR-100 (n = 100, K = 20: 3/12)
sigue sin validar. El texto dice que el acotado por régimen se adoptó **después** del barrido (§4 y A.5). Dominio de validez declarado:
n ≈ 1000, no validado por debajo de ~10 puntos por cluster (Limitación (i)). Actualizados: abstract, intro (ii), contribución 2, frase MERU
("its depth reading used the withdrawn isotropic star and is not part of the certified result"), frase NC ("answers yes for four ImageNet
backbones, three of them supervised ViTs"), Limitación (i), §7 (una cláusula), A.5 (párrafo "Real-data readings by regime"). Figuras: la
nueva figura del cuerpo es `fig_depth_main.pdf` ((a) ImageNet K = 30 real vs estrellas con z; (b) potencia n = 1000 y falsas alarmas
n = 100/1000) — queda numerada **Figura 4** por orden de aparición (la de texto pasa a ser la 5); `fig_depth_cifar100.pdf` (solo CIFAR-100)
en A.5; la figura de intervenciones (`fig3_causal.pdf`) pasa al apéndice, nueva subsección "Training interventions" (`app:interventions`),
con puntero de una línea en "The form tracks training". `phaseB_t4_apply.py` aplica todo esto desde expR56/expR55b
(`phaseB_depth_regime.json`); la rama estricta anterior sigue en `phaseB_t3_apply.py`.

**(2) §3, párrafo del null.** La construcción Haar descrita como registro ("random orthogonal coefficients recombined with the real
singular values, so the sample spectrum is exact") y la gaussiana entre paréntesis como alternativa reportada en A.14 (`app:robust`).
**(3)** Frase de agregación: "the same supremum over the same quadruple budget" → "the same statistic, the 99.9th percentile of the same
5×10⁵ sampled defects per seed". **(4)** Frase duplicada "A low raw value is not evidence." eliminada al final del párrafo homónimo;
también la cláusula obsoleta "a 99.9th-percentile variant preserves every sign" de los ajustes del estimador.

**(5) Barrido de clases bajo el registro (`expR60_c_sweep_record.py`, Haar × p99.9 × 200; mismos subconjuntos y semillas que exp19;
3 modelos de la tabla; 165 celdas, 67 min con 6 workers) → `expR60_c_sweep_record.csv`; NC adv copiado de exp19.** Viejo → nuevo:
"DINOv2-L at C = 50: excess −0.124 vs −0.093" (supremo) → -0.046 vs -0.042 bajo el registro (solo en la tabla; el
texto da los umbrales: los subconjuntos aleatorios llevan más exceso que los coherentes en CLIP-L a todo C, DINOv2-L hasta C = 100,
DINOv2-G hasta C = 20; más allá convergen). "Raw δ rises with C" → falso bajo el registro (δ̂₉₉.₉ plano en C: 0.030–0.034 para
DINOv2-L aleatorio); sustituido por "random subsets still score -0.08 to -0.14 at C = 10". "DINOv2-G reaches its null
at C ≥ 500 … reproducing the census exception" → "stays below its null at every C (excess -0.008 at C = 1000, r = 200)"
(bajo el supremo, +0.0035). §5: "The census exception is consistent with this…" → "This is consistent with the census: DINOv2's ImageNet
structure is genuine under the record and the least aligned with WordNet's top level". Tabla B3 regenerada (`gen_appendix2.py`, columna
"below" = semillas con p ≤ 0.05). La ventaja de explotabilidad (NC adv) de los aleatorios se mantiene en todo C.

**(6) DBpedia bajo el registro (`expR61_dbpedia_record.py`; embeddings re-extraídos como exp14, centroides cacheados en
`Platonic/results/text_cache/dbpedia_{m}.npz`) → `expR61_dbpedia_record.csv`.** −0.028 / −0.024 / −0.020 (supremo, null gaussiana,
3 réplicas) → -0.021 .. -0.017 (BGE / E5 / GTE), r = 200/200, p = 0.005, genuinos BH los tres; también Haar × sup y
Gauss × p99.9 en el CSV. Texto: "genuine excess (−0.020 to −0.028 under the original supremum reading)" → "beyond-null excess under the
record (−0.017 to −0.021, every replicate above the real value; Table A9)". Tabla A9 con las dos lecturas (CI95 fundido en la columna FS).

**Ajuste a 9 páginas.** Recortes: párrafo de profundidad compactado; "Negative curvature concentrates on bridges" fundido en el párrafo NC;
frases acortadas en §1 (calibración de similitud), §3 (panel, ORC k, resolución), §4 (extracción, anisotropía, coseno), §5 (WordNet, DBpedia,
Gröger), §7 (disociación, cláusula de profundidad); alturas: Fig. 1 placeholder 0.3 in, overview 1.5 in, panel de exceso 1.45 in, profundidad
1.5 in, text-nulls 0.75\linewidth, tree map 0.92\linewidth. Limitaciones: solo (i) editada (dominio de validez).

**Verificación.** `sweep_freeze.py`: checks nuevos de régimen (potencia/falsas alarmas por n, hits de ImageNet por nombre y z), de expR60
(umbrales por modelo, C = 10, DINOv2-G bajo el null a todo C, NC adv) y de expR61 (rango y r): **103/103 PASS**. Compilación: 33 páginas,
0 warnings, 0 overfull, texto principal termina en la p. 9; ningún nombre de script en el PDF. `main_iclr2027_final.pdf` exportado.

## 17. Hueco para la Figura 1 real (1.2 in)

Placeholder de la Fig. 1 a **1.2 in** (antes 0.3 in tras los recortes de la pasada final). Para que el texto principal vuelva a terminar
en la p. 9 con ese hueco: (a) el párrafo de §3 "Class count is a measured variable, not a nuisance" se funde en el de §4 "Hierarchy depth,
not class count" (mismo control; el párrafo de §4 abre ahora con la descripción del barrido: C ∈ {10, …, 1000}, composiciones aleatoria y
WordNet-coherente); (b) "Local agreement survives calibration" recortado en dos líneas (kNN, CKA y lectura Poincaré en una frase; ningún
número cambia). (c) No hizo falta reducir la Fig. 3 (sigue a 1.45 in); Figs. 1 y 4 intactas. Comprobado: texto principal termina en la p. 9
con el placeholder de 1.2 in; 33 páginas, 0 warnings; sweep 103/103 (sin números nuevos). PDF exportado.

## 18a. Reestructuración — memo de los reruns R7 y R8 (antes de tocar el texto)

**R7 — lectura a nivel de muestra bajo el registro** (`expR62_samplelevel_record.py` → `expR62_samplelevel_record.csv`; mismos subconjuntos que
expR37: CIFAR-100 10 img/clase, DTD 22 img/clase, semilla 0; Haar × p99.9 × 200, BH sobre 24 celdas; supremo crudo de la misma nube al lado).
Supremo crudo 0.087–0.145 (expR37: 0.087–0.145); δ̂₉₉.₉ 0.036–0.093;
exceso -0.0232..+0.0058, signo negativo en 18/24; |z| < 2 en 12/24; **genuinas BH 10/24**
(DINOv2-S/CIFAR-100, DINOv2-B/CIFAR-100, DINOv2-L/CIFAR-100, DINOv2-G/CIFAR-100, SigLIP-B/CIFAR-100, ViT-S/DTD, ViT-B/DTD, ViT-L/DTD, DINOv2-L/DTD, DINOv2-G/DTD); dentro del ruido (no genuinas) 14/24.

**R8 — control MERU bajo el registro** (`expR63_meru_record.py` → `expR63_meru_record.csv`; 6 modelos × 2 datasets × 2 modalidades; censo Haar ×
p99.9 × 200 con BH sobre 24 celdas; δ̂₉₉.₉ nativo (Lorentz/angular) vs euclídeo; test de profundidad anisotrópico, 10 semillas: ImageNet K = 30
validado, CIFAR-100 K = 20 no validado). Imágenes de ImageNet: exceso MERU -0.0101..-0.0075 vs CLIP
-0.0100..-0.0064 (r = 200 en todas: False); |nativo − euclídeo| ≤ 0.0002;
profundidad z MERU -1.54..-0.52 vs CLIP -1.85..-0.99;
celdas con z ≤ −2 en ImageNet: ninguna; MERU con profundidad más allá de su gemelo: ninguno.
Genuinas BH en total 12/24.

**Condiciones de parada del brief.** (a) exceso a nivel de muestra ya no dentro del ruido: no; (b) MERU con profundidad
más allá del gemelo: no. → GO: la lectura cualitativa se mantiene; sigue la fase de texto.
Checks en `sweep_freeze.py` (sección R7/R8) contra `phaseC_memo.json`.

## 18. Reestructuración: una tesis, tres actos (fase de texto)

Sin cambios en censo, test de profundidad, tree map ni texto; cambia el orden y la historia. Dos resultados suben del apéndice al cuerpo y por eso se
rehicieron bajo el registro (memo en §18a). Decisiones del autor aplicadas: subtítulo sin cambiar (pendiente); nivel de muestra al cuerpo (R7);
MERU al abstract (R8); "tool" eliminado del texto (el script queda en el repositorio y en una frase del Reproducibility Statement); corolario en el
apéndice, §6 lleva solo su consecuencia. Scripts: `rebuttal/scripts/phaseC_main_body.tex.tmpl` (cuerpo nuevo con marcadores) y
`phaseC_restructure.py` (rellena desde los ficheros, empalma, mueve los párrafos al apéndice, decide la Fig. 2(b), comprueba etiquetas y palabras
prohibidas); el cuerpo anterior queda en `rebuttal/results/phaseC_old_main_body.tex`.

**Mapa de secciones (párrafo antiguo → destino).**
- Abstract, caja, §1 (4 párrafos + 4 contribuciones): reescritos según el brief; "What is new, in order" → §2(b).
- §2 antiguo (5 párrafos) → 3: (a) "Latent hyperbolicity, the target claim" (hyperbolic DL + lecturas de latent hyperbolicity); (b) "Calibrating δ and
  geometry against nulls" (network science + Kornblith/Ansuini/Pope/NC + PRH/Koepke/Gröger/Park + qué es nuevo); (c) "Background" (cuatro puntos, ln 2,
  δ_norm, orden, caveats de Fournier, ORC en una frase con puntero a A.16).
- §3: "Classes are centroids…" → "Where the premise is read, and where hierarchy is measured" (misma prosa + frase nivel de muestra/nivel de clase);
  "A low raw value…", "The reading is an excess…" (sin la frase "The instrument is a single script"); nuevo "Projection and tasks" (dos frases, puntero A.10);
  párrafo ξ → A.21; "Class count…" ya estaba fundido en §4 → ahora A.7.
- §4 (7 párrafos): 1 "Sample-level readings sit within null noise in most cells" (nuevo, R7); 2 censo (antiguo "Vision: beyond-null…", + puntero al
  control de clases en A.7); 3 "The excess certifies clustering, not depth" (sin la frase MERU); 4 profundidad (sin cambios); 5 "Imposing the geometry
  does not create depth" (nuevo, R8; sustituye la frase MERU y la mención B30); 6 NC (−2 líneas: cláusula ORC → puntero); 7 "Scale and training"
  (afirmación de escala DINOv2 + una frase de intervenciones con puntero A.6). Salen de §4: "The angular reading agrees" → §5.3; "Text: recipe and
  scale…" → §5.6 (8 líneas); "Hierarchy depth, not class count" → A.7 (`app:csweep`); "The form tracks training" → A.6 (`app:interventions`);
  figura text-nulls → apéndice "Text census" (`app:textcensus`).
- §5 (7 párrafos): 1 isla naive (fundido: extracción, mapa, cortes degenerados, criterio); 2 "The island is an artifact; sharing is graded" (tres medidas,
  una frase cada una; cierre total bajo tripletes angulares); 3 "The self-supervised tree lives in angles" (nuevo párrafo unificado: tripletes cos/eucl
  DINOv2-L + censo coseno 56/72, 22/24, 65/72 + mecanismo de la isla; puente a §6); 4 WordNet (+ ARI 0.61; anti-alineación como última frase); 5 DBpedia
  (+ los tres controles de circularidad); 6 texto (movido de §4); 7 acuerdo local (5 líneas). HierarCaps → A.19 (`app:hierarcaps`).
- §6 "Consequences for imposing curvature" (nuevo, 3 párrafos): regla de Khrulkov, "measure before imposing", métrica de coste cero. Sustituye a
  "A corollary, in the appendix".
- §7: "What the paper establishes" (caja + tres actos), "Open questions" (nuevo), Limitaciones (i)–(vii) literales + (viii).
- Apéndice recibe, con puntero de una línea desde el cuerpo: ξ (A.21), intervenciones (A.6), ORC (A.16), censo coseno (B32), medidas sin corte (B33),
  corolario (A.10), barrido de clases (A.7, párrafo + B3), HierarCaps (A.19), 2×2 (B21), potencia (B35), A.18 (B17 reescrita desde R7), texto (frases
  de extracción/anisotropía/GTE-Qwen2/plantillas + figura text-nulls).

**Números que suben del apéndice al cuerpo (fuente).** Nivel de muestra (`expR62_samplelevel_record.csv`): supremo crudo 0.087–0.145;
no genuinas 14/24, genuinas 10/24 (DINOv2-S/CIFAR-100, DINOv2-B/CIFAR-100, DINOv2-L/CIFAR-100, DINOv2-G/CIFAR-100, SigLIP-B/CIFAR-100, ViT-S/DTD, ViT-B/DTD, ViT-L/DTD, DINOv2-L/DTD, DINOv2-G/DTD); exceso genuino hasta -0.023; columnas nuevas de la Tabla 1
(exc. C100 / exc. DTD a nivel de muestra, BH sobre 24). MERU (`expR63_meru_record.csv`, imágenes de ImageNet): exceso MERU -0.007..-0.010 vs
CLIP -0.006..-0.010; |nativo − euclídeo| ≤ 0.0002; z de profundidad MERU -1.5..-0.5, CLIP -1.8..-1.0.
Censo coseno (`expR57_census_cosine_haar_p999_200.csv`): 56/72, 22/24, acuerdo 65/72 (ya estaba en §4; ahora en §5.3). Ganancias
(`exp2b_normalized_stack.csv`, FS, Poincaré − coseno tras L2, conjuntos jerárquicos): contrastivos +0.1..+1.3 pp; resto −1.7..+1.1 pp. Regla de Khrulkov
(verificada en el PDF, ec. 5: c = (0.144/δ_rel)², δ_rel = 2δ/diam, c ≈ 0.33 en sus datos): sobre la banda gaussiana de `exp1_delta_controls.csv`
(δ_norm 0.104 en d = 192, 0.046 en d = 1536) da c de 0.48 a 2.5. Rango de exceso ImageNet a nivel de clase (`expR52`): −0.005..−0.020.

**Frases eliminadas del cuerpo y destino.** "The instrument is a single script…" (§3) → eliminada (queda la frase del Reproducibility Statement).
"Comparing the resulting trees…" (§1) → §1 párrafo 3, reformulada. Párrafo "Geometric analyses of representations" → fundido en §2(b). Párrafo
"Representational convergence" → fundido en §2(b). Frase "Those readings are sample-level and uncalibrated; transplanted…" (§2) → §2(a), reformulada.
Frase MERU de "The excess certifies…" → §4.5. Cláusula ORC del párrafo NC → puntero. Frase "Whether there is hierarchy above…" → §4.3 reformulada.
"Hierarchy depth, not class count" → A.7. "The form tracks training" → A.6 (una frase resumida queda en §4.7). Frases de extracción, anisotropía,
GTE-Qwen2 y plantillas del párrafo de texto → A "Text census". Frase HierarCaps → A.19. Controles de circularidad → §5.5. "A corollary, in the
appendix" → §6.3. "What the census establishes" → "What the paper establishes" (reescrito). Figura text-nulls → apéndice.

**Figura 2(b).** Decidido por los números: par a nivel de muestra ViT-T/DTD (imágenes; crudo 0.0873, exceso -0.0004) vs
SigLIP-B/CIFAR-10 (centroides; crudo 0.0874, exceso -0.050), diferencia de crudo 6.4e-05 frente a 2.4e-04 del par BGE/ViT-L;
`phaseC_fig2b.json` lo registra y `make_fig_overview.py` lo lee. (a) y (c) sin cambios.

**Lo que no se hizo y por qué.** Survey de Mettes et al. (IJCV 2024): el texto accesible (arXiv 2305.06611) define la hiperbolicidad de Gromov como
propiedad de espacios pero no enuncia la premisa de que las redes estándar sean latentemente hiperbólicas → no se cita, como pedía el brief.
Subtítulo: sin cambiar hasta decisión del autor.

**Matices respecto al borrador del brief.** (1) A nivel de muestra no son "dos o tres" celdas genuinas sino 10/24 (la familia DINOv2 en
CIFAR-100 y DTD, SigLIP-B en CIFAR-100 y los ViT S/B/L en DTD), con excesos pequeños (≥ -0.023); el abstract, §1, §4.1, §6.1 y §7 dicen
"dentro del ruido en la mayoría de las celdas; pequeño donde sobrevive" en lugar de "dentro del ruido". La condición de parada (a) no se disparó
(umbral fijado en > 12/24). (2) Las ganancias Poincaré − coseno no son "nada para el resto": DINOv2-S en CIFAR-10 da +1.1 pp y ViT-L en DTD +1.0 pp
(y DINOv2-G −1.7); §6.3 dice "inconsistente para el resto (−1.7 a +1.1 pp)". (3) Las celdas MERU/CLIP de imágenes de ImageNet no son todas genuinas
BH (r 171–197); el párrafo habla de "the same clustering excess" sin llamarlas genuinas.

**Verificación.** `sweep_freeze.py`: sección R7/R8 (4 checks) y sección de texto reestructurado (10 checks: frases con recuentos/rangos/ganancias/
curvatura, decisión Fig. 2(b), sin "tool"/"we believe"/"interestingly"): **116/116 PASS**. Compilación: 34 páginas, 0 warnings, 0 overfull, texto
principal termina en la p. 9 con el hueco de 1.2 in de la Fig. 1 (Ethics abre en la p. 9); sin nombres de scripts en el PDF. PDF exportado.
Entregables: `ICLR2027/REVIEWER_CHECKLIST_restructure.md`, `ICLR2027/SUMMARY_plain.md`.

## 19. Título

Decisión del autor: `\title{Is There a Platonic Tree? \\ Calibrating Latent Hyperbolicity in Foundation Models}` (antes "Calibrated Class Geometry in
Foundation Models"). Recompilado: 34 páginas, texto principal en la p. 9, 0 warnings; sweep sin cambios. PDF exportado.

## 20. Pasada de prosa (estilo Gröger et al.) — la §19 del brief; la 19 de este fichero ya era el título

Cambia cómo se lee, no lo que dice: estructura, secciones, figuras, tablas, números y afirmaciones congelados. Se reescribe la prosa del texto
principal (abstract, caja, §1–§7, pies de figura) con el vocabulario fijo del brief y las reglas de números, ritmo y frase. Scripts:
`rebuttal/scripts/phaseD_main_body.tex.tmpl` (cuerpo nuevo con marcadores para los números que quedan en la prosa) y `phaseD_prose.py`
(rellena desde los ficheros de resultados, empalma, aplica los renombrados del apéndice y da casa en el apéndice a los números que no la
tenían); el cuerpo anterior queda en `rebuttal/results/phaseD_old_main_body.tex`; `phaseD_changelog.py` lista los números que salieron de la
prosa y dónde están.

**Vocabulario fijo aplicado (renombrado en cuerpo, pies, títulos de figura y apéndice).** the premise (latent hyperbolicity) · the raw reading
(δ̂₉₉.₉ sin calibrar; sustituye a "the reading of record") · the instrument (nunca tool/method/approach) · the excess · clustered structure
(sustituye a "beyond-null (clustered) structure", "tree-like structure", "form": Tabla 1, Fig. 3 y su título de panel `make_figs.py`, A.6
"Clustered structure tracks training", pie de la figura text-nulls, párrafo A.7) · genuine (definido una vez en §3) · the star caveat · the
depth test ("certified" solo para sus veredictos positivos) · the island · the angular tree · la tesis, literal cinco veces (abstract, caja,
final de §1, final de §5, §7). Ideas bautizadas una vez y reutilizadas: dimension / spectrum / statistic confound (§1 → §3, §6).

**Mapa párrafo a párrafo (viejo → nuevo).** Abstract: mismo contenido, una cifra por frase, cierra con la tesis. Caja: la tesis literal.
§1: P1 premisa y quién la cita (citas al final de frase, no encadenadas) → pregunta; P2 "The raw reading is confounded" (bautiza los tres
confounds); P3 el instrumento y la pregunta del título; P4 "The answer has three parts" (49/72, 4/12; cierra con la tesis); contribuciones
sin números; puente a §2. §2: "The premise" / "Calibrating geometry against nulls" / "Background", mismas citas, puente a §3.
§3: apertura en tensión (dos frases) + "The premise lives at the sample level; hierarchy lives at the class level" (antes "Where the premise
is read…"; sin los tamaños 80–6,000, ahora en la nota del panel de modelos) / "A low raw reading is not evidence" (esferas 0.14–0.18 →
Tabla 5; queda el par de extremos gaussianos) / "The reading is an excess over a matched null" (define excess, rank, genuine, clustered
structure; sin la frase del script) / "Two choices changed after pre-registration" (nuevo párrafo, separado del anterior) / "Projection and
tasks"; puente a §4. §4: apertura en tensión; 4.1 "The premise does not survive calibration" (antes "Sample-level readings sit within null
noise in most cells"; 14/24 queda, banda/nombres/máximo → pie B17); 4.2 "Clustered structure is the rule" (el ejemplo del brief, + 18/24;
70/72 → título del panel Fig. 3a, 0.001 → pie B36, lista de celdas no genuinas → marcas ° de la Tabla 1); 4.3 "A star passes the census"
(−0.11 → B25); 4.4 "Depth is certified on ImageNet and not on CIFAR-100" (17 % y 4/12 quedan; barra, potencias, z, +6.8, 16 % → pies
B35/B34); 4.5 "Imposing the geometry does not create depth" (rangos → pie B30); 4.6 "Neural collapse is the flat limit, not the reading"
(−2 líneas); 4.7 "Scale deepens the excess for one family only" (−0.071→−0.130 etc. → Tabla 1 columnas C10/C100; 19–91 % → A.6); cierre
puente a §5. §5: 5.1 "The naive map manufactures an island" (0.82, ≤0.06, 97 % → pie B7; 0.13–0.38 → prosa A.9); 5.2 "The island is an
artifact, and sharing is graded" (0.38 vs 0.48 y 11/12 quedan; cofenética/tripletes → B33; 0.2–0.5 → pie B11); 5.3 "The self-supervised
tree lives in angles" (65/72 queda; 0.91/0.71 → pie B32; 56/72, 22/24 → pie B32); 5.4 "WordNet alignment follows supervision, not
tree-likeness" (0.61 queda; ρ por familia → columna ρ_WN de la Tabla 1); 5.5 "The recovery is not WordNet circularity" (11/12 y +0.26
quedan; 219, 70-way, 0.21–0.29, 0.65–0.82 → pie A9); 5.6 "In text, clustered structure depends on recipe, scale and probe" (rangos y p →
A9/B23); 5.7 "Local agreement survives calibration" (66 queda; 0.472/0.464/0.633/0.577/0.108/0.425/0.640/1/201/10/999 → B31); cierra con
la tesis. §6: apertura en tensión; 6.1 "The raw reading cannot select a curvature" (regla de Khrulkov, 0.48→2.5 y −0.005..−0.020 quedan;
c≈0.33 → A.10 "A diagnostic, not a policy"; −0.023 → B17); 6.2 "Measure before imposing"; 6.3 "A zero-cost readout collects what is
there" (+0.1..+1.3 y −1.7..+1.1 quedan). §7: tesis + tres actos (49/72, 18/24, 4/12), "Open questions", Limitaciones (i)–(viii) sin cifras
(n≈1000 y "ten points" → "class sets of ImageNet's size"; 0.018 → "of the order of the fragile excesses", vive en B28; "six datasets" →
"the datasets of Table 1"). Pies de figura: primera frase en negrita = conclusión, segunda = qué se dibuja (Figs. 1–5).

**Números que salieron de la prosa y dónde viven ahora** (`phaseD_changelog.py` los contrasta; el check del sweep exige que ninguno
desaparezca). Tabla 1: ρ_WN por familia (+0.57..+0.59, +0.49..+0.53, +0.36, +0.18..+0.22), C10/C100 de DINOv2 (−0.071→−0.130,
−0.031→−0.040), exceso IN (−0.005..−0.020), 10/24 genuinas a nivel de muestra. Fig. 3a (título): 70/72. B17: banda 0.087–0.145, 14/24,
nombres de las 10 genuinas, −0.023. B25: estrella de 30 clusters −0.103/−0.104 (la prosa decía ≈−0.11). B30: MERU −0.007..−0.010, CLIP
−0.006..−0.010, 0.0002, z −1.5..−0.5 / −1.8..−1.0. B34: z −2.4..−4.0 de los cuatro certificados, +6.8 iso. B35: barra (0.8, 0.3, 5 %),
1.00/0.90, 0 %, 17 %, marco superior 0.00 y 16 %. B36: 0.0011. B7: 0.82 (ViT-T vs CLIP-B), ≤0.06, 97 %. B11: 11/12 emparejados, 0.19–0.52.
B32: 56/72, 22/24, 65/72, tripletes 0.91/0.71. B33: 0.48/0.78, 0.77/0.76, 0.47/0.74, 0.36/0.80. A9: 219, 70-way, ≤10 %, 0.21–0.29,
0.65–0.82, −0.017..−0.021. B23: +0.002..+0.004, p 0.02/0.15. B31: todos los de acuerdo local. A.6: 19–91 %. A.9 (prosa): 0.13–0.38, 50 %.
A.10 (prosa): c≈0.33. A.3 (nota del panel): 100 imágenes/clase, 80–6,000. Tabla 5: esferas 0.14–0.18. B28: 0.018.

**Verificación.** `sweep_freeze.py`: los checks numéricos no cambian; los checks de texto de la §18 se reanclan a los pies B17/B30
(donde viven ahora los rangos) y se añaden tres checks de prosa: tesis literal ×5; métricas por párrafo (≤2 grupos numéricos por párrafo y
≤1 por frase en §4–§6, "X of Y" / "from A at d=B to C at d=D" cuentan como un grupo, dígitos pegados a nombres (CIFAR-100, GPT-2, DINOv2,
L2) y fórmulas con "=" no cuentan; solo 49/72, 18/24, 4/12 en §1 y §7; ≤1 paréntesis por párrafo, (i)–(viii) exentos; sin punto y coma en
§4–§6 salvo la tesis; sin tool/beyond-null/tree-like structure/form/reading of record/our approach/the method/essentially/largely/
substantially/somewhat/interestingly/notably/importantly/we believe/we note), imprime la frase infractora; conservación de números (todo
literal decimal del cuerpo anterior sigue en cuerpo, apéndice o tablas, con tolerancia de redondeo; única excepción declarada: ≈−0.11 →
−0.103 en B25). **119/119 PASS.** Compilación: 35 páginas, 0 warnings, texto principal en la p. 9 con el hueco de 1.2 in de la Fig. 1;
alturas: Fig. 2 1.35 in, Fig. 3 1.28 in, Fig. 5 0.86\linewidth (Figs. 1 y 4 intactas). `SUMMARY_plain.md` regenerado del texto nuevo.

## 21. Carpeta renombrada a `ICLR2027/`

Decisión del autor (2026-09-08) tras dos syncs de Overleaf que renombraban la carpeta y provocaban conflictos: la carpeta exterior pasa de
`iclr2027/` a `ICLR2027/` en el repo (`git mv`); la carpeta interior del paper sigue siendo `ICLR2027/iclr2027/`. Actualizadas todas las
referencias a la ruta exterior en `rebuttal/scripts/*.py` (sweep, phase*, expR55/55b/60/61/62/63, memos), `ICLR2027/iclr2027/fig_text_nulls.py`,
`CODE_MAP.md` y los .md de la carpeta; las referencias interiores (`HERE.parent/"iclr2027"/"figures"` en los scripts de figuras) no cambian.
Verificado tras el renombrado: generadores de tablas y figuras, sweep 119/119, compilación (35 páginas, texto principal en la p. 9, 0 warnings),
PDF exportado a `ICLR2027/main_iclr2027_final.pdf`. Ninguna línea de contenido cambia.

## 22. Cuatro arreglos de prosa (sin cambio de contenido)

(1) §1, párrafo 3: la frase encadenada se divide en "…what survives calibration, and is it shared? In the words of the title: is there a
Platonic tree?", con las dos citas tras "convergence work". (2) §1, párrafo 2: "(Gromov, 1987)" pasa a seguir a "Gromov δ assigns zero to
trees". (3) §4.1: tras "concentrated in the self-supervised family" se añade "and that family's structure is the angular tree of Section 5,
which cosine collects". (4) §2 "Calibrating geometry against nulls": la frase de network science se reescribe según lo que afirman los tres
trabajos, comprobados en sus textos: Narayan & Saniee (2011) y Kennedy et al. (2013) comparan la curvatura (δ) de redes de comunicación y
sociales con retículos, rejillas hiperbólicas y grafos aleatorios de Erdős–Rényi de tamaño comparable, leyendo δ respecto al diámetro del
grafo (y las redes de carreteras como control negativo); Adcock et al. (2013) miden δ junto con descomposiciones en árbol y concluyen que el
valor de δ a secas, "the simplest and most popular metrics", no basta para caracterizar la estructura arbórea. Se retira la atribución
"found long ago that the raw reading does not certify tree-likeness"; queda "Reading δ against a reference is therefore not new". Recompilado:
texto principal en la p. 9, 35 páginas, 0 warnings; sweep sin cambios, 119/119. PDF exportado.

## 23a. Positive-control pass — memo de la Fase A (sin prosa; `ICLR2027/MEMO_positive_control.md`)

**R9 — profundidad implantada en centroides reales de ImageNet** (`expR64_implanted_depth.py` → `expR64_implanted_depth.csv`, `_summary.csv`,
`fig_implanted_depth.pdf`; 12 backbones × s ∈ {0, 0.25, 0.5, 0.75, 1} × 5 semillas de implante = 300 tests de profundidad con el
protocolo exacto de B34, más censo Haar × p99.9 × 200 en la semilla 0; partición rand6 = 6 super-hubs de 5 (RandomState(0)); el 6-corte
WordNet existe y está anidado pero es 10/5/11/2/1/1, corrido como `wn6` para la semilla 0). Falsas alarmas a s = 0: **0** backbones
(z máx. +0.50). Potencia por s: s=0.0: 0.00, s=0.25: 0.00, s=0.5: 0.00, s=0.75: 0.10, s=1.0: 0.32. s* (primer s con z ≤ −2 en ≥ 4/5 semillas): i21k_t None, i21k_s None, i21k_b None, i21k_l 1.0, dinov1_b None, dinov2_s None, dinov2_b None, dinov2_l None, dinov2_g None, clip_b None, clip_l 0.75, siglip_b 1.0.
Censo sobre las mismas nubes: dispersión máxima entre s de 0.0076 (≤ 1.85 s.d. del null), r mínimo 172/200. Nube real por el
mismo código = B34 (|Δz| máx. 0.00). Cociente intra/entre de las nubes reales a K = 30 (el barrido sintético cubrió 0.1–0.6):
i21k_t 1.55, i21k_s 1.93, i21k_b 1.96, i21k_l 1.88, dinov1_b 2.12, dinov2_s 3.0, dinov2_b 3.71, dinov2_l 3.88, dinov2_g 3.92, clip_b 1.33, clip_l 1.49, siglip_b 1.54. Tamaños de cluster del marco WordNet-30: [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 10, 11, 12, 13, 16, 17, 18, 22, 23, 31, 67, 71, 158, 175, 316].
**Variante tight** (offsets reales encogidos a cociente 0.6, mismo implante, s ∈ {0, 0.5, 1}, 2 semillas; `--tight`, chain r14): i21k_t z(s=0/0.5/1) -0.6/-4.7/-10.2, s=0 hits 0/2; i21k_s z(s=0/0.5/1) -0.2/-7.0/-10.5, s=0 hits 0/2; i21k_b z(s=0/0.5/1) -1.0/-7.7/-11.1, s=0 hits 0/2; i21k_l z(s=0/0.5/1) -1.6/-8.7/-12.3, s=0 hits 0/2; dinov1_b z(s=0/0.5/1) -1.6/-9.4/-11.4, s=0 hits 0/2; dinov2_s z(s=0/0.5/1) -0.2/-6.6/-11.9, s=0 hits 0/2; dinov2_b z(s=0/0.5/1) -0.0/-8.5/-11.0, s=0 hits 0/2; dinov2_l z(s=0/0.5/1) +0.2/-7.4/-10.9, s=0 hits 0/2; dinov2_g z(s=0/0.5/1) +0.2/-9.1/-10.7, s=0 hits 0/2; clip_b z(s=0/0.5/1) -1.7/-6.5/-12.5, s=0 hits 0/2; clip_l z(s=0/0.5/1) -2.4/-9.8/-13.4, s=0 hits 2/2; siglip_b z(s=0/0.5/1) -2.1/-8.9/-11.8, s=0 hits 2/2.

**R11 — sensibilidad conjunta** (`expR66_joint_sensitivity.py` → `expR66_joint_sensitivity{,_summary}.csv`; 72 celdas × 30 remuestreos × 50 réplicas
Haar). Registro 49/72, 18/24. z_joint ≤ −2: **42/72**, 18/24 (caen: CLIP-B/cifar10, CLIP-L/cifar10, ViT-T/dtd, DINOv2-L/fashionmnist, ViT-S/fashionmnist, SigLIP-B/fashionmnist, SigLIP-B/mnist). BH por remuestreo, genuina en ≥ 27/30:
**47/72**, 17/24 (caen: CLIP-B/cifar10, ViT-T/dtd, CLIP-L/imagenet; entra: DINOv2-S/fashionmnist). s.d. bootstrap máx. 0.0026 (0.0011 en ImageNet).

**R10 — ViT-B/16 fine-tuned** (`expR65_hier_finetune.py` → `expR65_hier_finetune.csv`, `expR65_train_log.csv`; subconjunto del censo de 100 img/clase,
2 épocas, mismos batches para (a) CE y (b) CE + CE jerárquica sobre el 30-corte WordNet; el train completo no es legible a velocidad de
entrenamiento desde el disco): frozen exceso -0.0162 z -3.62; CE -0.0153 z -2.33; jerárquico -0.0162 z -2.77.
Criterio del brief (b certificado y a no, o z de b claramente por debajo de a, umbral 1): **False**.

Checks en `sweep_freeze.py` (sección positive-control) contra `positive_control_memo.json`. Sin ediciones de prosa.

## 23b. Positive-control pass — Fase B (texto) y las dos ejecuciones nuevas

**R9b — implante corregido** (`expR64b_implanted_depth_v2.py --frame wn30`; hub = s·super-hub + (1 − s + 0.4·s)·sorteo propio, los hermanos nunca
coinciden; mismos seeds y test que R9; `expR64b_wn30{,_summary}.csv`, `fig_implanted_depth_v2.pdf`). Potencia por s: s=0.0: 0.00, s=0.25: 0.00, s=0.5: 0.00, s=0.75: 0.00, s=1.0: 0.05;
falsas alarmas a s = 0: 0 de 60; detectado a s = 1 (≥ 4/5 semillas): **0/12**; s* = ninguno en los 12; certificados en datos reales
en este marco: 4/12 (mismos cuatro). Censo sobre las mismas nubes: dispersión máx. 0.0052, r mín. 188/200.
Cociente intra/entre real: i21k_t 1.55, i21k_s 1.93, i21k_b 1.96, i21k_l 1.88, dinov1_b 2.12, dinov2_s 3.00, dinov2_b 3.71, dinov2_l 3.88, dinov2_g 3.92, clip_b 1.33, clip_l 1.49, siglip_b 1.54. Variante tight (offsets encogidos a 0.6, s = 0/0.5/1, z medio): i21k_t -0.6/-1.9/-9.2; i21k_s -0.2/-3.4/-10.1; i21k_b -1.0/-3.9/-11.5; i21k_l -1.6/-4.4/-12.7; dinov1_b -1.6/-5.1/-12.1; dinov2_s -0.2/-3.1/-11.5; dinov2_b -0.0/-3.1/-11.2; dinov2_l +0.2/-2.7/-11.4; dinov2_g +0.2/-3.9/-11.6; clip_b -1.7/-3.8/-11.4; clip_l -2.4/-5.8/-13.5; siglip_b -2.1/-5.0/-11.0;
a s = 1 detectan los 12 (2/2 semillas); a s = 0, CLIP-L y SigLIP-B dan falsas alarmas (2/2).

**R12 — marco WordNet equilibrado, pre-registrado** (`--frame wn30bal`; regla: corte K = 30 de la matriz WordNet con mínima varianza de tamaños entre
average/complete/single, elegido y registrado en `expR67_frame_choice.json` antes de cualquier test: **complete** linkage, tamaños
1–154 clases frente a 1–316 del marco de registro; criterio: potencia a s = 1 ≥ 0.8 con cero falsas alarmas a s = 0).
Resultado: potencia a s = 1 0.02, falsas alarmas 0/60, certificados en datos reales 3/12 (ViT-T, ViT-B, ViT-L; ViT-S y DINOv2-L
quedan en z −1.7). **Regla no superada → apéndice tal como se corrió** (Tabla B37, columnas de la derecha); el marco de registro no cambia.

**Decisión del panel.** Figura 4 = (a) real vs estrellas (sin cambio) + (b) z frente a la intensidad del implante, colores de familia, variante tight a
trazos; el barrido sintético pasa al apéndice A.5 como `fig_depth_power_app.pdf` (pie propio). Alturas: Fig. 4 1.3 in, Fig. 3 1.05 in, Fig. 2 1.15 in,
Fig. 5 0.6\linewidth; Tabla 1 a scriptsize con pie abreviado. Recortes de prosa en §1–§7 para volver a la p. 9 (todos listados en el diff de
`phaseD_main_body.tex.tmpl`; ningún número ni afirmación cambia).

**Frases añadidas o cambiadas.** §4.4 en dos párrafos: "Depth is certified on ImageNet and not on CIFAR-100" (los cuatro certificados; "the other
eight are not detected, which is not the same as not hierarchical"; CIFAR-100 fuera del régimen; estrella isotrópica retirada) y nuevo "On real
clouds the test is conservative and weak" (implante sobre nubes reales; "none of the sixty zero-strength runs hierarchical"; "detected in none of the
twelve backbones"; cociente 1.3–3.9 frente a "ratios below one" del barrido sintético; con los offsets encogidos "every backbone is detected";
"conservative and weak at ImageNet's noise level: its positive verdicts stand, and a backbone it does not certify is not thereby flat"; control
entrenado "inconclusive, as Appendix A.5 reports"). Se retira del cuerpo la afirmación de régimen validado con la potencia sintética (queda en A.5
y B35). Abstract, §1, §7: "no additional depth" → "no detected depth"; abstract y §1: "The depth test, conservative and weak at ImageNet's noise
level, certifies…"; MERU: "neither model is detected as more hierarchical than a matched star… neither creates the clustering nor adds detectable
depth"; §6.2: "a positive verdict certifies and a negative one says nothing"; Limitación (i): "conservative and weak at ImageNet's noise level, leaves
hierarchy within the clusters untested, and its one trained positive control is inconclusive". R10 al apéndice A.5, un párrafo ("A trained control,
inconclusive": -3.62 / -2.33 / -2.77) y Tabla B39. R11: nuevo párrafo §4.2 "The count survives resampling" (42–47 de 72;
17–18 de 24; puntero A.14) y cláusula en el pie de la Tabla 1; Tabla B38. B3: §6.1 dividido, nuevo párrafo "The excess is small in absolute
terms and large against the null" (1 a 36 veces la s.d. del null en ImageNet). B4: frase del criterio en §5.1 (por dataset, ciego a los acuerdos, no
transferible; DBpedia/ImageNet). B5: glosa de los dos nulls en §3 y en el pie de la tabla A.1. B6: tesis literal ×3 (abstract, caja, §7) y forma
corta ×2 (final de §1 y de §5): "Clustered, occasionally hierarchical, moderately shared: not one common tree, and no license for curvature."

**Verificación.** Sweep: checks nuevos (R9/R9b/R10/R11 contra el memo; frases de R9b, R11, B3, B5 y el "no detected depth"; tesis 3+2; el listado
de fallos ahora imprime todos los checks) → **133/133 PASS**. Compilación: 36 páginas, 0 warnings, texto principal termina en la p. 9 con el hueco
de 1.2 in (la Ética abre la p. 10). `ICLR2027/REVIEWER_CHECKLIST_positive_control.md` con números de línea del PDF (`make_reviewer_checklist_pc.py`).
Memo actualizado con R9b y R12: `ICLR2027/MEMO_positive_control.md`.

## 23c. Tres frases en §4.4 (sin otro cambio)

(1) Tras "The other eight are not detected, which is not the same as not hierarchical": "For the four certified backbones the same clusters with
random hubs raise no alarm in any zero-strength run, 0 of 20, so the verdict comes from their real hub arrangement, which the test reads as deeper
than the implanted two-level tree, as Appendix A.5 shows. The certified set shifts with the choice of frame, as Table B37 shows." La fracción (0 de
20 = 4 backbones × 5 semillas a s = 0) y la comparación z real < z medio a s = 1 (ViT-S −2.36 vs −1.22; ViT-B −3.62 vs +0.37; ViT-L −4.04 vs −1.37;
DINOv2-L −2.39 vs +0.20) salen de `expR64b_wn30.csv` / `_summary.csv`. Respecto al texto del brief, el punto y coma pasa a punto y los dos punteros se
escriben "as Appendix A.5 shows" / "as Table B37 shows" para cumplir las reglas de prosa (sin punto y coma en §4, un paréntesis por párrafo).
(2) Tras "once the real offsets are shrunk into that range every backbone is detected": ", and two contrastive backbones also fire at zero strength,
so the shrunk variant is a diagnostic of power, not a substitute test" (CLIP-L y SigLIP-B, 2/2 semillas a s = 0 en la variante tight).
Sweep: dos checks nuevos (fracción exacta y z real < z implantado para los cuatro; los dos backbones que disparan a s = 0 son contrastivos) →
**135/135 PASS**. Compilación: texto principal termina en la p. 9 sin recortar §5.7; 36 páginas, 0 warnings. PDF exportado.

## 24a. Final pass — memo de la Fase A (sin prosa; `ICLR2027/MEMO_final_pass.md`, `rebuttal/results/final_pass_memo.json`)

**A1 — estrella con hubs Haar** (`expR69_depth_haarhubs.py`; `calibrated_delta.matched_star(variant="aniso_haarhubs")`: hubs = remuestreo Haar de los 30
hubs reales, clusters anisotrópicos Haar como antes; 12 backbones × {estrella gaussiana, estrella Haar} sobre las nubes reales + estrella Haar sobre
los implantes R9b a s = 0 y s = 1, 5 semillas cada uno; rango r_star = semillas de estrella con exceso B ≤ real, p = (1 + r)/11, resolución 1/11).
Certificados (z ≤ −2) con estrella gaussiana: ViT-S, ViT-B, ViT-L, DINOv2-L; con estrella Haar: ViT-S, ViT-B, ViT-L, DINOv2-L; intersección
= los mismos cuatro. z Haar: ViT-S -2.04, ViT-B -3.01, ViT-L -3.73, DINOv2-L -2.21. Rango r_star = 0/10 (p = 0.091) en los cuatro certificados y también en
4 de los ocho no certificados (el rango no descuenta el ruido de la lectura real). Implantes bajo la estrella Haar: falsas alarmas
0/60 a s = 0; potencia 1/60 a s = 1.

**A2 — ViTs supervisados solo con etiquetas hoja de IN-1k** (`expR70_inet1k_supervised.py`; extracción con el protocolo del caché del censo, 100
img/clase; `deit_base_patch16_224.fb_in1k` y `vit_base_patch16_224.augreg_in1k`): DeiT-B: exceso IN -0.0073 (r 200), C100 -0.0257 (r 200), z -1.26/-1.21, ρ_WN +0.079, ARI C100 máx 0.56; ViT-B augreg IN-1k: exceso IN -0.0070 (r 199), C100 -0.0117 (r 187), z -3.03/-2.65, ρ_WN +0.516, ARI C100 máx 0.48; ViT-B: exceso IN -0.0162 (r 200), C100 +0.0057 (r 43), z -3.62/-3.01, ρ_WN +0.480, ARI C100 máx 0.61.

**A3 — correlaciones del corolario sobre la lectura calibrada** (`expR68_corollary_excess.py`; Pearson r con IC95 Fisher-z, n = 10; demeaned por
familia como en B12). NC H−R sobre el exceso, IC sin 0: imagenet sí, cifar100 sí, cifar10 sí, dtd sí (r = -0.73 ImageNet, -0.80 CIFAR-100,
-0.67 CIFAR-10, -0.76 DTD; crudo -0.84 ImageNet); FS H−R sobre el exceso: imagenet no, cifar100 sí, cifar10 sí, dtd sí; el z de profundidad no predice
ninguna ganancia (ningún IC excluye 0). Figura 11 regenerada sobre el exceso.

**A4 — radios de MERU** (`expR71_meru_radii.py`): c = 0.100; ‖x‖·√c mediana 0.258–0.279, p95 0.263–0.287; Lorentz/Euclídeo
0.9972–0.9991 (centroides 0.9988–0.9997).

**A5 — barrido de presupuesto bajo el registro** (`expR72_budget_record.py`; 9 celdas × 6 presupuestos, Haar × p99.9 × 200 en cada uno; resuelve la
discrepancia de B15, que usaba supremo × gaussiana): deriva máxima con presupuesto ≥ 10⁵: ImageNet 0.0007 = 0.38 s.d., CIFAR-100
0.0006 = 0.10 s.d., DTD 0.0003 = 0.06 s.d.; signos estables en todo presupuesto; "budget-stable" en ImageNet: **True**.

**A6 — efecto normalizado exceso/δ_null**: ImageNet -33 % a -5 % (genuinas -33 % a -11 %); nivel de muestra
-39 % a +7 %; DINOv2-G/CIFAR-100 -57 %. Columnas añadidas en B14/B17/B23 y rejilla nueva B40 (`gen_review_tables.py`).

Sweep con checks A1–A6: **142/142 PASS**. Sin ediciones de prosa.

## 24b. Final pass — Fase B: memo al texto, simplificación, abstract, formato. Congelado el 17 de septiembre de 2026.

Brief: "Final pass — Phase B: memo into text, simplification, abstract, format. Freeze after this." (+ instrucción posterior del autor:
quitar la caja de la página 1, tesis literal solo en el abstract y en §7, sin formas cortas, hueco de la Figura 1 a 1.4 in).
El cuerpo sale entero de `rebuttal/scripts/phaseE_paper.tex.tmpl` rellenado por `phaseE_final.py` (todo número de la prosa viene de
un CSV; `rebuttal/results/phaseE_fills.json` guarda los valores); el apéndice, de `ICLR2027/iclr2027/gen_appendix_final.py`.

### Decisiones del autor (§1 del brief) llevadas al texto
- Profundidad: conjunto certificado = los cuatro que sobreviven a las dos estrellas (ViT-S/B/L, DINOv2-L); la estrella con hubs Haar es la
  confirmación (§4.4 "A spectrum-matched star confirms the four": z de -2.0 a -3.7, 0 of 60 falsas alarmas a s = 0, one implant in sixty a s = 1,
  cláusula del rango con resolución 1/11: el rango no separa certificados de no certificados, z sí). La gaussiana sigue siendo la original (Tabla 6).
- ViTs con etiquetas hoja: "supervision on leaf labels alone can produce this depth and need not" (§4.4) y "leaf-label supervision does not
  guarantee the alignment" (§5.4: augreg IN-1k alineado como los i21k, DeiT-B casi sin alineación pero recupera las superclases de CIFAR-100);
  "supervision deepens the resemblance on average across the supervised backbones tested" (§5.2); Limitación (iv) actualizada.
- Corolario: §6 gana el párrafo "The calibrated reading predicts the zero-cost gain" (NC en los cuatro conjuntos, FS en tres, la profundidad no
  predice nada; Tabla 13 con las correlaciones crudas y calibradas; Figura 11 sobre el exceso).
- MERU: lead-in "Imposing the geometry does not create detected depth."; frase del régimen casi plano con el cociente Lorentz/Euclídeo 0.997
  (el radio × √c va al pie de la Tabla 6).
- "Budget-stable" citado al barrido bajo el registro (§3, Tabla 4: deriva ≤ 0.38 s.d. en ImageNet).
- Tamaños de efecto: "small" sustituido por la fracción del nulo (§4.1: 7–39 % a nivel de muestra, DINOv2-G/CIFAR-100 nombrado;
  §6.1: 5–33 % en ImageNet); columna exceso/δ_null en las Tablas 3, 5 y 11.

### Abstract (§2 del brief)
- Sustituido literalmente. Cada número verificado contra su fichero por el sweep ("final: abstract numbers traced"): 12/6/16, 200,
  "at most 40 %" (máximo a nivel de muestra 39.4 %), 49/72, 4/12 = los mismos cuatro bajo ambas estrellas, 0 falsas alarmas con hubs
  aleatorizados, "a gap of about a third" = mediana de la fracción de acuerdo intra-bloque ausente sobre las configuraciones admisibles de
  ImageNet × tres medidas = 0.34 (comprobado en [0.25, 0.42]; `rebuttal/results/final_pass_island_gap.json`).
- Caja de la página 1 eliminada; la tesis aparece literal dos veces (última frase del abstract, primer párrafo de §7); formas cortas de
  final de §1 y §5 eliminadas; ambas secciones terminan en su última afirmación. Hueco de la Figura 1: 1.4 in.

### Ediciones de texto (§3 del brief)
- §5.2 lead-in del autor "The categorical island is an artifact; a moderate gap remains." escrito con coma en lugar de punto y coma (regla
  "sin punto y coma en §4–§6"); 0.38 frente a 0.48 en prosa; 0.13–0.17 frente a 0.50–0.55 y la mediana en el pie de la Tabla 9; tripletes
  cierran solo bajo cosine-average; "vanishes"/"closes entirely" eliminados (pie de la Figura 5 incluido); "self-supervised family as an
  island" → "the DINOv2 family".
- §5.6: un solo censo (Tabla 11); la Tabla 13 antigua (extracción original, 3 réplicas) eliminada y las columnas "original extraction"
  de la tabla del registro también; pie de la tabla de plantillas reescrito desde el registro; GPT-2 S genuino al margen bajo el registro
  y cambia bajo el supremo y bajo coseno, M no genuino bajo el registro, L/XL bajo toda construcción, Pythia a toda escala. §1 P2:
  "small GPT-2 models sit at their matched null" → "GPT-2 M sits at its matched null" (S es genuino bajo el registro).
- §3: "pre-registered" → "pre-specified" (no hay registro público fechado); §2: "trees < hyperbolic < spherical for hyperbolic regions
  of radius four and above (Table 2)"; glosa de los dos nulos presente en §3 y en A.2. A.3 (frase repetida) desaparece con el apéndice nuevo;
  Figura 12 (ilustración antigua) eliminada con su puntero; Tabla 1 con negrita en las celdas de signo positivo.

### Simplificación (§4 del brief)
- Puentes: quedan tres, al final de §3 ("With the instrument in place, we read the premise where it is read."), §4 ("...is the question we
  turn to next.") y §5 ("What a practitioner can collect from this local agreement is the subject of the next section.", penúltima frase; la
  última es la afirmación); el resto de párrafos termina en su afirmación. Check "final: exactly three bridges".
- Números en la prosa de §1–§7: 17 grupos (≤ 25), ≤ 2 por párrafo, ninguno entre paréntesis; recuentos pequeños en palabras ("none of the
  sixty", "all but one of twelve", "between one and four times"); todo número retirado vive en una tabla o pie (check de conservación
  respecto al cuerpo pre-prosa sigue en PASS).
- Apéndice: de 44 tablas a 14 (+ índice de procedencia), una por pregunta, numeradas en orden de primera cita del texto principal
  (Tabla 2 calibración → 3 censo → 4 robustez → 5 nivel de muestra → 6 profundidad → 7 potencia → 8 intervenciones/ORC → 9 mapa de árboles →
  10 taxonomía → 11 texto → 12 acuerdo local → 13 corolario → 14 ξ → 15 panel → 16 procedencia). Mapa antigua → nueva:
  calibration+B25 → 2; B20+B21+B40+census_extra(sup IN)+B32+B24 → 3; B15+expR72+B36+B27+B38+B3 → 4; B17 → 5; B34+expR69+marco
  equilibrado+expR70+B30+expR71+B39 → 6; B35+B37+implantes expR69 → 7; intervenciones (analysis4/e1/e5/arch_matched, antes solo figura y prosa)
  +B6 → 8; B7+B33 → 9; alineación exp3 (antes columna de Tabla 1 y prosa)+B11+grupos exp1+ρ imagen única exp8_p6+expR70+A9+A8 → 10;
  B23+A7+B14+B28+coseno texto → 11 (B1 eliminada por el brief); B31 → 12; A2+A3+A4+A5+exp2b+A6+B5+B12+expR68+census_extra(best−R)+B8+B9+B4 → 13;
  A10+B19+B26 → 14; A0 (+ bloque de controles) → 15. La tabla de nulos (antigua Tabla 2, sin números) es prosa en A.2.
  Conservación: los 992 tokens decimales distintos de las 44 tablas antiguas (instantánea en `rebuttal/results/final_pass_old_appendix/`,
  B1 y las columnas "original extraction" exceptuadas) aparecen en las tablas nuevas o en el texto (check "final: every decimal token...").
- Recortes: DBpedia fundido en §5.4 (dos frases finales), §5.7 en tres frases, §6 "Measure before imposing" en cuatro, párrafo de colapso
  neural sin su última frase, §5.6 en siete líneas; además frases redundantes en §1, §2, §3, §4.1, §4.2, §4.8, §5.1, §5.5, §7 y pies de
  Tabla 1 y Figura 4 para cerrar el presupuesto de página (nunca encogiendo figuras).

### Formato (§5 del brief)
- Figura 2 ≥ 1.3 in (1.51), Figura 3 ≥ 1.4 in (1.56), Figura 4 ≥ 1.5 in (1.53) con hueco opcional `figures/fig4_schematic.pdf`
  (5.5 × 0.9 in) encima del panel (a), sin marcador si falta; ticks a 7 pt, 2–3 marcas en y con dos decimales y sin "−0.00"; leyendas
  fuera de los ejes; ningún texto de figura por debajo de 7 pt. Tabla 1 en `\small` sin las columnas de nivel de muestra (Tabla 5).
- `\raggedbottom` global (los flotantes [t]/[H] dejaban páginas cortas: era la única fuente de avisos); tres URLs de NeurIPS irrompibles
  retiradas de `references.bib` y la entrada Gromov1987 pasa a `@incollection` (aviso de BibTeX). Compilación: 0 avisos, 38 páginas, texto
  principal acaba en la página 9 (Ethics abre la 9/10). Páginas rasterizadas a 100 dpi en `ICLR2027/qa_pages/`.
- §4 y §6 con la apertura de dos frases en forma de tensión; pies = takeaway en negrita + qué se dibuja; AI Use Statement en la forma
  de la plantilla ICLR 2027 (pendiente de que el autor coteje las categorías de la política); Reproducibility con `TODO(author)`.

### Verificación (§6 del brief)
- `sweep_freeze.py`: bloque `final_pass_prose_checks` (abstract literal y trazado, tesis ×2, puentes, números, frases ≤ 9, aperturas,
  vocabulario, pies, tamaños de figura, Tabla 1, statements, 14 tablas en orden de cita, etiquetas citadas, referencias cruzadas,
  Tabla 13 eliminada, conservación de la consolidación, nota de congelación + qa_pages); checks antiguos reanclados a las tablas nuevas.
  Generadores antiguos en `ICLR2027/iclr2027/legacy_generators/`. Índice de procedencia regenerado (`% prov:` en cada tabla).
- `REVIEWER_CHECKLIST_third.md` (con números de línea del PDF), `SUMMARY_plain.md` regenerado, `TODO_author.md` y `CODE_MAP.md` actualizados.

## 24c. Post-freeze (2026-09-17, ordenado por el autor): Figura 2 en dos paneles y paleta nueva. Ningún número cambia.

- **Figura 2** (`figures/make_fig_overview.py`): panel (c) eliminado (su contenido es la Figura 5). (a) los 12 backbones de ImageNet por
  dimensión d, lectura cruda (relleno, color de familia) y media del nulo emparejado (hueco, gris) unidas por un segmento vertical cuya
  longitud es el exceso (los backbones con la misma d se reparten en ±7 % de d para que todos los segmentos se vean); (b) las dos celdas con
  la misma lectura cruda de `phaseC_fig2b.json` (ViT-T en imágenes de DTD, SigLIP-B en centroides de CIFAR-10): dos puntos rellenos a la
  misma altura, sus nulos huecos a alturas muy distintas, segmentos entre cada par. Leyenda "hollow: a structureless cloud of the same
  shape". Misma altura total (1.3 in). Pie nuevo literal del autor. Datos: `exp1_delta_controls.csv`, `expR52_census_haar_p999_200.csv`,
  `expR62_samplelevel_record.csv`.
- **§3**: dos citas nuevas: "Figure 2a shows the same drift on the twelve ImageNet backbones: the null of each moves with its dimension, so
  the gap, not the level, is the reading." (tras la frase de las gaussianas) y "Figure 2b shows two cells with the same raw reading and
  opposite verdicts." (final de "The reading is an excess over a matched null").
- **Paleta** (`figures/palette.py`): supervised #2A6F97, self-supervised #C8553D, contrastive #6C8B3C, causal LM #7A4E9A, embedder
  #A67C52; grises de nulos y estrellas sin cambio; `make_figs.py` toma ahora los colores de familia de la paleta en vez de su copia local.
  Todas las figuras del paper regeneradas con los mismos scripts (Figs. 2–5, 6–11 del apéndice); fuentes STIX sin cambio.
- Sweep: el check del pie de la Figura 2 acepta la forma "takeaway en negrita + (a)/(b)" y la prueba de `fig2b` mira las celdas nombradas
  en el pie nuevo; todo lo demás igual (PASS completo). Compilación: 0 avisos, texto principal en la página 9; `qa_pages/` regenerado.
- 24c, retoque tipográfico (petición del autor): en la Figura 2(b) las etiquetas de las dos celdas van en tres líneas a 8 pt
  ("ViT-T / DTD / images", "SigLIP-B / CIFAR-10 / centroids"), el panel es más ancho (proporción 1.35:1) y hay más aire entre las dos
  posiciones. Recompilado: 0 avisos, página 9; `qa_pages/` regenerado.
- 24c, retoque tipográfico (petición del autor): en la Figura 2 los anillos superiores quedaban recortados por el borde del eje; margen
  vertical añadido en ambos paneles y marcadores sin recorte (`clip_on=False`). Además, el sync de Overleaf `overleaf-2026-09-17-1419`
  (commit 366218d) había introducido "withinynull" en el abstract y borrado los comentarios `% fichero.csv` de dos párrafos de §1;
  el cuerpo se ha regenerado desde `phaseE_paper.tex.tmpl` (fuente del texto), lo que restaura el abstract literal y la procedencia.
  Cualquier edición del cuerpo debe hacerse en la plantilla y re-aplicarse con `phaseE_final.py`.

## 24d. Post-freeze (2026-09-18, ordenado por el autor): Figura 3 como mapa de calor del censo de registro. Ningún número cambia.

- `figures/make_figs.py`: la Figura 3 es un único mapa de calor 12 × 6 (backbones agrupados por familia con barra de color de familia a
  la izquierda; datasets con ImageNet primero), color = exceso con mapa divergente centrado en cero (blanco en 0, azul más oscuro cuanto
  más negativo, rojo claro en positivo), punto relleno en las celdas genuinas y el exceso impreso a 7 pt con dos decimales (texto blanco
  sobre las celdas oscuras); barra de color a la derecha. Mismo ancho; alto 1.80 in en PDF (el que piden las etiquetas a 7 pt). Panel (b)
  (exceso frente a escala) eliminado: la lectura por escala se hace bajando por las filas de DINOv2. Datos: `expR52_census_haar_p999_200.csv`.
- Pie literal del autor. Frases de §4 que leían el panel (b): "Figure 3 and Table 1 give the census of record cell by cell." y
  "Figure 3 shows that scale does not deepen the excess in general, with one exception: down the DINOv2 rows the excess deepens with size
  on CIFAR-10 and, weakly, on CIFAR-100."
- Presupuesto de página (la figura crece 0.24 in): tres recortes sin números: §1 P3 pierde "and comparing trees across models needs a
  calibration of its own" (ya en §2), Limitación (ii) "The census covers class-centroid geometry only.", §6.4 "Changing the readout metric
  costs nothing"; el pie de la Tabla 1 pierde la fórmula del p (está en el pie de la Tabla 3) y acorta los punteros.
- Sweep: el check de pies exige takeaway en negrita seguido de descripción (la palabra "Plotted:" ya no es obligatoria); resto igual.
  Compilación: 0 avisos, texto principal en la página 9; `qa_pages/` regenerado.

## 24e. Post-freeze (2026-09-18, ordenado por el autor): Figura 3 como diagrama de puntos horizontal. Ningún número cambia.

- `figures/make_figs.py`: seis paneles estrechos, uno por dataset (ImageNet primero), eje y compartido con los 12 backbones agrupados
  por familia (color de familia en el marcador, separador gris fino entre familias), eje x común de −0.14 a +0.03 con línea vertical en
  cero; un marcador por celda en su exceso, relleno si la celda es genuina y hueco si no; sin texto dentro de los paneles salvo el título
  del dataset; etiquetas a 7 pt; mismo ancho; alto 1.87 in en PDF. Sustituye al mapa de calor de 24d (sin números en las celdas).
  Datos: `expR52_census_haar_p999_200.csv`. Pie literal del autor; las dos frases de §4 siguen leyendo de la figura ("cell by cell",
  "down the DINOv2 rows"). Compilación: 0 avisos, texto principal en la página 9; `qa_pages/` regenerado; sweep sin cambios.
- 24e, tres cambios (petición del autor, 2026-09-18): (1) leyenda de una línea bajo los seis paneles, al estilo de la Figura 2
  ("filled: genuine cell; hollow: not genuine" + los tres colores de familia con su nombre); (2) marca intermedia en −0.05 en el panel de
  ImageNet, algo más ancho (proporción 1.35) para que quepan las tres etiquetas a 7 pt, de modo que los excesos pequeños se lean como
  −0.01 a −0.02 y no como cero; (3) §4 "Clustered structure is the rule": tras "Table 1 gives the census of record" va "Figure 3 shows it:
  almost every marker sits left of zero, most are filled, the flat datasets scatter around it, and the DINOv2 rows reach furthest left on
  CIFAR-10." (la frase "Almost every cell is below its matched null" desaparece por redundante; 49/72 y 18/24 se mantienen); en "Scale
  deepens the excess for one family only" la escala de DINOv2 se cita a la figura: "the DINOv2 rows of Figure 3 reach further left with
  size on CIFAR-10 and, weakly, on CIFAR-100"; no queda ninguna referencia al antiguo panel (b) ni a "per-dataset excess vs scale".
  Alto de la figura 1.98 in en PDF. Compilación: 0 avisos, página 9; `qa_pages/` regenerado; sweep sin cambios.

## 24f. Post-freeze (2026-09-18, ordenado por el autor): Figura 2 al estilo de las figuras de dispersión de Huh et al. (2024) y Gröger et al. (2026). Ningún número cambia.

- `figures/make_fig_overview.py`: fondo blanco, sin rejilla, puntos etiquetados. (a) dos tercios del ancho: x = dimensión d en eje
  logarítmico, y = δ_norm; para cada uno de los 12 backbones de ImageNet un marcador relleno en su color de familia con el nombre del
  modelo al lado (7 pt, desplazamientos manuales por modelo para que ninguna etiqueta cruce marcadores ni segmentos; los backbones que
  comparten d se reparten unos puntos porcentuales en x), su media de nulo emparejado como marcador hueco del mismo color unido por un
  segmento vertical fino, y la curva de la gaussiana iid de la tabla de calibración (`exp1_delta_controls.csv`, variante gauss, estadístico
  supremo, como en la Tabla 2) como línea gris discontinua rotulada "structureless cloud, isotropic"; título "the null moves with the
  dimension; the gap is the reading"; leyenda dentro del panel ("filled: raw reading", "hollow: matched null"). (b) un tercio: las dos
  celdas ViT-T/DTD images y SigLIP-B/CIFAR-10 centroids en el mismo lenguaje, nombres junto a los puntos; título "same reading, opposite
  verdict". Colores de la paleta nueva; misma altura de figura (1.3 in de lienzo, 1.49 in en PDF). Datos: `exp1_delta_controls.csv`,
  `expR52_census_haar_p999_200.csv`, `expR62_samplelevel_record.csv`. Nota: la curva gaussiana es el estadístico supremo de la tabla de
  calibración mientras los marcadores son δ̂_99.9; se dibuja como referencia isotrópica, no como nulo de los puntos. Pie sin cambio.
  Compilación: 0 avisos, página 9; `qa_pages/` regenerado; sweep sin cambios (160/160).

## 25. Versión paralela v2 (2026-09-18): el paper reestilizado a la manera de Gröger et al. (2026) y Huh et al. (2024). El fichero de envío sigue siendo `main_iclr2027.tex`.

- Nuevo `ICLR2027/iclr2027/main_iclr2027_v2.tex`, generado por `rebuttal/scripts/phaseE_v2.py` desde `phaseE_paper_v2.tex.tmpl` con
  los mismos rellenos que v1 (`phaseE_fills.json`), el preámbulo de v1 (+ `amsthm`, `tcolorbox`, `caption`), las mismas figuras,
  tablas y bibliografía, y los statements y el apéndice copiados literalmente de la plantilla v1. `main_iclr2027.tex` no se toca.
- Esqueleto: 1 Introducción (v1 literal) · 2 Related Work (un párrafo) · 3 Background and Problem Setup (caja "Hypothesis under test";
  3.1 Gromov δ con Def. 1/Eq. 1; 3.2 Estimation con Def. 2/Eq. 2 y la estabilidad de presupuesto; 3.3 Objects and notation con tabla
  de notación sin numerar) · 4 The Instrument (4.1 nulo espectral, Def. 3; 4.2 exceso y rango, Def. 4/Eq. 3, Eq. 4, Def. 5/Eq. 5;
  4.3 test de profundidad, Def. 6, Def. 7/Eq. 6, régimen de validez medido; 4.4 proyección; caja "Definitions at a glance") ·
  5 Experimental Setup (modelos, datasets, extracción, réplicas, semillas, presupuestos, los dos niveles) · 6 Results (cajas Finding 1–4,
  Tabla 1, Figuras 3–5; 6.1 nivel de muestra, 6.2 nivel de clase, 6.3 profundidad, 6.4 whose tree, 6.5 acuerdo local) ·
  7 Implications (v1 §6 con la regla de curvatura de Khrulkov como Eq. 7) · 8 Discussion and Limitations (v1 §7).
- Prosa de v1 dentro de cada subsección; frases borradas solo donde una definición o ecuación dice lo mismo (lista en `V2_DIFF.md`);
  tres puentes; 17 grupos numéricos en §1–§8; vocabulario fijo; ningún número que no esté en v1.
- Compilación: 0 avisos, 41 páginas, texto principal hasta la página 12 (2.5 páginas sobre las 9 de v1; propuestas de recorte en
  `V2_DIFF.md`, nada recortado). `main_iclr2027_v2.pdf`, `qa_pages_v2/` (páginas 1–13), `SWEEP_REPORT_v1_v2.md`.
- Sweep: bloque `v2_checks` (abstract/statements/apéndice idénticos a v1, ningún número nuevo, mismos comentarios de procedencia,
  mismas figuras y tablas, tesis ×2, esqueleto 8/12 con 7 definiciones y 7 ecuaciones y las cajas, reglas de prosa de v1 sobre el
  esqueleto v2, tres puentes, referencias cruzadas, orden de citación de las tablas compartidas): 170/170 con v1.

## 26. Versión v3 (2026-09-18): reescritura en estructura clásica desde las listas de afirmaciones. El fichero de envío sigue siendo `main_iclr2027.tex`.

- Nuevo `ICLR2027/iclr2027/main_iclr2027_v3.tex`, generado por `rebuttal/scripts/phaseE_v3.py` desde `phaseE_paper_v3.tex.tmpl`.
  Literal de `main_local.tex` (fichero del autor, commit a30a880): abstract, §1 con la Figura 1 (recorte del autor), §2 y los párrafos
  "Gromov δ" (Eq. 1, emparejamientos) y "Estimation and normalization" (Eq. 2, δ_norm). El resto está escrito desde las listas de
  afirmaciones del brief con el vocabulario fijo; ningún párrafo de v1 pegado.
- Esqueleto: 1 Introducción · 2 Related Work · 3 Methodology (3.1 Gromov δ con Def. 1; 3.2 Estimation con Def. 2, Lema 1 (cota de
  rango), Corolario 1 (confound de dimensión, Beyer et al. 1999 y Aggarwal et al. 2001, entradas nuevas en `references.bib`), Remark
  (espectro y estadístico), calibración; 3.3 nulo Haar (Def. 3), exceso (Def. 4, Eq. 3), rango y p (Eq. 4, K = 200), genuino (Def. 5,
  Eq. 5), Figura 2; 3.4 test de profundidad (Def. 6, Def. 7, Eq. 6, star caveat); 3.5 comparación de árboles; 3.6 proyección y tareas) ·
  4 Experimental Setup · 5 Results (5.1–5.5, lead-ins en negrita, ≤ 2 números por párrafo y ≤ 1 por frase) · 6 Implications (Eq. 7) ·
  7 Conclusion and Limitations (tesis literal, limitaciones i–viii). Statements: Reproducibility (repositorio anonimizado, `tool/`),
  Ethics, AI Use en la forma ICLR 2027 con los cuatro usos del brief. Apéndice A: demostraciones del Lema 1 y del Corolario 1; después
  las catorce tablas por pregunta, reordenadas al orden de primera cita de v3 (Tabla 1 censo; 2 robustez, 3 calibración, 4 censo bajo
  las cuatro construcciones, 5 panel, 6 nivel de muestra, 7 intervenciones, 8 profundidad, 9 potencia, 10 mapa, 11 taxonomía, 12 texto,
  13 acuerdo local, 14 corolario, 15 ξ, 16 procedencia). Definiciones, lema y corolario con `amsthm`, sin cajas ni texto en color.
- Todo número de la prosa es un relleno leído de su CSV (`rebuttal/results/phaseE_v3_fills.json`); el sweep comprueba que ningún
  número de v3 falta en v1 o en los rellenos, que cada fichero citado en un comentario `%` existe, y recalcula los rellenos clave.
  Dos cifras del brief difieren del fichero y se escriben con el valor del fichero: "excess 3–8 null spreads" → 1 a 36 (|exceso|/s.d. del
  nulo, ImageNet) y "sibling agreement 0.88" (DBpedia) → 0.89; "power 0.05" y "star −0.11" se escriben como 0.05 y −0.115 (valores exactos).
- Figuras (§8 del brief): `style.mplstyle` con rejilla gris discontinua detrás de los ejes y bordes gris claro; paleta de familia sin
  cambio; bandas de nulo: Figura 2a (media del nulo ± 2 s.d. a lo largo de la dimensión), Figura 3 (± 2 s.d. medianas del nulo alrededor
  de cero en cada panel), Figura 4a (dispersión de la estrella como banda); mapa de calor de la Figura 5 sin rejilla. Todas las figuras
  regeneradas con los mismos scripts; al ser compartidas, v1 y v2 las heredan (v1 recompilada: 0 avisos, página 9, PDF y `qa_pages/`
  reexportados; el `.tex` de v1 no cambia).
- Presupuesto: tras los tres recortes prescritos (§5.5 a tres frases, §6 a tres párrafos, la tabla de modelos fuera del texto principal
  hacia la Tabla 5) el texto principal acaba en la página 11 (§3 ocupa 2.3 páginas: siete definiciones, siete ecuaciones, lema, corolario
  y los dos párrafos literales). No se ha recortado nada más; propuestas en el informe. Compilación: 0 avisos, 41 páginas.
- Entregables: `main_iclr2027_v3.pdf`, `qa_pages_v3/` (páginas 1–11), `V3_OUTLINE.md` (lista de afirmaciones con la línea de cada una),
  `SWEEP_REPORT_v1_v3.md` (sweep 183/183 sobre v1, v2 y v3), bloque `v3_checks` en `sweep_freeze.py`, `make_v3_outline.py`.
- 26, segunda ronda (petición del autor): (1) presupuesto: los tres recortes prescritos (§5.5 a tres frases, §6 a tres párrafos, tabla de
  modelos al apéndice) ya estaban aplicados y el texto principal sigue acabando en la página 11 (las referencias empiezan en la 11);
  no se ha recortado nada más, la decisión queda en `TODO_author.md`. (2) Figura 4 de v3 redibujada en el estilo de la Figura 3 en un
  fichero propio, `figures/fig_depth_test.pdf` (`make_fig_depth.py`; v1 y v2 conservan `fig_depth_main.pdf` y sus pies): (a) diagrama
  de puntos horizontal del z de profundidad por backbone bajo la estrella de registro (expR56, ImageNet K = 30), familias agrupadas,
  línea vertical en z = −2, marcador relleno cuando certificado; (b) fracción de ejecuciones con z ≤ −2 frente a la fuerza del implante
  en las nubes reales (expR64b, continua) y en la variante de offsets encogidos (expR64b `rand6_t06`, discontinua), con la tasa de falsas
  alarmas en s = 0 anotada. Pie literal del autor; las dos frases de §5.3 que leían la figura actualizadas. Etiquetas de tick en negro
  (matplotlib 3.3 no admite `labelcolor`); todas las figuras regeneradas y v1/v2/v3 recompiladas. Sweep 183/183.
- 26, tercera ronda (petición del autor): (1) §3.2: Lema 1 y Corolario 1 sustituidos por la Proposición 1 en dos partes con el mismo
  contenido, (a) cota de rango y (b) confound de dimensión (Beyer et al. 1999; Aggarwal et al. 2001), remark sin cambio, una sola
  demostración en el apéndice A; citada en la calibración de §3.2 y donde §3.3 discute que el nulo se mueve con la dimensión; las
  siete definiciones siguen en línea y sin marco. Sweep: el check del esqueleto exige la proposición con (a)/(b), su cita al menos dos
  veces y una única demostración. (2) Presupuesto: los tres recortes prescritos siguen aplicados; las referencias empiezan en la
  página 11 (la Proposición ahorra dos líneas respecto al lema y el corolario); no se ha recortado nada más. (3) La Figura 4 de v3 ya
  cumple la especificación de esta ronda (misma que la anterior); sin cambios. Compilación: 0 avisos, 41 páginas; sweep 183/183 sobre
  v1, v2 y v3; `qa_pages_v3/` y `V3_OUTLINE.md` regenerados.

## 27. Versión final (2026-09-18, brief "Final version — plain, short, nine pages"): `main_iclr2027_final.tex/.pdf`. El fichero congelado `main_iclr2027.tex` no cambia.

- Generación: `rebuttal/scripts/phaseE_paper_final.tex.tmpl` + `phaseE_submission.py` (ejecutado con `FINAL_CUTS=s55,s6,table,s2`) escriben
  `ICLR2027/iclr2027/main_iclr2027_final.tex`. Partes literales de `main_local.tex` (resumen, §1 con la Figura 1, §2, "Gromov δ",
  "Estimation and normalization") con las cinco ediciones que pide el brief, registradas en `rebuttal/results/final_verbatim_edits.json`:
  "cast the same shadow" → "all read the same"; "Why is the shadow low?" → "Why is the reading low?"; "geometric face of" → "geometric
  counterpart of"; "the intent of the supremum" → "as the supremum does"; la frase puente "What a structureless cloud reads on it is the
  next question." eliminada. Estructura de v3 sin cambios: siete secciones, once subsecciones, siete definiciones en línea sin marco, siete
  ecuaciones, Proposición 1 (a)(b) con su demostración en el Apéndice A, sin cajas.
- Prosa llana (regla del brief, comprobada por el bloque `final_checks` del sweep): media de 16.4 palabras por frase en la prosa no literal
  de §3–§7 (132 frases), ninguna por encima de 35; párrafos de 3 a 6 frases con entrada en negrita; en §5–§6 como máximo un número por
  frase y dos por párrafo, un solo puntero a tabla o figura al final de cada párrafo; en la prosa sólo los siete números de cabecera
  (49 de 72; 18 de 24; 4 de 12; 0 de 60; 7 de 15; 0.48 a 2.5; +0.9 a +1.3 pp); sin metáforas fuera de la Figura 1 y su leyenda, sin
  puentes, sin paréntesis de más de tres palabras, sin cadenas de punto y coma, sin "we note/interestingly/importantly/in plain terms".
- Figuras en lenguaje de barras (`ICLR2027/figures/make_figs_final.py`, ficheros `fig_{overview,excess,depth,treemap}_final`): Figura 2
  pares de barras lectura/nulo por backbone ordenados por dimensión con la referencia gaussiana escalonada a trazos y el par ViT-T/DTD
  frente a SigLIP-B/CIFAR-10; Figura 3 seis paneles estrechos de doce barras horizontales, rellenas si genuinas y rayadas si no, con la
  banda ±2 s.d. del nulo; Figura 4 barras de z con la línea a trazos en −2, cifras sólo sobre las certificadas, y curvas de detección con
  la variante encogida a trazos; Figura 5 dos matrices ARI más grandes, paleta secuencial, rectángulo sobre el bloque DINOv2, leyenda con
  0.03 frente a 0.38 (valores leídos de `rebuttal/results/final_fig5_values.json`). Leyendas = conclusión en negrita + qué se dibuja.
  La Figura 1 es el comando literal del autor (`\IfFileExists`, `fig1_concept.pdf` ausente en el repo → caja de 1.4 in).
- Apéndice: doce tablas citadas + demostraciones + índice de procedencia, en orden de primera cita, como copias flotantes por panel en
  `ICLR2027/iclr2027/appendix_tables/final/` (`[tbp]`, sin `\FloatBarrier`, un flotante por panel; los `[H]` de v1 dejaban media página
  en blanco por parte). Eliminado: ξ (tab_q12), la tabla de intervenciones/ORC (tab_q11), todas las figuras del apéndice y sus
  referencias, el panel de variantes del nulo de la tabla de robustez y el panel (b) del censo (veredicto bajo las cuatro construcciones),
  las filas supremum×Gaussian del panel de presupuesto y el panel (b) de la tabla del corolario (correlaciones del supremo, no citado),
  el barrido de número de clases salvo una fila (DINOv2-L, C=100, ambos modos) y las filas de intervención salvo una por intervención.
  Los comentarios `% prov:` y `% source` se conservan; `tab_q08_robust_final.tex` lo genera `gen_appendix_final.py` (conservación 992/992
  sobre las tablas de v1, intactas); índice `tab_z_provenance_final.tex` con `gen_provenance.py main_iclr2027_final.tex
  tab_z_provenance_final.tex final`. Tabla 1 = `tab_census_final.tex` (mismas filas que v1, leyenda de tres líneas).
- Presupuesto de página: la primera compilación terminaba en la página 11 (referencias en la línea 558). Los cuatro recortes del brief
  (§5.5 a tres frases, §6 en tres párrafos, tabla de modelos en el apéndice, §2 a sus seis primeras frases) liberan unas 25 de las 72
  líneas; el resto se ha obtenido recortando frase a frase la prosa no literal sin eliminar ninguna afirmación de `V3_OUTLINE.md` (las
  frases quitadas repetían una afirmación del mismo párrafo o del vecino; la limitación (viii) sobre ξ se va con el descriptor) y con
  espaciado tipográfico declarado en el preámbulo (saltos de sección/subsección/párrafo 1.2/1.0/0.5 ex frente a 2.0/1.8/1.5 ex del
  estilo, saltos de ecuación 4 pt, entornos de definición y proposición a 3 pt, separación de flotantes 12 pt, salto de leyenda 5 pt);
  fuentes, márgenes, interlineado y tamaño de figuras sin tocar. Resultado: §7 termina en la página 9 (línea 479), los statements empiezan
  en la 9 (línea 480) y las referencias en la 10 (línea 501); 28 páginas, 0 avisos. Apéndice: 16 páginas (demostraciones incluidas)
  frente a las 14 del brief; candidatos en `FINAL_CHECK.md`.
- Statements: AI Use con exactamente los tres usos declarados y la frase de responsabilidad; Reproducibility con el marcador del
  repositorio anónimo (`TODO(author)`); Ethics sin cambios.
- Sweep: `sweep_freeze.py` 207/207 (v1 160, v2 10, v3 13, final 24: partes literales módulo las ediciones registradas, reglas de prosa,
  vocabulario definido en §3, ningún número nuevo, ficheros de procedencia existentes, figuras y leyendas, conjunto y orden del apéndice,
  material eliminado ausente, copias por panel con los mismos números que v1, referencias cruzadas, Tabla 1, statements, preámbulo).
  Escribe `rebuttal/results/final_prose_stats.json`; `make_final_check.py` → `ICLR2027/FINAL_CHECK.md` (página de las referencias,
  estadísticas de longitud de frase por sección, números por párrafo en §5–§6, material eliminado, informe del sweep).
- Entregables: `ICLR2027/main_iclr2027_final.pdf`, `ICLR2027/qa_pages_final/` (100 dpi, 28 páginas), `FINAL_CHECK.md`, `SUMMARY_plain.md`
  regenerado en diez líneas desde el texto final. La exportación de v1 pasa a llamarse `ICLR2027/main_iclr2027_v1.pdf` (las entradas
  anteriores de este registro que dicen `main_iclr2027_final.pdf` se refieren a ese fichero).

## 27b. Versión final (2026-09-18, ordenado por el autor): Figura 4 con leyenda compartida y cifras dentro de las barras; Figura 5 con el panel (c) de tripletes. Ningún número del texto cambia.

- Figura 4 (`fig_depth_final`): la leyenda del panel (a) sale de los ejes a una fila común bajo los dos paneles (con las dos curvas del
  panel (b), como en la Figura 3); la z de las barras certificadas se imprime en blanco dentro de la barra, en vertical porque la barra
  (12 pt de ancho) no admite "−2.4" en horizontal a 8 pt; ninguna cifra queda tapada. Altura 1.9 in (antes 1.75 in) por la fila de leyenda.
- Figura 5 (`fig_treemap_final`): nuevo panel (c), misma altura que las matrices (1.4 in), dos barras por backbone, acuerdo de tripletes
  de hermanos en CIFAR-100 bajo coseno (rellena) y bajo distancia euclídea (rayada), colores de familia, azar 0.5 a trazos, título "where
  the tree lives: sibling triplets, CIFAR-100" en dos líneas; datos de `exp10_local_vs_global.csv` (columnas `c100_sibtrip_c/e`, el fichero
  del que sale la cifra de tripletes de la leyenda de la tabla del censo; no hay tabla por modelo en el apéndice final). La barra de color
  queda entre (b) y (c) con el rótulo "ARI" encima. Leyenda del autor, literal, con los dos valores de ARI como rellenos ({{NAIVE_BIG}},
  {{ARI_BIG}}). Los valores por modelo se guardan en `rebuttal/results/final_fig5_values.json` (`sibtrip_c100`): brecha coseno−euclídea
  DINOv2-B/L/G 0.13–0.22, DINOv2-S 0.03, resto ≤ 0.01.
- §5.4, párrafo "The self-supervised tree is angular": el puntero final pasa a "Figure 5c shows the triplets and Table 4 gives the cosine
  census" (un solo puntero, última frase).
- Compilación: §7 termina en la página 9 (línea 482), statements en la 9, referencias en la 10 (línea 504); 28 páginas, 0 avisos.
  `qa_pages_final/` regenerado; sweep 207/207 (mismos checks; la leyenda de la Figura 5 sigue leyendo 0.03 y 0.38 desde el JSON).
- Fe de erratas del commit cc49936: su PDF se compiló con la copia de la tabla de robustez vacía (`phaseE_submission.py` se ejecutó sin
  regenerar antes `tab_q08_robust_final.tex` y su partición por paneles no era idempotente); el sweep lo detectó (204/207). Corregido:
  la partición acepta copias ya partidas, y el PDF, `qa_pages_final/` y `FINAL_CHECK.md` se regeneran desde la cadena completa.
- El autor ha subido `ICLR2027/figures/fig1_concept.pdf` (commit 4dafd6a). Los `.tex` buscan `ICLR2027/iclr2027/figures/fig1_concept.pdf`,
  así que la Figura 1 sigue siendo la caja de 1.4 in hasta que se copie ahí. Probado en la copia de compilación: con el comando literal del
  autor (`width=\textwidth, trim=23 229 3 232`) la figura mide 1.36 in y la versión final no se mueve (referencias en la página 10); el
  fichero congelado `main_iclr2027.tex`, que la incluye sin recorte, pasaría a terminar en la página 10 con las referencias en la 11.
  No se ha copiado: decisión del autor (TODO).

## 28. Correcciones de la cuarta revisión (2026-09-18, ordenadas por el autor). Los números cambian sólo donde una tabla estaba mal.

- (1) `rebuttal/scripts/expR73_transfer_bootstrap_record.py`: bootstrap de centroides de CIFAR-100 y DTD bajo el registro (nulo Haar,
  p99.9, 200 réplicas por remuestreo, 30 remuestreos con reemplazo de las imágenes cacheadas por clase al presupuesto del censo: 500/clase
  en CIFAR-100, 80/clase en DTD; 12 procesos en segundo plano, ~4 h). El panel (b) de la tabla de robustez (Tabla 2) se construye desde
  `expR59_imagenet_bootstrap_summary.csv` y `expR73_transfer_bootstrap_record_summary.csv` (columnas dataset, imágenes/clase, réplicas);
  las filas "centroid bootstrap" de expR32 salen del panel (a) y del `% prov:`; la frase de la leyenda sobre la s.d. máxima y la fracción
  de remuestreos negativos se calcula sobre los tres conjuntos. [Pendiente de la fusión de expR73: véase la nota al final de esta entrada.]
- (2) §5.2: la frase del control de número de clases es la del brief; el panel (d) de la Tabla 2 vuelve a listar el barrido completo de
  $C$ (DINOv2-L, 10 a 1000, ambos modos) para que la leyenda, que ahora dice lo mismo que el texto, esté sostenida por las filas;
  `phaseE_submission.py` comprueba en `expR60_c_sweep_record.csv` que el exceso se encoge con $C$ en los dos modos. El párrafo de v1
  "Hierarchy depth, not class count" del apéndice, que afirmaba lo contrario, se elimina de la versión final.
- (3) 15 modelos de texto en todas partes: resumen ("6 datasets and 15 text models", edición registrada), §4 ("fifteen text models",
  "OLMo-1B" en vez de "two OLMo sizes", y la frase "OLMo-7B was not extracted."); las filas OLMo-7B de las copias finales de la tabla de
  texto y del panel de modelos desaparecen (`DROPLINE` en `phaseE_submission.py`).
- (4) Tabla 11(c): la cabecera dice "supremum, Haar". Nota: `exp14_dbpedia.csv`, de donde salían esas dos columnas, se calculó con
  coeficientes gaussianos (`exp14_dbpedia_text.py`), así que la cabecera antigua describía sus números. Para que la cabecera sea cierta,
  la copia final (`gen_appendix_final.py`, `FINAL=True`, `final/tab_q07_wordnet_final.tex`) toma el exceso del supremo de
  `expR61_dbpedia_record.csv` (`excess_haar_sup`, nulo Haar del registro); $\hat\delta_{\max}$ es el mismo. Los valores pasan de
  −0.028/−0.024/−0.020 (gaussiano) a −0.032/−0.028/−0.024 (Haar) para BGE/E5/GTE; la tabla de v1 no cambia.
- (5) §5.3: el párrafo "The certified set depends on the frame" describe el marco balanceado (ViT-T entra; ViT-S y DINOv2-L salen) y el
  barrido $K=10/30/60$, y termina en "Only ViT-B and ViT-L are certified under every frame" (comprobado en `phaseE_submission.py` desde
  `expR56_depth_variants.csv` y `expR64b_wn30bal_summary.csv`); la frase de CIFAR-100 pasa al párrafo del régimen. Tabla 8 (copia final
  `final/tab_q04_depth_final.tex`): todas las $z$ con dos decimales; el panel (a) se parte en (a) estrellas con hubs gaussianos con las
  columnas $K=10$ y $K=60$ (sólo $z$) y el marco balanceado, y (a$'$) hubs Haar; los paneles (b)–(d) con $z$ a dos decimales.
- (6) `\FloatBarrier` tras cada subsección del apéndice (los bloques de v1 lo traen; ya no se elimina). Apéndice: 17 páginas
  (demostraciones incluidas), una más que sin barreras; ningún encabezado queda vacío.
- (7) §3.3 y limitación (ii): la frase del brief sobre el nulo espectral y los segundos momentos. (8) Limitación (iv): el test de
  profundidad es euclídeo y puede ser conservador para el árbol angular de DINOv2. Las limitaciones vuelven a dos párrafos (instrumento,
  cinco puntos; alcance, tres) para respetar la regla de 3–6 frases.
- (9) §5.1 "The premise does not survive calibration on the two datasets tested." y el resumen (edición registrada en
  `final_verbatim_edits.json`, ahora seis ediciones). (10) §6 reescrito: "Both readings predict the gain, and cosine is the best simple
  policy" (la lectura cruda predice la ganancia tan bien como el exceso; el veredicto de profundidad no predice nada; "cosine everywhere"
  es la política simple, no necesita validación ni regla, y la regla por objetivo añade poco: +0.23 frente a +0.28 pp en la Tabla 13e,
  comprobado desde `exp24_val_metric_selection.csv`); el papel del instrumento es certificar si hay estructura que justifique una lectura
  no euclídea. Ninguna frase dice que el exceso predice mejor que la lectura cruda. (11) Resumen: véase (3) y (9).
- Presupuesto: las adiciones (≈10 líneas) se compensan con recortes de mi prosa sin afirmaciones (la frase de resumen de la conclusión,
  "The cells that move are on the transfer sets", "A cell is one pair...", la frase de cobertura de §6, "leaves hierarchy within the
  clusters untested" en (iii)) y con espaciado (saltos de párrafo 0.3 ex, de ecuación 3 pt, `textfloatsep` 8 pt, `abovecaptionskip` 3 pt).
  §7 termina en la página 9 (línea 479), statements en la 9, referencias en la 10 (línea 501); 30 páginas, 0 avisos, 0 `??`.
- `REVIEWER_CHECKLIST_third.md` (+ su generador): sección "Fourth review" con los cuatro puntos de réplica R4.1–R4.4 y los editoriales.
- Sweep: `final_checks` amplía con la comprobación de las tablas regeneradas (z a dos decimales, barrido K, "supremum, Haar", sin
  OLMo-7B, sin expR32, expR73 en el panel (b)).

## 29. Pase combinado: quinta revisión y restos de la cuarta (2026-09-18, ordenado por el autor).

- A1 Confundidores como nivel de referencia. §1 (edición registrada del texto literal): "three artifacts push it down" → "its reference
  level depends on dimension, spectrum and statistic, so a raw value cannot be called low on its own"; "Anisotropic spectra mimic
  low-dimensional behavior" → "lower the effective dimension and raise the reading"; el supremo "grows with the budget and does not
  converge". Resumen: la frase "a low raw reading is what high dimension ... produce on their own" pasa a la formulación del nivel de
  referencia. §3.2: párrafo "A raw value cannot be called low on its own" con los tres nombres entre paréntesis.
- A2 Regla de Khrulkov: `exp1_delta_controls.csv` sólo tiene `delta_max`, el supremo muestreado, y la banda gaussiana de la Tabla 3 sale
  de esa columna, así que el rango recalculado con el supremo es el mismo: 0.48 a 2.5 (`phaseE_submission.py` lo verifica). §6 dice ahora
  que la regla fue calibrada con el supremo.
- A3 §5.2: "the flat datasets have smaller and less consistent excesses; FMNIST is genuine in 7 of 12 backbones" (relleno FMNIST_GEN
  desde el censo); "49 of 72 cells and 30 of 36 on the three datasets with 47 classes or more" (comprobado: 30/36, n ≥ 47). §4: "With
  ten classes there are only 210 quadruples, so the 99.9th percentile coincides with the supremum there."
- A4 §6: "Both readings predict the gain, and the depth verdict predicts none" (la lectura cruda predice tan bien como el exceso; "The
  depth verdict predicts no hyperbolic gain."; la regla por objetivo gana a coseno, +0.41 frente a +0.28 pp en las 40 celdas jerárquicas
  de `exp24_val_metric_selection.csv`, pero se eligió sobre las mismas celdas; entre las políticas no circulares coseno es la mejor) y
  nuevo párrafo "The calibration certifies structure and does not choose the readout" (que absorbe las ganancias +0.9 a +1.3 pp).
- A5 Resumen (ediciones registradas; el mismo texto va a OpenReview): "certified in 4 of 12 ImageNet backbones, two of which survive every
  choice of frame"; "the raw reading is not evidence, and the calibrated reading is weak and model-dependent"; "49 of 72 cells, 30 of 36
  on datasets with 47 classes or more". La misma redacción en el párrafo "The answer has three parts" de §1 y en la entrada de §5.1.
- A6 Figura 2(b): ViT-B en imágenes de CIFAR-100 (0.082, exceso +0.003, no genuino) frente a DINO-B en centroides de CIFAR-100 (0.0822,
  exceso −0.032, genuino); celdas en `rebuttal/results/final_fig2b.json`, valores verificados contra expR62/expR52 por el script de la
  figura; título "(b) CIFAR-100: same reading, opposite verdict"; leyenda y frase de §3.3 ("two cells on the same dataset with the same
  reading can receive opposite verdicts").
- A7 Proposición 1(b): "with $n$ fixed".
- A8 §5.3: tras "The other eight are not detected": el implante es un árbol de dos niveles sobre los treinta hubs mientras que la jerarquía
  de WordNet sobre los mismos hubs tiene hasta catorce niveles (`final_wordnet_levels.py`: hipónimo común más bajo de cada clúster del
  corte K=30 de expR56; profundidad máxima 14, mediana 9, mínima 2; `rebuttal/results/final_wordnet_levels.json`), y el control de
  desacoplamiento (expR74) decide si el veredicto viene de los hubs o de su acoplamiento con los desplazamientos.
- A9 Menores: fuera "OLMo-7B was not extracted"; Tabla 2(a) sin la columna "protocol" y con título nuevo; título de la Tabla 2 sin
  "follows the hierarchy of the labels rather than their number" ("the excess shrinks with the class count"); título del panel (b)
  condicionado a expR73; Tabla 13 y §B.12 "Both readings predict the gain"; Tabla 5 "8 causal LMs"; Tabla 9(b) regenerada con z a dos
  decimales (`final/tab_q05_power_final.tex`); la negrita de la Tabla 1 (celda ViT-B/C100 sign-positiva) ya estaba en el `.tex`, sólo
  faltaba en el PDF exportado; Alper & Averbuch-Elor citado en §5.4; Sala et al. (ICML 2018, verificado contra proceedings.mlr.press) y
  Gu et al. (ICLR 2019; DBLP, OpenReview y Semantic Scholar inaccesibles desde esta máquina, entrada escrita desde las actas, ver TODO)
  añadidos a §2 con una frase registrada como edición.
- A10 AI Use Statement en los tres puntos del formulario ("writing assistance; retrieval of references; research ideation or execution,
  namely ... and LLM-simulated reviews used as methodological feedback"). A11 `TODO(author)` sigue.
- Presupuesto: las adiciones (~22 líneas) se compensan con: la Figura 1 del autor pasa de `[H]` a `[t]` (edición registrada; deja de
  perder seis líneas al pie de la página 1), fusión del párrafo de ganancias de §6 en el nuevo, recortes de mi prosa sin afirmaciones
  (§5.1, §3.3, §3.4, §5.4) y espaciado (`parskip` 4 pt frente a 6 pt del estilo, salto de sección 1.0 ex). §7 termina en la página 9
  (línea 474), statements en la 9, referencias en la 10 (línea 495); 30 páginas, 0 avisos.
- B1/B2 (`expR74_decoupling.py`, `expR75_census_centered_haar.py`): escritos; lanzados al terminar expR73. Resultado en 29b.
- `REVIEWER_CHECKLIST_third.md` (+ generador): sección "Fifth review" con las correcciones y la lista de réplica (i)–(v) sin acción.

## 29b. Cierre del pase combinado (2026-09-18/19): expR73, expR74 y expR75 integrados.

- expR73 (bootstrap CIFAR-100 y DTD bajo el registro, 24 celdas × 30 remuestreos, 200 réplicas): en la Tabla 2(b) junto a ImageNet;
  s.d. del bootstrap ≤ 0.0026; cada celda conserva el signo de su exceso de registro en al menos el 67 % de los remuestreos (la peor
  es una celda de CIFAR-100 cercana a cero). Ficheros `expR73_transfer_bootstrap_record{,_summary}.csv`.
- expR74 (control de desacoplamiento, hubs reales y desplazamientos de cada clúster rotados por una rotación Haar independiente,
  10 semillas, mismo test de profundidad que el registro): NINGUNA de las cuatro certificadas dispara (ViT-S −2.36 → −1.30 ± 0.19;
  ViT-B −3.62 → −1.40 ± 0.25; ViT-L −4.04 → −1.61 ± 0.19; DINOv2-L −2.39 → −0.20 ± 0.14; 0/10 semillas); las no certificadas
  tampoco. Según el brief, "certified hierarchy" pasa a "certified hub–offset structure": párrafo nuevo en §5.3 ("What is certified is
  hub–offset structure, not the hub arrangement alone", con los niveles de WordNet y el resultado), frase de §5.3 sobre los hubs
  aleatorizados ("needs the real hub arrangement as well as its coupling with the offsets"), leyenda de la Figura 4, limitación (iii),
  leyenda de la Tabla 8 y columna "decoupled z (cert.)" en el panel (a′), `SUMMARY_plain.md`. El resumen y su espejo en §1 NO se
  han tocado, como pide el brief: decisión del autor (TODO). `phaseE_submission.py` asegura que las cuatro tienen 0/10.
- expR75 (nulo Haar centrado, Q ⟂ 1, censo completo bajo el registro): 44/72 genuinas frente a 49/72; cambian 5 veredictos, todos
  en datasets de diez clases (CLIP-B y CLIP-L en CIFAR-10, ViT-S y DINOv2-L en FMNIST, SigLIP-B en MNIST); desplazamiento máximo del
  exceso 0.053; "30 of 36" y "18 of 24" no cambian. §3.3: "reproduces the sample spectrum up to a centering term of order 1/√n, so it
  is the record. Centering the null changes the verdict in 5 of 72 cells, all with ten classes (Table 2)"; panel (f) de la Tabla 2 con
  las cinco celdas. El registro sigue siendo expR52 (el brief pedía reportar los cambios; cambiar el registro es decisión del autor).
- Compilación: §7 termina en la página 9 (línea 483), statements en la 9, referencias en la 10 (línea 505); 31 páginas (el panel (f)
  añade una), 0 avisos, 0 `??`. Sweep 210/210 (dos checks nuevos: B.1/B.2 y las correcciones de la quinta revisión).
  `main_iclr2027_final.pdf`, `qa_pages_final/` (31) y `FINAL_CHECK.md` regenerados.

## 30. Decisiones del autor (2026-09-20): el registro es el nulo Haar centrado; el desacoplamiento cambia la afirmación de profundidad.

- (1) Registro = expR75 (nulo Haar con Q ⟂ 1, exacto en el espectro centrado). Regenerados desde él: Tabla 1 (`gen_main_table.py`
  con `CENSUS_SRC=expR75_census_centered_haar.csv FINAL_ONLY=1` → `tab_census_final.tex`; la de v1 sigue en expR52), Figura 3 y las
  barras de la Figura 2 (`make_figs_final.py`), Figura 2(b) (DINO-B/CIFAR-100: exceso −0.0315 bajo el registro centrado), los recuentos
  (44 de 72; 30 de 36 y 18 de 24 sin cambio; FMNIST 5 de 12, relleno `FMNIST_GEN`; signo negativo 68/72) y la sensibilidad conjunta
  (`expR66c_joint_sensitivity.py`: expR66 con el nulo centrado y el registro expR75; 12 partes en segundo plano, ~4.5 h; la Tabla 2(c)
  lee `expR66c_joint_sensitivity_summary.csv` cuando existe). La tabla del censo de la versión final (`final/tab_q01_census_final.tex`,
  `q_census` con `FINAL=True`) lleva el panel (a) sobre el registro centrado y el panel (b) con cinco construcciones (Haar centrado
  = registro, Haar sin centrar, Haar×supremo, gaussiano×p99.9, gaussiano×supremo: 44/49/46/52/52 genuinas); el panel (f) de la Tabla 2
  desaparece. §3.3: Definición 3 con Q ortogonal al vector de unos; "The centered Haar construction reproduces the centered spectrum
  exactly, so it is the record"; sin la frase del término de centrado. Acuerdo del censo coseno con el registro centrado 60/72 ("agrees
  ... in most cells"). Sin regenerar (nulo sin centrar, n ≥ 100 o texto): expR59/expR73 (bootstrap), expR72 (presupuesto), expR62
  (nivel de muestra, n=1000), expR53/expR61 (texto), expR68 (corolario), expR56/expR69/expR74 (test de profundidad, nulo de hubs);
  la Figura 2(b) cambia en 0.0005.
- (2) Redacción del autor para la afirmación de profundidad, literal: resumen (frase de centroides + "structure above the superclasses
  is certified in 4 of 12 ImageNet backbones, two of which survive every choice of frame; a decoupling control shows that it is the
  alignment of each cluster with its hub, not a hierarchy among the hubs: once cluster orientations are randomized, none of the four
  fires, and no hierarchy above the superclasses is certified in any backbone." + frase MERU "shows the same clustering, no such
  structure either, and embeddings that never leave the near-flat regime"), el espejo de §1, la contribución 2 ("Clustered structure in
  most models, hub-aligned structure in a few and no certified hierarchy above the superclasses, and a hyperbolic backbone as the
  control for imposing the geometry."), la tesis en el resumen y en §7 ("Read correctly, foundation models organize classes into
  clustered, hub-aligned structure that is moderately shared; no hierarchy above the superclasses is certified, they do not converge to
  one common tree, and their raw tree-likeness is not evidence for hyperbolic geometry."), las entradas de §5.3 ("Structure above the
  superclasses is certified in four backbones." / "It is the alignment of each cluster with its hub, not a hierarchy among the hubs."),
  la frase MERU de §5.3, la leyenda de la Figura 4, la limitación (iii) ("hub-aligned structure") y §3.4 ("certified against its
  matched star"). "certified" sólo para la estructura probada con el desacoplamiento. Ediciones registradas: 15 (`final_verbatim_edits.json`).
- (3) El resumen para OpenReview es el del `.tex` (318 palabras); se imprime en el informe de la sesión.
- Presupuesto: recortes en §5.3 (frase redundante con la entrada) y en la Definición 3, `parskip` 3 pt (antes 4 pt) y `textfloatsep`
  7 pt (antes 8 pt).
  [Compilación y sweep: véase el cierre 30b.]
- Sweep: tesis nueva; cabecera de números 44 of 72 / 5 of 12; comprobación de las decisiones (cinco construcciones, Tabla 1 desde expR75,
  redacción del autor en resumen/§1/§5.3/Figura 4, sin "hub--offset", sin "certified hierarchy", Tabla 2(c) desde expR66c).

## 31. Sexta revisión (2026-09-20, ordenada por el autor). Texto del fichero de envío.

- (1) La profundidad se declara no concluyente donde era un hallazgo: el test no tiene potencia al nivel de ruido de ImageNet, así que si
  las superclases forman una jerarquía queda abierto; lo que certifica en 4 de 12 backbones es la alineación de cada clúster con su hub.
  "no hierarchy above the superclasses is certified in any backbone" → "whether the superclasses form a hierarchy is left open: the test
  has no power at this noise level" (§1, §5.3; en el resumen con "the test having no power"). Entradas de §5.3: "The alignment of each
  cluster with its hub is certified in four backbones." / "Whether the superclasses form a hierarchy is left open."; contribución 2:
  "Structure beyond the second moments in most models, hub-aligned structure in a few, hierarchy left open by a test without power at this
  noise level, and a hyperbolic backbone..."; limitación (iii) reescrita (dos frases); leyendas de la Figura 4 y de la Tabla 8. Tesis
  nueva del autor en el resumen y en §7 ("whether that structure is hierarchical remains untested at the noise level of real clouds").
- (2) "certifies clustered structure" → "certifies structure beyond the second moments; clustering is its most plausible reading,
  supported by the superclass recovery of §5.4" (§3.3); "clustered structure" deja de presentarse como certificado en la entrada y el
  título de §5.2, la Figura 3, la Tabla 1, "says clustered, not deep" → "says structure beyond the second moments, not depth", §1 (edición
  registrada), las leyendas de las tablas del censo y de texto y el título de la subsección B.3. La tesis conserva "clustered, hub-aligned
  structure" (texto del autor).
- (3) §6, párrafo "Both readings predict the gain": "Calibration buys interpretation, a low $\delta$ is not hierarchy, and not prediction."
- (4) Resumen reescrito a ≤ 250 palabras: sin "two of which survive every choice of frame" ni "30 of 36…" (siguen en §5), "A cell is one
  model read on one dataset." en frase propia. El texto está en `final_verbatim_edits.json` ("abstract_sixth_review") y sustituye al
  resumen literal editado; las ediciones registradas del resumen desaparecen (quedan 11 ediciones, todas de §1 y §2). Es el texto para
  OpenReview.
- (5) Lista de réplica en `REVIEWER_CHECKLIST_third.md`: suelo de p en 1/201 y BH; z de 10 semillas de estrella; normalización por un
  percentil alto de distancias en vez del diámetro; multiplicidad sobre marcos.
- Pista paralela, prioridad 1b (CPU): `expR79_synthetic_deep_poincare.py` (nube sintética con el espectro real de ViT-L y una jerarquía
  implantada de tres niveles 2/6/30 a la razón intra/entre real; incrustaciones de Poincaré de WordNet de Nickel & Kiela entrenadas
  con gensim sobre el cierre transitivo del árbol que abarca las 1000 hojas de ImageNet, d = 10 y 50), ambas por el censo, el test de
  profundidad y el control de desacoplamiento; en marcha. Resultado en `PARALLEL_STATUS.md`.
- Compilación: §7 termina en la página 9 (línea 479), statements en la 9, referencias en la 10 (línea 501); 31 páginas, 0 avisos.
  Sweep: pendiente sólo del fichero de expR66c (cierre nocturno).

## 32. Sexta revisión, continuación (2026-09-20, ordenada por el autor).

- R1 "certifies clustering(, not depth)" → "certifies structure beyond the second moments(, not depth)" en el título de la Tabla 3 y en
  B.2 (sustitución en las copias finales y en la prosa del apéndice; `REPL_ALL` en `phaseE_submission.py`). R2 Tabla 9, título: "The
  depth test has full power on synthetic hierarchies at the leaf frame and none at the top-level frame used on real backbones" (copia
  final, `q_power` con `FINAL`). R3 "the validated regime" → "the regime where false alarms are controlled" en §5.3, B.7 y las copias de
  las tablas ("validated range" → "range where false alarms are controlled"; "depth validated/unvalidated" → "false alarms
  controlled/uncontrolled"; "is validated at" → "controls false alarms at"). R4 §5.2, colapso neuronal: "The census does not certify that
  frame: class means cluster around superclass hubs, which the superclass recovery of §5.4 shows and the depth test reads as hub
  alignment." R5 tesis (resumen y §7): "organize classes into clustered structure, hub-aligned in a few backbones, that is moderately
  shared" (§1 no contiene la tesis; el sweep exige exactamente dos apariciones). R6 §5.3: "The alignment is relational: it needs both
  the real hubs and the real orientation of each cluster relative to them, which is why randomizing either removes it, 0 of 60 and 0 of
  4." (cabecera de números ampliada con ese grupo).
- Pista paralela, prioridad 1c (CPU): `expR80_implanted_alignment.py` (desde la nube desacoplada de expR74, el eje principal de cada
  clúster girado hacia la dirección de su hub por una fracción s del ángulo, s ∈ {0, 0.25, 0.5, 0.75, 1}, 5 semillas, 12 backbones;
  espectro intra-clúster y hubs intactos, comprobado; test de profundidad del registro en cada s). Regla de decisión del brief
  (potencia ≥ 0.8 en s = 1 con falsas alarmas ≤ 0.05 en s = 0) codificada en `expR80_decision.csv`: si se cumple, `q_power` (Tabla 9c) y
  `make_figs_final.py` (tercera curva de la Figura 4b) la incorporan al envío y la frase "certifies hub alignment with measured power"
  entra en §5.3; si no, queda en la versión paralela. Cuatro procesos a prioridad baja desde las 19:15.
- Compilación y sweep: véase el cierre de esta entrada al terminar expR66c.

## 33. Prioridad 1c cableada al envío, y arranque del control positivo entrenado (2026-09-20, noche; brief del autor).

- Integración condicional de expR80, según la regla del brief (potencia ≥ 0.8 en s = 1 con falsas alarmas ≤ 0.05 en s = 0, leída de
  `expR80_decision.csv`). Si se cumple: `q_power` añade la Tabla 9c (tasa de detección frente a s, por backbone y agregada);
  `make_figs_final.py` añade la curva punteada "implanted alignment" a la Figura 4b (leyenda registrada en `final_fig4.json`);
  `phaseE_submission.py` sustituye en §5.3 por la redacción del autor, "certifies the alignment of each cluster with its hub, with
  measured power: implanted alignment is detected in x of y runs at full strength and in none at zero" (frase propia de 32 palabras,
  seguida de "It certifies 4 of 12 backbones…"; entradilla "…certified in four backbones, with measured power."), actualiza el pie de la
  Figura 4 ("…with measured power; whether…" y "(dotted)"), la limitación (iii) ("has measured power for alignment and none for
  hierarchy at this noise level") y el resumen ("plus a depth test whose power is measured for the structure it certifies"). El
  resumen tenía 249 palabras y la frase del autor añade seis: para respetar el tope de 250 la misma oración pierde cinco palabras
  ("and we build" → "and build"; "a random cloud of the same shape" → "a matched random cloud"; "the reference level of the raw
  reading" → "the raw reading's reference level"); el resto del resumen no cambia. Texto registrado como `abstract_final` en
  `final_verbatim_edits.json` (el sweep compara contra él; `abstract_sixth_review` se conserva). Si no se cumple: nada entra en el
  envío; `phaseE_rebuttal.py` escribe el párrafo y la tabla con los números en `main_iclr2027_rebuttal.tex`.
- Sweep: check nuevo de la prioridad 1c en ambas ramas (regla recomprobada desde el fichero de decisión, cada redacción, Tabla 9c y
  curva; o bien su ausencia total). `rebuttal_checks()`: el fichero de réplica es el envío congelado más los párrafos y la subsección
  insertados y nada más; cada número nuevo (expR77/78/79/80) se rederiva de su CSV; ninguna afirmación sin criterio cumplido.
  `rebuttal_rebuild.sh` compila la versión paralela y exporta `ICLR2027/main_iclr2027_rebuttal.pdf`.
- `phaseE_rebuttal.py`: ancla del control positivo actualizada a la entradilla de la sexta revisión; secciones nuevas para expR79
  (prioridad 1b) y expR80 (sólo si la regla no se cumple); `PLATONIC_RESULTS` respetado por los dos constructores.
- Prueba de la rama "regla cumplida" de extremo a extremo sobre una copia aislada de `rebuttal/results` en el scratchpad con
  ficheros sintéticos de expR80 (la carpeta real de resultados no se tocó): 31 páginas, 0 avisos, resumen de 249 palabras, sweep con
  todos los checks salvo el de expR66c; §7 sigue acabando en la página 9 (línea 483) y los tres statements pasan a la 10 con las
  referencias (antes empezaban en la 480 de la página 9). Después se regeneró el envío desde los resultados reales.
- Pista paralela, §1: decodificación terminada a las 20:42 (1.281.167 imágenes, 134,8 min; tamaño del memmap exacto). Los dos
  fine-tunings arrancaron a las 20:50 (GPU 0 CE de hoja, GPU 1 CE + jerárquica, semilla 0). Errata: el primer arranque cayó en el paso 0
  porque `gather` localizaba cada índice con una búsqueda ordenada sobre los inicios de chunk del pool, que va en orden aleatorio;
  corregido a búsqueda directa por inicio de chunk (prueba unitaria), sin cambiar los lotes (el plan depende sólo de semilla y época).
  60 img/s en el paso 50 con la máquina cargada (≈ 6 h por época; mejorará al acabar expR80/expR66c/expR79).
- Pista paralela, §2: el paso delta corría desde las 19:26 (un duplicado lanzado a las 19:42 por error se detuvo); primer lote de
  CIFAR-10: δ_rel de su estimador 0.297 frente al 0.26 publicado; exceso del registro −0.002 (rango 155/200). ~15 min por lote.
- Sweep del envío restaurado: 210/211, pendiente sólo del fichero de expR66c (cierre nocturno).
- **Cierre de la prioridad 1c (2026-09-21, 00:02)**: regla NO cumplida. Tasa de detección por s = 0 / 0.25 / 0.5 / 0.75 / 1:
  0.017 / 0.233 / 0.500 / 0.550 / 0.600 (12 backbones × 5 semillas); falsas alarmas en s = 0: 1 de 60 (dentro del 0.05); potencia en
  s = 1: 36 de 60 (0.60, por debajo del 0.8). Nada entra en el envío: `main_iclr2027_final.tex` queda como en `92f3c81` (sin Tabla 9c,
  sin tercera curva, sin "measured power"). Por backbone a fuerza plena: ViT-T/S/B/L y CLIP-B detectan en las 5 semillas (ViT-B/L ya en
  s = 0.25–0.5), DINO-B 4, SigLIP-B 3, CLIP-L y DINOv2-g 2, DINOv2-S/B/L 0: la potencia del test para la alineación depende del
  backbone. Resultado escrito en `main_iclr2027_rebuttal.tex` (párrafo "Implanted alignment on the real clouds." en §5.3 y tabla por
  backbone en la subsección de la pista paralela; lista de backbones en el orden del artículo) y en `PARALLEL_STATUS.md`; versión
  paralela compilada (`ICLR2027/main_iclr2027_rebuttal.pdf`, 32 páginas, 0 avisos). Sweep 213/214: los tres checks de la réplica
  pasan y sigue pendiente sólo el fichero de expR66c. `phaseE_rebuttal.py` exige además que expR79 haya terminado (fila de Poincaré
  d = 50) antes de escribir su párrafo: los resultados parciales no entran.

## 34. Cierre nocturno del envío con expR66c (2026-09-21, 02:40).

- `expR66c_joint_sensitivity.py --merge` (sensibilidad conjunta bajo el registro centrado, 12 partes, 30 remuestreos de centroides
  por celda): registro 44/72 (18/24 en ImageNet y CIFAR-100); z conjunta ≤ −2 en 39/72 (18/24); genuino en ≥ 27 de 30 remuestreos en
  42/72 (17/24). Bajo el registro anterior (expR66, no centrado) eran 49 / 42 / 47 con los mismos 18 / 18 / 17 en ImageNet y CIFAR-100.
  Entra sólo en el panel (d) de la Tabla 13 (copia final regenerada, `q_robust_final`); la frase de §5.2 ("leaves most of the count in
  place") no lleva números y sigue siendo cierta (39 y 42 de 44).
- Reconstrucción completa (`final_rebuild.sh`): 31 páginas, 0 avisos; §7 acaba en la página 9 (línea 460 de inicio), statements en
  la 9 (480), referencias en la 10 (501). Sweep 214/214: se cierra el único check pendiente desde la entrada §30 (fichero de expR66c
  en la tabla de robustez). Versión paralela reconstruida sobre el envío cerrado (`rebuttal_rebuild.sh`).

## 35. Pista paralela, prioridad 2 cerrada: réplica de Khrulkov et al. (2026-09-21, 03:00).

- expR78 (ResNet-34 de torchvision, 10 lotes balanceados de 1500 puntos, su estimador exacto y δ_rel = 2δ/diam): CIFAR-10 0.275,
  CIFAR-100 0.261, CUB 0.270, MiniImageNet 0.203 frente a 0.26 / 0.25 / 0.25 / 0.21 publicados; los cuatro dentro de 0.03, así que se
  cumple la condición fijada de antemano para afirmar algo. Calibrado con el registro sobre las mismas nubes: exceso −0.004 / −0.009 /
  −0.004 / −0.023; p máxima sobre los diez lotes 0.33 / 0.03 / 0.25 / 0.005. Sólo entra en `main_iclr2027_rebuttal.tex` (párrafo de
  §5.1 "A published reading, reproduced and calibrated." y tabla `tab:r2-khrulkov`); la frase calibrada dice exactamente lo que
  muestra la tabla (p ≤ 0.05 en CIFAR-100 y MiniImageNet). Sweep de la versión paralela 215/215. El envío no cambia.

## 36. Pista paralela, prioridad 1b cerrada: jerarquía profunda sintética y Poincaré (2026-09-21, 03:15).

- expR79: nube sintética con el espectro real de ViT-L (d = 1024) y jerarquía implantada de tres niveles (2/6/30) a la razón
  intra/entre real (1.88), 5 semillas: el test de profundidad dispara en K = 30 en 4 de 5 (z −1.80, −2.45, −2.38, −2.08, −2.20); el
  control plano (30 hubs iid, mismo espectro y razón) en 0 de 5. El control de desacoplamiento dispara en 5 de 5 nubes profundas con
  z ≈ −4.2 a −4.8: una jerarquía que vive en los hubs sobrevive a la aleatorización de orientaciones, que es justo lo que no hacen los
  backbones reales. Incrustaciones de Poincaré de WordNet (Nickel & Kiela, d = 10 y 50), leídas con distancias euclídeas sobre las
  coordenadas de la bola: exceso positivo (+0.083, +0.035), z de profundidad +2.36 y +1.53; no disparan (lectura euclídea, limitación iv).
- Sólo en `main_iclr2027_rebuttal.tex` (párrafo de §5.3 "A deep hierarchy at the real noise level." y tabla `tab:r1b-deep`; 33
  páginas, 0 avisos; sweep 216/216). El envío no cambia. Pendiente de decisión del autor, como pide el brief para los experimentos que
  acaban y pasan el sweep antes del 24: el resultado afecta a la frase "the test has no power at this noise level" (§5.3, resumen, §7,
  limitación iii), porque en este escenario sintético la potencia para una jerarquía profunda al nivel de ruido de ViT-L es 0.8, no cero.

## 37. Prioridades 1b y 2 integradas en el envío, 1c como limitación (2026-09-21, brief del autor).

- **1b, §5.3**: tras el párrafo del desacoplamiento (entradilla nueva "No hierarchy above the superclasses is found."), párrafo
  "A deep hierarchy at the real noise level is detected." (4 de 5 semillas en el marco de registro, control plano 0 de 5, el
  desacoplamiento dispara en todas las semillas profundas y más hondo, Poincaré no dispara en lectura euclídea) con la Tabla 9(d)
  (`q_power`, panel nuevo) y provenance `expR79_synthetic_deep_poincare.csv`. "the test has no power at this noise level" sustituido
  por la frase del autor ("The test misses an implanted two-level tree at this noise level but detects a three-level hierarchy with
  ViT-L's spectrum and noise in 4 of 5 seeds, and that detection survives orientation randomization, whereas none of the four real
  verdicts does") en §5.3; en §1 y en el resumen va la forma del resumen que dio el autor ("a three-level hierarchy implanted at the
  real noise level is detected and survives orientation randomization, whereas none of the four real verdicts does, so no hierarchy
  above the superclasses is found"); contribución 2: "no hierarchy above the superclasses where a deep synthetic one is detected".
  Tesis (resumen y §7): "…moderately shared; no hierarchy above the superclasses is found where a deep synthetic one is detected,
  they do not converge…". Limitación (iii) reescrita: potencia medida para una jerarquía sintética de tres niveles con el espectro y
  el ruido de ViT-L, no para las más someras; una de dos niveles se pierde a este nivel de ruido. Pie de la Figura 4 y de la Tabla 8:
  "no hierarchy above the superclasses is found". "left open" y "no power" no aparecen ya en el envío (el sweep lo exige).
- **2, §5.1**: entradilla "Raw readings are not evidence, ours or published, and calibrated ones are weak and model-dependent."
  y párrafo "A published reading is reproduced and calibrated." (los cuatro δ_rel publicados reproducidos dentro de 0.03; calibrados,
  exceso negativo en todos y sólo CIFAR-100 y MiniImageNet bajo el nulo en todos los lotes a 0.05) con la Tabla 3(b) (copia final
  nueva `tab_q03_sample_final.tex`, panel (a) = la tabla de v1, panel (b) = las cuatro filas; `q_sample` con `FINAL`). La frase
  "including a published reading reproduced and calibrated" no cabe en el resumen (250 palabras) y no se añadió. Limitación de
  alcance: se actualizó la (vi), que es la que habla de la lectura a nivel de muestra ("two datasets, one image budget and one
  published setting"); el brief decía (viii), que trata del alcance downstream y no cambió. Decidir si era otra cosa.
- **1c, limitaciones**: frase del autor en (iii) con el porcentaje como relleno ({{IMPL_PCT}} = 60, derivado de
  `expR80_implanted_alignment.csv`; la afirmación por familias, todas las semillas para los ViT supervisados y CLIP-B y ninguna para
  DINOv2-S/B/L, se comprueba contra el fichero en el constructor y en el sweep). Tabla 9(c) ya sin condición, con el resultado de la
  regla en el pie ("the pre-set bar of 0.8 power at full strength is not met"). La curva punteada de la Figura 4b sigue condicionada
  a la regla (no entró).
- **Resumen** (249 palabras, texto de OpenReview en `ICLR2027/OPENREVIEW_abstract.txt`, registrado como `abstract_final`): además de
  las dos frases del autor, la oración del instrumento y otras pierden palabras para respetar el tope: "rest on a premise we call
  latent hyperbolicity" → "rests on a premise, latent hyperbolicity"; "the reference level of the raw reading" → "the raw reading's
  reference level"; "and we build an instrument that reads" → "and build an instrument reading"; "a random cloud of the same shape"
  → "a matched random cloud"; "one model read on one dataset" → "one model on one dataset"; "the calibrated reading is weak" → "the
  calibrated one is weak"; "recovers the human taxonomy partially" → "partially recovers the taxonomy". Vetar lo que no convenga.
- **Presupuesto de página** (§5.5 y §6 primero, como pedía el brief): ecuación (7) de la curvatura en línea (quedan seis ecuaciones
  numeradas; `eq:curvature` no se citaba); en §6 cae la primera frase del párrafo de la calibración ("The instrument answers whether…
  presupposes."), se aprietan las dos frases del párrafo de las lecturas y el puntero ("Table 13 gives both"); §5.5 en tres líneas.
  Después: párrafo de §5.1 y de §5.3 (jerarquía profunda) apretados, "Under the spectrum-matched-hub star the certified set is the
  same four" fundido en la frase anterior ("under both matched stars"), párrafo del implante de dos niveles reescrito (entradilla
  "The two-level implant is missed at ImageNet's noise level."), "An earlier isotropic star produced false alarms and is withdrawn"
  eliminado de §5.3 (sigue en el pie de la Tabla 8), pie de la Figura 4 en cuatro líneas, y espaciado tipográfico algo más prieto
  (parskip 2 pt, saltos de sección 0.8 ex, abovecaptionskip 2 pt, textfloatsep 6 pt). Resultado: §7 empieza en la línea 460 de la
  página 9, statements en la 482 (página 9), referencias en la 501 (página 10); 32 páginas, 0 avisos: la misma disposición que antes
  del brief.
- **Sweep**: checks nuevos para 1b, 1c y 2 con cada número rederivado de su CSV (recuentos de expR79, porcentaje y familias de
  expR80, los cuatro valores y el umbral de expR78), la tesis y las frases del autor, ausencia de "left open"/"no power"; los dos
  enunciados literales del autor (§5.3 y limitación 1c) quedan exentos de la regla de 35 palabras como la tesis; las limitaciones
  enumeradas, exentas del recuento de 3–6 frases; seis ecuaciones; seis copias puras del apéndice (la Tabla 3 pasa a regenerada).
  `phaseE_rebuttal.py` omite las secciones ya integradas; la versión paralela vuelve a ser el envío congelado (más el control positivo
  cuando exista).
- Pendiente para el autor (TODO): el título de la Tabla 9 (R2, "…and none at the top-level frame used on real backbones") describe el
  barrido sintético de expR55; con expR79 el test sí tiene potencia en el marco superior cuando la nube lleva el espectro real y tres
  niveles. No se ha tocado porque el brief no lo pedía y R2 lo fijó literalmente.
- **Retoque del autor (2026-09-21, resumen)**: "A hyperbolic backbone shows the same, near-flat structure." → "A backbone trained
  in hyperbolic space shows the same structure and lives in the near-flat regime." La frase nueva tiene ocho palabras más (no cinco):
  257. Para respetar el tope de 250 se recortan siete palabras sin tocar contenido: "We show that the raw reading's reference level
  depends on dimension, spectrum and statistic, and build" → "The raw reading's reference level depends on dimension, spectrum and
  statistic; we build"; "but a star already produces it" → "but so does a star"; "the calibrated one is weak and model-dependent" →
  "the calibrated one weak and model-dependent"; "tree-like because their class geometry scores a low Gromov δ" → "tree-like, their
  class geometry scoring a low Gromov δ"; "once the cut is controlled" → "at a controlled cut". Resumen en 250 palabras exactas
  (`OPENREVIEW_abstract.txt` y `abstract_final` regenerados); envío recompilado.
- **Decisión del autor sobre los recortes (2026-09-21)**: se mantienen "The raw reading's reference level depends on …; we build"
  y "but so does a star"; se revierten los otros tres ("the calibrated one is weak", "tree-like because their class geometry scores a
  low δ", "once the cut is controlled"). El resumen queda en 253 palabras, tres por encima del tope de 250 que fijó la sexta revisión:
  el constructor acepta hasta 255 para poder compilar y el sweep sigue exigiendo 250, así que el check del resumen queda en FAIL
  hasta que el autor decida (aceptar 253, o recortar tres palabras de su elección). Los dos recortes limpios que quedan sin tocar
  contenido ("The trees are moderately shared" → "Trees are moderately shared"; "the naive comparison manufactures an island" →
  "naive comparison manufactures an island") sólo llegan a 251.

## 38. Séptima revisión (2026-09-21): titular acotado hasta 1d, control plano desacoplado, menores; expR81 en marcha.

- **Run 1d** `expR81_deep_per_backbone.py`: la jerarquía sintética profunda de expR79 con el espectro y la razón intra/entre de cada
  uno de los 12 backbones (expR64b), 5 semillas, test de profundidad en K = 30 y control de desacoplamiento en cada una; cada fila
  guarda el exceso B de la nube y de su estrella y la dispersión de la estrella. Diez shards a prioridad baja desde las 09:12 (las
  cinco semillas de ViT-L por separado, que reproducen las nubes de expR79). `--merge` da la potencia por backbone y por familia con la
  regla 0.8 del brief. Plazo: 23 de septiembre.
- **(1) Titular acotado**: "no hierarchy above the superclasses is found in the backbones whose noise level the control covers" en
  el resumen (frase de profundidad y tesis), §1 (espejo y contribución 2), entradilla de §5.3, pies de la Figura 4 y la Tabla 8, y tesis
  de §7 (`THESIS` del sweep). Limitación (iii): "The depth test's power for a three-level hierarchy was measured at ViT-L's spectrum
  and within/between ratio ({{RATIO_VITL}}) with five seeds, not at the DINOv2 ratios ({{RATIO_DINO_LO}}--{{RATIO_DINO_HI}}) or for
  shallower hierarchies", con los rellenos derivados de `expR64b_wn30_summary.csv` (ViT-L 1.9; DINOv2 3.0–3.9 sobre los cuatro
  DINOv2, el brief decía 3.7–3.9 que es el rango de B/L/G; DINOv2-S está en 3.0). Si 1d da potencia ≥ 0.8 en todas las familias se
  quita la acotación; si no, se mantiene y se nombran las familias.
- **(2) Párrafo del desacoplamiento (§5.3, "A deep hierarchy…")**: "A flat control with the same spectrum and ratio fires in 0 of 5,
  and once decoupled it fires in {{FA_DEC}} of 50 runs" (8 de 50, de `dec_frac_cert` de las filas planas de expR79); "The decoupling
  control fires on every deep seed and reads deeper than the intact cloud…"; y la frase {{DEC_OBS}}: mientras no existan las filas de
  ViT-L de expR81 es "Why the synthetic hierarchy reads deeper once decoupled is an open observation."; con ellas el constructor
  compara la dispersión de la estrella y el exceso B antes y después (cocientes con umbrales 0.8/1.2) y escribe la explicación que
  los datos sostienen (estrella, exceso o ambos), registrada en `final_dec_obs.json`. Cabecera de números: "8 of 50", "1.9",
  "3.0--3.9"; el párrafo pasa a (3, 2) en las exenciones.
- **(3) Menores**: etiqueta "no false alarms at s = 0 (real spread)" anclada a la curva de dispersión real en la Figura 4b; la frase no
  demostrada de la prueba de la Proposición 1(b) ("For a cloud with covariance Σ the same argument runs with d replaced by the
  effective dimension…") eliminada, queda "A maximum over a growing sample of quadruples is non-decreasing in the sample size."; pie de
  la Figura 1 "all read the same" → "all read alike, all low" (edición registrada modificada en el sitio); Tabla 2(e): estadístico
  declarado (supremo bruto δ_norm, mayor defecto de cuatro puntos sobre cuádruplas muestreadas dividido por el diámetro, 512 puntos;
  fila de fine-tuning: 1000 imágenes de test de CIFAR-100 antes y después del fine-tuning no jerárquico (color) de DINOv2-S, 5×10⁴
  cuádruplas; fila de profundidad: estados ocultos por capa de DINOv2-B), verificado en `delta_hyperbolicity.py` y
  `run_finetune_ablation.py` de Platonic.
- Sweep: check de la séptima revisión (titular acotado ≥ 6 veces, 8 de 50 rederivado, frase de observación según `final_dec_obs.json`,
  razones desde expR64b, pie de la Figura 1, prueba sin la extensión). El resumen sube de palabras con la acotación (ver el
  recuento en la nota de cierre); el tope del constructor pasa a 270 y el sweep mantiene 250 (decisión pendiente del autor).
- **Cierre (2026-09-21)**: compilación con §7 en la línea 459 de la página 9, statements en la 482 (página 9), referencias en la 500
  (página 10); 33 páginas (el apéndice gana la Tabla 9(d) y la 3(b)), 0 avisos. Para el presupuesto: "The other eight are not detected,
  which is not the same as no structure" (§5.3) y "The most aligned families carry the most fragile excess and DINOv2 the most robust
  one" (§5.4) eliminadas; Poincaré, "for coherent and random subsets alike", "Nothing helps on flat labels or at the sample level",
  la frase del implante encogido y la limitación (iii) apretadas; parskip 1 pt, abovecaptionskip 1 pt, textfloatsep 5 pt. Sweep
  215/216: sólo falla el tope de 250 palabras del resumen (264 con la acotación; decisión del autor pendiente). Versión paralela
  recompilada sobre el envío. La frase de observación se actualizará sola cuando existan las filas de ViT-L de expR81.

## 39. Séptima revisión, continuación (2026-09-21): las familias en el titular acotado; sesgo del test desacoplado.

- **(1)** "in the backbones whose noise level the control covers" → forma larga "in the supervised and contrastive backbones, whose
  noise level (ratios {{RATIO_SC_LO}}--{{RATIO_SC_HI}}) the control covers, and not tested in the DINOv2 family
  ({{RATIO_DINO_LO}}--{{RATIO_DINO_HI}})" en §1 (espejo y contribución 2), §5.3 y pie de la Figura 4 (pie de la Tabla 8 sin números);
  forma corta "in the supervised and contrastive backbones; the DINOv2 family lies outside the control's noise level" en el resumen
  (frase de profundidad y tesis) y, por ser la misma frase, en la tesis de §7, donde la coma que seguía pasa a punto y coma para que la
  lista de cláusulas se lea ("…noise level; they do not converge…"). Rellenos desde expR64b: 1.3–2.0 = ViT supervisados, CLIP-B/L y
  SigLIP-B (DINO-B, 2.1, queda fuera de ambas frases, como en el brief); DINOv2 3.0–3.9. La entradilla de §5.3 se queda en trece
  palabras ("…is found in the supervised and contrastive backbones.") por la regla de 16 palabras, y la forma larga va como frase
  propia al final del párrafo. Se ajustará el 23 si la potencia por backbone de expR81 dice otra cosa.
- **(2)** Tras "once decoupled it fires in 8 of 50 runs": la frase del autor con los rangos de z como rellenos de expR79
  ("the synthetic hierarchy at $z$ {{Z_DEEP_HI}} to {{Z_DEEP_LO}} stands well clear of that bias, the flat control at {{Z_FLAT_HI}}
  to {{Z_FLAT_LO}}"): −4.2 a −4.8 como decía el brief; el control plano va de −1.6 a −1.8 (el brief decía −1.7; la media más baja es
  −1.76). El paréntesis "(flat control …)" pasa a aposición por la regla de los paréntesis; la frase de Poincaré se funde con el
  puntero para no pasar de seis frases. Limitación (iii): "…would go undetected, and the decoupled test is biased toward firing,
  which makes the real verdicts under decoupling conservative".
- Sweep: tesis nueva; recuentos de las dos formas (larga ×3 con "is found" y ×4 en total, corta ×3); rangos de z rederivados de
  expR79; la expresión regular de números de la versión final acepta rangos con "--" sin espacios; exenciones de los dos párrafos de
  §5.3 ampliadas y las frases literales del autor exentas de la regla de 35 palabras.
- **Observación del desacoplamiento, resuelta con expR81 (ViT-L, 10:15)**: las cinco semillas de ViT-L de expR81 reproducen las nubes
  profundas de expR79 (z intactas −1.80, −2.45, −2.38, −2.08, −2.20). Bajo desacoplamiento la dispersión de la estrella baja a 0.69 de
  la intacta (0.0032 → 0.0022) y el exceso bajo la estrella casi se duplica (−0.019 → −0.036; el exceso B de la propia nube −0.020 →
  −0.040). Como cambian las dos partes, el constructor escribe "The deeper reading comes from both sides: decoupling shrinks the star
  spread and deepens the excess below the star." (`final_dec_obs.json`, kind = both) en lugar de la observación abierta; por qué el
  exceso de la nube crece al desacoplar no lo explican las corridas y no se afirma. Provenance de §5.3 incluye ya
  `expR81_deep_per_backbone.csv`. La ordenación "1.3--2.0" se escribe "ratios 1.3 to 2.0" en aposición: la expresión regular de números
  del sweep no acepta rangos con guiones y la regla de paréntesis no admite "(ratios 1.3 to 2.0)".
- **Cierre (2026-09-21, 10:40)**: §7 en la línea 460 de la página 9, statements en la 485 (página 9), referencias en la 503 (página 10);
  33 páginas, 0 avisos. Recortes para el presupuesto: "Scale deepens the excess for DINOv2 only…" (§5.2), "The reference level, not the
  size, decides what those readings mean" (§5.1), "yet it recovers the CIFAR-100 superclasses" (DeiT-B), la frase de los niveles de
  WordNet y el pie de la Figura 4 apretados, "outside that regime" en el párrafo del implante; saltos de sección 0.6 ex y parskip 0.
  Sweep 215/216: sólo falla el tope de 250 palabras del resumen (276). Versión paralela recompilada sobre el envío.

## 40. Octava revisión (2026-09-21): expR82 y la extensión de expR81 en marcha; tesis reordenada, acotación en una sola frase, ruido definido.

- **Run 1e** `expR82_radial_control.py` (seis shards desde las 11:44): test de profundidad del registro sobre los 12 backbones tras
  (a) normalizar L2 cada centroide y (b) quitar a cada offset su componente radial (proyección sobre la dirección del hub, hub
  intacto); las dos estrellas (hubs gaussianos al radio real, hubs Haar-remuestreados), 10 semillas de estrella, y el control de
  desacoplamiento (10 semillas) sobre cada nube transformada. Regla de decisión del brief codificada en `--merge`
  (`expR82_decision.csv`): si ViT-S/B/L y DINOv2-L mantienen z ≤ −2 bajo (a) o (b), se conserva "alignment of each cluster with its
  hub" y se añade "the alignment survives L2 normalization and the removal of the radial component, so it is not the spread of
  feature norms"; si no, "alignment" pasa a "the radial spread of feature norms within each superclass", el resumen deja la
  afirmación y la limitación (iii) lo dice. Primero se informa al autor, luego se edita. **Run 1d**: expR81 ampliado a 20 semillas
  para DINOv2-S/B/L/G (ocho shards desde las 11:36).
- **(1) Tesis** (resumen y §7): "…hub-aligned in a few backbones, that is moderately shared and does not converge to one common
  tree; no hierarchy above the superclasses is found in the supervised and contrastive backbones, and the DINOv2 family lies beyond
  the noise level at which the test was validated; their raw tree-likeness is not evidence for hyperbolic geometry."
- **(2)** La frase acotada completa, con las razones, una sola vez en §5.3; forma corta "in the supervised and contrastive backbones,
  not in the DINOv2 family, which lies beyond the noise level at which the test was validated" en el resumen, §1 (espejo y
  contribución 2), pie de la Figura 4 y pie de la Tabla 8. "Noise level" definido en §5.3 antes de su primer uso: "The noise level of
  a cloud is its within-cluster spread relative to the distance between its hubs." (primer párrafo de §5.3).
- **(3)** Párrafo del desacoplamiento: "…none of the four fires, and ViT-L reads −1.61, within the −1.61 to −1.76 of the flat control:
  without its orientations the real cloud reads like a flat one" (fundida con la frase anterior para no pasar de seis; −1.61 de
  `expR74_decoupling_summary.csv`, el rango de las filas planas de expR79). Todas las z de §5.3 pasan a dos decimales (la frase del
  sesgo dice ahora −4.19 to −4.82 y −1.61 to −1.76), como el resto del artículo desde la cuarta revisión.
- **(4)** Limitación (iii): "DINO-B (2.1) and ViT-B (2.0) lie at the edge of the covered range." con rellenos de expR64b.
- **(5)** Resumen, tras la frase del nivel de muestra: "The four values reported by Khrulkov et al. for latent hyperbolicity are
  reproduced and, calibrated, two are indistinguishable from a random cloud." (el constructor comprueba en expR78 que exactamente dos
  datasets tienen p máxima > 0.05: CIFAR-10 y CUB). En §1 la primera frase de "The answer has three parts" reescrita en llano ("On
  image features the calibrated reading lies within the noise of a random cloud of the same shape in most cells and removes at most two
  fifths of it elsewhere, so the raw reading is not evidence and the calibrated reading is weak and model-dependent.") seguida de la
  misma frase de Khrulkov con cita; sin "genuine", "certified", "record", "frame" ni "matched star".
- **(6)** "unregime" venía del orden de las sustituciones R3 ("validated regime" se aplicaba dentro de "unvalidated regime");
  "unvalidated regime" va ahora primero. Declaración de uso de IA: mantiene los tres puntos del formulario ("writing assistance",
  "retrieval of references", "research ideation or execution") y la aclaración de la quinta revisión; el texto exacto de las casillas
  no está disponible aquí (ver TODO).
- Presupuesto: "The genuine cells belong mostly to the DINOv2 family…" (§5.1), la cláusula "so a hierarchy carried by the hubs
  survives…" (§5.3) y dos frases apretadas. Sweep: tesis y formas nuevas, la frase de ViT-L, la de los bordes y la de Khrulkov
  rederivadas de sus ficheros, "unregime" ausente, la primera frase del espejo de §1 sin las cinco palabras vetadas.
- **Cierre (2026-09-21, 12:00)**: §7 en la línea 460 de la página 9, statements en la 485 (página 9), referencias en la 503 (página 10);
  33 páginas, 0 avisos. Recortes adicionales para el presupuesto: pie de la Figura 4 y de la Figura 3 apretados, "CIFAR-100 … carries no
  certification" (§5.3, sigue en el pie de la Tabla 8), "Training moves the raw reading within an architecture…" (§5.2, sigue en la
  Tabla 2(e)), "and order the levels by projection radius" (§5.4), "most for DINOv2-G on CIFAR-100" (§5.1), la cláusula de los
  magnitudes del control de clases (§5.2), "so the excess is conservative" en la limitación (ii); textfloatsep 4 pt, abovecaptionskip
  0, saltos de sección 0.5 ex. Sweep 215/216: sólo el tope del resumen (311 palabras). Versión paralela recompilada.

## 41. Correcciones de texto (2026-09-21, tras la octava revisión).

- **(1)** La forma corta del resumen, §1 (espejo y contribución 2), pie de la Figura 4 y pie de la Tabla 8 es ahora exactamente la de
  §7: "no hierarchy above the superclasses is found in the supervised and contrastive backbones; the DINOv2 family lies beyond the
  noise level at which the test was validated." (en la contribución 2 con "is found", como en §7).
- **(2)** §5.3: la frase larga se sustituye por "The control covers the supervised and contrastive backbones (within/between ratios
  {{RATIO_SC_LO}}--{{RATIO_SC_HI}}); DINO-B ({{RATIO_DINOB}}) and ViT-B ({{RATIO_VITB}}) sit at its edge, and the DINOv2 family
  ({{RATIO_DINO_LO}}--{{RATIO_DINO_HI}}) lies beyond it and is not tested." (rellenos de expR64b: 1.3–2.0, 2.1, 2.0, 3.0–3.9). Como
  DINO-B queda situado aquí, la frase "DINO-B (2.1) and ViT-B (2.0) lie at the edge of the covered range" sale de la limitación (iii).
  El sweep lee ahora los rangos con guion como rangos ("--" → "to" antes de contar números); el párrafo pasa a (7, 4).
- **(3)** §5.2: eliminado el puntero "Table 2 gives the training rows" (la frase de entrenamiento no vuelve por el presupuesto; la
  Tabla 2(e) sigue en el apéndice).
- **(4)** Limitación (iii), hasta que expR82 informe: "An alternative reading of the alignment is the radial spread of feature norms
  within each superclass, which would elongate every cluster toward its hub without semantics; a control that removes it is in
  progress." Se sustituirá por el resultado con la regla del brief.
- **(5)** §1, "The answer has three parts": "is genuine in 44 of 72 cells" → "is present in 44 of 72 cells, more than chance would give".
- Presupuesto: el párrafo "The count survives resampling" en dos frases más el puntero; "rather than its hierarchy" fuera de §6.
- **Cierre (2026-09-21, 12:35)**: §7 en la línea 459 de la página 9, statements en la 485 (página 9), referencias en la 503 (página 10);
  33 páginas, 0 avisos; sweep 215/216 (sólo el tope del resumen, 311 palabras). "The count survives resampling" vuelve a tres frases
  ("The census null accounts for replicate noise only. Folding image resampling and the estimator's seeds into that spread, …").
  Versión paralela recompilada.
- **Resumen, primera frase (autor, 2026-09-21)**: "Hyperbolic representation learning rests on a premise we call latent hyperbolicity:
  standard models are already tree-like because their class geometry scores a low Gromov δ, a measure of tree-likeness that is zero
  for a tree." (vuelve "we call" y se añade la definición de δ). 320 palabras; el check del tope de 250 sigue en FAIL a la espera del
  autor. `OPENREVIEW_abstract.txt` regenerado.
- **Resumen, segunda frase (autor, 2026-09-21)**: "That low value is also what a random cloud of the same dimension and spectrum
  scores, so we build an instrument that reads every score as its excess over 200 such clouds, and add a depth test whose power is
  measured." (sustituye a la frase del nivel de referencia). Recuento en la nota de cierre; el tope de 250 sigue pendiente.
- **Resumen completo del autor (2026-09-21, tarde)**: el resumen es ahora el texto que dio el autor (la frase de Khrulkov sale del
  resumen y queda en §1; "A cell is one model on one dataset" pasa a "model–dataset cells"; MERU nombrado; "where the premise is
  read"). Su última frase es la tesis nueva, "Read correctly, foundation models organize classes into clustered structure that is
  moderately shared and does not converge to one common tree; no hierarchy above the superclasses is found where the test has power;
  and their raw tree-likeness is not evidence for hyperbolic geometry.", y por la regla de la tesis idéntica en resumen y §7 se ha
  puesto también en §7 (cae "hub-aligned in a few backbones" de la tesis; el espejo de §1 no cambia). Sweep ajustado a las frases
  nuevas (definición de celda, orientaciones, "The four values" sólo en §1, "weak and model-dependent" al menos dos veces).

## 42. Resumen y §1 reescritos (2026-09-21, brief del autor).

- **(1)** Resumen literal del autor (299 palabras; el tope de 250 sigue pendiente). **(2)** Tesis de §7 = última frase del resumen.
- **(3)** §1 "The answer has three parts": párrafo entero sustituido por el texto del autor (edición registrada sobre `main_local.tex`,
  que ahora abarca el párrafo completo, incluida la frase de los árboles). **(4)** Contribución 2: "Structure beyond a random cloud in
  most models, clusters oriented toward their hubs in a few, no hierarchy above the superclasses where the test has power, and a
  hyperbolic backbone as the control for imposing the geometry."
- **(5)** Donde la alineación se enuncia como hallazgo: entradilla de §5.3 "The depth test certifies that clusters are oriented toward
  their hubs in four backbones.", pie de la Figura 4 "The depth test certifies in 4 of 12 ImageNet backbones that clusters are oriented
  toward their hubs, not that the hubs form a hierarchy; no hierarchy…", pie de la Tabla 8 igual con "under both matched stars".
  "Alignment" se conserva donde se define o se caracteriza (R6 "The alignment is relational…", §5.2 "reads as hub alignment",
  limitación (iii) y el control implantado).
- **(6)** Sweep: tesis nueva; "cell" definido por "model--dataset cells" en el resumen y "A model--dataset cell is genuine when" en la
  Definición 5 (§3.3); vocabulario: "hub-aligned" ausente, "clusters are oriented toward their hubs" al menos tres veces, la primera
  frase del espejo de §1 sin "genuine", "certified", "record", "frame", "matched star" ni "null"; la frase de Khrulkov sólo en §1
  ("two of them are indistinguishable"), "weak and model-dependent" al menos una vez (queda en la entradilla de §5.1).
- **(7)** `OPENREVIEW_abstract.txt` con el mismo texto.
- Presupuesto: "DINO-B does not fall in the island" (§5.4), los nombres de los cuatro datasets en el párrafo de Khrulkov (§5.1; están en
  la Tabla 3(b)) y las descripciones (a)/(b) del pie de la Figura 4 apretadas.

## 43. expR82 integrado: la alineación no es la dispersión radial de las normas (2026-09-21, tarde; brief del autor).

- Antes, `git pull --rebase --autostash origin main`: entra `ef48002` del autor (Figura 1 nueva, `fig1_concept.pdf` en las dos
  carpetas). El autostash reescribió los tres logs rastreados que estaban en escritura (los dos fine-tunings y el shard CLIP/SigLIP
  de expR81): los procesos siguen y sus CSV se actualizan, pero sus `.log` visibles quedan parados a las 16:15; los monitores pasan a
  vigilar los ficheros de salida (caché `vitb_ft_*_imagenet_train.npz`, filas del shard). Lección: no tocar con git los logs de
  procesos vivos.
- **Resultado de expR82 en los cuatro certificados** (los otros ocho, en marcha): quitando la componente radial de cada offset, los
  cuatro mantienen z ≤ −2 bajo las dos estrellas (aniso: ViT-S −2.26, ViT-B −3.60, ViT-L −3.99, DINOv2-L −2.59; Haar-hub: −2.08,
  −3.57, −3.47, −2.28); bajo normalización L2 sólo ViT-B (−2.45/−2.52) y ViT-L (−2.00/−2.23). Regla del brief cumplida por (b):
  "alignment" se conserva.
- **(1)** §5.3, párrafo nuevo tras el del desacoplamiento, "The alignment is not the spread of feature norms.": la frase del autor con
  el rango de z como rellenos de expR82 (aniso, cuatro certificados: {{Z_RAD_HI}} a {{Z_RAD_LO}} = −2.26 a −3.99), el paréntesis "(z
  from … under both stars)" en aposición por la regla de paréntesis, y el punto y coma partido en dos frases para que el párrafo tenga
  tres con el puntero a la Tabla 8. **(2)** Limitación (iii): "The alignment is not the radial spread of feature norms, which a control
  removes without changing the verdicts; full L2 normalization, a stronger transformation, keeps it in two of the four backbones."
  sustituye a la frase de "in progress" ("the sentence before it" leído como la primera parte de esa misma frase; "It certifies
  alignment above its frame only, and its one trained positive control is inconclusive." se mantiene). **(3)** Tabla 8, panel (e)
  (`q_depth`, ambas transformaciones, ambas estrellas, desacoplamiento) entra solo cuando el resumen de expR82 tenga los doce
  backbones. **(4)** Resumen intacto. **(5)** Sweep: rango, "los cuatro bajo ambas estrellas" y "ViT-B y ViT-L sólo" rederivados de
  los ficheros de expR82 (shards mientras no exista el merge), la frase de la limitación, y el panel (e) presente si y sólo si los doce
  están.
- **§1, párrafo 1**: la frase de δ partida en dos ("…which is zero for a metric tree and grows as a metric departs from one. Measured
  on CNN and ViT features…, it comes out low.") y la frase de la figura sustituida por la del observador ("Figure 1 frames the
  question: an observer who sees only a shadow, the raw δ, asks which of three worlds cast it… All three read alike."), ambas como
  ediciones registradas sobre `main_local.tex`; "shadow" sale de la lista de metáforas vetadas fuera de la Figura 1.
- **expR82 completo (2026-09-21, 17:00)**: los doce backbones, ambas transformaciones, 10 semillas de desacoplamiento cada una
  (`expR82_radial_control.csv`, `_summary.csv`, `_decision.csv`: regla cumplida por (b)). Tabla 8, panel (e) entra (12 filas, z bajo las
  dos estrellas y desacoplamiento por transformación). Fuera de los cuatro certificados: ViT-T cruza el umbral al quitar la componente
  radial (−2.06/−2.07) y DINO-B se queda en −1.61/−1.89; ningún backbone no certificado lo cruza bajo L2. Los desacoplados
  transformados no disparan (fracción certificada 0.0–0.3, ViT-T la más alta). Envío recompilado con el panel; sweep 215/216.

## 44. Control positivo entrenado: veredicto (2026-09-21, 20:35).

- Los dos fine-tunings de ViT-B/16 (5 épocas, CE de hoja frente a CE de hoja + CE jerárquica en los cortes 30/6/2, mismos lotes)
  terminaron a las 18:35; expR77 sobre sus centroides y el checkpoint congelado: jerárquico z −2.97/−3.57 (WordNet-30/equilibrado) y
  desacoplado −2.38/−3.51 (10 de 10 en ambos); CE de hoja z −2.45/−2.09 y desacoplado −0.72/−1.38 (0 de 10); congelado z −3.62/−2.82
  y desacoplado −1.40 (0 de 10) / −2.10 (7 de 10). Criterio fijado de antemano NO cumplido: la mitad "jerárquico certificado en ambos
  marcos y dispara desacoplado" se cumple entera, pero la mitad "CE y congelado no certificados" falla porque ViT-B ya es uno de los
  cuatro certificados. Lo que sí separa a los tres es el desacoplamiento en el marco de registro: sólo el jerárquico sigue disparando.
- Como manda el brief de la pista paralela: párrafo "A trained positive control." y tabla en `main_iclr2027_rebuttal.tex` (ancla del
  constructor actualizada a la entradilla actual de §5.3), sin tocar el envío hasta que el autor decida; el sweep rederiva las z y la
  frase del veredicto del CSV y del JSON. Informado el autor antes de cualquier edición.

## 45. expR81 cerrado con 20 semillas en DINOv2 (2026-09-22, 01:10).

- Potencia final por backbone: ViT-T/S/B 0.00, ViT-L 0.80; DINO-B 0.00; DINOv2-S 0.00, DINOv2-B 0.45, DINOv2-L 0.95, DINOv2-G 1.00
  (20 semillas); CLIP-B 0.20, CLIP-L 1.00, SigLIP-B 1.00. Ninguna familia ≥ 0.8. La acotación por familias del envío no se sostiene;
  decisión del autor el 23 (PARALLEL_STATUS). Sin cambios en el envío.

## 46. Pase de cierre (2026-09-22, brief del autor): acotación por backbone, la potencia sigue al backbone, control positivo en el envío.

- **(1)** La acotación por familias sale de sus siete sitios. Resumen y tesis (y §7): "no hub hierarchy is found in the five backbones
  where the control detects an implanted one, and the test is blind in the other seven"; §1 igual tras "whereas an implanted hierarchy,
  synthetic or trained in, survives it"; contribución 2: "no hub hierarchy in the five backbones where the control detects an implanted
  one, a blind test in the other seven"; pie de la Figura 4 y de la Tabla 8 con la forma del resumen; §5.3, párrafo nuevo "The power
  of the test follows the backbone, not its ratio or family." con la lista de expR81 como rellenos ({{P81_COVERED}} = "ViT-L, DINOv2-L,
  DINOv2-G, CLIP-L and SigLIP-B", potencia {{P81_COV_LO}}--{{P81_COV_HI}} = 0.80--1.00; {{P81_UNCOVERED}} = "ViT-T/S/B, DINO-B,
  DINOv2-S/B and CLIP-B", 0.00--0.45; el constructor comprueba que son cinco y siete). "noise level at which the test was validated"
  desaparece en todas partes (el sweep lo exige). La entradilla del párrafo del desacoplamiento pasa a "No hub hierarchy is found where
  the control has power."
- **(2)** "Three supervised ViTs at ratios {{RATIO_VIT3_LO}} to {{RATIO_VIT3_HI}} miss the implant, whereas DINOv2-L and DINOv2-G
  detect it at {{RATIO_DINOLG}}." (1.5 a 2.0; 3.9; de expR64b). Limitación (iii) reescrita: la potencia sigue al backbone, ≥ 0.80 en
  cinco y ≤ 0.45 en los otros siete, el veredicto vale en los cinco; sigue lo del implante de dos niveles y el sesgo del desacoplado;
  "its trained positive control, one seed, discriminates the objectives only once decoupled"; la frase de la dispersión radial.
- **(3)** Control positivo en el envío: párrafo "A trained hierarchy survives decoupling." en §5.3 tras la jerarquía sintética, con los
  recuentos como rellenos de expR77 (10 de 10, 0 de 10, 7 de 10) y la frase sobre el criterio previo tal cual pidió el autor; Tabla 8,
  panel (d) regenerado desde expR77 (sustituye al control de dos pasadas de expR65). Sale de la versión paralela (`phaseE_rebuttal.py`
  lo omite si está en el envío). Resumen: "whereas an implanted hierarchy, synthetic or trained in, survives it".
- Tabla 9, panel (e) nuevo: potencia por backbone de expR81 (razón, semillas, potencia, z medio, potencia y z desacoplados).
- **(4)** Sweep: el recuento de palabras del resumen quita las fórmulas y une los guiones; checks de la acotación por backbone (listas y
  rangos rederivados de expR81), de la frase de la potencia, del control positivo (recuentos de expR77, criterio no cumplido en el
  JSON) y de los dos paneles nuevos; exenciones de los tres párrafos. Lista del revisor: bloque nuevo (potencia por backbone, control
  radial, control entrenado). Presupuesto (§6 y §5.5 primero, luego §5.4): "Table 3 gives the band" (§6), la frase cofenética/tripletes
  de "Controlling the cut", las dos frases de la sonda y las plantillas de "Text depends…" (entradilla "Text depends on recipe and
  scale."), y el párrafo "The two-level implant is missed…" entero (su contenido está en la frase del autor, la Figura 4b y la Tabla 9;
  el puntero a la Figura 4b pasa al párrafo del desacoplamiento).
- **(5)** Segunda semilla de los dos fine-tunings lanzada a las 08:55 (GPU 0 CE, GPU 1 jerárquica, `--seed 1`), sólo para la versión
  paralela; ~22 h.
- **Cierre del pase (2026-09-22, 10:30)**: §7 en la línea 457 de la página 9, statements en la 482 (página 9), referencias en la 500
  (página 10); 34 páginas, 0 avisos. Sweep 216/217: sólo el tope de 250 palabras del resumen (299, contadas sin fórmulas ni guiones).
  El check de conservación de decimales del apéndice viejo exceptúa B39 (control de dos pasadas de expR65, sustituido por expR77).
  Versión paralela = envío congelado sin añadidos hasta que llegue la segunda semilla. Lista del revisor regenerada con el bloque nuevo.

## 47. Novena revisión (2026-09-22, brief del autor): un solo criterio de potencia, frame balanceado, vocabulario, Figura 4b.

- **Run previo, expR83** (`expR83_flat_balanced.py`, CPU, diez shards a nice 5, 10:05–11:11): el control plano de expR79 (espectro y
  razón de ViT-L, 5 semillas) leído en el frame balanceado de treinta superclases (enlace elegido en `expR67_frame_choice.json`, como
  en expR77), intacto y desacoplado (10 semillas, 50 runs). Dos construcciones: la emparejada, con los hubs planos asignados por el
  propio frame balanceado (como el 8 de 50 lo está a WordNet-30), y la literal, los clouds de expR79 con hubs por WordNet-30 leídos en
  el balanceado. Resultado: emparejada 0 de 5 intacto (z medio −0.46) y 0 de 50 desacoplado (z medio −0.65, rango +0.26 a −1.52);
  literal 5 de 5 intacto (z medio −3.13) y 38 de 50 desacoplado (z medio −2.33, rango −1.56 a −3.75). La literal mide el desajuste
  de frames, un cloud agrupado por otro frame dispara ya intacto, no profundidad; la tasa que va al texto es la emparejada. Tabla 9,
  panel (f) nuevo con las dos filas y su nota. Ficheros: `expR83_flat_balanced.csv`, `expR83_flat_balanced_summary.csv`.
- **(1)** Criterio único, la potencia desacoplada de la Tabla 9(e) (≥ 0.8): nueve cubiertos (ViT-S/B/L, DINOv2-B/L/G, CLIP-B/L y
  SigLIP-B; 0.83 a 1.00) y tres ciegos (ViT-T, DINO-B y DINOv2-S; 0.01 a 0.60). El builder rederiva las listas de `dec_power` (el
  compresor por familia admite CLIP) y comprueba nueve y tres. Forma larga "no hub hierarchy is found in the nine backbones where the
  decoupled control detects an implanted one, and the test is blind in the other three" en la frase propia del resumen, en §1 tras
  "survives it;", en la contribución 2 ("a blind test in the other three"), en §5.3 ("so no hub hierarchy is found in the nine and the
  test is blind in the other three") y en los pies de la Figura 4 y la Tabla 8; forma corta "no hub hierarchy is found where the test
  has power" como tesis en el resumen y en §7. La tasa de falsas alarmas desacoplada al lado: §5.3 "Its false-alarm rate on the flat
  control are {{FA_DEC}} of 50 decoupled runs on the frame of record." y limitación (iii) "Under the decoupled control it is
  {{P81_COV_LO}} or more in nine backbones and {{P81_UNC_HI}} or less in the other three. The verdict of no hub hierarchy therefore
  holds in the nine only, at {{FA_DEC}} of 50 false alarms on flat clouds." La frase de
  las razones pasa a "The three blind backbones sit at ratios {{RATIO_BLIND_LO}} to {{RATIO_BLIND_HI}}, inside the {{RATIO_COV_LO}}
  to {{RATIO_COV_HI}} of the nine covered." (1.5 a 3.0 dentro de 1.3 a 3.9, de expR64b; el builder comprueba la inclusión y que
  cada familia tiene cubiertos). Los rellenos RATIO_VIT3_* y RATIO_DINOLG desaparecen; el párrafo
  apunta a la Tabla 9 y a la Figura 4b.
- **(2)** ViT-B congelado en el frame balanceado: la tasa emparejada es baja (0 de 50), así que entra la segunda rama del brief, tras
  "On the balanced frame the frozen checkpoint also fires in 7 of 10.": "That frame's flat control fires once decoupled in 0 of 50 runs,
  so ViT-B shows hub structure under that grouping and the frame of record does not." Regla registrada en `final_bal_frame.json`:
  "alta" si la tasa es al menos el doble de la del frame de registro (16 de 50 o más); el sweep reproduce la regla y la frase. Matiz
  para el autor en el TODO: la construcción literal enseña que un cloud organizado por un frame dispara en el otro incluso desacoplado
  (38 de 50), así que el 7 de 10 del ViT-B congelado en el balanceado admite también esa lectura de desajuste.
- **(3)** B.7: el párrafo "A trained control, inconclusive." (control de dos pasadas, expR65) sale del apéndice; en su lugar "A trained
  positive control." con puntero a la Tabla 8(d).
- **(4)** Vocabulario: Definición 6 "measures structure above the clusters of the frame only"; entradillas "Leaf labels can produce the
  alignment but do not guarantee it." e "Imposing the geometry does not create hub structure."; contribución 1 "a nominally hyperbolic
  backbone as the control for imposing the geometry"; §5.4 "Under it the island shrinks to a moderate gap: the sibling-triplet
  agreement of DINOv2 with the block is {{TRIP_BIG}} against {{TRIP_WITHIN}} within the block. About a third of the within-block
  agreement is missing under the admissible configurations as a whole." (0.77 y 0.76 de `expR58_treemap_cutfree_summary.csv`, fila
  imagenet/cosine/average). La mediana del "about a third" (0.34, `final_pass_island_gap.json`) es sobre las doce medidas de las cuatro
  configuraciones admisibles, la seleccionada incluida; sobre las otras nueve medidas sale 0.50, por eso el texto dice "as a whole" y
  no "the other" (decisión del autor pendiente, TODO).
- **(5)** §6: la banda 0.104/0.061/0.046 de la Tabla 3 es la columna `delta_max` de `exp1_delta_controls.csv`, el supremo muestreado
  (línea 77 del generador), y las curvaturas {{C_LO}} a {{C_HI}} ya se calculaban con ella (assert del builder), así que no hay nada que
  recalcular. El texto lo dice: "The rule, calibrated with the supremum, assigns structureless clouds on the Gaussian band, itself the
  sampled supremum rather than the census percentile, a curvature from…"; pie de la Tabla 3: "Gaussian row: the sampled
  supremum from the released control file, the statistic the curvature rule uses".
- **(6)** Pies sin IDs de experimento: Tabla 8(a′) "the decoupling control of expR74" → "the decoupling control"; Tabla 10(b) "(expR59)"
  y "(expR73)" fuera; los ficheros siguen en la línea `% prov:` de cada tabla y en los comentarios `% source`. Final de la prueba de la
  Proposición 1(b): "Part (b) is the dimension confound alone. The statistic confound is separate and elementary: a maximum over a
  growing sample of quadruples is non-decreasing in the sample size, so the sampled supremum can only rise with the budget." Moreira,
  Marques, Costeira y Hauptmann (WACV 2024, pp. 2082–2090; entrada `moreira2024hyperbolic` verificada en CVF open access) en el tercer
  párrafo de §6: "In few-shot learning a fixed-radius Euclidean encoder does at least as well as hyperbolic prototypes
  \citep{moreira2024hyperbolic}."
- **(7)** Figura 4b: doce barras de potencia desacoplada por backbone (`dec_power` de expR81, colores de familia, línea en 0.8, rellenas
  a 0.8 o más con el valor dentro); leyenda común "certified (z ≤ −2) or power ≥ 0.8" / "not certified or power below 0.8"; pie "(b)
  Decoupled power per backbone for an implanted three-level hierarchy at the backbone's own spectrum and noise level, filled at 0.8 or
  more; line at 0.8." Las curvas del implante de dos niveles pasan a `fig_implant_final` en el apéndice, delante de la Tabla 9
  (`\label{fig:implant}`), con pie desde `final_fig_implant.json` (detección 0.05 a plena fuerza con la dispersión real contra 1.00 con
  la reducida; 0.00 en s = 0 con la real; el builder lo comprueba). El puntero de §5.3 "Figure 4b the detection rates" pasa a la figura
  del apéndice. `final_fig4.json` registra panel_b = decoupled_power_per_backbone y los nueve cubiertos.
- **Sweep**: split nueve/tres rederivado de `dec_power`; literales nuevos en los checks del cierre, de la séptima revisión, de 1b y de
  las decisiones; tesis corta dos veces; forma larga dos veces en el cuerpo (§1 y pie de la Figura 4); check nuevo de la novena revisión
  (nueve/tres, ningún "five backbones"/"other seven"/"in the five only" en el tex, la tasa desacoplada junto al split en §5.3 y (iii),
  la frase del frame balanceado igual al JSON y a la regla, vocabulario, 0.77/0.76 de expR58, supremo en §6 y Tabla 3, Moreira en la
  bib con páginas, final de la prueba, ningún `exp…` en ningún pie, `final_fig4.json`, figura del implante en el apéndice y exactamente
  un `\includegraphics` allí, B.7). HEADLINE con los rangos de potencia y de razones, "0.77 against 0.76" y el "0 of 50", leídos de los
  ficheros; exenciones (5, 2) para "The power of the test follows…" y "A trained hierarchy survives decoupling."; frase de 35+ palabras
  exenta "The decoupled control detects an implanted hierarchy in".
- **Presupuesto**: con los siete puntos el texto pasaba cuatro líneas a la página 10. Recortes, §6 primero: la frase de la regla
  objetiva contra coseno ("The objective rule beats cosine by +0.41 against +0.28 pp…; among non-circular policies cosine is best.",
  el puntero pasa a "Table 13 gives both predictions and the policies."), la frase del supremo fundida en una ("The rule, calibrated
  with the supremum, assigns structureless clouds on the Gaussian band, itself the sampled supremum rather than the census
  percentile, a curvature from…"), Moreira en una frase corta; en §5.3 sobra "and every family has covered members" y en (iii) las dos
  frases nuevas se aprietan ("Under the decoupled control it is 0.83 or more in nine backbones and 0.60 or less in the other three.
  The verdict of no hub hierarchy therefore holds in the nine only, at 8 of 50 false alarms on flat clouds."). Los rellenos
  POL_RULE_H/POL_COS_H quedan sin uso. Tablas 8(d) y 9(f) estrechadas (cabeceras "dec. z", etiquetas cortas, colsep 3pt): sin overfull.
- **Cierre del pase (2026-09-22, 11:25)**: §7 en la línea 463 de la página 9, statements en la 487 (página 10), referencias en la 506
  (página 10); 34 páginas; 0 `??`. Sweep 216/217: sólo falla el tope de 250 palabras del resumen (ahora 300 tokens brutos en
  `OPENREVIEW_abstract.txt`, unas 285 palabras sin fórmulas), decisión del autor pendiente. Versión paralela reconstruida sobre el
  envío nuevo (34 páginas, 0 `??`, sin párrafos insertados porque todo está en el envío; sólo la subsección del apéndice). Lista del
  revisor regenerada (anclas de la tercera revisión que ya no existen, cosmético). Commit y push.

## 48. Novena revisión, continuación (2026-09-22, tarde, brief del autor): frase del frame balanceado, tope del resumen a 300.

- **§5.3, frase del frame balanceado**, redacción del autor en lugar de la rama de la regla: "That frame's matched flat control fires
  once decoupled in {{FA_BAL}} of 50 runs, so the {{PC_FROZEN_BAL}} of 10 is not a false alarm: ViT-B shows hub structure under that
  grouping that the frame of record does not resolve; and a cloud organized under one frame fires under the other even once decoupled
  (38 of 50), so frames are not interchangeable." Los tres recuentos vienen de los ficheros, embebidos por el builder en {{BAL_SENT}} (0 de 50 y 38 de 50 de
  `expR83_flat_balanced_summary.csv`, 7 de 10 de expR77); el builder exige que la tasa emparejada quede por debajo de la barra de la
  regla (dos veces el 8 de 50) para "not a false alarm" y que la construcción literal dispare en más de la mitad de los runs para
  "fires under the other even once decoupled"; `final_bal_frame.json` guarda la frase con `author_sentence`. Sweep: la frase
  reconstruida desde los ficheros, exención (6, 3) del párrafo "A trained hierarchy survives decoupling." (seis recuentos, tres en la
  frase del autor), la frase exenta del tope de 35 palabras como frase del autor, "38 of 50" en HEADLINE.
- **§5.4**: "under the admissible configurations as a whole" se queda, por decisión del autor.
- **Sweep, tope del resumen**: 300 palabras (sin fórmulas ni guiones dobles), decisión del autor; el check pasa (unas 285).
- **Mañana**: integrar la semilla 1 de los dos fine-tunings con la regla ya escrita (`expR77_positive_control_tests.py --models
  ce_seed1 hier_seed1`, versión paralela, PARALLEL_STATUS; al envío sólo si el autor lo decide tras el informe).
- **Presupuesto**: la frase del autor añade dos líneas y el texto volvía a pasar a la página 10; recortes, §6 primero: fuera el puntero
  "Table 13 gives the gains." del tercer párrafo (el segundo ya apunta a la Tabla 13, ahora "Table 13 gives both."); §5.4: "The cosine
  census agrees with the Euclidean one in most cells" y los dos punteros de "Controlling the cut" fundidos en uno; §5.3: el párrafo de
  la potencia apunta sólo a la Tabla 9 (la Figura 4b lo dice en su pie).
- **Sweep**: el check del resumen conservaba un literal de la sexta revisión ("no hierarchy above the superclasses is found") que el
  tope de 250 venía enmascarando; ahora pide la cláusula de la tesis ("no hub hierarchy is found where the test has power").
- **Cierre (2026-09-22, 15:15)**: §7 en la línea 464 de la página 9, statements en la 487 (página 10), referencias en la 506 (página
  10); 34 páginas; 0 `??`; resumen a 300 palabras justas sin fórmulas. Sweep 217/217. Versión paralela reconstruida.
  Monitor persistente sobre las cachés de la semilla 1 (`Platonic/results/practical_tasks_cache/vitb_ft_{ce,hier}_seed1_…npz`).

## 49. Limpieza de pasada completa (2026-09-22, tarde, brief del autor): duplicados, Definición 8, §5.3 en siete párrafos, tabla de Khrulkov.

- **(1) Duplicados.** §3.3: la entradilla "The rank of the reading is the evidence." y las dos frases siguientes pasan a la frase del
  autor "The excess is a size; the evidence is the rank of the reading among its replicates and the left-tail $p$ that the rank gives,"
  (sigue la ecuación del rango); como cada párrafo lleva entradilla en negrita por regla, la nueva es "Size and evidence." (mía). La
  entradilla "A negative genuine excess means structure beyond the second moments." repetía la frase que sigue; ahora "What a genuine
  excess means." (mía) y se conserva sólo "A negative genuine excess certifies structure beyond the second moments; clustering is its
  most plausible reading…". §4: fuera "Every reading is calibrated against two hundred replicates."; entradilla nueva "Calibration budget
  and frames." (mía) y se conserva "Each cell is read against two hundred Haar replicates…". §5.2: fuera "The same structure appears
  without labels and in language models.". §5.3: "4 of 5", "8 of 50" y la frase del sesgo del desacoplado aparecen una sola vez.
- **(2) Definición 8 (decoupling control)**, tras la estrella emparejada, con la redacción del autor: "The decoupling control keeps the
  hubs of the frame and rotates each cluster's offsets by an independent Haar rotation; a verdict that survives the decoupling is
  carried by the arrangement of the hubs, one that does not is carried by the orientation of the clusters relative to their hubs,
  which we call alignment." Después, en el párrafo del test: "The power of the test is the fraction of implanted hierarchies it
  detects and its false-alarm rate the fraction of flat controls that fire; both are measured on real clouds in Section 5.3." (sustituye
  la cláusula anterior del mismo párrafo). Ocho definiciones; el sweep las cuenta y exige la Definición 8 entre la 7 y el párrafo.
- **(3) §5.3 en siete párrafos**, cada número una vez: (a) "The depth test certifies that clusters are oriented toward their hubs in
  four backbones." (sin cambios); (b) "What it certifies is alignment.": desacoplado (ViT-L −1.61 dentro de −1.61 a −1.76), relacional
  (0 de 60 / 0 de 4), control radial (−2.26 a −3.99, L2 sólo ViT-B y ViT-L; las dos frases del autor tal cual); (c) "What the test can
  see.": jerarquía sintética (frase del autor con el 4 de 5, sobrevive al desacoplado, más profunda desacoplada y la observación de
  expR81), potencia desacoplada por backbone (nueve a 0.83–1.00, tres a 0.01–0.60; razones 1.5 a 3.0 dentro de 1.3 a 3.9: la potencia
  sigue al backbone), control entrenado (10 de 10 contra 0 de 10 en el frame de registro; frase del criterio previo), sesgos (control
  plano 0 de 5 intacto y 8 de 50 desacoplado; la frase del sesgo con −4.19 a −4.82; frame balanceado emparejado 0 de 50 y desajustado
  38 de 50), Poincaré no dispara, puntero a las Tablas 9 y 8; (d) "No hub hierarchy is found where the test has power.": los nueve y los
  tres con nombre, la profundidad de los implantes frente a los catorce niveles de WordNet, puntero a la figura del implante y a la
  Tabla 9; (e) "The certified set depends on the frame.": frames, sólo ViT-B y ViT-L bajo todos, y el ViT-B congelado en el balanceado
  con la frase acordada; (f) "Leaf labels…"; (g) "Imposing the geometry…". Desaparecen las entradillas "No hub hierarchy is found where
  the control has power.", "The power of the test follows…", "The alignment is not the spread…", "A deep hierarchy at the real noise
  level is detected.", "A trained hierarchy survives decoupling." y "The test never fires on randomized hubs." (su contenido está en
  b, c y d). Dos desviaciones que señalo: (i) "each number once" obliga a que el 0 de 50 y el 38 de 50 estén sólo en (c), así que la
  frase acordada de (e) va sin sus dos recuentos entre paréntesis y con "so this is not a false alarm" en lugar de "so the 7 of 10 is
  not a false alarm", precedida de "On the balanced frame the frozen ViT-B fires once decoupled in 7 of 10 seeds."; el builder la
  genera con los recuentos de expR77/expR83 y `final_bal_frame.json` la registra como adaptada; (ii) el párrafo (c) tiene doce frases
  por diseño del brief, así que queda exento de la regla de 3–6 frases (registrado en el sweep, `SENT_EXEMPT`), igual que los dos
  párrafos de §5.2 recortados a dos frases.
- **(4) Tabla de Khrulkov en §5.1** (`tab_khrulkov_final.tex`, generada por `gen_appendix_final.py` desde
  `expR78_khrulkov_replication_summary.csv`; float [t], cuatro filas: dataset, su δ_rel, la nuestra con su estimador, exceso, r/200 y
  p mayor), citada por el párrafo ("Table 3 gives the four rows" apunta ahora a ella; el panel (b) de la tabla de muestra del apéndice
  se conserva con las desviaciones). Sitio: el párrafo de neural collapse queda en dos frases y "The count survives resampling." en
  dos ("Folding image resampling and the estimator's seeds into the null spread, or repeating the census on resampled centroid sets,
  leaves most of the count in place. Table 10 gives the counts and the bootstrap.").
- **(5) §3.2**: fuera "Gromov δ is the worst case, the supremum of the defect over all quadruples" (edición registrada número 13 sobre
  `main_local.tex`); la cita a Gromov (1987) pasa a la Definición 1.
- **Sweep**: definiciones ocho y término "decoupling control" definido una vez; orden de las siete entradillas de §5.3; cada recuento
  (4 de 5, 8 de 50, 0 de 50, 38 de 50, 7 de 10, 10 de 10, 0 de 10, 0 de 60, 0 de 4, 0 de 5) exactamente una vez en §5.3 y la frase
  del sesgo una vez; literales de los checks de 1b, expR81, control positivo, séptima y novena revisión rehechos con las frases
  nuevas (los números siguen rederivados de sus ficheros); celdas de la tabla de Khrulkov comparadas con expR78; exenciones (5, 2)
  para (b) y (12, 2) para (c). `phaseE_rebuttal.py` detecta con las frases nuevas que expR77 y expR79 ya están en el envío.
- **Cierre (2026-09-22, 15:35)**: §7 en la línea 464 de la página 9, statements en la 487 (página 10), referencias en la 506 (página
  10); 34 páginas; 0 `??`; la tabla de Khrulkov cabe sin recortes adicionales (los de §5.2 y la fusión de §5.3 compensan). Sweep
  218/218 (dos checks nuevos: la limpieza y la tabla). Versión paralela reconstruida sobre el envío nuevo (34 páginas, 0 `??`).
  Un check del sweep comparaba el bloque verbatim de §3.2 sin aplicar las ediciones registradas; ahora las aplica (la número 13).
  Numeración: la tabla de Khrulkov es ahora la Tabla 1 y el censo la Tabla 2, y las del apéndice corren una posición (todo va por
  `\ref`; los nombres de los checks del sweep y las notas anteriores de este registro siguen la numeración vieja, Tabla 8 = depth, Tabla 9
  = power). El rango de razones de los tres ciegos es 1.5 a 3.0 (ViT-T 1.55 redondea a 1.5 en el relleno).

## 50. Décima revisión (2026-09-22, noche, brief del autor): desajuste de frames, qué se sabe de la alineación, techo intra-modelo (expR84).

- **Run previo, expR84** (`expR84_tree_ceiling.py`, CPU, tres shards, unos 4 min): para cada uno de los 12 backbones de ImageNet, los
  30 conjuntos de centroides bootstrap de expR59 (RandomState(b), las 100 imágenes cacheadas por clase remuestreadas con reemplazo),
  dendrogramas bajo la configuración seleccionada (coseno, average; comprobada desde `exp23_config_diagnostics.json`) y las tres
  medidas de expR58 entre los 435 pares de remuestreos del mismo modelo (ARI al corte de 30, correlación cofenética, acuerdo en las
  mismas 10^4 tripletas de expR58). Techo de tripletas: media sobre modelos 0.902, backbone más bajo 0.849 (DINOv2-L); seis
  backbones por debajo de 0.9 (ViT-S 0.88, ViT-B 0.86, DINOv2-S 0.86, -B 0.86, -L 0.85, -G 0.88); ARI al corte 0.79 de media,
  cofenética 0.94. Ficheros: `expR84_tree_ceiling.csv`, `expR84_tree_ceiling_summary.csv`; Tabla 11, panel (c) nuevo.
- **(1) Frame balanceado**: la frase de (e) es la lectura de desajuste del autor, con los dos recuentos como rellenos: "A flat cloud
  clustered under the frame of record fires under the balanced frame in 38 of 50 decoupled runs, so a real cloud read under a frame
  that is not its own is expected to fire; the frozen ViT-B firing in 7 of 10 there is consistent with that mismatch and is not
  evidence of hub structure." En (c) queda sólo el 0 de 50 emparejado (cada número una vez).
- **(2) Qué se sabe de la alineación**: en (b) "The alignment is relational, needing both the real hubs and the real orientation of each
  cluster relative to them: randomizing either removes it, 0 of 60 with random hubs and 0 of 4 with random orientations." y, tras el
  control radial, "The implanted principal-axis alignment reproduces the verdict in the supervised ViTs and CLIP-B but not in DINOv2-L,
  and the exact geometric form of the alignment is not resolved here." En (iii): "The alignment is thus relational, not radial, and
  reproduced by a principal-axis implant in the supervised ViTs and CLIP-B but not in DINOv2-L; its exact geometric form is not
  resolved here."
- **(3) Qué separa el desacoplado**: en (d) "The decoupling separates structure among the hubs of the frame from structure carried by
  cluster orientations. A hierarchy below the frame expressed in how clusters open would be removed with the orientations and counted
  as alignment, so no hub hierarchy means no hierarchy among the frame's hubs."; en (iii) "It certifies alignment above its frame
  only: a hierarchy below the frame … so no hub hierarchy means none among the frame's hubs." Resumen y §1: "not that the hubs of the
  frame form a hierarchy"; el mismo titular en los pies de la Figura 4 y de la Tabla 9, por coherencia. El resumen sube a
  303 palabras sin fórmulas, tres por encima de tu tope de 300 (decisión pendiente, TODO).
- **(4) Techo y tesis**: la regla del brief se cumple por poco (0.902 ≥ 0.9), así que la tesis no cambia ("moderately shared", "does
  not converge to one common tree") y §5.4 dice el techo: "Both sit below the within-model ceiling of 0.90, the agreement two
  resamples of the same model reach, 0.85 for the lowest backbone." (rellenos de expR84; el builder exige la rama y que 0.77 y 0.76
  queden por debajo del mínimo; `final_tree_ceiling.json` registra la rama). Aviso al autor: es marginal, seis de doce backbones
  tienen su techo por debajo de 0.9.
- **(5) Limitación (vi)** nueva al final de las del instrumento: "The analysis rests on many choices, frames, stars and null variants
  among them, and a single pre-specified analysis is future work."; las de alcance pasan a (vii)–(ix).
- **(6) Tabla 14(b)** (correlaciones del supremo bruto con la ganancia, IC bootstrap y control de familia) restaurada: sale de `DROP`
  en el builder; el sweep la exige.
- **Sweep**: check nuevo de la décima revisión (frase de (e) desde los ficheros, frases de (b), (d) y (iii), "hubs of the frame" en
  resumen, §1 y pies, techo desde expR84 con la rama, panel (c) de la tabla q6, limitación (vi) y renumeración, Tabla 14(b));
  exenciones (3, 2) para "Controlling the cut…" y de 35 palabras para las dos frases largas del autor; techo en HEADLINE.
- **Presupuesto**: las adiciones (b, d, iii, vi, §5.4) pasaban diez líneas a la página 10. Recortes, §6 y §5.5 ya al mínimo de tres
  frases: en (d) sobra la frase de los niveles de WordNet (su punto lo hacen las dos frases nuevas del desacoplado), en (b) y (c) los
  punteros se acortan, §3.3 pierde dos frases que duplican la limitación (ii) y el pie de la Figura 2 ("The null moves with the
  dimension…", "The spectrum null conditions on the second moments…"), §3.4 la frase de la estrella que §5.2 ya da y la del frame que
  §4 ya da, §3 la lista de geometrías de referencia (queda "confirms this (Table 3)") y "proved in Appendix A", §4 "which also read
  DBpedia Classes and HierarCaps", §5.2 "Every family contributes genuine cells." y el puntero del censo, (a) "It never fires in the
  other direction.", (f) "and taxonomy alignment is recipe-dependent", (g) "The control therefore tests the objective rather than a
  curved geometry.", §5.4 "for DINOv2-L and DINOv2-G" y el criterio de selección referido a §3.5 (queda "Under the selected
  configuration, cosine distances with average linkage on ImageNet"); pies de las Figuras 2, 4b y 5c acortados; la tabla de Khrulkov a
  `\footnotesize`. Ninguna cifra ni frase del autor tocada.
- **Sweep**: la regla de la copia final de la tabla del corolario vuelve a exigir todos sus números (el panel (b) ya no se descarta);
  el check de la séptima revisión deja de prohibir "frame" en el arranque de §1 (tu "hubs of the frame"); "The alignment is
  relational, needing both the real hubs and the orientation…" a 35 palabras; exención (2, 2) para el párrafo del frame (tu frase con
  dos recuentos); el literal "its trained positive control" pasa a "trained positive control" (ahora empieza frase).
- **Cierre (2026-09-22, 16:05)**: §7 en la línea 455 de la página 9, statements en la 484 (página 9), referencias en la 503 (página
  10); 36 páginas; 0 `??`. Sweep 218/219: sólo falla el tope de 300 palabras del resumen (303 con "hubs of the frame"; TODO). Versión
  paralela reconstruida (36 páginas, 0 `??`); resumen exportado a OpenReview (303 palabras).

## 51. Tesis y §5.4 tras expR84 (2026-09-22, noche, brief del autor): topología compartida, métrica no.

- **(1) Tesis** (última frase del resumen y §7): "Read correctly, foundation models organize classes into clustered structure whose tree
  topology is largely shared, up to a gap beyond resampling noise, and whose metric is not; no hub hierarchy is found where the test
  has power; and their raw tree-likeness is not evidence for hyperbolic geometry." Frase del compartir en el resumen: "Across models
  the tree topology is largely shared and the metric is not: the naive comparison manufactures an island, triplet agreement reaches
  {{TRIP_BIG}} against a within-model ceiling of {{CEIL_TRIP_LO}}--{{CEIL_TRIP_HI}} once the clustering cut is controlled, and the
  self-supervised tree is angular." (0.77 de expR58; el techo por backbone de expR84 va de 0.85 a 0.96, no a 0.95 como en el brief:
  DINO-B 0.965 y ViT-L 0.955; los rellenos siguen al fichero).
- **(2) §5.4**, párrafo "Controlling the cut leaves a moderate gap.": el techo por medida con su rango entre backbones y su media,
  junto a los valores entre modelos de expR58 (fila coseno-average): tripletas {{CEIL_TRIP_LO}} a {{CEIL_TRIP_HI}}, media
  {{CEIL_TRIP}}, contra 0.77 (DINOv2 contra el bloque) y 0.76 (dentro del bloque); cofenética {{CEIL_COPH_LO}} a {{CEIL_COPH_HI}},
  media {{CEIL_COPH}}, contra 0.48 y 0.78; ARI al corte {{CEIL_ARI_LO}} a {{CEIL_ARI_HI}}, media {{CEIL_ARI}}, contra 0.39 y 0.48.
  Del fichero: tripletas 0.85–0.96 (media 0.90), cofenética 0.91–0.97 (media 0.94), ARI 0.64–0.94 (media 0.79). El brief traía
  "ARI 0.64–0.79" y "cophenetic 0.91–0.94", que son el mínimo y la media, no el rango; el texto da rango y media, ambos del fichero.
  Lectura: "The tree topology is therefore shared to within most of the ceiling and the metric agrees at about half of it."
  (0.77/0.90 = 0.86; 0.48/0.94 = 0.51; 0.39/0.79 = 0.49; el builder exige ≥ 0.8 y 0.4–0.6, y `final_tree_ceiling.json` guarda los
  tres cocientes). "the island shrinks to a gap beyond resampling noise" en lugar de "to a moderate gap". Se conservan "About a third…
  as a whole." y el puntero.
- **(3) §1**: la frase del compartir del párrafo "The answer has three parts" es la del resumen (edición registrada, con rellenos).
  Contribución 3: "The island is an artifact of the cut; the tree topology is largely shared and the metric is not, and the
  self-supervised tree is angular." (edición registrada número 14 sobre `main_local.tex`).
- **(4) Sweep**: tope del resumen 310; tesis nueva (dos veces exactas); literal de las decisiones actualizado; check nuevo con las
  frases del resumen, §1, contribución 3 y §5.4 rederivadas de expR58 y expR84 y los tres cocientes; exención (12, 4) y de recuento de
  frases para "Controlling the cut…" (siete frases, doce cifras, por diseño del brief); rangos, medias y pares en HEADLINE.
- **Presupuesto**: el párrafo de §5.4 crece tres líneas y el texto pasaba seis a la página 10. Recortes en colas de párrafo (sin
  tocar cifras ni frases del autor): §4 "Table 15 lists the panel." fundido en la frase anterior, "Two self-supervised ResNets" y
  "DeiT-B and the augreg ViT-B, trained on ImageNet-1k leaf labels"; §5.2 fuera "On class centroids the excess is negative in almost
  every cell." (la Figura 3 lo enseña) y "keeps most of the count … Table 5 gives the bootstrap."; §5.3 (b) fuera el puntero "Table 9
  gives both." (la Tabla 9 se cita en a, e, f y g) y "its exact geometric form"; (e) "Under a balanced frame ViT-T joins … and other
  WordNet cuts change the set again."; §5.4 fuera "with shuffle and Gaussian controls at zero" (están en la Tabla 12). Sweep:
  "largely" sale de la lista de palabras prohibidas (tu tesis dice "largely shared").
- **Cierre (2026-09-22, 16:20)**: §7 en la línea 457 de la página 9, statements en la 487 (página 10), referencias en la 506 (página
  10); 36 páginas; 0 `??`. Sweep 219/220: sólo falla el tope del resumen, que con tu frase nueva queda en 318 palabras sin fórmulas,
  ocho sobre el tope de 310 (TODO). Versión paralela reconstruida; resumen exportado a OpenReview.

## 52. Pasada consolidada (2026-09-22, noche, brief del autor): resumen, tesis, glosas de §1, frases, Figura 5.

- **(1) Resumen** verbatim del autor (con {{N_GEN}} = 44 y {{N_IN}} = 4 como rellenos, como siempre). Tiene 361 palabras sin fórmulas,
  51 por encima del tope de 310 que fija el mismo brief; el check del tope falla y no toco el texto (TODO). El assert del builder
  sube a 400 para poder construir.
- **(2) Tesis** = última frase del resumen, verbatim en §7; check de la tesis actualizado (dos veces exactas, y la última frase del
  resumen debe ser la tesis). Resumen exportado a OpenReview.
- **(3) §1**: glosas en el primer uso, "the superclass centers, which we call hubs" y "the grouping of classes into superclasses, which
  we call the frame"; "decoupled control" pasa a "once cluster orientations are randomized" en el párrafo de la respuesta y en la
  contribución 2; la frase del compartir del párrafo y la contribución 3 espejan la del resumen ("the trees agree on which classes
  group together almost as well as two readings of the same model do, but not on the distances between them"). El sweep exige las
  glosas antes del primer uso y que "decoupled control" no aparezca en §1.
- **(4) Frases de más de 45 palabras en §3–§6** (sin listas de citas): sólo dos, ambas partidas en una afirmación por frase: la
  Definición 8 ("… Haar rotation. A verdict that survives the decoupling is carried by the arrangement of the hubs; one that does not
  …") y la lectura de desajuste de frames de §5.3(e) ("… is expected to fire. The frozen ViT-B firing in 7 of 10 there …"). Regla
  nueva en el sweep: ninguna frase de más de 45 palabras en §3–§6 salvo las que llevan cita.
- **(5) Figura 5** redibujada en el lenguaje de las Figuras 3 y 4 (`make_figs_final.py`): (a) ARI medio de cada modelo con los otros
  once, hueco bajo la configuración ingenua y relleno bajo la seleccionada, unidos por un segmento (exp23); (b) acuerdo de tripletas
  bajo la seleccionada (expR58) contra la banda del techo intra-modelo de expR84 (rango sobre pares de remuestreos, raya en la
  media); (c) sin cambios. Pie del autor. Las dos matrices de ARI pasan al apéndice como figura (`fig_treemap_matrices_final`, junto a
  la Tabla 11) con su pie y los dos valores (0.03 y 0.38). La figura queda anclada ([t]) al principio de §5.4 y la primera frase de
  la sección la cita con la frase del autor; el check del puntero exime ese párrafo. `final_fig5_values.json` guarda además las
  medias por modelo y las bandas.
- **Sweep**: literales del resumen, §1, contribución 2 y 3, Definición 8 y frase del frame rehechos; forma larga "no hub hierarchy is
  found in the nine backbones where the decoupled control…" sólo en el pie de la Figura 4 (en §1 va con la glosa); check nuevo de la
  pasada; dos figuras en el apéndice; nombres de los checks con la numeración actual de tablas (clave arriba).
- **Presupuesto**: el resumen nuevo ocupa seis líneas más en la página 1 y las glosas de §1 dos, y el texto pasaba doce líneas a la
  página 10. Recortes sin tocar cifras ni frases del autor: Figura 5 compacta (2.1 in de alto, tres paneles en el ancho de línea),
  pies de la Tabla 1 y de las Figuras 3 y 4 acortados, §3 "as Proposition 1 states for a structureless cloud" y la lista de
  geometrías de referencia fundida en la frase anterior, §3.3 fuera "Rank-based calibration gives the score its finite-sample
  validity." y "The word tree-like is reserved…" y la cláusula de $n$ pequeño frente a $d$, §3.5 fuera "Each model's hierarchy is a
  dendrogram over the shared class centroids." y el detalle del nulo por permutación, §3.6 las dos frases de los readouts fundidas en
  una, §4 "in three sizes" (MERU), §5.2 "the two sets with a real hierarchy", §5.3 (g) "where Lorentz and Euclidean distances almost
  coincide" y (e) el puntero a la Tabla 9 (citada en a, f y g). Dos párrafos quedan en dos frases por el presupuesto ("What a genuine
  excess means.", "Three readouts are compared…"; exentos en el sweep, anotado). Aun así el texto principal termina seis líneas
  dentro de la página 10 (las limitaciones (vi)–(ix)); lo que falta es lo que añade el resumen de 361 palabras sobre el tope de 310:
  recortándolo a tu tope, la página 9 vuelve a cerrar. No he cortado contenido tuyo para forzarla (TODO).
- **Cierre (2026-09-22, 17:20)**: §7 en la línea 463 de la página 9, limitaciones (vi)–(ix) en la página 10, statements en la 493
  (página 10), referencias en la 512; 37 páginas; 0 `??`; Figura 5 en lo alto de la página 8, donde empieza §5.4, y las matrices
  en el apéndice (página 28). Sweep 220/221: sólo falla el tope del resumen (361 palabras). Versión paralela reconstruida; resumen
  exportado a OpenReview (361 palabras).
## 53. Resumen nuevo del autor (2026-09-22, 17:25).

- Resumen sustituido verbatim (rellenos {{N_GEN}} y {{N_IN}} como siempre; "model–dataset" como `model--dataset`). Misma tesis como
  última frase. 337 palabras sin fórmulas: 24 menos que el anterior, todavía 27 sobre el tope de 310, así que el check del tope sigue
  fallando (único fallo, sweep 220/221 tras reconstruir la versión paralela). Literales del sweep actualizados (frase de los nueve y
  los tres, "whereas a planted hierarchy, synthetic or trained in, survives", frase del compartir; la frase espejo de §1 y la
  contribución 3 conservan "but not on the distances between them", el resumen dice "but not on their distances").
- Página 9: sin cambio, el texto principal sigue terminando seis líneas dentro de la página 10 (limitaciones vi–ix); el recorte del
  resumen no mueve el salto de la página 1 a la 2 porque la Figura 1 lo absorbe. Sigue en el TODO. Resumen exportado a OpenReview.
## 54. Figura 5, sólo disposición (2026-09-22, 17:30, brief del autor).

- `make_figs_final.py`: altura 2.1 in (la que ya tenía, así que la figura no cuesta líneas nuevas); (a) y (b) comparten el eje y, con
  los nombres sólo en (a) a 7 pt y un hueco entre familias en lugar de líneas separadoras; `wspace` 0.45, de modo que ninguna etiqueta
  de (c) toca el eje de (b); en (c) los nombres van rotados a 7 pt con el color de su familia. Comprobado a 100 dpi en
  `qa_pages_final/p-08.png`: sin solapes. Sin cambios de datos.
- Presupuesto, en el orden que marcas (§5.5 y §6, nunca figuras): §5.5 en dos frases (las dos primeras fundidas); §6: fuera la
  primera frase de "Both readings predict…" (la entradilla ya lo dice), "tracking the width of the backbone", el puntero "Table 4
  gives the band." y la frase de consejo "Before imposing curvature, certify…" (la entradilla del párrafo la contiene); el puntero al
  corolario pasa al final del tercer párrafo. Además el pie de la Tabla 2 una línea más corto y dos punteros de §5.3 fundidos. Tres
  párrafos más quedan en dos frases (exentos, anotados: "The raw reading cannot select a curvature.", "Both readings predict…",
  "Models share neighborhoods, not metrics."). Con todo, la página 9 acaba en la limitación (vi) con una línea libre y el párrafo de
  alcance (vii)–(ix), cuatro líneas, pasa entero a la 10 (la clase pone `\widowpenalty` y `\clubpenalty` a 10000, así que no se
  parte con una sola línea libre). Faltan tres líneas, y ya no quedan en §5.5 ni en §6: sólo en contenido tuyo (TODO).
- **Cierre (2026-09-22, 17:45)**: §7 en la línea 460 de la página 9; limitación (vi) cierra la página 9; alcance (vii)–(ix),
  statements y referencias en la 10; 37 páginas; 0 `??`. Sweep 220/221: sólo falla el tope del resumen (337 palabras). Versión paralela
  reconstruida.
## 55. Semilla 1 del control entrenado en el envío (2026-09-23, 09:00, brief del autor).

- **§5.3 (c)**, las frases del control entrenado con las dos semillas, todo como rellenos de `expR77_positive_control.csv`: "ViT-B
  fine-tuned with a hierarchical cross-entropy at three WordNet cuts keeps firing once decoupled in {{PC_HIER_DEC}} of 10 runs in
  both seeds, $z$ ${{PC_HIER_Z0}}$ and ${{PC_HIER_Z1}}$. The leaf-only fine-tune fires in {{PC_CE_DEC}} of 10 and {{PC_CE_DEC1}} of
  10, $z$ ${{PC_CE_Z0}}$ and ${{PC_CE_Z1}}$, and the frozen checkpoint in {{PC_FROZEN_DEC}} of 10, on the frame of record.
  Fine-tuning itself adds some decoupled signal in one seed and the hierarchical objective adds more, so the trained control
  separates the objectives by degree." (10 de 10; −2.38 y −2.63; 0 de 10 y 6 de 10; −0.72 y −2.02; 0 de 10). El builder exige 10 de 10
  en las dos semillas, 6 de 10 y 0 de 10, que el leaf CE de la semilla 1 dispare (z ≤ −2) y el de la 0 no, que el jerárquico de la
  semilla 1 lea más profundo que su leaf CE, y que el criterio previo no se cumpla en ninguna semilla (JSON del veredicto).
- **Limitación (iii)**: "Its trained control separates the objectives by degree, not absolutely: in one of two seeds the leaf-only
  fine-tune also fires once decoupled." en lugar de "Its trained positive control, one seed, discriminates the objectives only once
  decoupled."
- **Tabla 9(d)** (depth; "Table 8(d)" en la numeración del brief): filas de la semilla 1 (leaf CE y jerárquica), etiquetas con la
  semilla, pie "two seeds, identical batches within a seed" y nota con el 0 de 10 / 6 de 10 del leaf CE por semilla.
- Resumen sin cambios. `phaseE_rebuttal.py` deja de insertar el párrafo de la segunda semilla cuando el envío ya dice "in both seeds".
- **Sweep**: literales de (c) y (iii) rederivados del csv y del JSON; "0 of 10" puede aparecer dos veces en §5.3 (el brief lo escribe
  dos veces) y "6 of 10" una; exención (16, 3) para (c); recuentos y pares de z en HEADLINE desde el csv; "(seed 1)" no aparece en el
  cuerpo; la Tabla 9(d) dice "two seeds".
- Ajustes de forma para el sweep: "fires in 0 of 10 in one seed and 6 of 10 in the other" (el contador de cifras leía "0 of 10 and 6
  of 10" como un grupo raro) y la frase a 35 palabras.
- **Cierre (2026-09-23, 09:10)**: §7 en la línea 464 de la página 9; el texto principal termina nueve líneas dentro de la página 10
  (parte de la limitación iii y las de alcance), dos más que antes por las frases nuevas; statements en la 494, referencias en la
  513; 37 páginas; 0 `??`. Sweep 220/221: sólo falla el tope del resumen (337 palabras contra 310). Versión paralela reconstruida sin
  el párrafo de la segunda semilla (ya en el envío).
## 56. Presupuesto de página (2026-09-23, 09:15, brief del autor): la página 9 cierra con (2); semillas agrupadas.

- Orden del brief: (1) resumen, (2) limitación (iii) a la mitad, (3) frase de ejemplos de §1, (4) neural collapse en una frase, (5)
  fundir las frases de las semillas; parar cuando quepa. **(1)**: el brief dice "the version above" pero no trae texto nuevo; el
  resumen vigente es el tuyo de las 17:25 del 22 (337 palabras), así que no cambia nada y el tope sigue en 310 (el check sigue
  fallando por 27 palabras). **(2)**: la limitación (iii) pasa de 256 a 142 palabras conservando la potencia por backbone ("Under the
  decoupled control it is {{P81_COV_LO}} or more in nine backbones and {{P81_UNC_HI}} or less in the other three, at {{FA_DEC}} of 50
  false alarms on flat clouds. The verdict of no hub hierarchy therefore holds in the nine only."), la cláusula radial ("The alignment
  is not the radial spread of feature norms, which a control removes without changing the verdicts; full L2 normalization keeps it
  in two of the four backbones, and its exact form is not resolved here.") y "by degree". Salen de (iii): la frase del 60 % del
  implante de alineación (el porcentaje sigue en la Tabla 10(c) y el sweep lo exige allí), la explicación larga de "alignment above
  its frame only" (queda la cláusula; la explicación sigue en §5.3 (d)), "which makes the real verdicts under decoupling
  conservative" y "The alignment is thus relational, not radial…". Con (2) sola el texto principal termina en la página 9, así que
  (3), (4) y (5) no se aplican.
- **Semillas agrupadas** (mensaje del autor de las 09:16): §5.3 (c) "ViT-B fine-tuned with a hierarchical cross-entropy at three
  WordNet cuts keeps firing once decoupled in {{PC_HIER_POOL}} of 20 runs over two seeds, mean $z$ ${{PC_HIER_ZM}}$, the leaf-only
  fine-tune in {{PC_CE_POOL}} of 20, mean $z$ ${{PC_CE_ZM}}$, and the frozen checkpoint in none, on the frame of record. Fine-tuning
  itself adds some decoupled signal and the hierarchical objective adds more, so the trained control separates the objectives by
  degree." (20 de 20, −2.51; 6 de 20, −1.37: recuentos sumados sobre semillas y z medio de las dos medias por semilla, del csv de
  expR77); (iii): "…not absolutely: the leaf-only fine-tune also fires once decoupled, in {{PC_CE_POOL}} of 20 runs." Las filas por
  semilla siguen en la Tabla 9(d). Asserts del builder sobre los recuentos agrupados (20, 6, 0; z jerárquico < z leaf < 0); sweep con
  la frase reconstruida del csv, "20 of 20" y "6 of 20" una vez en §5.3, exención de 35 palabras para la frase del autor.
- Ajustes de forma: la frase agrupada del autor pasaba de 45 palabras (46) y pierde "at three WordNet cuts" (los cortes están en el pie
  de la Tabla 9(d)); en (iii) la frase de la potencia y la radial se parten en dos para el tope de 35 palabras ("The verdict of no hub
  hierarchy therefore holds in the nine only."; "Its exact form is not resolved here."). `phaseE_rebuttal.py`: el indicador de que
  el control de alineación implantada ya está en el envío se lee ahora de la Tabla 10(c) (la frase de (iii) desapareció), y el
  párrafo de la segunda semilla se omite también cuando el envío dice "over two seeds".
- **Cierre (2026-09-23, 09:25)**: FINAL_CHECK: "Main text (through the Conclusion and Limitations section) ends on page 9"; §7 en la
  línea 464 de la página 9, la limitación (ix) es la última línea de la página 9, statements en la 487 (página 10), referencias en la
  506; 37 páginas; 0 `??`. Sweep 220/221: sólo falla el tope del resumen (337 palabras contra 310, sin texto nuevo). Versión paralela
  reconstruida.
## 57. Duodécima revisión (2026-09-23, 09:40, brief del autor): matices en el resumen y §5.3, tesis, MERU, normalización, Tabla 1, Tabla 3.

- **(1)** "none is found in the real model" → "none as strong as the planted one is found" en el resumen (frase de los nueve y los
  tres) y en §5.3 (d): "it detects an implanted hierarchy in {{P81_COVERED}}, where none as strong as the planted one is found, and
  the test is blind in {{P81_UNCOVERED}}."
- **(2)** "almost as well as two readings of the same model do" → "well above chance, though short of what two readings of the same
  model reach" en el resumen, en el párrafo de §1 y en la contribución 3 (ediciones registradas) y en §5.4 ("The tree topology is
  therefore shared well above chance, though short of what two readings of the same model reach, and the metric agrees at about half
  of the ceiling."). Tesis, resumen y §7: "agree on how those clusters nest well above chance, though not as much as two readings of
  one model, and not on the distances between them". El pie de la Figura 5 conserva tu "tree topology is shared almost up to noise"
  (no lo pedías; dímelo si también cambia). El resumen sube a 349 palabras sin fórmulas (tope 310, único fallo del sweep).
- **(3)** Entradilla de MERU en §5.3: "MERU's objective does not create hub structure."
- **(4)** Limitación (vii) nueva, la última del instrumento: "The reading divides by the diameter, so heavier tails in real clouds
  would lower $\delta_{\text{norm}}$ without any clustering; the cosine census mitigates this, and a percentile normalization is
  future work."; las de alcance pasan a (viii)–(x).
- **(5) Tabla 1**: los defectos de expR78 no están guardados (sólo los resúmenes por tanda), así que la columna se produce con un
  cálculo nuevo, **expR85** (`expR85_khrulkov_sup.py`, lanzado a las 09:41 en ocho shards, CPU): las mismas tandas de 1500 puntos
  (RandomState(t), como expR78) leídas con su propio estadístico, el supremo exacto por producto min-max (delta_rel = 2δ/diam), sobre
  el cloud real y sobre las mismas 200 réplicas Haar centradas del registro (semillas 0–199): exceso y rango del supremo. Un supremo
  exacto cuesta 10–17 s en 1500 puntos (unos 40 min por tanda, 40 tandas), fin previsto hacia las 13:00; la columna entra entonces
  (hasta ese momento la Tabla 1 no cambia y no se añade la cláusula alternativa, que sólo procede si no puede producirse hoy).
- **(6) Tabla 3** (robustez): sus cinco partes llevan ahora cada una el texto de su panel: (a) en el pie principal, (b) a (e) en los
  pies "(continued)" (antes tres de ellos estaban vacíos y el de (c) arrastraba (d) y (e)); las fuentes van al pie de su panel. El
  sweep exige cuatro pies continuados que empiecen por (b), (c), (d) y (e).
- **Sweep**: check nuevo de la duodécima revisión (frases nuevas, tesis dos veces, entradilla de MERU, limitación (vii) y renumeración,
  pies de la Tabla 3); literales anteriores actualizados.
- **Presupuesto**: (2) y (4) añadían tres líneas; recuperadas sin tocar cifras ni frases del autor: §5.3 (b) "reproduces it in … and its
  geometric form is not resolved here" (la última línea del párrafo era una palabra), §4 fuera "Two self-supervised ResNets read the
  vision census." (están en la Tabla 5), §5.3 (f) fuera el puntero "Table 9 gives the two models." y su primera frase partida en dos
  para mantener tres frases; la frase de (d) "it detects an implanted hierarchy in …, and the test is blind in …" partida: "Where
  the implant is detected, none as strong as the planted one is found." (tope de 35 palabras).
- **Cierre (2026-09-23, 10:00)**: FINAL_CHECK: "Main text (through the Conclusion and Limitations section) ends on page 9"; §7 en la
  línea 460 de la página 9, la limitación (x) es la última línea de la página 9; statements en la 487, referencias en la 506 (página
  10); 37 páginas; 0 `??`. Sweep 221/222: sólo falla el tope del resumen (349 palabras contra 310). Versión paralela reconstruida;
  resumen exportado a OpenReview. Pendiente de hoy: la columna del supremo en la Tabla 1 (expR85 en marcha).

## 58. Pasada de lenguaje llano (2026-09-23, 10:40, brief del autor): 21 párrafos reescritos, antes/después.

- Regla aplicada a §3.3 (excess, rank, genuine), §3.4 (test de profundidad), §4 (todo), §5.1, §5.3 (los siete), §5.4 (los tres primeros) y §6 (los tres): cada frase dice primero qué se hizo y luego qué salió; cada término acuñado se parafrasea en su primer uso por sección ("once the cluster orientations were randomized" para decoupled, "the superclass centers" para hubs, "the grouping into superclasses" para frame, "more than chance would give" para genuine); ninguna frase de más de 30 palabras; no más de dos números por frase; las entradillas se conservan. Sin cambios de contenido ni de cifras (los rellenos son los mismos). El brief citaba un párrafo de muestra que no llegó con el mensaje; se aplicó la regla tal cual está enunciada. Sin cambios en la lógica ni en los números del sweep: sólo los literales fijados siguen la redacción nueva; dos añadidos mecánicos (los pares "0.48 and 0.78" aparecen ahora como valores sueltos en HEADLINE; la frase de §5.4 que cita la Figura 5 en primer lugar ya estaba exenta). `SUMMARY_plain.md` regenerado a mano desde el texto final (no existe generador).
- Frases del autor partidas por el tope de 30 palabras, sin cambiar contenido: la del implante de dos y tres niveles (§5.3 c), la relacional (b), la radial (b), la de las dos semillas agrupadas (c), la del sesgo (c), la de "A hierarchy below the frame …" (d), la lectura de desajuste de frames (e, en el builder), la de la Figura 5 (§5.4), la de la potencia/falsas alarmas (§3.4).
- Presupuesto: la reescritura añadía unas seis líneas; recuperadas con la propia redacción (frases de apoyo mías acortadas), los dos punteros permitidos (§5.5 y §6.3; el puntero al corolario vuelve entre paréntesis porque el sweep exige que toda tabla del apéndice se cite) y la regla de numerales de §59 (numerales más cortos que palabras).

### Antes / después por párrafo (rellenos sustituidos por sus valores)

**The null is a cloud with the real shape and no structure.**

- antes: Gaussian coefficients reproduce the spectrum only in expectation and inflate the excess of low-rank clouds. The centered Haar construction reproduces the centered spectrum exactly, so it is the record. The uncentered and the Gaussian readings are reported beside it (Table~\ref{tab:q1-census}).
- después: Of the two constructions compared, Gaussian coefficients reproduce the spectrum only on average and inflate the excess of low-rank clouds. The centered Haar construction reproduces the centered spectrum exactly, so it is the record. The uncentered and the Gaussian readings are reported beside it (Table~\ref{tab:q1-census}).

**What a genuine excess means.**

- antes: A negative genuine excess certifies structure beyond the second moments; clustering is its most plausible reading, supported by the superclass recovery of Section~\ref{sec:content}. Two cells on the same dataset with the same reading can receive opposite verdicts (Figure~\ref{fig:overview}).
- después: A negative genuine excess, more than chance would give, certifies structure beyond the second moments; clustering is its most plausible reading, supported by the superclass recovery of Section~\ref{sec:content}. The same reading can therefore receive opposite verdicts in two cells of one dataset (Figure~\ref{fig:overview}).

**The depth test compares the real cloud with its matched star.**

- antes: With $e_B$ read on the real cloud and on its matched star, the depth statistic and its standardized form are \begin{equation} \mathrm{depth}=e_B(X)-e_B(X_{\mathrm{star}}),\qquad z=\mathrm{depth}\,/\,\mathrm{s.d.}, \label{eq:depth} \end{equation} The spread is taken over star seeds and null replicates, and a cloud is certified against its matched star when $z\le-2$. The power of the test is the fraction of implanted hierarchies it detects and its false-alarm rate the fraction of flat controls that fire; both are measured on real clouds in Section~\ref{sec:f-depth}.
- después: With $e_B$ read on the real cloud and on its matched star, the depth statistic and its standardized form are \begin{equation} \mathrm{depth}=e_B(X)-e_B(X_{\mathrm{star}}),\qquad z=\mathrm{depth}\,/\,\mathrm{s.d.}, \label{eq:depth} \end{equation} The spread is taken over star seeds and null replicates, and a cloud is certified against its matched star when $z\le-2$. The power of the test is the fraction of implanted hierarchies it detects, and its false-alarm rate is the fraction of flat controls that fire. Both are measured on real clouds in Section~\ref{sec:f-depth}.

**4 controls test the alternatives.**

- antes: DeiT-B and the augreg ViT-B, trained on ImageNet-1k leaf labels, test whether a hierarchical label set explains the depth. MERU and its Euclidean CLIP twin test whether training in hyperbolic space creates depth. The census ViT-B fine-tuned with and without a hierarchical objective is a trained control.
- después: 2 self-supervised ResNets read the vision census. DeiT-B and the augreg ViT-B, trained on ImageNet-1k leaf labels, test whether a hierarchical label set explains the depth. MERU and its Euclidean CLIP twin test whether training in hyperbolic space creates depth. The census ViT-B, fine-tuned with and without a hierarchical objective, is the trained control.

**The premise is read on images and hierarchy on centroids.**

- antes: The sample-level reading uses about a thousand stratified training images per cell on CIFAR-100 and DTD. Class centroids average a hundred cached training images per class on ImageNet and the full cached split elsewhere. Causal language models give the last-token state of ``a photo of a \{class\}'', extracted one prompt at a time because batching alters GPT-2's hidden states, and embedders use their recommended pooling (Table~\ref{tab:q2-text}).
- después: The sample-level reading uses about a thousand stratified training images per cell on CIFAR-100 and DTD. Class centroids average a hundred cached training images per class on ImageNet and the full cached split elsewhere. Causal language models give the last-token state of ``a photo of a \{class\}'', extracted one prompt at a time because batching alters GPT-2's hidden states. Embedders use their recommended pooling (Table~\ref{tab:q2-text}).

**Calibration budget and frames.**

- antes: Each cell is read against two hundred Haar replicates, with half a million sampled quadruples per seed and ten seeds. With ten classes there are only 210 quadruples, so $\hat\delta_{99.9}$ coincides with the supremum there. Benjamini--Hochberg runs over the vision census and, separately, over the text census. The depth test uses the WordNet cut into thirty superclasses on ImageNet, the twenty coarse labels on CIFAR-100, a balanced frame as a control, and ten star seeds.
- después: Each cell is read against 200 Haar replicates, with half a million sampled quadruples per seed and 10 seeds. With 10 classes there are only 210 quadruples, so $\hat\delta_{99.9}$ coincides with the supremum there. Benjamini--Hochberg runs over the vision census and, separately, over the text census. The frame, the grouping into superclasses, is the WordNet cut into 30 on ImageNet and the 20 coarse labels on CIFAR-100; a balanced grouping is the control, with 10 star seeds.

**Raw readings are not evidence, ours or published, and calibrated ones are weak and model-dependent.**

- antes: On per-image features, where the premise is read, the calibrated reading is not genuine in most cells. Where it is genuine, the excess removes at most two fifths of the null reading. Table~\ref{tab:q3-sample} gives the reading per cell.
- después: On per-image features, where the premise is read, the calibrated reading is not genuine in most cells, that is, no more than chance would give. Where it is genuine, the excess removes at most two fifths of the null reading. Table~\ref{tab:q3-sample} gives the reading per cell.

**A published reading is reproduced and calibrated.**

- antes: Their estimator on our extraction reproduces the raw $\delta_{\text{rel}}$ that \citet{Khrulkov_2020_CVPR} report for ResNet-34 on four datasets within 0.03. Calibrated on the same clouds, the excess is negative on every dataset, and only CIFAR-100 and MiniImageNet fall below the null in every batch at the 0.05 level. Table~\ref{tab:khrulkov} gives the four rows.
- después: Their estimator was run on our extraction; it reproduces the raw $\delta_{\text{rel}}$ that \citet{Khrulkov_2020_CVPR} report for ResNet-34 on 4 datasets within 0.03. The same clouds were then calibrated: the excess is negative on every dataset, and only CIFAR-100 and MiniImageNet fall below the null in every batch at the 0.05 level. Table~\ref{tab:khrulkov} gives the 4 rows.

**The depth test certifies that clusters are oriented toward their hubs in 4 of 12 backbones.**

- antes: It certifies {{N_IN}} of 12 backbones, ViT-S, ViT-B, ViT-L and DINOv2-L, under both matched stars. The noise level of a cloud is its within-cluster spread relative to the distance between its hubs. Figure~\ref{fig:depth}a shows the $z$ per backbone and Table~\ref{tab:q4-depth} gives both stars.
- después: The test was run on the 12 ImageNet centroid clouds under both matched stars, with the WordNet cut into superclasses as the frame and the superclass centers as hubs. It certifies {{N_IN}} of 12: ViT-S, ViT-B, ViT-L and DINOv2-L. The noise level of a cloud is its within-cluster spread relative to the distance between its hubs. Figure~\ref{fig:depth}a shows the $z$ per backbone and Table~\ref{tab:q4-depth} gives both stars.

**What it certifies is alignment.**

- antes: Under the decoupling control none of the four fires, and ViT-L reads $-1.61$, within the $-1.61$ to $-1.76$ of the flat control: without its orientations the real cloud reads like a flat one. The alignment is relational, needing both the real hubs and the orientation of each cluster relative to them: randomizing either removes it, 0 of 60 with random hubs and 0 of 4 with random orientations. The alignment survives removing the radial component of every offset in all four certified backbones, $z$ from $-2.26$ to $-3.99$ under both stars, which excludes the spread of feature norms as its source. Under full L2 normalization, which also moves the hubs, it survives in ViT-B and ViT-L only. The implanted principal-axis alignment reproduces it in the supervised ViTs and CLIP-B but not in DINOv2-L, and its geometric form is not resolved here.
- después: Once the cluster orientations were randomized, the decoupling control of Definition~\ref{def:decoupling}, none of the 4 backbones fires. ViT-L then reads $-1.61$, within the $-1.61$ to $-1.76$ of the flat control: without its orientations the real cloud reads like a flat one. The alignment is relational: 0 of 60 runs fire with random hubs and 0 of 4 with random orientations, so it needs both the real hubs and the real orientations. Removing the radial component of every offset leaves it in all 4 certified backbones, $z$ from $-2.26$ to $-3.99$ under both stars; feature norms are not its source. Under full L2 normalization, which also moves the hubs, it survives in ViT-B and ViT-L only. The implanted principal-axis alignment reproduces it in the supervised ViTs and CLIP-B but not in DINOv2-L, and its geometric form is not resolved here.

**What the test can see.**

- antes: The test misses an implanted two-level tree at this noise level but detects a three-level hierarchy with ViT-L's spectrum and noise in 4 of 5 seeds, and that detection survives orientation randomization, whereas none of the four real verdicts does. The decoupling control fires on every deep seed and reads deeper than the intact cloud. The deeper reading comes from both sides: decoupling shrinks the star spread and deepens the excess below the star. The decoupled control detects the same hierarchy, built at each backbone's own spectrum and ratio, in nine backbones (decoupled power 0.83--1.00) and not in the other three (power 0.01--0.60). The three blind backbones sit at ratios 1.5 to 3.0, inside the 1.3 to 3.9 of the nine covered, so the power follows the backbone, not its ratio or family. ViT-B fine-tuned with a hierarchical cross-entropy keeps firing once decoupled in 20 of 20 runs over two seeds, mean $z$ $-2.51$, the leaf-only fine-tune in 6 of 20, mean $z$ $-1.37$, and the frozen checkpoint in none, on the frame of record. Fine-tuning itself adds some decoupled signal and the hierarchical objective adds more, so the trained control separates the objectives by degree. The pre-set criterion asked the leaf-CE and frozen models not to be certified intact, which they are, by alignment, so the discriminating comparison is the decoupled one. A flat control with the same spectrum and ratio as the deep one fires in 0 of 5 intact and in 8 of 50 decoupled runs. The decoupled test is therefore biased toward firing, so the four real backbones not firing once decoupled is conservative evidence, and the synthetic hierarchy at $z$ $-4.19$ to $-4.82$ stands well clear of that bias. On the balanced frame the matched flat control fires in 0 of 50 decoupled runs. The WordNet Poincar\'{e} embeddings \citep{nickel2017poincare}, read with Euclidean distances, do not fire (Tables~\ref{tab:q5-power}, \ref{tab:q4-depth}).
- después: Given an implanted two-level tree at this noise level, the test misses it; given a three-level hierarchy with ViT-L's spectrum and noise, it detects it in 4 of 5 seeds. That detection survives orientation randomization, whereas none of the 4 real verdicts does. The decoupling control fires on every deep seed and reads deeper than the intact cloud. The deeper reading comes from both sides: decoupling shrinks the star spread and deepens the excess below the star. The decoupled control detects the same hierarchy, built at each backbone's own spectrum and ratio, in 9 of 12 backbones (decoupled power 0.83--1.00) and not in the other 3 (power 0.01--0.60). The 3 blind backbones sit at ratios 1.5 to 3.0, inside the 1.3 to 3.9 of the 9 covered backbones, so the power follows the backbone, not its ratio or family. ViT-B fine-tuned with a hierarchical cross-entropy keeps firing once decoupled in 20 of 20 runs over 2 seeds, mean $z$ $-2.51$, on the frame of record. The leaf-only fine-tune fires in 6 of 20, mean $z$ $-1.37$, and the frozen checkpoint in none. Fine-tuning itself adds some decoupled signal and the hierarchical objective adds more, so the trained control separates the objectives by degree. The pre-set criterion asked the leaf-CE and frozen models not to be certified intact, which they are, by alignment, so the discriminating comparison is the decoupled one. A flat control with the same spectrum and ratio as the deep one fires in 0 of 5 intact and in 8 of 50 decoupled runs. The decoupled test is therefore biased toward firing, so the 4 real backbones not firing once decoupled is conservative evidence. The synthetic hierarchy at $z$ $-4.19$ to $-4.82$ stands well clear of that bias. On the balanced frame the matched flat control fires in 0 of 50 decoupled runs. The WordNet Poincar\'{e} embeddings \citep{nickel2017poincare}, read with Euclidean distances, do not fire (Tables~\ref{tab:q5-power}, \ref{tab:q4-depth}).

**No hub hierarchy is found where the test has power.**

- antes: The power is the decoupled control's: it detects an implanted hierarchy in ViT-S/B/L, DINOv2-B/L/G, CLIP-B/L and SigLIP-B, and the test is blind in ViT-T, DINO-B and DINOv2-S. Where the implant is detected, none as strong as the planted one is found. The decoupling separates structure among the hubs of the frame from structure carried by cluster orientations. A hierarchy below the frame expressed in how clusters open would be removed with the orientations and counted as alignment, so no hub hierarchy means no hierarchy among the frame's hubs. Figure~\ref{fig:implant} and Table~\ref{tab:q5-power} give the detection rates and the power.
- después: The power is the decoupled control's: it detects an implanted hierarchy in ViT-S/B/L, DINOv2-B/L/G, CLIP-B/L and SigLIP-B, and the test is blind in ViT-T, DINO-B and DINOv2-S. Where the implant is detected, none as strong as the planted one is found. The decoupling separates structure among the hubs of the frame from structure carried by cluster orientations. A hierarchy below the frame expressed in how clusters open would be removed with the orientations and counted as alignment. No hub hierarchy therefore means no hierarchy among the frame's hubs. Figure~\ref{fig:implant} and Table~\ref{tab:q5-power} give the detection rates and the power.

**The certified set depends on the frame.**

- antes: Under a balanced frame ViT-T joins the certified set and ViT-S and DINOv2-L leave it, and other WordNet cuts change the set again. Only ViT-B and ViT-L are certified under every frame. A flat cloud clustered under the frame of record fires under the balanced frame in 38 of 50 decoupled runs. A real cloud read under a frame that is not its own is therefore expected to fire. The frozen ViT-B firing in 7 of 10 there is consistent with that mismatch and is not evidence of hub structure.
- después: Rerun under a balanced grouping into superclasses, the balanced frame, ViT-T joins the certified set and ViT-S and DINOv2-L leave it; other WordNet cuts change the set again. Only ViT-B and ViT-L are certified under every frame. A flat cloud clustered under the frame of record fires under the balanced frame in 38 of 50 decoupled runs. A real cloud read under a frame that is not its own is therefore expected to fire. The frozen ViT-B firing in 7 of 10 there is consistent with that mismatch and is not evidence of hub structure.

**Leaf labels can produce the alignment but do not guarantee it.**

- antes: The augreg ViT-B trained on ImageNet-1k leaf labels is certified and aligned with WordNet like the ImageNet-21k ViTs. DeiT-B, trained on the same labels with another recipe, is neither. A hierarchical label set therefore does not explain the certified ViTs' depth.
- después: Read like the census models, the augreg ViT-B trained on ImageNet-1k leaf labels is certified and aligned with WordNet like the ImageNet-21k ViTs. DeiT-B, trained on the same labels with another recipe, is neither. A hierarchical label set therefore does not explain the certified ViTs' depth.

**MERU's objective does not create hub structure.**

- antes: MERU embeds images on the Lorentz hyperboloid with an entailment objective, and its Euclidean twin shares backbone, data and recipe \citep{desai2023meru}. MERU shows the same clustering, no such structure either, and embeddings that never leave the near-flat regime. Table~\ref{tab:q4-depth} gives the readings and the radii.
- después: MERU embeds images on the Lorentz hyperboloid with an entailment objective, and its Euclidean twin shares backbone, data and recipe \citep{desai2023meru}. Read with the census and the depth test, MERU shows the same clustering, no such structure either, and embeddings that never leave the near-flat regime. Table~\ref{tab:q4-depth} gives the readings and the radii.

**The naive map manufactures an island.**

- antes: Figure~\ref{fig:treemap} compares the trees of the twelve models: (a) as usually compared, (b) under the criterion of Section~\ref{sec:f-treemap} against the within-model ceiling, and (c) by whether classes are grouped by direction or by distance. Under Euclidean distances with average linkage the DINOv2 family agrees with the supervised and contrastive block far less than the block agrees with itself. The cause is chaining: DINOv2's Euclidean dendrograms put almost every class in one cluster, forcing the index to zero.
- después: Figure~\ref{fig:treemap} compares the trees of the 12 models in three ways. Panel (a) compares them as usually done, (b) under the criterion of Section~\ref{sec:f-treemap} against the within-model ceiling, and (c) by whether classes are grouped by direction or by distance. Compared with Euclidean distances and average linkage, the DINOv2 family agrees with the supervised and contrastive block far less than the block agrees with itself. The cause is chaining: DINOv2's Euclidean dendrograms put almost every class in one cluster, forcing the index to zero.

**Controlling the cut leaves a moderate gap.**

- antes: Under the selected configuration, cosine distances with average linkage on ImageNet, the island shrinks to a gap beyond resampling noise. Two resamples of the same model agree on 0.85 to 0.96 of the triplets across backbones, mean 0.90, against 0.77 for DINOv2 against the block and 0.76 within it. The cophenetic correlation reaches 0.91 to 0.97 within a model, mean 0.94, against 0.48 and 0.78 across models. The ARI at the cut reaches 0.64 to 0.94, mean 0.79, against 0.39 and 0.48. The tree topology is therefore shared well above chance, though short of what two readings of the same model reach, and the metric agrees at about half of the ceiling. About a third of the within-block agreement is missing under the admissible configurations as a whole. Figure~\ref{fig:treemap}b and Table~\ref{tab:q6-treemap} give the map, the configurations and the ceiling.
- después: Under the selected configuration, cosine distances with average linkage on ImageNet, the island shrinks to a gap beyond resampling noise. Two resamples of the same model agree on 0.85 to 0.96 of the triplets across backbones, mean 0.90. Across models: 0.77 for DINOv2 against the block, 0.76 within it. The cophenetic correlation reaches 0.91 to 0.97 within a model, mean 0.94. Across models: 0.48 against the block, 0.78 within it. The ARI at the cut reaches 0.64 to 0.94 within a model, mean 0.79. Across models: 0.39 against the block, 0.48 within it. The tree topology is therefore shared well above chance, though short of what two readings of the same model reach, and the metric agrees at about half of the ceiling. About a third of the within-block agreement is missing under the admissible configurations as a whole. Figure~\ref{fig:treemap}b and Table~\ref{tab:q6-treemap} give the map, the configurations and the ceiling.

**The self-supervised tree is angular.**

- antes: DINOv2's semantics concentrate in the angular component: sibling triplets are resolved far better under cosine than under Euclidean distance, so the Euclidean dendrograms chain. The cosine census agrees with the Euclidean one in most cells, and DINOv2 on ImageNet is genuine under both. Figure~\ref{fig:treemap}c shows the triplets and Table~\ref{tab:q1-census} gives the cosine census.
- después: Sibling triplets were resolved under cosine and under Euclidean distance: DINOv2 resolves them far better under cosine, so its semantics concentrate in the angular component and its Euclidean dendrograms chain. Repeated on cosine geometry, the census agrees with the Euclidean one in most cells, and DINOv2 on ImageNet is genuine, more than chance would give, under both. Figure~\ref{fig:treemap}c shows the triplets and Table~\ref{tab:q1-census} gives the cosine census.

**The raw reading cannot select a curvature.**

- antes: \citet{Khrulkov_2020_CVPR} set the curvature of the Poincar\'{e} ball from the relative hyperbolicity, $c=(0.144/\delta_{\text{rel}})^{2}$ with $\delta_{\text{rel}}=2\delta/\mathrm{diam}$. The rule, calibrated with the supremum, assigns structureless clouds on the Gaussian band, itself the sampled supremum rather than the census percentile, a curvature from 0.48 to 2.5.
- después: \citet{Khrulkov_2020_CVPR} set the curvature of the Poincar\'{e} ball from the relative hyperbolicity, $c=(0.144/\delta_{\text{rel}})^{2}$ with $\delta_{\text{rel}}=2\delta/\mathrm{diam}$. The rule was calibrated with the supremum and applied to the Gaussian band, the sampled supremum rather than the census percentile; it assigns structureless clouds a curvature from 0.48 to 2.5.

**The calibration certifies structure and does not choose the readout.**

- antes: Cosine collects the self-supervised structure, and the Poincar\'{e} readout adds ${{GAIN_LO}}$ to ${{GAIN_HI}}$ pp over cosine for the contrastive VLMs on prototype tasks and nothing consistent elsewhere. In few-shot learning a fixed-radius Euclidean encoder does at least as well as hyperbolic prototypes \citep{moreira2024hyperbolic}. Table~\ref{tab:q9-corollary} gives the gains.
- después: Cosine collects the self-supervised structure. The Poincar\'{e} readout adds ${{GAIN_LO}}$ to ${{GAIN_HI}}$ pp over cosine for the contrastive VLMs on prototype tasks and nothing consistent elsewhere. In few-shot learning a fixed-radius Euclidean encoder does at least as well as hyperbolic prototypes \citep{moreira2024hyperbolic} (Table~\ref{tab:q9-corollary}).

## 59. Regla de numerales (2026-09-23, 10:55, mensaje del autor).

- Los recuentos de modelos, celdas, semillas y runs van siempre en cifras y con su denominador cuando lo hay ("9 of 12 backbones", "the other 3"); las palabras quedan para cantidades descriptivas ("a three-level hierarchy", "two readings of the same model", "three ways", "two fifths", "half a million quadruples", "hundreds of quadruples", "a thousand centroids"). Aplicado en el resumen ("in 9 of 12 backbones … in the other 3"), §1 (párrafo y contribución 2; "the 4 values … 2 of them"), §3.2 ("10 seeds"), §4 (entradilla "The census has 12 backbones, 6 class sets and 15 text models.", "4 supervised ViTs", "4 DINOv2 sizes", "4 GPT-2 sizes, 3 Pythia sizes … 7 sentence embedders", entradilla "4 controls test the alternatives.", "2 self-supervised ResNets", "200 Haar replicates", "10 seeds", "10 classes", "30 superclasses", "20 coarse labels", "10 star seeds"), §5.1 ("4 datasets", "the 4 rows"), §5.2 ("the 3 datasets with 47 classes or more"), §5.3 (entradilla "… in 4 of 12 backbones.", "the 12 ImageNet centroid clouds", "none of the 4 backbones fires", "all 4 certified backbones", "none of the 4 real verdicts", "the 4 real backbones", "9 of 12 backbones … the other 3", "The 3 blind backbones … of the 9 covered backbones", "over 2 seeds"), §5.4 ("the 12 models"), §5.5 ("66 model pairs"), §7 ("9 of 12 backbones … the other 3", "in those 9 only", "2 of the 4 backbones", "6 datasets", "2 datasets", "2 leaf-label ViTs"), pies de la Figura 4 y de la Figura 5 ("the other 11") y de las Tablas 3 ("9 cells") y 9 ("4 of 12 ImageNet backbones", "9 of 12 backbones … the other 3").
- Sweep: check nuevo de la regla (ninguna palabra-número delante de un sustantivo de recuento en resumen, texto principal y pies; "in 9 of 12 backbones" y "in the other 3" en el resumen; las cantidades descriptivas siguen en palabras); en la regla de "sólo números de titular" los recuentos con su sustantivo ("12 backbones", "200 Haar replicates", "the other 3", "those 9 only") no cuentan como números de resultado; literales fijados actualizados.
- **Cierre de §58–§59 (2026-09-23, 11:20)**: FINAL_CHECK: "Main text (through the Conclusion and Limitations section) ends on page 9";
  §7 en la línea 463 de la página 9, la limitación (x) es la última línea de la página 9; statements en la 487, referencias en la
  506 (página 10); 37 páginas; 0 `??`. Sweep 223/224: sólo falla el tope del resumen (351 palabras con los numerales, contra 310;
  ningún texto nuevo tuyo). `phaseE_rebuttal.py` reconoce la redacción nueva de §5.3 (c) para no reinsertar el párrafo de expR79;
  versión paralela reconstruida; resumen exportado a OpenReview. Pendiente de hoy: la columna del supremo de la Tabla 1 (expR85 en
  marcha, fin previsto ~16:00).
## 60. Dos ajustes (2026-09-23, 11:20, brief del autor): recuentos al inicio de frase en palabras; tope del resumen a 360.

- **(1)** Ninguna frase empieza por cifra: "Four controls test the alternatives." y "Two self-supervised ResNets read the vision
  census." (las dos únicas frases que la regla de numerales había dejado empezando por cifra). Excepción en el check de numerales:
  un recuento que abre frase va en palabras y con mayúscula (el check es sensible a mayúsculas, así que "Four controls" no cuenta como
  infracción) y ninguna frase de §3–§7 puede empezar por un dígito.
- **(2)** Tope del resumen en el sweep: 360 palabras (la regla de numerales añade tokens, no contenido; el resumen queda en 351).
- **Item 5 de la duodécima revisión, preparado a la espera de expR85**: `main_khrulkov()` añade dos columnas a la Tabla 1 (exceso y
  rango/p del supremo, contra las mismas 200 réplicas) cuando existe `expR85_khrulkov_sup_summary.csv` con 10 tandas por dataset
  (probado con un resumen simulado y restaurado); §5.1 lleva un relleno {{SUP_CLAUSE}}: mientras el run no está fusionado, la
  cláusula alternativa del brief ("; the calibration reads the 99.9th percentile, whereas their statistic is the supremum"), y después
  ", with their own statistic, the supremum, calibrated on the same replicates beside the record statistic"; el builder comprueba 10
  tandas y 200 réplicas; el sweep exige las celdas de la tabla desde el csv cuando existe y la cláusula alternativa mientras no. El
  comentario de procedencia de §5.1 nombra expR85 sólo cuando el fichero existe (relleno {{PROV85}}). Commit separado cuando expR85
  termine (fin previsto ~16:00).
- La cláusula alternativa de §5.1 se escribe "the calibration reads the percentile statistic $\\hat\\delta_{99.9}$, whereas their
  statistic is the supremum": con "99.9th" en texto llano el contador de cifras del sweep leía un "99" suelto.
- **Cierre (2026-09-23, 11:35)**: FINAL_CHECK: "Main text (through the Conclusion and Limitations section) ends on page 9"; §7 en la
  línea 463 de la página 9; statements en la 487, referencias en la 506 (página 10); 37 páginas; 0 `??`. **Sweep 224/224, 0 fallos**
  (resumen 351 palabras bajo el tope de 360). Versión paralela reconstruida; resumen exportado a OpenReview.
## 61. Legibilidad, segunda pasada (2026-09-23, 11:45, brief del autor): tope de 30 palabras en las páginas 3–9 y cuatro arreglos.

- **Tope de 30 palabras**: aplicado a toda frase de las páginas 3–9 (§2 en su parte de la página 3 y §3–§7), salvo dentro de las
  definiciones y las listas de citas (frases con dos o más citas, o una cita con varias claves). Mi recuento sobre el tex, con las
  entradillas separadas, daba 15 frases por encima (el brief cuenta 41, seguramente sobre el PDF y con otra tokenización); todas
  partidas en su giro natural sin cambiar contenido: §2 "…of comparable size \citep{…}. We bring the idea…" (edición registrada modificada; la frase "We are the geometric
  complement…" de `main_local.tex` no está en el envío, que conserva las seis primeras frases de §2); §3.1 "…is zero. The further a metric is from a tree…" (edición
  nueva); §3.2 "…fall below. It keeps the tail of the defects, as the supremum does…" (edición 4 modificada y una nueva); §3.5 las dos
  frases de la comparación de árboles; §4 la del frame; §5.3 (c) "The same hierarchy was then built at each backbone's own spectrum
  and ratio. The decoupled control detects it in 9 of 12 backbones…" y "…spanned by the 9 covered backbones. The power therefore
  follows the backbone…"; §5.5 "…improves neither. Models share who is near whom and differ in the metric \citep{…} (Table 13)."; §6
  "…census percentile. It assigns structureless clouds a curvature from…"; (iii) "…in the other 3. Its false alarms on flat clouds are
  8 of 50, and the verdict…"; (vii) "…without any clustering. The cosine census mitigates this…". Una excepción conservada y declarada: la tesis de §7 (56 palabras), que por tu regla es la última frase
  del resumen verbatim; la lista de citas de §2 ("Closest to us…", 60 palabras, cuatro citas) está exenta por el brief. Ediciones
  registradas: 16.
- **(1)** Definición 3 abre con "A null replicate is a cloud with the real shape and no structure." y sigue la construcción ("Let $X=U\Sigma
  V^{\top}$ …; the replicate is $X'=Q\Sigma V^{\top}$ …"). **(2)** §3.2, el párrafo de los confounds tras la Figura 2 queda en tu
  frase: "Proposition 1 makes the dimension confound precise; the calibration on reference geometries shows the spectrum and statistic
  confounds (Table 4)." (entradilla conservada). **(3)** §3.3: "Two cells of the same dataset can have the same reading and opposite
  verdicts (Figure 2)."; el check de numerales exime explícitamente las cantidades descriptivas (lista `DESCRIPTIVE`: "Two cells of
  the same dataset", "two readings of the same model", "three constructions", "two fifths", …). **(4)** §3.4 abre con tu párrafo "A
  genuine excess can come from clusters alone, so a second test asks whether the clusters are themselves arranged hierarchically. It
  needs three constructions." antes de la Definición 6, con la entradilla "Why a second test." (mía, por la regla de entradillas).
- **Sweep**: el tope pasa de 35 a 30 palabras y se mide con las entradillas separadas (antes la primera frase de cada párrafo se
  contaba con su entradilla); exentas las definiciones, las listas de citas y la tesis; los bloques verbatim de §2–§3 entran en la
  medida; check nuevo con los cuatro arreglos y las tres ediciones. **FINAL_CHECK**: columna "over 30" por sección en la tabla de
  longitudes y una línea con la regla, el recuento de frases por encima y la excepción de la tesis.
- La frase "We are the geometric complement…" sí está en el envío (es la sexta de §2 en la página 3) y se parte en dos con una edición
  registrada; para que la segunda mitad sobreviva al recorte de §2 del presupuesto de página (que conservaba seis frases), el
  builder conserva ahora siete. Ediciones registradas: 17. El check de las ediciones y el de §2 verbatim siguen pasando.
- **Cierre (2026-09-23, 12:08)**: FINAL_CHECK: "Main text (through the Conclusion and Limitations section) ends on page 9"; tabla de
  longitudes con la columna "over 30" y la línea de la regla ("sentences over the cap outside those: 0; the thesis sentence, verbatim
  from the abstract, is the one kept exception (56 words)"); §7 en la línea 463 de la página 9; 37 páginas; 0 `??`. **Sweep 225/225.**
  Versión paralela reconstruida.
## 62. Legibilidad, §4–§5.2 (2026-09-23, 12:10, brief del autor): seis arreglos de frase.

- **(1)** Entradilla de §5.1: "The premise does not survive calibration where it is read." (sustituye a "Raw readings are not evidence,
  ours or published, and calibrated ones are weak and model-dependent."). **(2)** Frase de Khrulkov en §5.1: "Calibrated, CIFAR-10
  and CUB-200 are indistinguishable from a random cloud, and CIFAR-100 and MiniImageNet fall below it." (el sweep comprueba desde
  `expR78_khrulkov_replication_summary.csv` que el p mayor supera 0.05 exactamente en CIFAR-10 y CUB-200 y queda por debajo en
  CIFAR-100 y MiniImageNet). **(3)** §5.2 "A star of clusters already passes the census.": fuera la primera frase, que repetía la
  entradilla; el párrafo queda en dos frases (exento del recuento). **(4)** §5.2 neural collapse: el párrafo termina en "…which the
  superclass recovery of Section 5.4 shows." (fuera "and the depth test reads as hub alignment"). **(5)** Entradilla de §5.2 "No
  family owns the structure." → "The structure depends on the class set more than on the model.". **(6)** §4: "explains the depth" →
  "explains the alignment".
- Sweep: literales de la entradilla de §5.1 y de la frase de Khrulkov actualizados; "weak and model-dependent" ya no se exige.
- **Cierre (2026-09-23, 12:15)**: FINAL_CHECK: "Main text (through the Conclusion and Limitations section) ends on page 9"; §7 en la línea
  463 de la página 9; 37 páginas; 0 `??`. Sweep 225/225. Versión paralela reconstruida.
## 63. Legibilidad, §5.3 (2026-09-23, 12:20, brief del autor): seis puntos.

- **(1)** Primer párrafo: fuera "It certifies 4 of 12:"; la lista sigue a la entradilla ("The depth test certifies that clusters are
  oriented toward their hubs in 4 of 12 backbones: ViT-S, ViT-B, ViT-L and DINOv2-L."), que el sweep exime del tope de 16 palabras;
  el párrafo queda en dos frases (exento). La definición del nivel de ruido pasa a "What the test can see", justo antes de la frase de
  las razones de los 3 ciegos y los 9 cubiertos.
- **(2)** "What the test can see": las dos frases sobre el desacoplado leyendo más profundo ("The decoupling control fires on every deep
  seed…" y "The deeper reading comes from both sides…") pasan a la nota (d) de la Tabla 10 (potencia), con los recuentos del fichero
  y el lado de la observación desde `final_dec_obs.json`. "The pre-set criterion asked…" sale de §5.3 y entra en la limitación (iii)
  como frase propia (el tope de 30 palabras impide la cláusula): "A pre-set criterion expecting no intact certification of the leaf
  and frozen models was not met, because they are aligned."
- **(3)** La frase de los Poincaré embeddings de WordNet pasa a la misma nota (d) con su explicación: "read with Euclidean distances,
  do not fire: the hierarchy they carry lives in the curvature and is not visible to a Euclidean reading, so they are not a positive
  control." El párrafo (c) cierra con "Tables 10 and 9 give the runs."
- **(4)** "No hub hierarchy is found where the test has power.": sin las listas repetidas; "In those 9 backbones, none as strong as
  the planted one is found." y las frases sobre qué significa "hub hierarchy". Las listas de cubiertos y ciegos quedan en la Tabla
  10(e) (los rellenos P81_COVERED/P81_UNCOVERED siguen calculados y comprobados).
- **(5)** Etiquetas de hoja: "…does not explain the certified ViTs' alignment."
- **(6)** Pie de la Figura 4: "\textbf{What the depth test certifies, and where it can see a hierarchy.} (a) Depth $z$ per backbone,
  filled when certified; line at $z=-2$. (b) Decoupled power for an implanted three-level hierarchy, filled at 0.8 or more; line at
  0.8." La forma larga de los nueve y los tres deja de estar en el cuerpo (queda en §1 con la glosa y en el pie de la Tabla 9).
- Sweep: literales de (a), (c), (d), (iii) y del pie de la Figura 4; las frases movidas se exigen en la nota (d) de la tabla de
  potencia; la definición del nivel de ruido debe preceder a la frase de las razones.
- **Cierre (2026-09-23, 12:35)**: FINAL_CHECK: "Main text (through the Conclusion and Limitations section) ends on page 9"; §7 en la línea
  455 de la página 9 y los statements empiezan en la 481, también en la página 9; referencias en la 500 (página 10); 37 páginas; 0
  `??`. Sweep 225/225. Versión paralela reconstruida.
## 64. Legibilidad, final de §5.3 a §7 (2026-09-23, 12:40, brief del autor): ocho puntos.

- **(1)** "Agreement with WordNet follows supervision and recipe." como entradilla de §5.4, y "agrees with WordNet" en el párrafo de las
  etiquetas de hoja (§5.3 f); el pie de la Tabla 12 (WordNet) pasa a "Agreement with the human taxonomy…" y su frase sobre los dos
  ViT-B de hoja a "the augreg recipe agrees with WordNet…", "DeiT-B barely agrees with it" y "agreement with WordNet is
  recipe-dependent". "Alignment" queda sólo con el sentido de la orientación de los clusters hacia sus hubs (barrido: ningún
  "align*" a menos de 80 caracteres de WordNet/taxonomy/correlation en plantilla, abstract ni pies).
- **(2)** §5.4, "Controlling the cut…" → entradilla "Controlled, the trees share their topology, not their metric." y cuerpo con tu
  frase, partida en tres por los topes de 30 palabras y de dos números por frase: "Across models, triplet agreement reaches
  {{TRIP_BIG}}, against {{CEIL_TRIP}} for two resamples of the same model. Chance gives 0.33. The cophenetic and cut agreements reach
  about half their within-model ceiling (Table 11)." Los demás números del párrafo (rangos y medias de las tres medidas, valores
  entre modelos) quedan en la Tabla 11(c); fuera la frase de "about a third". **Aviso sobre el azar**: tu frase decía "0.5 by
  chance", pero el acuerdo de tripletas de expR58 es sobre cuál de los tres pares de una tripleta se fusiona primero, y dos árboles
  independientes coinciden en un tercio de los casos; el 0.5 es el azar de las tripletas de hermanos de CIFAR-100 de la Figura 5(c)
  (elección binaria). El texto dice 0.33; si preferías otra lectura, dímelo.
- **(3)** §6, segundo párrafo: "The raw and calibrated readings predict the gain of a non-Euclidean readout equally well, and the depth
  verdict predicts none. Calibration tells what a low $\delta$ means, not which readout to use." (tu frase, partida en los dos puntos
  por el tope de 30 palabras: tenía 32); fuera la frase repetida y la de "buys interpretation".
- **(4)** §6, primer párrafo: "Applied, with the supremum it was calibrated for, to structureless Gaussian clouds, it assigns curvatures
  from {{C_LO}} to {{C_HI}}." en lugar de las dos frases anteriores.
- **(5)** MERU: "no hub structure". **(6)** §5.5: fuera "Models share who is near whom and differ in the metric"; la cita queda en la
  frase del párrafo. **(7)** Limitación (iii): "it tests hierarchy among the frame's hubs only". **(8)** Conclusión: "Cosine collects
  most of what is there."
- Sweep: literales de las revisiones quinta, novena, décima, duodécima y de la tesis actualizados a las frases nuevas (los cocientes
  entre modelos y techo siguen comprobados desde los ficheros); "0.33" en HEADLINE; "two resamples of the same model" como cantidad
  descriptiva; exenciones de la nueva entradilla de §5.4.
## 65. Figura 5, pie y panel (b) (2026-09-23, 13:10, brief del autor).

- Pie: "\textbf{The island is the cut; tree topology is shared well above chance, though short of the within-model ceiling; the
  self-supervised structure is angular.}" con las tres cláusulas de panel sin cambios (la de (b) sigue diciendo "against the
  within-model ceiling (band)"; la línea de azar la explica la leyenda).
- Panel (b): título "(b) topology, against chance and the ceiling" y línea vertical discontinua en 1/3, el azar de la elección a tres
  (qué par de la tripleta se fusiona primero); eje x ampliado de 0.6–1.0 a 0.25–1.0 con marca en 0.33. La entrada "chance" de la
  leyenda sirve ahora a (b) y a (c) (0.5 en (c), elección binaria). Sin cambio de datos ni de altura de la figura.
- §5.4 conserva "Chance gives 0.33.". Sweep: literal del pie actualizado y comprobación de que el script de figuras traza la línea en
  1/3 con ese título de panel y de que la frase de §5.4 sigue en el texto.
## 66. Bibliografía del autor (2026-09-23, 13:40): `iclr2027_conference.bib` revisada e integrada en `references.bib`.

- El fichero que compila el envío (y el de la vía paralela) es `references.bib` (`\bibliography{references}` en la plantilla); el sweep
  también lo lee. Tus 52 entradas de `iclr2027_conference.bib` pasan a `references.bib` en tu orden; ese fichero tuyo queda como lo
  dejaste. `references.bib` tenía 55 entradas: 46 idénticas a las tuyas, 6 solapadas con diferencias y 3 que las tuyas no tienen
  (sala2018representation, gu2019learning, moreira2024hyperbolic, las tres citadas), que se conservan al final del fichero.
- Solapadas con diferencias, verificadas: **CLIP** (tuya: ICML 2021 en PMLR; la anterior era el CoRR de bibsonomy; no se cita),
  **BGE** (guiones; no se cita), **cifar10** (tuya: techreport de Toronto; no se cita), **groger2026aristotelian** (citada): la
  página de arXiv 2602.14486 da los autores Fabian Gröger, Shuo Wen y Maria Brbić y "ICML 2026 camera-ready", así que la entrada
  queda como inproceedings de ICML 2026 con "Shuo Wen" (la anterior decía "Song Wen", error) y el DOI/eprint de arXiv de la tuya;
  **beyer1999nearest** (citada): tuya (DBLP, "Kevin S. Beyer", editores, DOI), con las comillas de apertura del título en LaTeX
  (DBLP escribe '' a ambos lados y se imprimiría ”Nearest Neighbor”); **aggarwal2001surprising** (citada): tuya (DBLP, editores,
  DOI) con el título en singular, "High Dimensional Space", como en el PDF de los autores y el repositorio de Konstanz (DBLP y
  Semantic Scholar escriben "Spaces").
- Dos defectos de impresión ya presentes: los DOI/URL de DBLP llevan `\_`, que bajo el `\doi` y `\url` verbatim del .bst imprime la
  barra ("7\_34" en sarkar2011low en el PDF actual): quitadas las barras en los tres DOI y URL; y el .bst pone los títulos en
  minúsculas, con lo que fournier2015computing imprimía "gromov": "{Gromov}" protegido; por la misma razón, protegidos "{Delaunay}"
  (sarkar2011low, imprimía "delaunay"), "{Plato's}" (koepke2026cave), "{GloVe}" (tifrea2019poincare, imprimía "glove"),
  "{Aristotelian}" y "{Platonic}" (groger2026aristotelian, huh2024platonic) y "{Chinese}" (BGE, no citada). Barrido de todas las
  palabras con mayúscula fuera de llaves en los títulos: el resto son mayúsculas de título (Title Case) que el .bst reduce bien.
- Sweep: comprobación nueva sobre `references.bib` (claves únicas, toda clave citada presente, Gröger como ICML 2026 con Shuo Wen,
  sin `\_`, Gromov protegido, las tres entradas conservadas). Sin cambio en el texto principal.
## 67. Abstract, cuatro ediciones (2026-09-23, 13:50, brief del autor); tesis sin cambios.

- (1) Frase 2: "…and a hierarchy test whose ability to detect a hierarchy is measured". (2) Frase 6: "the hierarchy test finds
  structure". (3) Frase 7: "no hierarchy as strong as the planted one is found". (4) Tras la frase 8: "In text, the result depends
  on the model's recipe and size." (5) Frase 9 partida en dos: "Across models, the usual comparison makes the self-supervised models
  look like outliers, an artifact of how trees are cut into groups. Compared properly, the trees agree on which classes group
  together well above chance, though less than two readings of the same model, and not on distances; the self-supervised models
  organize classes by direction rather than by distance."
- Abstract de 359 palabras (tope del sweep 360). `OPENREVIEW_abstract.txt` regenerado desde el tex. §1 conserva sus dos frases
  espejo con "though short of what two readings of the same model reach" (no estaban en el brief); §5.3(d) conserva "none as strong
  as the planted one is found".
- Sweep: literales del abstract (frases 7 y 9, recuento de la frase de acuerdo 3 → 2, "none as strong…" 2 → 1) y comprobaciones de las
  tres frases nuevas; sin cambio de números.
## 68. Tesis, una cláusula (2026-09-23, 15:30, brief del autor).

- Abstract y §7: "agree on how those clusters nest" → "agree on which classes group together" en la frase de la tesis (sigue en 56
  palabras, la excepción declarada a la regla de 30). `OPENREVIEW_abstract.txt` regenerado.
- Sweep: literal de la tesis actualizado (comprobación de la tesis y comprobación de la duodécima revisión, que la exigen idéntica en
  abstract y §7); sin cambio de números.
## 69. Abstract y tesis, dos ediciones (2026-09-23, 15:45, brief del autor).

- (1) Las dos frases entre modelos del abstract ("Across models, the usual comparison makes… by direction rather than by distance.")
  sustituidas por tu frase: "Across models, the trees agree on which classes group together well above chance, though less than two
  readings of the same model, and not on distances; the self-supervised models organize classes by direction rather than by
  distance, which a comparison by distance misses." La isla queda sólo en §5.4 (y en la contribución 3 de §1, "The island is an
  artifact of the cut; …", que no tocaba el brief: dime si también sale de ahí).
- Espejo en §1: el párrafo de respuesta usaba la misma redacción ("Across models, the usual way of comparing trees makes the
  self-supervised models look like outliers… by direction rather than by distance.") y lleva ahora la misma frase nueva, idéntica a
  la del abstract. La cláusula de la tesis no aparece en §1, así que (2) no tiene espejo allí.
- (2) Tesis "agree on which classes group together": ya estaba desde §68 (commit f3bb6e9); sin cambio adicional.
- Abstract: 344 palabras. `OPENREVIEW_abstract.txt` regenerado. Sweep: literal del abstract, la frase nueva exigida dos veces (abstract y
  §1, la segunda tras \section{Introduction}), "look like outliers" ausente del envío, y el recuento de "though short of what two
  readings of the same model reach" 2 → 1 (sólo la contribución 3).
## 70. Tesis nueva (2026-09-23, 16:15, brief del autor).

- Abstract y §7 (idéntica): "Read correctly, foundation models organize classes into clusters that they partly share, show no
  hierarchy among the superclasses where the test can see one, and their raw tree-likeness is not evidence for hyperbolic geometry."
  (34 palabras; antes 56). Sigue siendo la única excepción declarada a la regla de 30 palabras en las páginas 3–9 (FINAL_CHECK
  recalcula el recuento). La tesis no aparece en §1, así que no hay espejo que actualizar.
- Abstract: 322 palabras. `OPENREVIEW_abstract.txt` regenerado.
- Sweep: literal THESIS actualizado (comprobación de la tesis y de la duodécima revisión; la cláusula "though not as much as two
  readings of one model" sustituida por "clusters that they partly share" y el recuento de 34 palabras); sin cambio de números.
## 71. §1 reescrita (2026-09-23, 16:30, brief del autor): texto sólo, sin cambio de números.

- **(1) P1**: "…and they are read against the ideal values of a tree and of a non-hyperbolic space, never against what a structureless
  cloud of the same dimension and spectrum would score". **(2) P3**: fuera la última frase (GPT-2 M y los sentence embedders).
  **(3) P4**: "reads $\delta$ against the right reference"; "\citet{groger2026aristotelian} calibrate similarity across models";
  "\emph{hierarchy test}".
- **(4) "depth test" → "hierarchy test"** en todo el envío: plantilla (§3.4 pasa a titularse "Hierarchy test"; "The hierarchy test
  compares…", "The hierarchy test certifies…", MERU, limitaciones (iii) y (iv), el título de la sección del apéndice "Is the structure
  deeper than a star? The hierarchy test", el pie de la Figura 4), generador del apéndice (pies de las Tablas 3, 9, 10 y 14 y
  "hierarchy-test $z$"), etiqueta del eje de la Figura 4(a) ("hierarchy test $z$"), pies del fichero de la vía paralela, y
  "depth verdict" → "hierarchy verdict" (§6 y Tabla 14). El constructor aplica además el renombrado al documento ensamblado y
  comprueba que no queda ningún "depth test". "Depth" se conserva donde nombra el estadístico $z$ ("Depth $z$", "depth statistic",
  "depth ($z$)") y, **pendiente de tu decisión**, donde nombra el concepto: el título de §5.3 "Depth above the labelled clusters",
  el pie de la Figura 1 ("clusters without depth… with depth"), §4 "creates depth", §3.3 "not depth", limitación (i) "depth needs
  the separate test" y el pie del apéndice "misses implanted depth".
- **(5) P5** en tres partes con etiquetas en línea, tu texto verbatim (con los rellenos {{N_GEN}}/{{N_IN}} del constructor donde
  ya los usaba). La glosa del hub queda en P5 ("its superclass center, which we call its hub"); la de la frame sale de §1.
  La comprobación de "hub glosado en su primer uso" ignora el pie de la Figura 1 ("clusters around a hub", tu pie verbatim).
- **(6) Contribuciones**: los cuatro ítems sustituidos por los tuyos. Nota: "a nominally hyperbolic backbone" pasa a
  "a hyperbolic-trained backbone" (era vocabulario de la novena revisión; ahora no aparece en el envío).
- **(7) Glosa de la frame** en su primer uso, la Definición 6 de §3.4: "Given a grouping of classes into superclasses, which we call
  the frame, assigning each centroid $x_i$ to a cluster $k(i)$…". §4 conserva su paráfrasis propia ("The frame, the grouping into
  superclasses, is the WordNet cut…"). Es el primer "frame" del envío (§1 sólo tiene "frames the question").
- Mecánica: 11 ediciones nuevas en la lista del constructor (28 en total): 4 sustituyen ediciones anteriores (su `a` es la `b`
  anterior: P5, contribuciones 2 y 3, la frase de §2 "It adds a hierarchy test…") y 7 actúan sobre texto verbatim de
  `main_local.tex`. La comprobación de las ediciones salta las `b` reemplazadas por una edición posterior.
- Sweep: literales de §1 (P5, contribuciones, glosas, posición de la frase entre modelos: 1 vez en el abstract y en §1 con
  "\emph{Across models,}"), recuento de "though short of what two readings…" 1 → 0, "nominally hyperbolic" → "hyperbolic-trained",
  comprobación nueva del renombrado (envío, tablas del apéndice, script de figuras, título de §3.4), y las entradillas exentas con el
  nombre nuevo.
## 72. Figura 1, pie y separación (2026-09-23, 17:00, brief del autor).

- Última frase del pie: "The instrument reads the objects instead of the shadow: the excess separates the random cloud from the other
  two, and the hierarchy test separates the star from the tree." (antes "no structure beyond the null in the cloud, clusters without
  depth in the star, clusters with depth in the tree"; de paso desaparece uno de los "depth" conceptuales pendientes).
- `\vspace{2pt}` entre la imagen y el pie: la imagen, recortada con `trim`, tocaba la línea "Figure 1:" (página 2 de qa_pages).
- Mecánica: dos ediciones más en la lista del constructor (30), sobre el texto verbatim de `main_local.tex`; sweep con el recuento
  30 y la frase y el `\vspace` fijados en la comprobación de §1.
## 73. Tabla 1: columnas del supremo (expR85 terminado, 2026-09-23, 17:05) y la Figura 1 nueva del autor.

- expR85 (`expR85_khrulkov_sup.py`, 8 fragmentos, 10 ensayos por conjunto, 200 réplicas por ensayo, supremo exacto de $\delta_{\text{rel}}$
  en float32; unos 72 min por ensayo bajo carga, lanzado a las 10:59, terminado a las 17:04) fusionado en
  `expR85_khrulkov_sup.csv` y `expR85_khrulkov_sup_summary.csv`. La Tabla 1 lleva ahora dos columnas más, "excess (sup.)" y
  "$r$/200, largest $p$ (sup.)" (medias sobre los 10 ensayos; rango medio; el $p$ mayor de los 10): CIFAR-10 $-0.0205$
  (156/200, 0.701), CIFAR-100 $-0.0353$ (187/200, 0.299), CUB-200 $-0.0218$ (163/200, 0.488), MiniImageNet $-0.0700$ (200/200,
  0.005). §5.1 cierra con la cláusula acordada: "Table 1 gives the 4 rows, with their own statistic, the supremum, calibrated on the
  same replicates beside the record statistic." (fuera la cláusula provisional del percentil). Procedencia: `% prov` de la tabla y
  `% expR85_khrulkov_sup_summary.csv` en §5.1.
- Lectura para el autor: con su propio estadístico ninguno de los 4 conjuntos lee por encima de la nube aleatoria; sólo MiniImageNet
  queda por debajo en los 10 ensayos ($p$ mayor 0.005); CIFAR-100, que con el percentil "cae por debajo" (198/200, 0.030), con el
  supremo queda en 187/200 con $p$ mayor 0.299, es decir, indistinguible por el criterio del $p$ mayor. La frase de §5.1
  ("Calibrated, CIFAR-10 and CUB-200 are indistinguishable… CIFAR-100 and MiniImageNet fall below it") sigue refiriéndose al
  estadístico de registro; si quieres una frase sobre la lectura con el supremo, dímela.
- Figura 1: tu `fig1_concept.pdf` nuevo (commit f1c01ea, "All cast the same tree-shaped shadow", sombra "δ low") compila con el mismo
  `trim` y el `\vspace{2pt}` de §72; comprobado en la página 2 de qa_pages.
- Sweep: la comprobación del punto 5 de la duodécima revisión pasa a la rama "fusionado" (celdas de la tabla desde el resumen,
  10 ensayos, cláusula de §5.1); la comprobación de limpieza de la Tabla 1 (`kt_ok`) actualizada a las 7 columnas.
## 74. §5.1 y §1 con la lectura del supremo; pie de la Tabla 1; duplicado de la Figura 1 (2026-09-23, 17:20, brief del autor).

- §5.1, párrafo de Khrulkov: la cláusula de cierre pasa a tu frase "Under their own statistic, the supremum, calibrated against the
  same replicates, CIFAR-100 joins CIFAR-10 and CUB-200 among the indistinguishable, and only MiniImageNet falls below the random
  cloud." (27 palabras). Va antes del puntero "Table 1 gives the 4 rows.", que cierra el párrafo: la regla de prosa del sweep
  (un puntero a tabla, en la última frase) fallaba con la frase nueva al final. El constructor comprueba desde
  `expR85_khrulkov_sup_summary.csv` que el $p$ mayor de los 10 ensayos supera 0.05 en CIFAR-10, CIFAR-100 y CUB-200 y no en
  MiniImageNet.
- §1, respuesta (i): "…2 of them are indistinguishable from a random cloud, 3 under their own statistic" (comprobado: 3 conjuntos con
  $p$ mayor $>0.05$ bajo el supremo).
- Tabla 1: el rango del supremo es ahora la **mediana** de los 10 ensayos (`r_above_median`, columna nueva del resumen de expR85;
  antes la media), $p$ el mayor; pie: "For the supremum the excess is the mean and the rank the median over the 10 trials, and $p$
  the largest. Under the supremum, CIFAR-100 is not below the null at 0.05." **Aviso**: pedías "under either", pero con el
  estadístico de registro CIFAR-100 sí queda por debajo a 0.05 (198/200, $p$ mayor 0.030, y §5.1 dice que "falls below"); el pie
  dice sólo "under the supremum", y el generador y el sweep lo comprueban desde los dos resúmenes. Si querías otro criterio, dímelo.
- `ICLR2027/figures/fig1_concept.pdf` (copia antigua, distinta de la tuya) borrado con `git rm`; el tex lee
  `iclr2027/figures/fig1_concept.pdf`, que es tu fichero de f1c01ea; página 2 comprobada tras recompilar.
- Sweep: comprobación del punto 5 con la mediana, la frase de §5.1, el "3" de §1 y las dos frases del pie, todas desde los ficheros.
## 75. Lectura del supremo corregida: decide la mediana (2026-09-23, 17:40, brief del autor).

- Del fichero de ensayos (`expR85_khrulkov_sup.csv`): $p$ mediana de los 10 ensayos CIFAR-10 0.169, CIFAR-100 0.025, CUB-200 0.132,
  MiniImageNet 0.005; $p$ mayor de CIFAR-100 0.299 (por debajo de 0.05 en 7 de 10 ensayos). El resumen lleva ahora `p_left_median`.
- (1) §5.1: tu frase, partida en dos por el tope de 30 palabras (tenía 35): "Under their own statistic, the supremum, CIFAR-10 and
  CUB-200 stay indistinguishable from a random cloud and MiniImageNet stays below it. The verdict for CIFAR-100 changes from trial
  to trial, as the statistic confound predicts." Sigue antes del puntero "Table 1 gives the 4 rows.".
- (2) §1, respuesta (i): fuera ", 3 under their own statistic".
- (3) Tabla 1: para el supremo, excess = media, rango = mediana de los 10 ensayos y la columna de $p$ pasa a ser el **rango** de $p$
  entre ensayos (cabecera "$r$/200, $p$ range (sup.)"); pie: la redacción del estadístico de registro se mantiene, y para el supremo
  "the mean excess, the median rank over 10 trials and the range of $p$ across trials. CIFAR-100 is below the null in the median
  trial but not in every trial."
- (4) Sweep y constructor y generador: desde el resumen de expR85, $p$ mediana $<0.05$ en CIFAR-100 y MiniImageNet, $>0.05$ en
  CIFAR-10 y CUB-200, y $p$ mayor de CIFAR-100 $>0.05$; celdas de la tabla y las tres frases fijadas. El aviso de §74 queda resuelto
  con este criterio.
## 76. Pase final de exactitud (2026-09-23, 18:00, brief del autor): citas de modelos y datos, bibliografía, MERU, fuentes.

- **(A) Citas en la primera mención.** §4: ViT \citep{dosovitskiy2021an}, ImageNet \citep{imagenet}, DINO \citep{dinov1}, DINOv2
  \citep{dinov2}, CLIP \citep{CLIP}, SigLIP \citep{siglip}, CIFAR-10/100 \citep{cifar10}, DTD \citep{dtd}, FMNIST \citep{fashion},
  MNIST \citep{mnist}, GPT-2 \citep{gpt2}, Pythia \citep{pythia}, OLMo \citep{olmo}, ResNet \citep{resnet}, DeiT \citep{deit},
  augreg \citep{augreg}, MERU \citep{desai2023meru}, WordNet \citep{wordnet}; §3.3 Benjamini--Hochberg
  \citep{benjamini1995controlling} donde se define el umbral; §5.1 CUB-200 \citep{cub} y MiniImageNet \citep{miniimagenet}; §5.4
  DBpedia \citep{dbpedia}; los embedders (BGE, GTE, E5) citados en la tabla del panel del apéndice, único sitio donde se nombran.
  Las citas van dentro de las frases existentes; el texto principal sigue en la página 9 (ver FINAL_CHECK). La lista de
  referencias pasa de 35 a 59 entradas citadas y de 3 a 6 páginas (10–15; PDF de 40 páginas): las entradas de DBLP llevan editores,
  título completo de las actas y URL. Si quieres acortarla, dime si quito editores y URL en el .bib o cambio el .bst.
- **Entradas nuevas verificadas** (metadatos del registro del editor vía Crossref y clave DBLP vía Semantic Scholar; sin metadatos
  escritos a mano): siglip, deit, augreg, cub, miniimagenet, wordnet, dbpedia, resnet, benjamini1995controlling. Detalle de fuentes en
  la sección nueva de FINAL_CHECK. Fuentes usadas: Crossref (registros del editor: IEEE para SigLIP y ResNet, ACM para WordNet,
  Semantic Web para DBpedia, JRSS-B para Benjamini--Hochberg), páginas de actas de PMLR (DeiT) y NeurIPS (MiniImageNet), el
  registro de Caltech Authors (CUB: el registro da el número CNS-TR-2010-001, no el CNS-TR-2011-001 que suele citarse; se toma el
  del registro) y la referencia de revista del propio arXiv para augreg (TMLR, 05/2022). DBLP y OpenReview bloquean el acceso
  automático; la clave DBLP de gu2019learning (conf/iclr/GuSGR19) viene de Semantic Scholar.
- **(B) Bibliografía**: park2024geometry → inproceedings, International Conference on Learning Representations, 2025;
  koepke2026cave → note "arXiv preprint arXiv:2604.18572"; aggarwal2001surprising → título "…in High Dimensional Spaces" (tu
  decisión; DBLP).
- **(C) §5.3 MERU**: tu frase, partida en tres por el tope de 30 palabras: "Read with the census and the hierarchy test, MERU shows
  the same clustering as its twin, and no hub structure is detected. The test's power at MERU's spectrum, however, was not measured.
  Its embeddings stay in the near-flat regime, 95 per cent of them within {{MERU_P95}} of the curvature scale." El 0.29 es el
  relleno MERU_P95: máximo, sobre los 3 MERU y los 2 conjuntos, del percentil 95 de la norma espacial por $\sqrt{c}$
  (`expR71_meru_radii.csv`, columna radius_sqrtc_p95; 0.287 en MERU-L/ImageNet).
- **(D) FINAL_CHECK**: sección nueva "Cited bibliography entries and their sources" con la clave DBLP (de `biburl`), el DOI o la URL
  del editor de cada entrada citada; a las entradas citadas que no tenían campo de fuente se les añade `biburl`/`doi`/`url`
  verificados.
- Sweep: comprobación nueva (todas las claves de (A) citadas en el texto principal o en la tabla del panel; toda entrada citada con
  fuente; los tres cambios de (B); las frases de MERU con el relleno desde expR71); dos literales de §5.1 con las citas nuevas.
## 77. Bibliografía, dos comprobaciones (2026-09-23, 18:40, brief del autor).

- aggarwal2001surprising: título de vuelta al singular, "…in High Dimensional Space" (como en la página de Springer; DBLP escribe
  "Spaces"). Sweep: literal de la comprobación del pase de exactitud actualizado.
- koepke2026cave: confirmado, `note={arXiv preprint arXiv:2604.18572}` (más `eprint`/`url` de arXiv); la comprobación del sweep lo
  exige.
## 78. Dos frases de alcance y la lista W1-W5 (2026-09-23, 19:00, brief del autor).

- **§5.1**, tras la frase de que la premisa no sobrevive a la calibración, tu frase partida en dos por el tope de 30 palabras (tenía
  34): "The excess is conservative: the null keeps the spectrum, so a hierarchy carried by the spectrum alone would not show. What
  fails is the evidence the premise cites, not the possibility of a hierarchy." El párrafo queda en cinco frases (tope 6) y el
  puntero a la Tabla 5 sigue en la última.
- **§6**, al final de la sección: "We test zero-cost readouts only; whether training in hyperbolic space helps for reasons other than
  the latent hyperbolicity it cites is outside this study." (24 palabras). Va después de la frase que cita la Tabla 14, así que el
  último párrafo de §6 se añade a la excepción de la regla "el puntero a tabla va en la última frase" (ya la tenía la entradilla de
  §5.4, que cita la Figura 5 en la primera). Es la única regla de prosa que toca.
- **Lista de réplica W1-W5** en `REVIEWER_CHECKLIST_third.md` (sección nueva, y en su generador), sin acción, cada una con su diseño:
  W1 curva de potencia a nivel de muestra con árboles plantados (§5.1 da un negativo sin curva de potencia allí: la potencia está
  medida en centroides, Tabla 10); W2 efecto mínimo detectable por backbone (convierte "the test is blind in the other 3" en un
  número, barriendo la fuerza plantada de expR81); W3 frame derivado de los datos con partición de la muestra (frame en una mitad,
  test en la otra, frente al corte de WordNet y al control balanceado); W4 un único camino preespecificado en un conjunto retenido
  (limitación (vi): las decisiones de análisis se fijaron sobre la marcha); W5 normalización por la distancia mediana en lugar del
  diámetro (limitación (vii)).
- Sweep: comprobación nueva de las dos frases (la de §5.1 tras "no more than chance would give", y §6 acabando en la de alcance);
  excepción del puntero documentada. Sin cambio de números.
## 79. Limpieza del apéndice (2026-09-23, 20:00, brief del autor). Texto principal intacto byte a byte hasta `\appendix`.

- **(1) Página de lectura y glosario** (`B.1 How to read this appendix`): una línea por sección (pregunta → tabla → qué muestra,
  13 líneas) y un glosario de símbolos y columnas: "the reading used in the paper" (sustituye a "record"), excess, exc./null,
  $r$/200, $p$ de cola izquierda, BH y genuine, $^{\circ}$ / $^{\dagger}$ / $^{\ddagger}$, negrita, frame, hub, matched star,
  Haar-hub star, $z$, decoupled, power y false alarms. Las tablas ya no redefinen: remiten al glosario.
- **(2) Pies de tabla**: uno por tabla, de 28 a 56 palabras (tope 60), con una frase en negrita que da la respuesta y otra que
  nombra las columnas; los paneles continuados llevan sólo "(continued)". Cada tabla va precedida de un párrafo llano de 2 a 4
  frases ("What this table answers"), que sustituye a los párrafos de la versión v1 y a la explicación que vivía en los pies.
  Los títulos de panel pasan de 834 a 373 palabras: son etiquetas, no definiciones.
- **(3) Paneles borrados** (ningún enunciado del texto principal se apoya en ellos): Tabla 3(e) intervenciones sobre la lectura
  cruda dentro de una arquitectura; Tabla 7(b) plantillas de prompt; Tabla 8(b) la réplica de Khrulkov (ya es la Tabla 1 del texto
  principal); Tabla 10(a) jerarquías sintéticas en el frame de hojas y en el de nivel superior; Tabla 14(e) políticas de selección
  de métrica en few-shot, 14(f) las mismas en nearest-centroid y 14(g) sensibilidad al radio de proyección. **Comprobados y
  conservados**: Tabla 5(d) (el texto principal nombra los dos ResNets auto-supervisados) y Tabla 12(d) HierarCaps (§5.4 lo cita);
  los quité primero y la comprobación del sweep los recuperó. **Material retirado**: las columnas de la estrella isotrópica de la
  Tabla 9(a) y la figura de matrices ARI (`fig:treemapmat`), que no citaba nadie. El implante de dos niveles se queda: el texto
  principal lo cita (Figura 6 y §5.3).
- **(4) Vocabulario**: "of record" desaparece del apéndice ("the census", "the reading", "under the census reading"); el glosario
  dice una vez "the reading used in the paper". **Aviso**: el texto principal conserva "reading of record" y "frame of record"
  (no se puede tocar por el propio brief), así que apéndice y texto principal difieren en esa expresión; dime si quieres que la
  cambie también en el texto principal.
- **(5) Comprobación de conservación relajada**: la copia final de cada tabla sólo puede perder números respecto de la v1, nunca
  ganar, y debe abrir con el pie corto. Salen del apéndice 277 números (pies largos y paneles borrados). De ellos, cuatro los cita
  el texto principal y ya no aparecen escritos en el apéndice: 72 (las filas del censo, que la tabla sigue mostrando), 47, 34 y
  +0.9; los demás eran sólo del apéndice. Dime si quieres alguno de vuelta en un pie.
- **Páginas**: el apéndice pasa de 25 a 16 páginas (proofs incluidos; el PDF, de 40 a 31). Falta una para tu tope de 15, y con la
  regla de (3) aplicada no queda nada sin citar que borrar: para bajar a 15 habría que quitar material citado (p. ej. la Tabla 5(b)
  de construcciones o la Tabla 5(c) del censo coseno). Los paneles ahora se colocan donde caen, no flotan, lo que llenó las páginas
  medio vacías; `arraystretch` 0.92 y menos aire entre paneles.
- Sweep 230/230 con comprobación nueva de la estructura del apéndice (página de lectura, glosario, un párrafo por tabla, pies de
  60 palabras o menos, paneles borrados ausentes, sin "of record"); los pines que citaban frases de los pies largos se han
  eliminado o reapuntado. Texto principal en la página 9 y byte a byte idéntico hasta `\appendix`.
## 80. Glosario, "the reading", el panel del corolario y la versión con cajas (2026-09-23, 21:00, brief del autor).

- **(1) Glosario del apéndice**, tus tres correcciones, las dos primeras porque lo que yo había escrito era falso:
  "The reading used in the paper: the 99.9th-percentile statistic on Euclidean distances between class centroids, against the
  centered Haar null, 200 replicates; the cosine census is a robustness column (Table 5c)"; "excess: … Negative means more
  tree-like than a random cloud of the same shape; positive, less" (la Definición 4 resta la media del nulo, y $\delta$ crece al
  alejarse del árbol: negativo = más arbóreo, justo lo contrario de lo que yo puse); "no cell of an appendix table is bold; in the
  main-text census table bold marks cells above the null" (comprobado: la Tabla 2 tiene una celda en negrita y su pie dice
  "bold: above the null"). Barrido del apéndice en busca de otras frases sobre el signo o la geometría: no queda ninguna; el título
  del panel (c) del censo ("The same census on cosine geometry…") y el párrafo de esa tabla ya eran correctos.
- **(2) "reading of record" → "the reading"** en el texto principal: Definición 2 pasa a llamarse `[reading]` y dice
  "$\delta_{\text{norm}}$ …, is the statistic used throughout; we call it the reading"; el pie de la Tabla 2 dice "Excess of the
  reading over its matched null…". Cero apariciones de "reading of record" en el envío. La comprobación de vocabulario del sweep
  nombra ahora la Definición 2 "reading". **"frame of record" sigue** (2 veces, §5.3): no estaba en el brief, dime si también.
- **(3) El panel del corolario no había que restaurarlo**: el rango "+0.9 to +1.3 pp" sale de `exp2b_normalized_stack.csv`
  (columna FS_HN_COS_diff, backbones contrastivos en CIFAR-100, CIFAR-10 y DTD) y esos seis valores (+0.90, +0.91, +1.21, +1.23,
  +1.24, +1.29) están en la última columna de la Tabla 14(a), que está en el apéndice. Los paneles borrados (e), (f), (g) eran
  políticas de selección de métrica y el barrido del radio de proyección, que sostenían frases de prosa del apéndice ya retiradas.
  **72**: 12 backbones × 6 conjuntos (§4) y además escrito en el pie de la Tabla 2 ("BH over 72 cells"). **47**: es el número de
  clases de DTD; no lo dice ni §4 ni ninguna tabla, así que la frase "datasets with 47 classes or more" se apoya en un dato que el
  lector no ve: dime si lo añado a §4 o al panel de modelos. **34**: falsa alarma mía, sólo aparece en "ResNet-34", nombre de
  modelo, no es un número citado.
- **(4) `ICLR2027/iclr2027/main_iclr2027_boxes.tex`** (envío intacto), generado por `rebuttal/scripts/make_boxes.py`: las 8
  definiciones en caja azul claro y la Proposición 1 en ámbar, con `tcolorbox`, filete fino a la izquierda, sin marco, 3 pt de
  relleno y la misma fuente; el título en negrita ("Definition 3 (Haar null).") es la cabecera de amsthm dentro de la caja. Compila
  en 32 páginas: la Conclusión abre en la 9 y las declaraciones en la 10, es decir, **el texto principal acaba en la página 10**,
  una más que el envío. No he recortado nada, como pediste. PDF en `ICLR2027/main_iclr2027_boxes.pdf`.
- Envío recompilado: texto principal en la página 9, sweep 230/230, apéndice en 16 páginas.
## 81. Frame de WordNet, cuentas de clases, Figura 2 por familias y la variante con cajas (2026-09-23, 21:30, brief del autor).

- **(1)** §5.3: "the frame of record" → "the WordNet frame" en las dos frases (el control entrenado y la frase del frame balanceado;
  literal del sweep y `final_bal_frame.json` regenerados). §4 da ahora el número de clases en la primera mención: "The class sets are
  ImageNet with 1000 classes and CIFAR-100 with 100 classes, which carry a real hierarchy, and CIFAR-10 with 10 classes and DTD with
  47 classes. The flat sets are FMNIST and MNIST, 10 classes each." (dos frases por el tope de 30 palabras; las cinco cuentas son
  frases de cuenta, exentas de la densidad de números). Con esto el 47 de §5.2 ya está en el papel.
- **(2)** Figura 2(a): orden por familia y tamaño (ViT-T/S/B/L; DINO-B, DINOv2-S/B/L/G; CLIP-B, CLIP-L, SigLIP-B) en lugar de por
  dimensión, y la referencia gaussiana pasa de una línea escalonada a un guion corto sobre cada par de barras. Pie: "…by family and
  size; within each family the null falls as the dimension grows, and the gap is the excess."
- **(3)** Variante con cajas (`main_iclr2027_boxes.tex`, envío intacto): cajas más apretadas (2 pt de relleno, 2 pt antes y después,
  sin espacio extra tras el título), §5.5 movida al apéndice con un puntero de una frase al final de §5.4 ("Models share
  neighborhoods, not metrics. Permutation-calibrated agreement across models survives, and Appendix B.12 gives the reading."), y
  (1) y (2) aplicadas al heredarlas del envío. **No cabe en la página 9: el texto principal se desborda una línea** (la línea 486
  cae en la página 10, con las declaraciones detrás), así que no promociono el fichero y no toco nada más, como pediste.
- Envío recompilado: texto principal en la página 9, sweep 230/230.
## 82. Un panel, una tabla numerada (2026-09-23, 22:00, brief del autor).

- Cada panel de las tablas del apéndice es ahora una tabla numerada aparte, con su propio pie de **40 palabras o menos** (frase en
  negrita con la respuesta, luego qué son las columnas) y su propia etiqueta. El apéndice pasa de 12 tablas con paneles a **38
  tablas numeradas** (Tablas 3 a 40), repartidas en los mismos 12 ficheros y las mismas 12 subsecciones, una por pregunta.
- El título en negrita que encabezaba cada panel desaparece: lo dice el pie. Las etiquetas son la de la pregunta para la primera
  tabla del grupo y `-b`, `-c`, `-d`, `-e`, `-f`, `-ap` para las demás; cuando el panel (a) se había borrado (potencia), la etiqueta
  base viaja a la primera tabla superviviente, de modo que ninguna referencia queda colgando (`?? count: 0`).
- **Referencias actualizadas** (once en el texto principal): el exceso de la estrella → Tabla 8; el bootstrap → Tabla 4; los detalles
  del censo → Tablas 9 y 5; las lecturas sin centrar y gaussiana → Tabla 10; el censo coseno → Tabla 11; "both stars" → Tablas 16 y
  17; las corridas → Tablas 24 y 20; MERU → Tabla 19; las tasas de detección y la potencia → Tabla 25; los acuerdos cofenético y de
  corte → Tablas 27 y 28; DBpedia y HierarCaps → Tablas 31 y 32.
- La página de lectura sigue agrupando por pregunta: cada línea nombra ahora el rango de tablas de esa pregunta. El párrafo llano de
  2 a 4 frases se mantiene uno por pregunta, delante de su primera tabla, y ya no habla de "paneles" sino de tablas.
- Sweep: comprobaciones del apéndice adaptadas (pie de 40 palabras o menos, una etiqueta por tabla, orden de aparición comparado por
  pregunta, pines que citaban títulos de panel reapuntados a los pies nuevos); una relajación: un paréntesis que sólo contiene un
  puntero a tablas ("(Tables 27 and 28)") deja de contar como paréntesis de prosa. El constructor de la vía paralela detectaba el
  material ya incorporado por un título de panel: ahora lo detecta por el pie.
- Envío: texto principal en la página 9, apéndice en 16 páginas, PDF de 31, sweep 230/230.
## 83. La variante con cajas pasa a ser el envío (2026-09-24, brief del autor).

- **(1)** Cabecera de la tabla de potencia: "WordNet-30 frame of record" → "WordNet-30 frame".
- **(2) No había línea que recuperar: me equivoqué al medir.** Ayer informé de que la variante con cajas se desbordaba una línea;
  contaba como línea de texto el número de margen 486 de la página 10. Con la regla que usa la propia cadena (la ranura de la
  declaración de Reproducibilidad, 487, dividida entre las 54 líneas por página) el texto principal de la variante con cajas acaba
  en la **página 9**, igual que el envío sin cajas. Así que no he acortado ninguna frase: no hacía falta y el brief sólo lo pedía
  para recuperar esa línea.
- **Promoción hecha en el constructor**, no como post-proceso: `phaseE_submission.py` escribe ahora `main_iclr2027_final.tex` con
  las 8 definiciones en caja azul claro, la Proposición 1 en ámbar y §5.5 en el apéndice, y guarda el mismo texto sin cajas y con
  §5.5 en su sitio como `main_iclr2027_final_noboxes.tex`. `make_boxes.py` y los ficheros `*_boxes.*` desaparecen: la
  transformación vive en el constructor y la cadena la aplica en cada compilación.
- El puntero de §5.4 nombra la tabla además del apéndice ("…and Table~\ref{tab:q13-local} in Appendix~\ref{app:local} gives the
  reading"), de modo que el texto principal sigue citando todas las tablas del apéndice.
- Sweep: comprobación del preámbulo (ahora amsthm y tcolorbox, con el recuento de cajas 8 y 1), del esqueleto (siete secciones y
  **diez** subsecciones, §5.5 fuera del texto principal y su párrafo en el apéndice) y de la prosa llana (un párrafo que es una caja
  de enunciado es un enunciado, no prosa sin entradilla). 230/230.
- Envío: 31 páginas, texto principal en la página 9, apéndice de 16, qa_pages regeneradas.
## 84. Limpieza de jerga (2026-09-24, brief del autor). Sin cambio de contenido.

- **(1) "record" fuera de la prosa**: §3.3 "so it is the null used throughout"; pie de la Tabla 1 "for the reading (99.9th
  percentile)"; §5.4 "genuine under the reading"; cabecera de la tabla de texto "the reading" en vez de "record"; cabecera del
  apéndice "WordNet-30 frame" (y las otras tres apariciones de "frame of record" que quedaban en el generador); la etiqueta interna
  `def:record` pasa a `def:reading`. Comprobado: "record" no aparece ya en el texto del envío ni en las tablas. **Sigue en los
  nombres de fichero de los comentarios de procedencia** (`expR62_samplelevel_record.csv`, `expR63_meru_record.csv`, etc.), que son
  nombres de ficheros publicados y no prosa.
- **(2)** Definición 6 pasa a "(hub null and hub excess)" y el símbolo $e_B$ a $e_{\mathrm{hub}}$ en todas partes: definición,
  ecuación de la profundidad, §3.4 y las cabeceras y el pie de la tabla de la prueba de jerarquía ("star $e_{\mathrm{hub}}$",
  "depth = hub excess $e_{\mathrm{hub}}$…").
- **(3)** §5.4 y el pie de la Tabla 27: "the block" desaparece. **Aviso sobre el número**: pedías "the other 8 models"; la
  comparación de la figura y de la tabla enfrenta los 3 modelos DINOv2 con los 7 supervisados y contrastivos (los 12 menos DINO-B y
  DINOv2-S, que no entran en ninguno de los dos grupos), así que el texto dice "the other 7 models" y el pie "against the supervised
  and contrastive models". Si querías otra agrupación, dímelo.
- **(4)** §5.4 define la isla en su primer uso: "…the DINOv2 models then agree with none of the others, an island in the map of
  models." (frase partida en dos por el tope de 30 palabras: la segunda es la de (3)).
- **(5)** §4: "The census, the reading of every model--dataset cell, has 12 backbones, 6 class sets and 15 text models." La
  entradilla tiene 18 palabras, por encima del tope de 16, así que queda exenta como la de §5.3, por ser tuya.
- **(6)** El glosario del apéndice añade w/b (dispersión dentro de los clusters dividida por la dispersión entre hubs), tight o
  shrunk spread, $s^{*}$ (la fuerza de implante más pequeña a la que la prueba dispara) y $\bar z_1$ (el z medio a fuerza máxima).
- Envío: 31 páginas, texto principal en la página 9, sweep 230/230.
## 85. Estilo de las figuras (2026-09-24, brief del autor). Sin cambio de datos ni de contenido.

- **(1) Paleta**: `figures/palette.py` pasa a viridis discreto, supervisados #3B528B, auto-supervisados #21918C, contrastivos
  #5EC962, modelos de texto #440154 (un solo color para LMs causales y embedders, que en las figuras del envío no aparecen: viven en
  las tablas). Nulos, estrellas y controles siguen en gris #7F7F7F. Bandas y resaltes en #FDE725 al 35 %: la banda de dos
  desviaciones de la Figura 3 y el techo dentro del modelo de la Figura 5(b), cada una junto a la línea que lleva la misma
  información (el cero y la marca de la media), nunca como único portador.
- **(2) Ejes**: fondo #F4F6F8 con rejilla blanca sólida de 1 pt, marco superior y derecho ocultos, los otros en #BBBBBB.
- **(3) Marcas**: puntos con borde blanco de 0.8 pt y algo más grandes (4→5, y 4→4.5 los huecos); barras con borde blanco. Las
  barras rayadas conservan el borde del color de la familia: es lo que dice "no genuino" y "no certificado".
- **(4) Leyendas**: caja blanca redondeada con sombra suave (`frameon=True, fancybox=True, shadow=True, framealpha=0.95,
  facecolor='white', edgecolor='#DDDDDD'`) en las seis leyendas.
- **(5) Tipografía**: sans-serif con la lista TeX Gyre Heros, Helvetica, Nimbus Sans, Liberation Sans, DejaVu Sans; **en esta máquina
  no hay ni TeX Gyre Heros ni Helvetica, así que se usa Nimbus Sans**, el clon de Helvetica de URW, con las mismas métricas; en tu
  máquina tomará la primera que exista. Etiquetas de eje 9 pt, ticks 8 pt, leyendas 8 pt.
- **Dos ajustes que exigió el tamaño nuevo**: el conjunto matemático `stixsans` se comía los dígitos dentro de `$...$` (la leyenda de
  la Figura 4 decía "power ≥ ." en vez de "power ≥ 0.8"), así que las matemáticas usan `dejavusans`; y con ticks de 8 pt chocaban la
  etiqueta intermedia del eje x de la Figura 3 (queda la rejilla, se quita el "−0.06") y los títulos de los paneles (b) y (c) de la
  Figura 5, así que (b) pasa a "topology, chance and ceiling" (el literal del sweep va detrás).
- Revisado en qa_pages a 100 dpi y en escala de grises: los cuatro colores se separan también en gris (0.15, 0.28, 0.5, 0.68 de
  luminancia) y el relleno frente al rayado distingue las verdictos sin depender del color.
- Envío: 31 páginas, texto principal en la página 9, sweep 230/230.
## 86. Tres retoques de figura (2026-09-24, nota del autor).

- **Figura 4**: la leyenda tapaba los nombres de los modelos; baja fuera de ellos (`bbox_to_anchor` de $-0.01$ a $-0.13$) con un
  margen inferior algo mayor. Misma altura, mismos datos.
- **Figura 5(a)**: los nombres iban muy pegados. Las filas ganan aire desde los márgenes, no desde la altura: el eje pasa de ocupar
  el 49 % de la figura al 59 % (`top` 0.89→0.93, `bottom` 0.40→0.34) con la figura en 2.2 in. Lo probé primero subiendo la altura a
  2.45 in y el texto principal se iba a la página 10, así que lo hice por márgenes.
- **Figura 5(c)**: los nombres de los modelos van en negro; el color de familia lo llevan las barras.
- Envío: 31 páginas, texto principal en la página 9, sweep 230/230.
## 87. Figura 2 nueva, "el instrumento en una figura" (2026-09-24, brief del autor).

- Una fila, cuatro paneles, el mismo eje x: los 12 backbones de ImageNet agrupados por familia (supervisados; DINO/DINOv2;
  CLIP/SigLIP), unidos dentro de cada familia por una línea, al estilo de Gröger et al. (2026, Fig. 5), con la paleta y el estilo
  nuevos. (a) lectura cruda (punto de color) y media del nulo emparejado (rombo gris); (b) el exceso, barra llena cuando es genuino
  y rayada si no, con la marca de dos desviaciones del nulo por backbone; (c) el $z$ intacto (punto) y el decoupled (rombo), línea
  discontinua en $-2$; (d) la potencia decoupled por backbone, llena a partir de 0.8. Datos: expR75 (filas de ImageNet),
  expR74_decoupling_summary, expR81_deep_per_backbone_summary. Pie: el tuyo, verbatim.
- **Sustituye a las Figuras 2 y 4 anteriores.** El par de la misma lectura con veredictos opuestos (antigua 2b) pasa al apéndice
  como figura junto a la tabla del nivel de muestra. **La antigua 4b no se duplica**: era la potencia decoupled por backbone, que es
  exactamente el panel (d) nuevo; si querías además una figura de potencia sintética aparte en el apéndice, dímelo.
- **Etiquetas del eje**: doce nombres completos no caben bajo un panel de 1.1 in a un tamaño legible, así que cada modelo lleva su
  talla dentro de la familia (T, S, B, L | v1, S, B, L, G | B, L, Sig) y la familia va debajo (sup., SSL, contr.), como en la
  Tabla 2. Las leyendas van fuera de los datos, bajo la figura.
- Referencias actualizadas: §3.3 apunta ahora a la figura del apéndice ("Two cells … opposite verdicts (Figure 5)"), §5.3 a la
  Figura 2(c), y el pie de la tabla de la prueba de jerarquía a la Figura 2(c). Renumeración: Figura 1 concepto, 2 instrumento,
  3 exceso, 4 mapa de árboles, y en el apéndice 5 la misma lectura y 6 las curvas del implante.
- **La variante con cajas ya es el envío** (promovida el 24 por la mañana) y sigue acabando en la página 9, así que no hay nada que
  reintentar: lo confirmo con la regla de la cadena (declaraciones en la página 9, referencias en la 10).
- El apéndice pasa de 16 a 17 páginas por la figura que baja; el PDF, de 31 a 32. Sweep 230/230, texto principal en la página 9,
  qa_pages regeneradas.
## 88. Figura 2, cuatro arreglos (2026-09-24, brief del autor). Sin cambio de datos.

- **(1)** Una sola leyenda bajo los cuatro paneles, en caja blanca redondeada con sombra, con tus cinco entradas: punto (modelo,
  lectura cruda y prueba intacta), rombo (nube aleatoria de la misma forma / orientaciones aleatorizadas), llena (genuino o potencia
  $\ge 0.8$), rayada (no genuino o ciego) y discontinua (umbral).
- **(2)** Panel (b): fuera las líneas verticales de $\pm$2 s.d.; ahora cada barra lleva una marca horizontal gris corta en $-2$
  desviaciones del nulo, y la barra genuina la pasa por debajo. El eje deja un poco de aire sobre el cero para que se vea la línea.
- **(3)** Eje x: una posición vacía entre familias con un separador vertical fino y claro en cada hueco; las tallas por familia
  (T S B L | v1 S B L G | B L Sig) a 7 pt sobre su propia marca, y el nombre de la familia (sup., SSL, contr.) centrado debajo a
  7 pt. **Nota**: a 7 pt en horizontal "v1" y "Sig" son más anchos que su hueco (6.6 pt por modelo) y se tocaban, así que las
  etiquetas van giradas 90°, que a ese tamaño es lo único que cabe sin abreviar más. Si las prefieres en horizontal, hay que bajar a
  una letra por modelo (D para DINO-B y S para SigLIP-B).
- **(4)** Los paneles ocupan el ancho del texto: márgenes de 0.062 y 0.998 y separación entre paneles de 0.34 (antes 0.62).
- Comprobado en qa_pages a 100 dpi. Envío en 32 páginas, texto principal en la página 9, sweep 230/230.
## 89. Figura 2 en dos paneles (2026-09-24, brief del autor). Sin cambio de datos.

- Dos paneles en una fila, ancho del texto, 2.05 in de alto, al estilo de Gröger et al. (2026, Fig. 5): cada panel lleva la curva
  real y su control sobre los mismos ejes. (a) la lectura $\delta_{\mathrm{norm}}$ de cada modelo en puntos de color unidos por
  línea continua dentro de cada familia, y la media del nulo emparejado en rombos grises unidos por línea de puntos; el punto va
  lleno cuando la celda es genuina (BH, como en la Tabla 2) y hueco si no. (b) el $z$ intacto en puntos con línea continua, el $z$
  con las orientaciones aleatorizadas en rombos con línea de puntos, discontinua en $-2$; el punto va hueco donde la prueba es
  ciega (potencia decoupled por debajo de 0.8).
- Mismo eje x en los dos: los 12 backbones por familia con un hueco de una posición, separadores blancos, tallas a 6.3 pt sobre su
  marca y la familia debajo. A dos paneles las etiquetas caben en horizontal y se leen a 100 dpi (comprobado en qa_pages).
- Una sola leyenda debajo, dos columnas, caja blanca redondeada con sombra: "model", "random cloud (a) / orientations randomized
  (b)", "hollow: not genuine (a) / test blind (b)", "certified below". Pie: el tuyo, verbatim.
- **El exceso y la potencia salen de la figura** y se quedan en sus tablas, como pediste: el exceso en la Tabla 5 y la potencia en
  la Tabla 25; el hueco entre punto y rombo del panel (a) es el exceso, y el hueco del punto en (b) marca la ceguera.
- Referencias: §3.3 cita ahora el panel (a) en la frase de "Size and evidence" (partida en dos por el tope de 30 palabras) y §5.3
  apunta al panel (b), igual que el pie de la tabla de la prueba en el apéndice.
- Envío: 32 páginas, texto principal en la página 9, sweep 230/230, qa_pages regeneradas.
## 90. Figura 2: curva del modelo continua, control punteado (2026-09-24, nota del autor).

- Los estilos ya eran continuo para el modelo y punteado para el control, pero el modelo no se leía continuo: el borde blanco de
  los marcadores (0.8 pt) cortaba la línea en cada punto y parecía a trazos, también en la clave de la leyenda. La línea del modelo
  pasa de 1.0 a 1.5 pt y el borde blanco de los marcadores baja a 0.5 pt, así que la curva se lee de un trazo; el control usa ahora
  un punteado explícito (1 pt de punto, 1.6 de hueco) a 1.0 pt. La leyenda muestra las dos claves con esos mismos estilos.
- La figura tiene dos paneles, (a) y (b): el "(c)" de tu nota era de la versión de cuatro paneles. Los dos llevan el cambio.
- Envío: 32 páginas, texto principal en la página 9, sweep 230/230, qa_pages regeneradas.
## 91. Figura 2 en 2 × 2 (2026-09-24, brief del autor). Sin cambio de datos.

- Cuatro paneles en dos filas, ancho del texto, 3.2 in de alto; el color significa familia en los cuatro. (a) lectura cruda de
  ImageNet (puntos de color, línea continua) y media del nulo emparejado (rombos grises, punteada). (b) el exceso en barras, llena
  cuando la celda es genuina (BH) y rayada si no, con una marca horizontal oscura en $-2$ desviaciones del nulo junto a cada barra.
  (c) prueba de jerarquía: la nube real intacta (puntos, continua), la real con las orientaciones aleatorizadas (rombos grises,
  punteada) y la jerarquía plantada de tres niveles aleatorizada igual (triángulos amarillos #FDB813, continua, `dec_z_mean` de
  expR81), discontinua en $-2$. (d) potencia para la jerarquía plantada: dos barras por backbone, la intacta en gris claro y la
  aleatorizada en el color de la familia (rayada por debajo de 0.8), discontinua en 0.8.
- Mismo eje x en los cuatro, con las tallas y el nombre de familia también en la fila de arriba, como pediste. Una sola leyenda
  debajo, tres columnas, caja blanca redondeada con sombra, sin nada recortado, con tus seis entradas. Pie: el tuyo, verbatim.
- Referencias: §3.3 sigue citando el panel (a) en "Size and evidence" y §5.3 apunta ahora al panel (c), igual que el pie de la tabla
  de la prueba en el apéndice. La Figura 3 no se toca.
- **No hizo falta recortar §6**: con la figura de 3.2 in el texto principal sigue acabando en la página 9. Envío en 32 páginas,
  sweep 230/230, qa_pages regeneradas y revisadas a 100 dpi.
## 92. Figura 2, dos arreglos (2026-09-24, brief del autor). Sin cambio de datos.

- **(1)** El panel (a) lleva una tercera serie: la referencia gaussiana isótropa a la dimensión de cada backbone (sólo dimensión,
  sin espectro), en cuadrados abiertos gris claro sobre línea discontinua fina dentro de cada grupo, con los valores de
  `exp1_delta_controls.csv` (variante gauss), los mismos de la Figura 2 antigua y de la Tabla 4. La leyenda gana "random ball, same
  dimension (a)" y el pie dice tu frase: el hueco bola-nube es la parte del espectro y el hueco nube-modelo es el exceso. El título
  del panel sigue siendo el que fijaste ("raw reading and its random cloud"); dime si lo quieres también con la bola.
- **(2)** El eje x de los cuatro paneles separa el grupo auto-supervisado en dos: "DINO" bajo una sola marca "B" y "DINOv2" bajo
  "S B L G", con un hueco menor que el de familia y un separador fino entre ambos. "sup." y "contr." no cambian. Los dos subgrupos
  conservan el color de la familia, y la línea que une modelos se traza dentro de cada subgrupo.
- Envío: 32 páginas, texto principal en la página 9, sweep 230/230, qa_pages regeneradas y revisadas a 100 dpi.
## 93. Cumplimiento de la plantilla y vuelta a la página 9 (2026-09-24, brief del autor). Sin cambio de datos.

- **(A) Fuera todos los ajustes de espaciado que se apartaban del estilo ICLR 2027**: el bloque de `\abovedisplayskip` /
  `\belowdisplayskip` (antigua línea 51), `\textfloatsep`, `\abovecaptionskip` y `\parskip` (antigua línea 56) y `\floatsep`
  (antigua línea 380, las fracciones de colocación y los contadores se quedan). Rige ya el `\parskip .5pc` y el `\parindent 0pt`
  de la plantilla. Añado uno más por el mismo motivo: el texto principal ya no declara `\raggedbottom`, así que vale el
  `\flushbottom` del estilo; el apéndice lo conserva, donde las tablas `[H]` dejarían páginas a medias.
- Coste: unas 50 líneas, es decir una página entera. El texto principal acababa en la 10 con §6 y §7 completas allí.
- **(B) Recuperación de la página 9 en tu orden, moviendo al apéndice, y hasta que ha cabido:**
  1. **§7 limitaciones**: quedan cuatro en el texto principal, (i) el nulo condiciona a los segundos momentos, (ii) la potencia
     sigue al backbone y el veredicto de "ninguna jerarquía entre hubs" vale sólo en los 9 de 12, (iii) el análisis descansa en
     muchas decisiones y falta un análisis pre-especificado, (iv) el alcance del censo y de la lectura por imágenes. La lista
     entera, (i)–(x) verbatim, va a la nueva sección **Apéndice B "Limitations in full"** con puntero desde §7.
  2. **§6** pasa a un solo párrafo de seis frases: la regla de Khrulkov, las curvaturas de 0.48 a 2.5, las dos lecturas que
     predicen igual, "Calibration tells what a low δ means", la ganancia de $+0.9$ a $+1.3$ pp y la frase de alcance al final.
  3. **§5.4**: acuerdo con WordNet, DBpedia y HierarCaps quedan en una frase cada uno dentro de un solo párrafo con los tres
     punteros; el párrafo de texto queda en dos frases ("la receta y la escala, no la familia").
  4. **§5.3**: marcos, etiquetas de hoja y MERU quedan en una frase cada uno con su puntero a la Tabla 17, 19 y 20.
  5. **§5.2**: el párrafo del conjunto de clases y el del colapso neuronal van al apéndice.
- Los ocho párrafos movidos están **verbatim** en la nueva sección **Apéndice C "Detail from the main text"**, en el orden en que
  el texto principal los leía, cada uno con su comentario de procedencia: FMNIST y el control de número de clases, el colapso
  neuronal, los marcos (BAL_SENT entera), las etiquetas de hoja, MERU con el 95 por ciento dentro de 0.29 y la potencia no medida,
  el censo de texto modelo a modelo, WordNet/DBpedia/HierarCaps y la frase de Moreira.
- **(C) Cuatro líneas más allá de tu lista**: con los cinco pasos el texto acababa cuatro líneas dentro de la página 10. Las he
  sacado sin perder ninguna afirmación, porque las tres ya estaban dichas en el texto principal: las cuatro limitaciones quedan a
  una frase cada una (la potencia por backbone y las falsas alarmas 8 de 50 siguen en §5.3), el párrafo de texto baja a dos frases
  y en §6 cae "Cosine collects the self-supervised structure", que es la última frase de la conclusión.
- **(D) Lo que sigue apartándose del estilo**, todo tipográfico y reversible en `phaseE_submission.py`: los saltos de encabezado
  `\@startsection` de sección, subsección y párrafo (0.3/0.3/0 ex antes y 0.2/0.2/−1 em después, frente a 2.0/1.8/1.5 ex y
  1.5/0.8/−1 em del estilo); `\topsep`, `\partopsep` y `\parskip` a cero **dentro de las cajas** de definición y proposición; tu
  `\vspace{2pt}` entre la Figura 1 y su pie; y, tras `\appendix`, `\raggedbottom`, las fracciones y contadores de colocación de
  flotantes y `\arraystretch 0.92`. No hay ningún `\linespread` ni ningún `\vspace{-…}` en todo el fichero.
- Todas las afirmaciones del resumen siguen apoyadas en el texto principal (MERU con "a near-flat space", el texto con "la receta
  y la escala", 44 de 72, 9 de 12, la estrella de clústeres) y ninguna figura cambia de tamaño.
- Envío: **34 páginas**, texto principal hasta la página 9, declaraciones en la 10, referencias en la 10, apéndice de la 16 a la
  34. Sweep **231/231** con la comprobación nueva de cumplimiento (preámbulo limpio, cuatro limitaciones, lista de diez en el
  apéndice, ocho párrafos movidos, §6 en un párrafo). qa_pages regeneradas y revisadas.
## 94. Texto principal sin tablas y figura nueva en §5.1 (2026-09-24, brief del autor). Sin cambio de datos.

- **(1) Figura 2**, panel (a): el título pasa a "(a) raw reading, random ball and random cloud".
- **(2) Las dos tablas del texto principal se van al apéndice**, cada una junto a la pregunta que contesta: la Tabla 2 (censo,
  `tab_census_final`) encabeza ahora "Is the structure beyond the second moments genuine?" y la Tabla 1 (lectura publicada,
  `tab_khrulkov_final`) va con la pregunta del nivel de imagen. La Tabla 1 conserva **todas** sus columnas, incluida la del
  supremo, y con `tabcolsep` 2.0 pt ya no se sale del ancho del texto. El texto principal no tiene ninguna tabla.
- **(3) Figura 3 nueva, "The premise where it is read"**, ancho de texto, 1.9 in, estilo de la Figura 2, en §5.1:
  - (a) las 24 celdas de nivel de imagen, 12 backbones × CIFAR-100 y DTD, con el eje x por familias de la Figura 2. CIFAR-100 a la
    izquierda de cada backbone (círculos) y DTD a la derecha (cuadrados), llenos cuando la celda es genuina bajo BH, con la nube
    aleatoria emparejada en rombos grises y línea punteada. Son 10 de 24 genuinas, contadas del fichero.
  - (b) los cuatro conjuntos publicados: estrella gris = su $\delta_{\text{rel}}$ publicado; punto de color = nuestra
    reproducción con su estimador, con el valor impreso al lado (0.28, 0.26, 0.27, 0.20); rombo gris con barra de $\pm2$
    desviaciones = la nube aleatoria. El punto va hueco cuando la lectura es indistinguible de la nube: CIFAR-10 y CUB-200, que
    son los "two of four" del pie.
  - Pie: el tuyo, verbatim. Leyenda compartida debajo, tres columnas, como en la Figura 2.
- **Una decisión que te señalo**: en (b) la nube aleatoria está leída **con su estimador** (el supremo, expR85), no con el
  percentil, para que la estrella, el punto y el rombo estén en la misma escala y "two of four" se vea. Lo que decide el relleno
  (hueco = indistinguible) sí es la calibración del percentil, y coincide con la mediana del supremo (§5.1 ya lo dice). Si
  prefieres el rombo bajo el percentil, es una línea.
- **(4) Referencias**: §5.1 primer párrafo cierra con "Figure 3a shows the 24 cells and Table 16 gives the reading per cell" y el
  segundo con "Figure 3b shows the 4 datasets and Table 15 gives the rows"; el espejo de §1 añade "(Figure~\ref{fig:premise})"
  tras "2 of them are indistinguishable from a random cloud". Las figuras se renumeran solas: la del exceso es la 4 y el mapa de
  árboles la 5.
- **Página**: el texto principal sigue acabando en la página 9 y, al irse las dos tablas, las declaraciones vuelven a la 9 y las
  referencias a la 10. No ha hecho falta mover más contenido. Envío de 35 páginas (el apéndice crece una con las dos tablas),
  sweep **232/232** con la comprobación nueva (sin tablas en el texto principal, las dos en el apéndice, la figura y sus valores
  desde expR62, expR78 y expR85), qa_pages regeneradas y revisadas.
## 95. Repaso completo: doce arreglos de texto y maquetación (2026-09-24, brief del autor). Sin cambio de números.

- **(1)** Resumen: "…excess over many such clouds, and **add** a hierarchy test whose…". OpenReview actualizado (323 palabras).
- **(2)** "planted" en todo el paper: texto, pies, tablas, figuras y literales del sweep. Cambian 108 apariciones de
  "implanted"/"implant" en los seis ficheros fuente. No cambian los nombres de fichero (`expR80_implanted_alignment.csv`,
  `expR64_implanted_depth.csv`), las etiquetas (`fig:implant`, `tab:r1c-implant`) ni las claves de los JSON. El sustantivo suelto
  pasa a "planted two-level tree" y "planted-tree strength/seeds"; el título de la subsección del apéndice dice ya "planted".
- **(3)** §5.1: "The estimator of \citet{Khrulkov_2020_CVPR}, run on our extraction, reproduces the raw δ_rel they report…".
- **(4)** §5.3, "What the test can see": la definición del nivel de ruido abre el párrafo y el resto se parte en dos. El primero
  queda con el implante de dos niveles que se pierde, la jerarquía de tres niveles que se detecta y sobrevive a la
  aleatorización, la potencia por backbone (9 de 12 y 3 de 12) y la frase de que la potencia sigue al backbone. El segundo es
  nuevo, **"A trained hierarchy survives, and the decoupled test leans toward firing."**, con el control entrenado de 2 semillas
  (20 de 20, 6 de 20, ninguna, por grados), el control plano (8 de 50), la jerarquía sintética que queda lejos del sesgo y el
  marco balanceado (0 de 50), y cierra con los punteros.
- **(5)** §5.4: "They agree with the 7 supervised and contrastive models far less than those 7 agree among themselves (DINO-B is
  not part of the island)."
- **(6)** Fuera el párrafo "Models share neighborhoods, not metrics."; el párrafo del mapa de árboles cierra ahora con
  "Calibrated local agreement across models survives, as \citet{groger2026aristotelian} find for similarity (Table~\ref{…})".
  **Ojo**: esa tabla se imprime como **Tabla 31**, no 13; el 13 es el identificador interno (`tab:q13-local`), no su número.
- **(7)** §4: "The class sets are ImageNet (1000 classes) and CIFAR-100 (100), which carry a real hierarchy; DTD (47) and
  CIFAR-10 (10); and the flat FashionMNIST and MNIST (10 each)", con sus citas en su sitio.
- **(8)** Las cajas de definición y proposición son `unbreakable`: ninguna se parte entre páginas. La Proposición 1 sube entera al
  principio de la página 4 y la página 3 cierra con la Definición 2 completa.
- **(9)** Ética: fuera "and a metric-selection heuristic".
- **(10)** Tras la ecuación de profundidad: "…$z=\mathrm{depth}/\mathrm{s.d.}$, **where** the spread is taken over star seeds and
  null replicates, and a cloud is certified…".
- **(11)** Figura 3(b): se queda en la escala del supremo y el pie añade "on their statistic, the supremum; filled and hollow dots
  follow the reading, which gives the same verdict".
- **(12)** No he devuelto nada al texto principal: el hueco de la página 9 lo absorbe la partición del punto (4).
- Sweep **233/233** con una comprobación nueva del repaso y tres reglas afinadas: una ecuación en display cierra frase (si no, la
  frase del punto 10 contaba 44 palabras), "fires in N of M" y "N of M decoupled/intact runs" cuentan como recuentos, y se
  exceptúan tu paréntesis de DINO-B y los dos puntos y coma de la frase de §4. Envío de 35 páginas, texto principal hasta la
  página 9, declaraciones en la 9, referencias en la 10. Fichero de rebuttal reconstruido (61/61) y qa_pages revisadas.
## 96. FashionMNIST se presenta una vez y luego es FMNIST (2026-09-24, nota del autor).

- §4, primera mención: "and the flat **FashionMNIST (FMNIST)** \citep{fashion} and MNIST \citep{mnist} (10 each)". El resto del
  paper ya decía FMNIST: el texto (el párrafo del conjunto de clases, en el apéndice), las 14 apariciones de las tablas y los
  ejes de las figuras. El literal del sweep sigue la frase nueva.
- Sin cambio de números ni de maquetación: 35 páginas, texto principal hasta la página 9, declaraciones en la 9, referencias en
  la 10. Sweep 233/233, fichero de rebuttal reconstruido (61/61), qa_pages y FINAL_CHECK regenerados.
## 97. Cuatro frases completas en el texto principal (2026-09-24, brief del autor). Números verificados contra sus tablas.

- **(1)** §5.1: "the calibrated reading is not genuine in **14 of 24** cells". El 14 sale de `expR62_samplelevel_record.csv`
  (24 celdas, 10 genuinas) por relleno nuevo `SL_NOTGEN`, con aserto en el constructor y comprobación en el sweep.
- **(2)** §5.3, MERU: la frase dice ya que no se detecta estructura de hubs **aunque la potencia a su espectro no se midió**, y
  que el **95 por ciento** de sus embeddings queda a menos de **0.29** de la escala de curvatura, "a nearly flat space". Los dos
  números son los de `expR71_meru_radii.csv`, los mismos que el apéndice.
- **(3)** §5.4, WordNet: la correlación va ahora con sus rangos, leídos de `exp3_alignment.csv`: VLM contrastivos
  $+0.57$ a $+0.59$, ViT supervisados $+0.49$ a $+0.53$, DINOv2 $+0.18$ a $+0.22$ (DINO-B, $+0.36$, no es DINOv2 y queda fuera
  del rango, como pedías). Cierra con "Among leaf-supervised ViTs it depends on the recipe (Section~\ref{sec:f-depth})".
- **(4)** §5.4, texto: "GPT-2 L and XL are genuine under every reading while S and M change with it. Pythia is genuine at every
  size, and the sentence embedders are not on class names but are on DBpedia." Comprobado en las tres lecturas
  (`expR53_text_haar_p999_200.csv`, `expR53_text_haar_sup_200.csv`, `expR57_text_cosine_haar_p999_200.csv`): L y XL genuinos en
  las tres; S genuino sólo bajo la lectura del percentil y M sólo bajo coseno; Pythia genuino en los tres tamaños y en las tres
  lecturas; ninguno de los 7 embedders es genuino en los nombres de clase, y los 3 leídos en DBpedia sí lo son.
- **Dos frases tuyas las he partido**, sin tocar ni una palabra, porque pasaban de tus 30 palabras: la de MERU (52 palabras y dos
  puntos y coma) va en tres frases, y la de WordNet (40) en dos, cortando por los puntos y coma. Si las quieres enteras, es una
  línea. La frase de los tres rangos lleva excepción nombrada en el sweep para tu tope de dos números por frase.
- Sweep **234/234** con la comprobación nueva (cada número de las cuatro frases contra su fichero). Envío de 35 páginas, texto
  principal hasta la página 9, declaraciones en la 9, referencias en la 10. Rebuttal reconstruido (61/61) y qa_pages revisadas.
## 98. Tres limitaciones más en §7 (2026-09-24, brief del autor). Sin cambio de números.

- El hueco de la página 9 se llena con tus tres, una línea cada una, en tu orden: **(iii)** "The hierarchy test is Euclidean, and
  for the angular DINOv2 tree it may be conservative"; **(iv)** "The text census depends on the probe: template and batching move
  the reading, so verdicts at the margin are fragile"; **(vii)** "The objective--geometry link is correlational, resting chiefly
  on DINOv2's scale range and on 2 leaf-label ViTs". Las tres caben: el texto principal acaba justo al pie de la página 9 y la
  página 10 abre con la declaración de reproducibilidad.
- La (iii) y la (vii) son las del apéndice, verbatim. La (iv) la he escrito yo con lo que dice la tabla de texto: el panel de
  plantillas mide el rango de $\hat\delta$ por plantilla (causales frente a embedders) y el de extracción mide que el batching
  con relleno a la izquierda mueve los estados de GPT-2; de ahí "template and batching move the reading" y "verdicts at the
  margin", que es el caso de GPT-2 S. Procedencia: `expR47_extraction_variance.csv`, `expR49_template_nulls_bs1.csv`.
- Las siete van renumeradas (i)-(vii) y el puntero a la lista completa del apéndice sigue al final. La lista del apéndice no
  cambia: sigue con sus diez.
- Sweep **234/234** (la comprobación de cumplimiento verifica ya las siete y las tres frases nuevas). Envío de 35 páginas, texto
  principal hasta la página 9, declaraciones y referencias en la 10. Rebuttal reconstruido (61/61) y qa_pages regeneradas.
## 99. Párrafo de MERU en §5.3: entrada y cierre (2026-09-24, nota del autor). Sin cambio de números.

- Entradilla: "MERU's objective does not create hub structure." → **"Training in hyperbolic space leaves the clustering
  unchanged."**
- Última frase: "And 95 per cent of its embeddings stay within 0.29 of the curvature scale, a nearly flat space (Tabla 20)." →
  **"Its embeddings also stay nearly flat: 95 per cent lie within 0.29 of the curvature scale (Tabla 20)."** Los dos números
  siguen siendo los rellenos de `expR71_meru_radii.csv`, y la tabla es la del gemelo euclídeo, la 20, como escribiste.
- La entradilla nueva viaja a los cinco sitios del sweep que la fijaban (orden de las entradillas de §5.3, exención de números del
  párrafo, comprobaciones de la novena y la duodécima revisión y la del repaso completo).
- Sweep 234/234, envío de 35 páginas con el texto principal hasta la página 9, rebuttal reconstruido (61/61), qa_pages regeneradas.
## 100. Reducción del apéndice (2026-09-24, brief del autor). Texto principal intacto salvo tres punteros.

- **(1) Fuera las tablas que el texto principal no cita.** Las quince que listabas son exactamente las no citadas. Diez se han
  ido: q1-census (el censo por celda bajo la lectura del papel), q1-census-d, q2-text-c, q5-power (el árbol de dos niveles
  plantado, que ya es la Figura 9), q6-treemap (las seis configuraciones), q7-wordnet-b (recuperación de superclases),
  q8-robust-c (ruido conjunto) y q9-corollary-c, más las dos que pasan a figura.
- **Cinco quedan porque el sweep dice que un número del texto principal se apoya en ellas** (te las reporto): **q4-depth-e** (el
  control radial: "$z$ de $-2.26$ a $-3.99$ … feature norms are not its source" en §5.3), **q5-power-c** (la alineación plantada
  por eje principal: "reproduces it in the supervised ViTs and CLIP-B but not in DINOv2-L"), **q5-power-f** (el marco balanceado:
  "fires in 0 of 50 decoupled runs"), y **q9-corollary-b** y **q9-corollary-d** (las correlaciones cruda y calibrada, en las que
  se apoya "The raw and calibrated readings predict the gain … equally well" de §6).
- **(2) Cinco tablas de tendencia son ahora figuras**, con la paleta, el fondo gris, la rejilla blanca y la leyenda con sombra de
  la Figura 2, y sus valores leídos del mismo fichero que usaba la tabla: **Figura 7** presupuesto de cuádruples (exceso contra
  presupuesto, una línea por celda, 9 celdas), **Figura 8** número de clases (exceso contra tamaño del subconjunto, al azar y
  como hermanos de WordNet), **Figura 10** potencia por backbone contra el nivel de ruido de la nube (intacta hueca, desacoplada
  llena, discontinua en 0.8), **Figura 11** techo intramodelo (banda del peor par bootstrap a la media, con el acuerdo cruzado
  como punto) y **Figura 12** ganancias del corolario (barras de la mejor ventaja sin coste por backbone, con la métrica que la
  recoge). Siguen como tablas el censo por celda, la reproducción de la lectura publicada, el panel de modelos, el censo de texto
  y el índice de procedencia.
- **Tres punteros del texto principal cambian de palabra**, que es lo que obliga la conversión: §3.2 "(Table~\ref{tab:q8-robust})"
  → "(Figure~\ref{fig:budget})"; §5.3 "Figure 9 and Table …" → "Figures 9 and 10 give the detection rates and the power"; §5.4
  "(Tables 20 and 21)" → "(Table 20 and Figure 11)". Ni una palabra más del texto principal se toca.
- **(3) La sección B** pasa a llamarse "Limitations beyond Section 7" y guarda, con una frase de entrada, sólo las cuatro que §7
  no dice: el exceso certifica estructura más allá de los segundos momentos, el bloque largo de la prueba de jerarquía (falsas
  alarmas, sesgo del desacoplado, control entrenado, criterio previo, alineación no radial), la normalización por el diámetro y
  el alcance downstream. El puntero de §7 dice ahora "gives the rest".
- **(4) La sección C** guarda los cuatro párrafos que el texto principal aún necesita (conjunto de clases con FMNIST, colapso
  neuronal, el conjunto certificado bajo otros marcos con la frase del marco balanceado, y la frase de Moreira) y suelta los
  cuatro que el texto principal ya dice (etiquetas de hoja, MERU, censo de texto modelo a modelo, WordNet/DBpedia).
- **(5) La guía de lectura (D.1)** lista las trece preguntas en el orden en que el texto principal las cita, con sus tablas y sus
  figuras; dos títulos de subsección se ajustan a lo que queda dentro.
- **Páginas: el envío pasa de 35 a 32.** El apéndice (pruebas incluidas, de la 16 a la 32) pasa de **20 a 17 páginas**. El texto
  principal sigue acabando en la página 9. Sweep **235/235** con la comprobación nueva de la reducción. Rebuttal reconstruido
  (61/61) y qa_pages regeneradas.
## 101. Reducción final del apéndice (2026-09-24, brief del autor). 29 páginas, apéndice de 14.

- **(1) El apéndice B desaparece.** §7 gana tu **(viii)** "The reading divides by the diameter, so heavier tails in real clouds
  would lower $\delta_{\text{norm}}$ without any clustering; the cosine census mitigates this", y el puntero al apéndice se va.
  Las otras tres ya estaban dichas: el exceso certifica estructura más allá de los segundos momentos (§3.3 y §3.4), el bloque
  largo de la prueba de jerarquía (§5.3) y el alcance downstream (§6). **Una frase la he salvado**: "A pre-set criterion
  expecting no intact certification of the leaf and frozen models was not met, because they are aligned" no estaba en ningún
  otro sitio y es una frase de honestidad sobre un criterio previo no cumplido; va ahora en el párrafo de la pregunta de la
  prueba de jerarquía. Dime si prefieres que desaparezca.
- **(2) Fuera la guía de lectura y el glosario.** Cada símbolo que una tabla usa se define ahora en su propio pie: se han añadido
  las definiciones de $^{\circ}$, $r$/200, w/b, $s^{*}$ y $\bar z_1$ a los pies que los usaban sin definirlos. El tope de 40
  palabras sube a 50 sólo para esos pies.
- **(3) Fuera la tabla de procedencia.** Su contenido es ahora `ICLR2027/supplementary_code/README.md`, que genera
  `rebuttal/scripts/make_supp_readme.py` desde el paper construido (13 ficheros de tablas con sus 24 etiquetas, 11 figuras y los
  49 ficheros de resultados que cita el texto principal) y que la cadena regenera en cada compilación. La declaración de
  reproducibilidad dice ya tu frase.
- **(4)** Fuera la figura del número de clases. **(5)** Fuera la tabla de alineación plantada y, con ella, la frase de §5.3; queda
  "Its geometric form is not resolved here." **(6)** La tabla de correlaciones calibradas **sí** lleva la columna de la lectura
  cruda ($\hat\delta_{99.9}$), así que la de correlaciones crudas se va, como pedías.
- **(7)** Un solo censo por celda: la tabla que estaba en el texto principal se funde con la del apéndice y **Barlow-R50 y
  BYOL-R50 entran como filas** (leídas sólo en ImageNet, contra 20 réplicas, dicho en el pie). Esa tabla responde también a la
  referencia de la antigua Tabla 2. Los recuentos del marco balanceado son ahora **dos filas de la tabla de la estrella con hubs
  Haar** ("flat control, balanced frame" y "flat control, frame mismatch").
- **(8)** El apéndice C se disuelve: sus cuatro párrafos van a la pregunta que les toca (censo, censo, prueba de jerarquía,
  corolario). **(9)** Orden final: **A Proofs; B Implementation** (panel de modelos y extracción); **C Additional results**, una
  subsección por pregunta.
- **Páginas: 29** (eran 32). El apéndice, pruebas incluidas, ocupa de la 16 a la 29: **14 páginas** (eran 17). El texto principal
  sigue acabando en la página 9 y las declaraciones vuelven a la 9. Sweep **236/236**, rebuttal reconstruido (61/61), qa_pages
  regeneradas y README de suplementario escrito.
## 102. Las dos tablas sin citar quedan citadas (2026-09-24, nota del autor). Sin cambio de números.

- §5.3, en la frase del control radial: "…feature norms are not its source **(Table~\ref{tab:q4-depth-e})**", que se imprime
  como **Tabla 16**, junto a los $z$ de $-2.26$ a $-3.99$ que salen de ella.
- §6, en la frase de las dos lecturas: "…and the hierarchy verdict predicts none **(Table~\ref{tab:q9-corollary-d})**", que se
  imprime como **Tabla 24**, la de las correlaciones calibradas con la columna de la lectura cruda.
- Con eso **ninguna tabla del apéndice queda sin citar desde el texto principal**: la comprobación de la reducción pasa a exigir
  la lista vacía. El puntero del párrafo de la alineación va en la frase cuyos números respalda, no en la última, como pediste;
  queda anotado como excepción en el sweep.
- Envío de 29 páginas, texto principal hasta la página 9, declaraciones en la 9 y referencias en la 10. Sweep **236/236**,
  rebuttal reconstruido (61/61) y qa_pages regeneradas.
## 103. La declaración de reproducibilidad apunta al suplementario (2026-09-24, nota del autor).

- "The full pipeline with fixed seeds **is provided in the supplementary material; its `tool/` directory** ships the instrument
  as one script that reproduces any cell de la Tabla 2 desde su matriz de centroides." Fuera el "will be released in the
  anonymized repository (TODO(author))".
- **Cero apariciones** de "TODO", "anonymized repository" y "ANONYMIZED" en el `.tex` del envío, en los ficheros que incluye
  (tablas del apéndice), en el PDF, y también en el fichero y el PDF del rebuttal. El sweep lo comprueba ahora en cada pasada.
- Envío de 29 páginas, texto principal hasta la página 9. Sweep **236/236**, rebuttal reconstruido (61/61), qa_pages regeneradas.
