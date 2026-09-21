> **Frozen 17 Sept 2026; only typographical changes after this date.**

# CHANGELOG — pasada de versión final (2026-09-03)

Brief: "Revision brief — Is There a Platonic Tree? — final-version pass". Todo número del paper sale de
un fichero de `rebuttal/results/`; ningún número se ha teclado a mano salvo las filas árbol/H²/esfera de
la tabla de calibración (sin CSV; ver `TODO_author.md`). Los seis protocolos vigentes de `CODE_MAP.md`
no se han tocado.

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
