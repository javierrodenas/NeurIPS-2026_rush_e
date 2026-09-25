# Revisión del borrador ICLR 2027 (`ICLR2027/iclr2027/main_iclr2027.tex`) — 31 ago 2026

Compilado con tectonic sobre una copia (la máquina no tiene TeX): **compila sin errores, 21 páginas**.
Cifras del texto principal contrastadas una a una con `rebuttal/results/*.csv` (lista al final): **todas cuadran**.
Lo que sigue está ordenado por gravedad.

## A. Bloqueantes (no se puede enviar así)

1. **Se pasa del límite de 9 páginas.** La Discusión ("Excess measures existence…"), las Limitations y la
   Figura 7 caen en la página 10 (~0.6 página de exceso). Ethics/Reproducibility/LLM no cuentan, pero la
   Discusión sí.
2. **Abstract de 409 palabras** (`main_iclr2027.tex:47`): ocupa el 60 % de la página 1 y es exactamente lo
   que 1Eqj penalizó ("many numbers within the text, in parentheses, fragmented"). Objetivo: ≤ 220 palabras,
   tres hallazgos, como mucho 3-4 cifras.
3. **No hay ninguna tabla de resultados en el texto principal** (solo la de calibración, Tabla 1). El rebuttal
   a 1Eqj (A7) prometía "a main-text results table"; el borrador tiene 7 figuras y todos los números en
   paréntesis dentro de párrafos de 200-400 palabras. Propuesta: una tabla compacta por familia con
   exceso en ImageNet (z), exceso en transfer, ρ WordNet, ganancia NC/FS de la mejor métrica y métrica
   ganadora. Cabe en 8 filas y sustituye a la mitad de los paréntesis de §4-§6.
4. **Figura 1 sigue con el artwork viejo**: dentro del PDF pone "How is the reality?", "Hyperbolic" /
   "Euclidean" y nodo raíz "Reality"; el caption (`:73-76`) habla de "tree-like (left) … or flat (right)" y
   de "Is there a Platonic tree?". Contradice título y reescritura (COAUTHOR_TASKS #1). O se re-etiqueta o
   se quita; quitarla resuelve además el problema de páginas.
5. **Referencias `??` en el índice de procedencia** (p. 20): `tab_z_provenance.tex:10,17` referencian
   `tab:a1` y `tab:a8`, pero `tab_a1_excess` y `tab_a8_hierarcaps` no se incluyen con `\input`.
6. **HierarCaps no aparece en el paper.** La tabla está generada (`appendix_tables/tab_a8_hierarcaps.tex`,
   exp16) pero no se incluye ni se cita en el texto. Fue compromiso público con pux6 ("we commit to such an
   evaluation in the revision") y pux6 dijo que con ello el paper "reads as substantially more complete". Si
   los mismos reviewers reaparecen (habitual entre NeurIPS e ICLR), lo notarán. El resultado es además
   bueno: orden radial ρ(level, radius) +0.47..+0.87, monotonía estricta 19-54 % frente a 4.2 % de azar,
   tripletas 0.83-0.90. Añadir 2-3 frases en "Whose tree?" (`:190`) y el `\input` en el apéndice.
7. **Dos tablas del apéndice se salen del margen**: `tab_a0_models` (+10.7 pt) y `tab_b7_treemap`
   (+44 pt, claramente visible). `\scriptsize` + `\setlength{\tabcolsep}{3pt}` o `\resizebox`.

## B. Contenido y framing (decisiones de autor)

8. **§6 (Corollary) se desmonta a sí misma.** La contribución 4 (`:66`) promete "when to leave Euclidean
   distances (predicted by the descriptors) … and which zero-cost metric collects the gain", pero el propio
   texto dice que "the δ-gated rule … is the weakest policy in the table" en NC y FS, y que "at the headline
   t the Poincaré projection does not improve NC on average". Es honesto (y correcto: tab_b9 lo confirma),
   pero tal como está redactado un reviewer citará esa frase contra la contribución (pux6 ya lo hizo:
   "the structure exists yet is not useful downstream"). Recomendación: (a) redefinir la contribución 4
   como *diagnóstico* ("δ predicts where the sign of the non-Euclidean gain falls; the operational policy is
   objective + label flatness, with cosine as the safe default"); (b) reordenar "Which metric collects the
   gain" para abrir con lo que funciona (validation pick +0.78 pp, objective rule +0.41 vs always-cosine
   +0.28, p = 0.011) y cerrar con el negativo del δ-gated; (c) quitar "Metric-Selection Rule" del título de
   la sección si se opta por (a).
9. **La figura principal de §5 es el mapa ingenuo** (Fig. 5), que el propio texto declara artefacto; la
   corrección solo se ve en una tabla del apéndice. Mejor un panel (a) naive / (b) configuración
   seleccionada: el lector ve la isla desaparecer en vez de leerlo. El párrafo "A naive map of trees, and
   its artifact" (~420 palabras) lee como log de auditoría (criterio, umbral 0.5, sensibilidad
   [0.40-0.50], seis configuraciones); dejar el criterio en dos frases y mover la sensibilidad al apéndice.
10. **La respuesta a la pregunta del título aparece tres veces casi literal** ("to a substantial degree,
    yes; the shared tree is the human taxonomy; supervision deepens and stabilizes…"): abstract (`:47`),
    final de §5 (`:180`) y Discussion (`:221`). Una vez en Discussion basta; ahorra ~0.1 página.
11. **Panel de modelos incoherente.** §3 (`:95`) dice "4 additional contrastive vision backbones are
    measured on ImageNet" (OpenCLIP/MetaCLIP, Tabla A.1), pero no aparecen en ningún resultado del
    borrador (0 menciones; exp20 tiene 12 modelos). O entran en el censo o se quitan de A.1 y del
    recuento. El `% TODO-coauthors` de esa línea sigue abierto.
12. **Figura 7**: las etiquetas δ = 0.107 / 0.076 dentro del gráfico son del protocolo viejo (exp20
    CIFAR-100: DINOv2-S 0.113, DINOv2-G 0.070) y el nodo "Reality" sigue (COAUTHOR_TASKS #2). Quitar los δ
    del gráfico o regenerar. Es "ilustración" por caption y estamos pasados de página: candidata al apéndice.
13. **Intro, párrafo "The answer has two halves"** (`:58-59`): 230 palabras con 12 cifras. Es el patrón que
    1Eqj criticó. Con la tabla del punto 3, este párrafo puede quedarse en la mitad y sin números.

## C. Figuras (legibilidad)

El ancho de texto de ICLR es 5.5 in (396 pt). Factores de escala reales al incluirlas:
Fig. 2 y 6 (662 pt → ×0.60), Fig. 3 (633 pt a 0.9\linewidth → ×0.56), Fig. 4 (1320 pt → ×0.30),
Fig. 5 (627 pt → ×0.63). Con tipografía de ~7 pt en origen, Fig. 2, 4 y 6 quedan a 4-5 pt en el PDF:
ilegibles impresas (1Eqj: "Fig. 2 is too small … or it is useless"). Regenerar con
`figsize=(5.5, h)` y fuentes ≥ 7 pt (Fig. 4 en dos filas de 2 paneles).

14. Fig. 3: la anotación "genuine form emerges with scale (GPT-2 and OLMo)" tapa las barras de Pythia.
15. Fig. 2b muestra la media sobre datasets de transfer, pero el texto (`:133`) cita CIFAR-10
    (−0.095 → −0.157) "Figure 2b": el lector no encontrará esas cifras. Decir "mean over transfer
    datasets" en el caption o mostrar CIFAR-10.

## D. Menores

- Títulos de §4 y §5 ocupan dos líneas cada uno; acortar.
- `:118`: "5 null replicates" para δ; para ξ el último commit usa 20 réplicas + ruido del estimador. Decir
  ambos en el párrafo "Curvature sign".
- `:127`: el sweep exp19 tiene 5 semillas con dispersión grande (NC adv random 0.5-2.6 pp, coherente
  0.04-1.0 pp); el texto da solo medias (+1.58 / +0.53). Añadir "mean of 5 seeds" o ±.
- Reproducibility: "will be released" → enlace anónimo ya en submission (gratis y los reviewers lo valoran).
- Related Work: "None of these readings is calibrated against matched nulls; … without calibration they
  are not interpretable" es una afirmación fuerte sobre trabajo ajeno; suavizar a "are not calibrated…;
  our results suggest…".
- `\author{Anonymous authors}` es inocuo (la plantilla ya lo anonimiza).

## E. Plan de recorte para entrar en 9 páginas (≈ −0.9 pág, deja sitio para la tabla)

| Acción | Ahorro |
|---|---|
| Abstract 409 → ≤ 220 palabras | −0.30 |
| Fig. 7 al apéndice (o quitar Fig. 1) | −0.30 |
| Respuesta al título solo una vez | −0.10 |
| Sensibilidad del umbral (§5) al apéndice | −0.10 |
| Intro "two halves" a la mitad | −0.10 |
| Tabla resumen nueva en §4 | **+0.25** |

## F. Cifras verificadas contra CSV (todas ✓)

69/72, 51/72, 41/48, 34/48, celdas frontera −3.03/−2.97 (exp20); rangos ImageNet ViT −0.020..−0.030 y
CLIP/SigLIP −0.019..−0.025; DINOv2-L z = −1.1; excepciones ≤ +1.2 s.d.; S→G CIFAR-10 −0.095 → −0.157;
GPT-2 +0.048/−0.013/−0.034/−0.048, OLMo −0.001/−0.015, Pythia −0.035..−0.041, embedders +0.001..+0.011,
GTE-Qwen2 −0.019 (exp18); ξ DINOv2 −0.081 → −0.152, SigLIP +0.005 (z = 2.5), 9/12 (exp26); ρ WordNet
por familia (exp3); pooling 1 img ρ ≤ +0.26 (exp8_p6; nota: al rebuttal se dijo +0.34, de otro run);
DBpedia exceso/FS/NC/tripletas (exp14); exp19 C = 50 −0.124/−0.093 y +1.58/+0.53; McNemar 242/37,
p ≈ 4×10⁻³⁸ (exp13); NC 22/60 positivas, media −0.24, hier −0.00, flat −0.70; FS 56/60
(table1_regenerated); CLIP HN−COS +0.9..+1.3 transfer, +0.1..+0.2 ImageNet (exp2b); exp24 +0.63/+0.64,
+0.78/+0.78, +0.41 vs +0.28; políticas NC (tab_b9); exp21 kNN 0.474/0.010/0.464, CKA 0.636/0.107/0.529,
H 0.427/0.644.

Única discrepancia encontrada: las δ impresas dentro de la Figura 7 (punto 12).

## Cómo compilar en local

No hay TeX en la máquina. Funciona `conda create -p <dir> -c conda-forge tectonic` y luego
`tectonic main_iclr2027.tex` dentro de `ICLR2027/iclr2027/` (descarga paquetes la primera vez).

---

## Estado tras la pasada mecánica (31 ago, tarde)

Hecho en `main_iclr2027.tex` y `appendix_tables/` (compila con tectonic: **20 páginas, texto principal termina en la 9, cero warnings**):

- **A1 páginas**: ✅ Ethics Statement abre la página 10.
- **A2 abstract**: ✅ 409 → 225 palabras; mismos tres hallazgos, cuatro cifras.
- **A5 `??`**: ✅ `tab:a1` eliminado del índice de procedencia; `gen_provenance.py` ahora solo indexa tablas que el `.tex` incluye.
- **A6 HierarCaps**: ✅ frase con cifras (exp16) en "Whose tree?" + cita `alper2024hierarcaps` + nueva subsección A.13 con `tab_a8`.
- **A7 tablas anchas**: ✅ `tab_b7_treemap` (tabcolsep 3 pt, cabeceras cortas, "(deg.)"/"(selected)") y `tab_b2_ztable` (tabcolsep 4 pt); mismos cambios en `gen_appendix2.py` para que una regeneración no los deshaga.
- **B10 respuesta al título**: ✅ eliminada del final de §5; queda en abstract (corta) y Discussion.
- **B12 Fig. 7**: ✅ movida a un Apéndice B "Illustrative visualization" (`app:viz`); Limitations (vi) apunta allí. Las δ viejas dentro del gráfico y el nodo "Reality" siguen pendientes (requieren regenerar la figura).
- **B9 (parcial)**: la sensibilidad del umbral 0.5 pasa al apéndice A.5 (`app:treemap`); el criterio queda en dos frases en §5.
- **D títulos de §4/§5**: ✅ una línea cada uno ("Findings I: Genuine, Learned, Scale-Deepening Form" / "Findings II: Shared Content, Fragile Measurement").
- **B13 intro**: cifras repetidas de GPT-2/OLMo y el 41/48 quitadas del párrafo "two halves" (siguen en §4).
- Limitations (ii, iv, v, vi, vii, viii) y "What the census establishes" apretadas sin cambiar contenido.

Copia de seguridad del `.tex` original: scratchpad de la sesión (`main_iclr2027.tex.bak`); el diff completo está en `git diff`.

**Pendiente (decisiones de autor, no tocado):** A3 tabla resumen en el texto principal (hay ~0 líneas de margen: habrá que pagarla con más recortes en §5 o quitando Fig. 1); A4 Fig. 1 artwork viejo; B8 reformular §6/contribución 4; B9 Fig. 5 naive vs corregida; B11 OpenCLIP/MetaCLIP en el panel; C legibilidad de Figs. 2/4/6; 14-15 Fig. 3 anotación y Fig. 2b caption.

---

## Estado tras la pasada de pendientes (31 ago, noche)

Todo compilado con tectonic: **20 páginas, texto principal termina en la 9, cero warnings**. Commit anterior: `79da833`.

- **A3 tabla resumen en el texto principal**: ✅ nueva Tabla 2 (`tab_census.tex`, generada por `gen_main_table.py` desde exp20/exp26/exp3/exp28/exp2): 12 backbones × {δ̂ IN, exceso IN (z), exceso transfer, ξ-exceso, ρ_WN, ARI₂₀, mejor métrica − R (FS), métrica}. Referenciada desde §4, §5 y §6.
- **A4 Fig. 1 (artwork viejo)**: ✅ retirada del paper (opción B de COAUTHOR_TASKS); el fichero sigue en `figures/` por si se re-etiqueta. La frase de la intro que la citaba se reescribió.
- **B8 §6 / contribución 4**: ✅ reformulado como *diagnóstico + política*: contribución 4, abstract (3), frase de la intro, título de §6 ("A Diagnostic of Exploitability, and a Modest Policy"), párrafo de apertura, y "Which metric collects the gain" reordenado (abre con validation pick / objective rule, cierra con el negativo δ-gated en un párrafo propio "What the diagnostic is, and is not"). Ninguna cifra cambia.
- **B9 Fig. 5 naive vs. corregida**: ✅ nueva `fig_treemap_controls.pdf` (`ICLR2027/figures/make_treemap_fig.py`, desde las matrices ARI por configuración de exp23): (a) ImageNet naive, (b) ImageNet cosine-average, (c) CIFAR-100 naive, (d) CIFAR-100 cosine-complete, con el ARI medio DINOv2-B/L/G vs bloque sobre cada panel (0.03 → 0.38; 0.00 → 0.39). Sustituye a la figura naive; caption nuevo. Además la sensibilidad del umbral ya estaba en A.5.
- **B11 OpenCLIP/MetaCLIP**: ✅ fuera de la Tabla A.1 y de la frase de §3 (no hay resultados calibrados); el TODO-coauthors queda resuelto: censo 12 visión × 6 + 16 texto; malla de tareas 10 visión (sin DINO-B ni SigLIP-B).
- **C legibilidad**: ✅ Figs. 2, 3 y 6 regeneradas a 5.5 in de ancho con fuentes 6.5-8 pt (`make_figs.py`, `fig_text_nulls.py`; rutas ahora relativas al repo, con `PLATONIC_RESULTS` como override). ⚠️ **Fig. 4 (ablaciones) no se puede regenerar aquí**: el script de 4 paneles y los datos de la curva por capas no están en el repo (`figures_archive/gen_fig3_causal.py` es una versión antigua de 3 paneles). Queda para coautores: regenerar a `figsize=(5.5, 1.7)` con fuentes ≥ 7 pt.
- **14 Fig. 3 anotación**: ✅ reubicada (ya no tapa Pythia). **15 Fig. 2b**: ✅ caption dice "mean over the three hierarchical transfer datasets"; el texto cita también la media de transfer (−0.063 → −0.103).
- **12 Fig. 7 δ viejas**: ✅ tapadas con `overpic` (cajas blancas sobre "δ = 0.107 / 0.076"); el nodo "Reality" sigue (necesita el script fuente).
- **Menores**: ✅ "5 null replicates for δ, 20 for ξ"; exp19 "means over 5 draws"; frase de Related Work suavizada.
- Recortes adicionales para volver a 9 páginas tras añadir la tabla: intro "two halves", contribuciones 3-4, related work (convergencia), Discussion, Limitations (i)-(ii), §5 (criterio), §6 (frases sueltas). Sin cambios de contenido ni de cifras.

**Sigue pendiente (requiere a coautores):** Fig. 4 regenerada; Fig. 7 regenerada sin "Reality" (script no localizado); enlace anónimo al código en Reproducibility; decidir si Fig. 1 vuelve re-etiquetada.

---

## Nota de numeración (los números de figura de las secciones anteriores son los del borrador original)

Tras retirar la figura de la caverna y mover la visualización al Apéndice B, la numeración compilada es:
Figure 1 = `fig_excess_panel` · Figure 2 = `fig_text_nulls` · **Figure 3 = `fig3_causal` (ablaciones; la única aún ilegible ×0.30)** · Figure 4 = `fig_treemap_controls` (naive vs. seleccionada) · Figure 5 = `fig_bestmetric_scatter` · **Figure 6 = `fig_tree_visualization` (Apéndice B; nodo "Reality" pendiente, δ viejas ya tapadas)**. La tabla de calibración está ahora en el Apéndice A.1 y la Tabla 1 del texto principal es el censo.

---

## Respuesta a la revisión simulada (31 ago, madrugada) — TODOS los puntos atacados

Cuatro experimentos nuevos (scripts en `rebuttal/scripts/expR29-32*.py`, resultados en `rebuttal/results/expR*.csv`), corridos en esta máquina (los features estaban en `/media/HDD_4TB_2/javi/Platonic`, no en `/home/javi/Platonic` como asumían los scripts viejos; GPU 2080 Ti para GPT-2).

| Punto | Resultado | En el paper |
|---|---|---|
| **W1** null que aísle jerarquía de clustering | **expR29**: null que conserva la geometría de cada cluster y sustituye la configuración de hubs por una muestra gaussiana espectro-igualada. 24/24 celdas (IN wn30 + C100) siguen negativas; 14/24 con z<−2 (ViT-L IN −0.032, z=−5.1). En C100 el componente entre-hubs lleva el 56-94 % del exceso para sup./contrastivos y menos para DINOv2 (semántica angular). DINOv2-L en IN: marginal bajo null espectral → significativo bajo el plano (−0.034, z=−3.0) | Párrafo "Hierarchy, not just clusters" en §4; Tabla B13; abstract y contribución 1 mencionan el null; Fig. — |
| **W2** pseudo-replicación por familia | Demeaned por familia: within-dataset sobrevive (NC −0.31..−0.73; FS −0.63..−0.86; best−R −0.55..−0.82, sostenido por DINOv2); el pooled del mejor-métrica NO (−0.31→−0.18, −0.33→−0.04) | Frases en §6 (dos sitios); Tabla B12 |
| **W3** ¿sobrevive la emergencia GPT-2 a la plantilla? | **expR30**: 10 plantillas × 4 tamaños con nulls. S: 0/10 genuino (7/10 POR ENCIMA del null); M: 3/10; L: 9/10; XL: 9/10 (incl. gloss_only sin nombres de clase). El orden con la escala es robusto a plantilla; los valores no | Frase "Two scopes" reescrita en §4; Limitation (iii) matizada; Tabla B14 |
| **W4** desconexión instrumento↔corolario | z pooled predice ≥ δ (NC −0.50 vs −0.45; FS −0.36 vs −0.18); exceso within-transfer −0.55..−0.84; inversión SOLO en ImageNet (la celda anómala DINOv2) | Frase en §6 y en Discussion ("dissociation localized") |
| **W5** media de excesos incomparables | Tabla 1 con columnas exc. C100/C10/DTD; Fig. 1b con líneas por dataset | hecho |
| **W6** cuádruplas / percentil | **expR31**: exceso estable desde 10⁵ cuádruplas (sd ≤0.004 salvo la celda at-null); p99.9 conserva todos los signos | Frase en §3; Tabla B15 |
| **Q4** bootstrap de centroides | **expR32**: sd del exceso 0.002-0.005 con 30 remuestreos (C100 500 img/clase, DTD 80) — un orden por debajo del exceso | Frase en §3; Tabla B15 |
| Menores | CLIP-L "marginal (z=−1.9)"; GTE-Qwen2 anotado en Fig. 2 + texto ("patterning with the causal-LM scale story"); DINO-B unificado; **§3 corregido**: los centroides de transfer usan el split cacheado completo (500 C100 / 80 DTD / ~6000 MNIST), no "100 por clase" — verificar con coautores que ImageNet sí es 100 | hecho |
| Recomendación "mover §6 al apéndice" | No seguida (Xbn5/pux6 pidieron utilidad downstream); en su lugar: §6 reformulada como diagnóstico + política (commit anterior) + los resultados W2/W4 arriba | — |
| Densidad §5/§6 | Sin resolver del todo (falta de espacio); las tablas B12-B15 absorben parte | parcial |

Para caber en 9 páginas con el material nuevo: la tabla de calibración pasó al Apéndice A.1 (sus números clave quedan inline en §3) y hubo recortes de redacción distribuidos sin tocar cifras. Compila: 22 páginas, texto principal hasta la 9, 0 warnings.

---

## Ronda 2 del revisor simulado (nota 8) — los 6 puntos restantes, resueltos

1. **Abstract sobrevendía el null B** → ahora dice "significant in 14/24 cells, concentrated in CIFAR-100 and the supervised models". (El "5 de 12" del revisor era el −1.95 de ViT-T redondeado en la tabla; el recuento estricto es 4/12 IN + 10/12 C100 = 14/24.)
2. **Null B usa la taxonomía humana** → frase de alcance (i) en la caption de B13: certifica jerarquía *alineada con WordNet/superclases*; no detectaría jerarquía propia del modelo desalineada.
3. **"Estable desde 10⁵" no valía en ImageNet** → cierto (DINOv2-L IN deriva −0.024→−0.005; ViT-L y CLIP-B derivan en dirección contraria). Frase de §3 y caption de B15 reescopadas a transfer sets con el caveat de ImageNet explícito. Y **expR33** (nuevo): a presupuesto 4× (2×10⁶) los veredictos del null B en ImageNet persisten — ViT-L zB=−5.1, CLIP-B −2.4, **DINOv2-L −2.0** (la celda rescatada no es artefacto del presupuesto); citado en la caption de B13. El bootstrap de ImageNet requiere el caché por imagen (no está en esta máquina) → nota en B15 y pendiente coautores.
4. **Discrepancia B13 vs Tabla 3 en ImageNet** → diagnosticada: los `.npy` de `results/centroids/imagenet_train/` difieren del caché del censo **solo para los 4 ViT supervisados** (δ̂ +0.016..+0.025; el resto ±0.003). Caption (ii) de B13 lo declara; dentro de cada fila todo usa los mismos centroides. Pendiente coautores: regenerar la mitad ImageNet de B13 desde el caché canónico.
5. **Fig. 5 desincronizada** → títulos ahora "within-ds r −0.60..−0.83 (pooled −0.31)" (calculado en `make_figs.py`) y caption remite a B12.
6. **Limitación (iv)** → añade "the within-dataset δ–gain prediction rests chiefly on DINOv2's scale range".

Compila: 22 páginas, texto principal hasta la 9, 0 warnings, sin `??`.

---

## Ronda 3 (nota 8, aceptar — sostenida): tres menores de redacción, hechos

1. Caption B13: "Two scopes" → "Three scope notes" (eran tres).
2. Caption B15: fuera la frase de bitácora ("not available on this machine") → "reported on CIFAR-100 and DTD".
3. Trazabilidad de A.4: comentario de fuente en el .tex (expR29/expR33 + generador + almacén de centroides), ya que el índice de procedencia renderizado se retiró a petición.

Oferta del revisor para la siguiente pasada: solo legibilidad de §5-§6 (siguen densas). Decisión de Javi.

---

## Revisión "desde cero" (segundo agente; nota 6, subiría a 8 con W1-W2-W5) — atacada con 5 experimentos nuevos

- **W1/Q3** → **expR34**: ImageNet re-puntuado con p99.9 + 20 réplicas + rangos percentiles (Tabla B18); reordena honestamente (DINO/DINOv2 por debajo de las 20 réplicas; el exceso de ViT-T era del supremo) y coincide con el orden de ξ. Frase en §4.
- **W2 (z con 5 réplicas)** → §3: |z| grande se lee como "por debajo de todas las réplicas", no cola gaussiana; B18 aporta la versión de 20 réplicas. Pendiente coautores: censo completo a ≥30 réplicas desde el caché canónico.
- **W3** → "genuine" definido como relativo al null en §3 + tabla A.2 nueva (qué conserva/destruye cada null).
- **W4** → abstract añade "41/48 beyond 2σ on hierarchical datasets" junto al 14/24.
- **W5** → **expR37**: instrumento sobre features a nivel muestra (el objeto de Khrulkov/Yang): δ crudos en la misma banda baja, excesos casi todos en ruido (6/24 con |z|≥2; DINOv2/DTD positivo). Tabla B17 + related work reescrito ("our census is about inter-class geometry") + enlace desde §6.
- **Q4** → **expR36**: null plano con hubs k-means de los propios centroides (sin taxonomía humana): 24/24 negativos, 16/24 |z|≥2, DINOv2-L/G en ImageNet a −4.4/−4.6. Tabla B16 + §4.
- **Q2** → **expR38** (ξ vectorizado): en geometría angular el ξ negativo de DINOv2-S/B/L NO sobrevive (pasan a positivo); G sigue negativo (−12.5) y CLIP también. Tabla B19 + limitación (viii): la celda "no resuelta" se estrecha a efecto de normas.
- **Q1** → pooling detallado en el apéndice del panel (timm `num_classes=0`; last-token causal; pooling por embedder).
- **W7** → definición de *cell*, fórmula de ξ en el apéndice, nota de que el no-alineamiento cross-model bajo ϕ es esperable. Densidad §5-§6: parcial (limitada por espacio).
- **No seguido**: comprimir §6 a media página (conflicto entre revisores; decisión de coautores). **Pendiente coautores**: censo a ≥30 réplicas, iNaturalist (Q5), bootstrap ImageNet.

Compila: 25 páginas, texto principal hasta la 9, 0 warnings, sin `??`.

---

## Respuesta post-revisión (condición 6→8): reconciliación de ImageNet + censo a 20 réplicas

**Reconciliación del relato de ImageNet** (la condición del revisor, ~12 ediciones): la excepción del censo ya no se cuenta como "DINOv2 at null" sino como *supremum-specific*: §4 lo resuelve explícitamente ("DINOv2's ImageNet tree is real, in the bulk of its four-point distribution and among its own clusters; only the supremum read in the WordNet frame sits at null"), §5 lo pliega en forma-vs-contenido ("a real tree, least aligned with WordNet's top level"), y todas las frases "at null" (abstract-fig.1-ORC-§6-§7) quedan escopadas al supremo. La Tabla 1 muestra ahora el exceso supremo (z) **y la columna p99.9** lado a lado. La tensión δ/ξ se cierra igual (norm structure, expR38); caption de B19 con el resultado explícito.

**Censo a ≥20 réplicas** (segunda condición): **expR39** — las 72 celdas con 20 réplicas y rangos percentiles (Tabla B20): reproduce el censo de 5 réplicas con **signo 68/68** y significancia 62/68 en las celdas comparables; 56/72 por debajo de las 20 réplicas; mismas tres excepciones (DINOv2-S/B/G en IN). Los 4 ViT-ImageNet van marcados († almacén de centroides; no comparables directos con la Tabla 3). Frase en §3.

Ajustes de página por el material nuevo: título de §6 acortado ("Corollary: Diagnosing Exploitability"), limitaciones fusionadas (v-vi), recortes menores distribuidos. Compila: texto principal hasta la p. 9, 0 warnings, sin `??`.

**Para coautores**: regenerar las filas † de B13/B18/B20 desde el caché canónico de ImageNet; el resto de pendientes sin cambios.

---

## Revisión 4 (tercer agente, nota 5) — atacada con 2 experimentos + 1 análisis nuevos

- **W1/Q1 (estadístico primario)** → **expR40**: censo completo bajo p99.9 (20 réplicas, Tabla B21). Resultado decisivo: el censo NO depende del supremo — 70/72 negativos (vs 69/72), 40/48 jerárquicos con |z|≥2 (vs 41/48); solo cambia la identidad de las excepciones (ViT-T/IN +0.007 y ViT-B/C100 +0.001 en vez de DINOv2-S/B/G/IN), que es exactamente la historia supremo-vs-grueso ya contada en §4. Con esto, mantener el supremo como primario (pre-registrado, comparable en todo el paper) con p99.9 al lado es defendible con datos; frase añadida en §4.
- **W2 (¿δ cruda = proxy de rango efectivo/normas?)** → análisis nuevo (participation ratio + CV de normas por celda): el parcial de δ sobrevive within-dataset (−0.41/−0.53/−0.35/−0.20) pero el pooled NO (−0.33→−0.07), y el rango efectivo por sí solo correlaciona +0.99 con la ganancia en ImageNet. Reportado tal cual en §6 ("part of raw δ is spectral; its residual is the within-dataset signal").
- **W3 (ablaciones en δ cruda)** → los point clouds de pesos aleatorios/FT/capas no están en disco (solo CSVs con δ), así que no se pueden re-puntuar como exceso aquí. Caveat honesto añadido a "The form is learned" (el FT de superclases colapsa el rango efectivo 96→1; el label shuffle es null-invariante). Re-puntuar las ablaciones como exceso → coautores.
- **Q4 (Holm prometido y ausente)** → cierto; la promesa colgante eliminada del texto ("Per-cell values and z-scores are in Appendix").
- **W4 (§6 se autodebilita / comprimir)** → ya van 2 de 3 revisores frescos pidiéndolo; sigue siendo decisión de coautores, pero el balance ha cambiado (recomendación: comprimir a ~media página, políticas ya están en apéndice).
- **W5 (control positivo con MERU/HypViT)** → no ejecutado esta noche: requiere imágenes crudas + código del repo MERU (features no extraíbles desde los cachés). Propuesta concreta para coautores/otra sesión: MERU-ViT-B vs CLIP-B en CIFAR-100/10/DTD (datasets descargables por torchvision).
- **W6 (iNaturalist)** → coautores (ya en la lista).
- **W7 (legibilidad; ORC al apéndice)** → parcial una vez más; la sugerencia de mover ORC al apéndice queda anotada como opción de espacio para coautores.
- **Q3 (dos fuentes de centroides)** → ya documentado en B13(ii)/B18/B20; la Tabla 1 usa la caché del censo salvo la columna p99.9 (almacén, con caveat).

Ajuste de páginas tras integrar W2/W3/B21: limitación (i) duplicada eliminada, (iii)-(v) comprimidas, recortes de palabra en §5/§6. Compila: 26 páginas, texto principal hasta la 9, 0 warnings, sin `??`.

---

## Ronda 5 (seguimiento del tercer agente, 5→6): lo abierto, atacado

- **Ablaciones en δ cruda (su punto 3)** → tenía razón en que mi caveat empeoraba el caso: el brazo de fine-tuning *jerárquico* colapsa el rango efectivo 96→1 (una recta es un árbol trivial), así que "δ cerca del baseline" ahí no informa. **Brazo retirado como evidencia** (texto y caption de Fig. 3b); el brazo no-jerárquico (+19..+91 %), la profundidad y el shuffle (null-invariante) se mantienen, declarados como δ cruda. Re-puntuar las ablaciones como exceso → coautores (no hay point clouds en disco).
- **Inconsistencia nueva (abstract vs Tabla B21)** → cierta. **expR41**: null B (hubs WordNet) bajo p99.9 con 20 réplicas (Tabla B22): 23/24 negativos, **16/24 con |z_B|≥2**, pero los portadores cambian: ViT-T/S/B caen al null en ambos datasets (su exceso entre-hubs es de unas pocas cuádruplas extremas), DINO/DINOv2 pasan a z_B −5..−12 (grueso de la distribución). Abstract reescrito ("significant in 14/24 under the supremum and 16/24 under a supremum-robust statistic, with the carriers differing…") y frase de reconciliación en §4.
- **Legibilidad (punto 7)** → pasada de des-densificación sobre mis propios añadidos (§3 réplicas, §6 control espectral: cuatro cifras → un rango; los recuentos viven en las captions del apéndice). Texto principal sigue en 9 páginas.
- **MERU/HypViT (punto 5)** → NO ejecutado: requiere imágenes crudas (los cachés solo tienen features) y el código del repo de MERU; el disco raíz está al 100 % y un env nuevo con torch no cabe en /tmp. **Receta para coautores/otra máquina**: `pip install git+https://github.com/facebookresearch/meru` en un env aparte sobre el HDD; checkpoint `meru_vit_b`; extraer embeddings de imagen para CIFAR-100/CIFAR-10/DTD (torchvision) con el mismo preprocesado que CLIP-B; centroides → exp20-protocol (δ + null espectral, 20 réplicas) y null B; comparar exceso MERU-B vs CLIP-B. Predicción del paper: exceso MERU ≥ CLIP; si no, el instrumento tiene un problema. Es la validación positiva que falta y la más barata de las pendientes grandes.
- **iNaturalist (punto 6)** → coautores (imágenes + taxonomía; mismo pipeline que arriba).

Compila: 26 páginas, texto principal hasta la 9, 0 warnings, sin `??` (verificado antes del commit).

---

## Ronda 6 (seguimiento, nota 6 "sin asteriscos"): lo cerrado

- **Frase form-vs-content con números en §7** → añadida ("the models with the most WordNet content (supervised ViTs, CLIP) carry the most fragile form (supremum-borne), while the family with the most robust form (DINOv2) is the least WordNet-aligned").
- **Barras del FT jerárquico en la Fig. 3b** → **Fig. 3 regenerada desde los CSV de resultados** (`ICLR2027/figures/make_fig3_causal.py`, a 5.5 in y fuentes legibles, lo que además cierra el pendiente de legibilidad de esta figura): (a) pesos aleatorios (`e5_random_control.csv`: +65/+134/+144 %), (b) FT no-jerárquico (`analysis4_finetuning.csv`: +19/+69/+91 %), (c) profundidad (`e1_delta_by_layer.csv`). Sin el brazo jerárquico. El panel del shuffle no tiene CSV en disco → fuera de la figura, se mantiene en texto (p>0.12). ⚠️ **Cambio de cifras**: la caída con la profundidad sale del CSV como DINOv2-B −61 % / CLIP-B −46 % (la figura antigua, de una corrida no rastreada, decía −51/−29); el texto adopta el CSV por el estándar de trazabilidad del paper. Coautores: confirmar cuál es la corrida canónica. La figura vieja queda en `figures/fig3_causal_old4panel.pdf`.
- **Recuento 16 vs 17 en B22** → el 16 es correcto a precisión completa (CLIP-L/ImageNet z_B=−1.951 se mostraba como −2.0). B13/B16/B22 muestran ahora z con dos decimales, para que el recuento visible cuadre.
- Ajuste de páginas: Fig. 5 (scatter) 2.3→1.95 in de alto; recortes menores. Compila: 26 páginas, texto principal hasta la 9, 0 warnings, sin `??`.

**Sigue abierto (lo que separa el 6 del 7):** control positivo con MERU/HyCoCLIP (receta arriba) y densidad de la prosa.

---

## Ronda 7 (quinto agente, nota 6) — y dos RETRACCIONES importantes que su Q1 y su punto 3 destaparon

**1. Null B (hubs) retirado del paper.** Su Q1 ("¿qué exceso obtiene una mezcla de gaussianas con 30 componentes, estrella pura?") se contestó con **expR42** y la respuesta invalida el null B: una estrella sin jerarquía (clusters gaussianos alrededor de centros gaussianos) da exceso −0.08..−0.12 bajo el null A (ambas construcciones) **y también −0.03..−0.13 bajo el null B** (Tabla B25). Causa: 30 centros gaussianos iid forman un símplex casi regular, que ya *es* un árbol (estrella, δ≈0); cualquier aleatorización de hubs solo puede subir δ, así que "exceso B negativo" no certifica jerarquía. Además `spec_sample` con n=30 hubs no conserva el espectro (valores singulares inflados y desiguales), lo que agravaba el sesgo. **Consecuencia**: fuera del paper las tablas B13/B16/B22 y todas las frases "organization among hubs is tree-shaped", "24/24, 14/24, 16/24", hubs k-means, y el "resolves… among its own clusters" de DINOv2/ImageNet (que ahora se apoya solo en p99.9 y en el §5). En su lugar: párrafo "What the excess certifies" en §4 (el exceso espectral certifica estructura más allá de segundos momentos, clustering incluido; la profundidad la mide el control de submuestreo; el contenido, §5), limitación (i) nueva, fila anotada en la tabla de nulls, apéndice A.4 con B25. Los CSV de expR29/36/41 se conservan como registro.
**2. ξ degradado a descriptor de estructura de normas.** Su punto 3 se contestó con **expR43**: referencias *sin curvatura* pero con normas heterogéneas (gaussiana con radios log-normales, esfera con jitter radial, core+shell) dan ξ=−0.05..−0.32 con el 79-100 % de triángulos negativos, frente a +0.09 (gaussiana) y +0.12 (esfera) (Tabla B26). ξ mide dispersión de normas, no curvatura. **Consecuencia**: fuera del cuerpo las frases de "signo de curvatura" (§3, §4 "9 of 12 genuinely negative", "DINOv2 deepening", contribución 1, intro); §3 lo presenta como descriptor de normas; limitación (vi); apéndice renombrado. Coherente con expR38 (al normalizar normas, el ξ de DINOv2 desaparece).
**3. Null A: sensibilidad a la construcción** (**expR46**, Tabla B27): coeficientes gaussianos (paper) y PC-permutación coinciden; la construcción de espectro exacto (Haar) da excesos menores en nubes de bajo rango en datasets pequeños (DINOv2-L/C100 −0.054→−0.009). Frase de caveat en §3.
**4. Legibilidad (su punto 1)**: abstract reescrito sin jerga interna (254 palabras; un dato por hallazgo; sin la "tercera pata" del no-alineamiento cross-model, que él tenía razón en calificar de esperable); Tabla 1 a 10 columnas en `footnotesize` (ξ y ARI al apéndice).
**5. ConvNets (su Q2)**: **expR45** — Barlow Twins y BYOL (ResNet-50) por debajo del null en ImageNet (−0.039/−0.020, z −4.8/−2.4; Tabla B24); "near-universal" ahora acotado a "vision backbones tested (ViTs and two ConvNets)".
**6. Censo de texto a 20 réplicas (su punto 7)**: **expR44** en GPU (todos menos OLMo-7B, que no cabe en 11 GB) → Tabla B23.
**Sin cambio**: §6 (tres de cuatro revisores frescos piden comprimirlo; decisión de coautores — Q3 del revisor: no, la tesis no cambia sin §6). MERU (receta anotada). iNaturalist.

Compila: 26 páginas, texto principal termina en la p. 9 con holgura (Fig. 3 devuelta a 1.6 in), 0 warnings, sin `??`.

**expR44 integrado (Tabla B23)**: censo de texto a 20 réplicas, 15/15 signos coinciden con el de 3. Dos veredictos cambian y el texto lo refleja: OLMo-1B pasa a genuino (−0.039, z=−5.7) → la "emergencia con escala" queda solo en GPT-2 (OLMo genuino en ambas escalas); GTE-Qwen2 pasa a null (−0.002) → marcado como no robusto (texto, caption y anotación de Fig. 2). Abstract: "emerging with size in GPT-2".

---

## §6 comprimido a media página (decisión de Javi, 1 sept)

Tres párrafos: *Whether* (correlaciones within-dataset con controles de dimensión/accuracy/familia/espectro; pooled no usado como evidencia; flat datasets; CIs y McNemar), *Which metric* (control RT, división por objetivo, predicción del mejor métrica within-dataset, selección held-out, regla por objetivo vs coseno), *Scope and negatives* (diagnóstico ≠ política; δ-gated la peor; NC no mejora en media; coseno default; +0.5..+2 pp solo en prototipos; kNN/retrieval negativos; DBpedia; robustez a t en apéndice). Ninguna cifra nueva; las tablas de políticas, CIs y controles siguen en el apéndice. Compila: texto principal con holgura en la p. 9, 0 warnings.

---

## Ronda 8 (seguimiento del quinto agente, nota 6 → "7 con 4-5 limpios y abstract reescrito")

- **Abstract**: reescrito de nuevo, **199 palabras**, sin "beyond second moments"/"chaining artifact"/recuentos múltiples; una cifra (69/72) y lenguaje llano ("random cloud with the same dimension and spectrum", "an artifact of the clustering step").
- **Tabla 9 (B23) no cuadraba consigo misma** → cierto: expR44 *re-extrajo* los embeddings (inferencia fp16 por lotes), no solo cambió las réplicas. Caption lo declara; el texto ya no se apoya en OLMo en ningún sentido ("OLMo's verdicts differ between extractions and are not used"), la emergencia con escala queda solo en GPT-2 y la anotación de la Fig. 2 dice "(GPT-2)".
- **Restos**: Tabla 34 (a10) ya no se titula "curvature-sign estimator" (editado el .tex generado; el generador vive en la otra máquina); Tabla 24 (B18) remite al "norm-structure descriptor ξ"; Tabla 6 (B24) ya no anuncia "two further ViT-SSL recipes".
- **Tabla 27 → §4**: frase en el cuerpo: la *magnitud* (no el signo) de los excesos en datasets pequeños depende de la construcción del null (DINOv2/C100 −0.04..−0.08 → −0.007..−0.03; signos 35/36).
- **Ablaciones en δ crudo (punto 6)**: brazo de pesos aleatorios retirado como evidencia (features casi gaussianas ⇒ δ crudo mayor por construcción); Fig. 3 pasa a dos paneles (FT no-jerárquico, profundidad); "The form is learned" → "The form tracks training"; abstract/intro/contribución 2 en consonancia ("training-dependent").
- **Circularidad de la regla por objetivo (§6)**: frase explícita ("read off the same three families it is scored on… describes these families rather than validating a policy").
- **Su pregunta (¿null B calibrado contra la estrella como test de profundidad?)**: sí, es prometedor — bajo Haar la jerarquía 6×5 queda 2-3× por debajo de la estrella; anotado en la caption de B25 como trabajo futuro (requiere una estrella emparejada por celda en número y compacidad de clusters).

Compila: 26 páginas, texto principal hasta la p. 9, 0 warnings, sin `??`.

---

## Ronda 9 (seguimiento, nota 7): flecos + la pregunta de la variabilidad de extracción

- **Fig. 3**: la versión comprometida ya tenía dos paneles (el revisor vio una copia vieja); sin cambios.
- **Fig. 2 caption** → "GPT-2" (sin OLMo). **B23** → "between extractions". **Intro y contribución 3** → fuera "hyperbolic distances buy no cross-model agreement" (queda solo como nota "esperable" en §5).
- **Variabilidad extracción-a-extracción (su pregunta)** → **expR47** lo atribuye: en GPT-2 M la precisión es irrelevante (fp16≡fp32 a 4 decimales) y el *batch* mueve δ̂ hasta 0.018 (0.082 sin padding vs 0.091/0.100 con batch 32/16; coseno 0.98 con la extracción sin padding): HF no desplaza los position ids de GPT-2 con padding a la izquierda. OLMo-1B es invariante (0.082 en las seis condiciones, igual al valor impreso del paper; el 0.100 de exp18 era otra revisión del checkpoint). Tabla B28 + frase en §4. **Protocolo canónico → extracción sin padding (batch 1)**: **expR48** re-corre el censo de texto así (20 réplicas); B23 pasa a ese bloque al terminar, con la columna Δδ̂ frente a la extracción original.
- ⚠️ Para coautores: la caché del censo de texto original (exp18/exp4) se extrajo con batch 32 y padding a la izquierda; los valores de GPT-2 del cuerpo deberían re-derivarse de la extracción sin padding en la versión final (el trend con escala se verifica en B23).

**expR48 integrado (censo de texto sin padding, 20 réplicas → B23 y Fig. 2)**: con extracción sin padding, GPT-2 S y M quedan *en* su null (−0.015/−0.010; el "+0.048 por encima del null" de S era artefacto del padding), L y XL genuinos (−0.041/−0.049, por debajo de las 20 réplicas): la emergencia con la escala sobrevive como "genuino a partir de L". Pythia idéntico (−0.032..−0.036), OLMo-1B genuino (−0.039, dos re-extracciones coincidentes; se recupera OLMo en el texto), embedders idénticos a la extracción original (el padding a la derecha con máscara no afecta), GTE-Qwen2 sigue en null y sin uso. Texto de §4 e intro reescritos con los valores sin padding; Fig. 2 regenerada desde expR48 (OLMo-7B omitido, no re-extraído); captions de B14 (plantillas, batched) y B23 actualizadas. Compila: 26 páginas, texto principal hasta la p. 9, 0 warnings.

## Ronda 10 (2026-09-01) — revisión completa (score 5) + cabos

Prioridad del revisor: "moderadamente compartido" en abstract/§7 y pasar la calibración estrella de future work a resultado.

- **Framing "moderately shared"**: abstract (2) y cierre, intro ("The tree is moderately shared"), §5 "corrected picture" (recuperación parcial 0.2–0.5 vs 0.5–0.6 supervisado), §7 ("moderately shared and configuration-dependent", gap ImageNet 0.13–0.38 según umbral). Cierre honesto: "shared clustering and a partial, supervision-dependent recovery of the human taxonomy, not one common tree". Abstract 201 palabras.
- **Test de profundidad calibrado por estrella → resultado (expR50, Tabla B29)**: cada backbone vs una *estrella emparejada* (hubs gaussianos con el RMS real, nube isotrópica con la dispersión intra-superclase real, mismos tamaños) bajo la null B en construcción Haar, restando el sesgo propio de la null en la estrella. ImageNet (WN-30): sólo DINOv2-L supera su estrella más allá del ruido (z −2.5; DINOv2-G −1.7; resto |z|≤1.6). CIFAR-100 (coarse-20): todos los backbones son *menos* arbóreos que su estrella (z hasta +7). Conclusión: el exceso lo explica la organización en clusters/estrella; profundidad residual marginal. §4 "What the excess certifies", Limitación (i), abstract y leyenda B25 (incluye Q(a): star_mid −0.104 vs hier6x5 −0.14 bajo null gaussiana) actualizados.
- **Un estadístico primario**: §3 declara supremo = primario pre-registrado, p99.9 = robustez; §4 y §5 ya no "resuelven" la excepción DINOv2: se reportan ambos sin adjudicar.
- **Rango 0.13–0.38 en el cuerpo** (§5, §7). **Q(b)** en A.x tree map: el criterio se aplica por dataset (la configuración que elige en DBpedia es la degenerada en ImageNet).
- **Q(c)** sin DINOv2 (6 backbones): NC ImageNet −0.87, CIFAR-100 −0.65, CIFAR-10 +0.23, DTD −0.34 (FS −0.12/−0.63/−0.11/−0.50) → §6 y leyenda B12: la predicción aguanta en ImageNet/CIFAR-100, no en CIFAR-10/DTD.
- **Park et al. 2024** (geometría de conceptos jerárquicos en LMs) citado en related work + bib.
- **A.3**: nota de pooling → extracción sin padding, un prompt cada vez. **B1** (tab_b1_textnulls, generada en otra máquina): leyenda editada a mano como "original extraction, kept for comparison"; B23 canónica.
- **Limitación (vii)**: dependencia del protocolo de extracción (left-padding mueve δ̂ hasta 0.018).
- **Erratas**: "Whose tree, is §5's subject" reescrita; Fig. 6 "Reality" tapado con cajas overpic + "root" en (b) y (c) (la cadena "Reality" sigue en la capa de texto del PDF; coautores: regenerar la figura). Negrita de Tabla 1 y Fig. 3 de dos paneles ya estaban en el .tex (el revisor vio una versión anterior).
- **Pendiente en esta ronda**: expR49 (plantillas GPT-2 sin padding, 40 celdas) → B14 y la frase de §4 sobre plantillas (de momento sólo "the scale ordering is template-robust").
- Compila: 28 páginas, texto principal termina en p9, 0 warnings, 0 `??`.
- **expR49 terminado (plantillas GPT-2 sin padding, 40 celdas, 3 réplicas de null)**: exceso medio −0.012 (S) → −0.023 (M) → −0.040 (L) → −0.041 (XL); celdas con |z|≥2: 5/8/9/10 de 10. La ordenación por escala se mantiene; S ya no es "genuine on none": sus celdas son someras y, en la plantilla base, dentro del ruido del censo de 20 réplicas (B23: S z −1.6, M z −1.2). B14 regenerada desde expR49 (el generador cae a expR30 si faltan filas); §4 actualizado con los números. Commit 2 de la ronda.
- **Control con backbone hiperbólico (expR51, Tabla B30, opción A elegida por Javi)**: MERU S/B/L (Desai et al. 2023; Lorentz + entailment) frente a sus gemelos CLIP del mismo repo (mismo ViT, mismos datos RedCaps, misma receta sin lift hiperbólico), por el instrumento sin cambios, en CIFAR-100 imágenes (coarse-20), ImageNet 50 img/clase (WN-30) y prompts de clase. Resultado (24 celdas): entrenar en el hiperboloide no cambia nada a nivel de clases — mismo exceso de clustering que el gemelo (z −1.5..−4.9 en imágenes), profundidad frente a estrella emparejada nunca negativa más allá del ruido (a menudo positiva: menos arbóreo que su estrella), y la δ̂ de MERU en su propia métrica de Lorentz igual a la euclídea a 3 decimales. Lectura en el paper (§4 + leyenda B30): el exceso mide cómo se organizan las clases, no la geometría en la que se sumergen; la jerarquía de MERU es genérico→específico (texto⊃imagen), no una taxonomía de clases; el árbol sintético 6×5 de B25 sigue siendo la prueba de existencia de que el test de profundidad dispara cuando hay profundidad. Texto en ImageNet: ambas torres al null (+0.002..+0.009), consistente con el hallazgo de embedders. Script rebuttal/scripts/expR51_meru_control.py; cachés en Platonic/results/meru_cache. Commit 3 de la ronda.

## Pasada de versión final (2026-09-03) — brief "final-version pass"

Ejecutada íntegra; detalle en `ICLR2027/CHANGELOG_final.md` (números viejo→nuevo con fuente) y
`ICLR2027/TODO_author.md` (vetos pendientes: centroides de R2, figura best-metric al apéndice, tamaño de la
Fig. 1 de Canva; filas de calibración sin CSV; Fig. 6 sin script). Hitos: cachés de ImageNet regeneradas en
local con el pipeline original (fidelidad 4 decimales en los 12 modelos); censo homogéneo de 20 réplicas
`expR39b` (69/72 · 57/72 · 43/48); calibración Gröger K=200 `exp21b` (mKNN 0.472→0.464, CKA 0.633→0.577,
p=1/201 en los 66 pares); evidencia primaria = rango de percentil; caja p1; Figs. 1 (placeholder Canva) y 2
(overview) nuevas; Tabla 1 de 7 columnas; criterio del tree map ejecutable; tabla de calibración generada;
estilo/paleta de figuras; sweep_freeze 67/67; texto principal + Ethics en p9. Commit `1205a73`.
