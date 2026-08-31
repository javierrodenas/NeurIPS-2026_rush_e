# Revisión del borrador ICLR 2027 (`iclr2027/iclr2027/main_iclr2027.tex`) — 31 ago 2026

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
`tectonic main_iclr2027.tex` dentro de `iclr2027/iclr2027/` (descarga paquetes la primera vez).

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
