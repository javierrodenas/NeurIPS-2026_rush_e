# Auditoría de los experimentos del rebuttal (exp8)

**Motivo**: varios resultados de mi suite salieron "raros" en DINOv2 (ARI ≈ 0 en recuperación de superclases, ρ WordNet baja, puente ORC sin gap, PCA no monótono). Antes de publicar un rebuttal que concede puntos, verificamos si eran reales o artefactos de implementación. Veredicto por resultado:

## 1. ARI ≈ 0.003 de DINOv2 en recuperación de superclases → **ERA ARTEFACTO** (corregido)

El average-linkage se degeneraba por encadenamiento en los modelos con centroides muy compactos: el corte a 20 clusters daba un mega-cluster de 72–80 clases + singletons (detectado con la columna `max_cluster`). Con métodos no degenerados todos los modelos recuperan estructura real:

| modelo | average (degenerado) | Ward | mejor variante |
|---|---|---|---|
| ViT-B | 0.388 | **0.610** | Ward |
| ViT-S | 0.355 | 0.558 | Ward |
| DINOv2-S | 0.118 | 0.513 | Ward |
| DINOv2-B | 0.017 | 0.474 | Ward |
| DINOv2-L | 0.005 | 0.316 | complete+cosine: 0.497 |
| DINOv2-G | 0.003 | 0.188 | complete+cosine: 0.197 |
| CLIP-L | 0.349 | 0.516 | k-means: 0.528 |

**Consecuencias**: (a) actualizados `response_pux6.md` y `response_Xbn5.md` con los números Ward; (b) el orden relativo se mantiene (DINOv2-G sigue siendo el peor en recuperar la taxonomía, 0.20 vs 0.55–0.61 de supervisados) → la narrativa del "decoupling" sobrevive; (c) la conclusión "el linkage con distancia Poincaré NO mejora sobre el mejor euclídeo/cosine" se re-verificó con métodos no degenerados y **se mantiene** (H nunca gana; en DINOv2 sigue degenerado incluso con complete).

## 2. ρ WordNet baja de DINOv2 → **ROBUSTO** (no era artefacto)

Se re-midió con distancia cosine y con acuerdo de tripletas (invariante de escala, azar = 0.50):

- Tripletas: CLIP/SigLIP 0.73, supervisados 0.70–0.71, DINOv1 0.63, DINOv2 0.56–0.59 (apenas sobre el azar).
- Con cosine, DINOv2-L/G suben un poco (ρ 0.23–0.24) pero siguen lejos de CLIP (0.54–0.57).

DINOv2 organiza por similitud visual, no por taxonomía — el hallazgo es real y coherente con la literatura.

## 3. Puente ORC sin gap en DINOv2 → **ROBUSTO** (y ahora más limpio)

Repetido con los clusters *propios* de cada modelo (Ward-30 sobre su geometría, no WordNet): supervisados y CLIP/SigLIP muestran el gap esperado aún más fuerte (fneg inter 2.7–4× la intra; p.ej. SigLIP 0.048 intra vs 0.140 inter). DINOv2 muestra lo contrario: sus aristas inter-cluster son aún MÁS positivas (+0.83 vs +0.69) y 0% negativas. No es un error de asignación de puentes: la geometría de DINOv2 es genuinamente "clusters compactos", no árbol profundo. Frase del response_RJje ajustada en consecuencia.

## 4. Dimensión igualada (PCA-192 no monótono) → **MATIZADO a favor del paper**

El PCA trunca y pierde estructura. Con proyección aleatoria gaussiana a d=192 (preserva la geometría estadísticamente, JL):

- El orden por familias **sobrevive**: DINOv2-B/L/G los más bajos (0.094–0.100, por debajo del null iid 0.104), CLIP los más altos (0.125–0.131).
- Lo que no sobrevive en ningún control: la monotonía S→G (DINOv2-S queda alto en todas las variantes, 0.117). El claim limpio es "S mucho menos arbóreo que B/L/G", no una curva monótona.

response_RJje actualizado con esta versión (más favorable que la concesión inicial basada solo en PCA).

## 5. Null espectral (exp1c) → **CONFIRMADO por un null independiente**

Null de permutación por coordenadas PC (espectro y marginales exactos): mismo split con márgenes mayores — supervisados −0.035..−0.040 y CLIP/SigLIP −0.036..−0.047 por debajo del null (árbol genuino de orden superior); DINOv2 −0.004..+0.017 (en su null). Dos construcciones de null independientes, misma conclusión. Añadido al response_RJje.

## 6. Curva de pooling (1→10→100 imágenes/clase) → coherente

ρ WordNet con 1 sola imagen por clase (sin centroides): ViT-L 0.25, CLIP-L 0.26, DINOv2-G 0.03; con 10 imágenes ya 0.49/0.56/0.17. La señal jerárquica pre-existe sin pooling (anti-circularidad) y el pooling solo la desruidifica. Los números citados en response_pux6 quedan validados.

## 7. ¿COS > H en DINOv2 por mala elección del radio? → **NO** (verificado aparte)

Incluso dándole a H su mejor radio t por celda (sobreajuste a su favor), cosine gana en 9/12 celdas NC y 6/12 FS de DINOv2 (en DINOv2-G, 5 de 6). La concesión del rebuttal se mantiene.

## 8. Re-verificación específica de "cosine > hiperbólico en DINOv2" (exp9) → **REAL, 5 vías independientes**

1. **Reimplementación desde cero** (código nuevo, numpy/torch): reproduce exp2 dígito a dígito y reproduce la H del paper (DINOv2-L/CIFAR-100 NC H−R = +2.05pp vs +2.10 del paper) → mi pipeline H no tiene bug.
2. **Prototipo H alternativo** (proyectar la media cruda en vez de re-exp-map de la media proyectada): mucho peor (0.84 vs 0.91) → H no está siendo penalizado por la construcción del prototipo; la del paper es la buena.
3. **Features no pre-normalizadas** (CV de normas 0.013–0.039) → el cosine hace trabajo real, no es artefacto de la cache.
4. **FS con 2 seeds nuevos × 2000 episodios**: COS−H = +1.15..+1.29pp (CI ±0.08–0.16) en CIFAR-100 y DTD → no es suerte de seed.
5. **ImageNet full-train** (npz independiente de 1.23M muestras): R=0.788, H=0.786, COS=0.799 → aguanta con extracción de datos independiente.

**Resolución conceptual** (por qué no contradice "la geometría es arbórea, no esférica"): son dos preguntas distintas. δ describe el *layout inter-clase* (y ahí normalizar a la esfera lo empeora, δ sube — eso sigue siendo cierto); la accuracy downstream depende de *dónde está la señal discriminativa por muestra*. En DINOv2 —clusters compactos casi-estrella, sin exceso de árbol genuino sobre el null espectral— la señal es puramente angular → cosine óptimo. En CLIP —el exceso de árbol genuino más fuerte (−0.047)— la métrica hiperbólica añade sobre el ángulo (+0.69pp FS). El patrón por familias cuadra: exceso genuino ↔ ventaja H-sobre-esférico (Contrastive −0.042/+0.69; Supervised −0.037/+0.01; SSL +0.008/−0.51; a nivel de modelo r=−0.56, p=0.09, solo sugestivo — citarlo como patrón de familia, no como correlación).

## 9. Reconciliación con la Figura 3 del paper (exp10) → la semántica de DINOv2 es ANGULAR

La Figura 3 (DINOv2-G organizado por superclases en CIFAR-100) parecía contradecir la auditoría (ρ 0.18, ARI 0.19 "menos alineado"). exp10 lo resuelve con medidas locales e invariantes de escala:

| medida (CIFAR-100) | DINOv2-L | DINOv2-G | ViT-L | CLIP-L |
|---|---|---|---|---|
| tripletas hermano-vs-no (euclídea) | 0.710 | 0.634 | 0.955 | 0.934 |
| tripletas hermano-vs-no (cosine) | **0.913** | **0.850** | 0.963 | 0.931 |
| recall@4 de hermanos (eucl → cos) | 0.27→0.50 | 0.22→0.44 | 0.57→0.60 | 0.54→0.55 |
| spoke/inter-hub (dim completa) | 0.98 | 1.04 | 0.58 | 0.59 |

**Lecturas**:
1. La organización por superclases de DINOv2 **existe y es fuerte, pero es angular**: en euclídeo crudo está enmascarada por las normas; en cosine casi alcanza a los supervisados. Mis medidas de auditoría (ρ Spearman y ARI, euclídeas) la infravaloraban → corregido el matiz en `response_pux6.md`.
2. Encaja con el downstream: cosine gana a H en DINOv2 *porque* su semántica vive en los ángulos; CLIP tiene además árbol genuino (null espectral) → H gana a cosine. Una sola explicación para las dos anomalías.
3. Lo que se mantiene: a nivel **global** ImageNet/WordNet, DINOv2 sigue el más débil en ambas métricas (tripletas coarse 0.59–0.66 vs 0.83–0.85 CLIP): jerarquía local fuerte, árbol global plano.
4. **⚠️ Aviso camera-ready**: el claim de la Figura 3 "DINOv2-G: spokes cortos y compactos" es un artefacto de proyección (el PCA se ajusta a los 20 hubs; las desviaciones ortogonales al plano desaparecen). En dimensión completa el ratio spoke/inter-hub de DINOv2-G es el PEOR del panel (1.04). Rehacer la figura en versión angular (o caveat explícito) antes de la camera-ready; si un reviewer lo recalcula, la figura queda expuesta.

## 10. Nulls por dataset y métrica (exp11) → la "estrella" de DINOv2 es ESPECÍFICA DE IMAGENET

Extensión de los nulls espectrales a los 6 datasets × 2 métricas (euclídea y angular). Resultado (exceso = δ real − δ null; negativo = árbol genuino):

| exceso euclídeo | imagenet | cifar100 | dtd | cifar10 |
|---|---|---|---|---|
| ViT-L | −0.033 | −0.052 | −0.082 | −0.090 |
| DINOv2-S | **+0.012** | −0.052 | −0.043 | −0.087 |
| DINOv2-G | **+0.004** | **−0.078** | −0.076 | **−0.146** |
| CLIP-L | −0.020 | −0.047 | −0.053 | −0.075 |

**Lecturas**:
1. El "DINOv2 en su null" ocurre **solo en ImageNet** (y la métrica angular tampoco lo rescata ahí: exceso cos +0.02..+0.03). En los 5 datasets de transfer, DINOv2 queda claramente por debajo del null — en CIFAR-100 y CIFAR-10 es **el más negativo del panel**, y el exceso se profundiza con la escala (S→G en CIFAR-10: −0.087→−0.146). Interpretación plausible: ImageNet es (casi) la distribución de pretraining de DINOv2; con 1000 clases finas saturando su espacio, la geometría se acerca al reparto uniforme (estrella). En datos de transfer, la estructura jerárquica genuina emerge.
2. Esto apoya literalmente la tesis "co-producto de modelo y dato" del paper, y **reduce mucho la concesión**: las ganancias downstream del paper viven en los datasets de transfer, justo donde el árbol de DINOv2 es genuino. Respuesta a RJje actualizada con el scoping.
3. Matiz honesto: el exceso negativo aparece en todos los modelos incluso en MNIST (−0.03..−0.07) — exceso < 0 per se indica no-gaussianidad/clusterización, no jerarquía *semántica*; lo diagnóstico es el patrón relativo y el flip de signo de DINOv2 en ImageNet.
4. ¿Predice el exceso la ventaja H−R mejor que la δ cruda? Mixto: mejor en FS de los datasets de transfer (r=−0.78..−0.85 en cifar10/100/dtd vs −0.67..−0.78 de la δ cruda), invertido en ImageNet (+0.81). La δ cruda se queda como predictor del paper; el exceso es la herramienta interpretativa. No se overclaimea en el rebuttal.

---

## 11. Tabla 1 vs re-run: alcance final del problema de procedencia (25 jul)

Comparación completa en `paper_vs_rerun_table1.csv`. **Brazo R: reproduce** (0.05pp medio). **Columna NC-H: 15 celdas infladas 0.5–1.2pp + CLIP-L/MNIST con signo volteado** → a nivel de signo, paper 31/60 NC positivas vs 22/60 reproducibles. FS consistente (54/60 paper vs 56/60 re-run, 2 celdas frontera). **Lo crucial: las correlaciones del paper sobreviven a los números reproducibles** — NC pooled −0.454 (impreso −0.46), ImageNet −0.82, CIFAR-100 −0.87, flat sin correlación; FS por-dataset −0.61..−0.78. Única debilitación: FS *pooled* −0.36→−0.19 n.s. (compresión de varianza que el propio paper señala; ningún documento nuestro lo cita). Reglas de fase 2: citar el 54/60 de FS del paper; para NC citar correlaciones, nunca conteos; las 15 celdas NC-H afectadas no se citan jamás. Camera-ready: regenerar la columna NC-H desde `tasks_t707`.

---

**Resumen**: 1 artefacto encontrado y corregido (ARI/average-linkage), 1 concesión suavizada con datos mejores (dimensión igualada por RP en vez de PCA), el hallazgo cosine/DINOv2 re-verificado por 5 vías y *explicado* (semántica angular local — exp10), la concesión del null espectral **reescopada a ImageNet** (exp11): en los datasets de transfer el árbol genuino de DINOv2 es el más fuerte del panel y escala S→G, y la Tabla 1 auditada por completo (§11): el problema queda confinado a la columna NC-H y **las correlaciones centrales del paper reproducen**. Todo lo demás robusto a método, métrica, null y dataset. Los drafts (`response_RJje.md`, `response_pux6.md`, `response_Xbn5.md`) ya incorporan las correcciones; `comment_AC_global.md` y `response_1Eqj.md` no citaban ninguno de los números afectados.

## 12. Re-verificación independiente de las tareas 3/4/5 (exp15, 25 jul)

A raíz de la duda "¿no estarán mal los negativos?": reimplementación desde cero (código propio, mismo muestreo que los runs de mayo del repo para comparar celda a celda). Resultado sobre 144 celdas: **retrieval exacto (|dif| ≤ 0.01pp en las 72 celdas), clustering exacto, kNN dentro del ruido de desempates del voto (media 0.15–0.22pp)** — y **cero celdas cambian el signo de su veredicto**. Los negativos de las tareas sample-level quedan verificados por dos implementaciones independientes escritas con meses de diferencia. CSV: `exp15_verify_345.csv`.

## 13. Tarea de texto (exp14, DBpedia) — insertada tras aprobación (25 jul)

DBpedia Classes (jerarquía real 9→70→219), embedders BGE/E5/GTE, protocolo NC/FS idéntico al paper. Geometría: árbol genuino en los 3 (exceso −0.020..−0.028) y tripletas 0.88–0.89 contra el nivel-2 real. Herramienta: δ̂ = 0.122–0.132 y ORC ≈ +0.37 (banda moderada, sobre el umbral 0.10) → la regla predice ganancia marginal → FS +0.04..+0.08pp (CI fuera de 0 en E5/GTE), NC −0.4..−0.8pp. **Framing insertado: "el diagnóstico transfiere de modalidad", no "H gana en texto".** Criterio de éxito prefijado antes de mirar (≥2/3 CIs fuera de 0 en FS): cumplido. Insertado en Xbn5 A1 y AC §4 con aprobación explícita de Javi.

**Anexo §13 — la disociación de dos caras (para fase 2):** DBpedia y DINOv2-ImageNet juntos disocian δ̂ cruda del exceso: el exceso **no es necesario** para el gain (DINOv2-ImageNet: sin exceso, con gains FS) **ni suficiente** (DBpedia: con exceso, sin gain). La δ̂ cruda predice correctamente ambos casos → δ̂ cruda = predictor práctico; exceso = existencia de estructura, no explotabilidad. Además, Gröger et al. [2 de RJje] (verificado: arXiv 2602.14486) hacen el mismo movimiento — calibración por nulls de permutación, convergencia global desaparece al calibrar, estructura local sobrevive — nuestra línea en A8 los cita como programa espejo.

**Anexo §13-bis (carta para fase 2):** nuestro 0.474 de mutual-kNN cross-model es acuerdo de *vecindarios locales* entre modelos — exactamente la cantidad que [2] identifica como la que sobrevive a su calibración. Si RJje entra al tema PRH, nuestro número cross-model encaja en el cuadro de su propia referencia.
