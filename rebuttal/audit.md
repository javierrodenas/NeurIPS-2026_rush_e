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

---

**Resumen**: 1 artefacto encontrado y corregido (ARI/average-linkage), 1 concesión suavizada con datos mejores (dimensión igualada por RP en vez de PCA), el hallazgo cosine/DINOv2 re-verificado por 5 vías independientes, y todo lo demás robusto a método, métrica y null. Los drafts (`response_RJje.md`, `response_pux6.md`, `response_Xbn5.md`) ya incorporan las correcciones; `comment_AC_global.md` y `response_1Eqj.md` no citaban ninguno de los números afectados.
