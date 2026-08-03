# Plan de ataque ICLR 2027

Consolidación de TODO lo señalado por los 4 reviewers + meta-review + discusión,
organizado por temas. Cada tema: qué dijeron → qué tenemos → qué falta.
Estado: ✅ hecho y verificado · 📝 solo falta escribirlo · 🔬 falta cómputo/experimento · ⚠️ decisión pendiente.

---

## A. Framing y claims (RJje W1, Xbn5 W1-W2, pux6 W3, meta punto 5)
**Dijeron:** "interpretation too strong", "universal geometry" excesivo, PRH no establecido,
tree-like ≠ curvatura negativa, título no refleja el mensaje real.
**Tenemos:** ✅ título nuevo (comprometido en OpenReview), tesis form-not-content con su
experimento (mutual-kNN 0.474 vs 0.427), terminología "tree-like inter-class metric
structure", posición §2.4 lista para promocionar, literatura crítica PRH ([Koepke 2026],
[Gröger 2026]) leída y posicionada.
**Falta:** 📝 abstract + intro nuevos (los 3 pilares: forma / contenido / herramienta);
📝 barrido terminológico global: nunca "hyperbolic representation" como claim descriptivo,
nunca curvatura sin "we do not estimate curvature"; 📝 PRH reducido a un párrafo de
related work. ⚠️ Decidir: ¿"form, not content" va al abstract? (recomiendo sí, es memorable
y es nuestro).

## B. Nulls, baselines y calibración (RJje Q1-Q2, 1Eqj Q2-Q3, meta punto 2)
**Dijeron:** sin esfera/Gauss/whitening/dimensión-matched no se distingue de concentración;
constantes mal atribuidas; "low δ" sin referente.
**Tenemos:** ✅ tabla de calibración completa (árbol/H²/S⁹⁹/Gauss); ✅ nulls espectro-matched
+ PC-permutation, 72 celdas → 69/72; ✅ tendencia de capacidad en base de excesos
(S→G CIFAR-10: −0.087→−0.146); ✅ esfera 12/12 geodesic, 10/12 chord; ✅ ln 2 vs ln(1+√2)
con cita Nica-Spakula (regalo de 1Eqj); ✅ "low δ" = "below matched null".
**Falta:** 📝 sección Background con las dos constantes y las dos escalas;
🔬 **regenerar las figuras principales en base de excesos** (Fig. 2 de escalado y las
que comparen dimensiones distintas deben mostrar exceso, no δ̂ crudo — decisión de
presentación + regeneración de plots); 📝 el exceso como cantidad primaria en TODAS las tablas.

## C. Integridad de datos (auditoría interna — invisible para reviewers, crítico para nosotros)
**Tenemos:** ✅ auditoría completa (audit.md §11-12): columna NC-H de Table 1 con 15 celdas
infladas + sign flip CLIP-L/MNIST; δ ImageNet DINOv2-G .066→.080 y CLIP-L .108→.118;
brazo R reproduce a 0.05pp; correlaciones sobreviven (−0.454 vs −0.46); tasks 3/4/5
verificadas 144/144.
**Falta:** 🔬 **regenerar Table 1 desde tasks_t707** — PRIORIDAD 1: el paper de ICLR nace
con la tabla limpia; 🔬 re-verificar cada número titular contra la tabla regenerada
(⚠️ ALGUNOS TITULARES PUEDEN CAMBIAR, p.ej. el +2.1pp — avisar a coautores de esto);
🔬 rehacer Figura 3 (compacidad de spokes = artefacto PCA) con visualización honesta;
📝 pipeline número→script documentado (reproducibility statement con sustancia).

## D. La historia métrica (RJje Q3-Q4, pux6 Q3, Xbn5 Q1, toda la discusión)
**Dijeron:** ¿y si es solo rescaling? ¿y si coseno basta? (pux6: "near-inversion").
**Tenemos:** ✅ exp2/exp2b completos (R/H/RT/COS/COSC/HN/RTN × 60 celdas del grid del
paper); ✅ protocolo de dos pasos (cuándo salir de Euclídeo / qué métrica cosecha);
✅ correlaciones best-metric (FS −0.31/−0.33, NC −0.22, COS−R −0.39) certificadas;
✅ S→G +0.6..+0.8→+2.4..+2.6pp; ✅ MNIST metric-agnóstico; ✅ distinción
sphericize-configuración vs métrica-de-lectura; ✅ RT ≤ 0.25pp.
**Falta:** 📝 **sección 5 entera — es material nuevo nacido en la discusión, nunca tuvo
forma de paper**; 🔬 figura nueva: δ̂ (o exceso) vs ventaja del mejor métrica (el scatter
que sustituye/acompaña a la Fig. 4); ⚠️ decidir si el protocolo va como algorithm box.

## E. Estadística (Xbn5 Q3)
**Tenemos:** ✅ CIs pareados por episodio (en exp2 CSV); ✅ McNemar (242/37, p≈4×10⁻³⁸,
exp13 CSV); ✅ null s.d.
**Falta:** 📝 tabla por-celda en apéndice; 🔬 barras de error en las figuras de ganancias.

## F. Corrección semántica y circularidad (pux6 Q1-Q2, RJje Q6, meta punto 3)
**Tenemos:** ✅ WordNet ρ hasta +0.59 (controles ≈0); ✅ ARI 0.61 (4 métodos); ✅ groupings
11/12; ✅ no-pooling ρ+0.34; ✅ decoupling DINOv2 (más tree-like, menos alineado) +
descomposición angular (triplets 0.91/0.85 vs 0.71/0.63); ✅ prompts 6 templates.
**Falta:** ✅ **HierarCaps HECHO (exp16, 3-ago)** — compromiso público con pux6 ("commit to such an evaluation
in the revision"): montar pipeline (dataset ECCV'24, 73K imágenes, jerarquías de 4 niveles),
medir alineamiento nivel-a-nivel y orden radial. ⚠️ Es el mayor bloque de trabajo nuevo;
decidir alcance esta semana. 🔬 prompts WordNet-definition y no-visuales (comprometido
a RJje, barato: variante de exp4).

## G. ORC (RJje W3, 1Eqj L224-228, meta punto 2)
**Tenemos:** ✅ análisis de puentes (negativos 2-4× en bridges, CLIP-L .040 vs .084;
excepción DINOv2 ≈0% coherente con su null).
**Falta:** 📝 párrafo §3.2 reescrito (mean ORC = clusters prietos; negativos = puentes);
📝 el ranking de ORC con su interpretación (lo que 1Eqj pidió); ⚠️ ¿figura del grafo
con edges coloreados? (bonita pero no comprometida — solo si sobra tiempo).

## H. Presentación (1Eqj TODO, meta punto 1)
**Tenemos:** ✅ la tabla resumen de texto principal (diseñada y enseñada en el rebuttal);
✅ lista de revisiones detallada (intro 4→2 párrafos, verbo L59, §2.3→Background,
sin \mathbb{H} en prosa, citas §3.1, argmin, FS como algoritmo, Fig. 2 grande,
ref rota L242, protocolo junto a setup, A1-A4 con nombres descriptivos).
**Falta:** 📝 ejecutarla entera al escribir; 🔬 diagrama del exponential map (comprometido);
📝 apéndice troceado por secciones. Regla de oro nueva: cada sección abre con su
conclusión (lo pidió 1Eqj explícitamente).

## I. Negativos y dominio de validez (Xbn5 W1, pux6 Q3/W3, meta punto 4)
**Tenemos:** ✅ kNN −2.6pp, retrieval 0/60, con mecanismo (jerarquía a nivel prototipo,
el mapa radial distorsiona vecindarios); ✅ DBpedia (predicción negativa correcta,
excess genuino, triplets 0.88); ✅ dominio de validez formulado.
**Falta:** 📝 entra TODO al texto principal (en NeurIPS era material de rebuttal);
⚠️ profundidad de DBpedia: ¿sección corta propia o dentro de la sección 5? (recomiendo
media página en §5: es la prueba de transferencia del diagnóstico).

## J. Reserva no comprometida (decidir si entra)
- ✅ exp12 CORRIDO (3-ago): calibración separa signos; DINOv2 negativo y monótono con escala (ξ −0.02→−0.07, 66→89% neg). ENTRA en §4.
- ✅ correlaciones exceso-vs-ganancia (audit §10.4): ⚠️ ¿entran en §5? (miran justo la
  pregunta "¿el exceso predice mejor que δ̂ crudo?" — si el resultado es limpio, sí).
- Soundbite "two-sided dissociation" (DINOv2-ImageNet sin exceso pero con ganancias;
  DBpedia con exceso sin ganancias) — ⚠️ ¿va a Discussion? Es honesto y desarma
  sobre-lecturas del diagnóstico; recomiendo sí, en Limitations.

---

## Orden de ejecución propuesto
1. ✅ (3-ago) C hecho: Table 1 regenerada, titulares sobreviven (HEADLINE_CHANGES.md) — todo
   lo demás cita números que salen de aquí. En paralelo: decisiones ⚠️ con coautores.
2. **10-17 ago:** HierarCaps pipeline + Fig. 3 nueva + figuras en base de exceso + exp12
   + prompts WordNet-definition.
3. **17-31 ago:** escritura secciones 1-5 (borrador Claude → revisión Javi, sección a sección).
4. **31 ago-7 sept:** ablations, apéndice, statements, estilo, compilación Overleaf.
5. **8-15 sept:** lectura coautores, congelar. 18: abstract. 24: notificación. 25: envío o retirada.
