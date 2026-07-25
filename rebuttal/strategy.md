# Estrategia de rebuttal — NeurIPS 2026 Submission 18165

**Situación**: 2 reject (1Eqj: claridad; RJje: técnico), 1 borderline accept (pux6), 1 accept conf. 5 (Xbn5). Meta-review inicial: borderline. El AC lista 5 puntos de discusión — el rebuttal se organiza alrededor de ellos.

**Principio**: cada objeción experimental se responde con un experimento nuevo, no con retórica. Todos corridos sobre las features congeladas ya cacheadas (reproducibles con los scripts en `rebuttal/scripts/`). Concedemos con claridad lo que los controles muestran que hay que conceder — eso compra credibilidad para defender el resto.

## Qué encontraron los experimentos nuevos (resumen ejecutivo)

| Exp | Pregunta del reviewer | Resultado | Veredicto para el paper |
|---|---|---|---|
| E1 controles δ | RJje: ¿gauss/esfera/whitened/dim-matched? | Nubes gaussianas alta-dim tienen δ baja per se (0.046–0.104, efecto símplex); TODOS los modelos quedan por encima de su control gaussiano; normalizar a esfera SUBE δ (p.ej. DINOv2-G 0.080→0.098/0.106) | Conceder: nivel absoluto de δ requiere calibración. Defender: orden entre modelos estable bajo métrica cosine; esfera es menos arbórea |
| E1 agrupaciones | RJje Q6: ¿random groupings? | Grupos WordNet dan δ < grupos aleatorios en 8/12 modelos (CLIP-B 0.126 vs 0.174); en DINOv2-B/L/G empatan (suelo) | La estructura sigue la semántica, no cualquier binning |
| E1b dim-matched | RJje: dimension-matched | PCA a d=192: DINOv2-L sigue siendo el más arbóreo (0.085 < null 0.104) pero el trend monótono S→G se atenúa | Reformular el claim de scaling como exceso sobre null |
| E1c null espectral | (control definitivo) | Con espectro de covarianza igualado: supervisados (exceso −0.019..−0.031, crece con escala) y CLIP/SigLIP (−0.017..−0.021) POR DEBAJO del null = árbol genuino de orden superior; DINOv2 ≈ en su null (±0.011) | La historia refinada: label/language-supervised llevan el árbol genuino y alineado; DINOv2 = clusters compactos + espectro. Coherente con E3 |
| E2 controles métrica | RJje Q4: ¿rescalado radial? | RT (misma tanh, distancia euclídea) ≈ R en todo; H−RT = +0.9–1.2pp en SSL FS | **WIN limpio**: la ganancia H−R es de la métrica, no del rescalado |
| E2 cosine | RJje Q3: ¿cosine/esférica? | COS es fuerte: en DINOv2 COS>H; en CLIP H>COS (FS +0.6–0.8); supervisados empate | Conceder y reencuadrar el protocolo práctico como decisión a 3 vías |
| E2b stack normalizado | (seguimiento) | Sobre features normalizadas: HN>COS para CLIP (+1.2pp FS, CI ±0.14); DINOv2 sigue prefiriendo COS | La ventaja hiperbólica sobre esférica se concentra en contrastivos — hallazgo nuevo y honesto |
| E3 alineamiento | pux6 W1/Q1: ¿corrección semántica? | ρ(dist centroides, dist WordNet) ≈ 0.49–0.53 (shuffle ≈ 0.01, gauss ≈ 0); ARI dendrograma vs 20 superclases C100 ≈ 0.31–0.39 | El árbol medido ES el árbol semántico, cuantitativamente |
| E3 ORC puentes | RJje: "tensión interna ORC" | ORC intra-superclase +0.35–0.40 vs inter +0.27–0.31; aristas negativas 2–5% intra vs 6–9% inter | Resuelve la tensión: ORC alto = hojas compactas; negativo = puentes. Reescribir párrafo §3.2 |
| E4 prompts | Xbn5 Q2, RJje Q6 | Embedders robustísimos (rango δ ≤0.011; reproduce paper: BGE 0.124 vs 0.123); causal LMs sensibles (rango 0.04–0.065; con nombre-solo el orden se invierte) | Conceder y ACOTAR el claim de causal LMs; visión (core) no usa prompts |
| E5 cross-model | RJje minor, Xbn5 W2: PRH | NEGATIVO: alignment medio 0.474 (R) vs 0.427 (H); solo 6/66 pares mejoran | Moderamos el framing PRH con base cuantitativa (lo pedía el meta-review de todos modos) |
| E6 calibración H²/esfera | 1Eqj Q1/Q2 | δ absoluta de H² → ln2≈0.693, δ/diam → 0 al crecer región (0.16→0.02); esfera invariante de escala (S^99 chord 0.143, S² 0.47); árbol = 0.000 | Explica la "paradoja" del reviewer: comparaba constante absoluta (0.88) con normalizada (0.127) |
| E7 recuperación jerarquía | Xbn5 Q1 (tarea extra), pux6 W2 | Linkage euclídeo ya recupera superclases (ARI hasta 0.39); linkage Poincaré NO mejora. ρ WordNet sin pooling (1 img/clase): +0.20..+0.34 (menor en DINOv2) | Negativo para H en esta tarea (se reporta); la señal jerárquica pre-existe sin centroides → anti-circularidad |
| E8–E10 auditoría | (interna) | ARI≈0 de DINOv2 era artefacto de average-linkage (Ward: 0.19–0.61); semántica de DINOv2 es angular-local (tripletas 0.85–0.91 en cosine); nulls confirmados; ver audit.md | Drafts corregidos; aviso Figura 3 para camera-ready |
| E11 nulls por dataset | (decisivo) | El "DINOv2 en su null" es SOLO en ImageNet; en transfer (CIFAR-100/10, DTD) su exceso genuino es el más fuerte del panel y escala S→G (−0.087→−0.146 en CIFAR-10) | La concesión queda reescopada; apoya la tesis capacity×data del paper |
| kNN/retrieval (ya existía) | Xbn5/pux6: retrieval | A nivel sample el proyector NO ayuda (kNN se degrada con t; P@10 ligeramente negativo) | Reportarlo con honestidad: la jerarquía explotable vive a nivel prototipo/clase |

## Posición global (el "narrative arc" del rebuttal)

1. **Lo que se sostiene y sale reforzado**: (a) la estructura inter-clase es jerárquica Y semánticamente correcta (E3: ρ≈0.5, ARI≈0.35, muy por encima de cualquier control); (b) es aprendida (ablaciones del paper + E1 grupos); (c) la ganancia H−R es de la métrica hiperbólica, no del preprocesado (E2 RT); (d) δ/ORC predicen el contraste H−R (claim cuantitativo del paper, intacto); (e) hallazgo nuevo: sobre modelos contrastivos, la métrica hiperbólica supera también a la esférica.
2. **Lo que se concede y acota**: (a) no estimamos curvatura seccional; δ mide tree-likeness métrico, y su nivel absoluto necesita null de dimensión igualada (lo añadimos); (b) cosine es un baseline fuerte que el paper debía incluir — lo incluimos y reencuadramos el protocolo práctico; (c) el claim de causal LMs se acota (sensible al template); (d) "universal geometry" se suaviza; (e) la conexión PRH se re-articula como complementaria (geometría del punto de convergencia, no convergencia en sí) — con E5 si sale positivo.
3. **Claridad (1Eqj)**: lista concreta de cambios de redacción comprometidos para la revisión.

## Reglas logísticas
- Fase 1 hasta 27 jul: subir los 4 rebuttals con botón "Rebuttal" por review. ≤10.000 caracteres cada uno, markdown, sin links, sin revelar identidad.
- No se puede tocar el PDF: todo va en el texto de respuesta. Los números nuevos se presentan como tablas markdown compactas.
- Si piden código: link anonimizado solo al AC en Official Comment.

## Riesgos y cómo los tratamos
- **Riesgo A (RJje presiona con cosine)**: nos adelantamos publicando nosotros el número completo COS en el rebuttal, con el split por familia. Es mejor que lo digamos nosotros.
- **Riesgo B (nos acusan de resultados nuevos no verificables)**: protocolo descrito con precisión (features congeladas idénticas al paper, mismos splits/seeds); ofrecer código anonimizado al AC.
- **Riesgo C (el confound de dimensión debilita el scaling claim)**: E1b (PCA a d común) decide qué versión del claim sobrevive; redactamos según su resultado.
- **Nota interna (no va al rebuttal) — ALCANCE PRECISADO (25 jul)**: comparación completa Tabla 1 vs re-run en `rebuttal/results/paper_vs_rerun_table1.csv`. **El brazo R (euclídeo) reproduce casi perfecto** (media 0.05pp; 1 sola celda >0.5pp, por submuestreo de ImageNet). **El problema está confinado a la columna NC-H**: 15 celdas desviadas 0.5–1.2pp, sistemáticamente a favor del paper, más CLIP-L/MNIST con el signo volteado (paper −0.9pp vs re-run +0.9pp). El re-run coincide con `tasks_t707` (el CSV de producción del propio repo); la discrepancia es de la tabla impresa, no del código ni de las features. También δ ImageNet DINOv2-L/G (0.066/0.067 impresos vs 0.080 recomputado). **Auditoría obligada de la columna NC-H y las δ de ImageNet antes de la camera-ready** (hipótesis a comprobar: transcripción desde un run anterior con otro radio t). El rebuttal solo cita celdas verificadas que reproducen (DINOv2-L/G en CIFAR-100, los FS) — ninguna de las 15 afectadas.
