# Plan ICLR 2027

## Fechas (verificadas contra el CFP oficial)
- **18 sept 2026 AoE**: registro de abstract. Título + abstract reales (no placeholder). **No se pueden añadir autores después**; todos los coautores con perfil de OpenReview antes de esta fecha.
- **24 sept**: notificación NeurIPS. Reject → seguimos; Accept → retirar el abstract de ICLR antes del 25 (se borra sin rastro) y a camera-ready.
- **25 sept 2026 AoE**: deadline de paper completo.
- **Objetivo interno: paper terminado el 15 de septiembre.**

## Formato
- 9 páginas de texto principal en submission (10 en camera-ready). Referencias y apéndice ilimitados.
- Doble ciego estricto; `\iclrfinalcopy` comentado.
- Declaración de uso de IA/LLM **obligatoria**. Reproducibility y Ethics statements recomendados (no cuentan páginas).
- Plantilla oficial en `ICLR2027/` (bajada de media.iclr.cc, ago 2026).

## Título (compromiso público en OpenReview, respuesta a Xbn5)
"Tree-Like Class Geometry in Foundation Models: A Zero-Cost Diagnostic for Metric Selection"

## La reescritura = ejecutar lo ya comprometido en OpenReview
Fuente de verdad: rebuttal/ (respuestas + réplicas + audit.md). Cada sección del
esqueleto (`iclr2027_tree_geometry.tex`) lleva su lista COMMITTED en comentarios.
Los tres pilares del abstract: forma casi universal (69/72 vs nulls) · contenido
específico de cada modelo (decoupling, form-not-content) · protocolo de dos pasos
(cuándo salir de Euclídeo + qué métrica cosecha).

## Trabajo NUEVO imprescindible antes del 15-sept (no es solo texto)
1. **Regenerar Table 1 desde tasks_t707** (auditoría interna: columna NC-H con
   celdas infladas + sign flip CLIP-L/MNIST; celdas δ de ImageNet DINOv2-G
   .066→.080 y CLIP-L .108→.118). En NeurIPS nunca publicamos celdas afectadas;
   en ICLR la tabla nace limpia.
2. **Rehacer Figura 3** (la compacidad de los spokes era artefacto de la
   proyección PCA) o sustituirla por una visualización honesta.
3. Integrar los CSVs de rebuttal/results/ como tablas del apéndice.
4. Decidir si HierarCaps entra (mejor sí: era compromiso público y refuerza pux6-Q1).
5. exp12_curvature_sign.py (escrito, sin correr): correrlo y decidir si entra.

## Reparto orientativo
- Semana 10-17 ago: Table 1 regenerada + Fig. 3 nueva + HierarCaps pipeline.
- Semana 17-31 ago: secciones 1-5 escritas (Claude + Javi revisando).
- Semana 31 ago-7 sept: ablations, apéndice, statements, pasada de estilo.
- 8-15 sept: lectura de coautores, congelar. 18 sept: abstract. 25 sept: envío (si NeurIPS rechaza).
