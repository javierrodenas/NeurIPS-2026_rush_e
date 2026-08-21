# Tareas que requieren decisión/acción de coautores (21-ago)

## 1. Figura 1 (caverna, `fig_intro_concept.pdf`) — artwork manual, framing viejo dentro del dibujo
El texto incrustado dice: "How is the reality?", paneles "Hyperbolic" / "Euclidean", nodo raíz "Reality".
Con la espina nueva (forma-vs-contenido) ese lenguaje es el que hemos retirado.
**Opción A (relabel, quien tenga el fichero fuente — draw.io/Figma/Illustrator):**
- "How is the reality?" → "Is the class geometry tree-like or flat?"
- "Hyperbolic" → "Tree-like (low-distortion embedding in the Poincaré disk)"
- nodo "Reality" → eliminar o renombrar "root"
**Opción B (quitarla):** el mock review ya señaló que ocupa media página y aporta poco;
quitarla libera espacio para la tabla resumen en página 1-2. El caption actual del .tex
ya está reescrito para la opción A; si se elige B, aviso y reajusto la intro.

## 2. Figura 3 (`fig_tree_visualization.pdf`) — script no encontrado
El generador NO está en scripts/ (las 9 versiones de plot_euclid_vs_hyperbolic*.py
producen otra figura y ninguna contiene "Reality"). Se necesita:
- Localizar el script/notebook que la generó (¿Eduardo? ¿notebook local?), y
- Regenerarla con el nodo central "Reality" → "root" (único cambio necesario;
  los títulos "Poincaré disk" son correctos y se quedan).
Mientras tanto el caption del .tex ya la escopa como ilustración; si el script no
aparece, alternativa: Claude la reconstruye aproximada desde los features cacheados
(decisión de Javi, tras lo aprendido: no se sustituye ninguna figura sin OK explícito).

## 3. Título — elegir entre:
(a) "Form, not Content: Calibrated Measurement of Tree-Like Geometry in Foundation Models" (actual)
(b) "What Survives Calibration: Tree-Like Form, Model-Specific Content in Foundation Model Geometry"

## 4. Recuento del panel (TODO-coauthors en §3 del .tex)
Censo calibrado: 12 visión × 6 datasets + 16 texto = 28 modelos; + 4 contrastivos
visión solo-ImageNet (OpenCLIP-B/L, MetaCLIP-B/L) = 32 analizados. El "30 (14 vision)"
del paper de NeurIPS no cuadraba con sus propias tablas. Confirmar cómo contar y
que la tabla A.1 del apéndice es la lista canónica.

## 5. Perfiles de OpenReview activos antes del 18-sept (registro del abstract;
no se pueden añadir autores después de esa fecha).
