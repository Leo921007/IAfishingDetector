# ETAPA 8A — Caracterización de la mordida visual (dip del corcho)

> Análisis offline para calibrar el futuro trigger **100% visual**. **No** toca el loop. Conclusión
> adelantada: con los datos actuales **el dip NO es trackeable a 10 fps** y la señal de desaparición está
> **contaminada por el loot** → hace falta una **captura dedicada** antes de fijar el umbral (8B).

---

## 1. Método

`analyze_dip.py` corre el detector **nuevo** (`best_zona.onnx`, conf 0.25) **frame a frame** sobre las
secuencias de cada ciclo (`sessions/*/cycle_*_frames`, 40 frames @10 fps ≈ 4 s) y registra por frame
`(detectado, center_y, conf, bbox)`. Por ciclo mide:
- **magnitud del dip** = máx `(center_y − mediana móvil reciente)` hacia abajo (y crece hacia abajo);
- **vanish**: rachas sin detección (longitud, si es de cola, si hay desaparece-reaparece);
- `detect_frac`.

Se compara **`recogido`** (mordida) vs **`esperando`** (corcho presente sin mordida = **control**).
Sesión principal: `20260602_100757` (loop post-Etapa 7). Salidas (gitignored): `dataset_zona/dip_analysis/`
(`per_frame.csv`, `per_cycle.csv`, PNGs de QA).

## 2. Resultados (mediana / p25 / p75 / máx)

| outcome | n | max_dip_px | longest_vanish | trailing_vanish | detect_frac | mid_vanish |
|---|---|---|---|---|---|---|
| **recogido** (mordida) | 69 | **1.2** / 0.9 / 1.9 / 21.5 | 12 / 10 / 14 / 40 | **12** / 10 / 14 / 40 | 0.70 | 15/69 |
| **esperando** (control) | 20 | **1.2** / 0.9 / 1.4 / 134 | 6 / 2 / 6 / 8 | **0** / 0 / 0 / 0 | 0.80 | 3/20 |

(Apoyo era-audio: `corcho_no_detectado` n=49 bimodal; `sin_sonido` n=12 = solo agua, detect_frac 0.)

**Lectura:**
- **La caída de `center_y` NO discrimina:** mediana **1.2 px en ambos** = nivel de *jitter* del detector.
  No hay un descenso vertical trackeable; el corcho jitterea ±2 px y luego **desaparece** (ver PNGs
  `cy_*_recogido` vs `cy_*_esperando`).
- **El único discriminador es `trailing_vanish`:** recogido **med 12 frames** (~1.2 s desaparecido al
  final) vs esperando **0**. Es decir, la firma "visible" es la **desaparición**, no el hundimiento.

## 3. El gran caveat (honestidad)

El `trailing_vanish` **está confundido con el loot**, no es la mordida: en el loop de la Etapa 7 el
snapshot del buffer se toma **~2.3 s DESPUÉS del loot** (`delay_after_click` + recast). Así que la cola
sin detección de los `recogido` es muy probablemente **el corcho ya recogido/consumido**, no el dip de
la mordida. El control `esperando` (sin loot) no desaparece → la diferencia es casi tautológica
("hubo loot" vs "no hubo loot").

Por tanto, con estos datos **no se puede fijar el umbral del dip**:
- El dip como *deslizamiento de center_y* **no se ve a 10 fps** (señal ≈ jitter de 1 px). El hundimiento/
  salpicón es **más rápido que 10 fps**: el corcho pasa de presente a ausente sin frames intermedios de
  caída.
- La **desaparición** sí es señal, pero aquí está sesgada por el loot; y a 10 fps una desaparición real
  de mordida (~2-5 frames) **se solapa** con el *flicker* del detector en `esperando` (longest_vanish med 6).

## 4. Regla candidata (PROVISIONAL, a validar en 8B)

- **Descartado:** `center_y` sube > D px en ≤ K frames. No usable: D real ≈ jitter (1-2 px).
- **Candidato:** **trigger por desaparición** — el corcho, tras estar **establemente presente**,
  **deja de detectarse durante ≥ V frames** (salpicón/hundimiento). `V` **no se puede fijar aún** con
  datos sucios; a 10 fps un `V` que separe del flicker (~6) implicaría ~0.8 s de latencia → demasiado
  lento para la ventana de mordida de Cata.

## 5. Conclusión y qué capturar en la 8B

- **10 fps es insuficiente** para caracterizar el instante de la mordida: hay que muestrear **5-10×**
  más rápido (≈30-60 fps) para ver si existe un dip de 2-3 frames antes del salpicón y para separar la
  desaparición real del flicker.
- Los **buffers de 4 s post-loot no sirven** para esto (sesgo del loot). Se necesita una **captura
  dedicada**: continua, a más fps, y **etiquetando el instante de la mordida independientemente del
  loot** (p.ej. una pasada de pesca manual donde se marque cuándo pica, o registrar el frame de
  desaparición sin que medie un clic de loot).
- **Recomendación 8B:** primero **instrumentar una captura rápida y sin sesgo de loot** (modo de
  grabación dedicado), volver a medir `V` (y si aparece un dip a alta tasa, `D/K`), y recién entonces
  fijar el umbral del trigger visual. Adoptar hoy una regla de desaparición con los datos actuales sería
  calibrar sobre el artefacto del loot, no sobre la mordida.

## 6. Estado y siguiente paso

- ✅ `analyze_dip.py` + CSVs + PNGs de QA; firma medida (dip ≈ jitter; señal = desaparición, pero
  sesgada por el loot). Nada del loop/detector/dataset/audio tocado.
- ⏸️ **Decisión/siguiente (8B):** capturar a alta tasa sin sesgo de loot y re-medir antes de implementar
  el trigger visual. Si querés, en 8B preparo ese modo de captura dedicado.
