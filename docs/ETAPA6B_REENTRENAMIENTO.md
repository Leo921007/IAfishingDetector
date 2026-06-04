# ETAPA 6B — Reentrenamiento con la zona y comparación viejo vs nuevo

> Fine-tune del detector con el dataset etiquetado de la zona del usuario y comparación **honesta**
> contra el modelo anterior. **El modelo nuevo NO se promueve automáticamente** — lo decides tú con
> estos números.

---

## 1. Split por ciclo (anti-fuga)

Los frames de un mismo ciclo son casi idénticos: si caen en train **y** val, las métricas se inflan.

- **Provenance verificada:** `extract_zona.py --manifest-only` reconstruyó la selección determinista y
  escribió `_manifest.csv`. Verificación: **212/212 frames, hash-match total → PASS**.
- **Split por ciclo** (`build_zona_dataset.py`, seed 0, ~18 % de ciclos a val, estratificado pos/neg):
  - 72 ciclos → **59 train / 13 val**, **sin solape** (aserción).
  - train de zona: **173 frames** (91 pos, 82 neg) **+ dataset original (187)** = 360 imágenes.
  - val de zona: **39 frames** (21 pos, 18 neg). `val` = solo zona nueva (mide generalización in-domain).
- El **dataset original no se tocó** (additive). `data/corcho_zona.yaml` fusiona original+zona en train.

## 2. Reentrenamiento

`train_corcho_zona.py`: fine-tune de **YOLO11n desde `models/corcho_detector/best.pt`** en **GPU
(RTX 4050)**, `epochs=80 patience=20 seed=0`, misma augmentación que la Etapa 2 (sin
flipud/rotación/shear). **Early stopping en la época 32; mejor en la época 4** (converge enseguida: la
zona es fácil para el modelo una vez que la ha visto). Modelo nuevo en
`runs/corcho/zona_v1/weights/best.pt`, exportado a **`models/corcho_detector/best_zona.onnx`**
(separado; **`best.onnx` quedó intacto**, md5 verificado).

## 3. Comparación viejo vs nuevo (mismo val de zona)

| modelo | mAP50 | mAP50-95 | P | R | **FP@0.25 en negativos** |
|---|---|---|---|---|---|
| **VIEJO** (`best.pt`) | 0.063 | 0.022 | 0.150 | 0.286 | **16/18 imgs · 49 cajas** |
| **NUEVO** (`zona_v1`) | **0.989** | **0.616** | 0.954 | 0.995 | **0/18 imgs · 0 cajas** |
| Δ | +0.926 | +0.594 | — | — | **49 → 0 cajas espurias** |

Reproducir: `./.venv/bin/python -m tools.compare_zona`.

## 4. Conclusión HONESTA

**Mejora clara y grande en la métrica del problema real.** El modelo viejo prácticamente **no
funcionaba** en esta zona (mAP50 0.06) y disparaba falsos positivos en **16 de 18** frames de agua/espuma
(49 cajas) — exactamente lo que se veía en vivo. El nuevo localiza el corcho (mAP50 0.99) y **no dispara
ni un FP** en los negativos del val.

**Pero con cautela:**
- El **val es chico** (21 positivos / 18 negativos) y de **una sola zona** → mAP50 0.99 es **optimista
  e in-domain**; **no** refleja robustez en otras zonas/clima/hora.
- El split por ciclo evita la fuga obvia, pero 13 ciclos de val siguen compartiendo agua/encuadre con los
  de train → el número alto es esperable.
- Para robustez general hace falta **más variedad** (recolección futura). Esto arregla **tu zona**, no el
  caso general.

**Veredicto:** mejora real para la zona del usuario; recomendable promover el modelo **para operar en
esta zona**, asumiendo que fuera de ella habrá que recolectar más datos.

## 5. Cómo promover el modelo (TÚ decides — no ejecutado)

La operación carga `models/corcho_detector/best.onnx` (config.yaml). Para adoptar el nuevo:

```bash
# respaldo del viejo y promoción del nuevo (solo si decides adoptarlo)
cp models/corcho_detector/best.onnx models/corcho_detector/best_pre_zona.onnx
cp models/corcho_detector/best_zona.onnx models/corcho_detector/best.onnx
# (opcional) usar el nuevo .pt como base de futuros fine-tunes:
cp runs/corcho/zona_v1/weights/best.pt models/corcho_detector/best.pt
```

Todo esto queda gitignored (pesos). No lo hago yo: lo decides con la tabla de arriba.

## 6. Estado y siguiente paso

- ✅ Manifest verificado, split por ciclo sin fuga, fine-tune en GPU, `best_zona.onnx` exportado,
  `best.onnx` intacto, comparación honesta (mAP 0.06→0.99, FP 49→0).
- ⏸️ **Decisión tuya:** promover `best_zona.onnx`→`best.onnx` (§5) y probar en vivo en tu zona.
- ⏳ Fuera de alcance aquí: recalibración del audio (necesita que marques qué ciclos tuvieron mordida
  real) y recolección multi-zona para robustez general.
