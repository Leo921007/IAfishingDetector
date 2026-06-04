# ETAPA 6A — Dataset de la zona del usuario (preparación para reentrenar)

> El modelo actual no generaliza a la zona del usuario (1 mordida buena de muchas; FP sobre la espuma).
> Esta etapa **extrae** un dataset de la zona real desde las sesiones grabadas y deja todo listo para que
> **tú etiquetes** y luego (Etapa 6B) se reentrene. **No se reentrena aquí.**

---

## 1. Método de extracción (`extract_zona.py`)

- **Sesiones usadas** (misma escala = mismas dimensiones de frame **387×748**, ROI 748×387):
  `sessions/20260601_222727/` (51 ciclos) + `sessions/20260601_221053/` (21 ciclos) = **72 ciclos**.
  Se excluyen `20260601_131345` (50×80, stub) y `_muestra` (258×490, sintética): distinta escala.
- **Selección por ciclo** (de los 40 frames): el **frame del “dip”** (máximo movimiento inter-frame) +
  frames temporalmente separados; **dedup intra-ciclo** (average-hash 16×16) para no repetir agua estática.
- **Resultado:** **212 frames** en `dataset_zona/raw/` (`frame_0000.jpg`…), dentro del objetivo 150–250.
- **Propuestas:** el detector ONNX a conf 0.10 escribió una `.txt` YOLO por frame (**las 212**). A esa conf
  el modelo dispara sobre la espuma en todos → **propuestas POCO FIABLES**, hay que corregirlas.
- **Montaje:** `dataset_zona/montage.jpg` (rejilla con outcome + marca DIP) para juzgar calidad.
- `dataset_zona/` está **gitignored** (pesado, local para etiquetar). Reproducir:
  `./.venv/bin/python -m tools.extract_zona`.

## 2. ROI anclada

`config.yaml: roi` quedó en **left=586, top=126, width=748, height=387**, derivado de la sesión real
(dimensiones de los frames + `click − centro_bbox` de los eventos, unánime). Así
captura/entrenamiento/inferencia trabajan a la **misma escala** del corcho. **Este encuadre 748×387 es el
de operación**: si recapturas, usa esta ROI.

## 3. Cómo etiquetar (lo hace el usuario)

Herramienta: **labelImg** (local, formato YOLO) o **Roboflow** (alternativa web).

```bash
pip install labelImg        # en tu entorno; o usa el binario
labelImg dataset_zona/raw   # cambia el formato de salida a "YOLO"
```

Reglas (clase única **`corcho`**, id 0):
1. **Etiqueta SOLO el corcho real** (la boya). Ajusta la caja; **borra las propuestas que sean FP**.
2. **Deja la espuma SIN etiquetar.** No marques espuma ni reflejos.
3. **Incluye negativos:** en los frames de **solo agua/espuma sin corcho**, **borra la `.txt`** (una
   imagen sin label = fondo). Estos negativos son **lo que mata los falsos positivos** — deja unos cuantos.
4. Las propuestas `.txt` son una ayuda, **no** una verdad: revisa todas.

Pista: los ciclos `sin_sonido` (sin mordida) son buenos candidatos a negativo; los `recogido` /
`corcho_no_detectado` suelen contener corcho.

## 4. Reentrenamiento (Etapa 6B — tras etiquetar)

```bash
# 1) arma el split (val = zona nueva, incluye negativos) y el data yaml
./.venv/bin/python -m tools.build_zona_dataset
# 2) fine-tune de YOLO11n desde el modelo actual (best.pt)
./.venv/bin/python -m tools.train_corcho_zona
# 3) compara viejo vs nuevo en la zona (el dataset original queda intacto para comparar)
```

`build_zona_dataset.py` escribe `data/corcho_zona.yaml` con `train: [original, zona]` y
`val: [zona]`. `train_corcho_zona.py` reentrena desde `models/corcho_detector/best.pt` con la
augmentación de la Etapa 2 (sin flipud/rotación/shear). **Aún no ejecutados.**

## 5. Notas HONESTAS sobre este dataset

- **Pequeño y muy correlacionado:** 212 frames de **72 ciclos de una sola zona/sesión**, agua **oscura de
  baja textura**. Aunque se submuestreó (≈3/ciclo) y se dedup-licó, los frames comparten fondo → el modelo
  puede memorizar el fondo en vez de aprender el corcho. La diversidad real es baja.
- **Escala distinta del original:** el dataset original es 420×265; la zona es 748×387 → el corcho ocupa
  una fracción distinta del frame. Fusionar añade datos pero mezcla escalas; el **val es solo de la zona
  nueva** para medir lo que importa (generalización in-domain).
- **Los negativos son clave:** sin frames de agua/espuma sin label, los FP sobre la espuma seguirán.
- **Mejora esperable pero limitada:** este fine-tune debería reducir FP y mejorar la zona del usuario,
  pero **no** dará robustez general. Para eso hace falta **más variedad** (otras zonas/clima/hora), que es
  recolección futura. Mide siempre **viejo vs nuevo** antes de adoptar el modelo.

## 6. Estado y siguiente paso

- ✅ 212 frames + propuestas + montaje en `dataset_zona/raw/`; ROI anclada; scaffolding de 6B listo
  (no ejecutado); dataset original intacto.
- ⏸️ **Te toca etiquetar** `dataset_zona/raw/` siguiendo §3. Cuando termines, lanza la **Etapa 6B**
  (build + train) para reentrenar y comparar.
