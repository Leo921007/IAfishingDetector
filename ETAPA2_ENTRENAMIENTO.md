# ETAPA 2 — Entrenamiento del detector de corcho

> Reconstrucción del pipeline de entrenamiento (perdido en el repo original) y producción de un
> `best.pt` funcional con **Ultralytics YOLO11n**. **No se modificó la lógica del bot** (Etapa 3).

---

## 1. Entorno

| Componente | Detalle |
|---|---|
| SO | WSL2 (Ubuntu 24.04, kernel 6.6) |
| Python | 3.12.3 (del sistema; **sin pip/ensurepip**) |
| GPU | **NVIDIA RTX 4050 Laptop, 6 GB VRAM**, driver CUDA 12.8 |
| PyTorch | `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124` |
| Framework | `ultralytics==8.3.108` (+ `onnx`, `onnxruntime`) |
| Entorno | venv aislado en `.venv/` (no se tocó el Python del sistema) |

El sistema no traía `pip` ni `ensurepip` (falta el paquete `python3-venv`). Se resolvió **sin
sudo/apt**, bootstrapeando pip **dentro del venv**:

```bash
python3 -m venv .venv --without-pip
curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
.venv/bin/python /tmp/get-pip.py
.venv/bin/pip install -r requirements-train.txt
# Verificación CUDA:
.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# -> True NVIDIA GeForce RTX 4050 Laptop GPU
```

Dependencias fijadas en **`requirements-train.txt`** (UTF-8). El `requirements.txt` original
(operación del bot, UTF-16, deps de Windows) **no se tocó**.

---

## 2. Comando exacto de entrenamiento (reproducible)

Script versionado: **`train_corcho.py`** (semilla fija `seed=0`, `deterministic=True`).

```bash
./.venv/bin/python train_corcho.py
```

Equivalente por CLI:

```bash
./.venv/bin/yolo detect train model=yolo11n.pt data=data/corcho.yaml \
    imgsz=640 batch=16 epochs=150 patience=50 seed=0 deterministic=True \
    project=runs/corcho name=v1 exist_ok=True \
    hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 fliplr=0.5 flipud=0.0 \
    degrees=0.0 shear=0.0 perspective=0.0 mosaic=1.0 close_mosaic=10
```

**Augmentación** (justificada para dataset pequeño y objeto pequeño):
- `hsv_h/s/v`: variación de luz y color del agua (robustez ante iluminación).
- `fliplr=0.5`: espejo horizontal, válido para el corcho.
- `mosaic=1.0` + `close_mosaic=10`: ayuda con pocos datos; se desactiva en las últimas 10 épocas.
- **Desactivadas** las que deforman la apariencia: `flipud=0` (el corcho flota erguido),
  `degrees=0`, `shear=0`, `perspective=0`.

Config de datos: **`data/corcho.yaml`** (`nc: 1`, `names: [corcho]`, ruta absoluta al `dataset/`).
El `dataset/data.yaml` legacy (rutas Windows `C:/Users/...`) queda **obsoleto** y no se usa.

---

## 3. Dataset y sus límites reales

| | train | val |
|---|---|---|
| Imágenes | 187 | 20 |
| Labels (YOLO txt) | 187 | 20 |
| Bboxes | 188 | 20 |
| Clases | 1 (`corcho`) | 1 |

- **Integridad:** emparejado 1:1, sin huérfanos, sin labels vacíos, todos los bboxes en `0..1`.
- **Formato:** recortes pequeños de la región del corcho (~430×257, 443×242, 490×258 px), **no**
  capturas a pantalla completa. Bboxes pequeños (~0.05–0.08 normalizado → objeto pequeño).

**Límites honestos (lo que estas 187 imágenes NO dan):**
- Provienen de **una sola zona/sesión** → el modelo aprende ESE agua/iluminación/resolución.
- **No generaliza** a otras zonas, clima, día/noche, ángulos de cámara ni a pantalla completa.
- val de **20 imágenes** → métricas de **alta varianza y optimistas** (misma distribución que train).

---

## 4. Métricas obtenidas

Checkpoint `best.pt` = **época 74** (Ultralytics selecciona por *fitness* = 0.1·mAP50 + 0.9·mAP50-95).
Entrenamiento: 124 épocas (early stopping), ~4.7 min en RTX 4050.

| Métrica | Valor |
|---|---|
| **mAP50** | **≈ 0.86** |
| **mAP50-95** | **≈ 0.42** |
| Precision / Recall | ~0.74 / ~0.75–0.90 (punto de operación, ruidoso) |
| Inferencia | **~1.3 ms/img** (GPU) |

**Matriz de confusión @ conf=0.25:** TP=17, FP=8, FN=3 (sobre 20 instancias).
→ Buen recall pero **8 falsos positivos** a umbral bajo: en producción **subir el umbral**
(≈0.4–0.5) y/o **cruzar con la señal de audio** para descartarlos.

Detalle y caveats en **`reports/etapa2_metricas.md`**. Artefactos visuales (locales, no
versionados): `runs/corcho/v1/{confusion_matrix.png, results.png, PR_curve.png, val_batch0_pred.jpg}`.

**Verificación de carga + inferencia** (pasa):
```bash
.venv/bin/python -c "from ultralytics import YOLO; \
r=YOLO('models/corcho_detector/best.pt')('dataset/images/val/corcho_003.jpg'); \
print(r[0].boxes.xyxy)"
# -> 1 bbox, conf 0.556, clase 0
```

---

## 5. Ubicación del modelo

| Archivo | Uso |
|---|---|
| `models/corcho_detector/best.pt` | Pesos PyTorch (para Etapa 3). |
| `models/corcho_detector/best.onnx` | Export ONNX (baja latencia, Etapa 3). |
| `runs/corcho/v1/weights/best.pt` | Original generado por el entrenamiento. |

**No versionados** (`.gitignore`): `*.pt`, `*.onnx`, `models/`, `runs/`, `.venv/`, caches del
dataset. Se regeneran con el comando de §2. Decisión acordada: versionar solo scripts/configs/
métricas en texto, no binarios.

---

## 6. Plan de recolección de datos (requiere capturas nuevas tuyas — etapa posterior)

El cuello de botella **no es el modelo, es el dataset**. Para un detector robusto se necesitan
capturas nuevas en el juego cubriendo esta matriz (objetivo orientativo **≥ 1.500–2.000 imágenes**):

| Eje de variación | Cobertura objetivo |
|---|---|
| **Zonas** | ≥ 6–8 (distintos colores/texturas de agua: costa, río, lago, lava/zonas especiales, ciudad) |
| **Clima** | despejado, lluvia, niebla, tormenta |
| **Hora** | día, atardecer, noche (iluminación y reflejos distintos) |
| **Cámara** | varios ángulos/zoom y distancias del corcho |
| **UI/resolución** | resoluciones y escalas de UI que vayas a usar realmente |
| **Negativos** | frames **sin** corcho (agua, reflejos, NPCs) para reducir falsos positivos |

**Protocolo sugerido:**
1. Capturar con una herramienta de captura por zona/condición (reutilizable: `capturador.py`),
   guardando metadatos de zona/clima/hora en el nombre del archivo.
2. Etiquetar en formato YOLO (1 clase `corcho`); incluir **imágenes negativas** (label vacío).
3. Split estratificado train/val (≈85/15) **por zona** para que val mida generalización real.
4. Reentrenar con el mismo `train_corcho.py` (subir `epochs` y revisar augmentación).
5. Reportar mAP **por zona**, no solo global, para detectar zonas débiles.

> Esta recolección **requiere que tú captures y etiquetes nuevo material en el juego**; se aborda
> en una etapa posterior, no en la Etapa 2.

---

## 7. Estado y siguiente paso

- ✅ Entorno reproducible, dataset auditado, `best.pt` funcional (mAP50 ≈ 0.86) y export ONNX.
- ⏸️ **Detenido a la espera de aprobación para la Etapa 3** (adaptar la lógica del bot —
  `detect.py`/`main.py` — a este modelo Ultralytics/ONNX; umbral de confianza; integración con
  la detección de mordida).
