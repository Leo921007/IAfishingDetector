# ETAPA 3 — Inferencia reconectada y lógica de detección corregida

> Reconecta el bot al modelo de la Etapa 2 (ONNX), corrige el bug de interacción y deja la ruta de
> detección **importable y testeable offline en WSL2**. **No** se reentrena ni se reescribe la
> detección de mordida por audio.

---

## 1. Nueva ruta de inferencia

Antes (roto): `main.py`/`detect.py` cargaban el modelo con `torch.hub.load(..., 'yolov5', source='local')`
sobre la carpeta `yolov5/` **vacía**. Ahora:

```
config.yaml ──> config.py ──┐
                            ├─> corcho_detector.CorchoDetector  (ONNX, numpy, cv2)   ← ruta de DETECCIÓN (headless)
models/corcho_detector/     │        ▲
  best.onnx ────────────────┘        │ usa
                                     │
main.py (loop en vivo) ──> platform_io.ScreenCapturer / InputController  (mss, pyautogui, keyboard)
                       └─> funciones de audio (FFT, conservadas)         (sounddevice, scipy, pydub)
```

- **`corcho_detector.py`**: clase `CorchoDetector` sobre **onnxruntime**. Pre/post-proceso propio:
  letterbox a 640, salida YOLO11 `[1,5,8400]` (cx,cy,w,h,score; 1 clase), filtro por confianza,
  **NMS** (`cv2.dnn.NMSBoxes`) y reescalado a coordenadas de la ROI. Solo importa numpy/cv2/onnxruntime.
- **`platform_io.py`**: `ScreenCapturer` (mss) e `InputController` (pyautogui+keyboard) con **imports
  perezosos** → importar el módulo no exige display.
- **`main.py`**: loop reescrito que usa config + detector + adaptadores. La lógica de audio se conserva
  tal cual, solo parametrizada desde la config.

## 2. Decisión de runtime: ONNX + onnxruntime

Elegido frente a Ultralytics `best.pt` porque en el equipo de juego es **ligero (sin torch, ~2.5 GB
menos)**, de **menor latencia y arranque**, y deja la dependencia de inferencia mínima. Coste: implementar
letterbox + NMS a mano (hecho, compacto y cubierto por test). El `best.pt` sigue disponible para reentrenar/
exportar.

## 3. Reconciliación captura ↔ entrenamiento (ROI)

El modelo se entrenó sobre **recortes** de la región del corcho (~430×257), no sobre frames completos.
La captura del bot **ya era por región** (`mss.grab(region)`), así que la solución es una **ROI
configurable** en `config.yaml` (`roi: left/top/width/height`) que **calibras una vez** al encuadre de
tus capturas, en lugar de dibujarla a mano en cada arranque (se eliminó la selección Tkinter).

> **Honestidad:** la ROI resuelve el desajuste *pantalla-completa → región*, **pero no** un cambio fuerte
> de zoom/composición respecto al dataset. Si tu encuadre operativo difiere mucho del de entrenamiento, la
> detección caerá y habrá que **reentrenar con frames completos / nuevos encuadres** (tarea de datos de la
> Etapa 4). Además, el dataset es de **una sola zona** (ver Etapa 2): no esperes robustez entre zonas/clima.

## 4. Fix del bug de interacción

En WoW el corcho se recoge con **clic derecho**, no izquierdo. La versión original hacía
`pyautogui.click()` (izquierdo). Corregido: `config.yaml` → `input.loot_button: right`, aplicado en
`InputController.move_and_click(..., button=...)`.

## 5. Configuración centralizada

Todo en **`config.yaml`** (cargado por `config.py`, rutas relativas a la raíz, sin rutas Windows):
umbral de confianza (**0.5** por defecto, alto por la tasa de FP de la Etapa 2), IoU, imgsz, **ROI**,
keybind de lanzamiento (`2`), botón de loot, delays y parámetros de audio. Se **eliminó** el
`dataset/data.yaml` legacy (rutas `C:/Users/LEONARDO/...`, ya sustituido por `data/corcho.yaml`).

## 6. Verificación offline (sin juego, sin display, sin audio)

```bash
# CLI: imprime bbox + confianza sobre imágenes guardadas
./.venv/bin/python detect_offline.py --source dataset/images/val

# Pruebas
./.venv/bin/python -m pytest tests/ -v
```

Resultado obtenido (conf≥0.5): **14 detecciones en 20 imágenes** de validación (a umbral alto el recall
baja respecto a conf 0.25, como se documentó en la Etapa 2). `pytest`: **2 passed** — el detector se
importa **sin** mss/pyautogui/keyboard/sounddevice y detecta corcho en imágenes de `val`.

## 7. Qué queda por validar SOLO en Windows (equipo de juego)

La ruta de detección está verificada en WSL2, pero **lo siguiente requiere el juego en Windows** y no se
puede comprobar aquí:
- Captura real de la ventana del juego vía `mss` con la ROI calibrada a tu resolución/UI.
- Inyección de teclado (`keyboard`, requiere permisos) y ratón (`pyautogui`) sobre la ventana de WoW.
- **Recogida efectiva del pez** con clic derecho + autoloot, y los timings de la ventana de mordida.
- Detección de mordida por **audio de micrófono** (depende del audio del juego saliendo por altavoces).

**Dependencias de operación (Windows):** además de las de inferencia, instalar `onnxruntime`, `mss`,
`pyautogui`, `keyboard`, `sounddevice`, `pydub`, `scipy`. No se modificó el `requirements.txt` original
(UTF-16) en esta etapa.

## 8. Estado y siguiente paso

- ✅ Inferencia ONNX reconectada y verificada offline; clic derecho; config centralizada; ruta de
  detección headless y testeada; rutas Windows eliminadas; sin `torch.hub`/`yolov5` en el código.
- ⏸️ **Detenido a la espera de aprobación para la Etapa 4** (p. ej.: recolección de datos multi-zona y
  reentrenamiento, robustez de timings de la ventana de mordida, humanización del input, o validación en
  Windows). No se ha tocado la lógica de audio ni `sound_validator.py`.
