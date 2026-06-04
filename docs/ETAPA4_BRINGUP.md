# ETAPA 4 — Bring-up en Windows e instrumentación

> Hace el bot **observable** (logging estructurado) y añade la **infraestructura de validación**
> (grabador de sesión, replay offline, barrido de umbral) para una primera corrida controlada.
> **No** se cambian los algoritmos de detección ni de audio.

---

## 1. Entorno de ejecución en Windows (equipo de juego)

En Windows sí hay `pip`, display y audio (a diferencia del WSL2 de desarrollo).

```bat
cd C:\ruta\a\IAfishingDetector
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip

:: Inferencia (ONNX, sin torch) + operación (captura/input/audio)
.venv\Scripts\pip install onnxruntime==1.20.1 opencv-python numpy==2.1.1 scipy pyyaml ^
                          mss pyautogui keyboard sounddevice
```

Notas:
- El runtime de inferencia es **ONNX** → **no necesita torch** en el equipo de juego.
- El audio ya **no requiere `pydub` ni `ffmpeg`** (se carga con `scipy.io.wavfile`).
- `keyboard` puede requerir ejecutar la terminal **como administrador** para inyectar teclas.
- El modelo `models/corcho_detector/best.onnx` debe estar presente (se genera en la Etapa 2; está
  gitignored, cópialo al equipo de juego).

## 2. Correr el bot

```bat
.venv\Scripts\python main.py                 :: corrida normal
.venv\Scripts\python main.py --log-level DEBUG   :: ver scores de audio por ciclo
.venv\Scripts\python main.py --record        :: además, grabar la sesión
```

## 3. Observabilidad (logging)

Cada ciclo se registra con timestamp en **consola y `logs/pesca.log`** (rotativo, gitignored):
lanzamiento, ventana de escucha, **similitudes de audio por referencia**, resultado de detección
(bbox/conf o ninguno), coords de clic y desenlace. Nivel configurable (`config.yaml: logging.level`
o `--log-level`). Una corrida real deja un log analizable.

## 4. Grabador de sesión (captura de datos reales)

Con `--record` (o `config.yaml: session.enabled: true`) se persiste por ciclo en
`sessions/<timestamp>/` (gitignored):
- `cycle_NNNN_roi.png` — frame de la ROI capturado.
- `cycle_NNNN_audio.wav` — chunk de audio del momento de la decisión.
- `events.jsonl` — una línea por ciclo con scores de audio, detección, desenlace.

Es la materia prima para mejorar la mordida y **ampliar el dataset** (Etapa 5+).

## 5. Replay offline (validar el loop sin el juego, en WSL2)

```bash
# sintetizar una sesión-muestra mínima y reproducirla
./.venv/bin/python -m tools.replay --make-sample sessions/_muestra
./.venv/bin/python -m tools.replay --session sessions/_muestra
# reproducir una sesión real grabada en Windows (cópiala a sessions/)
./.venv/bin/python -m tools.replay --session sessions/<timestamp>
```

El replay pasa los frames y wavs por **la misma** lógica de detección y match de audio, sin I/O en
vivo. Útil para depurar desenlaces y comparar umbrales con datos reales.

## 6. Afinado del umbral de confianza

`./.venv/bin/python -m tools.tune_threshold` barre `conf_threshold` sobre `dataset/images/val` (con GT) e
imprime precisión/recall/F1. Resultado actual:

| conf | TP | FP | FN | precision | recall | F1 |
|------|----|----|----|-----------|--------|----|
| 0.25 | 18 | 5  | 2  | 0.783 | **0.900** | 0.837 |
| 0.30 | 17 | 5  | 3  | 0.773 | 0.850 | 0.810 |
| 0.35 | 17 | 5  | 3  | 0.773 | 0.850 | 0.810 |
| 0.50 | 13 | 1  | 7  | 0.929 | 0.650 | 0.765 |

**Recomendación:** en el loop la detección está *gated* por el audio (solo se busca el corcho tras una
mordida), así que interesa **recall alto** → **conf ≈ 0.30–0.35** (o 0.25). El valor por defecto actual
es **0.50** (conservador, recall 0.65). **Queda en `config.yaml` y no se fuerza**; cámbialo a mano si lo
decides. Recuerda que el `val` es de una sola zona → métricas optimistas.

## 7. Hallazgo: especificidad del audio (insumo para la Etapa 5)

El barrido instrumentado del match destapó que el algoritmo de audio (coseno de magnitudes FFT) tiene
**especificidad baja**: ruido de banda ancha puede superar el umbral 0.5 (la referencia genuina puntúa
más alto, pero el margen es estrecho). Esto explica falsos positivos de mordida. Mejorarlo
(p. ej. correlación temporal, plantilla espectral, captura por *loopback* del audio del juego en lugar
del micrófono) es trabajo de la **Etapa 5** — aquí solo se midió, no se cambió.

## 8. Checklist de primera corrida segura (Windows)

1. **ROI:** calibra `config.yaml: roi` al recuadro donde aparece el corcho, con un encuadre parecido al
   del dataset de entrenamiento (recortes ~430×257). Verifícalo con una sesión `--record` + `replay`.
2. **Keybind de Pesca:** `input.cast_key` (por defecto `2`) debe ser el slot del hechizo Pesca.
3. **Autoloot ON** en el juego (el bot recoge con **clic derecho**, no abre la ventana de loot).
4. **Ventana del juego** en primer plano y en la resolución para la que calibraste la ROI.
5. **Fail-safe de PyAutoGUI activo:** mover el ratón a una esquina aborta el bot. Ten a mano `Ctrl+C`.
6. Primera corrida con `--log-level DEBUG --record` para revisar el log y la sesión antes de confiar.

## 9. Qué sigue **solo validable en Windows**

La ruta de detección/replay/match está verificada en WSL2, pero requieren el juego en Windows:
- Captura real de la ventana (`mss`) con la ROI calibrada; inyección de teclado/ratón.
- **Recogida efectiva del pez** (clic derecho + autoloot) y los timings de la ventana de mordida.
- Detección de mordida por **audio de micrófono** y su tasa real de falsos positivos/negativos.

## 10. Estado y siguiente paso

- ✅ Logging estructurado, grabador de sesión, replay offline y barrido de umbral; ruta de
  detección/replay/match **importable headless**; I/O de plataforma aislado.
- ⏸️ **Detenido a la espera de aprobación para la Etapa 5** (candidatos: mejorar la especificidad del
  audio/mordida, robustez de timings, humanización del input, o recolección de datos multi-zona +
  reentrenamiento). No se tocó `sound_validator.py` ni los algoritmos.
