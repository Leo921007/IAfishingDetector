# ETAPA 5 — Mordida por audio robustecida + preparación de captura visual

> Cambia el discriminador de audio para ganar **especificidad** (menos falsas mordidas) y extiende el
> grabador para capturar **metraje de mordidas** que necesitará la mordida VISUAL (Etapa 6).

---

## 1. Problema (Etapa 4)

El match anterior comparaba **magnitudes FFT por similitud coseno**. Dos señales con energía repartida en
la banda 300–3000 Hz dan coseno alto **aunque el contenido sea distinto** → el ruido de banda ancha
puntuaba como el sonido genuino. Especificidad casi nula.

## 2. Nuevo enfoque: NCC de la envolvente de amplitud

Se compara la **envolvente de amplitud** (|Hilbert|, submuestreada ~1 kHz) mediante **correlación
cruzada normalizada (NCC)**. La envolvente captura la *forma del transitorio* del splash (onset agudo +
decaimiento), consistente entre grabaciones distintas del mismo sonido y **plana en el ruido**. Es un
discriminador **principista y de pocos parámetros** (banda, submuestreo, umbral), no un clasificador
entrenado → no se sobreajusta a las 3 referencias.

Se eliminó del camino de match la ganancia+saturación (el NCC es invariante a escala y el clipping
deforma la envolvente). La interfaz `match_audio(recording, fs, references, audio_cfg) -> (matched,
scores)` **no cambió**: `main.py` y `replay.py` siguen igual.

## 3. Separabilidad ANTES vs DESPUÉS (bench_audio.py)

Margen = `min(positivos) − max(negativos)` (positivo ⇒ separables). Positivos = refs cruzadas
(leave-one-out) + ref embebida en ruido; negativos = ruido blanco y ruido de banda.

| Discriminador | positivos | negativos | **margen** |
|---|---|---|---|
| Coseno-FFT (antiguo) | 0.767–0.787 | 0.758–0.761 | **+0.006** (al ras / solapados) |
| **NCC-envolvente (nuevo)** | **0.511–0.596** | **0.047–0.068** | **+0.442** |

Reproducir: `./.venv/bin/python bench_audio.py`.

## 4. Calibración del umbral

`config.yaml: audio.similarity_threshold` pasa de **0.5** (semántica coseno; estaba **por debajo** de los
negativos → falsos positivos) a **0.30** (NCC-envolvente). Margen explícito: negativos ≤ ~0.07, positivos
genuinos ≥ ~0.51 → 0.30 deja ~0.23 sobre el ruido y ~0.21 bajo la mordida genuina. El valor queda en
config; ajústalo si los datos reales lo piden.

## 5. Límites HONESTOS

- **Provisional:** la mejora se midió con **3 positivos** (refs) y **negativos sintéticos** (ruido). El
  margen real frente a **sonidos de gameplay** (música, hechizos, ambiente, otros NPC) **será menor** y
  hay que volver a medirlo con clips reales: `bench_audio.py --positives DIR --negatives DIR`.
- **El audio solo no basta:** incluso con buena especificidad, la confirmación robusta de la mordida
  necesita la señal **visual** (dip del corcho) y su **fusión** con el audio → **Etapa 6**.
- La cadencia de captura está limitada por la grabación de audio **bloqueante** (3 s); el grabador de
  frames corre en un hilo aparte para mitigarlo, pero el metraje de alta tasa fino es refinamiento de la
  Etapa 6.

## 6. Qué captura ahora el grabador

Con `--record` (o `config.yaml: session.enabled`), por ciclo en `sessions/<timestamp>/`:
- `cycle_NNNN_roi.png` — frame de la ROI en la decisión.
- `cycle_NNNN_audio.wav` — chunk de audio.
- **`cycle_NNNN_frames/`** — **secuencia** de frames de la ROI alrededor de la decisión (ring buffer a
  `session.frames.fps`, con `session.frames.max_frames` como cap de disco). Es el metraje para la Etapa 6.
- `events.jsonl` — scores de audio, detección, conteo de frames y desenlace por ciclo.

## 7. Cómo grabar una sesión de mordidas en Windows

```bat
:: en el equipo de juego, con la ROI ya calibrada (ver ETAPA4_BRINGUP.md)
.venv\Scripts\python main.py --record --log-level DEBUG
```

1. Pesca normalmente un rato; cada mordida deja `cycle_*_frames/` con el dip del corcho.
2. Copia `sessions/<timestamp>/` al repo en WSL2 y revísala: `./.venv/bin/python replay.py --session sessions/<timestamp>`.
3. Aparta clips de audio **positivos** (mordidas) y **negativos** (sonidos sin mordida) y re-mide la
   especificidad real: `./.venv/bin/python bench_audio.py --positives pos/ --negatives neg/`.

## 8. Estado y siguiente paso

- ✅ Discriminador NCC-envolvente (margen +0.44 vs ~0), umbral calibrado, interfaz estable, grabador con
  secuencia de frames; todo headless/testeable (9 tests verdes), `sound_validator.py` intacto.
- ⏸️ **Detenido a la espera de aprobación para la Etapa 6** (mordida VISUAL: detección del dip del corcho
  en la secuencia de frames + **fusión audio/visual**), que requiere el metraje real que ahora se puede
  grabar.
