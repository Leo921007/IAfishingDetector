# ETAPA 8C — Loop 100% VISUAL: trigger por splash (foam), audio eliminado

> El bot deja de usar audio (ruidoso). La mordida se detecta por el **chapuzón (foam)** sobre el corcho.
> El loop, el detector de mordida y la config cambian; el detector de corcho (YOLO) no se toca.

---

## 1. Nuevo trigger: foam

La mordida es un **splash** en el sitio del corcho. Métrica (calibrada en 8B, fuente única en `splash.py`,
idéntica al análisis offline): **foam = fracción de píxeles casi-blancos (V>200, S<60 en HSV) en un parche
×1.5 alrededor del bbox del corcho**. Baseline flotando ≈ 0.0000; en la mordida sube a 0.003–0.074;
separable hasta ~24 fps.

`bite_trigger.FoamBiteDetector(threshold, min_frames)` es stateful: `update(frame, bbox) -> (foam, fired)`
y **dispara una vez** cuando `foam > threshold` durante `>= min_frames` frames consecutivos; `reset()` tras
el loot.

## 2. Arquitectura: localizar-luego-foam (por qué)

Correr **YOLO en cada frame** para seguir el corcho es **caro en CPU** (en el equipo de juego el detector
va por onnxruntime CPU) y no aguanta ~30 fps. En cambio, calcular foam sobre un **parche fijo** es barato.
Por eso el loop (`main.LootLoop.run`):
- **LOCATE:** corre YOLO para ubicar el corcho (bbox → parche).
- **POLL:** samplea el foam a `poll_fps` sobre ese parche; **re-localiza** con YOLO cada `relocate_seconds`
  (el corcho deriva despacio) y confirma que sigue presente.
- Al disparar el foam → **loot** (clic derecho en el centro del corcho) → **park** → **recast**.

## 3. Watchdog y parkeo (conservados de la Etapa 7)

- **Watchdog:** si LOCATE no encuentra corcho, o se pierde en una re-localización, o pasa
  `max_wait_seconds` sin mordida → **recast**. Aviso (`WARNING`) tras N recasts seguidos sin corcho
  (deriva de cámara).
- **Parkeo:** el cursor se mueve a `mouse_park` (fuera de la ROI) tras cada loot y cada recast. El clic de
  loot es press+release sin arrastre (no rota la cámara).

## 4. Config nueva (`config.yaml: bite`) — afinar EN VIVO

```yaml
bite:
  foam_threshold: 0.005    # sube si hay falsos disparos por reflejos; baja si pierde mordidas
  foam_min_frames: 2       # más frames = menos falsos, más latencia
  poll_fps: 30             # tasa de sondeo del foam (8B: separable hasta ~24 fps)
  relocate_seconds: 0.5    # cada cuánto re-localizar el corcho con YOLO
  max_wait_seconds: 25     # safety: sin mordida en este tiempo -> recast
```

Parseo tolerante (`.get` con defaults). **El audio salió de la config** (sección `audio` y `AudioCfg`
eliminadas).

## 5. Audio eliminado del bot

Borrados: `audio_match.py`, `bench_audio.py`, `tests/test_audio_match.py`; `AudioRecorder` de
`platform_io.py`; la sección `audio` de la config. `replay.py` quedó **solo-detección**. El **legacy del
repo original** (`sound_validator.py`, `compare_*.py`) y los `Fishing_*.wav` **no se tocan** (tienen su
propio audio, ajenos al bot). El historial de git conserva todo por si hiciera falta revertir.

## 6. Verificación

- `pytest tests/` → **15 verdes**: trigger de foam (dispara/no-dispara/reset), watchdog/decide/park,
  detector, replay (solo detección), session. `analyze_splash.py --selftest` sigue **PASS** (foam idéntico,
  fuente única `splash.py`). `import main/bite_trigger/splash/replay` headless; grep sin audio en el bot.
- **Solo validable en Windows:** que el trigger de foam dispare en mordidas reales sin falsos por reflejos,
  y el ritmo real. Correr con `--record --log-level DEBUG` y revisar el log (foam por ciclo) para **afinar
  `foam_threshold`/`poll_fps`/`foam_min_frames`**.

## 7. Estado y siguiente paso

- ✅ Loop 100% visual con trigger de foam, watchdog/park conservados, audio eliminado; todo testeable
  headless; detector/dataset intactos.
- ⏸️ **Te toca probar en Windows** (pull + `python main.py --record --log-level DEBUG`) y **afinar los
  umbrales con el log**. Si hay falsos disparos o se pierden mordidas, ajustar `foam_threshold`/`foam_min_frames`
  y, si la captura no llega a la tasa, `poll_fps`.
