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
- **LOCATE:** **espera** a que el corcho aparezca tras castear, sondeando con YOLO a ~8 fps hasta
  `bite.locate_timeout` (~3 s; en WoW el corcho tarda ~1-1.5 s). Solo recastea si expira sin corcho — así
  no se auto-interrumpe el casteo. Devuelve el bbox → parche.
- **POLL:** samplea el foam a `poll_fps` sobre ese parche; **re-localiza** con YOLO cada `relocate_seconds`
  (el corcho deriva despacio). El detector corre a `conf 0.25` (el modelo nuevo no da FP a ese umbral) y el
  relocate **tolera hasta `relocate_tolerance` (3) fallos seguidos** antes de declarar "corcho perdido": un
  frame flaco del detector **no abandona el casteo** (se mantiene el último bbox y se sigue sondeando foam).
- Al disparar el foam → **loot** (clic derecho en el centro del corcho) → **park** → **recast**.

## 3. Watchdog y parkeo (conservados de la Etapa 7)

- **Watchdog:** si LOCATE no encuentra corcho, o se pierde en una re-localización, o pasa
  `max_wait_seconds` sin mordida → **recast**. Aviso (`WARNING`) tras N recasts seguidos sin corcho
  (deriva de cámara).
- **Parkeo:** el cursor se mueve a `mouse_park` (fuera de la ROI) tras cada loot y cada recast.
- **Clic de loot (timing, 8C-fix2):** `moveTo → move_settle → mouseDown → click_hold → mouseUp` (no
  instantáneo: WoW/DirectInput necesita registrar el cursor sobre el corcho); tras el clic se espera
  `loot_settle` ANTES de parkear, para que el loot se procese con el cursor todavía sobre el corcho. Sin
  arrastre intermedio (no rota la cámara). Afinar `move_settle`/`click_hold`/`loot_settle` en vivo.

## 4. Config nueva (`config.yaml: bite`) — afinar EN VIVO

```yaml
bite:
  foam_threshold: 0.005    # sube si hay falsos disparos por reflejos; baja si pierde mordidas
  foam_min_frames: 2       # más frames = menos falsos, más latencia
  poll_fps: 30             # tasa de sondeo del foam (8B: separable hasta ~24 fps)
  relocate_seconds: 0.5    # cada cuánto re-localizar el corcho con YOLO
  max_wait_seconds: 25     # safety: sin mordida en este tiempo -> recast
  locate_timeout: 3.0      # espera al corcho tras castear antes de recastear (8C-fix)
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
