# ETAPA 7 — Watchdog de recast + parkeo del mouse

> Robustez del loop (**uptime**): que un fallo de casteo no detenga al bot. **No** cambia el trigger
> (el audio sigue siendo ruido) — eso es etapa posterior.

---

## 1. El problema (stall)

En una corrida larga, tras ~30 ciclos el corcho dejó de caer dentro de la ROI (los casteos derivaron
fuera) y el bot quedó **95 ciclos girando sin recastear**, disparando sobre ruido de audio.

**Causa raíz:** la rama "audio OK pero corcho no detectado" del loop **no recasteaba**. Si el corcho ya
no estaba en la ROI, el audio (ruido) seguía matcheando, se veía agua vacía y el bot nunca relanzaba.

## 2. Watchdog de recast (Mejora A)

Ahora cada ~`watchdog_interval` (3 s, alineado con los chunks de audio) el loop hace **grab ROI + detect**
y decide (`decide(has_corcho, audio_matched)`):

| ¿corcho en ROI? | ¿audio matchea? | acción |
|---|---|---|
| **No** | — | **recast** (relanza "2") ← arregla el stall |
| Sí | Sí | **loot** (clic) + recast |
| Sí | No | **wait** (el corcho está, sin mordida) |

- **Aviso de deriva:** si recastea **N=`watchdog_warn_after` (5)** veces seguidas sin encontrar corcho,
  loguea un `WARNING` — señal de que la cámara derivó y los casteos caen fuera de la ROI (revisar ROI /
  posición de cámara).
- **Safety:** si hay corcho pero sin mordida durante más de `listen_window`, se fuerza un recast (corcho
  "muerto" que no desaparece).

Así **siempre hay un corcho presente** y un fallo aislado no detiene el casteo.

## 3. Parkeo del mouse (Mejora B)

`InputController.park(x, y)` mueve el cursor (sin clic) a `mouse_park` (config), un punto **fuera de la
ROI** (586,126,748×387; por defecto `{x:300, y:700}`). Se llama **tras cada loot y tras cada recast** para
que el cursor no tape el corcho ni interfiera con la captura.

## 4. Clic de loot limpio (Mejora C)

`move_and_click` = `moveTo(x,y)` instantáneo + `click(button='right')` = **press+release sin movimiento
intermedio**. Un right-click así **no arrastra ni rota la cámara** en WoW (la rotación requiere
click+drag). Ya era correcto; se confirma y se anota en el código. El parkeo posterior además aparta el
cursor enseguida.

## 5. Config nuevo (`config.yaml: input`)

```yaml
watchdog_interval: 3.0     # s entre sondeos (alineado con los chunks de audio)
watchdog_warn_after: 5     # avisar tras N recasts seguidos sin corcho
mouse_park: {x: 300, y: 700}   # punto fuera de la ROI (ajusta a tu pantalla)
```

`--record` y `--log-level` se mantienen.

## 6. ALCANCE / HONESTIDAD

- **Esto arregla el STALL (uptime), NO el trigger.** El audio sigue siendo **ruido**: su score no se
  correlaciona con mordidas reales (Etapa 5 lo dejó claro y provisional). Con el watchdog el bot ya no se
  cuelga, pero **el ritmo de capturas reales depende de la suerte del ruido** (un loot ocurre cuando el
  ruido matchea justo con un corcho presente).
- El **trigger real** (mordida **visual** / dip del corcho) es una etapa posterior, fuera de alcance aquí.
  El grabador (`--record`) ya deja el metraje para construirlo.

## 7. Verificación

- `pytest tests/test_watchdog.py` (6 tests): `decide`, recast sin corcho + parkeo, loot + park + recast,
  wait sin recast, **WARNING tras N recasts**, `mouse_park` fuera de la ROI. Suite completa: **15 verdes**.
- `import main` y la ruta de detección importan **headless** (sin mss/pyautogui/keyboard/sounddevice).
- **Solo validable en Windows:** que el recast suba el uptime en una corrida real y que el cursor quede
  efectivamente fuera de la ROI. Recomendado correr con `--record --log-level DEBUG` y revisar el log
  (los `recast (#n)` y el WARNING de deriva).

## 8. Estado y siguiente paso

- ✅ Watchdog de recast (no más stall), parkeo tras loot/recast, aviso de deriva, clic de loot anotado;
  lógica testeable; `config.roi` y el detector intactos.
- ⏸️ **Trigger real (mordida visual/dip + fusión audio/visual):** etapa posterior, con el metraje que el
  grabador captura. La recalibración de audio sigue pendiente (necesita marcar qué ciclos tuvieron
  mordida real).
