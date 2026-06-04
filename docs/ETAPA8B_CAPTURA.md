# ETAPA 8B — Captura dedicada de la mordida + análisis del splash

> Prepara **datos limpios** para calibrar el trigger visual: una captura a **alta tasa, sin sesgo de
> loot, con la mordida marcada por el usuario**, y mide una métrica de **splash** en el parche del corcho.
> **No toca loop/detector/dataset/audio** (el bot de audio sigue como fallback funcional).

Contexto: 8A (ver `ETAPA8A_DIP.md`) mostró que la mordida no es una caída de `center_y` sino un
**chapuzón en el sitio** (agua blanca + el detector pierde el corcho un instante), y que los buffers de
4 s a 10 fps post-loot no servían. Aquí capturamos a más fps y marcando la mordida a mano.

---

## 1. Capturar en Windows — `capture_bite.py`

`mss` y `keyboard` son de Windows (en WSL2 el script solo se importa/parsea, no se ejecuta).

```bat
:: en el equipo de juego, con el venv y la ROI ya calibrada en config.yaml
.venv\Scripts\pip install mss keyboard
:: keyboard suele requerir terminal como ADMINISTRADOR
.venv\Scripts\python capture_bite.py            :: pre 4s / post 0.5s / buffer 6s / jpeg q90
```

- Graba la **ROI de config** lo más rápido posible y **loguea los fps reales** alcanzados.
- Pesca normalmente. **Cuando veas el chapuzón, pulsa `b`** (no hace falta ser exacto: la ventana
  guardada se extiende ~4 s ANTES del keypress, así que el splash queda dentro aunque reacciones tarde).
- Cada `b` vuelca `captures_bite/<timestamp>/` con JPEGs + `manifest.json` (fps real, timestamps, índice
  del keypress). `q` = salir. **No** clickea ni castea; es captura pura.
- Trae las carpetas `captures_bite/<timestamp>/` al repo en WSL2 para analizarlas.

## 2. Analizar en WSL2 — `analyze_splash.py` (headless, usa `best_zona.onnx`)

```bash
.venv/bin/python analyze_splash.py            # analiza captures_bite/*/
.venv/bin/python analyze_splash.py --selftest # prueba el pipeline con una ventana sintética
```

Por ventana: localiza el corcho (frames previos al splash), define un **parche ×1.5** y calcula por frame:
- **frame-diff** (movimiento del agua) y **foam** (fracción de píxeles casi-blancos, HSV alto V / baja S);
- compara **baseline (flote)** vs **spike (ventana previa al keypress)**, mide la **duración** del pico y
  el **fps mínimo** que mantiene la separación; saca **PNGs de QA** y propone **regla + umbral**.
- Salidas en `captures_bite/analysis/` (CSV + PNG, gitignored).

Verificado headless: `--selftest` PASS (la métrica foam pega un pico claro en la ventana sintética). Sin
capturas reales, el script avisa y sale 0.

## 3. Alcance / honestidad

- Esto **solo prepara datos y mide**; **no** fija aún el trigger ni toca el loop. El umbral y el fps del
  trigger visual se decidirán con los datos reales (Etapa 8C / integración).
- Si tras analizar las capturas el pico no separa bien (p. ej. fps insuficiente o pocas mordidas), el
  reporte lo dirá y habrá que recapturar (más fps o más bites).

## 4. Estado y siguiente paso

- ✅ `capture_bite.py` (Windows) y `analyze_splash.py` (headless, con `--selftest`) listos; `captures_bite/`
  gitignored.
- ⏸️ **Te toca capturar en Windows** unas decenas de mordidas con `capture_bite.py`, traer las carpetas y
  correr `analyze_splash.py`. Con esos números decidimos la regla del trigger visual.
