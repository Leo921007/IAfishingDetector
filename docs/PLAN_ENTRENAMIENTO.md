# Plan de entrenamiento — detección de corcho + picadura

Ubicaciones objetivo: **Stormwind**, **Barrens (Oasis)**, **Un'Goro Crater**.
Dos problemas de detección por ubicación: **posición del corcho** (`corcho`) y **picadura/mordida** (`mordida`).
El énfasis del plan está en **validar la picadura de la mejor manera posible** (Sección 7).

---

## 1. Por qué este plan (resumen del diagnóstico)

Los datos en vivo (corridas de la noche, con lluvia) mostraron dos fallas con la **misma raíz**:

- **El foam está muerto** en agua oscura/lluvia: los píxeles casi-blancos (V>200) son ~0, no hay señal sobre la que disparar. El umbral de foam se calibró en agua clara (8B: mordida 0.003–0.074); fuera de esa condición no sirve.
- **YOLO pierde el corcho flotando** ("corcho perdido (5 fallos)") aunque el corcho es plenamente visible. El modelo se entrenó en agua clara/calma; la lluvia + oscuridad lo sacan de distribución.

Conclusión: la señal de picadura tiene que **aprenderse** (no un umbral de brillo), y el modelo de posición debe **ver el corcho en todas las condiciones**. Ambas cosas se arreglan con datos de esas condiciones.

---

## 2. Arquitectura objetivo

- **Modelos por ubicación**: `locations/<loc>/detector.onnx` (+ clasificador si aplica), `roi.yaml`, `dataset/`, `NOTES.md`.
- **Robustez por condición** dentro de cada ubicación: día/noche × claro/lluvia.
- **El disparo deja de ser foam** y pasa a ser una detección aprendida de la mordida.
- **Bake-off de arquitectura de mordida** (se decide con la validación, no a priori):
  - **A) YOLO de 2 clases** — `corcho` + `mordida` en un solo modelo. Ventaja: el salpicón es un evento distintivo, puede detectarse aun cuando el corcho flotando flaquea.
  - **B) YOLO `corcho` (posición) + clasificador binario** (picadura/no) sobre el parche del corcho. Ventaja: etiquetado binario trivial (las marcas `b` de `capture_bite` ya son los positivos) y validación más limpia.
- Las dos se evalúan con el **mismo arnés** (Sección 7) sobre los mismos datos retenidos de Stormwind; gana la que mida mejor y esa se replica en las otras ubicaciones.

---

## 3. Fase 0 — Tooling (antes de capturar)

1. **`capture_bite` extendido**: etiquetar cada sesión con su condición (`--cond dia_claro|dia_lluvia|noche_claro|noche_lluvia`) y **persistir la marca `b` como evento de verdad-terreno** en el `manifest.json` (timestamp + frame más cercano). Esa marca `b` es a la vez dato de entrenamiento y **ground-truth de validación**.
2. **Etiquetador de 2 clases** (extender `label_zona`): caja de `corcho` (ya existe) y, según la rama del bake-off, caja de `mordida` (rama A) o **tag binario** del frame (rama B). Atajos de teclado para marcar rápido.
3. **Arnés de validación** (lo más importante; ver Sección 7): script que corre un modelo sobre sesiones retenidas y calcula métricas **a nivel de evento** contra las marcas `b`.
4. **Split por sesión**: utilitario que separa train/val/test **por sesión completa**, nunca por frame (frames de una misma picadura están correlacionados → fuga de datos si se mezclan).

---

## 4. Protocolo de captura (por ubicación)

**Matriz de condiciones** (capturar lo que se pueda; el clima en WoW es parcialmente aleatorio):

| Condición       | Prioridad | Objetivo de picadas |
|-----------------|-----------|---------------------|
| día / claro     | alta      | ≥ 50                |
| noche / claro   | alta      | ≥ 50                |
| día / lluvia    | media     | ≥ 40                |
| noche / lluvia  | media     | ≥ 40                |

- **Objetivo total por ubicación**: **≥ 150–200 picadas marcadas con `b`**, repartidas en condiciones.
- **`corcho` (flotando)**: abundante y automático — cada ciclo deja miles de frames; no es el cuello de botella.
- **Sesiones**: ~6–10 por ubicación, 10–15 min cada una, etiquetadas por condición.
- **Variá la posición de caída** del corcho (incluyendo cerca de bordes) para robustez de posición; mantené el parqueo de cámara consistente.

---

## 5. Etiquetado

- **`corcho`**: caja sobre el corcho flotando, en todos los frames donde aparezca (todas las condiciones).
- **`mordida`** — definición precisa para que sea consistente: los frames **desde el inicio visible del chapuzón/salpicón hasta que el corcho vuelve al reposo** (típicamente los pocos frames alrededor de la marca `b`). La marca `b` (picadura confirmada por humano) **ancla** qué ventana es mordida real y evita subjetividad.
  - Rama A: caja sobre corcho+salpicón en esos frames.
  - Rama B: tag binario `picadura` en esos frames; el resto `flotando`.
- **Split por sesión 70/15/15** (train/val/test). Ninguna picadura debe quedar partida entre sets.
- **Desbalance de clases**: la mordida es rara frente a flotando → ponderar/oversamplear en train, pero **mantener una proporción realista flotando:mordida en val/test** (si no, las métricas mienten).

---

## 6. Entrenamiento

- **Posición (`corcho`)**: YOLO11n, transfer desde el `best` actual. **Augmentation fotométrico fuerte** (brillo/contraste/HSV) para cubrir día↔noche y **ruido/moteado** para cubrir lluvia. Esto ayuda a generalizar aun con pocos datos de noche/lluvia. Un modelo por ubicación.
- **Mordida (bake-off)**:
  - A) YOLO de 2 clases (mismo backbone, 2 cabezas de clase).
  - B) Clasificador binario liviano sobre el parche del corcho (entrada: parche ×1.5–2.0; salida: picadura/no). Para robustez ante un parpadeo de YOLO, el clasificador puede correr sobre la **última posición conocida** del corcho.
- Salidas a `locations/<loc>/detector.onnx` (+ `classifier.onnx` si gana B).

---

## 7. Validación — núcleo del plan (sobre todo la mordida)

**Principio**: lo que importa no es la exactitud por-frame, sino **"¿el bot atrapó la picadura?"**. Por eso se mide **a nivel de EVENTO**, no de frame.

**Ground-truth**: las marcas `b` de las **sesiones retenidas (test)** son las picaduras reales confirmadas por humano.

### 7.1 Replay offline (estándar de oro, repetible)
Correr el modelo sobre las sesiones retenidas (sin el juego) y, contra cada evento `b`:
- **Catch rate (recall de evento)**: ¿disparó `mordida` al menos una vez dentro de la **ventana de captura** (~5 s desde el inicio de la picadura)? → verdadero positivo.
- **False-fire rate**: disparos durante flotando sin picadura → falsos positivos. Se reporta como **FP por minuto de flotando**.
- **Latencia**: tiempo desde el primer frame de mordida real hasta el disparo (debe entrar en la ventana; ideal < 1.5 s).

### 7.2 Desglose por condición
Reportar catch rate / false-fire / latencia **por separado** para día/noche/claro/lluvia. Así sabemos exactamente dónde flaquea y si falta capturar esa condición.

### 7.3 Barrido de umbral (punto de operación)
El `conf` de la clase/`prob` del clasificador cambia el trade-off catch↔false-fire. Barrer en **val**, elegir el punto (p.ej. **máximo catch sujeto a false-fire < 0.2/min**), y **confirmar en test**. Nunca elegir el umbral en test.

### 7.4 Validación de posición (`corcho`)
- Métricas estándar de detección (mAP50, precisión/recall) en frames retenidos, **por condición**.
- **Continuidad de tracking offline**: tasa de "relocate fallido" sobre secuencias de flotando retenidas → mide directamente el problema de "corcho perdido".

### 7.5 Comparación contra baseline
Pasar el **mismo arnés** al trigger de foam → demostrar, con números y por condición, que la mordida aprendida **supera al foam** (y dónde el foam ya no servía).

### 7.6 Confirmación en vivo (chequeo final)
Corrida en vivo de ~50 ciclos: **loots/ciclos**, comparado con el baseline. Lo vivo es el chequeo final; el **replay offline es el validador riguroso y reproducible** (se puede correr mil veces sobre los mismos datos).

### 7.7 Bake-off A vs B
Ambas ramas se miden con 7.1–7.5 sobre los mismos datos retenidos de Stormwind. Gana la de mejor catch rate a igual false-fire (con latencia y costo de cómputo como desempate). Esa arquitectura se fija para Barrens y Un'Goro.

---

## 8. Criterios de aceptación (propuestos — a confirmar)

Por ubicación:
- **`corcho`**: mAP50 ≥ 0.95 en el conjunto de todas las condiciones; relocate-fail offline por debajo del baseline actual.
- **`mordida`**: catch rate offline **≥ 90 %** en test (**≥ 85 %** en la peor condición); **false-fire ≤ 0.2/min**; latencia < 1.5 s.
- **En vivo**: loots/ciclos ≥ baseline + margen, en 50 ciclos.

---

## 9. Plan por fases

- **Fase 0** — Tooling: `capture_bite` con condición + GT, etiquetador 2 clases, **arnés de validación**, split por sesión.
- **Fase 1 — Stormwind**: capturar (todas las condiciones) → etiquetar → entrenar `corcho` + bake-off `mordida` → **validar con el arnés** → elegir arquitectura ganadora → confirmar en vivo → cambiar el trigger de foam a mordida.
- **Fase 2 — Barrens (Oasis)**: repetir con la arquitectura elegida; su propio `roi.yaml` y dataset.
- **Fase 3 — Un'Goro Crater**: repetir.
- **Fase 4 (opcional)** — Modelo **general** entrenado sobre las 3 ubicaciones; comparar contra los específicos con el mismo arnés.

---

## 10. Riesgos y mitigaciones

- **El clima no se controla** → capturar de forma oportunista + augmentation para sintetizar lluvia/oscuridad.
- **Subjetividad al etiquetar la mordida** → regla estricta de la Sección 5 anclada en la marca `b`; revisar una muestra al azar.
- **Desbalance de clases** → ponderar en train; proporciones realistas en val/test.
- **Dependencia del parche (rama B)** → correr el clasificador sobre la última posición conocida durante caídas breves de YOLO.
- **Sobreajuste a la cámara de un spot** → variar posiciones de caída; el modelo general (Fase 4) como contramedida.

---

## Apéndice — mapeo a tooling existente

`capture_bite.py` (captura + marca `b`), `label_zona.py` (etiquetado), `train_corcho_zona.py` (entrenamiento), `analyze_splash.py`/`replay.py` (base del arnés offline), `data/corcho_zona.yaml` (dataset YOLO), `locations/<loc>/` (modelo + roi + dataset por ubicación). El arnés de validación a nivel de evento es **nuevo** y es el entregable central de la Fase 0.
