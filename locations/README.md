# Ubicaciones de pesca

Cada lugar de pesca tiene su propia agua/encuadre, así que su **modelo** y su **ROI** son por ubicación.

## Estructura

```
locations/<loc>/
  detector.onnx   # modelo de la zona (GITIGNORED; colócalo o entrénalo por máquina)
  roi.yaml        # ROI de captura (versionado)
  dataset/        # dataset de la zona (GITIGNORED; raw/labels/images, local)
  NOTES.md        # notas de la zona (versionado)
models/general/
  detector.onnx   # modelo general multi-zona (GITIGNORED)
  .gitkeep        # mantiene la carpeta en git
```

Solo se versionan `roi.yaml`, `NOTES.md`, `.gitkeep` y este README. Los `.onnx` y los `dataset/` son
pesados y **gitignored**: hay que colocarlos en cada máquina (o entrenarlos).

## Cómo se elige el modelo (config.yaml)

```yaml
detector_mode: specific   # general | specific
location: stormwind       # carpeta dentro de locations/
```

- `specific` → `locations/<location>/detector.onnx` + `locations/<location>/roi.yaml`.
- `general`  → `models/general/detector.onnx` (la ROI sigue saliendo de la ubicación activa).

Si el `detector.onnx` resuelto no existe, el bot aborta con un mensaje accionable (qué ubicación, dónde
colocar el modelo o cómo cambiar `detector_mode`/`location`).

## Ubicaciones actuales
- **stormwind**: entrenada y validada (mAP50 ≈ 0.99, conf 0.25 sin FP). Ver `stormwind/NOTES.md`.
- **barrens_oasis**: placeholder, pendiente de entrenar. Ver `barrens_oasis/NOTES.md`.
