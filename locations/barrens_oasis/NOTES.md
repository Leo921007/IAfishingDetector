# Ubicación: barrens_oasis

Lugar de pesca: oasis de Los Baldíos. **Pendiente de entrenar.**

## Estado
- `detector.onnx`: **no existe todavía** (hay que capturar, etiquetar y entrenar para esta zona).
- `roi.yaml`: placeholder con los valores de stormwind; **ajustar en la GUI** al encuadre de este oasis.

## Para activarla
1. Calibrar la ROI de esta zona y guardarla en `roi.yaml`.
2. Capturar/etiquetar un dataset (ver el flujo de stormwind: extract/label/build/train).
3. Colocar el `detector.onnx` resultante en `locations/barrens_oasis/detector.onnx`.
4. En `config.yaml`: `location: barrens_oasis`.
