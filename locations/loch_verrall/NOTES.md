# Ubicación: loch_verrall

Lugar de pesca: Loch Verrall (Twilight Highlands). **Pendiente de captura.**

## Estado
- `detector.onnx`: **no existe todavía** (hay que capturar, etiquetar y entrenar para esta zona).
- `roi.yaml`: placeholder con los valores de stormwind; **ajustar en la GUI** al encuadre de este lago.

## Para activarla
1. Calibrar la ROI de esta zona y guardarla en `roi.yaml`.
2. Capturar/etiquetar un dataset (ver el flujo de stormwind: capture_bite/extract/label/build/train).
3. Colocar el `detector.onnx` resultante en `locations/loch_verrall/detector.onnx`.
4. En `config.yaml`: `location: loch_verrall`.
