import torch
from pathlib import Path
from PIL import Image
import cv2
import numpy as np

# Cargar modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/corcho-detector3/weights/best.pt', source='local')
model.conf = 0.25  # umbral de confianza

# Ruta a las imágenes del juego
source_folder = Path('data/game_screenshots')

# Crear carpeta de salida
output_folder = Path('detections')
output_folder.mkdir(exist_ok=True)

# Procesar imágenes
for image_path in source_folder.glob('*.*'):
    if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue

    print(f"Procesando: {image_path.name}")
    results = model(str(image_path))

    # Mostrar resultados por consola
    results.print()

    # Guardar imagen con bounding box
    results.save(save_dir=output_folder)

print(f"\nDetecciones guardadas en: {output_folder.resolve()}")
