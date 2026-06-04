import cv2
import numpy as np
import pyautogui
import time
import os

# Obtener la ruta base del script actual
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construir la ruta de salida automáticamente
output_dir = os.path.join(base_dir, "dataset", "images", "train")

# Crear la carpeta si no existe
os.makedirs(output_dir, exist_ok=True)

# Seleccionar la región manualmente
screenshot = pyautogui.screenshot()
frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
region = cv2.selectROI("Selecciona la región donde aparece el corcho", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

x, y, w, h = region
print(f"✅ Región seleccionada: {region}")

# Número de capturas
num_capturas = 115
intervalo = 2  # segundos entre capturas

print(f"📸 Iniciando captura de {num_capturas} imágenes cada {intervalo} segundos...")

for i in range(num_capturas):
    i += 100
    imagen = pyautogui.screenshot(region=(x, y, w, h))
    imagen_np = np.array(imagen)
    imagen_bgr = cv2.cvtColor(imagen_np, cv2.COLOR_RGB2BGR)
    
    archivo = os.path.join(output_dir, f"corcho_{i:03d}.jpg")
    cv2.imwrite(archivo, imagen_bgr)
    print(f"🖼️ Imagen guardada: {archivo}")
    
    time.sleep(intervalo)

print("🎉 Captura finalizada.")
