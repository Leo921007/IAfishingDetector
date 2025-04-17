import cv2
import torch
import numpy as np
import pyautogui
import time
import sounddevice as sd
from scipy.io.wavfile import write
import os
import mss
import keyboard
from pydub import AudioSegment
from scipy.signal import correlate
from tkinter import Tk, Canvas
from scipy.signal import butter, lfilter

# Configuración del modelo
MODEL_PATH = 'yolov5/runs/train/corcho-detector3/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.25
# Lista de archivos de sonido de referencia en formato WAV
SOUND_REFERENCES = ['Fishing_1.wav', 'Fishing_2.wav', 'Fishing_3.wav']
DELAY_AFTER_CLICK = 2  # segundos

# --- Función para amplificar ganancia ---
def aplicar_ganancia(audio_array, ganancia=4.0):
    audio_float = audio_array.astype(np.float32)
    amplificado = audio_float * ganancia
    amplificado = np.clip(amplificado, -32768, 32767)  # Limitar para evitar saturación
    return amplificado.astype(np.int16)

# --- Filtro pasa banda (Filtro Butterworth) ---
def bandpass_filter(signal, fs, lowcut=300.0, highcut=3000.0, order=4):
    """
    Aplica un filtro Butterworth pasa banda a la señal.
    
    Parámetros:
      - signal: La señal de entrada (array de numpy).
      - fs: Frecuencia de muestreo (Hz).
      - lowcut: Frecuencia baja de corte.
      - highcut: Frecuencia alta de corte.
      - order: Orden del filtro.
      
    Retorna:
      - La señal filtrada.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal)
    return y

# --- Selección de región de pantalla ---
def select_screen_region():
    root = Tk()
    root.attributes('-fullscreen', True)
    root.attributes('-alpha', 0.3)
    canvas = Canvas(root, cursor="cross", bg='black')
    canvas.pack(fill="both", expand=True)
    
    region = {}
    rect = None

    def on_click(event):
        region["x1"] = event.x
        region["y1"] = event.y

    def on_drag(event):
        nonlocal rect
        if rect:
            canvas.delete(rect)
        rect = canvas.create_rectangle(region["x1"], region["y1"], event.x, event.y, outline='red')

    def on_release(event):
        region["x2"] = event.x
        region["y2"] = event.y
        root.quit()

    canvas.bind("<ButtonPress-1>", on_click)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)

    root.mainloop()
    root.destroy()

    x = min(region["x1"], region["x2"])
    y = min(region["y1"], region["y2"])
    w = abs(region["x1"] - region["x2"])
    h = abs(region["y1"] - region["y2"])
    return {"top": y, "left": x, "width": w, "height": h}

# --- Captura de pantalla ---
def capture_screen(region):
    with mss.mss() as sct:
        img = sct.grab(region)
        return np.array(img)

# --- Función para convertir audio a vector normalizado ---
def audio_to_np_mono(audio_path, target_fs):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(target_fs).set_channels(1).set_sample_width(2)  # mono, 16-bit
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    norm = np.max(np.abs(samples))
    if norm == 0:
        return samples
    return samples / norm

# --- Comparar sonido grabado con los de referencia usando correlación ---
def detect_fishing_sound(duration=3, fs=44100, ganancia=6.0, similarity_threshold=0.5):
    print("🎧 Escuchando...")
    # Grabación en tiempo real
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    
    # Aplicar ganancia
    recording_amplificado = aplicar_ganancia(recording, ganancia=ganancia)
    temp_file = "temp_audio.wav"
    write(temp_file, fs, recording_amplificado)
    
    # Convertir a vector normalizado
    recorded_np = audio_to_np_mono(temp_file, fs)
    
    # Aislar la banda de interés (por ejemplo 300 Hz a 3000 Hz)
    recorded_np_filtrado = bandpass_filter(recorded_np, fs, lowcut=300.0, highcut=3000.0, order=4)
    
    os.remove(temp_file)
    
    # Calcular la FFT de la señal filtrada
    fft_recorded = np.fft.rfft(recorded_np_filtrado)
    fft_recorded_mag = np.abs(fft_recorded)
    if np.max(fft_recorded_mag) == 0:
        fft_recorded_mag_norm = fft_recorded_mag
    else:
        fft_recorded_mag_norm = fft_recorded_mag / np.max(fft_recorded_mag)
    
    match_found = False
    # Comparar contra cada sonido de referencia
    for ref_path in SOUND_REFERENCES:
        ref_np = audio_to_np_mono(ref_path, fs)
        ref_np_filtrado = bandpass_filter(ref_np, fs, lowcut=300.0, highcut=3000.0, order=4)
        fft_ref = np.fft.rfft(ref_np_filtrado)
        fft_ref_mag = np.abs(fft_ref)
        if np.max(fft_ref_mag) == 0:
            fft_ref_mag_norm = fft_ref_mag
        else:
            fft_ref_mag_norm = fft_ref_mag / np.max(fft_ref_mag)
        
        # Igualar longitudes
        min_len = min(len(fft_recorded_mag_norm), len(fft_ref_mag_norm))
        rec_vec = fft_recorded_mag_norm[:min_len]
        ref_vec = fft_ref_mag_norm[:min_len]
        
        # Calcular similitud coseno
        similarity = np.dot(rec_vec, ref_vec) / (np.linalg.norm(rec_vec) * np.linalg.norm(ref_vec) + 1e-10)
        print(f"🔍 Comparando con {ref_path}: Cosine Similarity = {similarity:.4f}")
        if similarity > similarity_threshold:
            print(f"✅ Coincidencia detectada con {ref_path}")
            match_found = True
            break
    
    return match_found

# --- Inference con YOLOv5 ---
def detect_corcho(image, model):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(img)
    detections = results.xyxy[0].cpu().numpy()

    corchos = [d for d in detections if d[4] >= CONFIDENCE_THRESHOLD]
    if not corchos:
        return None

    # Devolver el corcho con mayor confianza
    best = max(corchos, key=lambda x: x[4])
    x_center = int((best[0] + best[2]) / 2)
    y_center = int((best[1] + best[3]) / 2)
    return (x_center, y_center)

# --- Automatización: mover el mouse y presionar tecla ---
def click_and_press(x, y):
    pyautogui.moveTo(x, y)
    pyautogui.click()
    time.sleep(DELAY_AFTER_CLICK)
    keyboard.press_and_release('2')

# --- Main ---
def main():
    print("Selecciona la región donde aparece el corcho en el juego.")
    region = select_screen_region()
    print(f"Región seleccionada: {region}")

    print("Cargando modelo YOLOv5...")
    model = torch.hub.load('yolov5', 'custom', path=MODEL_PATH, source='local')
    model.conf = CONFIDENCE_THRESHOLD

    # Presionar la tecla '2' para iniciar la pesca
    print("Iniciando pesca, presionando la tecla '2'...")
    keyboard.press_and_release('2')
    time.sleep(1)  # Pequeña pausa para asegurarse de que la acción se registre

    print("Esperando sonido de pesca (Ctrl+C para salir)...")
    try:
        while True:
            inicio = time.time()
            sonido_detectado = False

            # Intentar detectar el sonido durante 22 segundos
            while time.time() - inicio < 22:
                if detect_fishing_sound():
                    sonido_detectado = True
                    print("🎣 Sonido detectado, buscando corcho...")
                    frame = capture_screen(region)
                    coords = detect_corcho(frame, model)
                    if coords:
                        x_abs = region["left"] + coords[0]
                        y_abs = region["top"] + coords[1]
                        print(f"🟢 Corcho detectado en ({x_abs}, {y_abs}) - Haciendo clic")
                        click_and_press(x_abs, y_abs)
                    else:
                        print("❌ Corcho no detectado")
                    break  # Salir del ciclo de 22s si ya detectó
                else:
                    time.sleep(0.5)  # Esperar un poco antes de volver a escuchar

            if not sonido_detectado:
                print("⏳ No se detectó sonido en 22s, presionando tecla '2'...")
                keyboard.press_and_release('2')
                time.sleep(DELAY_AFTER_CLICK)  # Esperar después de presionar
    except KeyboardInterrupt:
        print("\nFinalizado por el usuario.")

if __name__ == "__main__":
    main()
