import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.signal import correlate
from scipy.io.wavfile import write

# Configuración
REFERENCE_DIR = 'normalized'      # Carpeta donde están Fishing_1.wav, Fishing_2.wav, Fishing_3.wav
DURATION = 3                      # Duración de la grabación en segundos
FS = 44100                        # Frecuencia de muestreo
THRESHOLD = 0.55                  # Umbral de similitud para considerar que hay coincidencia

# --- Funciones auxiliares ---

def aplicar_ganancia(audio_array, ganancia=6.0):
    audio_float = audio_array.astype(np.float32)
    amplificado = audio_float * ganancia
    amplificado = np.clip(amplificado, -32768, 32767)
    return amplificado.astype(np.int16)

def audio_to_np_mono(audio_path, target_fs):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(target_fs).set_channels(1).set_sample_width(2)  # mono, 16-bit
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    # Evitar división por 0 en caso de audio en silencio
    norm = np.max(np.abs(samples))
    if norm == 0:
        return samples
    return samples / norm

def get_fft(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    # Normalizar la magnitud FFT para que el máximo sea 1
    norm = np.max(fft_vals)
    if norm == 0:
        return freqs, fft_vals
    fft_vals = fft_vals / norm
    return freqs, fft_vals

def cosine_similarity(a, b):
    # Si alguna señal es cero, devolvemos 0
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Grabación en tiempo real ---
print("🎙️ Grabando sonido por {} segundos...".format(DURATION))
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
sd.wait()

recording_amplificado = aplicar_ganancia(recording, ganancia=6.0)
TEMP_PATH = 'temp_recording.wav'
write(TEMP_PATH, FS, recording_amplificado)
recorded_np = audio_to_np_mono(TEMP_PATH, FS)
os.remove(TEMP_PATH)

# Calcular FFT de la grabación
freqs_rec, fft_rec = get_fft(recorded_np, FS)

plt.figure(figsize=(10, 3))
plt.plot(freqs_rec, fft_rec, label="Grabación", color='blue')
plt.title("🎧 Espectro de la grabación")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud (normalizada)")
plt.tight_layout()
plt.show()

print("\n🔎 Comparando espectros de frecuencia con sonidos de referencia:\n")

for filename in os.listdir(REFERENCE_DIR):
    if filename.endswith(".wav"):
        ref_path = os.path.join(REFERENCE_DIR, filename)
        ref_np = audio_to_np_mono(ref_path, FS)
        freqs_ref, fft_ref = get_fft(ref_np, FS)

        # Para comparar, igualamos longitudes de los espectros
        min_len = min(len(fft_rec), len(fft_ref))
        fft_rec_cut = fft_rec[:min_len]
        fft_ref_cut = fft_ref[:min_len]
        
        similarity = cosine_similarity(fft_rec_cut, fft_ref_cut)
        print(f"📁 {filename} ➤ Similitud de espectros (coseno): {similarity:.4f}")
        if similarity > THRESHOLD:
            print(f"   ✅ Coincidencia detectada con {filename}\n")
        else:
            print(f"   ❌ No coincide bien con {filename}\n")
        
        # Mostrar gráficas comparativas
        plt.figure(figsize=(10, 4))
        plt.plot(freqs_rec[:min_len], fft_rec_cut, label="Grabación", color='blue')
        plt.plot(freqs_ref[:min_len], fft_ref_cut, label=filename, color='green')
        plt.title(f"🎼 Comparación de Espectros: {filename}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud")
        plt.legend()
        plt.tight_layout()
        plt.show()
