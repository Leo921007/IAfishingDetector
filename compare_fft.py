import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io.wavfile import write

# Configuración
REFERENCE_DIR = 'normalized'
DURATION = 3
FS = 44100
THRESHOLD = 0.8

# --- Funciones ---
def aplicar_ganancia(audio_array, ganancia=6.0):
    audio_float = audio_array.astype(np.float32)
    amplificado = audio_float * ganancia
    amplificado = np.clip(amplificado, -32768, 32767)
    return amplificado.astype(np.int16)

def audio_to_np_mono(audio_path, target_fs):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(target_fs).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    return samples / np.max(np.abs(samples))

def get_fft(signal, fs):
    n = len(signal)
    freqs = np.fft.rfftfreq(n, 1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_vals /= np.max(fft_vals)  # Normalizar
    return freqs, fft_vals

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# --- Grabación ---
print("🎙️ Grabando sonido por {} segundos...".format(DURATION))
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
sd.wait()

recording = aplicar_ganancia(recording)
TEMP_PATH = 'temp_recording.wav'
write(TEMP_PATH, FS, recording)
recorded_np = audio_to_np_mono(TEMP_PATH, FS)
os.remove(TEMP_PATH)

# --- FFT de la grabación ---
freqs_rec, fft_rec = get_fft(recorded_np, FS)

plt.figure(figsize=(10, 3))
plt.plot(freqs_rec, fft_rec, label="Grabación", color='blue')
plt.title("🎧 Espectro de la grabación")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud (normalizada)")
plt.tight_layout()
plt.show()

# --- Comparación con referencias ---
print("\n🔎 Comparando espectros de frecuencia con referencias:\n")

for filename in os.listdir(REFERENCE_DIR):
    if filename.endswith(".wav"):
        ref_path = os.path.join(REFERENCE_DIR, filename)
        ref_np = audio_to_np_mono(ref_path, FS)
        freqs_ref, fft_ref = get_fft(ref_np, FS)

        # Igualar tamaño de espectros
        min_len = min(len(fft_ref), len(fft_rec))
        similarity = cosine_similarity(fft_rec[:min_len], fft_ref[:min_len])

        print(f"📁 {filename} ➤ Similitud de espectros (coseno): {similarity:.4f}")
        if similarity > THRESHOLD:
            print(f"   ✅ Coincidencia en frecuencia con {filename}\n")
        else:
            print(f"   ❌ No coincide bien con {filename}\n")

        # --- Mostrar gráficas comparativas ---
        plt.figure(figsize=(10, 4))
        plt.plot(freqs_rec[:min_len], fft_rec[:min_len], label="Grabación", color='blue')
        plt.plot(freqs_ref[:min_len], fft_ref[:min_len], label=filename, color='green')
        plt.title(f"🎼 Comparación de Espectros: {filename}")
        plt.xlabel("Frecuencia (Hz)")
        plt.ylabel("Amplitud")
        plt.legend()
        plt.tight_layout()
        plt.show()
