import os
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.signal import correlate
from scipy.io.wavfile import write

# Configuración
REFERENCE_DIR = 'normalized'
DURATION = 5  # segundos
FS = 44100
THRESHOLD = 0.15

# --- Funciones auxiliares ---
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

# --- Grabación en tiempo real ---
print("🎙️ Grabando sonido por {} segundos...".format(DURATION))
recording = sd.rec(int(DURATION * FS), samplerate=FS, channels=1, dtype='int16')
sd.wait()

recording = aplicar_ganancia(recording, ganancia=6.0)
TEMP_PATH = 'temp_recording.wav'
write(TEMP_PATH, FS, recording)
recorded_np = audio_to_np_mono(TEMP_PATH, FS)
os.remove(TEMP_PATH)

# --- Visualización del audio grabado ---
plt.figure(figsize=(10, 2))
plt.plot(recorded_np, color='blue')
plt.title("🎧 Grabación en tiempo real")
plt.xlabel("Muestras")
plt.ylabel("Amplitud")
plt.tight_layout()
plt.show()

# --- Comparar con sonidos de referencia ---
print("\n🔎 Comparando con sonidos de referencia:\n")

for filename in os.listdir(REFERENCE_DIR):
    if filename.endswith(".wav"):
        ref_path = os.path.join(REFERENCE_DIR, filename)
        ref_np = audio_to_np_mono(ref_path, FS)

        correlation = correlate(recorded_np, ref_np, mode='valid')
        max_corr = np.max(np.abs(correlation))
        normalized_corr = max_corr / len(ref_np)

        print(f"📁 {filename} ➤ Correlación: {normalized_corr:.4f}")
        if normalized_corr > THRESHOLD:
            print(f"   ✅ Coincidencia detectada con {filename}\n")
        else:
            print(f"   ❌ No coincide con {filename}\n")

        # --- Mostrar gráficas ---
        fig, axs = plt.subplots(3, 1, figsize=(10, 6))
        axs[0].plot(recorded_np, color='blue')
        axs[0].set_title("🎧 Grabación en tiempo real")

        axs[1].plot(ref_np, color='green')
        axs[1].set_title(f"🎼 Referencia: {filename}")

        axs[2].plot(correlation, color='red')
        axs[2].set_title("📊 Correlación cruzada")
        axs[2].axhline(y=max_corr, color='gray', linestyle='--', label=f"Máx: {max_corr:.2f}")
        axs[2].legend()

        plt.tight_layout()
        plt.show()
