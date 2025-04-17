from pydub import AudioSegment
import numpy as np
import os

SOUND_REFERENCES = ['Fishing_1.wav', 'Fishing_2.wav', 'Fishing_3.wav']

def analizar_audio(path):
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(44100)
    samples = np.array(audio.get_array_of_samples())

    duration = len(samples) / 44100
    max_val = np.max(samples)
    min_val = np.min(samples)
    rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
    silence_ratio = np.sum(np.abs(samples) < 500) / len(samples)

    print(f"📁 {os.path.basename(path)}")
    print(f"  ➤ Duración: {duration:.2f} s")
    print(f"  ➤ Máximo: {max_val}, Mínimo: {min_val}")
    print(f"  ➤ RMS (potencia): {rms:.2f}")
    print(f"  ➤ Porcentaje de silencio (<500): {silence_ratio*100:.2f}%\n")

for ref in SOUND_REFERENCES:
    analizar_audio(ref)
