from pydub import AudioSegment
import os

# Nivel de referencia RMS (ajustable según necesidad)
TARGET_RMS = 3000

# Crear carpeta de salida si no existe
output_folder = "normalized"
os.makedirs(output_folder, exist_ok=True)

def match_target_rms(sound, target_rms):
    current_rms = sound.rms
    if current_rms == 0:
        return sound
    change_in_dBFS = 10 * (target_rms / current_rms) ** 0.5
    return sound.apply_gain(change_in_dBFS)

for filename in os.listdir():
    if filename.lower().endswith(".wav"):
        print(f"🎧 Procesando: {filename}")
        audio = AudioSegment.from_wav(filename)
        normalized_audio = match_target_rms(audio, TARGET_RMS)
        output_path = os.path.join(output_folder, filename)
        normalized_audio.export(output_path, format="wav")
        print(f"   ✅ Guardado como: {output_path}")
