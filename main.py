"""Loop principal del bot de pesca (Etapa 3).

Reconecta la inferencia al modelo de la Etapa 2 (ONNX) y centraliza la configuración en
config.yaml. La detección de mordida por audio (FFT) se conserva tal cual de la versión
original; solo se parametriza desde la config.

Nota: este loop es el punto de entrada en vivo y depende de mss/pyautogui/keyboard/
sounddevice (equipo de juego, Windows). La RUTA DE DETECCIÓN (corcho_detector) es
independiente y testeable en WSL2 (ver detect_offline.py).
"""
import os
import time

import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter

from config import load_config
from corcho_detector import CorchoDetector
from platform_io import InputController, ScreenCapturer

CFG = load_config()


# --- Audio: amplificar ganancia (conservado de la versión original) ---
def aplicar_ganancia(audio_array, ganancia=4.0):
    audio_float = audio_array.astype(np.float32)
    amplificado = audio_float * ganancia
    amplificado = np.clip(amplificado, -32768, 32767)
    return amplificado.astype(np.int16)


# --- Audio: filtro Butterworth pasa banda (conservado) ---
def bandpass_filter(signal, fs, lowcut=300.0, highcut=3000.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return lfilter(b, a, signal)


# --- Audio: convertir a vector mono normalizado (conservado) ---
def audio_to_np_mono(audio_path, target_fs):
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(target_fs).set_channels(1).set_sample_width(2)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    norm = np.max(np.abs(samples))
    if norm == 0:
        return samples
    return samples / norm


# --- Audio: detectar el sonido de pesca por similitud FFT (conservado, parametrizado) ---
def detect_fishing_sound(audio_cfg):
    print("🎧 Escuchando...")
    fs = audio_cfg.fs
    lowcut, highcut = audio_cfg.bandpass
    recording = sd.rec(int(audio_cfg.duration * fs), samplerate=fs, channels=1, dtype="int16")
    sd.wait()

    recording_amplificado = aplicar_ganancia(recording, ganancia=audio_cfg.gain)
    temp_file = "temp_audio.wav"
    write(temp_file, fs, recording_amplificado)
    recorded_np = audio_to_np_mono(temp_file, fs)
    recorded_np = bandpass_filter(recorded_np, fs, lowcut=lowcut, highcut=highcut, order=4)
    os.remove(temp_file)

    fft_recorded_mag = np.abs(np.fft.rfft(recorded_np))
    peak = np.max(fft_recorded_mag)
    fft_recorded_mag_norm = fft_recorded_mag / peak if peak else fft_recorded_mag

    for ref_path in audio_cfg.references:
        ref_np = audio_to_np_mono(ref_path, fs)
        ref_np = bandpass_filter(ref_np, fs, lowcut=lowcut, highcut=highcut, order=4)
        fft_ref_mag = np.abs(np.fft.rfft(ref_np))
        ref_peak = np.max(fft_ref_mag)
        fft_ref_mag_norm = fft_ref_mag / ref_peak if ref_peak else fft_ref_mag

        min_len = min(len(fft_recorded_mag_norm), len(fft_ref_mag_norm))
        rec_vec = fft_recorded_mag_norm[:min_len]
        ref_vec = fft_ref_mag_norm[:min_len]
        similarity = np.dot(rec_vec, ref_vec) / (
            np.linalg.norm(rec_vec) * np.linalg.norm(ref_vec) + 1e-10
        )
        print(f"🔍 {os.path.basename(str(ref_path))}: Cosine Similarity = {similarity:.4f}")
        if similarity > audio_cfg.similarity_threshold:
            print(f"✅ Coincidencia detectada con {os.path.basename(str(ref_path))}")
            return True
    return False


# --- Loop principal ---
def main():
    print("Cargando detector de corcho (ONNX)...")
    detector = CorchoDetector(
        CFG.model_onnx,
        conf_threshold=CFG.detector.conf_threshold,
        iou_threshold=CFG.detector.iou_threshold,
        imgsz=CFG.detector.imgsz,
    )
    capturer = ScreenCapturer()
    inputc = InputController()
    roi = CFG.roi

    print(f"ROI configurada: {roi.as_mss()}")
    print(f"Iniciando pesca, pulsando la tecla '{CFG.input.cast_key}'...")
    inputc.press_key(CFG.input.cast_key)
    time.sleep(1)

    print("Esperando sonido de pesca (Ctrl+C para salir)...")
    try:
        while True:
            inicio = time.time()
            sonido_detectado = False

            while time.time() - inicio < CFG.audio.listen_window:
                if detect_fishing_sound(CFG.audio):
                    sonido_detectado = True
                    print("🎣 Sonido detectado, buscando corcho...")
                    frame = capturer.grab(roi.as_mss())
                    center = detector.best_center(frame)
                    if center is not None:
                        x_abs = roi.left + center[0]
                        y_abs = roi.top + center[1]
                        print(f"🟢 Corcho en ({x_abs}, {y_abs}) - recogiendo")
                        inputc.move_and_click(x_abs, y_abs, button=CFG.input.loot_button)
                        time.sleep(CFG.input.delay_after_click)
                        inputc.press_key(CFG.input.cast_key)
                    else:
                        print("❌ Corcho no detectado")
                    break
                time.sleep(0.5)

            if not sonido_detectado:
                print(f"⏳ Sin sonido en {CFG.audio.listen_window}s, relanzando...")
                inputc.press_key(CFG.input.cast_key)
                time.sleep(CFG.input.delay_after_click)
    except KeyboardInterrupt:
        print("\nFinalizado por el usuario.")


if __name__ == "__main__":
    main()
