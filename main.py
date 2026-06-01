"""Loop principal del bot de pesca.

Reconecta la inferencia al modelo (ONNX) y centraliza la configuración. La captura de audio
(micrófono) está aislada en platform_io.AudioRecorder y el match FFT en audio_match.match_audio
(algoritmo conservado), lo que permite reproducir sesiones offline.

Nota: este loop es el punto de entrada en vivo y depende de mss/pyautogui/keyboard/sounddevice
(equipo de juego, Windows) solo al instanciar los adaptadores. La ruta de detección
(corcho_detector) y el match (audio_match) son independientes y testeables en WSL2.
"""
import time

from config import load_config
from audio_match import match_audio
from corcho_detector import CorchoDetector
from platform_io import AudioRecorder, InputController, ScreenCapturer

CFG = load_config()


def detect_fishing_sound(recorder, audio_cfg):
    """Graba un chunk y lo compara con las referencias. Devuelve (matched, scores)."""
    print("🎧 Escuchando...")
    recording = recorder.record(audio_cfg.duration, audio_cfg.fs)
    matched, scores = match_audio(recording, audio_cfg.fs, audio_cfg.references, audio_cfg)
    for name, score in scores.items():
        print(f"🔍 {name}: Cosine Similarity = {score:.4f}")
    if matched:
        print("✅ Coincidencia de audio detectada")
    return matched


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
    recorder = AudioRecorder()
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
                if detect_fishing_sound(recorder, CFG.audio):
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
