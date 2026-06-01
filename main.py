"""Loop principal del bot de pesca.

Reconecta la inferencia al modelo (ONNX), centraliza la configuración, registra cada ciclo con
logging estructurado (consola + archivo en logs/) y, opcionalmente, graba la sesión
(frames/audio/eventos en sessions/) para validación offline. La captura de audio está aislada en
platform_io.AudioRecorder y el match FFT en audio_match.match_audio (algoritmo conservado).

Nota: este loop es el punto de entrada en vivo y depende de mss/pyautogui/keyboard/sounddevice
(equipo de juego, Windows) solo al instanciar los adaptadores. La ruta de detección
(corcho_detector) y el match (audio_match) son independientes y testeables en WSL2.
"""
import argparse
import time

from config import load_config
from audio_match import match_audio
from corcho_detector import CorchoDetector
from logging_setup import setup_logging
from platform_io import AudioRecorder, InputController, ScreenCapturer
from session import SessionRecorder

CFG = load_config()
log = None  # logger; se inicializa en main()


def main():
    parser = argparse.ArgumentParser(description="Bot de pesca (loop en vivo)")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING (override de config)")
    parser.add_argument("--record", action="store_true", help="Grabar la sesión en sessions/")
    args = parser.parse_args()

    global log
    log = setup_logging(CFG.logging, level_override=args.log_level)

    log.info("Cargando detector ONNX: %s", CFG.model_onnx)
    detector = CorchoDetector(
        CFG.model_onnx,
        conf_threshold=CFG.detector.conf_threshold,
        iou_threshold=CFG.detector.iou_threshold,
        imgsz=CFG.detector.imgsz,
    )
    capturer = ScreenCapturer()
    inputc = InputController()
    recorder = AudioRecorder()
    session = SessionRecorder(
        CFG.session.dir, enabled=(args.record or CFG.session.enabled), fs=CFG.audio.fs
    )
    roi = CFG.roi
    log.info(
        "ROI=%s | conf=%.2f | loot=%s | escucha=%ds | grabando=%s",
        roi.as_mss(), CFG.detector.conf_threshold, CFG.input.loot_button,
        CFG.audio.listen_window, session.enabled,
    )
    if session.enabled:
        log.info("Sesión en: %s", session.dir)

    inputc.press_key(CFG.input.cast_key)
    log.info("Lanzamiento inicial (tecla '%s')", CFG.input.cast_key)
    time.sleep(1)

    ciclo = 0
    try:
        while True:
            ciclo += 1
            inicio = time.time()
            sonido_detectado = False
            last_recording = None
            last_scores = {}
            log.info("ciclo %d: ventana de escucha %ds", ciclo, CFG.audio.listen_window)

            while time.time() - inicio < CFG.audio.listen_window:
                recording = recorder.record(CFG.audio.duration, CFG.audio.fs)
                matched, scores = match_audio(
                    recording, CFG.audio.fs, CFG.audio.references, CFG.audio
                )
                last_recording, last_scores = recording, scores
                log.debug("ciclo %d: scores audio=%s", ciclo, {k: round(v, 4) for k, v in scores.items()})

                if matched:
                    sonido_detectado = True
                    log.info("ciclo %d: mordida por audio scores=%s", ciclo,
                             {k: round(v, 3) for k, v in scores.items()})
                    frame = capturer.grab(roi.as_mss())
                    dets = detector.detect(frame)
                    if dets:
                        best = max(dets, key=lambda d: d.conf)
                        cx, cy = best.center
                        x_abs, y_abs = roi.left + cx, roi.top + cy
                        log.info(
                            "ciclo %d: corcho bbox=(%.0f,%.0f,%.0f,%.0f) conf=%.3f -> clic %s en (%d,%d)",
                            ciclo, best.x1, best.y1, best.x2, best.y2, best.conf,
                            CFG.input.loot_button, x_abs, y_abs,
                        )
                        inputc.move_and_click(x_abs, y_abs, button=CFG.input.loot_button)
                        time.sleep(CFG.input.delay_after_click)
                        inputc.press_key(CFG.input.cast_key)
                        log.info("ciclo %d: recogido y relanzado", ciclo)
                        detection = {"bbox": [best.x1, best.y1, best.x2, best.y2],
                                     "conf": best.conf, "click": [x_abs, y_abs]}
                        outcome = "recogido"
                    else:
                        log.info("ciclo %d: audio OK pero corcho no detectado", ciclo)
                        detection = None
                        outcome = "corcho_no_detectado"

                    session.record_cycle(
                        ciclo, frame_bgr=frame, audio_int16=recording,
                        event={"matched": True, "scores": {k: round(v, 4) for k, v in scores.items()},
                               "detection": detection, "outcome": outcome},
                    )
                    break
                time.sleep(0.5)

            if not sonido_detectado:
                log.info("ciclo %d: sin sonido en %ds, relanzando", ciclo, CFG.audio.listen_window)
                session.record_cycle(
                    ciclo, frame_bgr=None, audio_int16=last_recording,
                    event={"matched": False,
                           "scores": {k: round(v, 4) for k, v in last_scores.items()},
                           "detection": None, "outcome": "sin_sonido"},
                )
                inputc.press_key(CFG.input.cast_key)
                time.sleep(CFG.input.delay_after_click)
    except KeyboardInterrupt:
        log.info("Finalizado por el usuario.")


if __name__ == "__main__":
    main()
