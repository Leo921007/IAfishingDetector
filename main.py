"""Loop principal del bot de pesca.

Reconecta la inferencia al modelo (ONNX), centraliza la configuración, registra cada ciclo con
logging estructurado y, opcionalmente, graba la sesión. La captura de audio está aislada en
platform_io.AudioRecorder y el match FFT en audio_match.match_audio (algoritmo conservado).

Robustez (Etapa 7): un **watchdog** sondea la ROI cada ~watchdog_interval (alineado con los chunks de
audio); si NO hay corcho, recastea — así un fallo de casteo no detiene el loop. Tras cada loot y cada
recast se **parkea** el cursor fuera de la ROI. La lógica (`decide`, `LootLoop`) es testeable headless.

Nota: el loop en vivo depende de mss/pyautogui/keyboard/sounddevice (Windows) solo al instanciar los
adaptadores. La ruta de detección (corcho_detector) y el match (audio_match) son independientes.
"""
import argparse
import time

from config import load_config
from audio_match import match_audio
from corcho_detector import CorchoDetector
from logging_setup import setup_logging
from platform_io import AudioRecorder, FrameGrabber, InputController, ScreenCapturer
from session import SessionRecorder

CFG = load_config()

CAST_SETTLE = 0.3  # s tras castear antes de parkear el cursor


def decide(has_corcho: bool, audio_matched: bool) -> str:
    """Decisión del watchdog: 'recast' si no hay corcho; 'loot' si corcho+mordida; 'wait' si corcho sin mordida."""
    if not has_corcho:
        return "recast"
    if audio_matched:
        return "loot"
    return "wait"


class LootLoop:
    """Loop de pesca con watchdog de recast y parkeo del cursor (adaptadores inyectados)."""

    def __init__(self, cfg, log, detector, capturer, inputc, recorder, session, grabber=None):
        self.cfg = cfg
        self.log = log
        self.detector = detector
        self.capturer = capturer
        self.inputc = inputc
        self.recorder = recorder
        self.session = session
        self.grabber = grabber
        self.roi = cfg.roi
        self.consecutive_recasts = 0
        self.last_cast = 0.0

    def _park(self):
        px, py = self.cfg.input.mouse_park
        self.inputc.park(px, py)

    def cast_and_park(self):
        self.inputc.press_key(self.cfg.input.cast_key)
        self.last_cast = time.monotonic()
        time.sleep(CAST_SETTLE)
        self._park()

    def _snapshot(self):
        return self.grabber.snapshot() if self.grabber else None

    def tick(self, ciclo, matched, recording, scores):
        """Un sondeo del watchdog: captura ROI, detecta y actúa. Devuelve (action, outcome)."""
        frame = self.capturer.grab(self.roi.as_mss())
        dets = self.detector.detect(frame)
        action = decide(bool(dets), matched)
        detection = None

        if action == "loot":
            self.consecutive_recasts = 0
            best = max(dets, key=lambda d: d.conf)
            cx, cy = best.center
            x_abs, y_abs = self.roi.left + cx, self.roi.top + cy
            self.log.info("ciclo %d: corcho conf=%.3f -> loot %s en (%d,%d)", ciclo, best.conf,
                          self.cfg.input.loot_button, x_abs, y_abs)
            self.inputc.move_and_click(x_abs, y_abs, button=self.cfg.input.loot_button)
            self._park()
            time.sleep(self.cfg.input.delay_after_click)
            self.cast_and_park()
            detection = {"bbox": [best.x1, best.y1, best.x2, best.y2], "conf": best.conf,
                         "click": [x_abs, y_abs]}
            outcome = "recogido"

        elif action == "recast":
            self.consecutive_recasts += 1
            self.log.info("ciclo %d: sin corcho en la ROI -> recast (#%d)", ciclo, self.consecutive_recasts)
            self.cast_and_park()
            if self.consecutive_recasts >= self.cfg.input.watchdog_warn_after:
                self.log.warning("ciclo %d: recast x%d sin corcho: ¿cámara derivó / casteos fuera de la ROI?",
                                 ciclo, self.consecutive_recasts)
            outcome = "recast_sin_corcho"

        else:  # wait: corcho presente, sin mordida
            self.consecutive_recasts = 0
            best = max(dets, key=lambda d: d.conf)
            detection = {"bbox": [best.x1, best.y1, best.x2, best.y2], "conf": best.conf}
            if time.monotonic() - self.last_cast > self.cfg.audio.listen_window:
                self.log.info("ciclo %d: corcho sin mordida > %ds -> recast de seguridad",
                              ciclo, self.cfg.audio.listen_window)
                self.cast_and_park()
                outcome = "recast_timeout"
            else:
                self.log.debug("ciclo %d: corcho presente (conf=%.3f), esperando", ciclo, best.conf)
                outcome = "esperando"

        self.session.record_cycle(
            ciclo, frame_bgr=frame, audio_int16=recording, frames=self._snapshot(),
            event={"matched": matched, "scores": {k: round(v, 4) for k, v in scores.items()},
                   "detection": detection, "outcome": outcome, "action": action,
                   "consecutive_recasts": self.consecutive_recasts},
        )
        return action, outcome

    def run(self):
        self.log.info("ROI=%s | conf=%.2f | loot=%s | watchdog=%.1fs | park=%s",
                      self.roi.as_mss(), self.cfg.detector.conf_threshold, self.cfg.input.loot_button,
                      self.cfg.input.watchdog_interval, self.cfg.input.mouse_park)
        self.cast_and_park()
        self.log.info("Lanzamiento inicial (tecla '%s')", self.cfg.input.cast_key)
        ciclo = 0
        while True:
            ciclo += 1
            t0 = time.monotonic()
            recording = self.recorder.record(self.cfg.audio.duration, self.cfg.audio.fs)
            matched, scores = match_audio(recording, self.cfg.audio.fs, self.cfg.audio.references, self.cfg.audio)
            self.log.debug("ciclo %d: scores audio=%s", ciclo, {k: round(v, 4) for k, v in scores.items()})
            self.tick(ciclo, matched, recording, scores)
            remaining = self.cfg.input.watchdog_interval - (time.monotonic() - t0)
            if remaining > 0:
                time.sleep(remaining)


def main():
    parser = argparse.ArgumentParser(description="Bot de pesca (loop en vivo)")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING (override de config)")
    parser.add_argument("--record", action="store_true", help="Grabar la sesión en sessions/")
    args = parser.parse_args()

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
    recording_enabled = args.record or CFG.session.enabled
    session = SessionRecorder(
        CFG.session.dir, enabled=recording_enabled, fs=CFG.audio.fs,
        frames_max=CFG.session.frames.max_frames,
    )
    grabber = None
    if recording_enabled and CFG.session.frames.enabled:
        grabber = FrameGrabber(CFG.roi.as_mss(), CFG.session.frames.fps, CFG.session.frames.max_frames)
        grabber.start()
        log.info("Captura de frames activa (%.0f fps, máx %d)", CFG.session.frames.fps,
                 CFG.session.frames.max_frames)

    loop = LootLoop(CFG, log, detector, capturer, inputc, recorder, session, grabber)
    try:
        loop.run()
    except KeyboardInterrupt:
        log.info("Finalizado por el usuario.")
    finally:
        if grabber is not None:
            grabber.stop()


if __name__ == "__main__":
    main()
