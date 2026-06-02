"""Loop principal del bot de pesca — 100% VISUAL (Etapa 8C).

El trigger de la mordida es el SPLASH (foam): la fracción de píxeles casi-blancos en un parche alrededor
del corcho (ver splash.py / bite_trigger.py; calibrado en 8B). El audio se eliminó por ruidoso.

Arquitectura **localizar-luego-foam** (robusta en CPU): correr YOLO para UBICAR el corcho es caro y no
aguanta ~30 fps por frame; en cambio el foam sobre un parche fijo es barato. Por eso el loop localiza el
corcho con el detector, samplea el foam a `poll_fps`, y **re-localiza** cada `relocate_seconds`.

Se mantienen el WATCHDOG (sin corcho -> recast) y el PARKEO del cursor (tras loot y recast). La lógica
(`decide`, `do_recast`, `do_loot`) es testeable headless. El loop en vivo depende de mss/pyautogui/keyboard
(Windows) solo al instanciar los adaptadores.
"""
import argparse
import time

from config import load_config
from bite_trigger import FoamBiteDetector
from corcho_detector import CorchoDetector
from logging_setup import setup_logging
from platform_io import FrameGrabber, InputController, ScreenCapturer
from session import SessionRecorder

CFG = load_config()

CAST_SETTLE = 0.3   # s tras castear antes de parkear el cursor
SESSION_FS = 44100  # fs nominal para SessionRecorder (sin audio: no se usa)


def decide(has_corcho: bool, bite: bool) -> str:
    """Decisión del loop: 'recast' si no hay corcho; 'loot' si corcho+mordida; 'wait' si corcho sin mordida."""
    if not has_corcho:
        return "recast"
    if bite:
        return "loot"
    return "wait"


class LootLoop:
    """Loop de pesca visual con watchdog de recast, parkeo del cursor y trigger de foam (adaptadores inyectados)."""

    def __init__(self, cfg, log, detector, capturer, inputc, session, bite, grabber=None):
        self.cfg = cfg
        self.log = log
        self.detector = detector
        self.capturer = capturer
        self.inputc = inputc
        self.session = session
        self.bite = bite
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

    def do_recast(self, ciclo, motivo):
        self.consecutive_recasts += 1
        self.log.info("ciclo %d: %s -> recast (#%d)", ciclo, motivo, self.consecutive_recasts)
        self.cast_and_park()
        if self.consecutive_recasts >= self.cfg.input.watchdog_warn_after:
            self.log.warning("ciclo %d: recast x%d sin corcho: ¿cámara derivó / casteos fuera de la ROI?",
                             ciclo, self.consecutive_recasts)

    def do_loot(self, ciclo, best):
        self.consecutive_recasts = 0
        cx, cy = best.center
        x_abs, y_abs = self.roi.left + cx, self.roi.top + cy
        self.log.info("ciclo %d: loot %s en (%d,%d) conf=%.3f", ciclo, self.cfg.input.loot_button,
                      x_abs, y_abs, best.conf)
        self.inputc.move_and_click(x_abs, y_abs, button=self.cfg.input.loot_button)
        self._park()
        time.sleep(self.cfg.input.delay_after_click)
        self.cast_and_park()
        self.bite.reset()

    def _record(self, ciclo, frame, detection, outcome, action):
        self.session.record_cycle(
            ciclo, frame_bgr=frame, audio_int16=None,
            frames=(self.grabber.snapshot() if self.grabber else None),
            event={"detection": detection, "outcome": outcome, "action": action,
                   "consecutive_recasts": self.consecutive_recasts},
        )

    def run(self):
        self.log.info("ROI=%s | conf=%.2f | loot=%s | poll=%.0ffps | foam>%.4f x%d | park=%s",
                      self.roi.as_mss(), self.cfg.detector.conf_threshold, self.cfg.input.loot_button,
                      self.cfg.bite.poll_fps, self.cfg.bite.foam_threshold, self.cfg.bite.foam_min_frames,
                      self.cfg.input.mouse_park)
        self.cast_and_park()
        interval = 1.0 / max(1.0, self.cfg.bite.poll_fps)
        ciclo = 0
        while True:
            ciclo += 1
            # --- LOCATE: ubicar el corcho con el detector ---
            frame = self.capturer.grab(self.roi.as_mss())
            dets = self.detector.detect(frame)
            if not dets:
                self.do_recast(ciclo, "sin corcho en la ROI")
                self._record(ciclo, frame, None, "recast_sin_corcho", "recast")
                continue
            best = max(dets, key=lambda d: d.conf)
            bbox = (best.x1, best.y1, best.x2, best.y2)
            self.bite.reset()
            self.consecutive_recasts = 0
            wait_start = last_relocate = time.monotonic()

            # --- POLL: samplear foam a poll_fps, re-localizando cada relocate_seconds ---
            last_foam, outcome, action = 0.0, None, None
            while True:
                t = time.monotonic()
                frame = self.capturer.grab(self.roi.as_mss())
                last_foam, fired = self.bite.update(frame, bbox)
                if fired:
                    self.do_loot(ciclo, best)
                    outcome, action = "recogido", "loot"
                    break
                if t - wait_start > self.cfg.bite.max_wait_seconds:
                    self.do_recast(ciclo, "sin mordida (timeout)")
                    outcome, action = "recast_timeout", "recast"
                    break
                if t - last_relocate >= self.cfg.bite.relocate_seconds:
                    dets = self.detector.detect(frame)
                    if not dets:
                        self.do_recast(ciclo, "corcho perdido")
                        outcome, action = "recast_perdido", "recast"
                        break
                    best = max(dets, key=lambda d: d.conf)
                    bbox = (best.x1, best.y1, best.x2, best.y2)
                    last_relocate = t
                slept = interval - (time.monotonic() - t)
                if slept > 0:
                    time.sleep(slept)

            detection = {"bbox": [round(v, 1) for v in bbox], "conf": round(best.conf, 3),
                         "foam": round(last_foam, 4)}
            self._record(ciclo, frame, detection, outcome, action)


def main():
    parser = argparse.ArgumentParser(description="Bot de pesca (loop visual en vivo)")
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
    recording_enabled = args.record or CFG.session.enabled
    session = SessionRecorder(
        CFG.session.dir, enabled=recording_enabled, fs=SESSION_FS,
        frames_max=CFG.session.frames.max_frames,
    )
    grabber = None
    if recording_enabled and CFG.session.frames.enabled:
        grabber = FrameGrabber(CFG.roi.as_mss(), CFG.session.frames.fps, CFG.session.frames.max_frames)
        grabber.start()
        log.info("Captura de frames activa (%.0f fps, máx %d)", CFG.session.frames.fps,
                 CFG.session.frames.max_frames)
    bite = FoamBiteDetector(CFG.bite.foam_threshold, CFG.bite.foam_min_frames)

    loop = LootLoop(CFG, log, detector, capturer, inputc, session, bite, grabber)
    try:
        loop.run()
    except KeyboardInterrupt:
        log.info("Finalizado por el usuario.")
    finally:
        if grabber is not None:
            grabber.stop()


if __name__ == "__main__":
    main()
