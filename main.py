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
        self.relocate_fails = 0
        self.last_cast = 0.0
        self.clock = time.monotonic  # reloj inyectable (los tests lo reemplazan por uno determinista)

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
        self.inputc.move_and_click(x_abs, y_abs, button=self.cfg.input.loot_button,
                                   move_settle=self.cfg.input.move_settle,
                                   click_hold=self.cfg.input.click_hold)
        time.sleep(self.cfg.input.loot_settle)  # que WoW registre el loot con el cursor sobre el corcho
        self._park()
        time.sleep(self.cfg.input.delay_after_click)
        self.cast_and_park()
        self.bite.reset()

    def note_relocate(self, best_or_none):
        """Tolera fallos aislados del relocate: 'ok' si hay corcho (resetea); 'keep' si falla pero
        aún por debajo de la tolerancia (mantener bbox y seguir); 'lost' tras relocate_tolerance fallos."""
        if best_or_none is not None:
            self.relocate_fails = 0
            return "ok"
        self.relocate_fails += 1
        return "lost" if self.relocate_fails >= self.cfg.bite.relocate_tolerance else "keep"

    def _record(self, ciclo, frame, detection, outcome, action):
        self.session.record_cycle(
            ciclo, frame_bgr=frame, audio_int16=None,
            frames=(self.grabber.snapshot() if self.grabber else None),
            event={"detection": detection, "outcome": outcome, "action": action,
                   "consecutive_recasts": self.consecutive_recasts},
        )

    def run_cycle(self, ciclo):
        """Un ciclo: castear ya hecho; muestrea foam de forma CONTINUA hasta loot o recast.

        Loop único a poll_fps: el foam (si ya hay bbox) tiene PRIORIDAD; el YOLO de tracking corre solo
        cada relocate_seconds para fijar/actualizar el bbox (no en cada frame, para no frenar el foam).
        La 1ª detección arranca el foam de inmediato. Devuelve (frame, best, bbox, last_foam, outcome, action).
        """
        self.bite.reset()
        self.relocate_fails = 0
        best = None
        bbox = None
        last_foam = 0.0
        interval = 1.0 / max(1.0, self.cfg.bite.poll_fps)
        start = self.clock()
        last_relocate = start - self.cfg.bite.relocate_seconds  # fuerza detect en la 1ª iteración

        while True:
            t = self.clock()
            frame = self.capturer.grab(self.roi.as_mss())

            # (1) FOAM primero (PRIORIDAD): solo si ya hay bbox.
            if bbox is not None:
                last_foam, fired = self.bite.update(frame, bbox)
                if fired:
                    self.do_loot(ciclo, best)
                    return frame, best, bbox, last_foam, "recogido", "loot"

            # (2) YOLO de tracking cada relocate_seconds (fijar el 1er bbox o actualizarlo).
            if t - last_relocate >= self.cfg.bite.relocate_seconds:
                last_relocate = t
                dets = self.detector.detect(frame)
                newbest = max(dets, key=lambda d: d.conf) if dets else None
                if bbox is None:
                    if newbest is not None:  # 1ª detección -> arranca el foam ya
                        best, bbox = newbest, (newbest.x1, newbest.y1, newbest.x2, newbest.y2)
                        self.relocate_fails = 0
                        self.consecutive_recasts = 0
                else:
                    status = self.note_relocate(newbest)
                    if status == "lost":
                        self.do_recast(ciclo, "corcho perdido (%d fallos)" % self.relocate_fails)
                        return frame, best, bbox, last_foam, "recast_perdido", "recast"
                    if status == "ok":
                        best, bbox = newbest, (newbest.x1, newbest.y1, newbest.x2, newbest.y2)
                    # "keep": mantener el último bbox y seguir muestreando foam

            # (2b) nunca apareció el corcho dentro de locate_timeout.
            if bbox is None and t - start > self.cfg.bite.locate_timeout:
                self.do_recast(ciclo, "sin corcho tras esperar")
                return frame, best, bbox, last_foam, "recast_sin_corcho", "recast"

            # (3) safety: corcho presente pero sin mordida demasiado tiempo.
            if t - start > self.cfg.bite.max_wait_seconds:
                self.do_recast(ciclo, "sin mordida (timeout)")
                return frame, best, bbox, last_foam, "recast_timeout", "recast"

            slept = interval - (self.clock() - t)
            if slept > 0:
                time.sleep(slept)

    def run(self):
        self.log.info("ROI=%s | conf=%.2f | loot=%s | poll=%.0ffps | foam>%.4f x%d | relocate=%.1fs | park=%s",
                      self.roi.as_mss(), self.cfg.detector.conf_threshold, self.cfg.input.loot_button,
                      self.cfg.bite.poll_fps, self.cfg.bite.foam_threshold, self.cfg.bite.foam_min_frames,
                      self.cfg.bite.relocate_seconds, self.cfg.input.mouse_park)
        self.cast_and_park()
        ciclo = 0
        while True:
            ciclo += 1
            frame, best, bbox, last_foam, outcome, action = self.run_cycle(ciclo)
            detection = None
            if bbox is not None:
                detection = {"bbox": [round(v, 1) for v in bbox], "conf": round(best.conf, 3),
                             "foam": round(last_foam, 4)}
            self._record(ciclo, frame, detection, outcome, action)


def main():
    parser = argparse.ArgumentParser(description="Bot de pesca (loop visual en vivo)")
    parser.add_argument("--log-level", default=None, help="DEBUG/INFO/WARNING (override de config)")
    parser.add_argument("--record", action="store_true", help="Grabar la sesión en sessions/")
    args = parser.parse_args()

    log = setup_logging(CFG.logging, level_override=args.log_level)
    log.info("Detector: modo=%s ubicación=%s -> %s", CFG.detector_mode, CFG.location, CFG.model_onnx)
    if not CFG.model_onnx.exists():
        log.error("No existe el modelo: %s", CFG.model_onnx)
        raise SystemExit(
            f"No existe el modelo para detector_mode='{CFG.detector_mode}' location='{CFG.location}':\n"
            f"  {CFG.model_onnx}\n"
            "Coloca ahí el .onnx (gitignored) o entrena la ubicación; o cambia detector_mode/location "
            "en config.yaml. Ver locations/README.md."
        )
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
