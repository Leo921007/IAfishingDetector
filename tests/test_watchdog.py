"""Pruebas de la lógica del watchdog, parkeo y decisión visual (headless, adaptadores fake)."""
import numpy as np
import pytest

import main
from bite_trigger import FoamBiteDetector
from config import load_config
from corcho_detector import Detection
from session import SessionRecorder

CFG = load_config()


class FakeInput:
    def __init__(self):
        self.casts = 0
        self.clicks = []
        self.parks = []

    def press_key(self, key):
        self.casts += 1

    def move_and_click(self, x, y, button="left"):
        self.clicks.append((x, y, button))

    def park(self, x, y):
        self.parks.append((x, y))


class FakeCapturer:
    def grab(self, roi):
        return np.zeros((10, 10, 3), dtype=np.uint8)


class FakeDetector:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, frame):
        return list(self.dets)


class SeqDetector:
    """Devuelve cada respuesta de la lista por llamada; repite la última al agotarse."""

    def __init__(self, responses):
        self.responses = [list(r) for r in responses]
        self.calls = 0

    def detect(self, frame):
        r = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return list(r)


class FakeLog:
    def __init__(self):
        self.warnings = []

    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def warning(self, *a, **k):
        self.warnings.append(a)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(main.time, "sleep", lambda *a, **k: None)


def _loop(dets=None):
    inp = FakeInput()
    log = FakeLog()
    loop = main.LootLoop(
        CFG, log, FakeDetector(dets or []), FakeCapturer(), inp,
        session=SessionRecorder(CFG.session.dir, enabled=False, fs=44100),
        bite=FoamBiteDetector(CFG.bite.foam_threshold, CFG.bite.foam_min_frames),
        grabber=None,
    )
    return loop, inp, log


def _loop_seq(responses):
    inp = FakeInput()
    log = FakeLog()
    loop = main.LootLoop(
        CFG, log, SeqDetector(responses), FakeCapturer(), inp,
        session=SessionRecorder(CFG.session.dir, enabled=False, fs=44100),
        bite=FoamBiteDetector(CFG.bite.foam_threshold, CFG.bite.foam_min_frames),
        grabber=None,
    )
    return loop, inp


def _det():
    return Detection(x1=10, y1=10, x2=20, y2=20, conf=0.9)


def test_locate_encuentra_tras_intentos_no_recasta():
    loop, inp = _loop_seq([[], [], [_det()]])  # aparece al 3er intento
    best, _ = loop.locate(1)
    assert best is not None
    assert inp.casts == 0  # locate no recastea


def test_locate_timeout_devuelve_none():
    loop, inp = _loop_seq([[]])  # nunca aparece
    best, _ = loop.locate(1)
    assert best is None
    # acotado: ~locate_timeout * LOCATE_FPS intentos
    assert loop.detector.calls == max(1, round(CFG.bite.locate_timeout * main.LOCATE_FPS))


def test_locate_none_dispara_recast():
    loop, inp = _loop_seq([[]])
    best, _ = loop.locate(1)
    assert best is None
    loop.do_recast(1, "sin corcho tras esperar")
    assert inp.casts == 1
    assert loop.consecutive_recasts == 1


def test_decide():
    assert main.decide(False, False) == "recast"
    assert main.decide(False, True) == "recast"
    assert main.decide(True, True) == "loot"
    assert main.decide(True, False) == "wait"


def test_do_recast_incrementa_y_parkea():
    loop, inp, _ = _loop()
    loop.do_recast(1, "sin corcho")
    assert inp.casts == 1
    assert inp.parks
    assert loop.consecutive_recasts == 1


def test_do_loot_clic_park_recast():
    loop, inp, _ = _loop()
    loop.consecutive_recasts = 3
    loop.do_loot(1, _det())
    assert len(inp.clicks) == 1
    assert inp.clicks[0][2] == CFG.input.loot_button
    assert inp.casts == 1          # recast tras loot
    assert len(inp.parks) >= 2     # park tras loot y tras recast
    assert loop.consecutive_recasts == 0


def test_warning_tras_n_recasts():
    loop, inp, log = _loop()
    for c in range(CFG.input.watchdog_warn_after):
        loop.do_recast(c + 1, "sin corcho")
    assert loop.consecutive_recasts == CFG.input.watchdog_warn_after
    assert log.warnings


def test_mouse_park_fuera_de_la_roi():
    r = CFG.roi
    px, py = CFG.input.mouse_park
    fuera = px < r.left or px > r.left + r.width or py < r.top or py > r.top + r.height
    assert fuera
