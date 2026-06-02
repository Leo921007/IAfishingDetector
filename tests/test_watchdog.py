"""Pruebas de la lógica del watchdog y el parkeo (headless, con adaptadores fake)."""
import numpy as np
import pytest

import main
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


def _loop(dets):
    inp = FakeInput()
    log = FakeLog()
    loop = main.LootLoop(
        CFG, log, FakeDetector(dets), FakeCapturer(), inp,
        recorder=None, session=SessionRecorder(CFG.session.dir, enabled=False, fs=CFG.audio.fs),
        grabber=None,
    )
    return loop, inp, log


def _det():
    return Detection(x1=10, y1=10, x2=20, y2=20, conf=0.9)


def test_decide():
    assert main.decide(False, False) == "recast"
    assert main.decide(False, True) == "recast"
    assert main.decide(True, True) == "loot"
    assert main.decide(True, False) == "wait"


def test_tick_sin_corcho_recasta_y_parkea():
    loop, inp, _ = _loop([])
    action, outcome = loop.tick(1, matched=True, recording=None, scores={})
    assert action == "recast"
    assert inp.casts == 1          # recast
    assert inp.parks               # parkeó
    assert inp.clicks == []        # no hubo loot
    assert loop.consecutive_recasts == 1


def test_tick_loot_y_parkea():
    loop, inp, _ = _loop([_det()])
    action, outcome = loop.tick(1, matched=True, recording=None, scores={})
    assert action == "loot" and outcome == "recogido"
    assert len(inp.clicks) == 1                     # un clic de loot
    assert inp.clicks[0][2] == CFG.input.loot_button
    assert inp.casts == 1                           # recast tras loot
    assert len(inp.parks) >= 2                      # park tras loot y tras recast
    assert loop.consecutive_recasts == 0


def test_tick_wait_no_recasta():
    loop, inp, _ = _loop([_det()])
    loop.last_cast = main.time.monotonic()          # cast reciente -> sin safety timeout
    action, outcome = loop.tick(1, matched=False, recording=None, scores={})
    assert action == "wait" and outcome == "esperando"
    assert inp.casts == 0 and inp.clicks == []


def test_warning_tras_n_recasts():
    loop, inp, log = _loop([])
    for c in range(CFG.input.watchdog_warn_after):
        loop.tick(c + 1, matched=False, recording=None, scores={})
    assert loop.consecutive_recasts == CFG.input.watchdog_warn_after
    assert log.warnings, "debería avisar tras N recasts seguidos sin corcho"


def test_mouse_park_fuera_de_la_roi():
    r = CFG.roi
    px, py = CFG.input.mouse_park
    fuera = px < r.left or px > r.left + r.width or py < r.top or py > r.top + r.height
    assert fuera, "mouse_park debe quedar fuera de la ROI"
