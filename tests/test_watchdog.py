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

    def move_and_click(self, x, y, button="left", move_settle=None, click_hold=None):
        self.clicks.append((x, y, button, move_settle, click_hold))

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


class HighFoamCapturer:
    def grab(self, roi):
        return np.full((30, 30, 3), 255, dtype=np.uint8)  # parche blanco -> foam alto


class FakeClock:
    """Reloj determinista: avanza dt en cada llamada (para timeouts en run_cycle sin esperar real)."""

    def __init__(self, dt):
        self.t = 0.0
        self.dt = dt

    def __call__(self):
        self.t += self.dt
        return self.t


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


def test_run_cycle_foam_dispara_apenas_hay_bbox():
    # detector fija bbox en la 1ª detección; capturer de foam alto -> dispara sin "fase POLL" aparte.
    loop, inp = _loop_seq([[_det()]])
    loop.capturer = HighFoamCapturer()
    loop.clock = FakeClock(0.01)
    _, _, bbox, _, outcome, action = loop.run_cycle(1)
    assert outcome == "recogido" and action == "loot"
    assert bbox is not None
    assert len(inp.clicks) == 1  # do_loot hizo el clic


def test_run_cycle_recast_sin_corcho():
    loop, inp = _loop_seq([[]])      # nunca aparece el corcho
    loop.clock = FakeClock(0.6)      # supera locate_timeout en pocas iteraciones
    _, _, bbox, _, outcome, _ = loop.run_cycle(1)
    assert outcome == "recast_sin_corcho"
    assert bbox is None
    assert inp.casts == 1


def test_run_cycle_recast_perdido():
    loop, inp = _loop_seq([[_det()], [], [], []])  # fija bbox y luego falla el relocate
    loop.clock = FakeClock(1.6)                     # relocate cada iteración (>relocate_seconds)
    _, _, _, _, outcome, _ = loop.run_cycle(1)      # capturer por defecto (foam 0): no dispara
    assert outcome == "recast_perdido"
    assert loop.relocate_fails == CFG.bite.relocate_tolerance


def test_run_cycle_recast_timeout():
    loop, inp = _loop_seq([[_det()]])  # corcho presente, foam 0 -> nunca muerde
    loop.clock = FakeClock(9.0)        # supera max_wait_seconds
    _, _, _, _, outcome, _ = loop.run_cycle(1)
    assert outcome == "recast_timeout"


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


def test_do_loot_parkea_despues_del_clic_con_settle(monkeypatch):
    events = []

    class TimedInput:
        def press_key(self, key):
            events.append(("cast",))

        def move_and_click(self, x, y, button="left", move_settle=None, click_hold=None):
            events.append(("click", move_settle, click_hold))

        def park(self, x, y):
            events.append(("park",))

    monkeypatch.setattr(main.time, "sleep", lambda s: events.append(("sleep", s)))
    loop = main.LootLoop(
        CFG, FakeLog(), FakeDetector([]), FakeCapturer(), TimedInput(),
        session=SessionRecorder(CFG.session.dir, enabled=False, fs=44100),
        bite=FoamBiteDetector(CFG.bite.foam_threshold, CFG.bite.foam_min_frames), grabber=None,
    )
    loop.do_loot(1, _det())

    # clic -> espera loot_settle -> park (park DESPUÉS del clic, con el settle entre medio)
    assert events[0] == ("click", CFG.input.move_settle, CFG.input.click_hold)
    assert events[1] == ("sleep", CFG.input.loot_settle)
    assert events[2] == ("park",)


def test_relocate_tolera_fallos_aislados():
    loop, _, _ = _loop()
    assert loop.note_relocate(None) == "keep"     # 1er fallo
    assert loop.note_relocate(None) == "keep"     # 2º fallo (tolerance=3)
    assert loop.note_relocate(_det()) == "ok"     # éxito resetea el contador
    assert loop.note_relocate(None) == "keep"     # vuelve a empezar


def test_relocate_perdido_tras_tolerance():
    loop, _, _ = _loop()
    res = [loop.note_relocate(None) for _ in range(CFG.bite.relocate_tolerance)]
    assert res[-1] == "lost"
    assert res[:-1] == ["keep"] * (CFG.bite.relocate_tolerance - 1)


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
