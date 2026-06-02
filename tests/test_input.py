"""Pruebas del timing del clic de loot (InputController, headless con pyautogui inyectado)."""
import platform_io


class FakePG:
    """pyautogui falso: registra el orden exacto de las llamadas."""

    FAILSAFE = False

    def __init__(self, events):
        self._events = events

    def moveTo(self, x, y, **k):
        self._events.append(("moveTo", x, y))

    def mouseDown(self, button=None, **k):
        self._events.append(("mouseDown", button))

    def mouseUp(self, button=None, **k):
        self._events.append(("mouseUp", button))


class FakeKB:
    pass


def test_move_and_click_orden(monkeypatch):
    events = []
    monkeypatch.setattr(platform_io.time, "sleep", lambda s: events.append(("sleep", s)))
    ic = platform_io.InputController(pg=FakePG(events), kb=FakeKB())

    ic.move_and_click(100, 200, button="right", move_settle=0.08, click_hold=0.06)

    assert events == [
        ("moveTo", 100, 200),
        ("sleep", 0.08),       # move_settle: el juego registra la posición
        ("mouseDown", "right"),
        ("sleep", 0.06),       # click_hold: press->release no instantáneo
        ("mouseUp", "right"),
    ]


def test_no_requiere_pyautogui_real():
    import sys

    assert not any(m in sys.modules for m in ("mss", "pyautogui", "keyboard", "sounddevice"))
