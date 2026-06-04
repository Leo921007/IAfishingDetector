"""Pruebas del detector de mordida por foam (headless, secuencias sintéticas)."""
import numpy as np

from pesca.bite_trigger import FoamBiteDetector

BBOX = (30, 30, 50, 50)  # en un frame 80x80; patch_box ~ (25,25,55,55)


def _water():
    return np.zeros((80, 80, 3), dtype=np.uint8)  # agua oscura: foam = 0


def _splash():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[20:60, 20:60] = 255  # blanco en el parche: foam alto
    return img


def test_no_requiere_display():
    import sys

    assert not any(m in sys.modules for m in ("mss", "pyautogui", "keyboard", "sounddevice"))


def test_flote_no_dispara():
    d = FoamBiteDetector(threshold=0.005, min_frames=2)
    for _ in range(10):
        foam, fired = d.update(_water(), BBOX)
        assert foam == 0.0
        assert not fired


def test_splash_dispara_una_sola_vez():
    d = FoamBiteDetector(threshold=0.005, min_frames=2)
    fired_seq = [d.update(_splash(), BBOX)[1] for _ in range(4)]
    # min_frames=2 -> dispara en el 2º frame de splash y NO se repite
    assert fired_seq == [False, True, False, False]


def test_foam_alto_en_splash():
    d = FoamBiteDetector(threshold=0.005, min_frames=1)
    foam, _ = d.update(_splash(), BBOX)
    assert foam > 0.5


def test_reset_re_arma():
    d = FoamBiteDetector(threshold=0.005, min_frames=1)
    assert d.update(_splash(), BBOX)[1] is True     # dispara
    assert d.update(_splash(), BBOX)[1] is False    # ya disparó
    d.reset()
    assert d.update(_splash(), BBOX)[1] is True      # re-armado
