"""Pruebas offline de la ruta de detección (sin display ni audio).

Se omiten (skip) si el modelo ONNX no está presente (es gitignored; se genera en la Etapa 2).
"""
from pathlib import Path

import pytest

from pesca.config import load_config

REPO = Path(__file__).resolve().parents[1]
MODEL = load_config().model_onnx  # ruta resuelta por detector_mode/location
VAL_DIR = REPO / "dataset" / "images" / "val"

needs_model = pytest.mark.skipif(
    not MODEL.exists(), reason="modelo ONNX ausente (gitignored); ejecutar la Etapa 2"
)


def test_detector_importable_sin_display_ni_audio():
    """Importar la ruta de detección no debe requerir mss/pyautogui/keyboard/sounddevice."""
    from pesca import corcho_detector

    assert hasattr(corcho_detector, "CorchoDetector")
    import sys

    for mod in ("mss", "pyautogui", "keyboard", "sounddevice"):
        assert mod not in sys.modules, f"{mod} no debería cargarse al importar el detector"


@needs_model
def test_deteccion_en_imagen_de_validacion():
    import cv2

    from pesca.corcho_detector import CorchoDetector

    detector = CorchoDetector(MODEL, conf_threshold=0.4)
    imgs = sorted(VAL_DIR.glob("*.jpg"))
    assert imgs, "faltan imágenes de validación"

    con_deteccion = 0
    for p in imgs[:5]:
        img = cv2.imread(str(p))
        if detector.detect(img):
            con_deteccion += 1
    assert con_deteccion >= 1, "ninguna de las primeras imágenes de val produjo detección"
