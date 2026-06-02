"""Pruebas de la resolución de modelo y ROI por ubicación (headless)."""
import pytest

from config import REPO_ROOT, load_config, load_roi, resolve_model_path


def test_resolve_model_path_specific():
    assert resolve_model_path("specific", "stormwind") == \
        REPO_ROOT / "locations" / "stormwind" / "detector.onnx"


def test_resolve_model_path_general():
    assert resolve_model_path("general", "stormwind") == \
        REPO_ROOT / "models" / "general" / "detector.onnx"


def test_load_roi_stormwind():
    roi = load_roi("stormwind")
    assert (roi.left, roi.top, roi.width, roi.height) == (586, 126, 748, 387)


def test_load_roi_ubicacion_inexistente_error_claro():
    with pytest.raises(FileNotFoundError):
        load_roi("no_existe_zzz")


def test_load_config_default_specific_stormwind():
    cfg = load_config()
    assert cfg.detector_mode == "specific"
    assert cfg.location == "stormwind"
    assert cfg.model_onnx == REPO_ROOT / "locations" / "stormwind" / "detector.onnx"
    assert (cfg.roi.left, cfg.roi.top, cfg.roi.width, cfg.roi.height) == (586, 126, 748, 387)


def test_load_config_general(tmp_path):
    text = (REPO_ROOT / "config.yaml").read_text(encoding="utf-8").replace(
        "detector_mode: specific", "detector_mode: general"
    )
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.detector_mode == "general"
    assert cfg.model_onnx == REPO_ROOT / "models" / "general" / "detector.onnx"
    assert cfg.roi.width == 748  # la ROI sigue saliendo de la ubicación
