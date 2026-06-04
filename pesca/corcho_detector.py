"""Detector de corcho por inferencia ONNX (onnxruntime).

Ruta de DETECCIÓN pura: solo depende de numpy, OpenCV y onnxruntime. **No** importa
mss/pyautogui/keyboard/sounddevice, por lo que es importable y testeable en WSL2 sin
display ni audio.

Modelo: YOLO11n exportado a ONNX (Etapa 2). Entrada [1,3,640,640]; salida [1,5,8400]
con filas (cx, cy, w, h, score) en el espacio de la imagen con letterbox (1 clase: corcho).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def center(self) -> Tuple[int, int]:
        return int((self.x1 + self.x2) / 2), int((self.y1 + self.y2) / 2)


class CorchoDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        providers: Optional[List[str]] = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo ONNX no encontrado: {self.model_path}. "
                "Genera el modelo con la Etapa 2 (train_corcho.py + export ONNX)."
            )
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.imgsz = int(imgsz)
        self.session = ort.InferenceSession(
            str(self.model_path), providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    # ---- pre/post-proceso -------------------------------------------------
    def _letterbox(self, img: np.ndarray, color=(114, 114, 114)):
        """Redimensiona manteniendo aspecto y rellena hasta (imgsz, imgsz)."""
        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        dw, dh = (self.imgsz - nw) / 2, (self.imgsz - nh) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded, r, left, top

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        """Devuelve las detecciones de corcho (xyxy en coords de la imagen de entrada)."""
        if image_bgr is None or image_bgr.size == 0:
            return []
        h0, w0 = image_bgr.shape[:2]
        padded, r, pad_w, pad_h = self._letterbox(image_bgr)

        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None])  # (1,3,H,W)

        out = self.session.run(None, {self.input_name: blob})[0]  # (1,5,8400)
        preds = out[0].T  # (8400, 5)
        scores = preds[:, 4]
        mask = scores >= self.conf_threshold
        preds, scores = preds[mask], scores[mask]
        if len(preds) == 0:
            return []

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        # xywh(centro) -> xyxy, deshaciendo padding y escala del letterbox
        x1 = (cx - bw / 2 - pad_w) / r
        y1 = (cy - bh / 2 - pad_h) / r
        x2 = (cx + bw / 2 - pad_w) / r
        y2 = (cy + bh / 2 - pad_h) / r
        # recortar a los límites de la imagen
        x1 = np.clip(x1, 0, w0); x2 = np.clip(x2, 0, w0)
        y1 = np.clip(y1, 0, h0); y2 = np.clip(y2, 0, h0)

        boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), self.conf_threshold, self.iou_threshold)
        if len(idxs) == 0:
            return []
        idxs = np.array(idxs).flatten()

        return [
            Detection(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(scores[i]))
            for i in idxs
        ]

    def best_center(self, image_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
        """Centro (x, y) del corcho de mayor confianza, o None si no hay detección."""
        dets = self.detect(image_bgr)
        if not dets:
            return None
        return max(dets, key=lambda d: d.conf).center
