from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except ImportError:  # pragma: no cover
    RapidOCR = None

@dataclass
class DetectionConfig:
    backend: str = "paddleocr"
    paddle_lang: str = "ch"
    score_threshold: float = 0.2
    min_area: int = 16
    min_box_size: int = 4
    ocr_device: str = "cpu"  # cpu | cuda


def _normalize_ocr_device(device: str | None) -> str:
    value = (device or "cpu").strip().lower()
    if value in {"cuda", "gpu"}:
        return "cuda"
    return "cpu"


def _build_rapidocr(device: str) -> Any:
    if RapidOCR is None:
        return None
    normalized = _normalize_ocr_device(device)
    if normalized != "cuda":
        return RapidOCR()
    try:
        return RapidOCR(
            det_use_cuda=True,
            rec_use_cuda=True,
            cls_use_cuda=True,
            det_model_path=None,
            rec_model_path=None,
            cls_model_path=None,
        )
    except TypeError:
        warnings.warn(
            "RapidOCR in this environment does not accept CUDA init kwargs; "
            "falling back to CPU.",
            RuntimeWarning,
        )
        return RapidOCR()
    except Exception as exc:
        warnings.warn(
            f"Failed to initialize RapidOCR with CUDA ({exc}); falling back to CPU.",
            RuntimeWarning,
        )
        try:
            return RapidOCR()
        except Exception:
            return None


@dataclass
class TextProposal:
    polygon: np.ndarray  # shape=(N, 2), dtype=float32
    score: float
    source: str

    def to_json(self) -> dict[str, Any]:
        return {
            "polygon": self.polygon.tolist(),
            "score": float(self.score),
            "source": self.source,
        }


class OrientedTextDetector:
    def __init__(self, config: DetectionConfig):
        self.config = config
        self._paddle = None
        self._paddle_error: str | None = None
        self._rapid = None
        if config.backend == "paddleocr":
            try:
                from paddleocr import PaddleOCR  # type: ignore
            except ImportError:
                PaddleOCR = None
            if PaddleOCR is None:
                # PaddleOCR isn't available in this environment; keep going so we can
                # fall back to RapidOCR (or MSER as a last resort).
                self._paddle = None
            try:
                # Older PaddleOCR API
                self._paddle = PaddleOCR(
                    use_angle_cls=False,
                    lang=config.paddle_lang,
                    det=True,
                    rec=False,
                    show_log=False,
                )
            except Exception:
                try:
                    # Newer PaddleOCR API (det/rec flags removed)
                    self._paddle = PaddleOCR(
                        use_angle_cls=False,
                        lang=config.paddle_lang,
                        show_log=False,
                    )
                except Exception as exc:  # pragma: no cover
                    self._paddle = None
                    self._paddle_error = str(exc)
        if RapidOCR is not None:
            self._rapid = _build_rapidocr(config.ocr_device)

    def detect(self, image_gray: np.ndarray) -> list[TextProposal]:
        if self._paddle is not None:
            return self._detect_paddle(image_gray)
        if self._rapid is not None:
            return self._detect_rapidocr(image_gray)
        return self._detect_fallback(image_gray)

    def _detect_paddle(self, image_gray: np.ndarray) -> list[TextProposal]:
        try:
            result = self._paddle.ocr(image_gray, cls=False, det=True, rec=False) or []
        except TypeError:
            # New API compatibility
            result = self._paddle.ocr(image_gray, cls=False) or []
        raw_boxes = result[0] if result else []
        proposals: list[TextProposal] = []
        for item in raw_boxes:
            if len(item) != 2:
                continue
            polygon, score_raw = item
            if isinstance(score_raw, (list, tuple)) and len(score_raw) >= 2:
                # output shape can be (text, score) when recognition is enabled
                score = float(score_raw[1])
            else:
                score = float(score_raw)
            if score < self.config.score_threshold:
                continue
            points = np.asarray(polygon, dtype=np.float32)
            if points.shape[0] < 4:
                continue
            area = cv2.contourArea(points)
            if area < self.config.min_area:
                continue
            proposals.append(
                TextProposal(polygon=points, score=float(score), source="paddleocr")
            )
        return proposals

    def _detect_rapidocr(self, image_gray: np.ndarray) -> list[TextProposal]:
        result, _ = self._rapid(image_gray)
        proposals: list[TextProposal] = []
        for item in result or []:
            if not isinstance(item, list) or len(item) < 3:
                continue
            polygon, _, score_raw = item
            score = float(score_raw)
            if score < self.config.score_threshold:
                continue
            points = np.asarray(polygon, dtype=np.float32)
            if points.shape[0] < 4:
                continue
            area = cv2.contourArea(points)
            if area < self.config.min_area:
                continue
            proposals.append(
                TextProposal(polygon=points, score=score, source="rapidocr")
            )
        return proposals

    def _detect_fallback(self, image_gray: np.ndarray) -> list[TextProposal]:
        """
        Fallback detector uses MSER regions and rotated min-area rectangles.
        It over-generates candidates intentionally for higher recall.
        """
        mser = cv2.MSER_create(min_area=self.config.min_area, max_area=20000)
        regions, _ = mser.detectRegions(image_gray)
        proposals: list[TextProposal] = []
        for region in regions:
            if len(region) < 5:
                continue
            contour = region.reshape((-1, 1, 2)).astype(np.float32)
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect).astype(np.float32)
            w, h = rect[1]
            if min(w, h) < self.config.min_box_size:
                continue
            score = min(1.0, len(region) / 120.0)
            proposals.append(TextProposal(polygon=box, score=score, source="mser"))
        return proposals


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot read page image: {path}")
    return image
