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
    allow_fallback: bool = True


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


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_paddle_det_items(result: Any) -> list[tuple[Any, float]]:
    """
    Normalize PaddleOCR detection outputs across API versions.

    Returns a list of (polygon_like, score).
    """
    normalized: list[tuple[Any, float]] = []

    def append_from_entry(entry: Any) -> None:
        # Classic entry: [polygon, score]
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            normalized.append((entry[0], _safe_float(entry[1], 1.0)))
            return

        # Newer structured dict outputs.
        if isinstance(entry, dict):
            polys = entry.get("dt_polys") or entry.get("polys") or entry.get("boxes")
            scores = entry.get("dt_scores") or entry.get("scores")
            if polys is None:
                return
            if isinstance(polys, np.ndarray):
                polys_iter = polys.tolist()
            else:
                polys_iter = list(polys)
            scores_iter = []
            if scores is not None:
                if isinstance(scores, np.ndarray):
                    scores_iter = scores.tolist()
                else:
                    try:
                        scores_iter = list(scores)
                    except Exception:
                        scores_iter = []
            for idx, poly in enumerate(polys_iter):
                score = _safe_float(scores_iter[idx], 1.0) if idx < len(scores_iter) else 1.0
                normalized.append((poly, score))
            return

        # Paddle result objects may expose attributes instead of dict keys.
        for poly_attr, score_attr in (
            ("dt_polys", "dt_scores"),
            ("polys", "scores"),
            ("boxes", "scores"),
        ):
            polys = getattr(entry, poly_attr, None)
            if polys is None:
                continue
            scores = getattr(entry, score_attr, None)
            if isinstance(polys, np.ndarray):
                polys_iter = polys.tolist()
            else:
                polys_iter = list(polys)
            scores_iter = []
            if scores is not None:
                if isinstance(scores, np.ndarray):
                    scores_iter = scores.tolist()
                else:
                    try:
                        scores_iter = list(scores)
                    except Exception:
                        scores_iter = []
            for idx, poly in enumerate(polys_iter):
                score = _safe_float(scores_iter[idx], 1.0) if idx < len(scores_iter) else 1.0
                normalized.append((poly, score))
            return

    if result is None:
        return normalized

    # Classic PaddleOCR 2.x: [ [ [poly, score], ... ] ]
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        if isinstance(first, (list, tuple)):
            # Could be either list-of-boxes or [poly,score].
            if len(first) == 2 and not isinstance(first[0], (list, tuple, np.ndarray)):
                append_from_entry(result)
                return normalized
            for item in first:
                append_from_entry(item)
            if normalized:
                return normalized
        for entry in result:
            append_from_entry(entry)
        return normalized

    append_from_entry(result)
    return normalized


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
                self._paddle_error = "paddleocr package is not importable"
            use_gpu = _normalize_ocr_device(config.ocr_device) == "cuda"
            ctor_options = [
                # Prefer disabling all orientation/unwarp helpers to keep box coordinates
                # in the same frame as the original rendered page.
                dict(
                    lang=config.paddle_lang,
                    det=True,
                    rec=False,
                    show_log=False,
                    use_angle_cls=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                ),
                dict(
                    lang=config.paddle_lang,
                    show_log=False,
                    use_angle_cls=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                ),
                dict(
                    lang=config.paddle_lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                ),
                # Compatibility fallbacks for older APIs.
                dict(
                    use_angle_cls=False,
                    lang=config.paddle_lang,
                    det=True,
                    rec=False,
                    show_log=False,
                ),
                dict(use_angle_cls=False, lang=config.paddle_lang, show_log=False, enable_mkldnn=False),
                dict(use_angle_cls=False, lang=config.paddle_lang, show_log=False),
                dict(lang=config.paddle_lang, enable_mkldnn=False),
                dict(lang=config.paddle_lang),
            ]
            if use_gpu:
                # Try GPU-enabled constructors first. Some PaddleOCR versions/builds
                # don't accept `use_gpu` or don't have CUDA support; we'll fall back.
                gpu_first = []
                for kw in ctor_options:
                    kw2 = dict(kw)
                    kw2["use_gpu"] = True
                    gpu_first.append(kw2)
                ctor_options = gpu_first + ctor_options
            last_exc: Exception | None = None
            for kwargs in ctor_options:
                try:
                    self._paddle = PaddleOCR(**kwargs)
                    last_exc = None
                    break
                except Exception as exc:  # pragma: no cover
                    last_exc = exc
                    self._paddle = None
            if self._paddle is None and last_exc is not None:
                self._paddle_error = str(last_exc)
        if config.allow_fallback and RapidOCR is not None:
            self._rapid = _build_rapidocr(config.ocr_device)

    def detect(self, image_gray: np.ndarray) -> list[TextProposal]:
        if self._paddle is not None:
            return self._detect_paddle(image_gray)
        if not self.config.allow_fallback:
            detail = self._paddle_error or "unknown paddle initialization error"
            raise RuntimeError(f"PaddleOCR detector unavailable: {detail}")
        if self._rapid is not None:
            return self._detect_rapidocr(image_gray)
        return self._detect_fallback(image_gray)

    def _detect_paddle(self, image_gray: np.ndarray) -> list[TextProposal]:
        # PaddleOCR 3.x/PaddleX pipelines expect 3-channel images.
        image_input = image_gray
        if isinstance(image_gray, np.ndarray) and image_gray.ndim == 2:
            image_input = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        try:
            result = self._paddle.ocr(image_input, cls=False, det=True, rec=False) or []
        except Exception:
            # New API compatibility
            try:
                result = self._paddle.ocr(image_input, cls=False) or []
            except Exception:
                try:
                    # PaddleOCR 3.x may reject cls/det/rec kwargs and route to predict().
                    result = self._paddle.ocr(image_input) or []
                except Exception:
                    if hasattr(self._paddle, "predict"):
                        predicted = self._paddle.predict(image_input)
                        try:
                            result = list(predicted)
                        except Exception:
                            result = predicted or []
                    else:
                        raise
        raw_boxes = _extract_paddle_det_items(result)
        proposals: list[TextProposal] = []
        for item in raw_boxes:
            polygon, score_raw = item
            score = _safe_float(score_raw, 1.0)
            if score < self.config.score_threshold:
                continue
            points = np.asarray(polygon, dtype=np.float32)
            if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] < 2:
                continue
            area = cv2.contourArea(points)
            if area < self.config.min_area:
                continue
            proposals.append(
                TextProposal(polygon=points, score=float(score), source="paddleocr")
            )
        if not proposals and not self.config.allow_fallback:
            raise RuntimeError(
                "PaddleOCR detector produced no usable polygons for this page. "
                "Check Paddle API compatibility / runtime logs."
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
