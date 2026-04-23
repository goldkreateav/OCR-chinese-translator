from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from typing import Any
import warnings

import cv2
import numpy as np

from .detect import TextProposal

try:
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except ImportError:  # pragma: no cover
    RapidOCR = None

_WORKER_RECOGNIZER: "RegionTextRecognizer | None" = None
_THREAD_LOCAL = threading.local()


@dataclass
class RecognitionConfig:
    backend: str = "rapidocr"
    paddle_python: str | None = None
    paddle_bridge_script: str | None = None
    lang: str = "ch"
    min_confidence: float = 0.0
    min_text_length: int = 1
    use_multipass: bool = False
    mode: str = "fast"  # fast | accurate | hybrid
    fast_confidence_threshold: float = 0.74
    retry_enabled: bool = True
    retry_confidence_threshold: float = 0.72
    accept_confidence_threshold: float = 0.90
    accept_score_threshold: float = 1.02
    accept_min_length: int = 6
    backend_cascade: bool = True
    cascade_probe_variants: int = 2
    parallel_min_regions: int = 16
    profile_ocr: bool = True
    profile_variant_calls: bool = True
    bridge_fallback_enabled: bool = True
    bridge_fallback_confidence_threshold: float = 0.80
    bridge_fallback_score_threshold: float = 0.96
    bridge_fallback_max_pixels: int = 220000
    parallel_ocr: bool = False
    max_workers: int = 1
    batch_size: int = 64
    ocr_device: str = "cpu"  # cpu | cuda
    allow_fallback: bool = True


def should_use_angle_classifier(config: RecognitionConfig) -> bool:
    mode = (config.mode or "").lower()
    if mode == "accurate":
        return True
    return bool(config.retry_enabled and config.backend == "paddleocr")


def _normalize_ocr_device(device: str | None) -> str:
    value = (device or "cpu").strip().lower()
    if value in {"cuda", "gpu"}:
        return "cuda"
    return "cpu"


def _rapidocr_init_kwargs(device: str) -> dict[str, Any]:
    use_cuda = _normalize_ocr_device(device) == "cuda"
    return {
        "det_use_cuda": use_cuda,
        "rec_use_cuda": use_cuda,
        "cls_use_cuda": use_cuda,
        # Current rapidocr_onnxruntime updater expects these keys when det/rec/cls
        # kwargs are provided; keep defaults by passing None.
        "det_model_path": None,
        "rec_model_path": None,
        "cls_model_path": None,
    }


def _build_rapidocr(device: str) -> Any:
    if RapidOCR is None:
        return None
    normalized = _normalize_ocr_device(device)
    if normalized != "cuda":
        return RapidOCR()
    try:
        return RapidOCR(**_rapidocr_init_kwargs(normalized))
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


class RegionTextRecognizer:
    def __init__(self, config: RecognitionConfig):
        self.config = config
        self._use_angle_cls = should_use_angle_classifier(config)
        self._paddle = None
        self._paddle_bridge: _PaddleOcrBridge | None = None
        self._rapid = None
        self._init_error: str | None = None
        self._last_trace: dict[str, Any] = {}
        if config.backend != "paddleocr":
            if config.backend == "rapidocr" and RapidOCR is not None:
                self._rapid = _build_rapidocr(config.ocr_device)
                if self._rapid is None and not config.allow_fallback:
                    raise RuntimeError("RapidOCR backend is unavailable in strict mode.")
            # Allow optional high-quality fallback through external Paddle bridge.
            if config.bridge_fallback_enabled and (config.paddle_python or os.getenv("OCR_PADDLE_PYTHON")):
                self._paddle_bridge = _maybe_build_paddle_bridge(config)
            return
        # If a separate Paddle environment is provided, prefer the subprocess bridge.
        # This avoids importing PaddleOCR into the current interpreter (which may be
        # incompatible or trigger heavyweight initialization).
        if config.paddle_python or os.getenv("OCR_PADDLE_PYTHON"):
            self._paddle_bridge = _maybe_build_paddle_bridge(config)
            if self._paddle_bridge is None:
                self._init_error = "Paddle bridge is configured but unavailable."
            if self._paddle_bridge is None and config.allow_fallback and RapidOCR is not None:
                self._rapid = _build_rapidocr(config.ocr_device)
            if self._paddle_bridge is None and not config.allow_fallback:
                raise RuntimeError(self._init_error or "Paddle bridge is unavailable in strict mode.")
            return
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except ImportError:
            PaddleOCR = None
        if PaddleOCR is None:
            self._paddle_bridge = _maybe_build_paddle_bridge(config)
            if self._paddle_bridge is None:
                self._init_error = "paddleocr package is not importable and bridge is unavailable."
            if self._paddle_bridge is None and config.allow_fallback and RapidOCR is not None:
                self._rapid = _build_rapidocr(config.ocr_device)
            if self._paddle_bridge is None and not config.allow_fallback:
                raise RuntimeError(self._init_error)
            return
        ctor_options = [
            dict(
                lang=config.lang,
                det=False,
                rec=True,
                show_log=False,
                use_angle_cls=self._use_angle_cls,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            ),
            dict(
                lang=config.lang,
                show_log=False,
                use_angle_cls=self._use_angle_cls,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            ),
            dict(
                lang=config.lang,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            ),
            dict(
                use_angle_cls=self._use_angle_cls,
                lang=config.lang,
                det=False,
                rec=True,
                show_log=False,
            ),
            dict(use_angle_cls=self._use_angle_cls, lang=config.lang),
            dict(lang=config.lang),
        ]
        last_exc: Exception | None = None
        for kwargs in ctor_options:
            try:
                self._paddle = PaddleOCR(**kwargs)
                last_exc = None
                break
            except Exception as exc:
                self._paddle = None
                last_exc = exc

        if self._paddle is None:
            self._paddle_bridge = _maybe_build_paddle_bridge(config)
            self._init_error = str(last_exc) if last_exc is not None else "Unknown Paddle init error"
            if self._paddle_bridge is None and config.allow_fallback and RapidOCR is not None:
                self._rapid = _build_rapidocr(config.ocr_device)
            if self._paddle_bridge is None and not config.allow_fallback:
                raise RuntimeError(
                    f"PaddleOCR recognizer initialization failed in strict mode: {self._init_error}"
                )

    def consume_last_trace(self) -> dict[str, Any]:
        trace = dict(self._last_trace)
        self._last_trace = {}
        return trace

    def _is_result_confident(self, text: str, confidence: float, score: float) -> bool:
        return (
            confidence >= float(self.config.accept_confidence_threshold)
            and score >= float(self.config.accept_score_threshold)
            and useful_length(text) >= int(self.config.accept_min_length)
        )

    @staticmethod
    def _dedupe_variants(
        variants: list[tuple[str, np.ndarray]], already_seen: set[str]
    ) -> list[tuple[str, np.ndarray]]:
        deduped: list[tuple[str, np.ndarray]] = []
        for variant_name, variant_img in variants:
            if variant_name in already_seen:
                continue
            already_seen.add(variant_name)
            deduped.append((variant_name, variant_img))
        return deduped

    def _record_trace(self, payload: dict[str, Any]) -> None:
        if self.config.profile_ocr:
            self._last_trace = payload
        else:
            self._last_trace = {}

    def _should_bridge_fallback(
        self,
        roi: np.ndarray,
        text: str,
        confidence: float,
        score: float,
    ) -> bool:
        if not self.config.bridge_fallback_enabled:
            return False
        if self._paddle_bridge is None:
            return False
        h, w = roi.shape[:2]
        if h <= 0 or w <= 0:
            return False
        if h * w > int(self.config.bridge_fallback_max_pixels):
            return False
        compact_len = useful_length(text)
        if compact_len == 0:
            return True
        if confidence < float(self.config.bridge_fallback_confidence_threshold):
            return True
        if score < float(self.config.bridge_fallback_score_threshold):
            return True
        return False

    def recognize_polygon(
        self, image_gray: np.ndarray, polygon: np.ndarray
    ) -> tuple[str, float, str, float]:
        roi = warp_polygon_roi(image_gray, polygon)
        if roi.size == 0:
            roi = rotate_crop_roi(image_gray, polygon)
        return self.recognize_roi(roi)

    def recognize_roi(self, roi: np.ndarray) -> tuple[str, float, str, float]:
        if roi.size == 0:
            return "", 0.0, "empty", 0.0

        roi_h, roi_w = roi.shape[:2]
        started_at = time.perf_counter()
        mode = (self.config.mode or "hybrid").lower()
        evaluated: list[dict[str, Any]] = []
        seen_variants: set[str] = set()

        def evaluate_stage(
            stage_name: str,
            variants: list[tuple[str, np.ndarray]],
            *,
            prefer_bridge: bool = False,
        ) -> tuple[str, float, str, float]:
            stage_start = time.perf_counter()
            text, conf, variant, score = self._recognize_backend(
                variants,
                allow_cascade=True,
                prefer_bridge=prefer_bridge,
            )
            stage_ms = (time.perf_counter() - stage_start) * 1000.0
            stage_trace = self.consume_last_trace()
            evaluated.append(
                {
                    "stage": stage_name,
                    "variant_count": len(variants),
                    "elapsed_ms": stage_ms,
                    "winner_variant": variant,
                    "winner_confidence": float(conf),
                    "winner_score": float(score),
                    "winner_text_len": useful_length(text),
                    "backend_trace": stage_trace,
                }
            )
            return text, conf, variant, score

        fast_variants = self._dedupe_variants(
            build_roi_variants_for_mode(roi, self.config, fast_only=True),
            seen_variants,
        )
        best_text, best_conf, best_variant, best_score = evaluate_stage("fast", fast_variants)

        def maybe_bridge_fallback() -> None:
            nonlocal best_text, best_conf, best_variant, best_score
            if not self._should_bridge_fallback(
                roi=roi,
                text=best_text,
                confidence=best_conf,
                score=best_score,
            ):
                return
            # Run bridge on an independent variant set even if rapid already evaluated
            # similarly named variants; the backend quality is different.
            bridge_variants = build_roi_variants(roi, multipass=False)
            if not bridge_variants:
                return
            text3, conf3, variant3, score3 = evaluate_stage(
                "bridge_fallback",
                bridge_variants,
                prefer_bridge=True,
            )
            if score3 > best_score:
                best_text, best_conf, best_variant, best_score = text3, conf3, variant3, score3

        # Fast extra pass: only if we got nothing or very low confidence.
        if (useful_length(best_text) == 0 or best_conf < 0.50) and roi.size > 0:
            roi_up = upscale_if_small(roi)
            h, w = roi_up.shape[:2]
            include_quarter_turns = bool(h > w * 1.15)
            binarized = cv2.adaptiveThreshold(
                roi_up,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31,
                15,
            )
            extra = self._dedupe_variants(
                oriented_variants(binarized, "binarize", include_quarter_turns=include_quarter_turns),
                seen_variants,
            )
            if extra:
                text2, conf2, variant2, score2 = evaluate_stage("fast_extra", extra)
                if score2 > best_score:
                    best_text, best_conf, best_variant, best_score = text2, conf2, variant2, score2

        if mode == "fast":
            if self._is_result_confident(best_text, best_conf, best_score):
                total_ms = (time.perf_counter() - started_at) * 1000.0
                self._record_trace(
                    {
                        "mode": mode,
                        "roi_hw": [int(roi_h), int(roi_w)],
                        "elapsed_ms": total_ms,
                        "selected_variant": best_variant,
                        "selected_confidence": float(best_conf),
                        "selected_score": float(best_score),
                        "stages": evaluated,
                    }
                )
                return best_text, best_conf, best_variant, best_score
            if not self.config.retry_enabled:
                maybe_bridge_fallback()
                total_ms = (time.perf_counter() - started_at) * 1000.0
                self._record_trace(
                    {
                        "mode": mode,
                        "roi_hw": [int(roi_h), int(roi_w)],
                        "elapsed_ms": total_ms,
                        "selected_variant": best_variant,
                        "selected_confidence": float(best_conf),
                        "selected_score": float(best_score),
                        "stages": evaluated,
                    }
                )
                return best_text, best_conf, best_variant, best_score
            if not should_retry_ocr(
                roi=roi,
                text=best_text,
                confidence=best_conf,
                config=self.config,
            ):
                maybe_bridge_fallback()
                total_ms = (time.perf_counter() - started_at) * 1000.0
                self._record_trace(
                    {
                        "mode": mode,
                        "roi_hw": [int(roi_h), int(roi_w)],
                        "elapsed_ms": total_ms,
                        "selected_variant": best_variant,
                        "selected_confidence": float(best_conf),
                        "selected_score": float(best_score),
                        "stages": evaluated,
                    }
                )
                return best_text, best_conf, best_variant, best_score

            retry_variants = self._dedupe_variants(build_retry_variants(roi), seen_variants)
            if retry_variants:
                text2, conf2, variant2, score2 = evaluate_stage("retry", retry_variants)
            else:
                text2, conf2, variant2, score2 = "", 0.0, best_variant, -1.0
            if score2 > best_score:
                best_text, best_conf, best_variant, best_score = text2, conf2, variant2, score2
            maybe_bridge_fallback()
            total_ms = (time.perf_counter() - started_at) * 1000.0
            self._record_trace(
                {
                    "mode": mode,
                    "roi_hw": [int(roi_h), int(roi_w)],
                    "elapsed_ms": total_ms,
                    "selected_variant": best_variant,
                    "selected_confidence": float(best_conf),
                    "selected_score": float(best_score),
                    "stages": evaluated,
                }
            )
            return best_text, best_conf, best_variant, best_score

        if self._is_result_confident(best_text, best_conf, best_score):
            total_ms = (time.perf_counter() - started_at) * 1000.0
            self._record_trace(
                {
                    "mode": mode,
                    "roi_hw": [int(roi_h), int(roi_w)],
                    "elapsed_ms": total_ms,
                    "selected_variant": best_variant,
                    "selected_confidence": float(best_conf),
                    "selected_score": float(best_score),
                    "stages": evaluated,
                }
            )
            return best_text, best_conf, best_variant, best_score

        full_variants = self._dedupe_variants(
            build_roi_variants_for_mode(roi, self.config, fast_only=False),
            seen_variants,
        )
        if full_variants:
            text2, conf2, variant2, score2 = evaluate_stage("full", full_variants)
            if score2 > best_score:
                best_text, best_conf, best_variant, best_score = text2, conf2, variant2, score2

        total_ms = (time.perf_counter() - started_at) * 1000.0
        self._record_trace(
            {
                "mode": mode,
                "roi_hw": [int(roi_h), int(roi_w)],
                "elapsed_ms": total_ms,
                "selected_variant": best_variant,
                "selected_confidence": float(best_conf),
                "selected_score": float(best_score),
                "stages": evaluated,
            }
        )
        return best_text, best_conf, best_variant, best_score

    def _recognize_backend(
        self,
        variants: list[tuple[str, np.ndarray]],
        allow_cascade: bool = True,
        prefer_bridge: bool = False,
    ) -> tuple[str, float, str, float]:
        if not variants:
            self._record_trace({"backend": "none", "elapsed_ms": 0.0, "variant_count": 0})
            return "", 0.0, "none", 0.0
        if prefer_bridge and self._paddle_bridge is not None and self._paddle is None:
            return self._recognize_paddle_bridge_variants(variants)
        if (self.config.backend or "").lower() == "rapidocr" and self._rapid is not None:
            return self._recognize_rapid_variants(variants)
        cascade_enabled = (
            allow_cascade
            and self.config.backend_cascade
            and (self.config.backend or "").lower() == "paddleocr"
            and self._rapid is not None
            and (self._paddle is not None or self._paddle_bridge is not None)
        )
        rapid_trace: dict[str, Any] | None = None
        if cascade_enabled:
            probe_count = max(1, int(self.config.cascade_probe_variants))
            probe_variants = variants[:probe_count]
            rapid_text, rapid_conf, rapid_variant, rapid_score = self._recognize_rapid_variants(
                probe_variants
            )
            rapid_trace = self.consume_last_trace()
            if self._is_result_confident(rapid_text, rapid_conf, rapid_score):
                self._record_trace(
                    {
                        "backend": "cascade_rapid_accept",
                        "probe_variant_count": len(probe_variants),
                        "rapid": rapid_trace,
                    }
                )
                return rapid_text, rapid_conf, rapid_variant, rapid_score
        if self._paddle is None and self._paddle_bridge is None:
            if not self.config.allow_fallback:
                detail = self._init_error or "No Paddle backend is available."
                raise RuntimeError(f"Paddle recognition backend unavailable: {detail}")
            if self._rapid is not None:
                return self._recognize_rapid_variants(variants)
            return "", 0.0, "none", 0.0
        if self._paddle_bridge is not None and self._paddle is None:
            result = self._recognize_paddle_bridge_variants(variants)
        else:
            result = self._recognize_paddle_variants(variants)
        if rapid_trace is not None:
            paddle_trace = self.consume_last_trace()
            self._record_trace(
                {
                    "backend": "cascade_paddle_fallback",
                    "probe_variant_count": max(1, int(self.config.cascade_probe_variants)),
                    "rapid": rapid_trace,
                    "fallback": paddle_trace,
                }
            )
        return result

    def _recognize_rapid_variants(
        self, variants: list[tuple[str, np.ndarray]]
    ) -> tuple[str, float, str, float]:
        started_at = time.perf_counter()
        best_text = ""
        best_conf = 0.0
        best_variant = "baseline"
        best_score = -1.0
        variant_calls: list[dict[str, Any]] = []
        for variant_name, variant_img in variants:
            t0 = time.perf_counter()
            result, _extra = self._rapid(variant_img)
            call_ms = (time.perf_counter() - t0) * 1000.0
            text, confidence = parse_rapid_result(result)
            score = score_ocr_result(text, confidence)
            if self.config.profile_variant_calls:
                variant_calls.append(
                    {
                        "variant": variant_name,
                        "backend": "rapidocr",
                        "call_ms": call_ms,
                        "score": float(score),
                        "confidence": float(confidence),
                        "text_len": useful_length(text),
                    }
                )
            if score > best_score:
                best_score = score
                best_text = text
                best_conf = confidence
                best_variant = variant_name
            if self._is_result_confident(text, confidence, score):
                break
        self._record_trace(
            {
                "backend": "rapidocr",
                "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
                "variant_count": len(variants),
                "winner_variant": best_variant,
                "winner_confidence": float(best_conf),
                "winner_score": float(max(0.0, best_score)),
                "variant_calls": variant_calls if self.config.profile_variant_calls else [],
            }
        )
        return best_text, best_conf, best_variant, float(max(0.0, best_score))

    def _recognize_paddle_variants(
        self, variants: list[tuple[str, np.ndarray]]
    ) -> tuple[str, float, str, float]:
        started_at = time.perf_counter()
        best_text = ""
        best_conf = 0.0
        best_variant = "baseline"
        best_score = -1.0
        variant_calls: list[dict[str, Any]] = []
        for variant_name, variant_img in variants:
            # PaddleOCR 3.x/PaddleX pipelines expect 3-channel images.
            paddle_img = variant_img
            if isinstance(variant_img, np.ndarray) and variant_img.ndim == 2:
                paddle_img = cv2.cvtColor(variant_img, cv2.COLOR_GRAY2BGR)
            t0 = time.perf_counter()
            try:
                result = self._paddle.ocr(
                    paddle_img,
                    det=False,
                    rec=True,
                    cls=self._use_angle_cls,
                )
            except Exception:
                try:
                    result = self._paddle.ocr(paddle_img, cls=self._use_angle_cls)
                except Exception:
                    try:
                        # PaddleOCR 3.x may reject cls and extra kwargs.
                        result = self._paddle.ocr(paddle_img)
                    except Exception:
                        if hasattr(self._paddle, "predict"):
                            result = self._paddle.predict(paddle_img)
                        else:
                            raise
            call_ms = (time.perf_counter() - t0) * 1000.0
            text, confidence = _parse_recognition_result(result)
            score = score_ocr_result(text, confidence)
            if self.config.profile_variant_calls:
                variant_calls.append(
                    {
                        "variant": variant_name,
                        "backend": "paddleocr",
                        "call_ms": call_ms,
                        "score": float(score),
                        "confidence": float(confidence),
                        "text_len": useful_length(text),
                    }
                )
            if score > best_score:
                best_score = score
                best_text = text
                best_conf = confidence
                best_variant = variant_name
            if self._is_result_confident(text, confidence, score):
                break
        self._record_trace(
            {
                "backend": "paddleocr",
                "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
                "variant_count": len(variants),
                "winner_variant": best_variant,
                "winner_confidence": float(best_conf),
                "winner_score": float(max(0.0, best_score)),
                "variant_calls": variant_calls if self.config.profile_variant_calls else [],
            }
        )
        return normalize_text(best_text), best_conf, best_variant, float(max(0.0, best_score))

    def _recognize_paddle_bridge_variants(
        self, variants: list[tuple[str, np.ndarray]]
    ) -> tuple[str, float, str, float]:
        if self._paddle_bridge is None:
            return "", 0.0, "none", 0.0
        started_at = time.perf_counter()
        best_text = ""
        best_conf = 0.0
        best_variant = "baseline"
        best_score = -1.0
        variant_calls: list[dict[str, Any]] = []
        for variant_name, variant_img in variants:
            t0 = time.perf_counter()
            text, confidence = self._paddle_bridge.recognize(variant_img, lang=self.config.lang)
            call_ms = (time.perf_counter() - t0) * 1000.0
            score = score_ocr_result(text, confidence)
            if self.config.profile_variant_calls:
                variant_calls.append(
                    {
                        "variant": variant_name,
                        "backend": "paddle_bridge",
                        "call_ms": call_ms,
                        "score": float(score),
                        "confidence": float(confidence),
                        "text_len": useful_length(text),
                    }
                )
            if score > best_score:
                best_score = score
                best_text = text
                best_conf = confidence
                best_variant = variant_name
            if self._is_result_confident(text, confidence, score):
                break
        self._record_trace(
            {
                "backend": "paddle_bridge",
                "elapsed_ms": (time.perf_counter() - started_at) * 1000.0,
                "variant_count": len(variants),
                "winner_variant": best_variant,
                "winner_confidence": float(best_conf),
                "winner_score": float(max(0.0, best_score)),
                "variant_calls": variant_calls if self.config.profile_variant_calls else [],
            }
        )
        return normalize_text(best_text), best_conf, best_variant, float(max(0.0, best_score))


class _PaddleOcrBridge:
    def __init__(self, python_exe: str, script_path: str):
        self.python_exe = python_exe
        self.script_path = script_path

    def recognize(self, image_gray: np.ndarray, lang: str = "ch") -> tuple[str, float]:
        if image_gray.size == 0:
            return "", 0.0
        with tempfile.TemporaryDirectory(prefix="paddle_bridge_") as tmp:
            img_path = Path(tmp) / "roi.png"
            cv2.imwrite(str(img_path), image_gray)
            try:
                env = dict(os.environ)
                # Avoid slow/hanging host connectivity checks in some Paddle distributions.
                env["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
                # Some Paddle/oneDNN builds crash on certain models; disable by default for bridge.
                env["FLAGS_use_mkldnn"] = "0"
                env["FLAGS_use_onednn"] = "0"
                proc = subprocess.run(
                    [self.python_exe, self.script_path, "--image", str(img_path), "--lang", str(lang)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=float(os.getenv("OCR_PADDLE_BRIDGE_TIMEOUT_S", "90")),
                )
            except subprocess.TimeoutExpired:
                return "", 0.0
            except Exception:
                return "", 0.0
            if proc.returncode != 0:
                return "", 0.0
            try:
                payload = json.loads(proc.stdout.strip() or "{}")
            except Exception:
                return "", 0.0
            text = str(payload.get("text") or "")
            conf = float(payload.get("confidence") or 0.0)
            return text, conf


def _maybe_build_paddle_bridge(config: RecognitionConfig) -> _PaddleOcrBridge | None:
    python_exe = config.paddle_python or os.getenv("OCR_PADDLE_PYTHON")
    if not python_exe:
        return None
    script_path = config.paddle_bridge_script or os.getenv("OCR_PADDLE_BRIDGE_SCRIPT")
    if not script_path:
        try:
            repo_root = Path(__file__).resolve().parents[2]
            script_path = str(repo_root / "scripts" / "paddle_ocr_bridge.py")
        except Exception:
            return None
    if not Path(script_path).exists():
        return None
    return _PaddleOcrBridge(python_exe=str(python_exe), script_path=str(script_path))

def warp_polygon_roi(image_gray: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    roi, _raw_roi, _rotation_deg = extract_region_roi_debug(image_gray, polygon)
    return roi


def extract_region_roi_debug(
    image_gray: np.ndarray,
    polygon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape[0] < 3:
        empty = np.empty((0, 0), dtype=np.uint8)
        return empty, empty, 0

    if points.shape[0] != 4:
        rect = cv2.minAreaRect(points)
        points = cv2.boxPoints(rect).astype(np.float32)

    ordered = order_quad_points(points)
    tl, tr, br, bl = ordered

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))
    if width <= 1 or height <= 1:
        empty = np.empty((0, 0), dtype=np.uint8)
        return empty, empty, 0

    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(
        image_gray,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )

    if warped.size == 0:
        fallback = rotate_crop_roi(image_gray, polygon)
        return fallback, warped, 0
    canonical, rotation_deg = canonicalize_roi_orientation(warped, ordered)
    if canonical.size == 0:
        fallback = rotate_crop_roi(image_gray, polygon)
        return fallback, warped, 0
    return canonical, warped, rotation_deg


def rotate_crop_roi(image_gray: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32)
    if points.shape[0] < 3:
        return np.empty((0, 0), dtype=np.uint8)
    rect = cv2.minAreaRect(points)
    (center_x, center_y), (width, height), angle = rect
    if width <= 1 or height <= 1:
        return np.empty((0, 0), dtype=np.uint8)
    if width < height:
        angle += 90.0
        width, height = height, width

    matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated = cv2.warpAffine(
        image_gray,
        matrix,
        (image_gray.shape[1], image_gray.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    crop = cv2.getRectSubPix(
        rotated,
        (max(1, int(round(width))), max(1, int(round(height)))),
        (center_x, center_y),
    )
    if crop.size == 0:
        return crop
    canonical, _rotation_deg = canonicalize_roi_orientation(crop)
    return canonical


def order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] != 4:
        raise ValueError("order_quad_points expects exactly four points.")

    idx_by_y = np.argsort(pts[:, 1], kind="mergesort")
    top = pts[idx_by_y[:2]]
    bottom = pts[idx_by_y[2:]]

    top = top[np.argsort(top[:, 0], kind="mergesort")]
    bottom = bottom[np.argsort(bottom[:, 0], kind="mergesort")]

    tl, tr = top
    bl, br = bottom
    return np.array([tl, tr, br, bl], dtype=np.float32)


def canonicalize_roi_orientation(
    roi: np.ndarray,
    ordered_quad: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    if roi.size == 0:
        return roi, 0
    height, width = roi.shape[:2]
    if width >= height:
        return roi, 0
    rotation_deg = choose_vertical_rotation(ordered_quad)
    if rotation_deg > 0:
        return cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE), rotation_deg
    return cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE), rotation_deg


def choose_vertical_rotation(ordered_quad: np.ndarray | None) -> int:
    if ordered_quad is None or ordered_quad.shape[0] != 4:
        return 90
    tl = ordered_quad[0]
    bl = ordered_quad[3]
    vertical_vec = bl - tl
    if vertical_vec[1] >= 0:
        return 90
    return -90


def build_roi_variants(
    roi: np.ndarray,
    multipass: bool,
) -> list[tuple[str, np.ndarray]]:
    roi = upscale_if_small(roi)
    h, w = roi.shape[:2]
    include_quarter_turns = bool(h > w * 1.15)
    variants = oriented_variants(roi, "baseline", include_quarter_turns=include_quarter_turns)
    if not multipass:
        return variants

    contrast = apply_clahe(roi)
    variants.extend(
        oriented_variants(contrast, "contrast", include_quarter_turns=include_quarter_turns)
    )

    binarized = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    variants.extend(
        oriented_variants(binarized, "binarize", include_quarter_turns=include_quarter_turns)
    )

    invert = cv2.bitwise_not(roi)
    variants.extend(
        oriented_variants(invert, "invert", include_quarter_turns=include_quarter_turns)
    )

    sharpen = apply_unsharp_mask(contrast)
    variants.extend(
        oriented_variants(sharpen, "sharpen", include_quarter_turns=include_quarter_turns)
    )

    denoise = cv2.fastNlMeansDenoising(contrast, None, h=8, templateWindowSize=7, searchWindowSize=21)
    variants.extend(
        oriented_variants(denoise, "denoise", include_quarter_turns=include_quarter_turns)
    )

    line_suppressed = suppress_drawing_lines(contrast)
    variants.extend(
        oriented_variants(
            line_suppressed, "line_suppressed", include_quarter_turns=include_quarter_turns
        )
    )
    return variants


def build_roi_variants_fast(roi: np.ndarray) -> list[tuple[str, np.ndarray]]:
    roi = upscale_if_small(roi)
    h, w = roi.shape[:2]
    include_quarter_turns = bool(h > w * 1.15)
    # Keep the first pass as cheap as possible; add heavier transforms only if needed.
    return oriented_variants(roi, "baseline", include_quarter_turns=include_quarter_turns)


def build_roi_variants_for_mode(
    roi: np.ndarray, config: RecognitionConfig, fast_only: bool = False
) -> list[tuple[str, np.ndarray]]:
    mode = (config.mode or "hybrid").lower()
    if fast_only or mode == "fast":
        return build_roi_variants_fast(roi)
    return build_roi_variants(roi, config.use_multipass)


def build_retry_variants(roi: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Medium-cost retry set for 'fast' mode.
    Keeps runtime reasonable while recovering long-line text cases.
    """
    roi = upscale_if_small(roi)
    h, w = roi.shape[:2]
    include_quarter_turns = bool(h > w * 1.15)
    variants: list[tuple[str, np.ndarray]] = []

    # Retry set: keep it limited; each variant triggers a full OCR forward pass.
    contrast = apply_clahe(roi)
    line_suppressed = suppress_drawing_lines(contrast)
    variants.extend(
        oriented_variants(
            line_suppressed, "line_suppressed", include_quarter_turns=include_quarter_turns
        )
    )

    return variants


def upscale_if_small(roi: np.ndarray) -> np.ndarray:
    """
    RapidOCR is noticeably better when small text is upscaled.
    Keep it conservative to avoid blowing up runtime for already-large crops.
    """
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return roi
    target_h = 96
    if h < 28:
        target_h = 128
    if h >= target_h:
        return roi
    scale = target_h / float(h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w > 1200:
        ratio = 1200.0 / new_w
        new_w = 1200
        new_h = max(1, int(round(new_h * ratio)))
    return cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(image)


def apply_unsharp_mask(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(image, 1.6, blurred, -0.6, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def suppress_drawing_lines(image: np.ndarray) -> np.ndarray:
    """
    Suppress long horizontal/vertical drawing lines while keeping text strokes.
    """
    h, w = image.shape[:2]
    if h < 12 or w < 12:
        return image
    binary_inv = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )
    horiz_len = max(15, min(45, w // 4))
    vert_len = max(15, min(45, h // 4))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_len, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
    horiz = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, horiz_kernel, iterations=1)
    vert = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    line_mask = cv2.bitwise_or(horiz, vert)
    if cv2.countNonZero(line_mask) == 0:
        return image
    # Dilate slightly to fully cover line thickness before inpaint.
    line_mask = cv2.dilate(line_mask, np.ones((2, 2), dtype=np.uint8), iterations=1)
    return cv2.inpaint(image, line_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)


def oriented_variants(
    image: np.ndarray, prefix: str, *, include_quarter_turns: bool = True
) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = [
        (prefix, image),
        (f"{prefix}_rotate180", cv2.rotate(image, cv2.ROTATE_180)),
    ]
    if include_quarter_turns:
        variants.extend(
            [
                (f"{prefix}_rotate90", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
                (f"{prefix}_rotate270", cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
            ]
        )
    return variants


def parse_rapid_result(result: Any) -> tuple[str, float]:
    if not result:
        return "", 0.0
    lines: list[tuple[float, float, str, float]] = []
    for item in result:
        if not isinstance(item, list) or len(item) < 3:
            continue
        polygon, raw_text, raw_score = item
        text = normalize_text(str(raw_text))
        if not text:
            continue
        score = float(raw_score)
        points = np.asarray(polygon, dtype=np.float32)
        if points.shape[0] >= 1:
            x = float(points[:, 0].mean())
            y = float(points[:, 1].mean())
        else:
            x = 0.0
            y = 0.0
        lines.append((y, x, text, score))
    if not lines:
        return "", 0.0
    lines.sort(key=lambda row: (round(row[0] / 8.0), row[1]))
    merged_text = " ".join(row[2] for row in lines)
    confidence = float(sum(row[3] for row in lines) / max(1, len(lines)))
    return normalize_text(merged_text), confidence


def _parse_recognition_result(result: Any) -> tuple[str, float]:
    if not result:
        return "", 0.0

    best_text = ""
    best_score = 0.0

    def handle_item(item: Any) -> None:
        nonlocal best_text, best_score
        # PaddleX / PaddleOCR 3.x sometimes returns dict/object-like items.
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("rec_text") or item.get("label") or "").strip()
            score_raw = item.get("score")
            if score_raw is None:
                score_raw = item.get("rec_score")
            try:
                score = float(score_raw) if score_raw is not None else 0.0
            except Exception:
                score = 0.0
            if score > best_score and text:
                best_text, best_score = text, score
            # Recurse into common containers
            for key in ("res", "result", "results", "rec_res", "items", "data"):
                if key in item:
                    handle_item(item[key])
            return

        # Object-like: try common attributes
        for text_attr, score_attr in (
            ("text", "score"),
            ("rec_text", "rec_score"),
            ("label", "score"),
        ):
            if hasattr(item, text_attr):
                try:
                    text = str(getattr(item, text_attr) or "").strip()
                except Exception:
                    text = ""
                try:
                    score = float(getattr(item, score_attr, 0.0) or 0.0)
                except Exception:
                    score = 0.0
                if score > best_score and text:
                    best_text, best_score = text, score
                return

        if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[0], str):
            text = item[0].strip()
            score = float(item[1])
            if score > best_score and text:
                best_text, best_score = text, score
            return
        if isinstance(item, list):
            # Common PaddleOCR formats include list pairs like ["TEXT", 0.98].
            if len(item) >= 2 and isinstance(item[0], str):
                text = str(item[0] or "").strip()
                try:
                    score = float(item[1])
                except Exception:
                    score = 0.0
                if score > best_score and text:
                    best_text, best_score = text, score
            for sub in item:
                handle_item(sub)

    handle_item(result)
    return normalize_text(best_text), best_score


def normalize_text(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def useful_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text or ""))


def score_ocr_result(text: str, confidence: float) -> float:
    """
    Score OCR candidates while preserving math symbols and mixed-script text.
    """
    text = normalize_text(text)
    if not text:
        return 0.0
    base = float(confidence)
    compact = re.sub(r"\s+", "", text)
    length_bonus = min(0.20, 0.004 * len(compact))
    replacement_penalty = 0.08 * compact.count("�")
    rare_ctrl_penalty = 0.0
    for ch in compact:
        if ord(ch) < 32:
            rare_ctrl_penalty += 0.02
    cjk_count = sum(1 for ch in compact if "\u4e00" <= ch <= "\u9fff")
    digit_count = sum(1 for ch in compact if ch.isdigit())
    cjk_bonus = min(0.12, 0.01 * cjk_count)
    digit_bonus = min(0.08, 0.008 * digit_count)
    return base + length_bonus + cjk_bonus + digit_bonus - replacement_penalty - rare_ctrl_penalty


def should_retry_ocr(
    roi: np.ndarray,
    text: str,
    confidence: float,
    config: RecognitionConfig,
) -> bool:
    """
    Decide whether to run a heavier retry pass.
    Targets long-line regions where fast mode often returns only a suffix.
    """
    if confidence >= float(config.retry_confidence_threshold):
        # High enough confidence; usually OK.
        return False

    compact_len = useful_length(text)
    h, w = roi.shape[:2]
    if h <= 0 or w <= 0:
        return True

    aspect = w / float(h)
    # If it's a long line but text is too short, retry.
    if aspect >= 6.0 and compact_len <= 6:
        return True
    if aspect >= 10.0 and compact_len <= 10:
        return True

    # If OCR returned nothing, retry.
    if compact_len == 0:
        return True
    return False


def should_parallelize_ocr(proposal_count: int, config: RecognitionConfig) -> bool:
    if not config.parallel_ocr:
        return False
    threshold = min(int(config.batch_size), max(1, int(config.parallel_min_regions)))
    return proposal_count >= threshold


def build_region_records(
    image_gray: np.ndarray,
    page_id: str,
    proposals: list[TextProposal],
    recognizer: RegionTextRecognizer,
    region_crops_root: Path | None = None,
    ocr_profile: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
    region_callback: Any | None = None,
) -> list[dict[str, Any]]:
    if should_parallelize_ocr(len(proposals), recognizer.config):
        return build_region_records_parallel(
            image_gray=image_gray,
            page_id=page_id,
            proposals=proposals,
            recognition_config=recognizer.config,
            region_crops_root=region_crops_root,
            ocr_profile=ocr_profile,
        )

    records: list[dict[str, Any]] = []
    page_crop_dir: Path | None = None
    if region_crops_root is not None:
        page_crop_dir = region_crops_root / page_id
        page_crop_dir.mkdir(parents=True, exist_ok=True)
    if ocr_profile is not None:
        ocr_profile.setdefault("mode", "serial")
        ocr_profile.setdefault("recognize_ms", 0.0)
        ocr_profile.setdefault("extract_ms", 0.0)
        ocr_profile.setdefault("regions", 0)
        ocr_profile.setdefault("slow_regions", [])

    for idx, proposal in enumerate(proposals):
        region_id = f"{page_id}_{idx:05d}"
        extract_t0 = time.perf_counter()
        roi, raw_roi, roi_rotation_deg = extract_region_roi_debug(image_gray, proposal.polygon)
        extract_ms = (time.perf_counter() - extract_t0) * 1000.0
        rec_t0 = time.perf_counter()
        text, confidence, variant, ocr_score = recognizer.recognize_roi(roi)
        recognize_ms = (time.perf_counter() - rec_t0) * 1000.0
        trace = recognizer.consume_last_trace()
        text_clean = text.strip()
        if len(text_clean) < recognizer.config.min_text_length:
            text_clean = "Текст не найден"
        if confidence < recognizer.config.min_confidence and text_clean == "Текст не найден":
            confidence = 0.0
        record = {
            "region_id": region_id,
            "page_id": page_id,
            "polygon": proposal.polygon.tolist(),
            "detection_score": float(proposal.score),
            "detection_source": proposal.source,
            "text": text_clean,
            "confidence": float(confidence),
            "ocr_variant": variant,
            "ocr_confidence": float(confidence),
            "ocr_score": float(ocr_score),
            "roi_rotation_deg": int(roi_rotation_deg),
        }
        if page_crop_dir is not None and roi.size > 0:
            crop_path = page_crop_dir / f"{region_id}.png"
            cv2.imwrite(str(crop_path), roi)
            record["crop_path"] = str(crop_path)
            if raw_roi.size > 0:
                raw_path = page_crop_dir / f"{region_id}__raw.png"
                cv2.imwrite(str(raw_path), raw_roi)
                record["crop_raw_path"] = str(raw_path)
            try:
                winning = next(
                    img
                    for name, img in build_roi_variants_for_mode(roi, recognizer.config)
                    if name == variant
                )
                winning_path = page_crop_dir / f"{region_id}__{variant}.png"
                cv2.imwrite(str(winning_path), winning)
                record["crop_winning_path"] = str(winning_path)
            except StopIteration:
                pass
        records.append(record)
        if region_callback:
            try:
                region_callback(record)
            except Exception:
                pass
        if progress_callback and ((idx + 1) % 3 == 0 or (idx + 1) == len(proposals)):
            try:
                progress_callback(
                    {
                        "current_region": int(idx + 1),
                        "total_regions": int(len(proposals)),
                    }
                )
            except Exception:
                pass
        if ocr_profile is not None:
            ocr_profile["regions"] += 1
            ocr_profile["recognize_ms"] += recognize_ms
            ocr_profile["extract_ms"] += extract_ms
            slow = ocr_profile["slow_regions"]
            slow.append(
                {
                    "region_id": region_id,
                    "recognize_ms": recognize_ms,
                    "extract_ms": extract_ms,
                    "ocr_variant": variant,
                    "ocr_score": float(ocr_score),
                    "trace": trace,
                }
            )
            slow.sort(key=lambda item: item["recognize_ms"], reverse=True)
            if len(slow) > 10:
                del slow[10:]
    return records


def save_page_regions(output_dir: Path, page_id: str, records: list[dict[str, Any]]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"page_id": page_id, "regions": records}
    out = output_dir / f"{page_id}_regions.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def extract_region_roi(image_gray: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    roi, _raw_roi, _rotation_deg = extract_region_roi_debug(image_gray, polygon)
    return roi


def build_region_records_parallel(
    image_gray: np.ndarray,
    page_id: str,
    proposals: list[TextProposal],
    recognition_config: RecognitionConfig,
    region_crops_root: Path | None = None,
    ocr_profile: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
    region_callback: Any | None = None,
) -> list[dict[str, Any]]:
    # RapidOCR benefits from threading here: avoids pickling large numpy arrays to subprocesses.
    if (recognition_config.backend or "").lower() == "rapidocr":
        return build_region_records_parallel_threaded(
            image_gray=image_gray,
            page_id=page_id,
            proposals=proposals,
            recognition_config=recognition_config,
            region_crops_root=region_crops_root,
            ocr_profile=ocr_profile,
            progress_callback=progress_callback,
            region_callback=region_callback,
        )
    items: list[dict[str, Any]] = []
    page_crop_dir: Path | None = None
    if region_crops_root is not None:
        page_crop_dir = region_crops_root / page_id
        page_crop_dir.mkdir(parents=True, exist_ok=True)

    for idx, proposal in enumerate(proposals):
        region_id = f"{page_id}_{idx:05d}"
        roi, raw_roi, roi_rotation_deg = extract_region_roi_debug(image_gray, proposal.polygon)
        crop_path = None
        crop_raw_path = None
        if page_crop_dir is not None and roi.size > 0:
            crop = page_crop_dir / f"{region_id}.png"
            cv2.imwrite(str(crop), roi)
            crop_path = str(crop)
            if raw_roi.size > 0:
                raw = page_crop_dir / f"{region_id}__raw.png"
                cv2.imwrite(str(raw), raw_roi)
                crop_raw_path = str(raw)
        items.append(
            {
                "region_id": region_id,
                "page_id": page_id,
                "polygon": proposal.polygon.tolist(),
                "detection_score": float(proposal.score),
                "detection_source": proposal.source,
                "roi": roi,
                "crop_path": crop_path,
                "crop_raw_path": crop_raw_path,
                "roi_rotation_deg": int(roi_rotation_deg),
            }
        )

    batches = list(_chunked(items, max(1, recognition_config.batch_size)))
    records: list[dict[str, Any]] = []
    workers = max(1, int(recognition_config.max_workers))
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker_recognizer,
        initargs=(recognition_config,),
    ) as executor:
        for output_batch in executor.map(_process_region_batch, batches):
            records.extend(output_batch)
    if ocr_profile is not None:
        ocr_profile.setdefault("mode", "process")
        ocr_profile["workers"] = workers
        ocr_profile["regions"] = len(records)
        recognize_ms = [
            float((record.get("ocr_profile") or {}).get("recognize_ms", 0.0))
            for record in records
        ]
        ocr_profile["recognize_ms"] = float(sum(recognize_ms))
        ranked = sorted(
            (
                {
                    "region_id": str(record.get("region_id", "")),
                    "recognize_ms": float((record.get("ocr_profile") or {}).get("recognize_ms", 0.0)),
                    "trace": (record.get("ocr_profile") or {}).get("trace", {}),
                    "ocr_variant": str(record.get("ocr_variant", "")),
                    "ocr_score": float(record.get("ocr_score", 0.0)),
                }
                for record in records
            ),
            key=lambda item: item["recognize_ms"],
            reverse=True,
        )
        ocr_profile["slow_regions"] = ranked[:10]
    return records


def _get_thread_recognizer(config: RecognitionConfig) -> RegionTextRecognizer:
    rec = getattr(_THREAD_LOCAL, "recognizer", None)
    if rec is None:
        cfg = RecognitionConfig(**config.__dict__)
        cfg.parallel_ocr = False
        rec = RegionTextRecognizer(cfg)
        _THREAD_LOCAL.recognizer = rec
    return rec


def _process_region_item_thread(item: dict[str, Any], config: RecognitionConfig) -> dict[str, Any]:
    recognizer = _get_thread_recognizer(config)
    roi = item["roi"]
    rec_t0 = time.perf_counter()
    text, confidence, variant, ocr_score = recognizer.recognize_roi(roi)
    rec_ms = (time.perf_counter() - rec_t0) * 1000.0
    trace = recognizer.consume_last_trace()
    text_clean = text.strip()
    if len(text_clean) < recognizer.config.min_text_length:
        text_clean = "Текст не найден"
    if confidence < recognizer.config.min_confidence and text_clean == "Текст не найден":
        confidence = 0.0

    record = {
        "region_id": item["region_id"],
        "page_id": item["page_id"],
        "polygon": item["polygon"],
        "detection_score": item["detection_score"],
        "detection_source": item["detection_source"],
        "text": text_clean,
        "confidence": float(confidence),
        "ocr_variant": variant,
        "ocr_confidence": float(confidence),
        "ocr_score": float(ocr_score),
        "roi_rotation_deg": int(item.get("roi_rotation_deg", 0)),
    }
    if item.get("crop_path"):
        record["crop_path"] = item["crop_path"]
        if item.get("crop_raw_path"):
            record["crop_raw_path"] = item["crop_raw_path"]
        try:
            winning = next(
                img
                for name, img in build_roi_variants_for_mode(roi, recognizer.config)
                if name == variant
            )
            winning_path = Path(item["crop_path"]).with_name(f"{item['region_id']}__{variant}.png")
            cv2.imwrite(str(winning_path), winning)
            record["crop_winning_path"] = str(winning_path)
        except StopIteration:
            pass
    if recognizer.config.profile_ocr:
        record["ocr_profile"] = {
            "recognize_ms": rec_ms,
            "trace": trace,
        }
    return record


def build_region_records_parallel_threaded(
    image_gray: np.ndarray,
    page_id: str,
    proposals: list[TextProposal],
    recognition_config: RecognitionConfig,
    region_crops_root: Path | None = None,
    ocr_profile: dict[str, Any] | None = None,
    progress_callback: Any | None = None,
    region_callback: Any | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page_crop_dir: Path | None = None
    if region_crops_root is not None:
        page_crop_dir = region_crops_root / page_id
        page_crop_dir.mkdir(parents=True, exist_ok=True)

    for idx, proposal in enumerate(proposals):
        region_id = f"{page_id}_{idx:05d}"
        roi, raw_roi, roi_rotation_deg = extract_region_roi_debug(image_gray, proposal.polygon)
        crop_path = None
        crop_raw_path = None
        if page_crop_dir is not None and roi.size > 0:
            crop = page_crop_dir / f"{region_id}.png"
            cv2.imwrite(str(crop), roi)
            crop_path = str(crop)
            if raw_roi.size > 0:
                raw = page_crop_dir / f"{region_id}__raw.png"
                cv2.imwrite(str(raw), raw_roi)
                crop_raw_path = str(raw)
        items.append(
            {
                "region_id": region_id,
                "page_id": page_id,
                "polygon": proposal.polygon.tolist(),
                "detection_score": float(proposal.score),
                "detection_source": proposal.source,
                "roi": roi,
                "crop_path": crop_path,
                "crop_raw_path": crop_raw_path,
                "roi_rotation_deg": int(roi_rotation_deg),
            }
        )

    workers = max(1, int(recognition_config.max_workers))
    total_items = len(items)
    ocr_device = str(getattr(recognition_config, "ocr_device", "cpu") or "cpu").strip().lower()
    # GPU runs many ROIs quickly; emit progress more frequently to avoid "jumpy" bars.
    if total_items <= 20:
        progress_every = 1
    elif ocr_device == "cuda":
        progress_every = 2
    else:
        progress_every = 5
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_process_region_item_thread, it, recognition_config): i for i, it in enumerate(items)
        }
        done_count = 0
        for fut in as_completed(futures):
            rec = fut.result()
            records.append(rec)
            if region_callback:
                try:
                    region_callback(rec)
                except Exception:
                    pass
            done_count += 1
            if progress_callback and (done_count % progress_every == 0 or done_count == total_items):
                try:
                    progress_callback(
                        {
                            "current_region": int(done_count),
                            "total_regions": int(total_items),
                        }
                    )
                except Exception:
                    pass
    if ocr_profile is not None:
        ocr_profile.setdefault("mode", "threaded")
        ocr_profile["workers"] = workers
        ocr_profile["regions"] = len(records)
        recognize_ms = [
            float((record.get("ocr_profile") or {}).get("recognize_ms", 0.0))
            for record in records
        ]
        ocr_profile["recognize_ms"] = float(sum(recognize_ms))
        ranked = sorted(
            (
                {
                    "region_id": str(record.get("region_id", "")),
                    "recognize_ms": float((record.get("ocr_profile") or {}).get("recognize_ms", 0.0)),
                    "trace": (record.get("ocr_profile") or {}).get("trace", {}),
                    "ocr_variant": str(record.get("ocr_variant", "")),
                    "ocr_score": float(record.get("ocr_score", 0.0)),
                }
                for record in records
            ),
            key=lambda item: item["recognize_ms"],
            reverse=True,
        )
        ocr_profile["slow_regions"] = ranked[:10]
    # Keep stable ordering by region_id to avoid UI churn.
    records.sort(key=lambda r: str(r.get("region_id", "")))
    return records


def _init_worker_recognizer(config: RecognitionConfig) -> None:
    global _WORKER_RECOGNIZER
    cfg = RecognitionConfig(**config.__dict__)
    cfg.parallel_ocr = False
    _WORKER_RECOGNIZER = RegionTextRecognizer(cfg)


def _process_region_batch(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recognizer = _WORKER_RECOGNIZER
    if recognizer is None:
        raise RuntimeError("OCR worker recognizer not initialized.")

    output: list[dict[str, Any]] = []
    for item in batch:
        roi = item["roi"]
        rec_t0 = time.perf_counter()
        text, confidence, variant, ocr_score = recognizer.recognize_roi(roi)
        rec_ms = (time.perf_counter() - rec_t0) * 1000.0
        trace = recognizer.consume_last_trace()
        text_clean = text.strip()
        if len(text_clean) < recognizer.config.min_text_length:
            text_clean = "Текст не найден"
        if confidence < recognizer.config.min_confidence and text_clean == "Текст не найден":
            confidence = 0.0

        record = {
            "region_id": item["region_id"],
            "page_id": item["page_id"],
            "polygon": item["polygon"],
            "detection_score": item["detection_score"],
            "detection_source": item["detection_source"],
            "text": text_clean,
            "confidence": float(confidence),
            "ocr_variant": variant,
            "ocr_confidence": float(confidence),
            "ocr_score": float(ocr_score),
            "roi_rotation_deg": int(item.get("roi_rotation_deg", 0)),
        }
        if item.get("crop_path"):
            record["crop_path"] = item["crop_path"]
            if item.get("crop_raw_path"):
                record["crop_raw_path"] = item["crop_raw_path"]
            try:
                winning = next(
                    img
                    for name, img in build_roi_variants_for_mode(roi, recognizer.config)
                    if name == variant
                )
                winning_path = Path(item["crop_path"]).with_name(
                    f"{item['region_id']}__{variant}.png"
                )
                cv2.imwrite(str(winning_path), winning)
                record["crop_winning_path"] = str(winning_path)
            except StopIteration:
                pass
        if recognizer.config.profile_ocr:
            record["ocr_profile"] = {
                "recognize_ms": rec_ms,
                "trace": trace,
            }
        output.append(record)
    return output


def _chunked(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
