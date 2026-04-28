from __future__ import annotations

import json
import os
import sys
import traceback


def _bool_env(name: str, default: bool = False) -> bool:
    raw = str(os.getenv(name, "")).strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def main() -> int:
    payload: dict[str, object] = {
        "ok": False,
        "paddle_import_ok": False,
        "paddle_is_compiled_with_cuda": None,
        "paddle_device": None,
        "paddle_version": None,
        "paddleocr_import_ok": False,
        "paddleocr_version": None,
        "paddleocr_init_ok": False,
        "paddleocr_use_gpu_requested": True,
        "error": None,
        "traceback": None,
    }

    verbose = _bool_env("GPU_PROBE_VERBOSE", False)

    try:
        import paddle  # type: ignore

        payload["paddle_import_ok"] = True
        payload["paddle_version"] = getattr(paddle, "__version__", None)
        try:
            payload["paddle_is_compiled_with_cuda"] = bool(paddle.is_compiled_with_cuda())
        except Exception:
            payload["paddle_is_compiled_with_cuda"] = None
        try:
            payload["paddle_device"] = str(paddle.get_device())
        except Exception:
            payload["paddle_device"] = None
    except Exception as exc:
        payload["error"] = f"paddle_import_failed: {exc}"
        if verbose:
            payload["traceback"] = traceback.format_exc()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    try:
        import paddleocr  # type: ignore

        payload["paddleocr_import_ok"] = True
        payload["paddleocr_version"] = getattr(paddleocr, "__version__", None)
    except Exception as exc:
        payload["error"] = f"paddleocr_import_failed: {exc}"
        if verbose:
            payload["traceback"] = traceback.format_exc()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 3

    try:
        from paddleocr import PaddleOCR  # type: ignore

        # If CUDA build is present, PaddleOCR should accept use_gpu=True.
        # If not, this may throw or silently run on CPU; we treat init failure as not-ok.
        _ocr = PaddleOCR(
            lang="ch",
            use_gpu=True,
            show_log=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
        payload["paddleocr_init_ok"] = True

        # Best-effort: try to see where Paddle thinks it runs.
        try:
            import paddle  # type: ignore

            payload["paddle_device"] = str(paddle.get_device())
        except Exception:
            pass

        payload["ok"] = True
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except TypeError as exc:
        payload["error"] = f"paddleocr_init_typeerror: {exc}"
        if verbose:
            payload["traceback"] = traceback.format_exc()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 4
    except Exception as exc:
        payload["error"] = f"paddleocr_init_failed: {exc}"
        if verbose:
            payload["traceback"] = traceback.format_exc()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 5


if __name__ == "__main__":
    raise SystemExit(main())

