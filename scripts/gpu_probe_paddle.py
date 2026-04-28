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

        # Probe constructor compatibility across PaddleOCR versions.
        # Some versions don't accept `show_log` or some doc-orientation kwargs.
        ctor_options = [
            dict(
                lang="ch",
                use_gpu=True,
                show_log=False,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            ),
            dict(
                lang="ch",
                use_gpu=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            ),
            dict(lang="ch", use_gpu=True, show_log=False),
            dict(lang="ch", use_gpu=True),
            dict(use_gpu=True),
        ]

        last_exc: Exception | None = None
        for kw in ctor_options:
            try:
                _ocr = PaddleOCR(**kw)
                payload["paddleocr_init_ok"] = True
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                payload["paddleocr_init_ok"] = False
        if last_exc is not None and not payload.get("paddleocr_init_ok"):
            raise last_exc

        # Best-effort: try to see where Paddle thinks it runs.
        try:
            import paddle  # type: ignore

            payload["paddle_device"] = str(paddle.get_device())
        except Exception:
            pass

        # If Paddle isn't compiled with CUDA, treat as not-ok even if ctor "worked".
        if payload.get("paddle_is_compiled_with_cuda") is False:
            payload["ok"] = False
            payload["error"] = "paddle_is_cpu_only"
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 6

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

