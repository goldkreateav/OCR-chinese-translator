from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--lang", type=str, default="ch")
    args = parser.parse_args()

    from paddleocr import PaddleOCR  # type: ignore

    # Newer PaddleX-backed pipelines expect 3-channel images.
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        print(json.dumps({"text": "", "confidence": 0.0}, ensure_ascii=False))
        return

    try:
        # PaddleOCR 2.x classic API (preferred for this bridge).
        ocr = PaddleOCR(use_angle_cls=False, lang=args.lang, det=False, rec=True, show_log=False)
    except Exception:
        # Compatibility fallback for variants where det/rec args are removed.
        try:
            ocr = PaddleOCR(use_angle_cls=False, lang=args.lang, show_log=False)
        except Exception:
            try:
                ocr = PaddleOCR(use_angle_cls=False, lang=args.lang)
            except Exception:
                # PaddleOCR 3.x may reject use_angle_cls/show_log entirely.
                ocr = PaddleOCR(lang=args.lang)

    # PaddleOCR API varies across versions:
    # - older: ocr(img, det=..., rec=..., cls=...)
    # - newer: ocr(img) delegates to predict(img) with different kwargs
    result = None
    try:
        result = ocr.ocr(img, det=False, rec=True, cls=False)
    except Exception:
        try:
            result = ocr.ocr(img, cls=False)
        except Exception:
            if hasattr(ocr, "predict"):
                result = ocr.predict(img)
            else:
                result = ocr.ocr(img)

    text = ""
    conf = 0.0
    try:
        # PaddleOCR rec-only formats vary; normalize to best candidate.
        # Common: [[("text", 0.98)]] or [("text", 0.98)] or [["text", 0.98]]
        if isinstance(result, list) and result:
            r0 = result[0]
            if isinstance(r0, list) and r0:
                r0 = r0[0]
            if isinstance(r0, (list, tuple)) and len(r0) >= 2:
                text = str(r0[0] or "")
                conf = float(r0[1] or 0.0)
    except Exception:
        text = ""
        conf = 0.0

    # Some Windows shells default to cp1251/cp866. Escaping avoids encoding errors.
    sys.stdout.write(json.dumps({"text": text, "confidence": conf}, ensure_ascii=True))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

