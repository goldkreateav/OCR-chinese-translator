from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from ocr_chinese.recognize import RecognitionConfig, RegionTextRecognizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Single OCR call via paddle bridge.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--backend", choices=["rapidocr", "paddleocr"], default="rapidocr")
    ap.add_argument("--mode", choices=["fast", "hybrid", "accurate"], default="accurate")
    ap.add_argument("--multipass", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    cfg = RecognitionConfig(
        backend=args.backend,
        mode=args.mode,
        use_multipass=bool(args.multipass),
        parallel_ocr=False,
        max_workers=1,
        backend_cascade=False,
        profile_ocr=True,
        profile_variant_calls=True,
    )
    rec = RegionTextRecognizer(cfg)
    text, conf, variant, score = rec.recognize_roi(img)
    trace = rec.consume_last_trace()
    print(
        json.dumps(
            {
                "text": text,
                "conf": conf,
                "variant": variant,
                "score": score,
                "has_bridge": bool(getattr(rec, "_paddle_bridge", None) is not None),
                "trace": trace,
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()

