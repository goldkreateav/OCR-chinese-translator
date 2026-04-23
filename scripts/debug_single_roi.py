from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np

from ocr_chinese.recognize import (
    RecognitionConfig,
    RegionTextRecognizer,
    apply_clahe,
    apply_unsharp_mask,
    build_roi_variants,
    build_roi_variants_fast,
    build_retry_variants,
    oriented_variants,
    score_ocr_result,
    suppress_drawing_lines,
    upscale_if_small,
)


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def _extra_variants(roi: np.ndarray) -> list[tuple[str, np.ndarray]]:
    roi = upscale_if_small(roi)
    h, w = roi.shape[:2]
    include_quarter_turns = bool(h > w * 1.15)
    out: list[tuple[str, np.ndarray]] = []

    contrast = apply_clahe(roi)
    out.extend(oriented_variants(contrast, "contrast", include_quarter_turns=include_quarter_turns))

    binarized = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    out.extend(oriented_variants(binarized, "binarize", include_quarter_turns=include_quarter_turns))

    inv = cv2.bitwise_not(roi)
    out.extend(oriented_variants(inv, "invert", include_quarter_turns=include_quarter_turns))

    sharp = apply_unsharp_mask(contrast)
    out.extend(oriented_variants(sharp, "sharpen", include_quarter_turns=include_quarter_turns))

    denoise = cv2.fastNlMeansDenoising(contrast, None, h=8, templateWindowSize=7, searchWindowSize=21)
    out.extend(oriented_variants(denoise, "denoise", include_quarter_turns=include_quarter_turns))

    ls = suppress_drawing_lines(contrast)
    out.extend(oriented_variants(ls, "line_suppressed", include_quarter_turns=include_quarter_turns))

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug OCR quality on a single ROI image.")
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("debug_single_roi_out"))
    ap.add_argument("--backend", choices=["rapidocr", "paddleocr"], default="paddleocr")
    ap.add_argument("--mode", choices=["fast", "hybrid", "accurate"], default="accurate")
    ap.add_argument("--multipass", action="store_true")
    ap.add_argument("--no-cascade", action="store_true")
    ap.add_argument("--topk", type=int, default=12)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # PaddleOCR sometimes performs a model host connectivity check on startup.
    # This can hang on restricted networks; disable it here to keep the debug loop fast.
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    roi = _read_gray(args.image)

    cfg = RecognitionConfig(
        backend=args.backend,
        mode=args.mode,
        use_multipass=bool(args.multipass),
        parallel_ocr=False,
        max_workers=1,
        backend_cascade=not bool(args.no_cascade),
        profile_ocr=True,
        profile_variant_calls=True,
        retry_enabled=True,
    )
    recognizer = RegionTextRecognizer(cfg)

    variant_sets: list[tuple[str, list[tuple[str, np.ndarray]]]] = [
        ("fast", build_roi_variants_fast(roi)),
        ("retry", build_retry_variants(roi)),
        ("full", build_roi_variants(roi, multipass=bool(args.multipass))),
        ("extra", _extra_variants(roi)),
    ]
    all_variants: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()
    for set_name, variants in variant_sets:
        for name, img in variants:
            key = f"{set_name}:{name}"
            if key in seen:
                continue
            seen.add(key)
            all_variants.append((key, img))

    results: list[dict] = []
    for key, img in all_variants:
        t0 = time.perf_counter()
        text, conf, chosen_variant, score = recognizer.recognize_roi(img)
        ms = (time.perf_counter() - t0) * 1000.0
        trace = recognizer.consume_last_trace()
        results.append(
            {
                "input_variant": key,
                "text": text,
                "confidence": float(conf),
                "score": float(score_ocr_result(text, conf)),
                "raw_score": float(score),
                "chosen_variant": chosen_variant,
                "elapsed_ms": ms,
                "trace": trace,
            }
        )

    results.sort(key=lambda r: (r["score"], r["confidence"]), reverse=True)
    topk = max(1, int(args.topk))
    top = results[:topk]

    print(f"backend={args.backend} mode={args.mode} multipass={bool(args.multipass)}")
    print(f"candidates={len(results)} topk={topk}")
    for i, r in enumerate(top, start=1):
        text = (r["text"] or "").strip().replace("\n", " ")
        safe_text = text.encode("unicode_escape").decode("ascii")
        print(
            f"{i:02d}. score={r['score']:.3f} conf={r['confidence']:.3f} "
            f"ms={r['elapsed_ms']:.1f} var={r['input_variant']} -> '{safe_text}'"
        )

    (out_dir / "results.json").write_text(
        json.dumps({"config": cfg.__dict__, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # Save top variants as images for visual inspection.
    for r in top[: min(topk, 8)]:
        key = str(r["input_variant"]).replace(":", "__").replace("/", "_")
        idx = str(top.index(r) + 1).zfill(2)
        # Recreate variant image by looking it up in all_variants
        img = next(img for k, img in all_variants if k == r["input_variant"])
        cv2.imwrite(str(out_dir / f"{idx}__{key}.png"), img)


if __name__ == "__main__":
    main()

