from __future__ import annotations

import argparse
import json
from pathlib import Path

from ocr_chinese.pipeline import precompute_region_text
from ocr_chinese.recognize import RecognitionConfig


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate adaptive OCR fallback on rendered/proposals.")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = RecognitionConfig(
        mode="fast",
        backend="rapidocr",
        parallel_ocr=False,
        max_workers=1,
        retry_enabled=True,
        retry_confidence_threshold=0.70,
        bridge_fallback_enabled=True,
        bridge_fallback_confidence_threshold=0.82,
        bridge_fallback_score_threshold=0.98,
        profile_ocr=True,
        profile_variant_calls=True,
    )
    res = precompute_region_text(output_dir=args.output_dir, recognition_config=cfg)
    profiling = res.get("profiling", {})
    print("ocr_ms", profiling.get("ocr_ms"))
    print("cache_hits", profiling.get("cache_hits"))
    pages = profiling.get("pages", [])
    print("pages", len(pages))
    if pages:
        page = pages[0]
        print("not_found_ratio", page.get("not_found_ratio"))
        print("variant_distribution", json.dumps(page.get("variant_distribution", {}), ensure_ascii=True))
        slow = ((page.get("ocr_profile") or {}).get("slow_regions") or [])[:3]
        print("slow_regions_top3", json.dumps(slow, ensure_ascii=True))


if __name__ == "__main__":
    main()

