from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ocr_chinese.pipeline import load_proposals, merge_and_dedupe_proposals
from ocr_chinese.detect import load_image
from ocr_chinese.recognize import RecognitionConfig, RegionTextRecognizer, build_region_records


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate adaptive OCR on a sampled subset of regions.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--page-id", type=str, default="page_0001")
    ap.add_argument("--max-regions", type=int, default=48)
    args = ap.parse_args()

    output_dir = args.output_dir
    page_id = args.page_id
    image = load_image(output_dir / "rendered_pages" / f"{page_id}.png")
    raw = load_proposals(output_dir / "proposals" / f"{page_id}_proposals.json")
    merged = merge_and_dedupe_proposals(raw, merge_lines=False)
    proposals = merged[: max(1, int(args.max_regions))]

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
    rec = RegionTextRecognizer(cfg)
    profile: dict = {"sample_size": len(proposals)}
    t0 = time.perf_counter()
    regions = build_region_records(
        image_gray=image,
        page_id=page_id,
        proposals=proposals,
        recognizer=rec,
        region_crops_root=None,
        ocr_profile=profile,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    low_conf = 0
    not_found = 0
    for region in regions:
        text = str(region.get("text", "")).strip()
        conf = float(region.get("confidence", 0.0))
        if text == "Текст не найден" or not text:
            not_found += 1
        if conf < 0.80:
            low_conf += 1

    summary = {
        "sample_size": len(regions),
        "elapsed_ms": elapsed_ms,
        "avg_ms_per_region": elapsed_ms / max(1, len(regions)),
        "low_conf_ratio": low_conf / max(1, len(regions)),
        "not_found_ratio": not_found / max(1, len(regions)),
        "variant_distribution": {
            k: sum(1 for r in regions if str(r.get("ocr_variant")) == k)
            for k in sorted({str(r.get("ocr_variant", "")) for r in regions})
        },
        "slow_regions_top3": (profile.get("slow_regions") or [])[:3],
    }
    print(json.dumps(summary, ensure_ascii=True))


if __name__ == "__main__":
    main()

