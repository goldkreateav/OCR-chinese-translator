from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from ocr_chinese.pipeline import PipelineConfig, run_mask_pipeline_with_regions
from ocr_chinese.recognize import RecognitionConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark end-to-end pipeline speed.")
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--baseline-report", type=Path, default=None)
    parser.add_argument("--render-backend", choices=["auto", "pymupdf", "poppler"], default="auto")
    parser.add_argument("--poppler-path", type=str, default=None)
    args = parser.parse_args()

    cfg = PipelineConfig(dpi=args.dpi)
    cfg.render.backend = args.render_backend
    cfg.render.poppler_path = args.poppler_path
    rec_cfg = RecognitionConfig(mode="hybrid", parallel_ocr=True, max_workers=4)

    t0 = time.perf_counter()
    report = run_mask_pipeline_with_regions(
        pdf_path=args.pdf,
        output_dir=args.out,
        config=cfg,
        recognition_config=rec_cfg,
    )
    elapsed = time.perf_counter() - t0
    pages = max(1, len(report.get("pages", [])))
    sec_per_page = elapsed / pages
    print(f"Current: elapsed={elapsed:.3f}s, pages={pages}, sec_per_page={sec_per_page:.3f}")

    if args.baseline_report and args.baseline_report.exists():
        baseline = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        baseline_elapsed = float(baseline.get("bench", {}).get("elapsed_s", 0.0))
        baseline_pages = int(max(1, baseline.get("bench", {}).get("pages", pages)))
        baseline_sec_per_page = baseline_elapsed / baseline_pages if baseline_elapsed > 0 else 0.0
        if baseline_sec_per_page > 0:
            speedup = baseline_sec_per_page / sec_per_page
            print(
                f"Baseline sec_per_page={baseline_sec_per_page:.3f}; speedup={speedup:.2f}x"
            )

    report["bench"] = {
        "elapsed_s": elapsed,
        "pages": pages,
        "sec_per_page": sec_per_page,
    }
    (args.out / "bench_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
