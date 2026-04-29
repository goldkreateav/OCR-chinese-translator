from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dotenv import load_default_env
from .labeling import SplitConfig, export_cvat_tasks_stub, prepare_labeling_manifest
from .pipeline import PipelineConfig, run_mask_pipeline
from .training import (
    FilterTrainingConfig,
    calibrate_threshold,
    train_filter_classifier,
    write_paddle_finetune_recipe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maskpdf",
        description="Generate text masks for Chinese text in scanned PDF drawings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="Run full inference pipeline.")
    run_cmd.add_argument("pdf", type=Path)
    run_cmd.add_argument("--out", type=Path, required=True)
    run_cmd.add_argument("--dpi", type=int, default=400)
    run_cmd.add_argument(
        "--render-backend",
        choices=["auto", "pymupdf", "poppler"],
        default="auto",
        help="PDF renderer backend.",
    )
    run_cmd.add_argument(
        "--poppler-path",
        type=Path,
        default=None,
        help="Path to folder containing pdftoppm(.exe). Used by poppler backend.",
    )
    run_cmd.add_argument("--gt-masks", type=Path, default=None)

    label_cmd = sub.add_parser("prepare-labeling", help="Prepare labeling manifest.")
    label_cmd.add_argument("--rendered-dir", type=Path, required=True)
    label_cmd.add_argument("--manifest", type=Path, required=True)
    label_cmd.add_argument("--dev-ratio", type=float, default=0.5)
    label_cmd.add_argument("--seed", type=int, default=42)
    label_cmd.add_argument("--cvat-index-dir", type=Path, default=None)

    train_filter_cmd = sub.add_parser("train-filter", help="Train text/non-text filter.")
    train_filter_cmd.add_argument("--features", type=Path, required=True)
    train_filter_cmd.add_argument("--model-out", type=Path, required=True)
    train_filter_cmd.add_argument("--trees", type=int, default=300)

    calibrate_cmd = sub.add_parser("calibrate", help="Calibrate threshold from probabilities.")
    calibrate_cmd.add_argument("--scores-json", type=Path, required=True)
    calibrate_cmd.add_argument("--target-recall", type=float, default=1.0)

    recipe_cmd = sub.add_parser(
        "write-finetune-recipe",
        help="Write PaddleOCR fine-tune steps for detector adaptation.",
    )
    recipe_cmd.add_argument("--dataset-root", type=Path, required=True)
    recipe_cmd.add_argument("--out", type=Path, required=True)

    web_cmd = sub.add_parser("web", help="Run local web UI.")
    web_cmd.add_argument("--host", default="127.0.0.1")
    web_cmd.add_argument("--port", type=int, default=8000)
    web_cmd.add_argument("--data-root", type=Path, default=Path("web_jobs"))
    web_cmd.add_argument(
        "--render-backend",
        choices=["auto", "pymupdf", "poppler"],
        default="auto",
    )
    web_cmd.add_argument(
        "--poppler-path",
        type=Path,
        default=None,
        help="Path to folder containing pdftoppm(.exe).",
    )
    web_cmd.add_argument(
        "--ocr-mode",
        choices=["eco", "balanced", "max"],
        default="eco",
        help="OCR resource profile for web generation.",
    )
    web_cmd.add_argument(
        "--ocr-workers",
        type=int,
        default=1,
        help="OCR worker processes (used for balanced/max modes).",
    )
    web_cmd.add_argument(
        "--ocr-device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="OCR inference device for RapidOCR backends.",
    )
    web_cmd.add_argument(
        "--ocr-fallback-cpu-on-oom",
        action="store_true",
        help="If GPU runs out of memory, automatically retry OCR on CPU.",
    )
    web_cmd.add_argument(
        "--allow-fallback",
        action="store_true",
        help=(
            "Allow non-Paddle fallbacks (RapidOCR/MSER). "
            "By default web generation is strict Paddle-only and fails on Paddle errors."
        ),
    )
    return parser


def main() -> None:
    # Auto-load env vars for Web UI / translation / OCR bridge.
    # This keeps deployment simple: put variables into .env in the working dir.
    load_default_env()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        config = PipelineConfig(dpi=args.dpi)
        config.render.backend = args.render_backend
        config.render.poppler_path = str(args.poppler_path) if args.poppler_path else None
        report = run_mask_pipeline(
            pdf_path=args.pdf,
            output_dir=args.out,
            config=config,
            gt_masks_dir=args.gt_masks,
        )
        print(json.dumps(report.get("aggregate_metrics", {}), ensure_ascii=False, indent=2))
        return

    if args.command == "prepare-labeling":
        split = SplitConfig(dev_ratio=args.dev_ratio, seed=args.seed)
        manifest = prepare_labeling_manifest(args.rendered_dir, args.manifest, split)
        if args.cvat_index_dir:
            export_cvat_tasks_stub(args.manifest, args.cvat_index_dir)
        print(f"Prepared {len(manifest['items'])} items.")
        return

    if args.command == "train-filter":
        cfg = FilterTrainingConfig(n_estimators=args.trees)
        info = train_filter_classifier(args.features, args.model_out, cfg)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return

    if args.command == "calibrate":
        payload = json.loads(args.scores_json.read_text(encoding="utf-8"))
        result = calibrate_threshold(
            probabilities=payload["probabilities"],
            labels=payload["labels"],
            target_recall=args.target_recall,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "write-finetune-recipe":
        path = write_paddle_finetune_recipe(args.out, args.dataset_root)
        print(f"Wrote fine-tune recipe to {path}")
        return

    if args.command == "web":
        try:
            import uvicorn
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install uvicorn and fastapi to run web UI.") from exc
        from .web.app import create_app

        app = create_app(
            data_root=args.data_root,
            default_render_backend=args.render_backend,
            default_poppler_path=str(args.poppler_path) if args.poppler_path else None,
            default_ocr_mode=args.ocr_mode,
            default_ocr_workers=args.ocr_workers,
            default_ocr_device=args.ocr_device,
            default_ocr_fallback_to_cpu_on_oom=bool(args.ocr_fallback_cpu_on_oom),
            allow_fallback=args.allow_fallback,
        )
        uvicorn.run(app, host=args.host, port=args.port)
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
