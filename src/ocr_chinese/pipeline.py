from __future__ import annotations

from dataclasses import dataclass, asdict, field
import hashlib
import json
from pathlib import Path
import time
from typing import Callable
import os
import cv2

import numpy as np

from .detect import DetectionConfig, OrientedTextDetector, TextProposal, load_image
from .filter import FilterConfig, TextCandidateFilter
from .mask import MaskConfig, draw_overlay, postprocess_mask, rasterize_polygons, save_mask
from .metrics import compute_pixel_metrics, load_mask, save_metrics_report
from .recognize import (
    RecognitionConfig,
    RegionTextRecognizer,
    build_region_records,
    save_page_regions,
)
from .render import PdfRenderOptions, render_pdf_to_images


@dataclass
class PipelineConfig:
    dpi: int = 400
    save_overlays: bool = True
    render: PdfRenderOptions = field(default_factory=PdfRenderOptions)
    detector: DetectionConfig = field(default_factory=DetectionConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)


def _save_proposals(page_id: str, proposals: list[TextProposal], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"page_id": page_id, "proposals": [proposal.to_json() for proposal in proposals]}
    path = output_dir / f"{page_id}_proposals.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_proposals(path: Path) -> list[TextProposal]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    proposals: list[TextProposal] = []
    for item in payload.get("proposals", []):
        polygon = np.asarray(item.get("polygon", []), dtype=np.float32)
        if polygon.shape[0] < 3:
            continue
        proposals.append(
            TextProposal(
                polygon=polygon,
                score=float(item.get("score", 0.0)),
                source=str(item.get("source", "unknown")),
            )
        )
    return proposals


def run_mask_pipeline(
    pdf_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    gt_masks_dir: Path | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_dir = output_dir / "rendered_pages"
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    proposals_dir = output_dir / "proposals"

    render_options = config.render
    render_options.dpi = config.dpi
    detector = OrientedTextDetector(config.detector)
    candidate_filter = TextCandidateFilter(config.filter)

    page_reports: list[dict] = []
    profiling = {
        "render_ms": 0.0,
        "detect_ms": 0.0,
        "mask_ms": 0.0,
        "pages": [],
    }
    aggregate = {
        "recall_sum": 0.0,
        "precision_sum": 0.0,
        "fpr_sum": 0.0,
        "evaluated_pages": 0,
    }

    t_render = time.perf_counter()
    page_paths = render_pdf_to_images(pdf_path, rendered_dir, render_options)
    render_ms = (time.perf_counter() - t_render) * 1000.0
    profiling["render_ms"] = render_ms

    for page_path in page_paths:
        page_profile = {"page_id": page_path.stem}
        image = load_image(page_path)
        t0 = time.perf_counter()
        proposals = detector.detect(image)
        proposals = candidate_filter.filter(proposals, image)
        detect_ms = (time.perf_counter() - t0) * 1000.0
        profiling["detect_ms"] += detect_ms
        page_profile["detect_ms"] = detect_ms
        _save_proposals(page_path.stem, proposals, proposals_dir)

        t1 = time.perf_counter()
        raw_mask = rasterize_polygons(image.shape, proposals)
        mask = postprocess_mask(raw_mask, config.mask)
        mask_path = masks_dir / f"{page_path.stem}_mask.png"
        save_mask(mask, mask_path)
        mask_ms = (time.perf_counter() - t1) * 1000.0
        profiling["mask_ms"] += mask_ms
        page_profile["mask_ms"] = mask_ms

        if config.save_overlays:
            overlay = draw_overlay(image, mask)
            overlays_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(overlays_dir / f"{page_path.stem}_overlay.png"), overlay)

        page_report = {
            "page_id": page_path.stem,
            "image_path": str(page_path),
            "mask_path": str(mask_path),
            "proposal_count": len(proposals),
        }
        page_profile["proposal_count"] = len(proposals)

        if gt_masks_dir:
            gt_path = gt_masks_dir / f"{page_path.stem}_mask.png"
            if gt_path.exists():
                gt_mask = load_mask(gt_path)
                metric = compute_pixel_metrics(mask, gt_mask).to_dict()
                page_report["metrics"] = metric
                aggregate["recall_sum"] += metric["recall"]
                aggregate["precision_sum"] += metric["precision"]
                aggregate["fpr_sum"] += metric["false_positive_rate"]
                aggregate["evaluated_pages"] += 1
        page_reports.append(page_report)
        profiling["pages"].append(page_profile)

    result = {
        "pdf_path": str(pdf_path),
        "output_dir": str(output_dir),
        "config": asdict(config),
        "pages": page_reports,
        "profiling": profiling,
    }
    if aggregate["evaluated_pages"] > 0:
        n = aggregate["evaluated_pages"]
        result["aggregate_metrics"] = {
            "mean_recall": aggregate["recall_sum"] / n,
            "mean_precision": aggregate["precision_sum"] / n,
            "mean_false_positive_rate": aggregate["fpr_sum"] / n,
            "evaluated_pages": n,
        }

    save_metrics_report(result, output_dir / "report.json")
    return result


def precompute_region_text(
    output_dir: Path,
    recognition_config: RecognitionConfig | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    region_ready_callback: Callable[[str, dict], None] | None = None,
    page_done_callback: Callable[[str, list[dict]], None] | None = None,
) -> dict:
    rendered_dir = output_dir / "rendered_pages"
    proposals_dir = output_dir / "proposals"
    merged_dir = output_dir / "merged_regions"
    regions_dir = output_dir / "regions"
    region_crops_dir = output_dir / "region_crops"
    cache_dir = output_dir / "cache" / "ocr"
    recognition_config = recognition_config or RecognitionConfig()

    # RapidOCR often uses multi-threaded runtimes internally (OpenCV/OMP/ORT).
    # Unbounded internal threads + external parallelism can cause oversubscription and slowdowns.
    if (recognition_config.backend or "").lower() == "rapidocr" and recognition_config.parallel_ocr:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass

    # Opportunistic speedup: enable *small* parallelism only if requested.
    # Auto-enabling was shown to regress on some machines; keep it conservative.
    if (
        (recognition_config.backend or "").lower() == "rapidocr"
        and recognition_config.parallel_ocr
        and int(getattr(recognition_config, "max_workers", 1) or 1) > 2
    ):
        recognition_config.max_workers = 2
    recognizer = RegionTextRecognizer(recognition_config)
    cache_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []
    profiling = {
        "ocr_ms": 0.0,
        "merge_ms": 0.0,
        "cache_hits": 0,
        "pages": [],
        "stage_budget_ms": {
            "render_detect_mask": 20000.0,
            "ocr": 35000.0,
            "other": 5000.0,
            "total": 60000.0,
        },
    }
    page_images = sorted(rendered_dir.glob("page_*.png"))
    for index, page_image in enumerate(page_images, start=1):
        page_id = page_image.stem
        proposal_file = proposals_dir / f"{page_id}_proposals.json"
        if not proposal_file.exists():
            continue
        image = load_image(page_image)
        raw_proposals = load_proposals(proposal_file)
        t_merge = time.perf_counter()
        merge_lines = recognition_config.mode in {"hybrid", "accurate"}
        proposals = merge_and_dedupe_proposals(raw_proposals, merge_lines=merge_lines)
        merge_ms = (time.perf_counter() - t_merge) * 1000.0
        profiling["merge_ms"] += merge_ms
        save_merged_proposals(page_id, proposals, merged_dir)

        key = build_ocr_cache_key(page_image, proposal_file, recognition_config, len(proposals))
        cache_file = cache_dir / f"{page_id}_{key}.json"
        if cache_file.exists():
            regions = json.loads(cache_file.read_text(encoding="utf-8")).get("regions", [])
            profiling["cache_hits"] += 1
            out_path = save_page_regions(regions_dir, page_id, regions)
            if page_done_callback:
                try:
                    page_done_callback(page_id, list(regions))
                except Exception:
                    pass
            page_summary = {
                "page_id": page_id,
                "raw_regions_count": len(raw_proposals),
                "merged_regions_count": len(proposals),
                "regions_count": len(regions),
                "regions_path": str(out_path),
                "region_crops_dir": str(region_crops_dir / page_id),
                "cache_hit": True,
            }
            summary.append(page_summary)
            profiling["pages"].append(page_summary | {"merge_ms": merge_ms, "ocr_ms": 0.0})
            if progress_callback:
                progress_callback(
                    {
                        "stage": "ocr",
                        "current_page": index,
                        "total_pages": len(page_images),
                        "page_id": page_id,
                        "status": "cache_hit",
                        "current_region": 1,
                        "total_regions": 1,
                    }
                )
            continue

        page_ocr_profile: dict = {"page_id": page_id}
        t_ocr = time.perf_counter()
        def on_region_progress(update: dict) -> None:
            if not progress_callback:
                return
            payload = {
                "stage": "ocr",
                "current_page": index,
                "total_pages": len(page_images),
                "page_id": page_id,
                "status": "running",
            }
            payload.update(update or {})
            progress_callback(payload)

        regions = build_region_records(
            image,
            page_id,
            proposals,
            recognizer,
            region_crops_root=region_crops_dir,
            ocr_profile=page_ocr_profile,
            progress_callback=on_region_progress,
            region_callback=(
                (lambda rec, pid=page_id: region_ready_callback(pid, rec)) if region_ready_callback else None
            ),
        )
        ocr_ms = (time.perf_counter() - t_ocr) * 1000.0
        profiling["ocr_ms"] += ocr_ms
        for region in regions:
            region.pop("ocr_profile", None)
        out_path = save_page_regions(regions_dir, page_id, regions)
        cache_file.write_text(json.dumps({"regions": regions}, ensure_ascii=False), encoding="utf-8")
        if page_done_callback:
            try:
                page_done_callback(page_id, list(regions))
            except Exception:
                pass
        variant_distribution = count_variants(regions)
        not_found_ratio = count_not_found_ratio(regions)
        page_summary = {
            "page_id": page_id,
            "raw_regions_count": len(raw_proposals),
            "merged_regions_count": len(proposals),
            "regions_count": len(regions),
            "regions_path": str(out_path),
            "region_crops_dir": str(region_crops_dir / page_id),
            "cache_hit": False,
            "variant_distribution": variant_distribution,
            "not_found_ratio": not_found_ratio,
        }
        summary.append(page_summary)
        profiling["pages"].append(
            page_summary
            | {
                "merge_ms": merge_ms,
                "ocr_ms": ocr_ms,
                "ocr_profile": page_ocr_profile,
            }
        )
        if progress_callback:
            progress_callback(
                {
                    "stage": "ocr",
                    "current_page": index,
                    "total_pages": len(page_images),
                    "page_id": page_id,
                    "status": "page_done",
                    "current_region": int(len(proposals)),
                    "total_regions": int(len(proposals)),
                }
            )
    total_render_detect_mask = 0.0
    total_ocr = float(profiling["ocr_ms"])
    total_pages = max(1, len(summary))
    for page in profiling.get("pages", []):
        total_render_detect_mask += float(page.get("merge_ms", 0.0))
    profiling["timing_budget_check"] = {
        "ocr_ms": total_ocr,
        "ocr_budget_ms": float(profiling["stage_budget_ms"]["ocr"]),
        "ocr_budget_met": total_ocr <= float(profiling["stage_budget_ms"]["ocr"]),
        "avg_ocr_ms_per_page": total_ocr / total_pages,
        "merge_ms": total_render_detect_mask,
    }
    return {"regions": summary, "total_pages": len(summary), "profiling": profiling}


def run_mask_pipeline_with_regions(
    pdf_path: Path,
    output_dir: Path,
    config: PipelineConfig,
    gt_masks_dir: Path | None = None,
    recognition_config: RecognitionConfig | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    region_ready_callback: Callable[[str, dict], None] | None = None,
    page_done_callback: Callable[[str, list[dict]], None] | None = None,
) -> dict:
    started_at = time.perf_counter()
    if progress_callback:
        try:
            progress_callback(
                {
                    "stage": "mask",
                    "status": "running",
                    "current_page": 0,
                    "total_pages": 0,
                    "page_id": None,
                }
            )
        except Exception:
            pass
    report = run_mask_pipeline(
        pdf_path=pdf_path,
        output_dir=output_dir,
        config=config,
        gt_masks_dir=gt_masks_dir,
    )
    total_pages = len(report.get("pages", []) or [])
    if progress_callback:
        try:
            progress_callback(
                {
                    "stage": "ocr",
                    "status": "running",
                    "current_page": 0,
                    "total_pages": int(total_pages),
                    "page_id": None,
                }
            )
        except Exception:
            pass
    report["region_precompute"] = precompute_region_text(
        output_dir=output_dir,
        recognition_config=recognition_config,
        progress_callback=progress_callback,
        region_ready_callback=region_ready_callback,
        page_done_callback=page_done_callback,
    )
    total_ms = (time.perf_counter() - started_at) * 1000.0
    mask_profile = report.get("profiling", {})
    region_profile = report.get("region_precompute", {}).get("profiling", {})
    mask_stage_ms = float(mask_profile.get("render_ms", 0.0)) + float(
        mask_profile.get("detect_ms", 0.0)
    ) + float(mask_profile.get("mask_ms", 0.0))
    ocr_stage_ms = float(region_profile.get("ocr_ms", 0.0))
    other_ms = max(0.0, total_ms - mask_stage_ms - ocr_stage_ms)
    report["timing"] = {
        "total_ms": total_ms,
        "mask_stage_ms": mask_stage_ms,
        "ocr_stage_ms": ocr_stage_ms,
        "other_ms": other_ms,
        "target_total_ms": 60000.0,
        "meets_target": total_ms <= 60000.0,
    }
    save_metrics_report(report, output_dir / "report.json")
    return report


def bbox_from_polygon(polygon: np.ndarray) -> tuple[float, float, float, float]:
    x_min = float(np.min(polygon[:, 0]))
    y_min = float(np.min(polygon[:, 1]))
    x_max = float(np.max(polygon[:, 0]))
    y_max = float(np.max(polygon[:, 1]))
    return x_min, y_min, x_max, y_max


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-6, (bx2 - bx1) * (by2 - by1))
    return inter / (area_a + area_b - inter)


def merge_and_dedupe_proposals(
    proposals: list[TextProposal],
    iou_threshold: float = 0.55,
    merge_lines: bool = True,
) -> list[TextProposal]:
    if not proposals:
        return []
    # Hard cap candidates to keep CPU predictable on personal machines.
    sorted_props = sorted(proposals, key=lambda p: p.score, reverse=True)[:2000]

    boxes_xywh: list[list[float]] = []
    scores: list[float] = []
    for proposal in sorted_props:
        x1, y1, x2, y2 = bbox_from_polygon(proposal.polygon)
        boxes_xywh.append([x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1)])
        scores.append(float(proposal.score))

    kept: list[TextProposal] = []
    if boxes_xywh:
        nms_idx = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh,
            scores=scores,
            score_threshold=0.05,
            nms_threshold=max(0.1, float(iou_threshold)),
            top_k=1200,
        )
        if len(nms_idx) > 0:
            for idx in np.array(nms_idx).reshape(-1).tolist():
                kept.append(sorted_props[int(idx)])
        else:
            kept = sorted_props[:1200]
    kept = sorted(kept, key=lambda p: p.score, reverse=True)

    if not merge_lines:
        return kept

    # lightweight line grouping
    rows: list[list[TextProposal]] = []
    for proposal in sorted(kept, key=lambda p: bbox_from_polygon(p.polygon)[1]):
        x1, y1, x2, y2 = bbox_from_polygon(proposal.polygon)
        h = max(1.0, y2 - y1)
        cy = (y1 + y2) * 0.5
        matched_row = None
        for row in rows:
            r_boxes = [bbox_from_polygon(r.polygon) for r in row]
            r_cy = sum((rb[1] + rb[3]) * 0.5 for rb in r_boxes) / len(r_boxes)
            r_h = sum(max(1.0, rb[3] - rb[1]) for rb in r_boxes) / len(r_boxes)
            if abs(cy - r_cy) <= max(h, r_h) * 0.6:
                matched_row = row
                break
        if matched_row is None:
            rows.append([proposal])
        else:
            matched_row.append(proposal)

    merged: list[TextProposal] = []
    for row in rows:
        row.sort(key=lambda p: bbox_from_polygon(p.polygon)[0])
        current = [row[0]]
        for proposal in row[1:]:
            prev = current[-1]
            px1, py1, px2, py2 = bbox_from_polygon(prev.polygon)
            cx1, cy1, cx2, cy2 = bbox_from_polygon(proposal.polygon)
            gap = cx1 - px2
            h = max(1.0, (py2 - py1 + cy2 - cy1) * 0.5)
            if gap <= h * 1.6:
                current.append(proposal)
            else:
                merged.append(merge_group(current))
                current = [proposal]
        merged.append(merge_group(current))
    return merged


def merge_group(group: list[TextProposal]) -> TextProposal:
    xs = []
    ys = []
    score = 0.0
    for item in group:
        xs.extend(item.polygon[:, 0].tolist())
        ys.extend(item.polygon[:, 1].tolist())
        score = max(score, item.score)
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    polygon = np.asarray([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    return TextProposal(polygon=polygon, score=score, source="merged")


def save_merged_proposals(page_id: str, proposals: list[TextProposal], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"page_id": page_id, "proposals": [proposal.to_json() for proposal in proposals]}
    (output_dir / f"{page_id}_merged.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_ocr_cache_key(
    page_image: Path,
    proposal_file: Path,
    recognition_config: RecognitionConfig,
    merged_count: int,
) -> str:
    h = hashlib.sha1()
    h.update(page_image.read_bytes())
    h.update(proposal_file.read_bytes())
    h.update(str(merged_count).encode("utf-8"))
    h.update(json.dumps(recognition_config.__dict__, sort_keys=True).encode("utf-8"))
    h.update(b"ocr_v2")
    return h.hexdigest()[:20]


def count_variants(regions: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for region in regions:
        variant = str(region.get("ocr_variant", "unknown"))
        counts[variant] = counts.get(variant, 0) + 1
    return counts


def count_not_found_ratio(regions: list[dict]) -> float:
    if not regions:
        return 0.0
    nf = 0
    for region in regions:
        text = str(region.get("text", "")).strip()
        if text == "Текст не найден" or not text:
            nf += 1
    return nf / len(regions)
