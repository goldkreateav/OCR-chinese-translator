from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

from fastapi import HTTPException, UploadFile

from ..pipeline import PipelineConfig, run_mask_pipeline_with_regions
from ..recognize import RecognitionConfig, RegionTextRecognizer, extract_region_roi


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class GenerateOptions:
    dpi: int = 360
    render_backend: str = "auto"
    poppler_path: str | None = None
    ocr_mode: str = "eco"
    ocr_workers: int | None = None


class ProjectService:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_project(self, upload: UploadFile) -> dict:
        project_id = uuid.uuid4().hex[:12]
        project_dir = self.root_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=False)

        filename = upload.filename or "input.pdf"
        pdf_path = project_dir / "input.pdf"
        with pdf_path.open("wb") as file_handle:
            file_handle.write(upload.file.read())

        status = {
            "project_id": project_id,
            "filename": filename,
            "status": "uploaded",
            "error": None,
            "pages": 0,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self._write_status(project_id, status)
        return status

    def generate(self, project_id: str, options: GenerateOptions) -> dict:
        status = self.get_status(project_id)
        project_dir = self._project_dir(project_id)
        pdf_path = project_dir / "input.pdf"
        output_dir = project_dir / "output"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Input PDF not found.")

        status["status"] = "running"
        status["error"] = None
        status["stage"] = "mask"
        status["progress"] = {
            "status": "running",
            "indeterminate": True,
            "current_page": 0,
            "total_pages": 0,
            "page_id": None,
            "current_region": 0,
            "total_regions": 0,
        }
        status["updated_at"] = _utc_now()
        self._write_status(project_id, status)

        try:
            config = PipelineConfig(dpi=options.dpi)
            config.render.backend = options.render_backend
            config.render.poppler_path = options.poppler_path
            self._tune_pipeline_config(config, options)
            rec_cfg = self._build_recognition_config(options)

            def on_progress(update: dict) -> None:
                status_inner = self.get_status(project_id)
                status_inner["stage"] = update.get("stage", "ocr")
                status_inner["progress"] = {
                    "current_page": int(update.get("current_page", 0)),
                    "total_pages": int(update.get("total_pages", 0)),
                    "page_id": update.get("page_id"),
                    "status": update.get("status"),
                    "current_region": int(update.get("current_region", 0) or 0),
                    "total_regions": int(update.get("total_regions", 0) or 0),
                }
                # Per-page progress map for UI (page_id -> {percent, current_region, total_regions, status})
                page_id = update.get("page_id")
                if page_id:
                    pages_map = status_inner.get("progress_pages") or {}
                    try:
                        cur_r = int(update.get("current_region", 0) or 0)
                        tot_r = int(update.get("total_regions", 0) or 0)
                    except Exception:
                        cur_r, tot_r = 0, 0
                    percent = None
                    if tot_r > 0:
                        raw_pct = int(round((float(cur_r) * 100.0) / float(max(1, tot_r))))
                        percent = max(0, min(100, raw_pct))
                    pages_map[str(page_id)] = {
                        "percent": percent,
                        "current_region": cur_r,
                        "total_regions": tot_r,
                        "status": str(update.get("status") or ""),
                        "current_page": int(update.get("current_page", 0) or 0),
                        "total_pages": int(update.get("total_pages", 0) or 0),
                    }
                    status_inner["progress_pages"] = pages_map
                status_inner["updated_at"] = _utc_now()
                self._write_status(project_id, status_inner)

            report = run_mask_pipeline_with_regions(
                pdf_path=pdf_path,
                output_dir=output_dir,
                config=config,
                recognition_config=rec_cfg,
                progress_callback=on_progress,
            )
            status["status"] = "done"
            status["pages"] = len(report.get("pages", []))
            status["updated_at"] = _utc_now()
            self._write_status(project_id, status)
            return {"status": status, "report": report}
        except Exception as exc:
            status["status"] = "error"
            status["error"] = str(exc)
            status["updated_at"] = _utc_now()
            self._write_status(project_id, status)
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    def _build_recognition_config(self, options: GenerateOptions) -> RecognitionConfig:
        mode = (options.ocr_mode or "eco").lower()
        cpu = int(os.cpu_count() or 4)
        auto_workers = max(2, min(8, cpu // 2))
        workers = options.ocr_workers if options.ocr_workers is not None else auto_workers
        workers = max(1, int(workers))
        if mode == "max":
            return RecognitionConfig(
                mode="hybrid",
                use_multipass=True,
                parallel_ocr=True,
                max_workers=min(8, workers),
                batch_size=24,
                parallel_min_regions=12,
                cascade_probe_variants=2,
            )
        if mode == "balanced":
            return RecognitionConfig(
                mode="hybrid",
                use_multipass=False,
                parallel_ocr=True,
                max_workers=min(4, workers),
                batch_size=28,
                parallel_min_regions=14,
                cascade_probe_variants=2,
                bridge_fallback_enabled=True,
                bridge_fallback_confidence_threshold=0.82,
                bridge_fallback_score_threshold=0.98,
            )
        # eco (default): keep CPU usage low for personal machines
        return RecognitionConfig(
            mode="fast",
            use_multipass=False,
            parallel_ocr=False,
            max_workers=1,
            batch_size=24,
            parallel_min_regions=12,
            retry_confidence_threshold=0.70,
            accept_confidence_threshold=0.92,
            accept_score_threshold=1.04,
            accept_min_length=6,
            cascade_probe_variants=2,
            bridge_fallback_enabled=True,
            bridge_fallback_confidence_threshold=0.82,
            bridge_fallback_score_threshold=0.98,
        )

    def _tune_pipeline_config(self, config: PipelineConfig, options: GenerateOptions) -> None:
        mode = (options.ocr_mode or "eco").lower()
        if mode == "eco":
            config.dpi = min(int(config.dpi), 360)
            config.detector.score_threshold = max(0.24, float(config.detector.score_threshold))
            config.detector.min_area = max(20, int(config.detector.min_area))
            return
        if mode == "balanced":
            config.dpi = min(int(config.dpi), 380)
            config.detector.score_threshold = max(0.22, float(config.detector.score_threshold))
            config.detector.min_area = max(18, int(config.detector.min_area))

    def get_status(self, project_id: str) -> dict:
        status_path = self._project_dir(project_id) / "status.json"
        if not status_path.exists():
            raise HTTPException(status_code=404, detail="Project not found.")
        return json.loads(status_path.read_text(encoding="utf-8"))

    def list_pages(self, project_id: str) -> list[str]:
        output_dir = self._project_dir(project_id) / "output" / "rendered_pages"
        pages = [page.stem for page in sorted(output_dir.glob("page_*.png"))]
        if not pages:
            raise HTTPException(status_code=404, detail="No generated pages found.")
        return pages

    def load_page_regions(self, project_id: str, page_id: str) -> list[dict]:
        region_path = self._project_dir(project_id) / "output" / "regions" / f"{page_id}_regions.json"
        if not region_path.exists():
            return []
        payload = json.loads(region_path.read_text(encoding="utf-8"))
        return payload.get("regions", [])

    def load_region_by_id(self, project_id: str, region_id: str) -> dict | None:
        output_dir = self._project_dir(project_id) / "output" / "regions"
        for regions_file in sorted(output_dir.glob("page_*_regions.json")):
            payload = json.loads(regions_file.read_text(encoding="utf-8"))
            for region in payload.get("regions", []):
                if region.get("region_id") == region_id:
                    return region
        return None

    def retry_region_ocr(self, project_id: str, region_id: str) -> dict:
        region = self.load_region_by_id(project_id, region_id)
        if region is None:
            raise HTTPException(status_code=404, detail="Region not found.")
        crop_path = region.get("crop_path")
        crop_file: Path | None = None

        if crop_path:
            raw = Path(str(crop_path))
            candidates: list[Path] = []
            if raw.is_absolute():
                candidates.append(raw)
            # Sometimes stored relative to repo root, e.g. "web_jobs\\<id>\\output\\..."
            candidates.append(Path.cwd() / raw)
            # Sometimes stored relative to project dir
            candidates.append(self._project_dir(project_id) / raw)
            # If it includes "web_jobs/<project_id>/...", strip prefix and treat as project-relative
            prefix = Path("web_jobs") / project_id
            try:
                rel_after_prefix = raw.relative_to(prefix)
                candidates.append(self._project_dir(project_id) / rel_after_prefix)
            except Exception:
                pass

            for cand in candidates:
                try:
                    resolved = cand.resolve()
                except Exception:
                    continue
                if resolved.exists():
                    crop_file = resolved
                    break

        import cv2
        import numpy as np

        roi = None
        if crop_file is not None:
            roi = cv2.imread(str(crop_file), cv2.IMREAD_GRAYSCALE)

        # Fallback: regenerate ROI crop from rendered page + polygon
        if roi is None:
            page_id = region.get("page_id")
            polygon = region.get("polygon")
            if not page_id or not polygon:
                raise HTTPException(status_code=404, detail="Crop file not found.")
            page_img = cv2.imread(str(self.image_path(project_id, str(page_id))), cv2.IMREAD_GRAYSCALE)
            if page_img is None:
                raise HTTPException(status_code=500, detail="Failed to read rendered page image.")
            poly = np.asarray(polygon, dtype=np.float32)
            roi = extract_region_roi(page_img, poly)
            if roi is None or roi.size == 0:
                raise HTTPException(status_code=500, detail="Failed to extract ROI from polygon.")

        # Accurate single-region retry (still CPU-friendly because it's one ROI).
        # If PaddleOCR is not available in this interpreter (e.g. Python 3.14),
        # you can provide OCR_PADDLE_PYTHON pointing to a Python 3.10–3.12 env
        # with paddleocr installed, and we'll call it via subprocess bridge.
        rec_cfg = RecognitionConfig(
            backend="paddleocr",
            mode="accurate",
            use_multipass=True,
            parallel_ocr=False,
            max_workers=1,
        )
        recognizer = RegionTextRecognizer(rec_cfg)
        text, confidence, variant, score = recognizer.recognize_roi(roi)
        text_clean = (text or "").strip() or "Текст не найден"
        if text_clean == "Текст не найден":
            confidence = 0.0
            score = 0.0
        return {
            "region_id": region_id,
            "text": text_clean,
            "ocr_confidence": float(confidence),
            "ocr_variant": variant,
            "ocr_score": float(score),
        }

    def image_path(self, project_id: str, page_id: str) -> Path:
        path = self._project_dir(project_id) / "output" / "rendered_pages" / f"{page_id}.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Rendered page not found.")
        return path

    def mask_path(self, project_id: str, page_id: str) -> Path:
        path = self._project_dir(project_id) / "output" / "masks" / f"{page_id}_mask.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Mask not found.")
        return path

    def normalize_page_id(self, page: str) -> str:
        if page.startswith("page_"):
            return page
        if page.isdigit():
            return f"page_{int(page):04d}"
        raise HTTPException(status_code=400, detail="Page must be number or page_XXXX format.")

    def _project_dir(self, project_id: str) -> Path:
        project_dir = self.root_dir / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found.")
        return project_dir

    def _write_status(self, project_id: str, status: dict) -> None:
        path = self._project_dir(project_id) / "status.json"
        path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
