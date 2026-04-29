from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectCreateResponse(BaseModel):
    project_id: str
    filename: str
    status: str


class GenerateRequest(BaseModel):
    dpi: int = 400
    render_backend: str | None = None
    poppler_path: str | None = None
    ocr_mode: str = "eco"  # eco | balanced | max
    ocr_workers: int | None = None
    ocr_device: str | None = None  # cpu | cuda


class ImportRenderRequest(BaseModel):
    dpi: int = 400
    render_backend: str | None = None
    poppler_path: str | None = None


class ProjectStatusResponse(BaseModel):
    project_id: str
    status: str
    error: str | None = None
    pages: int = 0
    pipeline_status: str | None = None
    stage: str | None = None
    progress: dict | None = None
    progress_pages: dict | None = None
    ocr_runtime: dict | None = None
    paddle_runtime: dict | None = None
    translation: dict | None = None
    updated_at: str | None = None


class RegionRecord(BaseModel):
    region_id: str
    page_id: str
    polygon: list[list[float]]
    detection_score: float
    detection_source: str
    text: str
    confidence: float
    ocr_variant: str | None = None
    ocr_confidence: float | None = None
    ocr_score: float | None = None
    crop_path: str | None = None
    crop_raw_path: str | None = None
    crop_winning_path: str | None = None
    roi_rotation_deg: int | None = None


class PageAssetsResponse(BaseModel):
    page_id: str
    image_url: str
    mask_url: str
    regions: list[RegionRecord] = Field(default_factory=list)
