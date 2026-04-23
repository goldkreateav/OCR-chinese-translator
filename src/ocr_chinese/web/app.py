from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .schemas import (
    GenerateRequest,
    PageAssetsResponse,
    ProjectCreateResponse,
    ProjectStatusResponse,
    RegionRecord,
)
from .service import GenerateOptions, ProjectService


def create_app(
    data_root: Path | None = None,
    default_render_backend: str = "auto",
    default_poppler_path: str | None = None,
    default_ocr_mode: str = "eco",
    default_ocr_workers: int = 1,
) -> FastAPI:
    package_dir = Path(__file__).resolve().parent
    static_dir = package_dir / "static"
    jobs_root = data_root or (Path.cwd() / "web_jobs")
    service = ProjectService(jobs_root)

    app = FastAPI(title="OCR Chinese Mask Web UI")
    app.state.default_render_backend = default_render_backend
    app.state.default_poppler_path = default_poppler_path
    app.state.default_ocr_mode = default_ocr_mode
    app.state.default_ocr_workers = default_ocr_workers
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/import")
    async def index_import() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/version")
    async def version() -> JSONResponse:
        try:
            from importlib.metadata import version as pkg_version  # py3.10+

            v = pkg_version("ocr-chinese-masker")
        except Exception:
            v = "unknown"
        return JSONResponse(
            {
                "app_version": v,
                "quality_bridge_enabled": bool(os.getenv("OCR_PADDLE_PYTHON")),
                "bridge_python": os.getenv("OCR_PADDLE_PYTHON"),
            }
        )

    @app.post("/api/projects", response_model=ProjectCreateResponse)
    async def create_project(file: UploadFile = File(...)) -> ProjectCreateResponse:
        created = service.create_project(file)
        return ProjectCreateResponse(
            project_id=created["project_id"],
            filename=created["filename"],
            status=created["status"],
        )

    @app.post("/api/import/projects")
    async def import_project(
        file: UploadFile = File(...),
        dpi: int = Form(400),
        render_backend: str | None = Form(None),
    ) -> JSONResponse:
        chosen_backend = render_backend or app.state.default_render_backend
        poppler_path = app.state.default_poppler_path
        payload = service.create_import_project_and_render_pages(
            file,
            dpi=int(dpi or 400),
            render_backend=chosen_backend,
            poppler_path=poppler_path,
        )
        return JSONResponse(payload)

    @app.post("/api/projects/{project_id}/generate")
    async def generate(project_id: str, request: GenerateRequest) -> JSONResponse:
        render_backend = request.render_backend or app.state.default_render_backend
        poppler_path = request.poppler_path
        if poppler_path is None:
            poppler_path = app.state.default_poppler_path
        ocr_mode = request.ocr_mode or app.state.default_ocr_mode
        ocr_workers = (
            request.ocr_workers
            if request.ocr_workers is not None
            else app.state.default_ocr_workers
        )
        options = GenerateOptions(
            dpi=request.dpi,
            render_backend=render_backend,
            poppler_path=poppler_path,
            ocr_mode=ocr_mode,
            ocr_workers=ocr_workers,
        )
        payload = service.start_generate_background(project_id, options)
        return JSONResponse(payload)

    @app.get("/api/projects/{project_id}/status", response_model=ProjectStatusResponse)
    async def status(project_id: str) -> ProjectStatusResponse:
        data = service.get_status(project_id)
        return ProjectStatusResponse(
            project_id=data["project_id"],
            status=data["status"],
            error=data.get("error"),
            pages=int(data.get("pages", 0)),
            stage=data.get("stage"),
            progress=data.get("progress"),
            progress_pages=data.get("progress_pages"),
            updated_at=data.get("updated_at"),
        )

    @app.get("/api/projects/{project_id}/pages")
    async def pages(project_id: str) -> JSONResponse:
        return JSONResponse({"pages": service.list_pages(project_id)})

    @app.get("/api/projects/{project_id}/pages/{page}/regions")
    async def regions(project_id: str, page: str) -> JSONResponse:
        page_id = service.normalize_page_id(page)
        return JSONResponse({"page_id": page_id, "regions": service.load_page_regions(project_id, page_id)})

    @app.get("/api/projects/{project_id}/pages/{page}/assets", response_model=PageAssetsResponse)
    async def assets(project_id: str, page: str) -> PageAssetsResponse:
        page_id = service.normalize_page_id(page)
        regions_payload = service.load_page_regions(project_id, page_id)
        region_models = [RegionRecord(**region) for region in regions_payload]
        return PageAssetsResponse(
            page_id=page_id,
            image_url=f"/api/projects/{project_id}/pages/{page_id}/image",
            mask_url=f"/api/projects/{project_id}/pages/{page_id}/mask",
            regions=region_models,
        )

    @app.get("/api/projects/{project_id}/pages/{page}/translations/status")
    async def translations_status(project_id: str, page: str, lang: str = "ru") -> JSONResponse:
        page_id = service.normalize_page_id(page)
        return JSONResponse(service.load_translation_status(project_id, page_id, lang=lang))

    @app.get("/api/projects/{project_id}/pages/{page}/translations/region/{region_id}")
    async def translations_region(
        project_id: str, page: str, region_id: str, lang: str = "ru"
    ) -> JSONResponse:
        page_id = service.normalize_page_id(page)
        return JSONResponse(service.load_region_translation(project_id, page_id, region_id, lang=lang))

    @app.get("/api/projects/{project_id}/pages/{page}/image")
    async def page_image(project_id: str, page: str) -> FileResponse:
        page_id = service.normalize_page_id(page)
        return FileResponse(service.image_path(project_id, page_id))

    @app.get("/api/projects/{project_id}/pages/{page}/mask")
    async def page_mask(project_id: str, page: str) -> FileResponse:
        page_id = service.normalize_page_id(page)
        return FileResponse(service.mask_path(project_id, page_id))

    @app.post("/api/projects/{project_id}/regions/{region_id}/retry")
    async def retry_region(project_id: str, region_id: str) -> JSONResponse:
        payload = service.retry_region_ocr(project_id, region_id)
        return JSONResponse(payload)

    return app

