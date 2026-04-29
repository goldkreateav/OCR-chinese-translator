from __future__ import annotations

import os
from pathlib import Path
import sys

from fastapi import Body, FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, Response
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
    default_ocr_device: str = "cpu",
    default_ocr_fallback_to_cpu_on_oom: bool = False,
    allow_fallback: bool = False,
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
    app.state.default_ocr_device = default_ocr_device
    app.state.default_ocr_fallback_to_cpu_on_oom = bool(default_ocr_fallback_to_cpu_on_oom)
    app.state.allow_fallback = bool(allow_fallback)
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
        paddle_device_probe = service._probe_paddle_device_runtime()
        paddle_probe = service._probe_paddle_runtime()
        enable_retry_ocr = str(os.getenv("WEB_ENABLE_RETRY_OCR", "") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        return JSONResponse(
            {
                "app_version": v,
                "quality_bridge_enabled": bool(os.getenv("OCR_PADDLE_PYTHON")),
                "bridge_python": os.getenv("OCR_PADDLE_PYTHON"),
                "default_ocr_device": app.state.default_ocr_device,
                "default_ocr_fallback_to_cpu_on_oom": bool(app.state.default_ocr_fallback_to_cpu_on_oom),
                "allow_fallback": bool(app.state.allow_fallback),
                "web_enable_retry_ocr": bool(enable_retry_ocr),
                **paddle_device_probe,
                **paddle_probe,
            }
        )

    @app.get("/api/health/paddle")
    async def health_paddle() -> JSONResponse:
        payload = service._probe_paddle_runtime()
        payload["allow_fallback"] = bool(app.state.allow_fallback)
        return JSONResponse(payload)

    @app.post("/api/projects", response_model=ProjectCreateResponse)
    async def create_project(file: UploadFile = File(...)) -> ProjectCreateResponse:
        created = service.create_project(file)
        return ProjectCreateResponse(
            project_id=created["project_id"],
            filename=created["filename"],
            status=created["status"],
        )

    @app.get("/api/projects/{project_id}/export/ocpkg")
    async def export_ocpkg(project_id: str) -> Response:
        data, download_name = service.export_ocpkg_bytes(project_id)
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
        )

    @app.post("/api/import/ocpkg")
    async def import_ocpkg(file: UploadFile = File(...), dpi: int = Form(400)) -> JSONResponse:
        """
        One-file import: ocpkg contains input.pdf + report.json.
        We render pages from embedded PDF and return a project_id + pages list,
        plus the parsed report so the UI can show regions/translations immediately.
        """
        created = service.import_ocpkg_and_create_project(file)
        project_id = created["project_id"]
        report = created["report"]
        chosen_backend = app.state.default_render_backend
        poppler_path = app.state.default_poppler_path
        rendered = service.render_pages_for_existing_project(
            project_id,
            dpi=int(dpi or 400),
            render_backend=chosen_backend,
            poppler_path=poppler_path,
        )
        return JSONResponse({"project_id": project_id, "pages": rendered.get("pages") or [], "report": report})

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

    @app.post("/api/import/projects/{project_id}/translations/enqueue")
    async def import_enqueue_translations(
        project_id: str,
        report: dict = Body(...),
        lang: str = "ru",
    ) -> JSONResponse:
        payload = service.enqueue_missing_translations_from_report(project_id, report, lang=lang)
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
        ocr_device = request.ocr_device or app.state.default_ocr_device
        ocr_fallback = app.state.default_ocr_fallback_to_cpu_on_oom
        options = GenerateOptions(
            dpi=request.dpi,
            render_backend=render_backend,
            poppler_path=poppler_path,
            ocr_mode=ocr_mode,
            ocr_workers=ocr_workers,
            ocr_device=ocr_device,
            ocr_fallback_to_cpu_on_oom=bool(ocr_fallback),
            allow_fallback=bool(app.state.allow_fallback),
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
            pipeline_status=data.get("pipeline_status"),
            stage=data.get("stage"),
            progress=data.get("progress"),
            progress_pages=data.get("progress_pages"),
            ocr_runtime=data.get("ocr_runtime"),
            translation=data.get("translation"),
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
        # Best-effort: ignore malformed region entries instead of failing the whole response.
        region_models: list[RegionRecord] = []
        for region in regions_payload:
            try:
                region_models.append(RegionRecord(**(region or {})))
            except Exception:
                continue
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

    @app.get("/api/projects/{project_id}/pages/{page}/translations")
    async def translations_page(project_id: str, page: str, lang: str = "ru") -> JSONResponse:
        page_id = service.normalize_page_id(page)
        return JSONResponse(service.load_page_translations(project_id, page_id, lang=lang))

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
        payload = service.retry_region_ocr(
            project_id,
            region_id,
            allow_fallback=bool(app.state.allow_fallback),
        )
        return JSONResponse(payload)

    return app

